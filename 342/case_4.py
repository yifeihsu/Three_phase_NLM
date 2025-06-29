"""
4‑bus feeder with a *mid‑span* High‑Impedance Fault (HIF) injected on
Line.2 (n3–n4) at its midpoint.  The script

1. creates  “4Bus-YY-Bal_HIF.dss” on‑the‑fly,
2. parses both the original and the faulted circuit,
3. runs a time‑domain simulation of the stochastic Mayr‑Emanuel arc,
4. at every time step:
      • updates the 1‑φ HIF load power,
      • solves the circuit
5. performs Kron reduction to give measurements whose size/order match
   the 4‑bus network.
"""
import numpy as np, math, opendssdirect as dss
from scipy.sparse import csc_matrix
import time as _time
from pathlib import Path
from midspan_hif_utils import make_midspan_hif_dss, kron_reduce
from parse_opendss_file import (
    parse_opendss_to_mpc, build_global_y_per_unit,
    merge_closed_switches_in_mpc_and_dss,
)
from cal_pf import run_newton_powerflow_3p
from utilities.mea_fun        import measurement_function
from utilities.report_results import report_results

def dss_cmd(cmd: str):
    return dss.Command(cmd)

# ───────────────────────── 1.  File names & constants ──────────────────
PRISTINE_DSS = "4Bus-YY-Bal.DSS"
FAULT_DSS    = "4Bus-YY-Bal_HIF.dss"     # auto‑generated below
TARGET_LINE  = "line.2"                  # ← split this one
FAULT_BUS    = "FaultBus"                # new mid‑span node

import os
FAULT_DSS_ABS = os.path.abspath(FAULT_DSS).replace(os.sep, "/")

# Mayr/Emanuel parameters
V_ARC_P, V_ARC_N = 1000.0, 1100.0        # [V_ln]
TAU, P_COOL       = 300e-6, 2.8e6
G_MIN, G_MAX      = 1e-7, 0.02           # [S]
SIGMA_NOISE       = 5e-6
R_MEAN, R_SIGMA   = 300.0, 0.6
DT, N_STEPS       = 20e-6, 5_000         # 20µs × 5000 ≈ 0.1s

# ───────────────────────── 2.  Helpers (arc model) ─────────────────────
def hif_current(v: float, g: float) -> float:
    if   v >  V_ARC_P: return (v -  V_ARC_P) * g
    elif v < -V_ARC_N: return (v +  V_ARC_N) * g # Corrected line
    return 0.0

def mayr_update(g: float, p: float, dt: float) -> float:
    g_dot = ((p / P_COOL) - g) / TAU
    noise = np.random.normal(0.0, SIGMA_NOISE * math.sqrt(dt))
    g_new = g + g_dot * dt + noise
    return float(np.clip(g_new, G_MIN, G_MAX))

# ───────────────────────── 3.  Build the faulted DSS file ──────────────
make_midspan_hif_dss(PRISTINE_DSS, FAULT_DSS, TARGET_LINE)

# ───────────────────────── 4.  Parse both circuits ─────────────────────
mpc_orig = parse_opendss_to_mpc(PRISTINE_DSS, baseMVA=1.0, lc_filename="LineConstantsCode4.txt",
                               slack_bus="sourcebus")
merge_closed_switches_in_mpc_and_dss(mpc_orig, 0.1)

mpc_fault = parse_opendss_to_mpc(FAULT_DSS,   baseMVA=1.0, lc_filename="LineConstantsCode4.txt",
                                 slack_bus="sourcebus")
merge_closed_switches_in_mpc_and_dss(mpc_fault, 0.1)

# ───────────────────────── 5.  One‑shot PF on faulted topology ─────────
Vr_f, Vi_f, bp_fault = run_newton_powerflow_3p(mpc_fault, 1e-6, 20)
Vc_f  = Vr_f + 1j * Vi_f
report_results(Vr_f, Vi_f, bp_fault, mpc_fault)

# ───────────────────────── 6.  Kron reduction --------------------------
Y_full, _ = build_global_y_per_unit(mpc_fault)
fault_id  = mpc_fault["busname_to_id"][FAULT_BUS.lower()]
elim_idx  = [bp_fault[(fault_id, ph)] for ph in range(3)]
Y_red     = kron_reduce(Y_full, elim_idx)

# --- keep the original bus ordering for the SE / measurement stage -----
bp_orig   = {(int(r[0]), ph):   3*idx + ph
             for idx, r in enumerate(mpc_orig["bus3p"])
             for ph in range(3)}
Vc_red    = np.array([Vc_f[ bp_fault[(int(r[0]), ph)] ]
                      for r in mpc_orig["bus3p"]  for ph in range(3)])
x_red     = np.hstack([np.abs(Vc_red), np.angle(Vc_red)])

z         = measurement_function(x_red, Y_red, mpc_orig, bp_orig)

# Add white noise identical to the template in case_4.py
std_P, std_Q, std_V = 1e-4, 1e-4, 1e-5
n_busph   = len(bp_orig)
n_PQflow  = 4 * 3 * len(mpc_orig["line3p"])
Sigma         = np.diag([std_P**2]*n_busph + [std_Q**2]*n_busph +
                    [std_P**2]*n_PQflow + [std_V**2]*n_busph)
z_noisy   = z + np.random.normal(0.0, np.sqrt(np.diag(Sigma)), size=len(z))

print(f"Measurement length  : {len(z_noisy)}  (unchanged)")
print(f"Original bus count  : {len(mpc_orig['bus3p'])}")
print("Kron‑reduced FaultBus; ready for state‑estimation, etc.")

# ------------------------------------------------------------------
# 7.  Time‑domain HIF simulation
# ------------------------------------------------------------------
dss_cmd("clear")                                    # start from a clean slate
dss_cmd(f'compile "{FAULT_DSS_ABS}"')               # absolute path, quoted

if not dss.Circuit.Name():                          # empty ⇒ compile failed
    raise RuntimeError("OpenDSS compile failed:\n" + dss.Text.Result())

dss_cmd("calcvoltagebases")
dss_cmd("reset monitors")                           # optional clean‑up

# --- initialise the snapshot solver -------------------------------
dss.Solution.Mode(0)        # 0 = SNAP
dss.Solution.Number(1)      # exactly one step per SolveSnap()
DT_SEC = DT                 # DT was defined earlier (20e‑6 s)
dss.Solution.StepSize(DT_SEC)

t_log, v_log, i_log, g_log, p_log = [], [], [], [], []

# --- arc model state ----------------------------------------------
sim_time = 0.0
g_arc    = G_MIN
i_prev   = 0.0

print(f"\nRunning dynamic arc simulation: {N_STEPS} steps * "
      f"{DT_SEC*1e6:.0f} us")

tic = _time.time()
for k in range(N_STEPS):
    dss.Solution.SolveSnap()

    # instantaneous voltage at FaultBus.1 ---------------------------
    dss.Circuit.SetActiveBus(FAULT_BUS)
    kv_ln_base = dss.Bus.kVBase()
    mag_pu, ang_deg = dss.Bus.puVmagAngle()[:2]
    v_peak  = kv_ln_base * 1000.0 * (2.0 ** 0.5)
    theta   = 2.0 * math.pi * 60.0 * sim_time + math.radians(ang_deg)
    v_inst  = v_peak * mag_pu * math.cos(theta)

    # Mayr‑Emanuel arc model ---------------------------------------
    i_arc  = hif_current(v_inst, g_arc)
    p_arc  = v_inst * i_arc
    if (i_prev * i_arc) < 0.0 and abs(i_prev) > 1e-6:
        g_arc = 1.0 / np.random.lognormal(math.log(R_MEAN), R_SIGMA)
    g_arc  = mayr_update(g_arc, p_arc, DT_SEC)
    i_prev = i_arc

    dss_cmd(f'edit Load.HIF_Load kW={p_arc/1000.0:.9f}')

    t_log.append(sim_time)
    v_log.append(v_inst)
    i_log.append(i_arc)
    g_log.append(g_arc)
    p_log.append(p_arc)

    sim_time += DT_SEC
    if (k + 1) % 1000 == 0:
        print(f"  {k+1}/{N_STEPS} steps complete", end="\r")

print(f"\nDynamic simulation finished in {(_time.time() - tic):.2f} s")
# ------------------------------------------------------------------
# 9.  Plot V(t), i(t) and g(t)  (matplotlib, one figure per signal)
# ------------------------------------------------------------------
import matplotlib.pyplot as plt

# Convert lists to NumPy arrays (not strictly necessary, but handy)
t_arr = np.array(t_log)        # seconds
v_arr = np.array(v_log)        # volts (instantaneous)
i_arr = np.array(i_log)        # amps
g_arr = np.array(g_log)        # siemens
p_arr = np.array(p_log)

# 9‑A.  Voltage
plt.figure()
plt.plot(t_arr, v_arr)
plt.title("Phase‑A Voltage at FaultBus.1")
plt.xlabel("time [s]")
plt.ylabel("v_inst [V]")
plt.grid(True)


print(f"min(i_arc) = {i_arr.min():.3e} A,  max(i_arc) = {i_arr.max():.3e} A")

# 9‑B.  Arc current
plt.figure()
plt.plot(t_arr, i_arr)
plt.title("Arc Current i(t)")
plt.xlabel("time [s]")
plt.ylabel("i_arc [A]")
plt.grid(True)

# 9‑C.  Arc conductance
plt.figure()
plt.plot(t_arr, g_arr)
plt.title("Arc Conductance g(t)")
plt.xlabel("time [s]")
plt.ylabel("g_arc [S]")
plt.grid(True)

# 9‑D.  Arc power
plt.figure()
plt.plot(t_arr, p_arr)
plt.title("Arc Power p(t)")
plt.xlabel("time [s]")
plt.ylabel("p_arc [W]")
plt.grid(True)
plt.tight_layout()
# Show all plots
# --- At the end of the script, before the plots ---

# 1. Calculate the overall average active power
P_active_overall = np.mean(p_arr)

print(f"\n--- Power Calculations ---")
print(f"Peak Instantaneous Power: {p_arr.max() / 1000:.2f} kW")
print(f"Overall Active Power (P): {P_active_overall / 1000:.2f} kW")

plt.show()
print("All done – z_noisy ready for your state estimator.")