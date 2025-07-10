import math, os, time
import numpy as np
from opendssdirect import dss
from midspan_hif_utils import make_midspan_hif_dss, kron_reduce
from parse_opendss_file import (
    parse_opendss_to_mpc, merge_closed_switches_in_mpc_and_dss,
    build_global_y_per_unit,
)
from cal_pf import run_newton_powerflow_3p
from utilities.mea_fun import measurement_function
from lagrangian_m import run_lagrangian_polar
from utilities.report_results import report_results

PRISTINE_DSS = "4Bus-YY-Bal.DSS"
FAULT_DSS    = "4Bus-YY-Bal_HIF.dss"
TARGET_LINE  = "line.2"              # split this one
FAULT_BUS    = "FaultBus"
R_HIF        = 150.0               # ohms line‑to‑neutral (≈ 0.5A @2.4kV)
SWITCH_THRES = 0.1                   # Ω threshold to merge DSS switches
STD_P, STD_Q, STD_V = 1e-3, 1e-3, 1e-5

def add_kw_for_constant_resistance(dss_file: str, r_ohm: float, fault_bus: str):
    txt = open(dss_file, "r").read().splitlines()
    for k, line in enumerate(txt):
        if f"New Load.HIF_Load" in line:
            kv_ln = 4.16 / math.sqrt(3)
            p_kw = (kv_ln * 1e3) ** 2 / r_ohm / 1e3
            txt[k] = (f"New Load.HIF_Load Bus1={fault_bus}.1 Phases=1 "
                      f"Conn=Wye Model=2 kW={p_kw:.6g} kvar=0 kV={kv_ln:.4g}")
            break
    open(dss_file, "w").write("\n".join(txt))

def main():
    # 1)  create a DSS file with a constant‑R HIF
    make_midspan_hif_dss(PRISTINE_DSS, FAULT_DSS, TARGET_LINE)
    add_kw_for_constant_resistance(FAULT_DSS, R_HIF, FAULT_BUS)

    # 2)  parse pristine and faulted circuits
    mpc_orig = parse_opendss_to_mpc(PRISTINE_DSS, baseMVA=1.0, lc_filename="LineConstantsCode4.txt",
                                    slack_bus="sourcebus")
    mpc_fault = parse_opendss_to_mpc(FAULT_DSS, baseMVA=1.0, lc_filename="LineConstantsCode4.txt",
                                     slack_bus="sourcebus")
    merge_closed_switches_in_mpc_and_dss(mpc_fault, SWITCH_THRES)

    # 3)  run PF on augmented network to get "true" faulted voltages
    Vr, Vi, bp_full = run_newton_powerflow_3p(mpc_fault, 1e-6, 20)
    Vc_full = Vr + 1j * Vi
    report_results(Vr, Vi, bp_full, mpc_fault)

    # 4)  Build the full Y-Bus for the faulted network
    dss.Command("Clear")
    dss.Command(f'Redirect "{FAULT_DSS}"')
    Y_full, node_order_list = build_global_y_per_unit(mpc_fault)

    # --- Add HIF shunt admittance to the Y matrix ---
    hif_admittance = 1 / R_HIF  # S (line-to-neutral)
    fault_bus_id = mpc_fault["busname_to_id"][FAULT_BUS.lower()]
    node_phase_map = {node_name: i for i, node_name in enumerate(node_order_list)}
    hif_node_name = f"bus{fault_bus_id}.1"

    if hif_node_name in node_phase_map:
        hif_idx = node_phase_map[hif_node_name]
        v_base_ln = (4.16e3 / math.sqrt(3))
        s_base = mpc_fault["baseMVA"] * 1e6
        z_base = v_base_ln ** 2 / s_base
        y_base = 1 / z_base
        y_hif_pu = hif_admittance / y_base
        Y_full = Y_full.tolil()
        Y_full[hif_idx, hif_idx] += y_hif_pu
        Y_full = Y_full.tocsc()

    # --- COMPARISON OF POWER INJECTION CALCULATION METHODS ---
    print("\n" + "=" * 50)
    print("--- Comparing Injection Calculation Methods ---")
    print("=" * 50)

    # Method 1: Generate from Full Model, then Delete FaultBus measurements
    print("Calculating injections using full faulted model...")
    x_full = np.hstack([np.abs(Vc_full), np.angle(Vc_full)])
    z_full = measurement_function(x_full, Y_full, mpc_fault, bp_full)
    n_nodes_full = len(bp_full)

    indices_to_remove = []
    for ph in range(3):
        node_idx = bp_full.get((fault_bus_id, ph))
        if node_idx is not None:
            indices_to_remove.append(node_idx)  # P
            indices_to_remove.append(n_nodes_full + node_idx)  # Q
    injections_from_full = np.delete(z_full[:2 * n_nodes_full], sorted(indices_to_remove))

    # Method 2: Reduce Model with Kron Reduction, then Generate
    print("Calculating injections using Kron-reduced model...")
    bp_red = {(int(row[0]), ph): 3 * i + ph for i, row in enumerate(mpc_orig["bus3p"]) for ph in range(3)}
    elim_idx = [bp_full[(fault_bus_id, ph)] for ph in range(3)]
    Y_red = kron_reduce(Y_full, elim_idx)
    V_red = np.array([Vc_full[bp_full[(int(row[0]), ph)]] for row in mpc_orig["bus3p"] for ph in range(3)])
    x_red = np.hstack([np.abs(V_red), np.angle(V_red)])
    z_reduced = measurement_function(x_red, Y_red, mpc_orig, bp_red)
    injections_from_reduced = z_reduced[:2 * len(bp_red)]

    print("\n" + "=" * 50)
    print("--- Verifying Kron Reduction Correctness ---")
    print("=" * 50)

    # 1. Get the full voltage vector from the power flow solution
    # Vc_full is already available

    # 2. Calculate the true current injections for the ENTIRE network
    I_full_all_nodes = Y_full @ Vc_full

    # 3. Partition the full currents and voltages
    # Create a list of retained indices
    retained_idx = sorted([v for k, v in bp_full.items() if k[0] != fault_bus_id])
    # The eliminated indices 'elim_idx' are already available and sorted
    # elim_idx = sorted([bp_full[(fault_bus_id, ph)] for ph in range(3)])

    # Extract the "true" currents and voltages for the retained nodes
    I_m_true = I_full_all_nodes[retained_idx]
    V_m = Vc_full[retained_idx]  # This should be identical to V_red

    # 4. Calculate equivalent current injections using the reduced matrix
    I_m_equivalent = Y_red @ V_red

    if np.allclose(I_m_true, I_m_equivalent):
        print("SUCCESS: Kron Reduction is correct. Current injections match.")
    else:
        print("ERROR: Kron Reduction is incorrect or the underlying assumptions are violated.")
        print(f"Max absolute difference in currents: {np.max(np.abs(I_m_true - I_m_equivalent)):.6g}")

    print("=" * 50 + "\n")

    # --- COMPARISON OF POWER INJECTION CALCULATION METHODS ---
    print("\n" + "=" * 60)
    print("--- Verifying Injection Calculation Equivalence ---")
    print("=" * 60)

    # --- Method 1: Generate from Full Model, then extract injections ---
    x_full = np.hstack([np.abs(Vc_full), np.angle(Vc_full)])
    z_full = measurement_function(x_full, Y_full, mpc_fault, bp_full)

    # --- Method 2: Reduce Model with Kron Reduction, then Generate ---
    bp_red = {(int(row[0]), ph): 3 * i + ph for i, row in enumerate(mpc_orig["bus3p"]) for ph in range(3)}
    elim_idx = [bp_full[(fault_bus_id, ph)] for ph in range(3)]
    Y_red = kron_reduce(Y_full, elim_idx)
    V_red = np.array([Vc_full[bp_full[(int(row[0]), ph)]] for row in mpc_orig["bus3p"] for ph in range(3)])
    x_red = np.hstack([np.abs(V_red), np.angle(V_red)])
    z_reduced = measurement_function(x_red, Y_red, mpc_orig, bp_red)

    # --- Detailed Comparison for Debugging ---
    print("Debugging injections for Bus 3 and Bus 4, Phase A (node 1)...")

    # Get indices for Bus 3, Phase A (0)
    idx_full_b3pA = bp_full.get((3, 0))
    idx_red_b3pA = bp_red.get((3, 0))

    # Get indices for Bus 4, Phase A (0)
    idx_full_b4pA = bp_full.get((4, 0))
    idx_red_b4pA = bp_red.get((4, 0))

    n_nodes_full = len(bp_full)
    n_nodes_red = len(bp_red)

    # Extract P and Q injections for comparison
    P_full_b3 = z_full[idx_full_b3pA]
    Q_full_b3 = z_full[n_nodes_full + idx_full_b3pA]
    P_red_b3 = z_reduced[idx_red_b3pA]
    Q_red_b3 = z_reduced[n_nodes_red + idx_red_b3pA]

    P_full_b4 = z_full[idx_full_b4pA]
    Q_full_b4 = z_full[n_nodes_full + idx_full_b4pA]
    P_red_b4 = z_reduced[idx_red_b4pA]
    Q_red_b4 = z_reduced[n_nodes_red + idx_red_b4pA]

    print("\n--- Bus 3, Phase A ---")
    print(f"Voltage (Full Model):  {Vc_full[idx_full_b3pA]:.6f}")
    print(f"Voltage (Red. Model):  {V_red[idx_red_b3pA]:.6f}")
    print(f"P_inj (Full Model):    {P_full_b3:.6f}")
    print(f"P_inj (Red. Model):    {P_red_b3:.6f}")
    print(f"Q_inj (Full Model):    {Q_full_b3:.6f}")
    print(f"Q_inj (Red. Model):    {Q_red_b3:.6f}")

    print("\n--- Bus 4, Phase A ---")
    print(f"Voltage (Full Model):  {Vc_full[idx_full_b4pA]:.6f}")
    print(f"Voltage (Red. Model):  {V_red[idx_red_b4pA]:.6f}")
    print(f"P_inj (Full Model):    {P_full_b4:.6f}")
    print(f"P_inj (Red. Model):    {P_red_b4:.6f}")
    print(f"Q_inj (Full Model):    {Q_full_b4:.6f}")
    print(f"Q_inj (Red. Model):    {Q_red_b4:.6f}")

    # Overall check
    injections_from_full = np.delete(z_full[:2 * n_nodes_full],
                                     [idx for idx in sorted(indices_to_remove) if idx < 2 * n_nodes_full])
    injections_from_reduced = z_reduced[:2 * len(bp_red)]

    if np.allclose(injections_from_full, injections_from_reduced):
        print("\nSUCCESS: Power injections from both methods are identical.")
    else:
        print("\nERROR: Power injections from the two methods DO NOT match.")
    print("=" * 60 + "\n")

    # 5)  Measurement generation for State Estimation (using Method 2 as per your setup)
    z = z_reduced  # Use the measurements from the reduced model

    # Add noise
    R = np.diag([STD_P ** 2] * len(bp_red) + [STD_Q ** 2] * len(bp_red) + [STD_V ** 2] * len(bp_red))
    z_noisy = z + np.random.normal(0.0, np.sqrt(np.diag(R)), size=len(z))

    # 6)  State estimation using the PRISTINE model
    dss.Command("Clear")
    dss.Command(f'Redirect "{PRISTINE_DSS}"')
    Y_orig, _ = build_global_y_per_unit(mpc_orig)

    # Compare Y_orig and Y_red
    if not np.allclose(Y_orig.toarray(), Y_red.toarray()):
        Y_diff = Y_orig.toarray() - Y_red.toarray()

    x_est, converged, lambdaN = run_lagrangian_polar(
        z_noisy, V_red, bp_red, Y_orig, R, mpc_orig
    )
    if not converged:
        print("State estimator did not converge.")
        return

    # 7) Report the largest NLM
    idx = int(np.argmax(np.abs(lambdaN)))
    node_to_busphase = {v: k for k, v in bp_red.items()}
    node_idx = idx // 2
    param_type = "Conductance (G)" if (idx % 2) == 0 else "Susceptance (B)"
    bus_id, phase_idx = node_to_busphase[node_idx]
    phase_char = chr(ord('A') + phase_idx)

    print(f"Largest multiplier: |lambda| = {abs(lambdaN[idx]):.3g} "
          f"at Bus {bus_id}, Phase {phase_char}, for parameter: {param_type}")


# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()