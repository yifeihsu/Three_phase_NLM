import math
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

# Constants for the case
PRISTINE_DSS = "Secondary_Equivalent.dss"
FAULT_DSS = "Secondary_Equivalent_HIF.dss"
TARGET_LINE = "Line.156"  # split this one
FAULT_BUS = "FaultBus"
R_HIF = 1
SWITCH_THRES = 2
STD_P, STD_Q, STD_V = 1e-5, 1e-5, 1e-5


def add_kw_for_constant_resistance(dss_file: str, r_ohm: float, fault_bus: str):
    """Updates a DSS load to model a constant resistance fault."""
    txt = open(dss_file, "r").read().splitlines()
    for k, line in enumerate(txt):
        if f"New Load.HIF_Load" in line:
            kv_ln = 0.120  # kV line-to-neutral
            p_kw = (kv_ln * 1e3) ** 2 / r_ohm / 1e3
            txt[k] = (f"New Load.HIF_Load Bus1={fault_bus}.1 Phases=1 "
                      f"Conn=Wye Model=2 kW={p_kw:.6g} kvar=0 kV={kv_ln:.4g}")
            break
    open(dss_file, "w").write("\n".join(txt))


def main():
    # 1) Create a DSS file with a constant-R HIF
    make_midspan_hif_dss(PRISTINE_DSS, FAULT_DSS, TARGET_LINE)
    add_kw_for_constant_resistance(FAULT_DSS, R_HIF, FAULT_BUS)

    # 2) Parse pristine and faulted circuits
    mpc_orig = parse_opendss_to_mpc(PRISTINE_DSS, baseMVA=1.0, lc_filename="LineConstantsCode.txt",
                                    slack_bus="s148")
    merge_closed_switches_in_mpc_and_dss(mpc_orig, SWITCH_THRES)
    mpc_fault = parse_opendss_to_mpc(FAULT_DSS, baseMVA=1.0, lc_filename="LineConstantsCode.txt",
                                     slack_bus="s148")
    merge_closed_switches_in_mpc_and_dss(mpc_fault, SWITCH_THRES)

    # 3) Run PF on the faulted network to get "true" faulted voltages
    Vr, Vi, bp_full = run_newton_powerflow_3p(mpc_fault, 1e-6, 20)
    Vc_full = Vr + 1j * Vi
    # report_results(Vr, Vi, bp_full, mpc_fault)

    # 4) Build the full Y-Bus for the faulted network
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
        v_base_ln = (1.2e2 / math.sqrt(3))
        s_base = mpc_fault["baseMVA"] * 1e6
        z_base = v_base_ln ** 2 / s_base
        y_base = 1 / z_base
        y_hif_pu = hif_admittance / y_base
        Y_full = Y_full.tolil()
        Y_full[hif_idx, hif_idx] += y_hif_pu
        Y_full = Y_full.tocsc()

    # 5) Generate measurements using the Kron-reduced model
    bp_red = {(int(row[0]), ph): 3 * i + ph for i, row in enumerate(mpc_orig["bus3p"]) for ph in range(3)}
    elim_idx = [bp_full[(fault_bus_id, ph)] for ph in range(3)]
    Y_red = kron_reduce(Y_full, elim_idx)
    V_red = np.array([Vc_full[bp_full[(int(row[0]), ph)]] for row in mpc_orig["bus3p"] for ph in range(3)])
    x_red = np.hstack([np.abs(V_red), np.angle(V_red)])
    z = measurement_function(x_red, Y_red, mpc_orig, bp_red)

    # Add noise to measurements (P, Q, and V)
    R = np.diag([STD_P ** 2] * len(bp_red) + [STD_Q ** 2] * len(bp_red) + [STD_V ** 2] * len(bp_red))
    z_noisy = z + np.random.normal(0.0, np.sqrt(np.diag(R)), size=len(z))
    # z_noisy = z
    # 6) Run state estimation using the PRISTINE model
    dss.Command("Clear")
    dss.Command(f'Redirect "{PRISTINE_DSS}"')
    Y_orig, _ = build_global_y_per_unit(mpc_orig)

    x_est, converged, lambdaN = run_lagrangian_polar(
        z_noisy, V_red, bp_red, Y_orig, R, mpc_orig
    )
    if not converged:
        print("State estimator did not converge.")
        return

if __name__ == "__main__":
    main()