import numpy as np
import time
from system_y import build_global_y_per_unit
from wls_estimation import run_wls_state_estimation
from wls_estimation_p import run_wls_state_estimation_polar
from lagrangian_m import run_lagrangian_polar
from utilities.mea_fun import measurement_function
from utilities.report_results import report_results
from utilities.cal_pf import run_newton_powerflow_3p

##############################################################################
# 1. Load the 4-bus test case (matpower-like data)
##############################################################################
def load_t_case3p_a():
    mpc = {}
    mpc["version"] = "2"
    mpc["baseMVA"] = 1
    mpc["freq"] = 60.0
    # bus3p columns: [bus_id, type, base_kV_LL, VmA, VmB, VmC, VaA, VaB, VaC]
    # type=3 => slack, type=1 => PQ, type=2 => PV
    # base_kV_LL: line-to-line base
    mpc["bus3p"] = np.array([
        [1, 3, 12.47, 1.0, 1.0, 1.0,   0.0,   -120.0, 120.0],
        [2, 1, 12.47, 1.0, 1.0, 1.0,   0.0,   -120.0, 120.0],
        [3, 1,  4.16, 1.0, 1.0, 1.0,   0.0,   -120.0, 120.0],
        [4, 1,  4.16, 1.0, 1.0, 1.0,   0.0,   -120.0, 120.0],
    ], dtype=float)
    # line3p columns: [id, fbus, tbus, status, lcid, length (miles)]
    mpc["line3p"] = np.array([
        [1, 1, 2, 1, 1, 2000.0/5280.0],
        [2, 3, 4, 1, 1, 2500.0/5280.0],
    ], dtype=float)
    # xfmr3p columns: [xfid, fbus, tbus, status, R, X, basekVA, basekV_LL(HV)]
    mpc["xfmr3p"] = np.array([
        [1, 2, 3, 1, 0.01, 0.06, 6000.0, 12.47, 4.16],
    ], dtype=float)
    # load3p columns: [ldid, ldbus, status, PdA, PdB, PdC, pfA, pfB, pfC]
    mpc["load3p"] = np.array([
        [1, 4, 1, 1275.0, 1800.0, 2375.0, 0.85, 0.90, 0.95],
    ], dtype=float)
    # gen3p columns: [genid, gbus, status, VgA, VgB, VgC, PgA, PgB, PgC, QgA, QgB, QgC]
    mpc["gen3p"] = np.array([
        [1, 1, 1, 1.0, 1.0, 1.0, 2000.0, 2000.0, 2000.0, 0.0, 0.0, 0.0],
    ], dtype=float)
    # line construction (3x3) data for lcid=1
    # [lcid, R11, R21, R31, R22, R32, R33, X11, X21, X31, X22, X32, X33, ...C11...]
    mpc["lc"] = np.array([
        [1,
         0.457541, 0.15594,  0.153474, 0.466617, 0.157996, 0.461462,
         1.078,    0.501648, 0.384909, 1.04813,  0.423624, 1.06502,
         15.0671, -4.86241, -1.85323, 15.875, -3.09098, 14.3254]
    ], dtype=float)
    return mpc
##############################################################################
# 3. Newton Power Flow in Rectangular Coordinates
##############################################################################
def main():
    # 1) Load the 4-bus data
    mpc = load_t_case3p_a()
    nbus = len(mpc["bus3p"])
    from opendssdirect import dss
    dss.run_command('Redirect "4Bus-YY-Bal.DSS"')
    dss.Solution.Solve()
    # 2) Run Newton PF
    Vr, Vi, busphase_map = run_newton_powerflow_3p(mpc, tol=1e-6, max_iter=20)
    # 3) Print final PF results
    x_f = Vr + 1j * Vi
    x = np.concatenate([np.abs(x_f), np.angle(x_f)])
    report_results(Vr, Vi, busphase_map, mpc)

    # 4) Generate the measurement data from PF results (Add noise)
    Y_pu_s, _ = build_global_y_per_unit()
    z = measurement_function(x, Y_pu_s, mpc, busphase_map)
    std_P = 0.0001
    std_Q = 0.0001
    std_V = 0.00001
    nnodephase = 3 * nbus
    num_P_inj = nnodephase
    num_Q_inj = nnodephase
    num_PQ_flow = 4 * 3 * len(mpc["line3p"])
    num_Vmag = nnodephase
    covariance_matrix = np.diag([std_P**2]*num_P_inj + [std_Q**2]*num_Q_inj + [std_P**2]*num_PQ_flow + [std_V**2]*num_Vmag)
    z_noisy = z + np.random.multivariate_normal(np.zeros(len(z)), covariance_matrix)
    x_est, success, lambdaN = run_lagrangian_polar(
        z_noisy, x_f, busphase_map, Y_pu_s, covariance_matrix, mpc
    )

    dss.Command('Redirect "4Bus-YY-Bal.DSS"')
    dss.Text.Command("New LineCode.Kersting nphases=3 units=mi")
    dss.Text.Command("~ Rmatrix=[0.457552 |0.155951  0.466628 |0.153485  0.158007  0.461473 ]")
    dss.Text.Command("~ Xmatrix=[1.07805  |0.51679   1.04818  |0.384938  0.423653  1.06507  ]")
    dss.Text.Command("~ Cmatrix=[15.0675  |-4.86254  15.8754  |-1.85328  -3.09107  14.3258  ]")
    dss.Text.Command("Edit line.line1 Linecode=Kersting")
    dss.Solution.Solve()
    Y_pu, _ = build_global_y_per_unit()
    # tmp1 = np.array(Y_pu_s.todense())
    # tmp2 = np.array(Y_pu.todense())
    # diff = np.abs(tmp1 - tmp2)
    x_est, success, lambdaN = run_lagrangian_polar(
        z_noisy, x_f, busphase_map, Y_pu, covariance_matrix, mpc
    )

    # 6) find the largest index
    largest_idx = np.argmax(np.abs(lambdaN))
    largest_val = lambdaN[largest_idx]
    print(f"Largest Lagrangian param => index={largest_idx}, value={largest_val:.4g}")
    if success:
        print("DSSE converged with the modified X_aa.\n")
    else:
        print("DSSE did not converge.\n")
if __name__ == "__main__":
    main()
