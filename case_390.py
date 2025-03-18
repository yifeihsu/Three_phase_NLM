import numpy as np
import time
from opendssdirect import dss
# from system_y import build_global_y_per_unit
# from wls_estimation import run_wls_state_estimation
# from wls_estimation_p import run_wls_state_estimation_polar
from lagrangian_m import run_lagrangian_polar
from utilities.mea_fun import measurement_function
from utilities.report_results import report_results
from cal_pf import run_newton_powerflow_3p
from parse_opendss_file import parse_opendss_to_mpc
from report_results import report_results
from parse_opendss_file import build_global_y_per_unit, merge_closed_switches_in_mpc_and_dss

def main():
    # 1) Load the 4-bus data
    dss_filename = "4Bus-YY-Bal1.dss"
    mpc = parse_opendss_to_mpc(dss_filename, baseMVA=1.0)
    # dss.run_command('Redirect "Master.DSS"')
    # dss.Solution.Solve()
    # 2) merges and Run Newton PF
    merge_closed_switches_in_mpc_and_dss(mpc, switch_threshold=2)
    Vr, Vi, busphase_map = run_newton_powerflow_3p(mpc, tol=6e-6, max_iter=20)
    # 3) Print final PF results
    x_f = Vr + 1j * Vi
    x = np.concatenate([np.abs(x_f), np.angle(x_f)])
    report_results(Vr, Vi, busphase_map, mpc)
    # 4) Generate the measurement data from PF results (Add noise)
    Y_pu_s, _ = build_global_y_per_unit(mpc, dss_filename)
    z = measurement_function(x, Y_pu_s, mpc, busphase_map)
    std_P = 0.0001
    std_Q = 0.0001
    std_V = 0.00001
    nnodephase = len(busphase_map)
    num_P_inj = nnodephase
    num_Q_inj = nnodephase
    num_PQ_flow = 4 * 3 * len(mpc["line3p"])
    num_Vmag = nnodephase
    covariance_matrix = np.diag([std_P**2]*num_P_inj + [std_Q**2]*num_Q_inj + [std_P**2]*num_PQ_flow + [std_V**2]*num_Vmag)
    # Print the shape of the covariance matrix
    print("Covariance matrix shape:", covariance_matrix.shape)
    z_noisy = z + np.random.normal(0, np.sqrt(np.diag(covariance_matrix)), size=len(z))
    x_est, success, lambdaN = run_lagrangian_polar(
    z_noisy, x_f, busphase_map, Y_pu_s, covariance_matrix, mpc
    )

    # # NLM Test for different lines
    # dss.run_command('Redirect "4Bus-YY-Bal1.DSS"')
    
    # linecode_name = "Kersting"
    
    # old_xmatrix = [
    #    1.07805,      # Xaa
    #    0.501679, 1.04818,
    #    0.384938, 0.423653, 1.06507
    # ]
    # # Multiply the first entry (Xaa) by 10:
    # old_xmatrix[0] *= 10.0  # => 1.07805 => 10.7805
    
    # # re-build a string for Xmatrix
    # # Xmatrix in DSS is typically bracketed as [Xaa  | Xab Xbb  | Xac Xbc Xcc ]
    # xmat_str = f"[{old_xmatrix[0]:.6f} | {old_xmatrix[1]:.6f} {old_xmatrix[2]:.6f} | {old_xmatrix[3]:.6f} {old_xmatrix[4]:.6f} {old_xmatrix[5]:.6f}]"
    
    # # Now we do an Edit linecode command
    # cmd = f'Edit linecode.{linecode_name} Xmatrix={xmat_str}'
    # dss.run_command(cmd)
    # # print("Modified linecode Kersting Xmatrix =>", xmat_str)
    # # dss.Solution.Solve()
    # Y_pu, _ = build_global_y_per_unit(line_changes=None)
    # tmp1 = np.array(Y_pu_s.todense())
    # tmp2 = np.array(Y_pu.todense())
    # diff = np.abs(tmp1 - tmp2)
    # x_est, success, lambdaN = run_lagrangian_polar(
    #     z_noisy, x_f, busphase_map, Y_pu, covariance_matrix, mpc
    # )
    
    # # 6) find the largest index
    # largest_idx = np.argmax(np.abs(lambdaN))
    # # largest_val = lambdaN[largest_idx]
    # print(f"Largest Lagrangian param => index={largest_idx}, value={largest_val:.4g}")
    # if success:
    #     print("DSSE converged with the modified X_aa.\n")
    # else:
    #     print("DSSE did not converge.\n")
if __name__ == "__main__":
    main()
