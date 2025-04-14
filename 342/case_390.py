import numpy as np
import time
from opendssdirect import dss
from lagrangian_m import run_lagrangian_polar
from utilities.mea_fun import measurement_function
from utilities.report_results import report_results
from cal_pf import run_newton_powerflow_3p
from parse_opendss_file import parse_opendss_to_mpc
from report_results import report_results
from parse_opendss_file import build_global_y_per_unit, merge_closed_switches_in_mpc_and_dss

def main():
    # 1) Load the data from OpenDSS, func parse_opendss_to_mpc will solve the circuit
    # dss_filename = "Master.DSS"
    # mpc = parse_opendss_to_mpc(dss_filename, baseMVA=1.0, slack_bus="p1")
    # dss_filename = "4Bus-YY-Bal.DSS"
    # mpc = parse_opendss_to_mpc(dss_filename, baseMVA=1.0, lc_filename="LineConstantsCode4.txt", slack_bus="sourcebus")
    dss_filename = "6bus.DSS"
    mpc = parse_opendss_to_mpc(dss_filename, baseMVA=1.0, lc_filename="LineConstantsCode4.txt", slack_bus="sourcebus")
    # 2) merges and Run Newton PF
    merge_closed_switches_in_mpc_and_dss(mpc, switch_threshold=0.1)
    Vr, Vi, busphase_map = run_newton_powerflow_3p(mpc, tol=1e-6, max_iter=20)
    # 3) Print final PF results
    x_f = Vr + 1j * Vi
    x = np.concatenate([np.abs(x_f), np.angle(x_f)])
    # report_results(Vr, Vi, busphase_map, mpc)
    # 4) Generate the measurement data from PF results (Add noise)
    Y_pu_s, _ = build_global_y_per_unit(mpc, dss_filename)
    z = measurement_function(x, Y_pu_s, mpc, busphase_map)
    # Inject noise into the measurements
    std_P, std_Q, std_V = 0.0001, 0.0001, 0.00001
    num_P_inj = num_Q_inj = num_Vmag = len(busphase_map)
    num_PQ_flow = 4 * 3 * len(mpc["line3p"])
    covariance_matrix = np.diag([std_P**2]*num_P_inj + [std_Q**2]*num_Q_inj + [std_P**2]*num_PQ_flow + [std_V**2]*num_Vmag)
    # Print the shape of the covariance matrix
    print("Covariance matrix shape:", covariance_matrix.shape)
    z_noisy = z + np.random.normal(0, np.sqrt(np.diag(covariance_matrix)), size=len(z))

    # Pre-estimation if needed
    # x_est, success, lambdaN = run_lagrangian_polar(
    #     z_noisy, x_f, busphase_map, Y_pu_s, covariance_matrix, mpc
    # )

    # NLM Test for different lines
    # Test for 4Bus Case
    dss.Command('Redirect "6bus.DSS"')
    dss.Text.Command("New LineCode.Kersting nphases=3 units=mi")
    dss.Text.Command("~ Rmatrix=[0.457552  |0.155951  0.466628  |0.153485  0.158007  0.461473]")
    dss.Text.Command("~ Xmatrix=[1.07805  |0.551679  1.04818  |0.484938  0.423653  0.906507  ]")
    dss.Text.Command("~ Cmatrix=[15.0675  |-4.86254  15.8754  |-1.85328  -3.09107  14.3258  ]")
    dss.Text.Command("Edit line.7 Linecode=Kersting")
    dss.Solution.Solve()

    # Test for 342 bus Case
    # dss.Command('Redirect "Master.DSS"')
    # dss.Text.Command("New LineCode.NLM nphases=3 units=mi")
    # dss.Text.Command("~ Rmatrix=[0.33619  |0.128579  1.328163  |0.132719  2.128579  0.33619]")
    # dss.Text.Command("~ Xmatrix=[0.710307  |0.168686  0.775146  |0.193614  0.268686  0.510307]")
    # dss.Text.Command("~ Cmatrix=[34.0866  |-12.0789  34.1952  |-4.12143  -12.0789  34.0866]")
    # dss.Text.Command("Edit line.307 Linecode=NLM")
    # dss.Text.Command("Edit line.307_2 Linecode=NLM")
    # dss.Text.Command("Edit line.307_3 Linecode=NLM")
    # dss.Text.Command("Edit line.307_4 Linecode=NLM")
    # dss.Text.Command("Edit line.307_5 Linecode=NLM")
    # dss.Text.Command("Edit line.307_6 Linecode=NLM")
    # dss.Solution.Solve()
    ### Rebuild the Y Matrix *!
    Y_pu, _ = build_global_y_per_unit(mpc, dss_filename)
    x_est, success, lambdaN = run_lagrangian_polar(
        z_noisy, x_f, busphase_map, Y_pu, covariance_matrix, mpc
    )
    # 6) find the largest index
    largest_idx = np.argmax(np.abs(lambdaN))
    largest_val = lambdaN[largest_idx]
    error_type = "X" if (largest_idx%12 + 1)//6 == 1 else "R"
    print(f"Largest Lagrangian param => index={(largest_idx%12 + 1)%6}, value={largest_val:.4g}, "
          f"line={largest_idx//12 + 1}, error_type={error_type}")
if __name__ == "__main__":
    main()
