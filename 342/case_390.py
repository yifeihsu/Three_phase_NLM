import numpy as np
import time
from opendssdirect import dss
from lagrangian_m import run_lagrangian_polar
from utilities.mea_fun import measurement_function
from utilities.report_results import report_results
from cal_pf import run_newton_powerflow_3p
from parse_opendss_file import parse_opendss_to_mpc, build_global_y_per_unit, merge_closed_switches_in_mpc_and_dss


def main():
    # 1) Load the data from OpenDSS, func parse_opendss_to_mpc will solve the circuit
    dss_filename = "Master.DSS"
    mpc = parse_opendss_to_mpc(dss_filename, baseMVA=1.0, slack_bus="p1")

    # 2) Merges and Run Newton PF
    merge_closed_switches_in_mpc_and_dss(mpc, switch_threshold=2)  ### Be careful with the switch threshold
    Vr, Vi, busphase_map = run_newton_powerflow_3p(mpc, tol=1e-6, max_iter=20)

    # 3) Create final PF results state vector
    x_f = Vr + 1j * Vi
    x = np.concatenate([np.abs(x_f), np.angle(x_f)])

    # 4) Generate the measurement data from PF results (Add noise)
    Y_pu_s, _ = build_global_y_per_unit(mpc)
    z = measurement_function(x, Y_pu_s, mpc, busphase_map)

    # Inject noise into the measurements
    std_P, std_Q, std_V = 0.0001, 0.0001, 0.00001
    num_P_inj = num_Q_inj = num_Vmag = len(busphase_map)
    num_PQ_flow = 4 * 3 * len(mpc["line3p"])
    covariance_matrix = np.diag(
        [std_P ** 2] * num_P_inj + [std_Q ** 2] * num_Q_inj + [std_P ** 2] * num_PQ_flow + [std_V ** 2] * num_Vmag)

    print("Covariance matrix shape:", covariance_matrix.shape)
    z_noisy = z + np.random.normal(0, np.sqrt(np.diag(covariance_matrix)), size=len(z))

    # --- NLM Test for a line with perturbed parameters ---

    # For primary network
    dss.Command('Redirect "Master.DSS"')
    dss.Text.Command("New LineCode.NLM nphases=3 units=mi")
    dss.Text.Command("~ Rmatrix = [0.13018  |-4.14314E-006  0.130177  |-8.11898E-006  -4.14314E-006  0.13018]")
    dss.Text.Command("~ Xmatrix = [0.0668195  |-0.000224891  2.8703539  |-4.2835E-005  -0.000224891  6.13821694]")
    dss.Text.Command("~ Cmatrix = [566.228 | 0  566.228 | 0  0  566.228]")
    dss.Text.Command("Edit line.14 Linecode=NLM")
    dss.Solution.Solve()

    # Rebuild the Y Matrix with the perturbed line
    Y_pu, _ = build_global_y_per_unit(mpc)

    # Run the Lagrangian state estimator and time it
    print("\nRunning Lagrangian state estimator...")
    start_time_se = time.time()
    x_est, success, lambdaN = run_lagrangian_polar(
        z_noisy, x_f, busphase_map, Y_pu, covariance_matrix, mpc
    )
    end_time_se = time.time()
    elapsed_time_se = end_time_se - start_time_se
    print(f"State estimation completed in {elapsed_time_se:.4f} seconds.")

    # 6) Find and print the maximum NLM for each line
    if lambdaN is not None and success:
        num_lines = len(mpc["line3p"])
        n_params_per_line = 12  # 6 for R-matrix, 6 for X-matrix

        print("\n--- Maximum Normalized Lagrangian Multipliers per Line ---")

        max_nlms_per_line = []
        for i in range(num_lines):
            # Extract the slice of lambdaN for the current line
            start_idx = i * n_params_per_line
            end_idx = start_idx + n_params_per_line

            if end_idx > len(lambdaN):
                print(f"Warning: Index out of bounds for Line {i + 1}. Skipping.")
                continue

            line_lambdaN = lambdaN[start_idx:end_idx]

            # Find the maximum absolute value in this slice
            if len(line_lambdaN) > 0:
                max_line_lambdaN = np.max(np.abs(line_lambdaN))
                max_nlms_per_line.append((i + 1, max_line_lambdaN))
            else:
                max_line_lambdaN = 0.0

            # Get the line name for better reporting (assuming it's the last column)
            line_name = mpc["line3p"][i][-1]

            print(f"Line {i + 1} ('{line_name}'): Max NLM = {max_line_lambdaN:.4g}")

        # Also print the overall largest NLM for comparison
        if max_nlms_per_line:
            overall_max_line, overall_max_val = max(max_nlms_per_line, key=lambda item: item[1])
            print("\n---------------------------------------------------------")
            print(f"Overall largest NLM is {overall_max_val:.4g} on Line {overall_max_line}")
            print("---------------------------------------------------------")

    else:
        print("State estimation failed or did not produce Lagrangian multipliers.")


if __name__ == "__main__":
    main()
