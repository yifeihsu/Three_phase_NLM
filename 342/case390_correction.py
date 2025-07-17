import numpy as np
from opendssdirect import dss
from utilities.mea_fun import measurement_function
from cal_pf import run_newton_powerflow_3p
from parse_opendss_file import parse_opendss_to_mpc, build_global_y_per_unit, merge_closed_switches_in_mpc_and_dss
from utilities.correct_parameter_group_multi_scan import correct_parameter_group_multi_scan
from lagrangian_m import run_lagrangian_polar


def main():
    # 1) Load the base data from OpenDSS
    dss_filename = "Master.DSS"
    lc_filename = "LineConstantsCode.txt"
    baseMVA = 1.0
    slack_bus = "p1"

    mpc_correct = parse_opendss_to_mpc(dss_filename, baseMVA=baseMVA, slack_bus=slack_bus, lc_filename=lc_filename)
    merge_closed_switches_in_mpc_and_dss(mpc_correct, switch_threshold=2)
    Y_pu_correct, _ = build_global_y_per_unit(mpc_correct)

    num_scans = 3
    multi_scan_measurements_z = []
    initial_states_multi_scan = []
    load_variation_factor = 0.95

    print(f"--- Generating {num_scans} Measurement Scans ---")
    for i in range(num_scans):
        print(f"  Generating scan {i + 1}/{num_scans}...")
        mpc_scan = mpc_correct.copy()
        mpc_scan["load3p"] = mpc_correct["load3p"].copy()

        current_load_factor = load_variation_factor + (i * 0.02)
        mpc_scan["load3p"][:, 3:6] *= current_load_factor

        Vr_true, Vi_true, busphase_map = run_newton_powerflow_3p(mpc_scan, tol=1e-6, max_iter=20)
        x_f_true = Vr_true + 1j * Vi_true
        x_true_polar = np.concatenate([np.abs(x_f_true), np.angle(x_f_true)])

        z_true = measurement_function(x_true_polar, Y_pu_correct, mpc_scan, busphase_map)

        std_P, std_Q, std_V = 0.0001, 0.0001, 0.00001
        num_P_inj = num_Q_inj = num_Vmag = len(busphase_map)
        num_PQ_flow = 4 * 3 * len(mpc_scan["line3p"])

        if len(z_true) != (num_P_inj + num_Q_inj + num_PQ_flow + num_Vmag):
            num_PQ_flow = len(z_true) - (num_P_inj + num_Q_inj + num_Vmag)

        covariance_matrix = np.diag(
            [std_P ** 2] * num_P_inj + [std_Q ** 2] * num_Q_inj +
            [std_P ** 2] * num_PQ_flow + [std_V ** 2] * num_Vmag
        )
        R_variances_vec = np.diag(covariance_matrix)

        z_noisy = z_true + np.random.normal(0, np.sqrt(R_variances_vec))
        multi_scan_measurements_z.append(z_noisy)
        initial_states_multi_scan.append(x_true_polar)

    multi_scan_measurements_z = np.array(multi_scan_measurements_z).T
    initial_states_multi_scan = np.array(initial_states_multi_scan).T
    print("--- Multi-scan measurement generation complete ---")

    print("\n--- Introducing parameter error into the model ---")
    dss.Command('Redirect "Master.DSS"')
    dss.Command("New LineCode.NLM nphases=3 units=mi")
    dss.Command("~ Rmatrix = [0.13018  |-4.14314E-006  0.130177  |-8.11898E-006  -4.14314E-006  0.13018]")
    dss.Command("~ Xmatrix = [0.0668195  |-0.000224891  2.8703539  |-4.2835E-005  -0.000224891  6.13821694]")
    dss.Command("~ Cmatrix = [566.228 | 0  566.228 | 0  0  566.228]")
    dss.Command("Edit line.14 Linecode=NLM")
    dss.Solution.Solve()

    # ### Rebuild the Y Matrix *!
    # Y_pu, _ = build_global_y_per_unit(mpc_scan)
    # x_est, success, lambdaN = run_lagrangian_polar(
    #     z_noisy, x_f_true, busphase_map, Y_pu, covariance_matrix, mpc_scan
    # )
    # # 6) find the largest index
    # largest_idx = np.argmax(np.abs(lambdaN))
    # largest_val = lambdaN[largest_idx]
    # error_type = "X" if (largest_idx%12 + 1)//6 == 1 else "R"
    # print(f"Largest Lagrangian param => index={(largest_idx%12 + 1)%6}, value={largest_val:.4g}, "
    #       f"line={largest_idx//12 + 1}, error_type={error_type}")

    mpc_with_error = parse_opendss_to_mpc(dss_filename, baseMVA=baseMVA, slack_bus=slack_bus, lc_filename=lc_filename)
    merge_closed_switches_in_mpc_and_dss(mpc_with_error, switch_threshold=2)

    line_name_to_correct = '14'
    line_to_correct_idx = np.where(mpc_with_error["line3p"][:, 6] == line_name_to_correct.lower())[0][0]

    print(f"\n--- Starting Group Parameter Correction for line index {line_to_correct_idx} ---")

    corrected_params, success = correct_parameter_group_multi_scan(
        dss_filename,
        mpc_with_error,
        line_to_correct_idx,
        multi_scan_measurements_z,
        initial_states_multi_scan,
        R_variances_vec,
        busphase_map
    )

    if success:
        print("\n--- Parameter Correction Successful! ---")
        print("Corrected Parameters (R, X):")
        print("R:", corrected_params[:6])
        print("X:", corrected_params[6:])
    else:
        print("\n--- Parameter Correction Failed to Converge ---")


if __name__ == "__main__":
    main()