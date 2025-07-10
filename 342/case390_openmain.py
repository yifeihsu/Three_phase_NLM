import numpy as np
import time
from opendssdirect import dss
from lagrangian_m import run_lagrangian_polar
from utilities.mea_fun import measurement_function
from cal_pf import run_newton_powerflow_3p
from parse_opendss_file import parse_opendss_to_mpc, build_global_y_per_unit, merge_closed_switches_in_mpc_and_dss

def main():
    """
    This script simulates an open main fault scenario where all parallel lines
    of a given line are opened. It then uses a Lagrangian-based state
    estimator with all measurement types (P, Q, and V) to detect and
    identify the fault.
    """
    dss_filename = "Master.DSS"
    dss_filename_ft = "Master_Circuit.dss"
    baseMVA = 1.0
    slack_bus = "p1"

    # Define the base name of the line and all its parallel instances
    lines_to_open = ["280", "280_2", "280_3", "280_4", "280_5", "280_6"]

    # --- 1) SIMULATE REALITY: OPEN MAIN FAULT AND POWER FLOW ---
    print("--- Step 1: Simulating the true system with an open main fault on all parallel lines ---")
    print("Parsing the 'true' (faulted) model from OpenDSS...")
    mpc_true = parse_opendss_to_mpc(dss_filename_ft, baseMVA=baseMVA, slack_bus=slack_bus)
    merge_closed_switches_in_mpc_and_dss(mpc_true, switch_threshold=2)

    Vr_true, Vi_true, busphase_map_true = run_newton_powerflow_3p(mpc_true, tol=1e-6, max_iter=20)
    x_f_true = Vr_true + 1j * Vi_true
    x_true = np.concatenate([np.abs(x_f_true), np.angle(x_f_true)])
    print("...Power flow for the faulted scenario is complete.")

    # --- 2) GENERATE NOISY MEASUREMENTS FROM THE "TRUE" STATE -----------------
    print("\n--- Step 2: Generating noisy measurements from the true state ---")

    Y_pu_true, _ = build_global_y_per_unit(mpc_true)
    z = measurement_function(x_true, Y_pu_true, mpc_true, busphase_map_true)

    num_bus_phases = len(busphase_map_true)
    num_P_inj = num_Q_inj = num_Vmag = num_bus_phases
    num_PQ_flow = 4 * 3 * len(mpc_true["line3p"])

    # standard deviations
    std_P, std_Q, std_V = 1e-4, 1e-4, 1e-4

    # Include all measurement types (P, Q, V) in the covariance matrix
    diag_R = (
        [std_P ** 2] * num_P_inj
        + [std_Q ** 2] * num_Q_inj
        + [std_V ** 2] * num_Vmag
    )
    R = np.diag(diag_R)

    z_noisy = z + np.random.normal(0.0, np.sqrt(np.diag(R)), size=len(z))
    print("...Noisy measurements generated.")

    # --- 3) SETUP THE "ASSUMED CORRECT" (UNMODIFIED) MODEL ---
    print("\n--- Step 3: Setting up the assumed-correct (unmodified) model for estimation ---")

    dss.Command("clear")
    mpc_assumed = parse_opendss_to_mpc(dss_filename, baseMVA=baseMVA, slack_bus=slack_bus)
    merge_closed_switches_in_mpc_and_dss(mpc_assumed, switch_threshold=2)

    Y_pu_assumed, _ = build_global_y_per_unit(mpc_assumed)

    Vr_assumed, Vi_assumed, busphase_map_assumed = run_newton_powerflow_3p(mpc_assumed, tol=1e-6, max_iter=20)
    x_f_assumed = Vr_assumed + 1j * Vi_assumed
    print("...Assumed-correct model is ready.")

    # --- 4) RUN LAGRANGIAN ESTIMATOR TO FIND THE DISCREPANCY ---
    print("\n--- Step 4: Running Lagrangian Polar State Estimator ---")

    x_est, success, lambdaN = run_lagrangian_polar(
        z_noisy, x_f_assumed, busphase_map_assumed,
        Y_pu_assumed, R,
        mpc_assumed
    )

    if not success:
        print("!!! State estimation did not converge, which may be expected due to the model error.")
    else:
        print("...State estimation converged.")

    print("\n--- Step 5: Analyzing results to locate the error ---")

    if lambdaN is None or len(lambdaN) == 0:
        print("Estimation did not produce Lagrange multipliers. Cannot locate error.")
        return

    # Analyze the individual NLM values
    largest_idx = np.argmax(np.abs(lambdaN))
    largest_val = lambdaN[largest_idx]
    line_index_in_mpc = largest_idx // 12
    error_type = "X (reactance)" if (largest_idx % 12) >= 6 else "R (resistance)"

    print(f"Largest Individual NLM value: {largest_val:.4g}")
    print(f"  - Corresponding Line Index in MPC: {line_index_in_mpc + 1}")
    print(f"  - Parameter type: {error_type}")

    # Analyze the grouped Mahalanobis distance
    print("\n--- Calculating Grouped Index Mahalanobis Distances ---")
    # line_distances = compute_line_mahalanobis_distances(lambdaN, np.eye(len(lambdaN)), mpc_assumed)

    # max_dist = -1
    # faulty_line_index = -1
    # for i_line, dist in line_distances:
    #     if dist > max_dist:
    #         max_dist = dist
    #         faulty_line_index = i_line + 1
    #
    # print(f"\nLine with the largest Mahalanobis distance: Line {faulty_line_index}")
    # print(f"  - Mahalanobis distance value: {max_dist:.4f}")
    #
    # if faulty_line_index == 280:
    #     print("\nSuccessfully identified the faulted line!")
    # else:
    #     print("\nCould not identify the faulted line correctly.")


if __name__ == "__main__":
    main()