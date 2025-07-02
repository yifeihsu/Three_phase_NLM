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
    """
    This script simulates a cable burn-out scenario to test the parameter
    error detection capabilities of a Lagrangian-based state estimator.

    The process is as follows:
    1. Simulate Reality (Cable Burn-out): Modify the OpenDSS model to
       represent a "burn-out" of several parallel cables at Line 208 by
       disabling them. Run a power flow on this modified grid to get the
       "true" system state (voltages, currents, etc.).
    2. Generate Measurements: Based on this "true" state from the burnt-out
       model, generate a set of measurements (power injections, flows,
       voltage magnitudes) and add realistic noise.
    3. Assume Correct Model: Load the original, unmodified OpenDSS model,
       which represents the "assumed" or "correct" state of the grid from
       an operator's perspective (i.e., they don't know about the burn-out).
    4. Run State Estimation: Use the measurements from the burnt-out reality
       (Step 2) as input to the state estimator, which operates on the
       assumed correct model (Step 3).
    5. Detect Anomaly: Analyze the Lagrange multipliers from the state
       estimation. A large multiplier value is expected to correspond to the
       Kirchhoff's law constraints of the line that was burnt out, thus
       identifying the location of the modeling error.
    """
    dss_filename = "Master.DSS"
    dss_filename_ft = "Master_ft.DSS"
    baseMVA = 1.0
    slack_bus = "p1"

    # lines_to_burn_out = ["208_2", "208_3", "208_4"]
    lines_to_burn_out = ["208_4"]

    # --- 1) SIMULATE REALITY: CABLE BURN-OUT AND POWER FLOW ---
    print("--- Step 1: Simulating the true system with burnt-out cables ---")

    # Use opendssdirect to compile the master file and then disable the lines
    dss.Command("clear")
    dss.Command(f'Redirect "{dss_filename_ft}"')

    # print(f"Disabling the following parallel cables: {', '.join(lines_to_burn_out)}")
    # for line_name_suffix in lines_to_burn_out:
    #     dss.Command(f"Edit line.{line_name_suffix} enabled=no")

    dss.Solution.Solve()

    print("Parsing the 'true' (burnt-out) model from OpenDSS...")
    # NOTE: We assume parse_opendss_to_mpc works on the active dss circuit
    mpc_true = parse_opendss_to_mpc(dss_filename_ft, baseMVA=baseMVA, slack_bus=slack_bus)
    merge_closed_switches_in_mpc_and_dss(mpc_true, switch_threshold=2)

    Vr_true, Vi_true, busphase_map_true = run_newton_powerflow_3p(mpc_true, tol=1e-6, max_iter=20)
    x_f_true = Vr_true + 1j * Vi_true
    x_true = np.concatenate([np.abs(x_f_true), np.angle(x_f_true)])
    print("...Power flow for the burnt-out scenario is complete.")

    # --- 2) GENERATE NOISY MEASUREMENTS FROM THE "TRUE" STATE -----------------
    print("\n--- Step 2: Generating noisy measurements from the true state ---")

    Y_pu_true, _ = build_global_y_per_unit(mpc_true)
    z = measurement_function(x_true, Y_pu_true, mpc_true, busphase_map_true)

    num_bus_phases = len(busphase_map_true)
    num_P_inj = num_Q_inj = num_Vmag = num_bus_phases

    # standard deviations
    std_P, std_Q, std_V = 1e-4, 1e-4, 1e-4

    diag_R = (
            [std_P ** 2] * num_P_inj +
            [std_Q ** 2] * num_Q_inj +
            [std_V ** 2] * num_Vmag
    )
    R = np.diag(diag_R)  # <-- renamed

    z_noisy = z + np.random.normal(0.0, np.sqrt(diag_R))
    print("...Noisy measurements generated.")

    # --- 3) SETUP THE "ASSUMED CORRECT" (UNMODIFIED) MODEL ---
    print("\n--- Step 3: Setting up the assumed-correct (unmodified) model for estimation ---")

    # Re-compile the original master file to get the model WITHOUT the burn-out
    # This represents the model the operator believes to be correct.
    dss.Command("clear")
    mpc_assumed = parse_opendss_to_mpc(dss_filename, baseMVA=baseMVA, slack_bus=slack_bus)
    merge_closed_switches_in_mpc_and_dss(mpc_assumed, switch_threshold=2)

    # Build the admittance matrix for this assumed-correct model
    Y_pu_assumed, _ = build_global_y_per_unit(mpc_assumed)

    # Calculate a power flow solution for the assumed model. This will be used
    # as the initial guess (flat start) for the state estimator.
    Vr_assumed, Vi_assumed, busphase_map_assumed = run_newton_powerflow_3p(mpc_assumed, tol=1e-6, max_iter=20)
    x_f_assumed = Vr_assumed + 1j * Vi_assumed
    print("...Assumed-correct model is ready.")

    # --- 4) RUN LAGRANGIAN ESTIMATOR TO FIND THE DISCREPANCY ---
    print("\n--- Step 4: Running Lagrangian Polar State Estimator ---")

    # We provide the estimator with:
    # - z_noisy: Measurements from the "real" (burnt-out) world.
    # - x_f_assumed: An initial state guess from the "assumed" world.
    # - Y_pu_assumed, mpc_assumed: The model of the "assumed" world.
    x_est, success, lambdaN = run_lagrangian_polar(
        z_noisy, x_f_assumed, busphase_map_assumed,
        Y_pu_assumed, R,  # <-- pass R, not covariance_matrix
        mpc_assumed
    )

    if not success:
        print("!!! State estimation did not converge, which may be expected due to the model error.")
    else:
        print("...State estimation converged.")

    print("\n--- Step 5: Analyzing Lagrangian multipliers to locate the error ---")

    if lambdaN is None or len(lambdaN) == 0:
        print("Estimation did not produce Lagrange multipliers. Cannot locate error.")
        return

    largest_idx = np.argmax(np.abs(lambdaN))
    largest_val = lambdaN[largest_idx]

    # The structure of lambdaN is assumed to be 12 parameters per line (6 for R, 6 for X)
    line_index_in_mpc = largest_idx // 12
    param_index_within_line = largest_idx % 12

    # Determine if it's an R or X matrix error
    error_type = "X (reactance)" if param_index_within_line >= 6 else "R (resistance)"

    # The index within the 3x3 symmetric matrix (represented as a 6-element vector)
    param_index_in_matrix = param_index_within_line % 6

    print(f"Largest Lagrangian param => index={(largest_idx%12 + 1)%6}, value={largest_val:.4g}, "
          f"line={largest_idx//12 + 1}, error_type={error_type}")
    # Get the line name from the mpc_assumed structure for better reporting
    # try:
    #     line_info = mpc_assumed['line3p'][line_index_in_mpc]
    #     line_name = line_info['name']
    #     from_bus = line_info['from_bus_name']
    #     to_bus = line_info['to_bus_name']
    # except IndexError:
    #     print(f"Error: Cannot find line with MPC index {line_index_in_mpc}.")
    #     return



if __name__ == "__main__":
    main()