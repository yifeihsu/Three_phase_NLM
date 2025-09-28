import os
import numpy as np
import time
import copy
from opendssdirect import dss
from lagrangian_m import run_lagrangian_polar
from utilities.mea_fun import measurement_function
from cal_pf import run_newton_powerflow_3p
from parse_opendss_file import parse_opendss_to_mpc, build_global_y_per_unit, merge_closed_switches_in_mpc_and_dss
from config import DATA_DIR

# --- Define MPC constants for clarity ---
# See MATPOWER documentation for column definitions
# https://matpower.org/docs/ref/matpower7.0/mpc.html
LINE_STATUS = 10  # bus 'i', bus 'j', R, X, B, rateA, rateB, rateC, ratio, angle, status


def main():
    """
    This script simulates an open main fault, uses a Lagrangian-based state
    estimator to detect the anomaly, and then uses the true fault model to
    verify that the anomaly disappears when the correct model is used.
    """
    dss_filename = os.fspath(DATA_DIR / "Master.dss")
    dss_filename_ft = os.fspath(DATA_DIR / "Master_Circuit.dss")
    baseMVA = 1.0
    slack_bus = "p1"

    # --- 1) SIMULATE REALITY: OPEN MAIN FAULT AND POWER FLOW ---
    print("--- Step 1: Simulating the true system with an open main fault ---")
    print("Parsing the 'true' (faulted) model from OpenDSS...")
    mpc_true = parse_opendss_to_mpc(
        dss_filename_ft, baseMVA=baseMVA,
        lc_filename=os.fspath(DATA_DIR / "LineConstantsCode.txt"),
        slack_bus=slack_bus,
    )
    merge_closed_switches_in_mpc_and_dss(mpc_true, switch_threshold=2)

    Vr_true, Vi_true, busphase_map_true = run_newton_powerflow_3p(mpc_true, tol=1e-6, max_iter=20)
    x_f_true = Vr_true + 1j * Vi_true
    x_true = np.concatenate([np.abs(x_f_true), np.angle(x_f_true)])
    print("...Power flow for the faulted scenario is complete.")

    # --- 2) GENERATE NOISY MEASUREMENTS FROM THE "TRUE" STATE ---
    print("\n--- Step 2: Generating noisy measurements from the true state ---")

    Y_pu_true, _ = build_global_y_per_unit(mpc_true)
    # We only consider injection measurements here, so the number of measurements won't change for the true model.
    z = measurement_function(x_true, Y_pu_true, mpc_true, busphase_map_true)

    num_bus_phases = len(busphase_map_true)
    num_P_inj = num_Q_inj = num_Vmag = num_bus_phases

    # Standard deviations for measurements
    std_P, std_Q, std_V = 1e-4, 1e-4, 1e-4

    # Define measurement covariance matrix R
    diag_R = (
            [std_P ** 2] * num_P_inj
            + [std_Q ** 2] * num_Q_inj
            + [std_V ** 2] * num_Vmag
    )
    R = np.diag(diag_R)

    # Add Gaussian noise to the true measurements
    z_noisy = z + np.random.normal(0.0, np.sqrt(np.diag(R)), size=len(z))
    print("...Noisy measurements generated.")

    # --- 3) SETUP THE "ASSUMED CORRECT" (PRE-FAULT) MODEL ---
    print("\n--- Step 3: Setting up the assumed-correct (unmodified) model for estimation ---")

    dss.Command("clear")
    mpc_assumed = parse_opendss_to_mpc(
        dss_filename, baseMVA=baseMVA,
        lc_filename=os.fspath(DATA_DIR / "LineConstantsCode.txt"),
        slack_bus=slack_bus,
    )
    merge_closed_switches_in_mpc_and_dss(mpc_assumed, switch_threshold=2)

    Y_pu_assumed, _ = build_global_y_per_unit(mpc_assumed)

    # Get a starting point for the state estimator
    Vr_assumed, Vi_assumed, busphase_map_assumed = run_newton_powerflow_3p(mpc_assumed, tol=1e-6, max_iter=20)
    x_f_assumed = Vr_assumed + 1j * Vi_assumed
    print("...Assumed-correct model is ready.")

    # --- 4) RUN STATE ESTIMATOR ON ASSUMED MODEL TO FIND DISCREPANCY ---
    print("\n--- Step 4: Running Lagrangian Polar State Estimator on the assumed (pre-fault) model ---")

    x_est_initial, success_initial, lambdaN_initial, _ = run_lagrangian_polar(
        z_noisy, x_f_assumed, busphase_map_assumed,
        Y_pu_assumed, R,
        mpc_assumed
    )

    if not success_initial:
        print("!!! State estimation did not converge, which is expected due to the large model error.")
    else:
        print("...State estimation converged, but we expect large anomaly indices.")

    # --- 5) ANALYZE ANOMALY INDICES TO LOCATE THE FAULT ---
    print("\n--- Step 5: Analyzing anomaly indices (Lagrange Multipliers) to locate the error ---")

    if lambdaN_initial is None or len(lambdaN_initial) == 0:
        print("Estimation did not produce Lagrange multipliers. Cannot locate error.")
        return

    # The norm of the Lagrange multipliers serves as an overall anomaly score
    anomaly_score_initial = max(abs(lambdaN_initial))
    print(f"Initial Anomaly Score (Norm of NLM): {anomaly_score_initial:.4f}")

    # Find the individual parameter with the largest error signal for reporting
    largest_idx = np.argmax(np.abs(lambdaN_initial))
    line_index_in_mpc = largest_idx // 12
    print(f"Highest individual anomaly index corresponds to Line Index {line_index_in_mpc}.")

    # --- 6) VERIFY FAULT BY RE-RUNNING SE ON THE TRUE FAULTED MODEL ---
    print("\n--- Step 6: Verifying that measurements match the true (faulted) model ---")
    print("Re-running state estimator using the ground-truth model (mpc_true)...")
    dss.Command("Clear")
    x_est_verify, success_verify, lambdaN_verify, _ = run_lagrangian_polar(
        z_noisy, x_f_true, busphase_map_true,
        Y_pu_true, R,
        mpc_true
    )

    if not success_verify:
        print("!!! Verification failed: State estimation did not converge on the true model.")
    else:
        print("...State estimation on the true model converged successfully.")
        # Find the largest lambdaN index in the verified model
        anomaly_score_verified = max(abs((lambdaN_verify)))

        print("\n--- Verification Results ---")
        print(f"Initial Anomaly Score (using pre-fault model): {anomaly_score_initial:.4f}")
        print(f"Verified Anomaly Score (using true faulted model): {anomaly_score_verified:.4f}")
if __name__ == "__main__":
    main()
