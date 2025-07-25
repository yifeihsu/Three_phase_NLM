import numpy as np
import os  # <-- Import the os module
from opendssdirect import dss
from utilities.mea_fun import measurement_function
from cal_pf import run_newton_powerflow_3p
from parse_opendss_file import parse_opendss_to_mpc, build_global_y_per_unit, merge_closed_switches_in_mpc_and_dss
from utilities.correct_parameter_group_multi_scan import correct_parameter_group_multi_scan, correct_parameters_with_known_states


def generate_and_save_measurement_data(dss_filename, lc_filename, baseMVA, slack_bus, num_scans, output_filename):
    """
    Generates multi-scan measurement data by running power flows with varying loads
    and saves the results to a compressed .npz file.

    This is the computationally intensive part that should only be run once.
    """
    print("--- Starting: Generate and Save Measurement Data ---")

    # 1) Load the base data from OpenDSS (the "correct" model)
    mpc_correct = parse_opendss_to_mpc(dss_filename, baseMVA=baseMVA, slack_bus=slack_bus, lc_filename=lc_filename)
    merge_closed_switches_in_mpc_and_dss(mpc_correct, switch_threshold=2)
    Y_pu_correct, _ = build_global_y_per_unit(mpc_correct)

    multi_scan_measurements_z = []
    initial_states_multi_scan = []
    load_variation_factor = 0.95

    # Placeholder for the first busphase_map, assuming it's constant across scans
    busphase_map = None

    print(f"--- Generating {num_scans} Measurement Scans ---")
    for i in range(num_scans):
        print(f"  Generating scan {i + 1}/{num_scans}...")
        mpc_scan = mpc_correct.copy()
        # Ensure deep copy of mutable structures like the load array
        mpc_scan["load3p"] = mpc_correct["load3p"].copy()

        current_load_factor = load_variation_factor + (i * 0.02)
        mpc_scan["load3p"][:, 3:6] *= current_load_factor

        Vr_true, Vi_true, current_busphase_map = run_newton_powerflow_3p(mpc_scan, tol=1e-6, max_iter=20)

        if busphase_map is None:
            busphase_map = current_busphase_map  # Store the map from the first run

        x_f_true = Vr_true + 1j * Vi_true
        x_true_polar = np.concatenate([np.abs(x_f_true), np.angle(x_f_true)])

        z_true = measurement_function(x_true_polar, Y_pu_correct, mpc_scan, busphase_map)

        std_P, std_Q, std_V = 0.0001, 0.0001, 0.00001
        num_P_inj = num_Q_inj = num_Vmag = len(busphase_map)

        # Calculate the number of flow measurements dynamically
        num_PQ_flow = len(z_true) - (num_P_inj + num_Q_inj + num_Vmag)

        # Build covariance vector
        R_variances_vec = np.concatenate([
            [std_P ** 2] * num_P_inj,
            [std_Q ** 2] * num_Q_inj,
            [std_P ** 2] * num_PQ_flow,  # Assuming same std for P and Q flows
            [std_V ** 2] * num_Vmag
        ])

        z_noisy = z_true + np.random.normal(0, np.sqrt(R_variances_vec))
        multi_scan_measurements_z.append(z_noisy)
        initial_states_multi_scan.append(x_true_polar)

    # Convert lists to numpy arrays and transpose for correct shape
    multi_scan_measurements_z_arr = np.array(multi_scan_measurements_z).T
    initial_states_multi_scan_arr = np.array(initial_states_multi_scan).T

    # Save all necessary data to a single compressed file
    np.savez_compressed(
        output_filename,
        multi_scan_measurements_z=multi_scan_measurements_z_arr,
        initial_states_multi_scan=initial_states_multi_scan_arr,
        R_variances_vec=R_variances_vec,
        busphase_map=busphase_map,
        # Save the correct mpc for reference if needed
        mpc_correct=mpc_correct
    )
    print(f"--- Measurement data successfully generated and saved to {output_filename} ---")


def run_parameter_correction_from_file(dss_filename_with_error, saved_data_filename, baseMVA, slack_bus, lc_filename):
    """
    Loads pre-computed measurement data and runs the parameter correction algorithm.
    """
    print("\n--- Starting: Parameter Correction Using Pre-computed Data ---")

    # 1) Load the pre-computed data
    print(f"Loading data from {saved_data_filename}...")
    try:
        data = np.load(saved_data_filename, allow_pickle=True)
        multi_scan_measurements_z = data['multi_scan_measurements_z']
        initial_states_multi_scan = data['initial_states_multi_scan']
        R_variances_vec = data['R_variances_vec']
        busphase_map = data['busphase_map'].item()  # .item() to extract from 0-d array
    except FileNotFoundError:
        print(f"Error: Data file not found at {saved_data_filename}.")
        print("Please run the script once to generate the data file.")
        return

    # 2) Introduce parameter error into the model
    print("Introducing parameter error into the DSS model...")
    dss.Command(f'Redirect "{dss_filename_with_error}"')
    dss.Command("New LineCode.NLM nphases=3 units=mi")
    dss.Command("~ Rmatrix = [0.13018  |-4.14314E-006  0.130177  |-8.11898E-006  -4.14314E-006  0.13018]")
    dss.Command("~ Xmatrix = [0.0668195  |-0.000224891  2.8703539  |-4.2835E-005  -0.000224891  6.13821694]")
    dss.Command("~ Cmatrix = [566.228 | 0  566.228 | 0  0  566.228]")
    dss.Command("Edit line.14 Linecode=NLM")
    dss.Solution.Solve()

    # 3) Parse the model with the error
    mpc_with_error = parse_opendss_to_mpc(dss_filename_with_error, baseMVA=baseMVA, slack_bus=slack_bus,
                                          lc_filename=lc_filename)
    merge_closed_switches_in_mpc_and_dss(mpc_with_error, switch_threshold=2)

    # 4) Set up and run the correction algorithm
    line_name_to_correct = '14'
    line_to_correct_idx = np.where(mpc_with_error["line3p"][:, 6] == line_name_to_correct.lower())[0][0]
    print(f"\n--- Starting Group Parameter Correction for line index {line_to_correct_idx} ---")

    corrected_params, success = correct_parameters_with_known_states(
        dss_filename_with_error, mpc_with_error, line_to_correct_idx,
        multi_scan_measurements_z, initial_states_multi_scan,
        R_variances_vec, busphase_map)

    if success:
        print("\n--- Parameter Correction Successful! ---")
        print("Corrected R/mi (p.u.):", corrected_params[:6])
        print("Corrected X/mi (p.u.):", corrected_params[6:])
    else:
        print("\n--- Parameter Correction Failed ---")


def main():
    # --- Configuration ---
    dss_filename = "Master.DSS"
    lc_filename = "LineConstantsCode.txt"
    baseMVA = 100
    slack_bus = "p1"
    num_scans = 20

    # Define a filename for the pre-computed data
    saved_data_filename = "multi_scan_data.npz"

    # --- Workflow ---

    # 1. Check for pre-computed data. If it doesn't exist, generate it.
    if not os.path.exists(saved_data_filename):
        print(f"Data file '{saved_data_filename}' not found. Generating new data...")
        generate_and_save_measurement_data(
            dss_filename,
            lc_filename,
            baseMVA,
            slack_bus,
            num_scans,
            saved_data_filename
        )
    else:
        print(f"Found pre-computed data file: '{saved_data_filename}'. Skipping generation.")

    # 2. Run the analysis using the (now existing) data file.
    # We use the same dss_filename because the error is introduced programmatically.
    run_parameter_correction_from_file(
        dss_filename,
        saved_data_filename,
        baseMVA,
        slack_bus,
        lc_filename
    )


if __name__ == "__main__":
    main()