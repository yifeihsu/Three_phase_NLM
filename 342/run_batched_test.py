import numpy as np
import re
import time
import os  # <-- Import the os module for file operations
from opendssdirect import dss

# Assume the following custom modules are in the path
from lagrangian_m import run_lagrangian_polar
from utilities.mea_fun import measurement_function
from cal_pf import run_newton_powerflow_3p
from parse_opendss_file import parse_opendss_to_mpc, build_global_y_per_unit, merge_closed_switches_in_mpc_and_dss


def create_faulted_dss_file(original_dss_path, line_to_burn_out, temp_dss_path):
    """
    Reads an original DSS file and writes a temporary version with one line commented out.

    Args:
        original_dss_path (str): Path to the source Master.DSS file.
        line_to_burn_out (str): The name of the line to disable (e.g., '136_2').
        temp_dss_path (str): The path where the temporary faulted DSS file will be written.
    """
    # Compile a regular expression to find the exact line definition.
    # It looks for "new Line.[line_name]" at the start of a line, ignoring case and whitespace.
    line_regex = re.compile(r"^\s*new\s+Line\." + re.escape(line_to_burn_out), re.IGNORECASE)

    with open(original_dss_path, 'r') as f_in, open(temp_dss_path, 'w') as f_out:
        for line in f_in:
            if line_regex.search(line):
                # If the line matches, comment it out with '!'
                f_out.write('! ' + line)
            else:
                # Otherwise, write the line as is.
                f_out.write(line)


def find_parallel_line_groups(dss_file_path):
    """
    Parses a DSS file to find groups of parallel lines, excluding switches.
    (This function is correct and remains unchanged)
    """
    try:
        with open(dss_file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: DSS file not found at {dss_file_path}")
        return {}
    all_line_names = re.findall(r"new\s+Line\.([\w_]+)", content, re.IGNORECASE)
    cable_names = [name for name in all_line_names if not name.lower().endswith('_sw')]
    groups = {}
    for name in cable_names:
        base_name = name.split('_')[0]
        if base_name not in groups:
            groups[base_name] = []
        groups[base_name].append(name)
    parallel_groups = {base: lines for base, lines in groups.items() if len(lines) > 1}
    return parallel_groups


def run_single_burnout_test(dss_filename, line_to_burn_out, true_base_name, mpc_assumed, x_f_assumed,
                            busphase_map_assumed, Y_pu_assumed):
    """
    Runs a single burnout test by generating a temporary faulted DSS file for parsing.
    """
    baseMVA = 1.0
    slack_bus = "p1"
    temp_dss_filename = f"temp_fault_{line_to_burn_out}.dss"

    try:
        # --- PRE-STEP: Generate the faulted DSS file ---
        create_faulted_dss_file(dss_filename, line_to_burn_out, temp_dss_filename)

        # --- 1) SIMULATE REALITY (using the temp file) ---

        # First, check if the power flow converges for this reality
        dss.Command("Clear")
        dss.Command(f'Redirect "{temp_dss_filename}"')
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            print(
                f"    - WARNING: Power flow for reality with '{line_to_burn_out}' burnt-out did not converge. Skipping test.")
            return None

        # Now parse the state from the faulted file
        mpc_true = parse_opendss_to_mpc(temp_dss_filename, baseMVA=baseMVA, slack_bus=slack_bus)

        # The merge function also modifies the live DSS object, which is fine here.
        merge_closed_switches_in_mpc_and_dss(mpc_true, switch_threshold=2)

        Vr_true, Vi_true, busphase_map_true = run_newton_powerflow_3p(mpc_true, tol=1e-6, max_iter=20)
        x_f_true = Vr_true + 1j * Vi_true
        x_true = np.concatenate([np.abs(x_f_true), np.angle(x_f_true)])

        # --- 2) GENERATE NOISY MEASUREMENTS ---
        Y_pu_true, _ = build_global_y_per_unit(mpc_true)
        z = measurement_function(x_true, Y_pu_true, mpc_true, busphase_map_true)
        num_bus_phases = len(busphase_map_true)
        std_P, std_Q, std_V = 1e-4, 1e-4, 1e-5
        diag_R = ([std_P ** 2] * num_bus_phases + [std_Q ** 2] * num_bus_phases + [std_V ** 2] * num_bus_phases)
        R = np.diag(diag_R)
        z_noisy = z + np.random.normal(0.0, np.sqrt(diag_R))

        # --- 3) RUN ESTIMATOR ---
        x_est, success, _, line_distances = run_lagrangian_polar(
            z_noisy, x_f_assumed, busphase_map_assumed, Y_pu_assumed, R, mpc_assumed
        )

        # --- 4) ANALYZE RESULTS ---
        if not success or line_distances is None or len(line_distances) == 0:
            print(
                f"    - DETECTION FAILED: Estimator did not converge or produce distances for line {line_to_burn_out}.")
            return False

        valid_distances = [d for d in line_distances if np.isfinite(d[1])]
        if not valid_distances:
            print(f"    - DETECTION FAILED: All computed distances were NaN or Inf.")
            return False

        detected_line_index_mpc, max_dist = max(valid_distances, key=lambda item: item[1])
        detected_line_info = mpc_assumed['line3p'][detected_line_index_mpc]
        detected_line_name = detected_line_info[-1]

        is_successful = detected_line_name.lower() == true_base_name.lower()

        if is_successful:
            print(
                f"    - DETECTION SUCCESS: Fault on '{line_to_burn_out}' correctly identified as Line '{detected_line_name}' (Max Distance: {max_dist:.3f}).")
        else:
            print(
                f"    - DETECTION FAILED: Fault on '{line_to_burn_out}' was misidentified as Line '{detected_line_name}' (Max Distance: {max_dist:.3f}).")

        return is_successful

    except Exception as e:
        print(f"    - An unexpected error occurred during test for '{line_to_burn_out}': {e}")
        return False
    finally:
        # --- 5) CLEANUP: Ensure temporary file is always deleted ---
        if os.path.exists(temp_dss_filename):
            os.remove(temp_dss_filename)


def main():
    """
    Main function to orchestrate the representative cable burnout analysis.
    """
    dss_filename = "Master.DSS"
    baseMVA = 1.0
    slack_bus = "p1"

    print("--- Step 1: Identifying all parallel line groups from DSS file ---")
    parallel_line_groups = find_parallel_line_groups(dss_filename)
    if not parallel_line_groups:
        print("No parallel line groups found to test. Exiting.")
        return
    print(f"Found {len(parallel_line_groups)} groups of parallel lines to be tested.")

    print("\n--- Step 2: Preparing the 'assumed-correct' base model ---")
    mpc_assumed = parse_opendss_to_mpc(dss_filename, baseMVA=baseMVA, slack_bus=slack_bus)
    merge_closed_switches_in_mpc_and_dss(mpc_assumed, switch_threshold=2)
    Vr_assumed, Vi_assumed, busphase_map_assumed = run_newton_powerflow_3p(mpc_assumed, tol=1e-6, max_iter=20)
    x_f_assumed = Vr_assumed + 1j * Vi_assumed
    Y_pu_assumed, _ = build_global_y_per_unit(mpc_assumed)
    print("...'Assumed-correct' model and initial state are ready.\n")

    print("--- Step 3: Running representative burnout simulation for each line group ---")

    successful_detections = 0
    total_tests = 0

    for base_name, cables in parallel_line_groups.items():
        total_tests += 1
        cable_to_burn = cables[1]

        print(f"\nTesting representative burnout for Line Group '{base_name}' (disabling '{cable_to_burn}')...")

        result = run_single_burnout_test(
            dss_filename=dss_filename,
            line_to_burn_out=cable_to_burn,
            true_base_name=base_name,
            mpc_assumed=mpc_assumed,
            x_f_assumed=x_f_assumed,
            busphase_map_assumed=busphase_map_assumed,
            Y_pu_assumed=Y_pu_assumed
        )

        if result is True:
            successful_detections += 1
        elif result is None:
            total_tests -= 1

        time.sleep(0.1)

    print("\n" + "---" * 12)
    print("---   Representative Analysis Report   ---")
    print("---" * 12)

    if total_tests > 0:
        success_rate = (successful_detections / total_tests) * 100
        print(f"Total Groups Tested:    {total_tests}")
        print(f"Successful Detections:  {successful_detections}")
        print(f"Failed Detections:      {total_tests - successful_detections}")
        print(f"Detection Success Rate: {success_rate:.2f}%")
    else:
        print("No valid tests were completed.")


if __name__ == "__main__":
    main()