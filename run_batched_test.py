import numpy as np
import re
import time
import os
from opendssdirect import dss
from lagrangian_m import run_lagrangian_polar
from utilities.mea_fun import measurement_function
from cal_pf import run_newton_powerflow_3p
from parse_opendss_file import parse_opendss_to_mpc, build_global_y_per_unit, merge_closed_switches_in_mpc_and_dss
from copy import deepcopy  # NEW


def reorder_measurements_to_map(z, src_map, dst_map, fill_v=1.0):
    """
    Reorder measurement vector z=[P;Q;|V|] from src_map ordering to dst_map ordering.
    fill_v is used for |V| if a (bus,phase) key is missing in src_map.
    """
    N_dst = len(dst_map)
    # split source
    N_src = len(src_map)
    P_src = z[:N_src]
    Q_src = z[N_src:2*N_src]
    V_src = z[2*N_src:2*N_src + N_src]

    P_dst = np.zeros(N_dst, dtype=float)
    Q_dst = np.zeros(N_dst, dtype=float)
    V_dst = np.full(N_dst, float(fill_v))

    missing = []
    # map by keys
    for key, j_dst in dst_map.items():
        if key in src_map:
            i_src = src_map[key]
            P_dst[j_dst] = P_src[i_src]
            Q_dst[j_dst] = Q_src[i_src]
            V_dst[j_dst] = V_src[i_src]
        else:
            missing.append(key)
    if missing:
        print(f"[reorder_measurements_to_map] WARNING: {len(missing)} nodes exist in dst_map but not in src_map; "
              f"P/Q set to 0.0, |V| set to {fill_v}. Examples: {missing[:5]}{'...' if len(missing)>5 else ''}")

    return np.concatenate([P_dst, Q_dst, V_dst])


def build_mpc_with_measured_injections(mpc_base, busphase_map, P_meas_by_key, Q_meas_by_key):
    """
    Return a deep-copied MPC whose per-(bus,phase) net injections equal measured values.
    Positive => generator; negative => load. Always keep a generator row for slack buses.
    """
    mpc_pf = deepcopy(mpc_base)

    # bus type map
    bus_type = {int(row[0]): int(row[1]) for row in mpc_pf["bus3p"]}

    load3p_pf, gen3p_pf = [], []
    load_id, gen_id = 1, 1

    bus_ids = [int(row[0]) for row in mpc_pf["bus3p"]]
    for bid in bus_ids:
        # gather per-phase measured injections (defaults 0 if missing)
        def Pv(ph): return P_meas_by_key.get((bid, ph), 0.0)
        def Qv(ph): return Q_meas_by_key.get((bid, ph), 0.0)

        P_A, P_B, P_C = Pv(0), Pv(1), Pv(2)
        Q_A, Q_B, Q_C = Qv(0), Qv(1), Qv(2)

        # split into loads (negative part) and gens (positive part)
        PdA, PdB, PdC = max(-P_A, 0.0), max(-P_B, 0.0), max(-P_C, 0.0)
        QdA, QdB, QdC = max(-Q_A, 0.0), max(-Q_B, 0.0), max(-Q_C, 0.0)
        PgA, PgB, PgC = max(P_A, 0.0),  max(P_B, 0.0),  max(P_C, 0.0)
        QgA, QgB, QgC = max(Q_A, 0.0),  max(Q_B, 0.0),  max(Q_C, 0.0)

        if (PdA + PdB + PdC + QdA + QdB + QdC) > 0:
            load3p_pf.append([load_id, bid, 1, PdA, PdB, PdC, QdA, QdB, QdC])
            load_id += 1

        is_slack = (bus_type.get(bid, 1) == 3)
        if (PgA + PgB + PgC + QgA + QgB + QgC) > 0 or is_slack:
            # keep Vg placeholders at 1.0 pu
            gen3p_pf.append([gen_id, bid, 1, 1.0, 1.0, 1.0, PgA, PgB, PgC, QgA, QgB, QgC])
            gen_id += 1

    mpc_pf["load3p"] = load3p_pf
    mpc_pf["gen3p"]  = gen3p_pf
    return mpc_pf


def remap_by_busphase_map(vec_src, map_src, map_dst):
    """
    Remap a per-node vector from source map to destination map using (bus,phase) keys.
    """
    out = np.zeros(len(map_dst), dtype=float)
    missing = []
    for key, j_dst in map_dst.items():
        if key in map_src:
            out[j_dst] = vec_src[map_src[key]]
        else:
            missing.append(key)
    if missing:
        print(f"[remap_by_busphase_map] WARNING: {len(missing)} keys missing in src map. Examples: {missing[:5]}{'...' if len(missing)>5 else ''}")
    return out

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
    slack_bus = "s148"
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
        std_P, std_Q, std_V = 1e-4, 1e-4, 1e-4
        diag_R = ([std_P ** 2] * num_bus_phases + [std_Q ** 2] * num_bus_phases + [std_V ** 2] * num_bus_phases)
        R = np.diag(diag_R)
        z_noisy_true = z + np.random.normal(0.0, np.sqrt(diag_R))
        # --- 3a) Reorder measurements to the assumed ordering (critical!) ---  # NEW
        z_noisy = reorder_measurements_to_map(
            z_noisy_true,
            src_map=busphase_map_true,
            dst_map=busphase_map_assumed,
            fill_v=1.0
        )

        # Rebuild R for the assumed ordering (diagonal with same stds)          # NEW
        N_assumed = len(busphase_map_assumed)
        diag_R_assumed = (
                [std_P ** 2] * N_assumed +
                [std_Q ** 2] * N_assumed +
                [std_V ** 2] * N_assumed
        )
        R = np.diag(diag_R_assumed)

        # --- 3b) Build PF case on assumed network using measured P/Q injections ---  # NEW
        # Split the (reordered) measurement vector
        P_meas = z_noisy[:N_assumed]
        Q_meas = z_noisy[N_assumed:2 * N_assumed]
        # Map to dict keyed by (bus, phase) for the assumed network
        P_meas_by_key = {key: P_meas[idx] for key, idx in busphase_map_assumed.items()}
        Q_meas_by_key = {key: Q_meas[idx] for key, idx in busphase_map_assumed.items()}

        mpc_pf_from_meas = build_mpc_with_measured_injections(
            mpc_assumed, busphase_map_assumed, P_meas_by_key, Q_meas_by_key
        )
        Vr_pf_meas, Vi_pf_meas, busphase_map_pf = run_newton_powerflow_3p(mpc_pf_from_meas, tol=1e-6, max_iter=20)

        # If the PF routine returned a different internal ordering, remap to the estimator's order
        if busphase_map_pf != busphase_map_assumed:
            print("...Re-mapping PF solution from measured P/Q to estimator ordering.")
            Vr_pf_meas = remap_by_busphase_map(Vr_pf_meas, busphase_map_pf, busphase_map_assumed)
            Vi_pf_meas = remap_by_busphase_map(Vi_pf_meas, busphase_map_pf, busphase_map_assumed)

        # Initial guess for estimator (polar)
        x0_est = np.concatenate([np.abs(Vr_pf_meas + 1j * Vi_pf_meas),
                                 np.angle(Vr_pf_meas + 1j * Vi_pf_meas)])
        # --- 3) RUN ESTIMATOR ---
        x_est, success, _, line_distances = run_lagrangian_polar(
            z_noisy, x0_est, busphase_map_assumed, Y_pu_assumed, R, mpc_assumed
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
    np.random.seed(42)
    dss_filename = "Secondary_Equivalent.dss"
    baseMVA = 1.0
    slack_bus = "s148"

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

    failed_line_ids = []   # <-- collect misdetections
    skipped_line_ids = []  # <-- collect PF-not-converged / skipped tests

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
        elif result is False:
            # Detection ran but was wrong -> record the actual burnt Line ID
            failed_line_ids.append((base_name, cable_to_burn))
        elif result is None:
            # Power flow for the "reality" didn't converge -> record as skipped
            skipped_line_ids.append((base_name, cable_to_burn))
            total_tests -= 1  # keep your success-rate denominator consistent

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

    if failed_line_ids:
        print("\nLine IDs with FAILED detection:")
        for base_name, line_id in failed_line_ids:
            print(f"  - Group '{base_name}': Line '{line_id}'")

    if skipped_line_ids:
        print("\nLine IDs SKIPPED (PF did not converge):")
        for base_name, line_id in skipped_line_ids:
            print(f"  - Group '{base_name}': Line '{line_id}'")

if __name__ == "__main__":
    main()
