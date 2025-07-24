import numpy as np
import os
from opendssdirect import dss
from utilities.mea_fun import measurement_function
from cal_pf import run_newton_powerflow_3p
from parse_opendss_file import parse_opendss_to_mpc, build_global_y_per_unit, merge_closed_switches_in_mpc_and_dss
from utilities.correct_parameter_group_multi_scan import correct_parameter_group_multi_scan

# --- Sub-network Definition ---
sub_network_bus_names = {
    'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p11', 'p12', 'p13', 'p14', 'p15'
}
sub_network_line_names = {
    '1', '2', '4', '14', '15', '16'
}


def generate_and_save_measurement_data(dss_filename, lc_filename, baseMVA, slack_bus, num_scans, output_filename):
    """
    Generates multi-scan measurement data for the FULL network.
    """
    print("--- Starting: Generate and Save Full Measurement Data ---")
    if os.path.exists(output_filename):
        print(f"Full data file '{output_filename}' already exists. Skipping generation.")
        return

    mpc_correct = parse_opendss_to_mpc(dss_filename, baseMVA=baseMVA, slack_bus=slack_bus, lc_filename=lc_filename)
    merge_closed_switches_in_mpc_and_dss(mpc_correct, switch_threshold=2)
    Y_pu_correct, _ = build_global_y_per_unit(mpc_correct)

    multi_scan_measurements_z = []
    initial_states_multi_scan = []
    load_variation_factor = 0.95
    busphase_map = None

    print(f"--- Generating {num_scans} Measurement Scans for Full Network ---")
    for i in range(num_scans):
        print(f"  Generating scan {i + 1}/{num_scans}...")
        mpc_scan = mpc_correct.copy()
        mpc_scan["load3p"] = mpc_correct["load3p"].copy()
        current_load_factor = load_variation_factor + (i * 0.02)
        mpc_scan["load3p"][:, 3:6] *= current_load_factor

        Vr_true, Vi_true, current_busphase_map = run_newton_powerflow_3p(mpc_scan, tol=1e-6, max_iter=20)
        if busphase_map is None:
            busphase_map = current_busphase_map

        x_f_true = Vr_true + 1j * Vi_true
        x_true_polar = np.concatenate([np.abs(x_f_true), np.angle(x_f_true)])
        z_true = measurement_function(x_true_polar, Y_pu_correct, mpc_scan, busphase_map)

        std_P, std_Q, std_V = 0.00001, 0.00001, 0.00001
        num_busphases = len(busphase_map)
        num_lines = mpc_scan['line3p'].shape[0]

        num_P_inj = num_Q_inj = num_Vmag = num_busphases
        num_Pf = num_Pt = num_Qf = num_Qt = num_lines * 3

        R_variances_vec = np.concatenate([
            [std_P ** 2] * num_P_inj, [std_Q ** 2] * num_Q_inj,
            [std_P ** 2] * num_Pf, [std_P ** 2] * num_Pt,
            [std_Q ** 2] * num_Qf, [std_Q ** 2] * num_Qt,
            [std_V ** 2] * num_Vmag
        ])

        z_noisy = z_true + np.random.normal(0, np.sqrt(R_variances_vec))
        multi_scan_measurements_z.append(z_noisy)
        initial_states_multi_scan.append(x_true_polar)

    multi_scan_measurements_z_arr = np.array(multi_scan_measurements_z).T
    initial_states_multi_scan_arr = np.array(initial_states_multi_scan).T

    np.savez_compressed(
        output_filename,
        multi_scan_measurements_z=multi_scan_measurements_z_arr,
        initial_states_multi_scan=initial_states_multi_scan_arr,
        R_variances_vec=R_variances_vec,
        busphase_map=busphase_map,
        mpc_correct=mpc_correct
    )
    print(f"--- Full measurement data successfully saved to {output_filename} ---")


def _calculate_true_boundary_injections(mpc, busphase_map, x_full, boundary_bus_ids, sub_network_bus_ids):
    """
    Calculates the true, noise-free EQUIVALENT injection for each boundary bus.
    """
    num_busphases = len(busphase_map)
    num_lines = mpc['line3p'].shape[0]
    num_scans = x_full.shape[1]

    Y_pu, _ = build_global_y_per_unit(mpc)
    z_true_all_scans = np.array(
        [measurement_function(x_full[:, i], Y_pu, mpc, busphase_map) for i in range(num_scans)]).T

    tie_lines = [{'line_idx': i, 'fbus': int(line[1]), 'tbus': int(line[2])}
                 for i, line in enumerate(mpc['line3p'])
                 if (int(line[1]) in sub_network_bus_ids) != (int(line[2]) in sub_network_bus_ids)]

    boundary_injections = {}

    for b_id in boundary_bus_ids:
        for phase_idx in [0, 1, 2]:
            bp_tuple = (b_id, phase_idx)
            bp_idx = busphase_map.get(bp_tuple)
            if bp_idx is None: continue

            true_net_Pinj = z_true_all_scans[bp_idx, :]
            true_net_Qinj = z_true_all_scans[bp_idx + num_busphases, :]
            S_net_original = true_net_Pinj + 1j * true_net_Qinj

            S_from_tie_lines = np.zeros(num_scans, dtype=complex)
            for tie in tie_lines:
                if tie['fbus'] == b_id or tie['tbus'] == b_id:
                    line_idx, fbus = tie['line_idx'], tie['fbus']

                    pf_idx = 2 * num_busphases + (line_idx * 3) + phase_idx
                    pt_idx = pf_idx + (3 * num_lines)
                    qf_idx = pt_idx + (3 * num_lines)
                    qt_idx = qf_idx + (3 * num_lines)

                    S_from = z_true_all_scans[pf_idx, :] + 1j * z_true_all_scans[qf_idx, :]
                    S_to = z_true_all_scans[pt_idx, :] + 1j * z_true_all_scans[qt_idx, :]

                    if fbus == b_id:
                        S_from_tie_lines -= S_from
                    else:
                        S_from_tie_lines += S_to

            boundary_injections[bp_tuple] = S_net_original + S_from_tie_lines

    return boundary_injections


def extract_subnetwork_measurements(full_data_filename, sub_network_output_filename):
    """
    Extracts and saves the measurement data for the sub-network.
    """
    print("\n--- Starting: Sub-network Measurement Extraction ---")
    data = np.load(full_data_filename, allow_pickle=True)
    z_full, x_full, R_full, busphase_map_full, mpc_full = \
        data['multi_scan_measurements_z'], data['initial_states_multi_scan'], \
            data['R_variances_vec'], data['busphase_map'].item(), data['mpc_correct'].item()

    num_busphases_full = len(busphase_map_full)
    num_lines_full = mpc_full['line3p'].shape[0]
    busname_to_id_full = mpc_full['busname_to_id']
    id_to_busname_full = {v: k for k, v in busname_to_id_full.items()}

    sub_network_bus_ids = {busname_to_id_full[name] for name in sub_network_bus_names}

    boundary_bus_ids = {int(line[1]) for line in mpc_full['line3p'] if
                        (int(line[1]) in sub_network_bus_ids) != (int(line[2]) in sub_network_bus_ids)}
    boundary_bus_ids.update({int(line[2]) for line in mpc_full['line3p'] if
                             (int(line[1]) in sub_network_bus_ids) != (int(line[2]) in sub_network_bus_ids) and int(
                                 line[2]) in sub_network_bus_ids})

    print(f"Identified Boundary Buses (by ID): {boundary_bus_ids}")

    true_boundary_injections = _calculate_true_boundary_injections(mpc_full, busphase_map_full, x_full,
                                                                   boundary_bus_ids, sub_network_bus_ids)

    sub_busphase_indices_mask = sorted(
        [idx for bp_tuple, idx in busphase_map_full.items() if bp_tuple[0] in sub_network_bus_ids])
    sub_line_indices_mask = sorted(
        [i for i, line in enumerate(mpc_full['line3p']) if str(line[6]).lower() in sub_network_line_names])

    p_inj_indices = [idx for idx in sub_busphase_indices_mask]
    q_inj_indices = [idx + num_busphases_full for idx in sub_busphase_indices_mask]
    pf_indices = [2 * num_busphases_full + (line_idx * 3) + phase for line_idx in sub_line_indices_mask for phase in
                  range(3)]
    pt_indices = [pf_indices[i] + (3 * num_lines_full) for i in range(len(pf_indices))]
    qf_indices = [pt_indices[i] + (3 * num_lines_full) for i in range(len(pf_indices))]
    qt_indices = [qf_indices[i] + (3 * num_lines_full) for i in range(len(pf_indices))]
    v_mag_indices = [2 * num_busphases_full + 12 * num_lines_full + idx for idx in sub_busphase_indices_mask]

    final_meas_mask = sorted(
        p_inj_indices + q_inj_indices + pf_indices + pt_indices + qf_indices + qt_indices + v_mag_indices)

    z_sub = z_full[final_meas_mask, :]
    R_sub = R_full[final_meas_mask]

    final_state_mask = sorted(sub_busphase_indices_mask + [i + num_busphases_full for i in sub_busphase_indices_mask])
    x_sub = x_full[final_state_mask, :]

    sub_busphase_map_old_numbering = {bp_tuple: new_idx for new_idx, (bp_tuple, old_idx) in enumerate(
        sorted([(k, v) for k, v in busphase_map_full.items() if v in sub_busphase_indices_mask],
               key=lambda item: item[1]))}

    pseudo_meas_variance = 1e-12
    num_sub_busphases = len(sub_busphase_map_old_numbering)
    num_sub_flows_per_type = len(pf_indices)

    for bp_tuple_target, true_equivalent_injection in true_boundary_injections.items():
        sub_bp_idx = sub_busphase_map_old_numbering.get(bp_tuple_target)
        if sub_bp_idx is None: continue

        full_bp_idx = busphase_map_full.get(bp_tuple_target)
        true_vmag = x_full[full_bp_idx, :]

        p_inj_sub_offset = sub_bp_idx
        q_inj_sub_offset = num_sub_busphases + sub_bp_idx
        v_mag_sub_offset = 2 * num_sub_busphases + 4 * num_sub_flows_per_type + sub_bp_idx

        z_sub[p_inj_sub_offset, :] = true_equivalent_injection.real
        R_sub[p_inj_sub_offset] = pseudo_meas_variance

        z_sub[q_inj_sub_offset, :] = true_equivalent_injection.imag
        R_sub[q_inj_sub_offset] = pseudo_meas_variance

        z_sub[v_mag_sub_offset, :] = true_vmag
        R_sub[v_mag_sub_offset] = pseudo_meas_variance

    np.savez_compressed(
        sub_network_output_filename,
        multi_scan_measurements_z=z_sub,
        initial_states_multi_scan=x_sub,
        R_variances_vec=R_sub,
        busphase_map_old_numbering=sub_busphase_map_old_numbering,
        id_to_busname_full=id_to_busname_full,
        sub_network_line_names=np.array(list(sub_network_line_names))
    )
    print(f"--- Sub-network measurement data successfully saved to {sub_network_output_filename} ---")


def remap_subnetwork_data(loaded_data, mpc_sub):
    """
    Remaps the loaded sub-network data (z, x, map) to match the internal
    numbering of the newly parsed mpc_sub.
    """
    print("--- Remapping sub-network data to match new MPC numbering ---")
    z_old = loaded_data['multi_scan_measurements_z']
    x_old = loaded_data['initial_states_multi_scan']
    R_old = loaded_data['R_variances_vec']
    map_old = loaded_data['busphase_map_old_numbering'].item()
    id_to_busname_full = loaded_data['id_to_busname_full'].item()
    sub_network_line_names_loaded = loaded_data['sub_network_line_names']

    # Create maps for the NEW sub-network MPC
    busname_to_id_sub = mpc_sub['busname_to_id']
    id_to_busname_sub = {v: k for k, v in busname_to_id_sub.items()}
    busname_to_old_id = {v: k for k, v in id_to_busname_full.items()}

    _, _, map_new = run_newton_powerflow_3p(mpc_sub.copy(), tol=1e-8)

    num_scans = x_old.shape[1]
    num_sub_busphases_new = len(map_new)
    num_sub_lines_new = mpc_sub['line3p'].shape[0]

    # Initialize new, correctly ordered arrays
    x_new = np.zeros((2 * num_sub_busphases_new, num_scans))
    z_new = np.zeros((3 * num_sub_busphases_new + 12 * num_sub_lines_new, num_scans))
    R_new = np.zeros(z_new.shape[0])

    # Remap States (x) and Bus-related measurements (z, R)
    for bp_tuple_new, idx_new in map_new.items():
        new_bus_id, phase_idx = bp_tuple_new
        bus_name = id_to_busname_sub.get(new_bus_id)
        if bus_name is None: continue

        old_bus_id = busname_to_old_id.get(bus_name)
        if old_bus_id is None: continue

        bp_tuple_old = (old_bus_id, phase_idx)
        idx_old = map_old.get(bp_tuple_old)
        if idx_old is None: continue

        x_new[idx_new, :] = x_old[idx_old, :]
        x_new[idx_new + num_sub_busphases_new, :] = x_old[idx_old + len(map_old), :]

        p_inj_sub_offset = idx_old
        q_inj_sub_offset = len(map_old) + idx_old
        v_mag_sub_offset = 2 * len(map_old) + 4 * (len(sub_network_line_names_loaded) * 3) + idx_old

        z_new[idx_new, :] = z_old[p_inj_sub_offset, :]
        R_new[idx_new] = R_old[p_inj_sub_offset]

        z_new[idx_new + num_sub_busphases_new, :] = z_old[q_inj_sub_offset, :]
        R_new[idx_new + num_sub_busphases_new] = R_old[q_inj_sub_offset]

        vmag_new_offset = 2 * num_sub_busphases_new + 4 * (num_sub_lines_new * 3)
        z_new[vmag_new_offset + idx_new, :] = z_old[v_mag_sub_offset, :]
        R_new[vmag_new_offset + idx_new] = R_old[v_mag_sub_offset]

    # Remap Flow Measurements
    # old_line_name_to_idx = {name: i for i, name in enumerate(sub_network_line_names_loaded)}
    old_line_name_to_idx = {str(line[6]).lower(): i for i, line in enumerate(mpc_sub['line3p'])}
    new_line_name_to_idx = {str(line[6]).lower(): i for i, line in enumerate(mpc_sub['line3p'])}

    for line_name, idx_old in old_line_name_to_idx.items():
        idx_new = new_line_name_to_idx.get(line_name)
        if idx_new is None: continue

        for phase in range(3):
            pf_old_offset = 2 * len(map_old) + (idx_old * 3) + phase
            pt_old_offset = pf_old_offset + (3 * len(sub_network_line_names_loaded))
            qf_old_offset = pt_old_offset + (3 * len(sub_network_line_names_loaded))
            qt_old_offset = qf_old_offset + (3 * len(sub_network_line_names_loaded))

            pf_new_offset = 2 * num_sub_busphases_new + (idx_new * 3) + phase
            pt_new_offset = pf_new_offset + (3 * num_sub_lines_new)
            qf_new_offset = pt_new_offset + (3 * num_sub_lines_new)
            qt_new_offset = qf_new_offset + (3 * num_sub_lines_new)

            z_new[pf_new_offset, :] = z_old[pf_old_offset, :]
            R_new[pf_new_offset] = R_old[pf_old_offset]
            z_new[pt_new_offset, :] = z_old[pt_old_offset, :]
            R_new[pt_new_offset] = R_old[pt_old_offset]
            z_new[qf_new_offset, :] = z_old[qf_old_offset, :]
            R_new[qf_new_offset] = R_old[qf_old_offset]
            z_new[qt_new_offset, :] = z_old[qt_old_offset, :]
            R_new[qt_new_offset] = R_old[qt_old_offset]

    print("--- Remapping complete ---")
    return z_new, x_new, R_new, map_new


def main():
    full_dss_filename = "Master.dss"
    sub_dss_filename = "Master_sub.dss"
    sub_dss_filename1 = "Master_sub1.dss"
    lc_filename = "LineConstantsCode.txt"
    baseMVA = 1.0
    slack_bus = "p1"
    num_scans = 10

    full_data_filename = "full_system_multi_scan_data.npz"
    sub_network_data_filename = "sub_network_measurements.npz"

    generate_and_save_measurement_data(
        full_dss_filename, lc_filename, baseMVA, slack_bus, num_scans, full_data_filename
    )
    extract_subnetwork_measurements(
        full_data_filename, sub_network_data_filename
    )
    print("\nExtraction complete.")

    print("\n--- Preparing Data for Parameter Correction ---")

    dss.Command("Clear")
    mpc_sub = parse_opendss_to_mpc(sub_dss_filename, baseMVA=baseMVA, slack_bus=slack_bus, lc_filename=lc_filename)
    merge_closed_switches_in_mpc_and_dss(mpc_sub, switch_threshold=2)

    dss.Command("Clear")
    mpc_sub1 = parse_opendss_to_mpc(sub_dss_filename1, baseMVA=baseMVA, slack_bus=slack_bus, lc_filename=lc_filename)
    merge_closed_switches_in_mpc_and_dss(mpc_sub1, switch_threshold=2)

    print(f"Loading data from {sub_network_data_filename}...")
    try:
        loaded_data = np.load(sub_network_data_filename, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {sub_network_data_filename}.")
        return

    # ADDED remapping step
    z_remapped, x_remapped, R_remapped, map_remapped = remap_subnetwork_data(loaded_data, mpc_sub)

    # --- Run Parameter Correction ---
    line_name_to_correct = '14'
    line_to_correct_idx = np.where(mpc_sub["line3p"][:, 6] == line_name_to_correct.lower())[0][0]
    print(f"\n--- Starting Group Parameter Correction for line index {line_to_correct_idx} ---")

    corrected_params, success = correct_parameter_group_multi_scan(
        sub_dss_filename1, mpc_sub1, line_to_correct_idx,
        z_remapped, x_remapped,
        R_remapped, map_remapped)

    if success:
        print("\n--- Parameter Correctio n Successful! ---")
        print("Corrected R/mi (p.u.):", corrected_params[:6])
        print("Corrected X/mi (p.u.):", corrected_params[6:])
    else:
        print("\n--- Parameter Correction Failed ---")


if __name__ == "__main__":
    main()