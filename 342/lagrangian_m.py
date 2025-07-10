import numpy as np
from utilities.mea_fun import measurement_function
from utilities.make_jacobian import jacobian
from utilities.make_jacobian_shunt import jacobian_shunt_params
from utilities.make_jacobian_p import jacobian_line_params
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve, splu
import cupy as cp

import cupyx.scipy.sparse as cpsparse
import cupyx.scipy.sparse.linalg as cpsparse_linalg
import scipy.sparse as sp # Assuming Gain originates as a SciPy sparse matrix


def find_zero_injection_nodes(mpc, busphase_map, tol=1.0):
    """
    Identify phase-level nodes (bus-phase) that have effectively zero net injection,
    i.e. Generation - Load is within +/- tol (kW or kVar).

    Args:
        mpc : dict containing "bus3p", "load3p", "gen3p" data, etc.
        busphase_map : dict mapping (bus_id, phase_id) -> global node index
        tol : float, absolute threshold for net P,Q (kW,kVar) to consider zero injection

    Returns:
        zero_inj_nodes : list of node indices (0-based) in the global 3-phase node ordering
                         whose net injection magnitude is < tol.
    """

    bus3p = mpc["bus3p"]  # columns: [bus_id, type, base_kV_LL, VmA, VmB, VmC, VaA, VaB, VaC]
    load3p = mpc["load3p"]  # columns: [ldid, ldbus, status, PdA, PdB, PdC, QdA, QdB, QdC]
    gen3p = mpc["gen3p"]  # columns: [genid, gbus, status, VgA, VgB, VgC, PgA, PgB, PgC, QgA, QgB, QgC]

    # 1) Gather bus IDs and identify the slack bus
    bus_ids = [int(row[0]) for row in bus3p]
    slack_buses = [int(row[0]) for row in bus3p if row[1] == 3]
    # If there's always exactly one slack bus in your system:
    # slack_id = slack_buses[0]

    # 2) Initialize netPQ_phase = Generation - Load
    #    We'll store netPQ_phase[(b, ph)] = [P_net(kW), Q_net(kVar)]
    netPQ_phase = {}
    for b in bus_ids:
        for ph in [0, 1, 2]:
            netPQ_phase[(b, ph)] = [0.0, 0.0]  # [P_net, Q_net]

    # 3) Process loads -> subtract from net injection
    for row in load3p:
        ldid, ldbus, status, PdA, PdB, PdC, QdA, QdB, QdC = row
        if status == 0:
            continue

        bus_id = int(ldbus)
        # Phase A load
        if abs(PdA) > 1e-9 or abs(QdA) > 1e-9:
            netPQ_phase[(bus_id, 0)][0] -= PdA
            netPQ_phase[(bus_id, 0)][1] -= QdA
        # Phase B load
        if abs(PdB) > 1e-9 or abs(QdB) > 1e-9:
            netPQ_phase[(bus_id, 1)][0] -= PdB
            netPQ_phase[(bus_id, 1)][1] -= QdB
        # Phase C load
        if abs(PdC) > 1e-9 or abs(QdC) > 1e-9:
            netPQ_phase[(bus_id, 2)][0] -= PdC
            netPQ_phase[(bus_id, 2)][1] -= QdC

    # 4) Process generations -> add to net injection
    for row in gen3p:
        genid, gbus, status, VgA, VgB, VgC, PgA, PgB, PgC, QgA, QgB, QgC = row
        if status == 0:
            continue

        bus_id = int(gbus)
        # Phase A
        if abs(PgA) > 1e-9 or abs(QgA) > 1e-9:
            netPQ_phase[(bus_id, 0)][0] += PgA
            netPQ_phase[(bus_id, 0)][1] += QgA
        # Phase B
        if abs(PgB) > 1e-9 or abs(QgB) > 1e-9:
            netPQ_phase[(bus_id, 1)][0] += PgB
            netPQ_phase[(bus_id, 1)][1] += QgB
        # Phase C
        if abs(PgC) > 1e-9 or abs(QgC) > 1e-9:
            netPQ_phase[(bus_id, 2)][0] += PgC
            netPQ_phase[(bus_id, 2)][1] += QgC

    # 5) Determine which nodes have near-zero injection
    zero_inj_nodes = []
    for (b, ph), (p_net, q_net) in netPQ_phase.items():
        if b in slack_buses:
            continue
        if (abs(p_net) < tol) and (abs(q_net) < tol):
            node_idx = busphase_map[(b, ph)]
            zero_inj_nodes.append(node_idx)

    return zero_inj_nodes


def compute_group_mahalanobis_distances_per_phase(
        lambda_vec,  # Un-normalized Lagrange multipliers
        EA,  # Full covariance matrix of lambda_vec
        mpc,  # MPC case file
        param_map,  # The NEW per-phase parameter map
        epsilon_reg=1e-8):
    """
    Computes group Mahalanobis distance where a group is a line's series impedance
    PLUS the PER-PHASE shunt admittances of its two terminal buses.
    """
    if 'line' not in param_map or 'shunt' not in param_map:
        return []

    line_param_info = param_map['line']
    shunt_param_info = param_map['shunt']  # keys are (bus_id, phase_idx)

    distances = []
    for i_line, line_data in enumerate(mpc["line3p"]):
        from_bus = int(line_data[1])
        to_bus = int(line_data[2])

        group_indices = []

        # a) Add indices for the line's series impedance (12 params)
        line_start_idx, line_num_params = line_param_info[i_line]
        group_indices.extend(range(line_start_idx, line_start_idx + line_num_params))

        # b) Add indices for all 3 phases of the 'from' bus shunt (3 * 2 = 6 params)
        for phase_idx in range(3):
            key = (from_bus, phase_idx)
            if key in shunt_param_info:
                shunt_start_idx, shunt_num_params = shunt_param_info[key]
                group_indices.extend(range(shunt_start_idx, shunt_start_idx + shunt_num_params))

        # c) Add indices for all 3 phases of the 'to' bus shunt (3 * 2 = 6 params)
        for phase_idx in range(3):
            key = (to_bus, phase_idx)
            if key in shunt_param_info:
                shunt_start_idx, shunt_num_params = shunt_param_info[key]
                group_indices.extend(range(shunt_start_idx, shunt_start_idx + shunt_num_params))

        group_indices = sorted(list(set(group_indices)))

        # A full group should have 12 (line) + 6 (from_bus) + 6 (to_bus) = 24 parameters
        if len(group_indices) != 24:
            print(f"Warning: Group for line {i_line} has {len(group_indices)} params instead of the expected 24.")

        # ... The rest of the calculation is identical ...
        lam_sub = lambda_vec[group_indices]
        EA_sub = EA[np.ix_(group_indices, group_indices)]

        dist_line = np.nan
        try:
            try:
                L = np.linalg.cholesky(EA_sub)
            except np.linalg.LinAlgError:
                EA_sub_reg = EA_sub + np.eye(EA_sub.shape[0]) * epsilon_reg
                L = np.linalg.cholesky(EA_sub_reg)
            y = np.linalg.solve(L, lam_sub)
            dist_line = np.linalg.norm(y)
        except np.linalg.LinAlgError:
            print(f"ERROR: Cholesky decomposition failed for group of line {i_line}. Dist set to inf.")
            dist_line = np.inf

        distances.append((i_line, dist_line))

    return distances
# def compute_line_mahalanobis_distances(lambdaN, EA, mpc, epsilon_reg=1e-10):
#     """
#     Compute the group Mahalanobis distance for each line's parameters.
#
#     Args:
#         lambdaN (np.ndarray): Normalized Lagrange multipliers (CPU).
#         EA (np.ndarray): Full covariance matrix (CPU).
#         epsilon_reg (float): Small regularization value for EA_sub diagonal.
#
#     Returns:
#       distances : list of (line_index, distance_value)
#     """
#     if np.any(np.isnan(lambdaN)) or np.any(np.isinf(lambdaN)):
#         print("Warning: NaN or Inf detected in input lambdaN!")
#     if np.any(np.isnan(EA)) or np.any(np.isinf(EA)):
#         print("Warning: NaN or Inf detected in input EA!")
#
#     line_data = mpc["line3p"]
#     num_lines = len(line_data)
#     nparam_line = 12
#
#     distances = []
#     for i_line in range(num_lines):
#         start_idx = i_line * nparam_line
#         end_idx = start_idx + nparam_line
#
#         lam_vec = lambdaN[start_idx:end_idx]
#         EA_sub = EA[start_idx:end_idx, start_idx:end_idx]
#
#         dist_line = np.nan
#         try:
#             try:
#                 np.linalg.cholesky(EA_sub)
#                 EA_sub_to_invert = EA_sub
#                 is_regularized = False
#             except np.linalg.LinAlgError:
#                 print(f"Line {i_line}: EA_sub is not PSD. Applying regularization (epsilon={epsilon_reg}).")
#                 EA_sub_to_invert = EA_sub + np.eye(EA_sub.shape[0]) * epsilon_reg
#                 is_regularized = True
#
#             EA_sub_inv = np.linalg.inv(EA_sub_to_invert)
#             quadratic_form_value = lam_vec @ EA_sub_inv @ lam_vec
#             if quadratic_form_value < 0:
#                 print(f"Line {i_line}: Warning - Quadratic form value slightly negative ({quadratic_form_value:.2e}) after inversion {'(regularized)' if is_regularized else ''}. Clamping to 0.")
#                 quadratic_form_value = 0.0
#             elif np.isnan(quadratic_form_value):
#                  print(f"Line {i_line}: Error - Quadratic form value is NaN after inversion {'(regularized)' if is_regularized else ''}. Setting dist=NaN.")
#                  quadratic_form_value = np.nan
#             if not np.isnan(quadratic_form_value):
#                  dist_line = np.sqrt(quadratic_form_value)
#
#         except np.linalg.LinAlgError:
#             print(f"Line {i_line}: ERROR - Matrix inversion failed for EA_sub {'(regularized)' if is_regularized else ''}. Setting dist=inf.")
#             dist_line = np.inf
#
#         except Exception as e:
#              print(f"Line {i_line}: An unexpected error occurred: {e}. Setting dist=NaN.")
#              dist_line = np.nan
#
#         distances.append((i_line, dist_line))
#
#     return distances

def run_lagrangian_polar(z, x_init, busphase_map, Ybus, R, mpc, max_iter=20, tol=1e-4):
    """
    DSSE solver that also assesses both line impedance and shunt admittance parameters.
    """
    # ... (The first part of the function for state estimation remains exactly the same) ...
    # ... (Assume the DSSE converges and we have the final x_current and dxl) ...

    ########################################################
    # Data Preparation
    ########################################################
    nbus = len(mpc["bus3p"])
    nnodephase = len(busphase_map)
    nstate = 2 * nnodephase

    x_init = np.zeros(nstate)
    for i in range(nbus):
        x_init[3 * i] = 1.0;
        x_init[3 * i + nnodephase] = 0.0
        x_init[3 * i + 1] = 1.0;
        x_init[3 * i + nnodephase + 1] = np.deg2rad(-120)
        x_init[3 * i + 2] = 1.0;
        x_init[3 * i + nnodephase + 2] = np.deg2rad(120)
    x_est = x_init.copy()

    W = np.linalg.inv(R)
    zi_nodes = find_zero_injection_nodes(mpc, busphase_map, tol=1.0)
    indices_to_remove = []
    if zi_nodes:
        for i_node in zi_nodes:
            indices_to_remove.extend([i_node, i_node + nnodephase])

    W = np.delete(W, indices_to_remove, axis=0)
    W = np.delete(W, indices_to_remove, axis=1)
    W_sp = sps.csr_matrix(W)
    z = np.delete(z, indices_to_remove)

    ########################################################
    # State Estimation Iteration
    ########################################################
    success = True
    x_current = x_est.copy()
    for it in range(max_iter):
        hval = measurement_function(x_current, Ybus, mpc, busphase_map)
        Hmat = jacobian(x_current, Ybus, mpc, busphase_map)
        slack_cols = [nnodephase, nnodephase + 1, nnodephase + 2]
        keep_cols = np.delete(np.arange(Hmat.shape[1]), slack_cols)
        Hmat = Hmat[:, keep_cols]
        C = Hmat[indices_to_remove, :]
        keep_rows = np.delete(np.arange(Hmat.shape[0]), indices_to_remove)
        Hmat = Hmat[keep_rows, :]
        H_sp, C_sp = Hmat.tocsr(), C.tocsr()
        cval = hval[indices_to_remove]
        hval = np.delete(hval, indices_to_remove)
        r_meas = z - hval
        size_top = 2 * nnodephase - 3
        r = np.concatenate((np.zeros(size_top), r_meas, -cval))
        I_H_sp = sps.eye(H_sp.shape[0], format='csc')
        zero_block_sp = sps.csr_matrix((size_top, size_top))
        H_W_sp = H_sp.transpose().dot(W_sp)
        Gain = sps.bmat([[zero_block_sp, H_W_sp, C_sp.transpose()], [H_sp, I_H_sp, None], [C_sp, None, None]],
                        format='csc')

        try:
            dxl = spsolve(Gain, r)
        except sps.linalg.MatrixRankWarning:
            success = False
            print("Solver rank warning.")
            break

        dx = dxl[:size_top]
        dx0 = np.zeros(2 * nnodephase)
        dx0[:nnodephase] = dx[:nnodephase]
        dx0[nnodephase + 3:] = dx[nnodephase:]
        x_current += dx0
        print(f"DSSE polar iter {it + 1}: max dx={np.max(np.abs(dx)):.3e}")
        if np.max(np.abs(dx)) < tol:
            print(f"DSSE polar converged in {it + 1} iterations.")
            break
    else:
        success = False
        print("DSSE polar did not converge.")
    if not success:
        return x_current, False, None

    # --- 1. Form a COMBINED Parameter Jacobian (Hp) ---
    print("\nForming combined Jacobian for Line Impedances and Shunt Admittances...")
    Hp_line = jacobian_line_params(x_current, Ybus, mpc, busphase_map)
    Hp_shunt = jacobian_shunt_params(x_current, mpc, busphase_map)

    Hp = sps.hstack([Hp_line, Hp_shunt], format='csc')
    print(f"Combined Jacobian Hp formed – shape {Hp.shape}")
    num_line_params = Hp_line.shape[1]

    # --- 2. Create a Robust Parameter Map (NEW PER-PHASE SHUNT LOGIC) ---
    param_map = {'line': {}, 'shunt': {}}

    # a) Map for line parameters (12 per line)
    n_lines = len(mpc['line3p'])
    for i in range(n_lines):
        param_map['line'][i] = (i * 12, 12)

    # b) Map for shunt parameters, assuming a PER-PHASE model
    #    The jacobian creates columns for Bus1-PhA, Bus1-PhB, Bus1-PhC, Bus2-PhA, ...
    shunt_bus_phase_ordered = []
    for bus_row in mpc['bus3p']:
        bus_id = int(bus_row[0])
        for phase_idx in range(3):  # 0:A, 1:B, 2:C
            shunt_bus_phase_ordered.append((bus_id, phase_idx))

    # Check if the number of columns in Hp_shunt matches our new assumption
    expected_shunt_params = len(shunt_bus_phase_ordered) * 2  # 12 nodes * 2 params/node = 24
    if Hp_shunt.shape[1] != expected_shunt_params:
        print(
            f"CRITICAL WARNING: Shunt Jacobian size ({Hp_shunt.shape[1]}) does not match per-phase model assumption ({expected_shunt_params}). The following analysis may be incorrect.")

    # The map key is now a tuple (bus_id, phase_idx)
    for i, (bus_id, phase_idx) in enumerate(shunt_bus_phase_ordered):
        # The start index for shunts is after all line parameters
        start_idx = num_line_params + (i * 2)
        param_map['shunt'][(bus_id, phase_idx)] = (start_idx, 2)

    # --- 3. Proceed with GPU-accelerated calculation for lambda and EA ---
    # ... (This entire section is unchanged as it's general) ...
    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
    dtype = cp.float64

    Hp_dense = Hp.toarray()
    all_rows_hp = np.arange(Hp.shape[0])
    keep_rows_hp = np.delete(all_rows_hp, indices_to_remove) if len(indices_to_remove) > 0 else all_rows_hp
    Cp_dense = Hp[indices_to_remove, :].toarray() if len(indices_to_remove) > 0 else np.zeros((0, Hp.shape[1]))
    Hp_dense = Hp_dense[keep_rows_hp, :]
    W_dense = W_sp.toarray()
    W_gpu, Hp_gpu, Cp_gpu = cp.asarray(W_dense, dtype), cp.asarray(Hp_dense, dtype), cp.asarray(Cp_dense, dtype)
    WHp_gpu = W_gpu @ Hp_gpu
    S_gpu = -cp.concatenate([WHp_gpu, Cp_gpu], axis=0).T
    D_inv = 1e-6  # Or another suitable small number
    S_gpu *= D_inv
    Gain_gpu = cpsparse.csc_matrix(Gain)
    Gain_factor_gpu = cpsparse_linalg.splu(Gain_gpu)
    a, b, c = zero_block_sp.shape[0], I_H_sp.shape[0], C_sp.shape[0]
    cols_to_solve_np = np.arange(a, a + b)
    n_total = Gain_gpu.shape[0]
    rhs_gpu = cp.zeros((n_total, len(cols_to_solve_np)), dtype);
    rhs_gpu[cp.asarray(cols_to_solve_np), cp.arange(len(cols_to_solve_np))] = 1.0
    solutions_gpu = Gain_factor_gpu.solve(rhs_gpu)
    phi_gpu = cp.concatenate(
        [solutions_gpu[cp.asarray(np.arange(a, a + b)), :], solutions_gpu[cp.asarray(np.arange(a + b, a + b + c)), :]],
        axis=0)
    W_inv_sub_gpu = cp.diag(1.0 / cp.asarray(W_sp.diagonal()))
    step1_gpu = phi_gpu @ W_inv_sub_gpu @ phi_gpu.T
    ea_gpu_full = S_gpu @ step1_gpu @ S_gpu.T
    try:
        cp.linalg.cholesky(ea_gpu_full)
    except cp.linalg.LinAlgError:
        ea_gpu_full += cp.eye(ea_gpu_full.shape[0], dtype=dtype) * 1e-10

    lam_part_gpu = cp.asarray(dxl[a:], dtype)
    lambda_vec_gpu = S_gpu @ lam_part_gpu
    lambdaN_gpu = lambda_vec_gpu / cp.sqrt(cp.diag(ea_gpu_full) + 1e-12)

    lambda_vec, lambdaN, ea = cp.asnumpy(lambda_vec_gpu), cp.asnumpy(lambdaN_gpu), cp.asnumpy(ea_gpu_full)

    # --- 4. Assess the Multipliers and Group Distances (NEW PER-PHASE LOGIC) ---
    print("\n--- Combined Parameter Assessment Results ---")

    # a) Individual Parameter Test
    max_lambdaN_val = np.max(np.abs(lambdaN))
    max_lambdaN_idx = np.argmax(np.abs(lambdaN))
    print(f"Normalized Multiplier Test (All Individual Parameters):")
    print(f"  - Max absolute normalized multiplier (lambda_N): {max_lambdaN_val:.4f}")

    # Interpret the index using the per-phase mapping
    if max_lambdaN_idx < num_line_params:
        line_idx = max_lambdaN_idx // 12
        param_type = "Line Impedance"
        print(f"  - This corresponds to a {param_type} parameter in Line Index {line_idx}.")
    else:
        # CORRECTED LOGIC FOR PER-PHASE SHUNTS
        shunt_idx_local = max_lambdaN_idx - num_line_params
        # bus_phase_idx is the index into our ordered list of (bus, phase) tuples
        bus_phase_idx = shunt_idx_local // 2
        param_type = "Shunt Admittance"

        if bus_phase_idx < len(shunt_bus_phase_ordered):
            bus_id, phase_idx = shunt_bus_phase_ordered[bus_phase_idx]
            phase_str = {0: 'A', 1: 'B', 2: 'C'}.get(phase_idx, '?')
            print(f"  - This corresponds to a {param_type} parameter at Bus ID {bus_id}, Phase {phase_str}.")
        else:
            print(f"  - Error: Corresponds to an invalid shunt parameter index ({max_lambdaN_idx}).")

    # b) Group Parameter Test (Mahalanobis Distance for Line + Terminal Shunts)
    print("\nGroup Mahalanobis Distance Test (Line Impedance + Terminal Shunts):")

    # We must also update the M-distance function to handle this new mapping.
    # Pass the per-phase map to a revised function.
    group_distances = compute_group_mahalanobis_distances_per_phase(lambda_vec, ea, mpc, param_map)

    if group_distances:
        for i_line, d_val in group_distances:
            fbus, tbus = mpc['line3p'][i_line][1], mpc['line3p'][i_line][2]
            print(f"  - Group for Line {i_line} ({fbus}-{tbus}): Mahalanobis distance = {d_val:.4f}")

        if any(not np.isnan(d[1]) for d in group_distances):
            max_item = max(group_distances, key=lambda item: item[1] if not np.isnan(item[1]) else -np.inf)
            max_line_idx, max_dist = max_item
            print(
                f"\nConclusion: The component group for Line Index {max_line_idx} has the largest error score ({max_dist:.4f}) and is the most likely source of parameter error.")
    else:
        print("Could not compute group Mahalanobis distances.")

    return x_current, success, lambdaN