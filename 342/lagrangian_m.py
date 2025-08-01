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


def compute_terminal_shunt_mahalanobis_distances(lambda_vec, EA, mpc, param_map, epsilon_reg=1e-8):
    """
    Computes group Mahalanobis distance for shunts at the terminals of each line.
    A group consists of all shunt admittance parameters (3 phases * 2 params/phase)
    for the two buses connected by a line.

    Args:
        lambda_vec (np.ndarray): Un-normalized Lagrange multipliers for shunts.
        EA (np.ndarray): Covariance matrix corresponding to lambda_vec.
        mpc (dict): The MATPOWER case file.
        param_map (dict): Maps shunt parameters, with (bus_id, phase_idx) as keys.
        epsilon_reg (float): Regularization value for Cholesky decomposition.

    Returns:
        list: A list of tuples (line_index, distance_value).
    """
    shunt_param_info = param_map.get('shunt')
    if not shunt_param_info:
        print("Warning: Shunt parameter map is missing.")
        return []

    distances = []
    for i_line, line_data in enumerate(mpc["line3p"]):
        from_bus = int(line_data[1])
        to_bus = int(line_data[2])

        group_indices = []
        # Gather indices for all 3 phases of the 'from' bus and 'to' bus shunts
        for bus_id in [from_bus, to_bus]:
            for phase_idx in range(3):
                key = (bus_id, phase_idx)
                if key in shunt_param_info:
                    start_idx, num_params = shunt_param_info[key]
                    group_indices.extend(range(start_idx, start_idx + num_params))

        group_indices = sorted(list(set(group_indices)))

        # A full group should have 6 (from_bus) + 6 (to_bus) = 12 parameters
        if not group_indices:
            continue

        # Extract the subset of multipliers and the covariance sub-matrix
        lam_sub = lambda_vec[group_indices]
        EA_sub = EA[np.ix_(group_indices, group_indices)]

        dist_line = np.nan
        try:
            # Use Cholesky decomposition for numerical stability
            try:
                L = np.linalg.cholesky(EA_sub)
            except np.linalg.LinAlgError:
                # Apply regularization if the matrix is not positive definite
                EA_sub_reg = EA_sub + np.eye(EA_sub.shape[0]) * epsilon_reg
                L = np.linalg.cholesky(EA_sub_reg)

            # The index is lambda^T * Sigma^-1 * lambda - k
            # This is computed as ||L^-1 * lambda||_2^2 - k
            y = np.linalg.solve(L, lam_sub)
            quadratic_form_value = np.dot(y, y)
            k = len(lam_sub)
            dist_line = quadratic_form_value - k

        except np.linalg.LinAlgError:
            print(f"ERROR: Cholesky decomposition failed for terminal shunts of line {i_line}. Dist set to inf.")
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

    # --- 1. Form the Parameter Jacobian (Hp) for SHUNTS ONLY ---
    print("\nForming Jacobian for Shunt Admittances ONLY...")
    Hp = jacobian_shunt_params(x_current, mpc, busphase_map)
    print(f"Shunt Jacobian Hp formed – shape {Hp.shape}")

    # --- 2. Create a Parameter Map for SHUNTS ONLY ---
    param_map = {'shunt': {}}
    shunt_bus_phase_ordered = []
    for bus_row in mpc['bus3p']:
        bus_id = int(bus_row[0])
        for phase_idx in range(3):
            shunt_bus_phase_ordered.append((bus_id, phase_idx))

    # The map key is (bus_id, phase_idx), value is (start_index, num_params)
    for i, (bus_id, phase_idx) in enumerate(shunt_bus_phase_ordered):
        # Indexing starts at 0 since there are no preceding line parameters
        start_idx = i * 2
        param_map['shunt'][(bus_id, phase_idx)] = (start_idx, 2)

    # --- 3. Proceed with GPU-accelerated calculation for lambda and EA ---
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

    # --- 4. Assess the Multipliers and Group Distances for SHUNTS ---
    print("\n--- Shunt Parameter Assessment Results ---")

    # a) Individual Shunt Parameter Test
    max_lambdaN_idx = np.argmax(np.abs(lambdaN))
    max_lambdaN_val = np.abs(lambdaN[max_lambdaN_idx])
    print(f"Normalized Multiplier Test (Individual Shunt Parameters):")
    print(f"  - Max absolute normalized multiplier (λ_N): {max_lambdaN_val:.4f}")

    bus_phase_idx = max_lambdaN_idx // 2
    if bus_phase_idx < len(shunt_bus_phase_ordered):
        bus_id, phase_idx = shunt_bus_phase_ordered[bus_phase_idx]
        phase_str = {0: 'A', 1: 'B', 2: 'C'}.get(phase_idx, '?')
        param_type = "Conductance (G)" if (max_lambdaN_idx % 2) == 0 else "Susceptance (B)"
        print(f"  - Corresponds to {param_type} at Bus ID {bus_id}, Phase {phase_str}.")
    else:
        print(f"  - Error: Corresponds to an invalid shunt parameter index ({max_lambdaN_idx}).")

    # b) Group Test: Mahalanobis Distance for Terminal Shunts of each Line
    print("\nGroup Mahalanobis Distance Test (Shunts at Line Terminals):")
    group_distances = compute_terminal_shunt_mahalanobis_distances(lambda_vec, ea, mpc, param_map)

    if group_distances:
        sorted_distances = sorted(group_distances, key=lambda item: item[1], reverse=True)
        max_line_idx, max_dist = sorted_distances[0]
        fbus, tbus = mpc['line3p'][max_line_idx][1], mpc['line3p'][max_line_idx][2]

        print(f"Top 5 most likely erroneous line-terminal groups:")
        for i_line, d_val in sorted_distances[:5]:
            f, t = mpc['line3p'][i_line][1], mpc['line3p'][i_line][2]
            print(f"  - Shunts at terminals of Line {i_line+1} ({f}-{t}): Mahalanobis distance = {d_val:.4f}")

        print(
            f"\nConclusion: The shunt admittances at the terminals of Line {max_line_idx+1} ({fbus}-{tbus}) have the largest group error score ({max_dist:.4f}).")
    else:
        print("Could not compute group Mahalanobis distances.")

    return x_current, success, lambdaN