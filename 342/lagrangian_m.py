import numpy as np
from utilities.mea_fun import measurement_function
from utilities.make_jacobian import jacobian
from utilities.make_jacobian_p import jacobian_line_params
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve, splu
import cupy as cp

import cupyx.scipy.sparse as cpsparse
import cupyx.scipy.sparse.linalg as cpsparse_linalg
import scipy.sparse as sp


def find_zero_injection_nodes(mpc, busphase_map, tol=1):
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


def compute_line_mahalanobis_distances(lambdaN, EA, mpc, epsilon_reg=1):
    """
    Compute the group Mahalanobis distance for each line's parameters.

    Args:
        lambdaN (np.ndarray): Normalized Lagrange multipliers (CPU).
        EA (np.ndarray): Full covariance matrix (CPU).
        epsilon_reg (float): Small regularization value for EA_sub diagonal.

    Returns:
      distances : list of (line_index, distance_value)
    """
    if np.any(np.isnan(lambdaN)) or np.any(np.isinf(lambdaN)):
        print("Warning: NaN or Inf detected in input lambdaN!")
    if np.any(np.isnan(EA)) or np.any(np.isinf(EA)):
        print("Warning: NaN or Inf detected in input EA!")

    line_data = mpc["line3p"]
    num_lines = len(line_data)
    nparam_line = 12

    distances = []
    for i_line in range(num_lines):
        start_idx = i_line * nparam_line
        end_idx = start_idx + nparam_line

        lam_vec = lambdaN[start_idx:end_idx]
        EA_sub = EA[start_idx:end_idx, start_idx:end_idx]

        dist_line = np.nan
        try:
            try:
                np.linalg.cholesky(EA_sub)
                EA_sub_to_invert = EA_sub
                is_regularized = False
            except np.linalg.LinAlgError:
                # print(f"Line {i_line}: EA_sub is not PSD. Applying regularization (epsilon={epsilon_reg}).")
                EA_sub_to_invert = EA_sub + np.eye(EA_sub.shape[0]) * epsilon_reg
                is_regularized = True

            EA_sub_inv = np.linalg.inv(EA_sub_to_invert)
            quadratic_form_value = lam_vec @ EA_sub_inv @ lam_vec
            if quadratic_form_value < 0:
                print(f"Line {i_line}: Warning - Quadratic form value slightly negative ({quadratic_form_value:.2e}) after inversion {'(regularized)' if is_regularized else ''}. Clamping to 0.")
                quadratic_form_value = 0.0
            elif np.isnan(quadratic_form_value):
                 print(f"Line {i_line}: Error - Quadratic form value is NaN after inversion {'(regularized)' if is_regularized else ''}. Setting dist=NaN.")
                 quadratic_form_value = np.nan
            if not np.isnan(quadratic_form_value):
                 dist_line = np.sqrt(quadratic_form_value)

        except np.linalg.LinAlgError:
            print(f"Line {i_line}: ERROR - Matrix inversion failed for EA_sub {'(regularized)' if is_regularized else ''}. Setting dist=inf.")
            dist_line = np.inf

        except Exception as e:
             print(f"Line {i_line}: An unexpected error occurred: {e}. Setting dist=NaN.")
             dist_line = np.nan

        distances.append((i_line, dist_line))

    return distances

def run_lagrangian_polar(z, x_init, busphase_map, Ybus, R, mpc, max_iter=20, tol=1e-4):
    """
    DSSE solver using polar coordinates.
    Args:
      z : 1D numpy array of measurements (size m)
      x_init : initial guess (2*nnodephase),
               x_init[i]   = V_m,i
               x_init[i+N] = angle_i
      busphase_map : dictionary mapping (bus_id, phase_id) -> node index
      Ybus : NxN complex bus admittance matrix
      R : measurement covariance or weighting matrix, shape (m,m)
    Returns:
      x_est : final estimated state in polar => [V_m(0..N-1), Va(0..N-1)]
      success : boolean
    """
    def check_PSD(covLa):
        """
        Check if the matrix is positive semi-definite (PSD).
        """
        try:
            # Compute eigenvalues for the symmetric matrix
            eigenvalues = np.linalg.eigvalsh(covLa)
            # Define a tolerance for floating-point comparisons
            tolerance = 1e-9  # Adjust tolerance if needed
            # Check if the minimum eigenvalue is non-negative within the tolerance
            is_psd = np.min(eigenvalues) >= -tolerance
            if is_psd:
                print(f"Matrix appears to be PSD (min eigenvalue: {np.min(eigenvalues):.2e})")
            else:
                print(f"Matrix is NOT PSD (min eigenvalue: {np.min(eigenvalues):.2e})")
        except np.linalg.LinAlgError:
            # Handle cases where eigenvalue computation fails
            print("Eigenvalue computation failed.")
            is_psd = False


    nbus = len(mpc["bus3p"])
    nnodephase = len(busphase_map)
    nstate = 2 * nnodephase
    # Number of parameters that needs to be monitored, assume only X and R for now
    fbus_self = []
    tbus_self = []
    fbus_mut  = []
    tbus_mut  = []
    for line in mpc["line3p"]:
        from_bus = int(line[1]) - 1  # convert to 0-based
        to_bus   = int(line[2]) - 1

        # self (phases 0..2 => A,B,C)
        for phase in range(3):
            fbus_self.append(from_bus*3 + phase)
            tbus_self.append(to_bus  *3 + phase)

        # mutual: ab=(0,1), bc=(1,2), ac=(0,2)
        fbus_mut.append(from_bus*3 + 0)
        tbus_mut.append(from_bus*3 + 1)

        fbus_mut.append(from_bus*3 + 1)
        tbus_mut.append(from_bus*3 + 2)

        fbus_mut.append(from_bus*3 + 0)
        tbus_mut.append(from_bus*3 + 2)

    fbus_self = np.array(fbus_self, dtype=int)
    tbus_self = np.array(tbus_self, dtype=int)
    fbus_mut  = np.array(fbus_mut,  dtype=int)
    tbus_mut  = np.array(tbus_mut,  dtype=int)

    # Combine into one index set
    fbus = np.concatenate([fbus_self, fbus_mut])
    tbus = np.concatenate([tbus_self, tbus_mut])
    npara_x = len(fbus)  # total X-parameters for all lines
    npara   = 2 * npara_x  # double to account for R-parameters
    x_init = np.zeros(nstate)
    # x_init[nnodephase:] angles in radians, three phases 0, -120, 120 --> 0, -2pi/3, 2pi/3
    for i in range(nbus):
        x_init[3*i] = 1.0
        x_init[3*i+nnodephase] = 0.0
        x_init[3*i+1] = 1.0
        x_init[3*i+nnodephase+1] = np.deg2rad(-120)
        x_init[3*i+2] = 1.0
        x_init[3*i+nnodephase+2] = np.deg2rad(120)
    x_est = x_init.copy()
    W = np.linalg.inv(R)
    # Find the zero injection buses
    zi_nodes = find_zero_injection_nodes(mpc, busphase_map, tol=1.0)
    indices_to_remove = []
    #### We assume the normal equations without constraints
    zi_nodes = []

    for i_node in zi_nodes:
        # i_node is the node index in [0..nnodephase-1] for that phase
        # P row => i_node
        # Q row => i_node + nnodephase
        indices_to_remove.append(i_node)
        indices_to_remove.append(i_node + nnodephase)
    W = np.delete(W, indices_to_remove, axis=0)
    W = np.delete(W, indices_to_remove, axis=1)
    W_sp = sps.csr_matrix(W)
    z = np.delete(z, indices_to_remove)
    # Assuming only one slack bus (3-phases setting)
    xl = np.zeros(2*nnodephase - 3 + 3 * nnodephase, dtype=float)
    for i in range(nnodephase):
        xl[3*i] = 1
        xl[3*i + nnodephase] = 0
        xl[3*i + 1] = 1
        xl[3*i + nnodephase + 1] = np.deg2rad(-120)
        xl[3*i + 2] = 1
        xl[3*i + nnodephase + 2] = np.deg2rad(120)

    ########################################################
    # Start iteration
    ########################################################
    success = True
    x_current = x_est.copy()
    for it in range(max_iter):
        # 1) Compute the measurement function and its Jacobian
        hval = measurement_function(x_current, Ybus, mpc, busphase_map)
        # print("Start forming Jacobian")
        Hmat = jacobian(x_current, Ybus, mpc, busphase_map)
        # print("Jacobian formed")

        # 2) Remove slack-bus angle columns (3 angles for 3-phase slack)
        slack_cols = [nnodephase, nnodephase + 1, nnodephase + 2]
        all_cols = np.arange(Hmat.shape[1])
        keep_cols = np.delete(all_cols, slack_cols)
        Hmat = Hmat[:, keep_cols]  # keep all rows, subset columns

        # 3) Remove zero-injection rows
        #    We'll first gather them in "C" for constraints:
        C = Hmat[indices_to_remove, :]

        #    Then keep the rest in Hmat
        all_rows = np.arange(Hmat.shape[0])
        keep_rows = np.delete(all_rows, indices_to_remove)
        Hmat = Hmat[keep_rows, :]

        # 4) Now Hmat and C are sparse. If you want them in CSR format:
        H_sp = Hmat.tocsr()  # ensure it's still CSR after slicing
        C_sp = C.tocsr()

        # 2) Remove columns for the slack bus angles (example: angles # nnodephase, nnodephase+1, nnodephase+2)
        cval = hval[indices_to_remove]
        hval = np.delete(hval, indices_to_remove)

        # 4) Form residual vector r = [zeros_for_lagrange; (z - h); -cval]
        r_meas = z - hval
        r = np.concatenate(( np.zeros(2*nnodephase - 3), r_meas, -cval ))
        # We'll need an identity matrix matching Hmat's row count:
        nH = H_sp.shape[0]
        I_H_sp = sps.eye(nH, format='csc')

        # Make a zero block for the top-left corner: shape (2*nnodephase-3, 2*nnodephase-3)
        size_top = 2*nnodephase - 3
        zero_block_sp = sps.csr_matrix((size_top, size_top))

        # "H_W" = H' * W, but we need W in sparse as well
        # If W_sp is your weighting matrix, do:
        H_W_sp = H_sp.transpose().dot(W_sp)

        # For the third block dimension:
        nC = C_sp.shape[0]

        # Next we form the big "Gain" matrix in block-sparse form:
        #      [  0          H'W         C'
        #        H           I          0
        #        C           0          0  ]
        Gain = sps.bmat([
            [ zero_block_sp, H_W_sp,     C_sp.transpose() ],
            [ H_sp,          I_H_sp,     None             ],
            [ C_sp,          None,       None             ]
        ], format='csc')

        # 5) Solve the linear system on GPU (CuPy) and reuse the factorization later
        Gain_csc = Gain.tocsc() if not isinstance(Gain, sps.csc_matrix) else Gain

        # Move to GPU
        Gain_gpu = cpsparse.csc_matrix(Gain_csc)  # GPU sparse CSC
        r_gpu = cp.asarray(r, dtype=cp.float64)  # GPU RHS

        # Factorize + solve on GPU
        try:
            Gain_factor_gpu = cpsparse_linalg.splu(Gain_gpu)  # LU on GPU
            dxl_gpu = Gain_factor_gpu.solve(r_gpu)
            dxl = cp.asnumpy(dxl_gpu)  # back to CPU for the state update

            # Save for reuse after convergence (EA/λ step)
            last_Gain_gpu = Gain_gpu
            last_factor_gpu = Gain_factor_gpu
            last_size_top = size_top
            last_nH = nH
            last_nC = C_sp.shape[0]
        except Exception as e:
            success = False
            print(f"GPU sparse solve failed: {e}")
            break

        # 6) Extract dx from dxl
        # The first part of dxl corresponds to the "2*nnodephase - 3" states,
        # which we then map back into dx0 with zeros for the slack bus angles
        dx = dxl[:size_top]
        dx0 = np.zeros(2*nnodephase)
        # put dx into the correct spots for magnitude and angle (excluding the 3 slack angles)
        dx0[:nnodephase] = dx[:nnodephase]
        dx0[nnodephase+3:] = dx[nnodephase:]

        # Update x_current
        x_current += dx0
        # print_estimation_results(x_current, mpc)
        # Print the max dx
        print(f"DSSE polar iter {it+1}: max dx={np.max(np.abs(dx)):.3e}")
        if np.max(np.abs(dx))<tol:
            print(f"DSSE polar converged in {it+1} iterations, max dx={np.max(np.abs(dx)):.3e}")
            break
    else:
        success = False
        print("DSSE polar did not converge in max_iter steps.")

    # cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
    dtype = cp.float64

    Hp = jacobian_line_params(x_current, Ybus, mpc, busphase_map)
    Hp_dense = Hp.toarray()  # from sparse to dense on CPU
    all_rows_hp = np.arange(Hp.shape[0])
    keep_rows_hp = (np.delete(all_rows_hp, indices_to_remove)
                    if len(indices_to_remove) > 0 else all_rows_hp)
    if len(indices_to_remove) > 0:
        Cp_dense = Hp[indices_to_remove, :].toarray()
    else:
        Cp_dense = np.zeros((0, Hp.shape[1]))
    Hp_dense = Hp_dense[keep_rows_hp, :]

    W_dense = W_sp.toarray()
    # 3) Transfer dense arrays to GPU as CuPy arrays
    W_gpu = cp.asarray(W_dense).astype(dtype, copy=False)  # shape (n_meas, n_meas)

    #### Remove the row/col if zero injection (C not empty)

    Hp_gpu = cp.asarray(Hp_dense).astype(dtype, copy=False)  # shape (n_meas-n_zi, n_params)
    Cp_gpu = cp.asarray(Cp_dense).astype(dtype, copy=False)  # shape (n_zi, n_params)

    # 4) Compute WHp = W_gpu @ Hp_gpu on GPU (dense)
    WHp_gpu = W_gpu @ Hp_gpu  # shape: (n_meas, n_params)

    # 5) Build S = [WHp; Cp], then transpose => S = -(S^T)
    S_gpu = cp.concatenate([WHp_gpu, Cp_gpu], axis=0)
    S_gpu = -S_gpu.T  # (n_params, n_meas + n_zi)
    # Scale for numerical stability
    # D_inv = 1e-6 # For impedance based multipliers
    # D_inv = 1e6
    D_inv = 1
    S_gpu *= D_inv

    # if not isinstance(Gain, sp.csc_matrix):
    #     print("Converting Gain to CSC format on CPU...")
    #     Gain_csc = Gain.tocsc()
    # else:
    #     Gain_csc = Gain
    #
    # Gain_gpu = cpsparse.csc_matrix(Gain_csc)
    #
    # # Perform the sparse LU factorization on the GPU
    # Gain_factor_gpu = cpsparse_linalg.splu(Gain_gpu)
    #
    # a, b, c = zero_block_sp.shape[0], I_H_sp.shape[0], C_sp.shape[0]
    # n_total = Gain_gpu.shape[0]  # Total size from the GPU matrix

    # Reuse the factorization & sizes from the last DSSE iteration
    Gain_gpu = last_Gain_gpu
    Gain_factor_gpu = last_factor_gpu
    a, b, c = last_size_top, last_nH, last_nC
    n_total = Gain_gpu.shape[0]

    row_E5_np = np.arange(a, a + b)
    col_E5_np = row_E5_np  # Columns to solve for to get E5
    row_E8_np = np.arange(a + b, a + b + c)
    cols_to_solve_np = col_E5_np  # Unique columns needed for solves
    # 2. Create the multi-RHS matrix on GPU (columns of Identity)
    # Columns we solved for were exactly the measurement block (size b)
    rhs_gpu = cp.zeros((n_total, b), dtype=cp.float64)
    col_idx = cp.arange(b)
    rhs_gpu[a + col_idx, col_idx] = 1.0  # Identity columns at measurement block

    solutions_gpu = Gain_factor_gpu.solve(rhs_gpu)  # (n_total, b)
    E5_gpu = solutions_gpu[a:a + b, :]  # (b, b)
    E8_gpu = solutions_gpu[a + b:a + b + c, :]  # (c, b)
    phi_gpu = cp.concatenate([E5_gpu, E8_gpu], axis=0)  # (b+c, b)

    # Vector (no dense matrix) for W^{-1}
    W_diag = W_sp.diagonal().astype(np.float64)  # CPU vector of length b
    W_inv_cols_gpu = cp.asarray(1.0 / W_diag)  # GPU vector, shape (b,)

    # step1 = phi * diag(W_inv) * phi^T
    step1_gpu = (phi_gpu * W_inv_cols_gpu) @ phi_gpu.T  # (b+c, b+c)

    # EA = S * step1 * S^T
    print("Computing EA = S * step1 * S.T on GPU...")
    S_cov_gpu = S_gpu @ step1_gpu
    ea_gpu_full = S_cov_gpu @ S_gpu.T
    ### Now ea is scaled by D_inv^2

    try:
        cp.linalg.cholesky(ea_gpu_full)
        print("ea_gpu_full is initially PSD.")
        ea_gpu_reg = ea_gpu_full  # No regularization needed initially
    except cp.linalg.LinAlgError:
        print("ea_gpu_full is NOT initially PSD. Applying regularization.")
        # Add small value to the diagonal (Diagonal Loading)
        epsilon = 1e-10  # Adjust this value as needed (start small)
        diag_indices = cp.arange(ea_gpu_full.shape[0])
        ea_gpu_reg = ea_gpu_full + cp.eye(ea_gpu_full.shape[0], dtype=ea_gpu_full.dtype) * epsilon
        print(f"Added epsilon={epsilon} * I to ea_gpu_full diagonal.")
        # Optional: Verify PSD after regularization
        try:
            cp.linalg.cholesky(ea_gpu_reg)
            print("ea_gpu_reg is now PSD after regularization.")
        except cp.linalg.LinAlgError:
            print(
                f"WARN: ea_gpu_reg is STILL NOT PSD even after adding epsilon={epsilon}. Consider increasing epsilon or using eigenvalue clipping.")

    ea_diag_gpu = cp.diag(ea_gpu_reg)
    ea = cp.asnumpy(ea_gpu_reg)  # Get the full regularized matrix if needed
    # 9) LAMBDA computations
    # Assuming lam_part is defined on CPU and corresponds to the rows of phi (b+c dimensions)
    # Or perhaps it corresponds to the full n_total dimension? The original slicing suggests otherwise.
    # Original: dxl[2 * nnodephase - 3:] - this implies it doesn't directly match a,b,c structure.
    # Let's assume lam_part corresponds to the rows aggregated in phi (dimension b+c) for S @ lam_part to work.
    # You might need to adjust how lam_part is selected based on your specific problem.
    # Placeholder: assume lam_part has size b+c
    # Adjust lam_part extraction based on actual structure if needed
    # This assumes lam_part should match the rows of S_gpu, which is b+c
    lam_part_start_idx = a  # Assuming lam_part corresponds to rows a -> a+b+c like phi
    lam_part_end_idx = a + b + c
    lam_part = dxl[lam_part_start_idx:lam_part_end_idx]
    # Ensure lam_part matches S_gpu's columns (b+c)
    if len(lam_part) != S_gpu.shape[1]:
        print(
            f"Warning: Dimension mismatch for S@lam_part. S_gpu columns={S_gpu.shape[1]}, len(lam_part)={len(lam_part)}. Adjusting lam_part extraction or S_gpu.")
        # Example: Pad or truncate lam_part if appropriate, or redefine S_gpu
        # This padding/truncation is problem-specific, using zeros as placeholder:
        correct_len = S_gpu.shape[1]
        if len(lam_part) < correct_len:
            lam_part = np.pad(lam_part, (0, correct_len - len(lam_part)))
        else:
            lam_part = lam_part[:correct_len]
    lam_part_gpu = cp.asarray(lam_part, dtype=cp.float64)
    # lam_part_gpu *= col_norms  # multiply, not divide!
    lambda_vec_gpu = S_gpu @ lam_part_gpu
    lambda_vec = cp.asnumpy(lambda_vec_gpu)

    # Add small epsilon for numerical stability before division
    tt_gpu = cp.sqrt(ea_diag_gpu + cp.finfo(ea_diag_gpu.dtype).eps)
    lambdaN_gpu = lambda_vec_gpu / tt_gpu # Note that ea_diag is scaled with d_inv^2, lambda_vec is scaled with d_inv so cancelled
    lambdaN = cp.asnumpy(lambdaN_gpu)  # Final result on CPU
    print("Lambda computations finished.")

    line_distances = compute_line_mahalanobis_distances(lambda_vec, ea, mpc)
    for i_line, dval in line_distances:
        print(f"Line {i_line + 1} => Mahalanobis distance = {dval:.3f}")
    max_line, max_dist = max(line_distances, key=lambda x: x[1])
    print(f"Line {max_line + 1} has the largest Mahalanobis distance: {max_dist:.6f}")
    # --- Bad Data Processing (Using Residual Covariance)
    # Compute final measurement residual vector
    # h_final = measurement_function(x_current)
    # r_final = z - h_final
    #
    # # Build the Gain matrix from the last iteration: Gain = H^T * W * H
    # Gain = Hmat.T @ W @ Hmat
    # # Measurement covariance is R = inv(W)
    # R_meas = np.linalg.inv(W)
    # # Residual covariance: omega = R_meas - H * inv(Gain) * H^T
    # omega = R_meas - Hmat @ np.linalg.inv(Gain) @ Hmat.T
    # # Normalized residuals
    # diag_omega = np.diag(omega)
    # norm_resid = np.abs(r_final) / np.sqrt(diag_omega)
    # max_norm = np.max(norm_resid)
    # idx_max = np.argmax(norm_resid)
    # bad_data_threshold = 3.0
    # if max_norm > bad_data_threshold:
    #     print(f"Bad data detected at measurement index {idx_max}: normalized residual = {max_norm:.2f} exceeds threshold {bad_data_threshold}")
    #     success = False
    # else:
    #     print(f"No bad data detected: max normalized residual = {max_norm:.2f} below threshold {bad_data_threshold}")

    return x_current, success, lambdaN, line_distances
