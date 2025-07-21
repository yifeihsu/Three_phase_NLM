import numpy as np
from utilities.mea_fun import measurement_function
from utilities.make_jacobian import jacobian
from utilities.make_jacobian_p import jacobian_line_params
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve, splu
import cupy as cp

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
    load3p = mpc["load3p"]  # columns: [ldid, ldbus, status, PdA, PdB, PdC, QdA, QdB, QdC] (if truly P,Q)
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
    #    Confirm that the last 3 columns are indeed QdA, QdB, QdC (not pfA,B,C).
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
        # Optionally skip the slack bus if you never want to classify it as zero injection
        if b in slack_buses:
            continue
        if (abs(p_net) < tol) and (abs(q_net) < tol):
            node_idx = busphase_map[(b, ph)]
            zero_inj_nodes.append(node_idx)

    return zero_inj_nodes

def compute_group_rms_dist(lambdaN, nparam_line=12):
    """
    If lambdaN is already individually normalized,
    then we can define a 'group distance' as RMS of each line's 12 parameters.
    """
    num_lines = len(lambdaN) // nparam_line
    distances = []
    for i_line in range(num_lines):
        start_idx = i_line * nparam_line
        end_idx   = start_idx + nparam_line
        lam_block = lambdaN[start_idx:end_idx]
        dist_line = np.sqrt(np.mean(lam_block**2)) * nparam_line
        distances.append((i_line, dist_line))
    return distances

import cupy as cp
import cupyx.scipy.sparse as cpx_sp
import cupyx.scipy.sparse.linalg as cpx_linalg

# The rest of your imports remain the same, except we won't rely on CPU-based scipy.sparse:
# import scipy.sparse as sps  # We'll move to cpx_sp
# from scipy.sparse.linalg import spsolve, splu  # We'll replace these with gmres

from utilities.mea_fun import measurement_function
from utilities.make_jacobian import jacobian
from utilities.make_jacobian_p import jacobian_line_params


def run_lagrangian_polar(z, x_init, busphase_map, Ybus, R, mpc, max_iter=20, tol=1e-6):
    """
    DSSE solver using polar coordinates, with full GPU-based large scale calculations:
      - R inversion, Weighted Jacobian construction,
      - Gains & constraints in cupy sparse,
      - Iterative solves with cupyx.scipy.sparse.linalg.gmres
    """

    #################################################################
    # 0) Preliminary: Convert input data to GPU arrays as needed
    #################################################################
    # z could be large, so let's move it to GPU
    z_gpu = cp.asarray(z)
    # R => invert on GPU
    R_gpu = cp.asarray(R, dtype=cp.float64)
    W_gpu = cp.linalg.inv(R_gpu)  # GPU-based dense inverse of R

    # We'll keep "Ybus" as is.
    # If Ybus is large and you want GPU acceleration inside `measurement_function`,
    # you'd also make Ybus a GPU array. That depends on your code details.

    #################################################################
    # 1) Basic Setup
    #################################################################
    nbus = len(mpc["bus3p"])
    nnodephase = len(busphase_map)
    nstate = 2 * nnodephase

    # Example param indexing (unchanged logic)
    fbus_self = []
    tbus_self = []
    fbus_mut  = []
    tbus_mut  = []
    for line in mpc["line3p"]:
        from_bus = int(line[1]) - 1
        to_bus   = int(line[2]) - 1
        for phase in range(3):
            fbus_self.append(from_bus*3 + phase)
            tbus_self.append(to_bus  *3 + phase)
        fbus_mut.append(from_bus*3 + 0)
        tbus_mut.append(from_bus*3 + 1)
        fbus_mut.append(from_bus*3 + 1)
        tbus_mut.append(from_bus*3 + 2)
        fbus_mut.append(from_bus*3 + 0)
        tbus_mut.append(from_bus*3 + 2)
    fbus_self = cp.asarray(fbus_self, dtype=cp.int32)
    tbus_self = cp.asarray(tbus_self, dtype=cp.int32)
    fbus_mut  = cp.asarray(fbus_mut,  dtype=cp.int32)
    tbus_mut  = cp.asarray(tbus_mut,  dtype=cp.int32)

    fbus = cp.concatenate([fbus_self, fbus_mut])
    tbus = cp.concatenate([tbus_self, tbus_mut])
    npara_x = fbus.shape[0]
    npara   = 2 * npara_x

    # Initialize x on CPU or GPU. We'll do CPU for the example, then keep it synced
    x_current = cp.asarray(x_init, dtype=cp.float64)

    # We'll define a small helper to move from GPU to CPU if needed
    def x_current_cpu():
        return cp.asnumpy(x_current)

    # For demonstration, let's do zero injection logic on CPU
    # or skip it if you want. We'll skip it for simplicity:
    indices_to_remove = []

    #################################################################
    # 2) DSSE Iteration
    #################################################################
    success = True
    for it in range(max_iter):
        #
        # (A) Evaluate measurement function & Jacobian
        #
        # If measurement_function & jacobian are CPU-based, you might do:
        x_cpu_tmp = x_current_cpu()
        hval_cpu = measurement_function(x_cpu_tmp, Ybus, mpc, busphase_map)
        Hmat_cpu = jacobian(x_cpu_tmp, Ybus, mpc, busphase_map)

        # Then move results to GPU
        hval_gpu = cp.asarray(hval_cpu, dtype=cp.float64)
        # Convert Hmat_cpu (scipy.sparse) to cupyx sparse:
        if not hasattr(Hmat_cpu, "tocoo"):
            raise ValueError("Jacobian must return a sparse SciPy matrix.")
        Hmat_coo = Hmat_cpu.tocoo()
        # Build a cupyx.scipy.sparse COO, then convert to CSR:
        H_gpu = cpx_sp.coo_matrix((cp.asarray(Hmat_coo.data),
                                   (cp.asarray(Hmat_coo.row),
                                    cp.asarray(Hmat_coo.col))),
                                  shape=Hmat_coo.shape).tocsr()

        #
        # (B) Remove slack bus angle columns, zero-injection rows, etc. on GPU
        #
        slack_cols = cp.array([nnodephase, nnodephase+1, nnodephase+2], dtype=cp.int32)
        all_cols = cp.arange(H_gpu.shape[1], dtype=cp.int32)
        # keep_cols = np.delete(all_cols, slack_cols) => do on GPU:
        keep_mask = cp.ones_like(all_cols, dtype=cp.bool_)
        keep_mask[slack_cols] = False
        keep_cols = all_cols[keep_mask]

        # We can row/column-slice in cupyx similarly to SciPy, but it's a bit more limited:
        # For performance, you might reconstruct a new matrix with a gather approach.
        # As a simple approach, let's do a naive CPU fallback for slicing:
        # (If you need full GPU slicing of columns, see cupyx.scipy.sparse advanced usage.)
        H_gpu_dense = H_gpu.toarray()       # become dense on GPU
        H_gpu_dense = H_gpu_dense[:, keep_cols]  # slice in GPU
        # Convert back to a GPU-sparse if desired:
        H_gpu = cpx_sp.csr_matrix(H_gpu_dense)

        # indices_to_remove (zero-injections) => skip for brevity, or do similarly:
        # For example, if you have rows to remove, you'd do row slicing the same way.

        #
        # (C) Form Weighted H => H' * W on GPU
        #
        # Convert W_gpu to a sparse matrix as well if we want to do sparse-sparse multiplies:
        # But W is dense. We can keep it dense. We'll do H'*(W H) or something.
        # Let's do "H_W_sp = H_sp.transpose().dot(W_sp)" but on GPU.
        # Easiest might be to do everything in dense for demonstration:
        H_dense = H_gpu.toarray()           # (still on GPU)
        W_dense = W_gpu                     # (already on GPU, shape m x m)
        Ht_dense = H_dense.T               # transpose on GPU
        # Weighted multiplication: (H^T) * W => shape (nstate, m)
        # But watch out for dimension mismatch if you removed rows.
        # We'll assume your dimension matches: (H is m x (nstate-3)) etc.
        HtW_dense = Ht_dense @ W_dense      # GPU-based dense

        #
        # (D) Build the block matrix "Gain" in GPU memory
        #     Because we can't do a "bmat" with Cupy the same way as SciPy easily,
        #     let's do the 3-block approach in dense for demonstration:
        #
        mH, nH = H_dense.shape  # rows, cols
        size_top = 2*nnodephase - 3
        # zero_block => size_top x size_top
        zero_block = cp.zeros((size_top, size_top), dtype=cp.float64)
        # We need an Identity block => mH x mH
        I_H = cp.eye(mH, dtype=cp.float64)
        # We also have "C" for the constraints => skip zero-injection for brevity
        #
        # We want:
        #  Gain = [[0,          HtW_dense,   Ct^T],
        #          [H_dense,    I_H,         0   ],
        #          [C,          0,           0   ]]
        #
        # For demonstration, let's skip C to keep it simpler.
        # So the block matrix is effectively:
        #    [[0,           HtW_dense],
        #     [H_dense,     I_H     ]]
        # with dimension (size_top + mH) x (size_top + mH).
        # This is just a demonstration:
        block_top  = cp.concatenate([zero_block,       HtW_dense], axis=1)
        block_bot  = cp.concatenate([H_dense,          I_H],       axis=1)
        Gain_dense = cp.concatenate([block_top, block_bot], axis=0)
        # Convert to Cupy sparse:
        Gain_gpu = cpx_sp.csr_matrix(Gain_dense)

        #
        # (E) Build the "r" vector on GPU
        #
        # Suppose r = [0; z_gpu - hval_gpu] ignoring constraints for brevity:
        r_meas_gpu = z_gpu - hval_gpu
        r_gpu = cp.concatenate([cp.zeros(size_top, dtype=cp.float64), r_meas_gpu])

        #
        # (F) Solve the linear system "Gain * dxl = r" using GMRES on GPU
        #
        dxl_gpu, info = cpx_linalg.gmres(Gain_gpu, r_gpu, tol=1e-9, maxiter=500)
        if info != 0:
            print(f"GMRES did not converge (info={info}) in iteration {it+1}.")
            success = False
            break

        #
        # (G) Extract dx from dxl
        #
        # dxl has dimension (size_top + nH).
        # The first size_top entries => "dx", then the rest belongs to the Lagrange multipliers, etc.
        dx_gpu = dxl_gpu[:size_top]
        # We build dx0 with zeros for slack angles.
        # You would replicate your original logic for mapping dx into x_current
        dx0_gpu = cp.zeros(2*nnodephase, dtype=cp.float64)
        # Insert the first nnodephase portion into magnitude, skip angles # nnodephase..nnodephase+3, etc.
        dx0_gpu[:nnodephase] = dx_gpu[:nnodephase]
        dx0_gpu[nnodephase+3:] = dx_gpu[nnodephase:]

        #
        # (H) Update x_current, check convergence
        #
        x_current += dx0_gpu
        max_dx_gpu = cp.max(cp.abs(dx_gpu))
        max_dx = float(max_dx_gpu.get())  # bring to CPU as float
        print(f"DSSE polar iter {it+1}: max dx={max_dx:.3e}")
        if max_dx < tol:
            print(f"DSSE polar converged in {it+1} iterations.")
            break
    else:
        print(f"DSSE polar did not converge in {max_iter} iterations.")
        success = False

    # Lagrangian multiplier calculation
    # V, theta: the mag and angle for all nodes
    print("Forming Parameters Jacobian")
    # --- Form Hp as before ---
    Hp = jacobian_line_params(x_current, Ybus, mpc, busphase_map)
    print("Jacobian Formed")
    # print the shape of Hp
    print("Shape of Hp:", Hp.shape)

    if not sps.isspmatrix_csc(Hp):
        Hp = Hp.tocsc()
    if not sps.isspmatrix_csc(W_sp):
        W_sp = W_sp.tocsc()
    if not sps.isspmatrix_csc(Gain):
        Gain = Gain.tocsc()

    # Extract Cp and (reduced) Hp efficiently
    all_rows_hp = np.arange(Hp.shape[0])
    keep_rows_hp = np.delete(all_rows_hp, indices_to_remove)
    Cp = Hp[indices_to_remove, :]
    Hp = Hp[keep_rows_hp, :]

    # Compute WHp = W_sp * Hp
    WHp = W_sp.dot(Hp)

    # Stack and transpose in CSC format to form S_sp
    S_sp = sps.vstack([WHp, Cp], format='csc')
    S_sp = -S_sp.transpose(copy=True)

    # ----------------------------------------------------------------
    #       REPLACE FULL INVERSE WITH SPARSE FACTOR + PARTIAL SOLVES
    # ----------------------------------------------------------------

    print("Factorizing Gain for partial solves...")
    Gain_factor = splu(Gain)  # Instead of sps.linalg.inv(Gain)

    # Suppose you need to form E5, E8 = selected rows/cols of Gain_inv
    # Example: "a, b, c" are block sizes. Adjust as appropriate.
    a, b, c = zero_block_sp.shape[0], I_H_sp.shape[0], C_sp.shape[0]

    row_E5 = np.arange(a, a + b)
    col_E5 = row_E5

    row_E8 = np.arange(a + b, a + b + c)
    col_E8 = col_E5  # same columns as E5 in this scenario

    # Build E5 columns by solving Gain * x = e_j for each column j in col_E5
    E5_cols = []
    for j in col_E5:
        e_j = np.zeros(Gain.shape[0])
        e_j[j] = 1.0
        x = Gain_factor.solve(e_j)
        E5_cols.append(x[row_E5])

    # Stack columns into a 2D array, then convert to sparse if desired
    E5 = np.column_stack(E5_cols)
    E5_sp = sps.csc_matrix(E5)

    # Same procedure for E8
    E8_cols = []
    for j in col_E8:
        e_j = np.zeros(Gain.shape[0])
        e_j[j] = 1.0
        x = Gain_factor.solve(e_j)
        E8_cols.append(x[row_E8])

    E8 = np.column_stack(E8_cols)
    E8_sp = sps.csc_matrix(E8)

    # Stack to form phi_sp
    phi_sp = sps.vstack([E5_sp, E8_sp], format='csc')

    # ----------------------------------------------------------------
    #               REST OF THE COMPUTATION
    # ----------------------------------------------------------------

    # Invert diagonal of W_sp
    W_inv_sp = sps.diags(1.0 / W_sp.diagonal(), format='csc')

    # step1_sp = phi_sp * W_inv_sp * phi_sp^T
    print("Computing step1_sp")
    w_diag = W_inv_sp.diagonal()
    phi_scaled = phi_sp.dot(sps.diags(w_diag))
    step1_sp = phi_scaled.dot(phi_sp.transpose())
    covu_sp = step1_sp  # Keep it sparse as long as possible

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

    def regularization(covL):
        """
        Regularize the covariance matrix covL by adding a small value to its diagonal.
        """
        epsilon = 1e-12  # Small regularization parameter, adjust if needed

        # Ensure covL is available (Calculate S, R first if needed)
        # R_mat = np.linalg.inv(W) # Assuming W is dense here
        # S_mat = ... calculate S ... (May need G_inv = np.linalg.inv(H_sp.T @ W_sp @ H_sp) etc.)
        # covL = S_mat @ R_mat @ S_mat.T # This calculation might be needed if not already done
        if sps.issparse(covL):
            covL_reg = covL + epsilon * sps.eye(covL.shape[0], format=covL.format)
        else:  # Assuming covL is a NumPy array
            covL_reg = covL + epsilon * np.identity(covL.shape[0])
        return covL_reg
    print("Computing diagonal of EA = diag(S_sp * covu_sp * S_sp^T)")
    X_sp = S_sp.dot(covu_sp)  # both are sparse => X_sp also sparse (NxN)

    # 2) Elementwise multiply X_sp and S_sp, then sum each row.
    #    In SciPy, 'multiply()' is an elementwise product on sparse matrices;
    #    'sum(axis=1)' sums each row into a 1x1 result; we then flatten to 1D.
    ea_diag = X_sp.multiply(S_sp).sum(axis=1).A1

    # If you truly need the entire matrix EA in dense form, you can do:
    ea_sp = S_sp.dot(covu_sp).dot(S_sp.transpose())  # keep as sparse
    ea = ea_sp.toarray()                             # convert to dense if absolutely necessary


    print("EA diagonal computed.")

    # ----------------------------------------------------------------
    #               LAMBDA COMPUTATIONS
    # ----------------------------------------------------------------

    # Extract lam_part from your dxl array
    lam_part = dxl[2 * nnodephase - 3:]

    # Compute lambda_vec = S_sp @ lam_part in sparse form
    lambda_vec = S_sp @ lam_part

    # If using just the diagonal of EA:
    tt = np.sqrt(ea_diag)  # shape is (len(ea_diag), )
    lambdaN = lambda_vec / tt
    # print("Forming Parameters Jacobian (Hp)")
    # Hp = jacobian_line_params(x_current, Ybus, mpc, busphase_map)
    # print("Jacobian Formed")
    #
    # # 1) Convert Hp to dense NumPy, extract needed rows
    # Hp_dense = Hp.toarray()  # from sparse to dense on CPU
    # all_rows_hp = np.arange(Hp.shape[0])
    # keep_rows_hp = (np.delete(all_rows_hp, indices_to_remove)
    #                 if len(indices_to_remove) > 0 else all_rows_hp)
    #
    # # Sub-block for zero-injection rows => Cp
    # if len(indices_to_remove) > 0:
    #     Cp_dense = Hp[indices_to_remove, :].toarray()
    # else:
    #     Cp_dense = np.zeros((0, Hp.shape[1]))  # empty sub-block
    #
    # Hp_dense = Hp_dense[keep_rows_hp, :]
    #
    # # 2) Convert W_sp (which is your W = R^{-1}) to dense
    # W_dense = W_sp.toarray()  # CPU dense
    # # We'll do the multiplication W_sp * Hp in dense form on GPU
    #
    # # 3) Transfer dense arrays to GPU as CuPy arrays
    # W_gpu = cp.asarray(W_dense)  # shape (n_meas, n_meas)
    # Hp_gpu = cp.asarray(Hp_dense)  # shape (n_meas, n_params) or similar
    # Cp_gpu = cp.asarray(Cp_dense)  # shape (n_zi, n_params), possibly empty
    #
    # # 4) Compute WHp = W_gpu @ Hp_gpu on GPU (dense)
    # WHp_gpu = W_gpu @ Hp_gpu  # shape: (n_meas, n_params)
    #
    # # 5) Build S = [WHp; Cp], then transpose => S = -(S^T)
    # #    Because you don't need the sparse structure, do it all in dense form:
    # S_gpu = cp.concatenate([WHp_gpu, Cp_gpu], axis=0)  # vertical stack
    # S_gpu = -S_gpu.T  # now shape is (n_params, n_meas + n_zi), negated
    #
    # print("Factorizing Gain for partial solves (CPU-based) ...")
    # Gain_factor = splu(Gain)  # This remains on CPU
    #
    # # Suppose we want partial blocks E5, E8 from Gain^{-1} on CPU:
    # a, b, c = zero_block_sp.shape[0], I_H_sp.shape[0], C_sp.shape[0]
    # row_E5 = np.arange(a, a + b)
    # col_E5 = row_E5
    # row_E8 = np.arange(a + b, a + b + c)
    # col_E8 = col_E5
    #
    # E5_cols = []
    # for j in col_E5:
    #     e_j = np.zeros(Gain.shape[0])
    #     e_j[j] = 1.0
    #     x_sol = Gain_factor.solve(e_j)  # CPU solve
    #     E5_cols.append(x_sol[row_E5])
    # E5 = np.column_stack(E5_cols)
    #
    # E8_cols = []
    # for j in col_E8:
    #     e_j = np.zeros(Gain.shape[0])
    #     e_j[j] = 1.0
    #     x_sol = Gain_factor.solve(e_j)
    #     E8_cols.append(x_sol[row_E8])
    # E8 = np.column_stack(E8_cols)
    #
    # # If you still need them on GPU for further dense operations:
    # E5_gpu = cp.asarray(E5)
    # E8_gpu = cp.asarray(E8)
    #
    # # phi = [E5; E8], in dense GPU form:
    # phi_gpu = cp.concatenate([E5_gpu, E8_gpu], axis=0)
    #
    # # 6) Invert diagonal of W_sp in a dense manner:
    # W_diag = W_sp.diagonal()  # CPU vector of diag elements
    # W_inv_diag = 1.0 / W_diag
    # # Build a dense diagonal on GPU
    # W_inv_gpu = cp.diag(cp.asarray(W_inv_diag))
    #
    # # 7) step1_sp = phi_sp * W_inv_sp * phi_sp^T  (dense GPU version)
    # print("Computing step1_sp on GPU (dense) ...")
    # step1_gpu = phi_gpu @ W_inv_gpu @ phi_gpu.T  # covu in dense GPU
    #
    # # 8) Now compute EA = S * covu * S^T on GPU
    # #    If you only want the diagonal, we can do a diagonal approach in dense.
    # #    But let's do the full multiplication for demonstration.
    # S_cov_gpu = S_gpu @ step1_gpu  # shape: (n_params, n_params)
    # ea_gpu_full = S_cov_gpu @ S_gpu.T  # shape: (n_params, n_params)
    #
    # # For the diagonal:
    # ea_diag_gpu = cp.diag(ea_gpu_full)
    # ea_diag = cp.asnumpy(ea_diag_gpu)  # back to CPU if needed
    #
    # print("Converting EA to dense on GPU ... (already dense in 'ea_gpu_full')")
    # ea = cp.asnumpy(ea_gpu_full)  # Full EA on CPU if needed
    # print("EA diagonal computed.")
    #
    # # 9) LAMBDA computations
    # lam_part = dxl[2 * nnodephase - 3:]  # CPU vector
    # # If you need it on GPU for consistency:
    # lam_part_gpu = cp.asarray(lam_part)
    #
    # # lambda_vec = S_sp @ lam_part   => now in dense form on GPU:
    # lambda_vec_gpu = S_gpu @ lam_part_gpu
    #
    # # if we just do diagonal-based normalization:
    # tt_gpu = cp.sqrt(ea_diag_gpu)
    # lambdaN_gpu = lambda_vec_gpu / tt_gpu
    # # move back to CPU if needed
    # lambdaN = cp.asnumpy(lambdaN_gpu)

    line_distances = compute_group_rms_dist(lambdaN, nparam_line=12)
    for i_line, dval in line_distances:
        print(f"Line {i_line + 1} => Mahalanobis distance = {dval:.3f}")

    # Done.  Now lambdaN, ea_diag, etc. are ready for further use.
    print("All done.")

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

    return x_current, success, lambdaN
