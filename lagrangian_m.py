import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve, splu
import cupy as cp
import cupyx.scipy.sparse as cpsparse
import cupyx.scipy.sparse.linalg as cpsparse_linalg
from utilities.mea_fun import measurement_function
from utilities.make_jacobian import jacobian
from utilities.make_jacobian_p import jacobian_line_params

def find_zero_injection_nodes(mpc, busphase_map, tol=1e6):
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

def compute_line_mahalanobis_distances(lambdaN, EA, mpc,
                                       epsilon_reg=1,   # relative jitter multiplier (see notes)
                                       rtol=1e-12,      # relative eig cutoff for pseudoinverse
                                       atol=0.0,        # absolute eig cutoff for pseudoinverse
                                       max_jitter_tries=8,
                                       debug=False):
    """
    Compute the (squared) Mahalanobis distance for each line block using
    numerically robust operations that avoid explicit inversion.

    Args:
        lambdaN (np.ndarray): Normalized Lagrange multipliers (CPU), shape (n_total,).
        EA (np.ndarray): Full covariance matrix (CPU), shape (n_total, n_total).
        mpc (dict-like): Must contain key "line3p" to determine number of lines.
        epsilon_reg (float): Relative jitter multiplier. The actual initial jitter is
            epsilon_reg * eps * scale, where eps is machine epsilon and
            scale ~ mean(diag(block)). Set to 0 to try without jitter first.
        rtol (float): Relative eigenvalue tolerance for PSD pseudoinverse fallback.
        atol (float): Absolute eigenvalue tolerance for PSD pseudoinverse fallback.
        max_jitter_tries (int): Max geometric escalations of jitter for Cholesky.
        debug (bool): If True, prints extra diagnostics.

    Returns:
        distances: list of (line_index, distance_value) with NaN/inf on hard failures.
    """

    # Basic input sanity
    if np.any(~np.isfinite(lambdaN)):
        print("Warning: NaN or Inf detected in input lambdaN!")
    if np.any(~np.isfinite(EA)):
        print("Warning: NaN or Inf detected in input EA!")

    line_data = mpc["line3p"]
    num_lines = len(line_data)
    nparam_line = 12

    distances = []

    # Precompute machine epsilon for dtype
    eps = np.finfo(float).eps  # assume float64; adjust if you use another dtype globally

    for i_line in range(num_lines):
        start_idx = i_line * nparam_line
        end_idx = start_idx + nparam_line

        lam_vec = np.asarray(lambdaN[start_idx:end_idx], dtype=float)
        A = np.asarray(EA[start_idx:end_idx, start_idx:end_idx], dtype=float)

        # Guard against bad inputs at block level
        if np.any(~np.isfinite(lam_vec)) or np.any(~np.isfinite(A)):
            if debug:
                print(f"Line {i_line}: Non-finite entries in lam_vec/EA_sub → distance = NaN.")
            distances.append((i_line, np.nan))
            continue

        # Enforce symmetry explicitly; quadratic form depends only on symmetric part
        A = 0.5 * (A + A.T)

        # Set a scale to size the jitter relative to the block
        diagA = np.diag(A)
        # Robust scale: mean of |diag|, fall back to 1.0 if not positive/finite
        scale = float(np.mean(np.abs(diagA))) if np.all(np.isfinite(diagA)) else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = max(1.0, float(np.linalg.norm(A, ord=np.inf)))

        # Initial jitter: interpret epsilon_reg as a *multiplier* on eps*scale
        jitter0 = max(0.0, float(epsilon_reg)) * eps * scale

        dist_line = np.nan
        succeeded = False
        jitter = jitter0

        # Try Cholesky with adaptive jitter escalation
        for t in range(max_jitter_tries + 1):
            Ak = A.copy()
            if t > 0 or jitter > 0.0:
                # First retry adds jitter; subsequent retries increase it geometrically
                if t > 1:
                    jitter *= 10.0
                elif t == 1 and jitter == 0.0:
                    jitter = max(rtol * scale, eps * scale)
                Ak += jitter * np.eye(A.shape[0])
                if debug:
                    print(f"Line {i_line}: Cholesky attempt {t} with jitter={jitter:.3e}")

            try:
                # Cholesky factorization (SPD only)
                L = np.linalg.cholesky(Ak)  # may raise LinAlgError if not SPD
                # Triangular solves: (Ak) y = lam_vec
                z = np.linalg.solve(L, lam_vec)
                y = np.linalg.solve(L.T, z)
                q = float(lam_vec @ y)

                # Clip tiny negatives due to roundoff: bound scale ~ eps * ||Ak|| * ||lam||^2
                err_bound = 100.0 * eps * (np.linalg.norm(Ak, ord=np.inf) * (np.linalg.norm(lam_vec) ** 2))
                if q < 0 and -q <= err_bound:
                    if debug:
                        print(f"Line {i_line}: tiny negative quadratic form {q:.3e} → 0 (bound {err_bound:.3e})")
                    q = 0.0

                dist_line = q
                succeeded = True
                break

            except np.linalg.LinAlgError:
                # Will escalate jitter and retry
                continue

        if not succeeded:
            # Fallback: PSD pseudoinverse via eigen-decomposition with clipping
            try:
                w, U = np.linalg.eigh(A)  # symmetric/hermitian eigendecomposition
                wmax = np.max(np.abs(w)) if w.size else 0.0
                tol = float(atol + rtol * wmax)
                mask = w > tol

                if debug:
                    wmin = np.min(w) if w.size else np.nan
                    print(f"Line {i_line}: eig-clip fallback: min_eig={wmin:.3e}, tol={tol:.3e}, kept={int(mask.sum())}/{len(w)}")

                if not np.any(mask):
                    dist_line = 0.0  # entirely in (approx) nullspace → quadratic form 0
                else:
                    v = U.T @ lam_vec
                    q = float(np.sum((v[mask] ** 2) / w[mask]))
                    # Numerical safety
                    dist_line = 0.0 if q < 0 and abs(q) <= 100.0 * eps * (np.linalg.norm(A, ord=np.inf) * (np.linalg.norm(lam_vec) ** 2)) else max(q, 0.0)

            except np.linalg.LinAlgError:
                print(f"Line {i_line}: ERROR - eigen-decomposition failed. Setting dist=inf.")
                dist_line = np.inf
            except Exception as e:
                print(f"Line {i_line}: Unexpected error in eig-clip fallback: {e}. Setting dist=NaN.")
                dist_line = np.nan

        distances.append((i_line, dist_line))

    return distances

def run_lagrangian_polar(z, x_init, busphase_map, Ybus, R, mpc, max_iter=20, tol=1e-4):
    """
    DSSE solver using polar coordinates.
    Args:
      z : 1D numpy array of measurements (size m), ordered consistently with `busphase_map`:
          [P_inj for all nodes, Q_inj for all nodes, |V| for all nodes]
      x_init : initial guess (2*N),
               x_init[i]   = |V|_i
               x_init[i+N] = angle_i (rad)
      busphase_map : dict (bus_id, phase_id)-> node index (0..N-1)
      Ybus : NxN complex admittance
      R : measurement covariance (m x m)
    Returns:
      x_est : final state [|V|; θ], success : bool, lambdaN : np.ndarray, line_distances : list
    """
    nnodephase = len(busphase_map)
    nstate = 2 * nnodephase

    # --- initial state: use caller-provided x_init as-is; fall back to 3φ flat if invalid
    def flat_start():
        x0 = np.zeros(nstate)
        for (bus_id, ph), idx in busphase_map.items():
            x0[idx] = 1.0
            if ph == 0:      # A
                x0[idx + nnodephase] = 0.0
            elif ph == 1:    # B
                x0[idx + nnodephase] = np.deg2rad(-120.0)
            else:            # C
                x0[idx + nnodephase] = np.deg2rad(+120.0)
        return x0

    if x_init is None or len(x_init) != nstate:
        x_current = flat_start()
    else:
        x_current = x_init.copy()

    # --- weights
    W = np.linalg.inv(R)
    W_sp = sps.csr_matrix(W)

    # --- (optional) zero-injection constraints — currently disabled in your code
    indices_to_remove = []  # keep as-is; if you reinstate ZI, repopulate this list

    # reduce measurement vector
    z = np.delete(z, indices_to_remove)

    # --- robust slack detection (get all (bus,phase) angles at slack buses)
    slack_bus_ids = [int(row[0]) for row in mpc["bus3p"] if int(row[1]) == 3]
    slack_angle_cols = []
    for b in slack_bus_ids:
        for ph in (0, 1, 2):
            key = (b, ph)
            if key in busphase_map:
                slack_angle_cols.append(busphase_map[key] + nnodephase)
    slack_angle_cols = sorted(set(slack_angle_cols))

    # columns we actually keep in the KKT (all except slack angles)
    all_state_cols = np.arange(nstate)
    keep_cols = np.setdiff1d(all_state_cols, np.array(slack_angle_cols, dtype=int))
    size_top = len(keep_cols)  # top-left block dimension (= number of free state variables)

    ########################################################
    # Gauss-Newton iteration
    ########################################################
    success = True
    last_Gain_gpu = None
    last_factor_gpu = None
    last_size_top = None
    last_nH = None
    last_nC = None
    dxl = None

    for it in range(max_iter):
        # 1) Measurement function and Jacobian
        hval = measurement_function(x_current, Ybus, mpc, busphase_map)   # full rows
        Hmat = jacobian(x_current, Ybus, mpc, busphase_map)               # (m x 2N) sparse

        # 2) Column elimination (remove slack angles)
        H_reduced_cols = Hmat[:, keep_cols]

        # 3) Row removal (zero-injection constraints -> C; rest -> H)
        C = H_reduced_cols[indices_to_remove, :] if len(indices_to_remove) else sps.csr_matrix((0, size_top))
        all_rows = np.arange(H_reduced_cols.shape[0])
        keep_rows = np.setdiff1d(all_rows, np.array(indices_to_remove, dtype=int))
        H_sp = H_reduced_cols[keep_rows, :].tocsr()
        C_sp = C.tocsr()

        # residuals aligned to H_sp (measurement rows kept)
        cval = hval[indices_to_remove] if len(indices_to_remove) else np.zeros(0)
        hval_kept = np.delete(hval, indices_to_remove)
        r_meas = z - hval_kept

        # 4) Assemble KKT
        I_H_sp = sps.eye(H_sp.shape[0], format='csc')
        zero_block_sp = sps.csr_matrix((size_top, size_top))
        H_W_sp = H_sp.transpose().dot(W_sp)
        Gain = sps.bmat(
            [[zero_block_sp, H_W_sp,     C_sp.transpose()],
             [H_sp,          I_H_sp,     None           ],
             [C_sp,          None,       None           ]],
            format='csc'
        )
        r = np.concatenate((np.zeros(size_top), r_meas, -cval))

        # 5) Solve on GPU and cache factorization for multiplier step
        Gain_csc = Gain if isinstance(Gain, sps.csc_matrix) else Gain.tocsc()
        Gain_gpu = cpsparse.csc_matrix(Gain_csc)
        r_gpu = cp.asarray(r, dtype=cp.float64)
        try:
            Gain_factor_gpu = cpsparse_linalg.splu(Gain_gpu)
            dxl_gpu = Gain_factor_gpu.solve(r_gpu)
            dxl = cp.asnumpy(dxl_gpu)
            last_Gain_gpu = Gain_gpu
            last_factor_gpu = Gain_factor_gpu
            last_size_top = size_top
            last_nH = I_H_sp.shape[0]
            last_nC = C_sp.shape[0]
        except Exception as e:
            success = False
            print(f"GPU sparse solve failed: {e}")
            break

        # 6) Extract Δx for the kept columns and lift back to full state
        dx_reduced = dxl[:size_top]
        dx_full = np.zeros(nstate)
        dx_full[keep_cols] = dx_reduced
        x_current += dx_full

        print(f"DSSE polar iter {it+1}: max |Δx|={np.max(np.abs(dx_reduced)):.3e}")
        if np.max(np.abs(dx_reduced)) < tol:
            print(f"DSSE polar converged in {it+1} iterations, max |Δx|={np.max(np.abs(dx_reduced)):.3e}")
            break
    else:
        success = False
        print("DSSE polar did not converge in max_iter steps.")

    if not success:
        return x_current, False, None, []

    # =========================
    # Multiplier / covariance
    # =========================
    dtype = cp.float64

    Hp = jacobian_line_params(x_current, Ybus, mpc, busphase_map)
    Hp_dense = Hp.toarray()
    all_rows_hp = np.arange(Hp.shape[0])
    keep_rows_hp = np.delete(all_rows_hp, indices_to_remove) if len(indices_to_remove) else all_rows_hp
    Cp_dense = Hp[indices_to_remove, :].toarray() if len(indices_to_remove) else np.zeros((0, Hp.shape[1]))
    Hp_dense = Hp_dense[keep_rows_hp, :]

    W_dense = W_sp.toarray()
    W_gpu  = cp.asarray(W_dense).astype(dtype, copy=False)
    Hp_gpu = cp.asarray(Hp_dense).astype(dtype, copy=False)
    Cp_gpu = cp.asarray(Cp_dense).astype(dtype, copy=False)

    WHp_gpu = W_gpu @ Hp_gpu
    S_gpu = cp.concatenate([WHp_gpu, Cp_gpu], axis=0)
    S_gpu = -S_gpu.T      # (n_params, b+c)
    D_inv = 1.0
    S_gpu *= D_inv

    # Reuse the last KKT factorization
    Gain_gpu = last_Gain_gpu
    Gain_factor_gpu = last_factor_gpu
    a, b, c = last_size_top, last_nH, last_nC
    n_total = Gain_gpu.shape[0]

    # Build RHS to extract E5/E8 (measurement block identity columns)
    rhs_gpu = cp.zeros((n_total, b), dtype=dtype)
    col_idx = cp.arange(b)
    rhs_gpu[a + col_idx, col_idx] = 1.0
    solutions_gpu = Gain_factor_gpu.solve(rhs_gpu)
    E5_gpu = solutions_gpu[a:a+b, :]          # (b, b)
    E8_gpu = solutions_gpu[a+b:a+b+c, :]      # (c, b)
    phi_gpu = cp.concatenate([E5_gpu, E8_gpu], axis=0)  # (b+c, b)

    # W^{-1} diagonal (vector form)
    W_diag = W_sp.diagonal().astype(np.float64)
    W_inv_cols_gpu = cp.asarray(1.0 / W_diag)

    # step1 = phi * diag(W_inv) * phi^T
    step1_gpu = (phi_gpu * W_inv_cols_gpu) @ phi_gpu.T

    print("Computing EA = S * step1 * S.T on GPU...")
    S_cov_gpu = S_gpu @ step1_gpu
    ea_gpu_full = S_cov_gpu @ S_gpu.T

    # Ensure PSD via gentle diagonal loading if needed
    try:
        cp.linalg.cholesky(ea_gpu_full)
        ea_gpu_reg = ea_gpu_full
    except cp.linalg.LinAlgError:
        epsilon = 1e-10
        ea_gpu_reg = ea_gpu_full + cp.eye(ea_gpu_full.shape[0], dtype=ea_gpu_full.dtype) * epsilon

    ea_diag_gpu = cp.diag(ea_gpu_reg)
    ea = cp.asnumpy(ea_gpu_reg)

    # λ (unnormalized) and normalized λ_N
    lam_part = dxl[a:a+b+c]               # rows corresponding to [measurements; constraints]
    if len(lam_part) != S_gpu.shape[1]:
        # pad/truncate as a last-resort guard
        target = S_gpu.shape[1]
        if len(lam_part) < target:
            lam_part = np.pad(lam_part, (0, target - len(lam_part)))
        else:
            lam_part = lam_part[:target]
    lam_part_gpu = cp.asarray(lam_part, dtype=dtype)

    lambda_vec_gpu = S_gpu @ lam_part_gpu                   # (n_params,)
    lambda_vec = cp.asnumpy(lambda_vec_gpu)
    lambdaN_gpu = lambda_vec_gpu / cp.sqrt(ea_diag_gpu + cp.finfo(ea_diag_gpu.dtype).eps)
    lambdaN = cp.asnumpy(lambdaN_gpu)

    print("Lambda computations finished.")

    line_distances = compute_line_mahalanobis_distances(lambda_vec, ea, mpc)
    for i_line, dval in line_distances:
        print(f"Line {i_line + 1} => Mahalanobis distance = {dval:.3f}")
    max_line, max_dist = max(line_distances, key=lambda x: x[1])
    print(f"Line {max_line + 1} has the largest Mahalanobis distance: {max_dist:.6f}")

    return x_current, success, lambdaN, line_distances
