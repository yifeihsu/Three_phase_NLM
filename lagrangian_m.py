import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu, LinearOperator
import scipy.sparse.linalg as spla
import cupy as cp
from scipy.linalg import solve_triangular
import cupyx.scipy.sparse as cpsparse
import cupyx.scipy.sparse.linalg as cpsparse_linalg
from utilities.mea_fun import measurement_function
from utilities.make_jacobian import jacobian
from utilities.make_jacobian_p import jacobian_line_params, jacobian_shunt_params

def find_zero_injection_nodes(mpc, busphase_map, tol=1e6):
    """
    Identify phase-level nodes (bus-phase) that have effectively zero net injection,
    i.e. Generation - Load is within +/- tol (kW or kVar).
    """
    bus3p = mpc["bus3p"]
    load3p = mpc["load3p"]
    gen3p  = mpc["gen3p"]

    bus_ids = [int(row[0]) for row in bus3p]
    slack_buses = [int(row[0]) for row in bus3p if row[1] == 3]

    netPQ_phase = {(b, ph): [0.0, 0.0] for b in bus_ids for ph in (0, 1, 2)}

    for row in load3p:
        ldid, ldbus, status, PdA, PdB, PdC, QdA, QdB, QdC = row
        if status == 0: continue
        bus_id = int(ldbus)
        if abs(PdA) > 1e-9 or abs(QdA) > 1e-9:
            netPQ_phase[(bus_id, 0)][0] -= PdA; netPQ_phase[(bus_id, 0)][1] -= QdA
        if abs(PdB) > 1e-9 or abs(QdB) > 1e-9:
            netPQ_phase[(bus_id, 1)][0] -= PdB; netPQ_phase[(bus_id, 1)][1] -= QdB
        if abs(PdC) > 1e-9 or abs(QdC) > 1e-9:
            netPQ_phase[(bus_id, 2)][0] -= PdC; netPQ_phase[(bus_id, 2)][1] -= QdC

    for row in gen3p:
        genid, gbus, status, VgA, VgB, VgC, PgA, PgB, PgC, QgA, QgB, QgC = row
        if status == 0: continue
        bus_id = int(gbus)
        if abs(PgA) > 1e-9 or abs(QgA) > 1e-9:
            netPQ_phase[(bus_id, 0)][0] += PgA; netPQ_phase[(bus_id, 0)][1] += QgA
        if abs(PgB) > 1e-9 or abs(QgB) > 1e-9:
            netPQ_phase[(bus_id, 1)][0] += PgB; netPQ_phase[(bus_id, 1)][1] += QgB
        if abs(PgC) > 1e-9 or abs(QgC) > 1e-9:
            netPQ_phase[(bus_id, 2)][0] += PgC; netPQ_phase[(bus_id, 2)][1] += QgC

    zero_inj_nodes = []
    for (b, ph), (p_net, q_net) in netPQ_phase.items():
        if b in slack_buses: continue
        if (abs(p_net) < tol) and (abs(q_net) < tol):
            node_idx = busphase_map[(b, ph)]
            zero_inj_nodes.append(node_idx)
    return zero_inj_nodes
def quadratic_form_with_target_cond(A, lam_vec,
                                    kappa_target=1e8,
                                    rtol=None, atol=0.0,
                                    equilibrate=True,
                                    debug=False):
    """
    Computes q = lam^T A^{-1} lam with conditioning control.
    Returns (q, Ak_used, tau_used, kappa_est).
    """
    A = np.asarray(A, float)
    lam = np.asarray(lam_vec, float)

    # Symmetrize & remove tiny imaginary parts
    A = 0.5 * (A + A.T)
    A = np.real_if_close(A, tol=1e-12)
    lam = np.real_if_close(lam, tol=1e-12)

    # Optional diagonal equilibration
    if equilibrate:
        d = np.diag(A).copy()
        wmax_diag = float(np.max(np.abs(d))) if d.size else 0.0
        tol_diag  = float((rtol or 0.0) * wmax_diag + atol)
        d_safe = np.sqrt(np.maximum(d, tol_diag if tol_diag > 0 else 1e-30))
        Dinv = np.diag(1.0 / d_safe)
        Aeq = Dinv @ A @ Dinv
        lameq = Dinv @ lam
    else:
        Aeq = A
        lameq = lam

    # Eigenvalues to plan tau
    w = np.linalg.eigvalsh(Aeq)
    w = np.real_if_close(w, tol=1e-12)
    wmax = float(np.max(w))
    # If rtol not specified, tie it to the desired condition number
    if rtol is None:
        rtol = 1.0 / kappa_target
    tol_ev = float(atol + rtol * wmax)

    wmin_all = float(np.min(w))
    wpos = w[w > tol_ev]
    if wpos.size == 0:
        # Everything is at/below tol -> return 0
        return 0.0, Aeq, tol_ev - wmin_all, np.inf

    wmin = float(np.min(wpos))

    # Ridge to meet condition and to lift negative/tiny modes to tol_ev
    tau_cond = max(0.0, (wmax - kappa_target * wmin) / (kappa_target - 1.0))
    tau_neg  = max(0.0, tol_ev - wmin_all)
    tau = max(tau_cond, tau_neg)

    Ak = Aeq + tau * np.eye(Aeq.shape[0])

    # Estimate conditioned kappa quickly
    wAk_min = float(np.min(np.linalg.eigvalsh(Ak)))
    kappa_est = (wmax + tau) / wAk_min if wAk_min > 0 else np.inf

    # Try Cholesky on Ak; if it fails or kappa is still huge, fall back to eig-clip
    try:
        L = np.linalg.cholesky(Ak)
        z = np.linalg.solve(L, lameq)
        y = np.linalg.solve(L.T, z)
        q = float(lameq @ y)
        # Safety: clip tiny negatives
        if q < 0 and -q <= 100*np.finfo(float).eps*np.linalg.norm(Ak, np.inf)*np.linalg.norm(lameq)**2:
            q = 0.0
        return q, Ak, tau, kappa_est
    except np.linalg.LinAlgError:
        # PSD eigen-clip (keeps > tol_ev)
        w, U = np.linalg.eigh(Aeq)
        mask = w > tol_ev
        if not np.any(mask):
            return 0.0, Aeq, tau, np.inf
        v = U.T @ lameq
        q = float(np.sum((v[mask]**2) / w[mask]))
        return max(q, 0.0), Aeq, tau, np.inf

def compute_line_mahalanobis_distances(lambdaN, EA, mpc,
                                       epsilon_reg=1,
                                       rtol=1e-12,
                                       atol=0.0,
                                       max_jitter_tries=8,
                                       debug=False):
    if np.any(~np.isfinite(lambdaN)):
        print("Warning: NaN or Inf detected in input lambdaN!")
    if np.any(~np.isfinite(EA)):
        print("Warning: NaN or Inf detected in input EA!")

    line_data = mpc["line3p"]
    num_lines = len(line_data)
    nparam_line = 12

    distances = []
    eps = np.finfo(float).eps

    for i_line in range(num_lines):
        start_idx = i_line * nparam_line
        end_idx = start_idx + nparam_line

        lam_vec = np.asarray(lambdaN[start_idx:end_idx], dtype=float)
        A = np.asarray(EA[start_idx:end_idx, start_idx:end_idx], dtype=float)

        if np.any(~np.isfinite(lam_vec)) or np.any(~np.isfinite(A)):
            if debug:
                print(f"Line {i_line}: Non-finite entries in lam_vec/EA_sub → distance = NaN.")
            distances.append((i_line, np.nan))
            continue

        A = 0.5 * (A + A.T)
        diagA = np.diag(A)
        scale = float(np.mean(np.abs(diagA))) if np.all(np.isfinite(diagA)) else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = max(1.0, float(np.linalg.norm(A, ord=np.inf)))
        jitter0 = max(0.0, float(epsilon_reg)) * eps * scale

        dist_line = np.nan
        succeeded = False
        jitter = jitter0

        for t in range(max_jitter_tries + 1):
            Ak = A.copy()
            if t > 0 or jitter > 0.0:
                if t > 1:
                    jitter *= 10.0
                elif t == 1 and jitter == 0.0:
                    jitter = max(rtol * scale, eps * scale)
                Ak += jitter * np.eye(A.shape[0])

            try:
                L = np.linalg.cholesky(Ak)
                z = np.linalg.solve(L, lam_vec)
                y = np.linalg.solve(L.T, z)
                q = float(lam_vec @ y)
                err_bound = 100.0 * eps * (np.linalg.norm(Ak, ord=np.inf) * (np.linalg.norm(lam_vec) ** 2))
                if q < 0 and -q <= err_bound:
                    q = 0.0
                dist_line = q
                succeeded = True
                break
            except np.linalg.LinAlgError:
                continue

        if not succeeded:
            try:
                w, U = np.linalg.eigh(A)
                wmax = np.max(np.abs(w)) if w.size else 0.0
                tol = float(atol + rtol * wmax)
                mask = w > tol
                if not np.any(mask):
                    dist_line = 0.0
                else:
                    v = U.T @ lam_vec
                    q = float(np.sum((v[mask] ** 2) / w[mask]))
                    dist_line = 0.0 if q < 0 and abs(q) <= 100.0 * eps * (np.linalg.norm(A, ord=np.inf) * (np.linalg.norm(lam_vec) ** 2)) else max(q, 0.0)
            except np.linalg.LinAlgError:
                print(f"Line {i_line}: ERROR - eigen-decomposition failed. Setting dist=inf.")
                dist_line = np.inf
            except Exception as e:
                print(f"Line {i_line}: Unexpected error in eig-clip fallback: {e}. Setting dist=NaN.")
                dist_line = np.nan

        distances.append((i_line, dist_line))
    return distances
def compute_terminal_shunt_mahalanobis_distances(lambda_vec, EA, mpc, param_map, epsilon_reg=1e-8):
    shunt_param_info = param_map.get('shunt')
    if not shunt_param_info:
        print("Warning: Shunt parameter map is missing.")
        return []

    distances = []
    for i_line, line_data in enumerate(mpc["line3p"]):
        from_bus = int(line_data[1])
        to_bus = int(line_data[2])

        group_indices = []
        for bus_id in [from_bus, to_bus]:
            for phase_idx in range(3):
                key = (bus_id, phase_idx)
                if key in shunt_param_info:
                    start_idx, num_params = shunt_param_info[key]
                    group_indices.extend(range(start_idx, start_idx + num_params))

        group_indices = sorted(list(set(group_indices)))
        if not group_indices:
            continue

        lam_sub = lambda_vec[group_indices]
        EA_sub = EA[np.ix_(group_indices, group_indices)]

        dist_line = np.nan
        try:
            try:
                L = np.linalg.cholesky(EA_sub)
            except np.linalg.LinAlgError:
                EA_sub = EA_sub + np.eye(EA_sub.shape[0]) * epsilon_reg
                L = np.linalg.cholesky(EA_sub)

            y = np.linalg.solve(L, lam_sub)
            chi2 = float(np.dot(y, y))
            k = len(lam_sub)
            dist_line = chi2 - k
        except np.linalg.LinAlgError:
            print(f"ERROR: Cholesky failed for terminal shunts of line {i_line}. Dist set to inf.")
            dist_line = np.inf

        distances.append((i_line, dist_line))
    return distances
def compute_line_group_mahalanobis_distances(lambda_vec, EA, mpc, param_map,
                                             epsilon_reg=0, rtol=1e-15, atol=0.0,
                                             max_jitter_tries=12, debug=False):
    """
    Group χ² index for each line that INCLUDES:
      - the line's own impedance parameters, and
      - shunt admittance parameters at BOTH terminals (all phases).

    Uses the same robust numerical strategy as compute_line_mahalanobis_distances:
    jittered Cholesky with geometric escalation, then PSD eig-clip fallback.

    Returns:
      list[(i_line, chi2_minus_k)]
    """
    line_info  = param_map.get('line', {})
    shunt_info = param_map.get('shunt', {})

    if not line_info:
        print("Warning: Line parameter map is missing.")
        return []

    distances = []
    for i_line, line_row in enumerate(mpc["line3p"]):
        from_bus = int(line_row[1])
        to_bus   = int(line_row[2])

        # 1) line-parameter block for this line
        if i_line not in line_info:
            continue
        s_line, n_line = line_info[i_line]
        idx_line = list(range(s_line, s_line + n_line))

        # 2) shunt blocks for both terminals (all 3 phases)
        idx_shunt = []
        for bus in (from_bus, to_bus):
            for ph in (0, 1, 2):
                key = (bus, ph)
                if key in shunt_info:
                    s_b, n_b = shunt_info[key]
                    idx_shunt.extend(range(s_b, s_b + n_b))

        group_indices = sorted(set(idx_line + idx_shunt))
        if not group_indices:
            continue

        lam_sub = np.asarray(lambda_vec[group_indices], dtype=float)
        EA_sub  = np.asarray(EA[np.ix_(group_indices, group_indices)], dtype=float)

        # Robust q = λ^T Σ^{-1} λ
        q, Ak_used, tau_used, kappa_est = quadratic_form_with_target_cond(
            EA_sub, lam_sub,
            kappa_target=1e8,  # or 1e10 if you’re comfortable
            rtol=1e-8,  # rtol ≈ 1/κ_target; else set None to auto
            atol=0.0,
            equilibrate=True,
            debug=False
        )

        # χ² index with k = group dimension (for zero-mean null, E[χ²-k] ≈ 0)
        k = len(lam_sub)
        dist = (q - k) if np.isfinite(q) else q

        distances.append((i_line, float(dist)))

    return distances

def _diag_of_matrix(M):
    if sps.issparse(M):
        return np.asarray(M.diagonal()).ravel()
    return np.asarray(np.diag(M)).ravel()


def _is_diagonal_matrix(M, tol=0.0):
    if sps.issparse(M):
        A = M.tocoo()
        return np.all(np.abs(A.data[np.where(A.row != A.col)]) <= tol)
    else:
        off = M.copy()
        np.fill_diagonal(off, 0.0)
        return np.all(np.abs(off) <= tol)


def _build_zero_injection_row_indices(zero_nodes, nnodephase):
    """
    Given zero-injection node indices (0..N-1), return the measurement row indices
    (P and Q) to be removed from the regular measurement block and moved to C.
    z ordering: [P_inj (N), Q_inj (N), |V| (N)]
    """
    zero_nodes = np.asarray(zero_nodes, dtype=int)
    p_rows = zero_nodes
    q_rows = nnodephase + zero_nodes
    return np.sort(np.r_[p_rows, q_rows])


def run_lagrangian_polar(
    z, x_init, busphase_map, Ybus, R, mpc,
    max_iter=20, tol=1e-4,
    zi_enable=False, zi_tol_kw=1.0,
    alpha=1e-8,
    gpu_permc="COLAMD", gpu_diag_pivot_thresh=1.0,
    debug=False
):
    """
    DSSE solver (polar coordinates) with whitened augmented/Hachtel KKT and
    equality constraints for zero-injections.

    Now computes normalized multipliers (NLMs) for BOTH line impedances and
    shunt admittances, and evaluates per-line group χ² indices that include
    the line's params + the two terminal shunts.
    """

    # ---------- dimensions & initial state ----------
    nnodephase = len(busphase_map)
    nstate = 2 * nnodephase

    def flat_start():
        x0 = np.zeros(nstate)
        for (bus_id, ph), idx in busphase_map.items():
            x0[idx] = 1.0
            if ph == 0:
                x0[idx + nnodephase] = 0.0
            elif ph == 1:
                x0[idx + nnodephase] = np.deg2rad(-120.0)
            else:
                x0[idx + nnodephase] = np.deg2rad(+120.0)
        return x0

    x_current = flat_start() if (x_init is None or len(x_init) != nstate) else x_init.copy()

    # ---------- slack angle elimination ----------
    slack_bus_ids = [int(row[0]) for row in mpc["bus3p"] if int(row[1]) == 3]
    slack_angle_cols = []
    for b in slack_bus_ids:
        for ph in (0, 1, 2):
            key = (b, ph)
            if key in busphase_map:
                slack_angle_cols.append(busphase_map[key] + nnodephase)
    slack_angle_cols = sorted(set(slack_angle_cols))
    all_state_cols = np.arange(nstate)
    keep_cols = np.setdiff1d(all_state_cols, np.array(slack_angle_cols, dtype=int))
    size_top = len(keep_cols)

    # ---------- zero-injection selection ----------
    indices_to_remove = []
    if zi_enable:
        zero_nodes = find_zero_injection_nodes(mpc, busphase_map, tol=zi_tol_kw)
        if len(zero_nodes):
            indices_to_remove = _build_zero_injection_row_indices(zero_nodes, nnodephase).tolist()

    # ---------- constants ----------
    success = True
    last_Gain_gpu = None
    last_factor_gpu = None
    last_size_top = None
    last_nH = None
    last_nC = None
    dxl = None

    # ---------- Gauss-Newton iterations ----------
    for it in range(max_iter):
        # (1) h(x) and H
        hval = measurement_function(x_current, Ybus, mpc, busphase_map)
        Hmat = jacobian(x_current, Ybus, mpc, busphase_map)  # (m x 2N) sparse

        # (2) remove slack angle columns
        H_reduced_cols = Hmat[:, keep_cols]

        # (3) split equality constraints (C) and regular measurement rows (H_sp)
        all_rows = np.arange(H_reduced_cols.shape[0], dtype=int)
        rem_idx = np.array(indices_to_remove, dtype=int) if len(indices_to_remove) else np.array([], dtype=int)
        keep_rows = np.setdiff1d(all_rows, rem_idx)

        C_sp = H_reduced_cols[rem_idx, :].tocsr() if len(rem_idx) else sps.csr_matrix((0, size_top))
        H_sp = H_reduced_cols[keep_rows, :].tocsr()

        # residuals aligned with kept measurement rows
        z_kept   = np.asarray(z)[keep_rows]
        h_kept   = np.asarray(hval)[keep_rows]
        r_meas   = z_kept - h_kept
        cval     = np.asarray(hval)[rem_idx] if len(rem_idx) else np.zeros(0)

        # (4) Whitening
        sqrt_alpha = float(np.sqrt(alpha))
        sigma2_all  = _diag_of_matrix(R)
        sigma2_kept = sigma2_all[keep_rows]
        sigma2_kept = np.where(sigma2_kept > 0.0, sigma2_kept, np.finfo(float).eps)
        rhalf = 1.0 / np.sqrt(sigma2_kept)

        F_sp    = sps.diags(sqrt_alpha * rhalf) @ H_sp
        rhs_meas = (sqrt_alpha * rhalf) * r_meas

        # (5) Assemble augmented/Hachtel KKT
        I_meas = sps.eye(F_sp.shape[0], format='csc')
        zero_block = sps.csr_matrix((size_top, size_top))
        Gain = sps.bmat(
            [[zero_block,       F_sp.transpose().tocsr(),   C_sp.transpose()],
             [F_sp,             I_meas,                     None           ],
             [C_sp,             None,                       None           ]],
            format='csc'
        )
        r_rhs = np.concatenate((np.zeros(size_top), rhs_meas, -cval))

        # (6) Solve on GPU (fallback CPU)
        Gain_csc = Gain if isinstance(Gain, sps.csc_matrix) else Gain.tocsc()
        r_gpu = cp.asarray(r_rhs, dtype=cp.float64)
        try:
            Gain_gpu = cpsparse.csc_matrix(Gain_csc)
            Gain_factor_gpu = cpsparse_linalg.splu(
                Gain_gpu, permc_spec=gpu_permc, diag_pivot_thresh=gpu_diag_pivot_thresh
            )
            dxl_gpu = Gain_factor_gpu.solve(r_gpu)
            dxl = cp.asnumpy(dxl_gpu)

            last_Gain_gpu   = Gain_gpu
            last_factor_gpu = Gain_factor_gpu
            last_size_top   = size_top
            last_nH         = I_meas.shape[0]
            last_nC         = C_sp.shape[0]
        except Exception as e:
            try:
                Gain_cpu = Gain_csc
                lu = splu(Gain_cpu, permc_spec="COLAMD", diag_pivot_thresh=1.0)
                dxl = lu.solve(r_rhs)
                last_Gain_gpu   = None
                last_factor_gpu = None
                last_size_top   = size_top
                last_nH         = I_meas.shape[0]
                last_nC         = C_sp.shape[0]
                if debug:
                    print(f"[Fallback CPU LU] {e}")
            except Exception as e2:
                success = False
                print(f"Augmented KKT solve failed (GPU & CPU): {e2}")
                break

        # (7) update state
        dx_reduced = dxl[:size_top]
        dx_full = np.zeros(nstate)
        dx_full[keep_cols] = dx_reduced
        x_current += dx_full

        print(f"DSSE polar iter {it+1}: max |Δx|={np.max(np.abs(dx_reduced)):.3e}")
        if np.max(np.abs(dx_reduced)) < tol:
            if debug:
                print(f"Converged in {it+1} iterations.")
            break
    else:
        success = False
        print("DSSE polar did not converge within max_iter.")

    if not success:
        return x_current, False, None, []

    # =========================
    # Parameter multipliers & covariance (LINE + SHUNT)
    # =========================
    dtype = cp.float64

    # --- 1) Build parameter Jacobians ---
    # Line impedances
    Hp_line  = jacobian_line_params(x_current, Ybus, mpc, busphase_map)   # (m x n_line_params) sparse
    n_line_params_total = Hp_line.shape[1]
    n_lines = len(mpc['line3p'])
    # Derive per-line parameter count robustly (falls back to 12 if needed)
    nparam_line = (n_line_params_total // n_lines) if (n_lines and n_line_params_total % n_lines == 0) else 12

    # Shunt admittances
    Hp_shunt = jacobian_shunt_params(x_current, mpc, busphase_map)        # (m x n_shunt_params) sparse
    n_shunt_params_total = Hp_shunt.shape[1]

    # Combined parameter Jacobian: [Hp_line | Hp_shunt]
    Hp = sps.hstack([Hp_line, Hp_shunt], format='csr')

    # --- 2) Parameter map for decoding combined λ/EA ---
    param_map = {'line': {}, 'shunt': {}}

    # line indices
    for i_line in range(n_lines):
        start_idx = i_line * nparam_line
        param_map['line'][i_line] = (start_idx, nparam_line)

    # shunt indices (start AFTER all line params)
    shunt_offset = n_line_params_total
    shunt_bus_phase_ordered = []
    for bus_row in mpc['bus3p']:
        bus_id = int(bus_row[0])
        for phase_idx in range(3):
            shunt_bus_phase_ordered.append((bus_id, phase_idx))
    for i, (bus_id, phase_idx) in enumerate(shunt_bus_phase_ordered):
        start_idx = shunt_offset + i * 2  # 2 params/phase: G and B
        param_map['shunt'][(bus_id, phase_idx)] = (start_idx, 2)

    # --- 3) Partition rows (keep vs constraints), whiten Hp rows same as KKT ---
    all_rows_hp = np.arange(Hp.shape[0])
    rem_idx = np.array(indices_to_remove, dtype=int) if len(indices_to_remove) else np.array([], dtype=int)
    keep_rows_hp = np.setdiff1d(all_rows_hp, rem_idx)

    Hp_dense = Hp[keep_rows_hp, :].toarray()
    Cp_dense = Hp[rem_idx, :].toarray() if len(rem_idx) else np.zeros((0, Hp.shape[1]))

    if _is_diagonal_matrix(R):
        sigma2_all = _diag_of_matrix(R)
        sigma2_kept = sigma2_all[keep_rows_hp]
        sigma2_kept = np.where(sigma2_kept > 0.0, sigma2_kept, np.finfo(float).eps)
        rhalf_kept = 1.0 / np.sqrt(sigma2_kept)
        Hp_wh = (np.atleast_2d(sqrt_alpha * rhalf_kept).T) * Hp_dense
    else:
        R_dense = np.asarray(R, dtype=float)
        L = np.linalg.cholesky(R_dense)
        Hp_wh = solve_triangular(L.T, Hp_dense, lower=False)
        Hp_wh = sqrt_alpha * Hp_wh

    # ---- Build S = -[Hp_wh; Cp]^T ----
    Hp_gpu = cp.asarray(Hp_wh).astype(dtype, copy=False)
    Cp_gpu = cp.asarray(Cp_dense).astype(dtype, copy=False)
    S_gpu = -cp.concatenate([Hp_gpu, Cp_gpu], axis=0).T

    # ---- Reuse last factorization to get phi = [E5; E8] ----
    a, b, c = last_size_top, last_nH, last_nC
    n_total = a + b + c

    if last_factor_gpu is None:
        # CPU path
        rhs = np.zeros((n_total, b))
        for j in range(b):
            rhs[a + j, j] = 1.0
        Gain_cpu = Gain_csc
        lu = splu(Gain_cpu, permc_spec="COLAMD", diag_pivot_thresh=1.0)
        sols = lu.solve(rhs)                      # (n_total x b)
        E5 = sols[a:a+b, :]
        E8 = sols[a+b:a+b+c, :] if c > 0 else np.zeros((0, b))
        phi_gpu = cp.asarray(np.vstack([E5, E8])).astype(dtype, copy=False)
    else:
        rhs_gpu = cp.zeros((n_total, b), dtype=dtype)
        col_idx = cp.arange(b)
        rhs_gpu[a + col_idx, col_idx] = 1.0
        solutions_gpu = last_factor_gpu.solve(rhs_gpu)
        E5_gpu = solutions_gpu[a:a+b, :]
        E8_gpu = solutions_gpu[a+b:a+b+c, :] if c > 0 else cp.zeros((0, b), dtype=dtype)
        phi_gpu = cp.concatenate([E5_gpu, E8_gpu], axis=0)

    # ---- step1 = α * (phi @ phi^T) ----
    step1_gpu = alpha * (phi_gpu @ phi_gpu.T)

    # ---- EA = S * step1 * S^T ----
    S_cov_gpu  = S_gpu @ step1_gpu
    ea_gpu_full = S_cov_gpu @ S_gpu.T

    # Gentle diagonal loading if needed
    try:
        cp.linalg.cholesky(ea_gpu_full)
        ea_gpu_reg = ea_gpu_full
    except cp.linalg.LinAlgError:
        eps = cp.finfo(ea_gpu_full.dtype).eps
        jitter = 1e3 * eps
        ea_gpu_reg = ea_gpu_full + cp.eye(ea_gpu_full.shape[0], dtype=ea_gpu_full.dtype) * jitter

    ea = cp.asnumpy(ea_gpu_reg)
    ea_diag_gpu = cp.diag(ea_gpu_reg)

    # ---- Parameter multipliers λ and normalized λ_N (combined: [line | shunt]) ----
    lam_part      = dxl[a:a+b+c]               # rows corresponding to [residuals; constraints]
    lam_part_gpu  = cp.asarray(lam_part, dtype=dtype)
    lambda_vec_gpu = S_gpu @ lam_part_gpu
    lambda_vec     = cp.asnumpy(lambda_vec_gpu)

    lambdaN_gpu = lambda_vec_gpu / cp.sqrt(ea_diag_gpu + cp.finfo(ea_gpu_full.dtype).eps)
    lambdaN     = cp.asnumpy(lambdaN_gpu)

    # ---------------------------
    # Assessments / Diagnostics
    # ---------------------------
    print("\n--- Parameter Assessment Results (Line Impedances + Shunt Admittances) ---")

    # a) Individual parameter tests (category-wise maxima)
    if n_line_params_total > 0:
        line_slice = slice(0, n_line_params_total)
        idx_line_max = int(np.argmax(np.abs(lambdaN[line_slice])))
        val_line_max = float(np.abs(lambdaN[idx_line_max]))
        print(f"Max |λ_N| over LINE parameters: {val_line_max:.4f} (global index {idx_line_max})")
    else:
        print("No line parameters detected in Jacobian.")

    if n_shunt_params_total > 0:
        shunt_slice = slice(n_line_params_total, n_line_params_total + n_shunt_params_total)
        idx_shunt_rel = int(np.argmax(np.abs(lambdaN[shunt_slice])))
        idx_shunt_max = n_line_params_total + idx_shunt_rel
        val_shunt_max = float(np.abs(lambdaN[idx_shunt_max]))
        # Decode bus/phase if mapping aligns with full bus-phase ordering
        bus_phase_idx = idx_shunt_rel // 2
        if bus_phase_idx < len(shunt_bus_phase_ordered):
            bus_id, phase_idx = shunt_bus_phase_ordered[bus_phase_idx]
            phase_str = {0: 'A', 1: 'B', 2: 'C'}.get(phase_idx, '?')
            param_type = "G" if (idx_shunt_rel % 2) == 0 else "B"
            print(f"Max |λ_N| over SHUNT parameters: {val_shunt_max:.4f} "
                  f"(Bus {bus_id}, Phase {phase_str}, Param {param_type}, global index {idx_shunt_max})")
        else:
            print(f"Max |λ_N| over SHUNT parameters: {val_shunt_max:.4f} (global index {idx_shunt_max})")
    else:
        print("No shunt parameters detected in Jacobian.")

    # b) Group test: Line-group χ² index (line params + terminal shunts)
    print("\nGroup χ² Test: (Line parameters) ∪ (Shunts at both terminals)")
    line_group_distances = compute_line_group_mahalanobis_distances(
        lambda_vec, ea, mpc, param_map,
        epsilon_reg=1.0, rtol=1e-12, atol=0.0, max_jitter_tries=8, debug=False
    )
    sorted_distances = []
    if line_group_distances:
        sorted_distances = sorted(line_group_distances, key=lambda item: item[1], reverse=True)
        print("Top 5 most likely erroneous line groups:")
        for i_line, d_val in sorted_distances[:5]:
            fbus, tbus = int(mpc['line3p'][i_line][1]), int(mpc['line3p'][i_line][2])
            print(f"  - Line {i_line+156} ({fbus}-{tbus}): χ² - k = {d_val:.4f}")
        best_idx, best_val = sorted_distances[0]
        fbus_top, tbus_top = int(mpc['line3p'][best_idx][1]), int(mpc['line3p'][best_idx][2])
        print(f"\nConclusion: Line {best_idx+156} ({fbus_top}-{tbus_top}) "
              f"(its params + terminal shunts) has the largest group score ({best_val:.4f}).")
    else:
        print("Could not compute line-group Mahalanobis distances.")

    # Return the final state, success flag, combined normalized multipliers, and the sorted group list
    return x_current, True, lambdaN, sorted_distances


# ---------- 1-norm condition-number estimate (cheap & robust) ----------
def cond1_estimate_sparse(A, lu=None, itmax=8):
    """
    Estimate cond_1(A) ~= ||A||_1 * ||A^{-1}||_1 for a sparse matrix A.
    """
    if not sps.isspmatrix(A):
        A = sps.csc_matrix(A)
    if lu is None:
        lu = spla.splu(A.tocsc(), permc_spec="COLAMD", diag_pivot_thresh=1.0)

    normA1 = spla.norm(A, 1)

    n = A.shape[0]
    def matvec_inv(x):
        return lu.solve(x)
    def rmatvec_inv(x):
        return lu.solve(x, 'T')

    Ainv = LinearOperator((n, n), matvec=matvec_inv, rmatvec=rmatvec_inv, dtype=A.dtype)
    normAinv1 = spla.onenormest(Ainv, t=2, itmax=itmax)
    return float(normA1 * normAinv1)
