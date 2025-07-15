import numpy as np
import scipy.sparse as sp

from .mea_fun import measurement_function
from .make_jacobian import jacobian
from .make_jacobian_p import jacobian_line_params, build_param_info
from parse_opendss_file import build_global_y_per_unit


def _expand_state_vector(x_sub, nb, ref_bus):
    """Expand reduced state vector back to full dimension."""
    n_states_per_scan = 2 * nb - 1
    assert len(x_sub) == n_states_per_scan
    angles = np.zeros(nb)
    mags = np.zeros(nb)
    # layout: [angles_without_ref; mags]
    angles_idx = list(range(nb))
    angles_idx.pop(ref_bus)
    angles[angles_idx] = x_sub[: nb - 1]
    mags[:] = x_sub[nb - 1:]
    return angles, mags


def _update_line_params_in_mpc(mpc, line_idx, new_R, new_X):
    """Helper to overwrite a line's R/X in the mpc structure.

    This assumes each line has an associated line code stored in ``mpc['lc']``.
    The line row contains a ``lcid`` in column 4.  All six entries of the R and
    X matrices for that line code are replaced with ``new_R`` and ``new_X``.
    """
    line_row = mpc["line3p"][line_idx]
    lcid = int(line_row[4])
    mask = mpc["lc"][:, 0] == lcid
    if np.any(mask):
        idx = np.where(mask)[0][0]
        mpc["lc"][idx, 1:7] = new_R
        mpc["lc"][idx, 7:13] = new_X

    # Try updating the active OpenDSS line object as well so that
    # ``build_global_y_per_unit`` reflects the new parameters.
    try:
        import opendssdirect as dss

        line_name = str(line_row[6])
        dss.Lines.Name(line_name)

        # Convert the symmetric-lower vector [r11, r21, r31, r22, r32, r33]
        # to a full 3x3 matrix in row-major order.
        rmat = [
            new_R[0], new_R[1], new_R[2],
            new_R[1], new_R[3], new_R[4],
            new_R[2], new_R[4], new_R[5],
        ]
        xmat = [
            new_X[0], new_X[1], new_X[2],
            new_X[1], new_X[3], new_X[4],
            new_X[2], new_X[4], new_X[5],
        ]

        dss.Lines.RMatrix(rmat)
        dss.Lines.XMatrix(xmat)
    except Exception:
        # If OpenDSS is not available or the update fails, ignore.
        pass


def calculate_param_jacobian_for_line(x, Ybus, mpc, busphase_map, line_idx):
    """Return Jacobian of measurements w.r.t. [R,X] of a given line."""
    Hp_full = jacobian_line_params(x, Ybus, mpc, busphase_map)
    param_info = build_param_info(mpc)
    cols = []
    start_idx = line_idx * 12
    cols.append(start_idx + 0)  # Zaa_R
    cols.append(start_idx + 6)  # Zaa_X
    return Hp_full[:, cols]


def correct_parameter_group_multi_scan(
    mpc_with_error,
    line_to_correct_idx,
    multi_scan_measurements_z,
    initial_states_multi_scan,
    R_variances_vec,
    busphase_map,
    baseMVA,
    max_iter=30,
    tol=5e-3,
):
    """Correct R and X of a specific line using multiple measurement scans.

    Parameters are adapted from the MATLAB reference implementation but
    tailored for the three-phase distribution ``mpc`` structure.
    ``line_to_correct_idx`` is 0-based.
    Returns ``(corrected_params, success_flag)`` where ``corrected_params`` is
    ``[R_est, X_est]``.
    """

    nb = len(busphase_map)
    s = multi_scan_measurements_z.shape[1]

    success_correction = 0

    # extract current parameter guess from the mpc structure
    line_row = mpc_with_error["line3p"][line_to_correct_idx]
    lcid = int(line_row[4])
    mask = mpc_with_error["lc"][:, 0] == lcid
    idx = np.where(mask)[0][0]
    R_init = float(mpc_with_error["lc"][idx, 1])
    X_init = float(mpc_with_error["lc"][idx, 7])
    p_g_current = np.array([R_init, X_init], dtype=float)
    corrected_params_group = p_g_current.copy()

    nz_single = multi_scan_measurements_z.shape[0]
    R_single = sp.diags(R_variances_vec, 0, shape=(nz_single, nz_single))
    R_inv_single = sp.linalg.inv(R_single)

    n_states_per_scan = 2 * nb - 1
    N = s * n_states_per_scan + 2

    ref_bus = 0
    for i, row in enumerate(mpc_with_error["bus3p"]):
        if int(row[1]) == 3:
            ref_bus = i
            break

    x_states_all = np.zeros(s * n_states_per_scan)
    for k in range(s):
        v_init = initial_states_multi_scan[:nb, k]
        a_deg = initial_states_multi_scan[nb:, k]
        a_rad = np.radians(a_deg)
        a_sub = np.delete(a_rad, ref_bus)
        x_states_all[k * n_states_per_scan : (k + 1) * n_states_per_scan] = np.hstack(
            [a_sub, v_init]
        )

    x_v_current = np.concatenate([x_states_all, p_g_current])

    for it in range(max_iter):
        G_v = sp.csc_matrix((N, N))
        RHS = np.zeros(N)

        for k in range(s):
            idx_start = k * n_states_per_scan
            idx_end = (k + 1) * n_states_per_scan
            x_scan_k = x_v_current[idx_start:idx_end]
            theta_full, V_full = _expand_state_vector(x_scan_k, nb, ref_bus)
            mpc_iter = mpc_with_error.copy()
            _update_line_params_in_mpc(mpc_iter, line_to_correct_idx, [p_g_current[0]] * 6, [p_g_current[1]] * 6)
            Ybus_iter, _ = build_global_y_per_unit(mpc_iter)
            x_full = np.hstack([V_full, theta_full])
            hx_k = measurement_function(x_full, Ybus_iter, mpc_iter, busphase_map)
            z_k = multi_scan_measurements_z[:, k]
            delta_z_k = z_k - hx_k
            J_full = jacobian(x_full, Ybus_iter, mpc_iter, busphase_map)
            J_full = np.delete(J_full, ref_bus + nb, axis=1)
            J_full = np.delete(J_full, ref_bus, axis=1)
            Hp_k = calculate_param_jacobian_for_line(x_full, Ybus_iter, mpc_iter, busphase_map, line_to_correct_idx)

            Wblock = R_inv_single
            Ht_W = J_full.T @ Wblock
            Hp_t_W = Hp_k.T @ Wblock

            row_s = slice(idx_start, idx_end)
            row_p = slice(s * n_states_per_scan, s * n_states_per_scan + 2)

            G_v[row_s, row_s] = G_v[row_s, row_s] + Ht_W @ J_full
            G_v[row_s, row_p] = G_v[row_s, row_p] + Ht_W @ Hp_k
            G_v[row_p, row_s] = G_v[row_p, row_s] + Hp_t_W @ J_full
            G_v[row_p, row_p] = G_v[row_p, row_p] + Hp_t_W @ Hp_k

            RHS[row_s] = RHS[row_s] + Ht_W @ delta_z_k
            RHS[row_p] = RHS[row_p] + Hp_t_W @ delta_z_k

        if np.linalg.cond(G_v.toarray()) > 1e14:
            return corrected_params_group, 0

        delta_x_v = sp.linalg.spsolve(G_v, RHS)
        x_v_current += 0.1 * delta_x_v
        p_g_current = x_v_current[-2:]
        if np.max(np.abs(delta_x_v)) < tol:
            success_correction = 1
            corrected_params_group = p_g_current.copy()
            break

    return corrected_params_group, success_correction
