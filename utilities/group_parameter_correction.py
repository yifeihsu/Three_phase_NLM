import numpy as np
import scipy.sparse as sp

from .mea_fun import measurement_function
from .make_jacobian import jacobian
from .make_jacobian_p import jacobian_line_params, build_param_info
from parse_opendss_file import build_global_y_per_unit

# Impedance matrices for the erroneous LineCode ``Kersting`` (ohm/mile)
KERSTING_R = np.array(
    [0.457552, 0.155951, 0.153485, 0.466628, 0.158007, 0.461473], dtype=float
)
KERSTING_X = np.array(
    [1.07805, 0.501679, 0.384938, 1.04818, 0.423653, 1.06507], dtype=float
)



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


def calculate_param_jacobian_for_line(x, Ybus, mpc, busphase_map, line_idx):
    """Return Jacobian of measurements w.r.t. the 12 impedance parameters of a line."""
    Hp_full = jacobian_line_params(x, Ybus, mpc, busphase_map)
    param_info = build_param_info(mpc)
    start_idx = line_idx * 12
    cols = list(range(start_idx, start_idx + 12))
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

    nb = len(busphase_map)
    s = multi_scan_measurements_z.shape[1]

    success_correction = 0

    # extract current parameter guess from the mpc structure

    # start from the erroneous linecode parameters directly
    R_init_vec = KERSTING_R.copy()
    X_init_vec = KERSTING_X.copy()
    p_g_current = np.hstack([R_init_vec, X_init_vec])
    corrected_params_group = p_g_current.copy()

    nz_single = multi_scan_measurements_z.shape[0]
    R_single = sp.diags(R_variances_vec, 0, shape=(nz_single, nz_single))
    R_inv_single = sp.linalg.inv(R_single)

    n_states_per_scan = 2 * nb - 1
    N = s * n_states_per_scan + 12


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
            _update_line_params_in_mpc(
                mpc_iter,
                line_to_correct_idx,
                p_g_current[:6],
                p_g_current[6:12],
            )
            Ybus_iter, _ = build_global_y_per_unit(mpc_iter)
            x_full = np.hstack([V_full, theta_full])
            hx_k = measurement_function(x_full, Ybus_iter, mpc_iter, busphase_map)
            z_k = multi_scan_measurements_z[:, k]
            delta_z_k = z_k - hx_k
            J_full = jacobian(x_full, Ybus_iter, mpc_iter, busphase_map)
            J_full = np.delete(J_full, ref_bus, axis=1)
            J_full = np.delete(J_full, ref_bus, axis=1)
            Hp_k = calculate_param_jacobian_for_line(x_full, Ybus_iter, mpc_iter, busphase_map, line_to_correct_idx)

            Wblock = R_inv_single
            Ht_W = J_full.T @ Wblock
            Hp_t_W = Hp_k.T @ Wblock

            row_s = slice(idx_start, idx_end)
            row_p = slice(s * n_states_per_scan, s * n_states_per_scan + 12)

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
        p_g_current = x_v_current[-12:]
        if np.max(np.abs(delta_x_v)) < tol:
            success_correction = 1
            corrected_params_group = p_g_current.copy()
            break

    return corrected_params_group, success_correction
