import numpy as np
import scipy.sparse as sp

import os
import sys
from importlib import util

from .mea_fun import measurement_function
from .make_jacobian import jacobian
from .make_jacobian_p import jacobian_line_params, build_param_info

_342_path = os.path.join(os.path.dirname(__file__), "..", "342")
if _342_path not in sys.path:
    sys.path.append(_342_path)
_spec = util.spec_from_file_location(
    "parse_opendss_file_342", os.path.join(_342_path, "parse_opendss_file.py")
)
_mod = util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_global_y_per_unit = _mod.build_global_y_per_unit

# Impedance matrices for the erroneous LineCode ``Kersting`` (ohm/mile)
KERSTING_R = np.array(
    [0.457552, 0.155951, 0.153485, 0.466628, 0.158007, 0.461473], dtype=float
)
KERSTING_X = np.array(
    [1.07805, 0.501679, 0.384938, 1.04818, 0.423653, 1.06507], dtype=float
)


def _expand_state_vector(x_sub, nb, slack_indices):
    """Expand reduced state vector back to full dimension."""
    n_states = 2 * nb - len(slack_indices)
    assert len(x_sub) == n_states
    angles = np.zeros(nb)
    mags = np.zeros(nb)
    angle_idx = [i for i in range(nb) if i not in slack_indices]
    angles[angle_idx] = x_sub[: nb - len(slack_indices)]
    mags[:] = x_sub[nb - len(slack_indices) :]
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

    # We do not persist these changes to the active OpenDSS model here.
    # ``build_global_y_per_unit`` will therefore rely on this modified copy
    # of ``mpc`` when computing the admittance matrix for each iteration.


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
    """Correct the 6 ``R`` and 6 ``X`` impedance values of a specific line.

    Parameters are adapted from the MATLAB reference implementation but
    tailored for the three-phase distribution ``mpc`` structure.  The line
    index is 0-based.  The routine returns ``(corrected_params, success_flag)``
    where ``corrected_params`` contains the 12 updated impedance parameters in
    the order ``[Raa, Rab, Rac, Rbb, Rbc, Rcc, Xaa, Xab, Xac, Xbb, Xbc, Xcc]``.
    The initial guess for the impedance values comes from the predefined
    LineCode ``Kersting`` rather than the ``mpc`` data.
    """

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

    ref_bus = 0
    for i, row in enumerate(mpc_with_error["bus3p"]):
        if int(row[1]) == 3:
            ref_bus = int(row[0])
            break

    slack_indices = [busphase_map[(ref_bus, ph)] for ph in range(3)]
    n_states_per_scan = 2 * nb - len(slack_indices)
    N = s * n_states_per_scan + 12

    x_states_all = np.zeros(s * n_states_per_scan)
    for k in range(s):
        v_init = initial_states_multi_scan[:nb, k]
        a_deg = initial_states_multi_scan[nb:, k]
        a_rad = np.radians(a_deg)
        a_sub = np.delete(a_rad, slack_indices)
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
            theta_full, V_full = _expand_state_vector(x_scan_k, nb, slack_indices)
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
            angle_cols = [nb + idx for idx in slack_indices]
            J_full = np.delete(J_full, angle_cols, axis=1)
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
