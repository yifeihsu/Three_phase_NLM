import numpy as np
import scipy.sparse as sp
from opendssdirect import dss

from .mea_fun import measurement_function
from .make_jacobian import jacobian
from .make_jacobian_p import jacobian_line_params
from parse_opendss_file import build_global_y_per_unit

# Per-mile impedance matrices for the erroneous LineCode ``Kersting``
# KERSTING_R_PER_MILE = np.array(
#     [0.457552, 0.155951, 0.153485, 0.466628, 0.158007, 0.461473], dtype=float
# )
# KERSTING_X_PER_MILE = np.array(
#     [10.7805, 0.501679, 0.384938, 10.4818, 0.423653, 1.06507], dtype=float
# )


### For 342 case
KERSTING_R_PER_MILE = np.array(
    [0.13018, -4.14314E-006, -8.11898E-006, 0.130177, -4.14314E-006, 0.13018], dtype=float
)

KERSTING_X_PER_MILE = np.array(
    [0.0668195, -0.000224891, -4.2835E-005, 2.8703539, -0.000224891, 6.13821694], dtype=float
)
# KERSTING_X_PER_MILE = np.array(
#     [0.0668195, -0.000224891, -4.2835E-005, 0.066985, -0.000224891, 0.0668195], dtype=float
# )
# KERSTING_C_STR = "[15.0675 |-4.86254 15.8754 |-1.85328 -3.09107 14.3258]"
KERSTING_C_STR = "[566.228 | 0  566.228 | 0  0  566.228]"

def _build_dss_matrix_string(vals):
    """Builds a matrix string for OpenDSS from a vector of 6 lower-triangle values."""
    return f"[{vals[0]} |{vals[1]} {vals[3]} |{vals[2]} {vals[4]} {vals[5]}]"


def _update_line_params_in_dss(line_name, new_R_per_mile, new_X_per_mile):
    """
    Creates a temporary LineCode in OpenDSS with the given per-mile R and X
    parameters and assigns it to the specified line.
    """
    temp_linecode = f"temp_{line_name}"
    r_matrix_str = _build_dss_matrix_string(new_R_per_mile)
    x_matrix_str = _build_dss_matrix_string(new_X_per_mile)

    dss.Command(f"New LineCode.{temp_linecode} nphases=3 Units=mi "
                f"Rmatrix={r_matrix_str} Xmatrix={x_matrix_str} Cmatrix={KERSTING_C_STR}")

    dss.Command(f"Edit Line.{line_name} LineCode={temp_linecode}")
    dss.Solution.Solve()


def calculate_param_jacobian_for_line(x, Ybus, mpc, busphase_map, line_idx):
    """Return Jacobian of measurements w.r.t. the 12 total impedance parameters of a line."""
    Hp_full = jacobian_line_params(x, Ybus, mpc, busphase_map)
    start_idx = line_idx * 12
    cols = list(range(start_idx, start_idx + 12))
    return Hp_full[:, cols]


def correct_parameter_group_multi_scan(
        dss_filename,
        mpc_with_error,
        line_to_correct_idx,
        multi_scan_measurements_z,
        initial_states_multi_scan,
        R_variances_vec,
        busphase_map,
        max_iter=50,
        tol=1e-2,
):
    """
    Corrects the 6 R and 6 X impedance values (per mile) of a specific line by estimating
    the total line parameters and then converting back.
    """
    nnodephase = len(busphase_map)
    s = multi_scan_measurements_z.shape[1]

    line_row = mpc_with_error["line3p"][line_to_correct_idx]
    line_name = line_row[6]
    line_length_miles = float(line_row[5]/5280)  # Convert feet to miles

    p_g_current = np.hstack([
        KERSTING_R_PER_MILE * line_length_miles,
        KERSTING_X_PER_MILE * line_length_miles
    ])

    W_single = sp.diags(1.0 / R_variances_vec)
    n_states_per_scan = 2 * nnodephase - 3
    N = s * n_states_per_scan + 12

    ref_bus_idx = np.where(mpc_with_error["bus3p"][:, 1] == 3)[0][0]

    x_states_all = np.zeros(s * n_states_per_scan)
    for k in range(s):
        vm_init = initial_states_multi_scan[:nnodephase, k]
        va_init = initial_states_multi_scan[nnodephase:, k]
        va_sub = np.delete(va_init, ref_bus_idx * 3 + np.arange(3))
        x_states_all[k * n_states_per_scan:(k + 1) * n_states_per_scan] = np.hstack([vm_init, va_sub])

    x_v_current = np.concatenate([x_states_all, p_g_current])

    print(f'    Starting iterative correction for line "{line_name}" with {s} scans...')
    for it in range(max_iter):
        G_v = sp.lil_matrix((N, N))
        RHS = np.zeros(N)

        for k in range(s):
            dss.Command("Clear")
            dss.Command(f'Redirect "{dss_filename}"')
            _update_line_params_in_dss(line_name, p_g_current[:6] / line_length_miles,
                                       p_g_current[6:] / line_length_miles)
            Ybus_iter, _ = build_global_y_per_unit(mpc_with_error)

            idx_start = k * n_states_per_scan
            # *** CORRECTED: Definition of idx_end was missing ***
            idx_end = (k + 1) * n_states_per_scan
            x_scan_k = x_v_current[idx_start:idx_end]

            vm_full = x_scan_k[:nnodephase]
            va_sub = x_scan_k[nnodephase:]
            ref_angles = np.deg2rad([0, -120, 120])
            va_full = np.insert(va_sub, ref_bus_idx * 3, ref_angles)
            x_full = np.hstack([vm_full, va_full])

            hx_k = measurement_function(x_full, Ybus_iter, mpc_with_error, busphase_map)
            delta_z_k = multi_scan_measurements_z[:, k] - hx_k

            J_full = jacobian(x_full, Ybus_iter, mpc_with_error, busphase_map)
            slack_angle_cols = nnodephase + ref_bus_idx * 3 + np.arange(3)
            H_x_k = np.delete(J_full.toarray(), slack_angle_cols, axis=1)

            Hp_k_total = calculate_param_jacobian_for_line(x_full, Ybus_iter, mpc_with_error, busphase_map,
                                                           line_to_correct_idx)

            Ht_W = H_x_k.T @ W_single
            Hp_t_W = Hp_k_total.T @ W_single

            row_s = slice(idx_start, idx_end)
            row_p = slice(s * n_states_per_scan, N)

            G_v[row_s, row_s] += Ht_W @ H_x_k
            G_v[row_s, row_p] += Ht_W @ Hp_k_total
            G_v[row_p, row_s] += Hp_t_W @ H_x_k
            G_v[row_p, row_p] += Hp_t_W @ Hp_k_total

            RHS[row_s] += Ht_W @ delta_z_k
            RHS[row_p] += Hp_t_W @ delta_z_k

        try:
            delta_x_v = sp.linalg.spsolve(G_v.tocsr(), RHS)
        except (np.linalg.LinAlgError, sp.linalg.MatrixRankWarning):
            print(f"    G_v is ill-conditioned at iteration {it}. Aborting correction.")
            return p_g_current, 0

        x_v_current += 0.01 * delta_x_v
        # Print the last 12 values difference (the parameters being corrected)
        print(delta_x_v[-12:])

        p_g_current = x_v_current[-12:]

        max_update = np.max(np.abs(delta_x_v))
        R_per_mile_est = p_g_current[0] / line_length_miles
        X_per_mile_est = p_g_current[6] / line_length_miles
        print(
            f'    Iter {it + 1}: max update={max_update:.2e}, R_per_mile_est={R_per_mile_est:.4f}, X_per_mile_est={X_per_mile_est:.4f}')

        if max_update < tol:
            print(f'    Correction converged in {it + 1} iterations.')
            return p_g_current / line_length_miles, 1

    print(f'    Correction did NOT converge after {max_iter} iterations.')
    return p_g_current / line_length_miles, 0