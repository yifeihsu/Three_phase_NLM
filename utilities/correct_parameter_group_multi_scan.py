import numpy as np
import cupy as cp
import scipy.sparse as sp
from opendssdirect import dss
from .mea_fun import measurement_function
from .make_jacobian import jacobian
from .make_jacobian_p import jacobian_line_params
from parse_opendss_file import build_global_y_per_unit

# --- Constants (unchanged) ---
KERSTING_R_PER_MILE = np.array(
    [0.13018, -4.14314E-006, -8.11898E-006, 0.130177, -4.14314E-006, 0.13018]
)
# The user's manipulated X matrix
KERSTING_X_PER_MILE = np.array(
    [0.0668195, -0.000224891, -4.2835E-005, 0.066985, -0.000224891, 0.0668195]
)
KERSTING_C_STR = "[566.228 | 0  566.228 | 0  0  566.228]"


# --- Helper Functions (unchanged) ---
def to_dense_gpu(mat):
    if sp.issparse(mat):
        mat = mat.toarray()
    else:
        mat = np.asarray(mat, dtype=np.float64, order='C')
    return cp.asarray(mat)


def _build_dss_matrix_string(vals):
    # Limit the digits to 6 decimal places for DSS compatibility
    vals = [f"{v:.6f}" for v in vals]
    return f"[{vals[0]} |{vals[1]} {vals[3]} |{vals[2]} {vals[4]} {vals[5]}]"


def _update_line_params_in_dss(dss_filename, line_name, new_R, new_X, mpc):
    """ Helper to update DSS and return the new Ybus """
    temp = f"temp_{line_name}"
    # It's safer to re-redirect the file to ensure a clean state
    dss.Command("Clear")
    dss.Command(f'Redirect "{dss_filename}"')
    dss.Command("New LineCode.NLM nphases=3 units=mi")
    dss.Command(f"~ Rmatrix = {_build_dss_matrix_string(new_R)}")
    dss.Command(f"~ Xmatrix = {_build_dss_matrix_string(new_X)}")
    dss.Command("~ Cmatrix = [566.228 | 0  566.228 | 0  0  566.228]")
    dss.Command("Edit line.14 Linecode=NLM")
    dss.Solution.Solve()
    Ybus, _ = build_global_y_per_unit(mpc)
    return Ybus


def calculate_param_jacobian_for_line(x, Ybus, mpc, busphase_map, line_idx):
    """ Your original analytical jacobian function (unchanged). """
    Hp = jacobian_line_params(x, Ybus, mpc, busphase_map)
    c0 = line_idx * 12
    return Hp[:, c0: c0 + 12]


def correct_parameters_with_known_states(
        dss_filename, mpc_with_error, line_to_correct_idx,
        multi_scan_measurements_z, true_states_multi_scan,
        R_variances_vec, busphase_map,
        max_iter=60, tol=1e-4, lambda_reg=0.1):
    """
    Highly efficient parameter correction that assumes system states are known and fixed.
    This version now uses Gain Matrix Normalization to handle severe ill-conditioning.
    """
    scans = multi_scan_measurements_z.shape[1]
    line_row = mpc_with_error["line3p"][line_to_correct_idx]
    line_name = line_row[6]
    miles = float(line_row[5] / 5280)

    p_g = cp.hstack([cp.asarray(KERSTING_R_PER_MILE) * miles,
                     cp.asarray(KERSTING_X_PER_MILE) * miles])

    # Convert the p_g values to per-unit (p.u.) for the correction
    baseMVA = mpc_with_error["baseMVA"]
    V_nom = float(line_row[4])
    Z_base = V_nom ** 2 / baseMVA
    # Print Z_base for debugging
    print(f"Z_base for line {line_name}: {Z_base:.6f}")
    # Convert p_g to per-unit
    p_g = p_g / Z_base

    z_gpu = cp.asarray(multi_scan_measurements_z)
    x_true_gpu = cp.asarray(true_states_multi_scan)
    var_inv_full = 1.0 / cp.asarray(R_variances_vec)
    print(f'GPU correction for line "{line_name}" ({scans} scans) with KNOWN states [USING MATRIX NORMALIZATION]')

    for it in range(max_iter):
        G_p = cp.zeros((12, 12), dtype=cp.float64)
        RHS_p = cp.zeros(12, dtype=cp.float64)
        p_cpu = cp.asnumpy(p_g)

        Ybus_cpu = _update_line_params_in_dss(dss_filename, line_name, p_cpu[:6] / miles * Z_base, p_cpu[6:] / miles * Z_base,
                                              mpc_with_error)

        for k in range(scans):
            x_true_k_cpu = cp.asnumpy(x_true_gpu[:, k])

            hx_cpu = measurement_function(x_true_k_cpu, Ybus_cpu, mpc_with_error, busphase_map)
            Hp_cpu = calculate_param_jacobian_for_line(x_true_k_cpu, Ybus_cpu, mpc_with_error, busphase_map,
                                                       line_to_correct_idx)

            hx = cp.asarray(hx_cpu)
            Hp_full = to_dense_gpu(Hp_cpu)

            dz_full = z_gpu[:, k] - hx

            # 1. Measurement Selection: Find measurements sensitive to parameters
            sensitivity_mask = cp.sum(cp.abs(Hp_full), axis=1) > 1e-6

            # 2. Filter the Jacobian, residuals, and weights
            Hp_filtered = Hp_full[sensitivity_mask, :]
            dz_filtered = dz_full[sensitivity_mask]
            var_inv_filtered = var_inv_full[sensitivity_mask]

            # 3. Build the Normal Equations using only filtered measurements
            Hp_t_W = Hp_filtered.T * var_inv_filtered
            G_p += Hp_t_W @ Hp_filtered
            RHS_p += Hp_t_W @ dz_filtered
        # --- GAIN MATRIX NORMALIZATION ---
        # 1. Get the diagonal elements
        diag_G = G_p.diagonal()
        # Prevent division by zero if any diagonal is zero
        diag_G[cp.abs(diag_G) < 1e-12] = 1.0
        # 2. Create the normalization matrix S = diag(1 / sqrt(diag(G)))
        S = cp.diag(1.0 / cp.sqrt(diag_G))
        # 3. Normalize the matrix and RHS vector
        G_p_norm = S @ G_p @ S
        RHS_p_norm = S @ RHS_p

        cond_num_norm = cp.linalg.norm(G_p_norm) * cp.linalg.norm(cp.linalg.inv(G_p_norm))
        print(f"    Condition Number of NORMALIZED G_p: {cond_num_norm:.2e}")

        # 4. Apply regularization to the NORMALIZED matrix
        G_p_reg = G_p_norm + lambda_reg * cp.eye(12)

        # 5. Solve the stable, normalized system
        delta_p_norm = cp.linalg.solve(G_p_reg, RHS_p_norm)

        # 6. Un-scale the result to get the actual parameter update
        delta_p = S @ delta_p_norm
        # --- END NORMALIZATION ---

        print(f"Δp={cp.asnumpy(delta_p)}")
        # Do not use scientific notation for small values, print out the p_g before and after update
        print("    p_g before update:", np.array2string(cp.asnumpy(p_g), precision=6, suppress_small=True))
        indices = [0, 3, 5, 6, 9, 11]  # elements 1,4,6,7,10,12 in 1-based indexing
        if it <= 10:
            damping_factor = 0.5
        elif it <= 20:
            damping_factor = 2
        else:
            damping_factor = 10
        p_g[indices] += damping_factor * delta_p[indices]
        # p_g += damping_factor * delta_p
        print("    p_g after update:", np.array2string(cp.asnumpy(p_g), precision=6, suppress_small=True))

        max_upd = float(cp.max(cp.abs(delta_p)).get())
        print(f"    iter {it + 1}: max Δp={max_upd:.2e}")

        if max_upd < tol:
            print("    converged")
            return cp.asnumpy(p_g) / miles, 1

    print(" did not converge")
    return cp.asnumpy(p_g) / miles, 0


# --- ORIGINAL Function (Unchanged) ---
def correct_parameter_group_multi_scan(
        dss_filename, mpc_with_error, line_to_correct_idx,
        multi_scan_measurements_z, initial_states_multi_scan,
        R_variances_vec, busphase_map,
        max_iter=50, tol=1e-2):
    # ... (Your original code remains here, untouched) ...
    nnodephase = len(busphase_map)
    scans = multi_scan_measurements_z.shape[1]

    line_row = mpc_with_error["line3p"][line_to_correct_idx]
    line_name = line_row[6]
    miles = float(line_row[5] / 5280)

    p_g = cp.hstack([cp.asarray(KERSTING_R_PER_MILE) * miles,
                     cp.asarray(KERSTING_X_PER_MILE) * miles])

    z_gpu = cp.asarray(multi_scan_measurements_z)
    x0_gpu = cp.asarray(initial_states_multi_scan)
    var_inv = 1.0 / cp.asarray(R_variances_vec)

    n_state = 2 * nnodephase - 3
    N = scans * n_state + 12

    ref_bus = int(np.where(mpc_with_error["bus3p"][:, 1] == 3)[0][0])

    x_states = cp.zeros(scans * n_state, dtype=cp.float64)
    for k in range(scans):
        vm = x0_gpu[:nnodephase, k]
        va = x0_gpu[nnodephase:, k]
        del_idx = ref_bus * 3 + cp.arange(3)
        va_sub = cp.delete(va, del_idx)
        x_states[k * n_state:(k + 1) * n_state] = cp.hstack([vm, va_sub])

    x_v = cp.concatenate([x_states, p_g])

    print(f'GPU correction for line "{line_name}" ({scans} scans)')

    for it in range(max_iter):
        G_dense = cp.zeros((N, N), dtype=cp.float64)
        RHS = cp.zeros(N, dtype=cp.float64)

        for k in range(scans):
            p_cpu = cp.asnumpy(p_g)
            dss.Command("Clear")
            dss.Command(f'Redirect "{dss_filename}"')
            _update_line_params_in_dss(dss_filename, line_name,
                                       p_cpu[:6] / miles,
                                       p_cpu[6:] / miles, mpc_with_error)

            Ybus_cpu, _ = build_global_y_per_unit(mpc_with_error)

            s0, s1 = k * n_state, (k + 1) * n_state
            x_k = x_v[s0:s1]

            vm = x_k[:nnodephase]
            va_sub = x_k[nnodephase:]
            ref_ang = cp.deg2rad(cp.array([0.0, -120.0, 120.0]))
            idx_ref = ref_bus * 3
            va = cp.concatenate([va_sub[:idx_ref], ref_ang,
                                 va_sub[idx_ref:]])
            x_full_gpu = cp.hstack([vm, va])

            x_full_cpu = cp.asnumpy(x_full_gpu)
            hx_cpu = measurement_function(x_full_cpu, Ybus_cpu,
                                          mpc_with_error, busphase_map)
            J_cpu = jacobian(x_full_cpu, Ybus_cpu,
                             mpc_with_error, busphase_map)
            Hp_cpu = calculate_param_jacobian_for_line(
                x_full_cpu, Ybus_cpu, mpc_with_error,
                busphase_map, line_to_correct_idx)

            hx = cp.asarray(hx_cpu)
            J_dense = to_dense_gpu(J_cpu)
            Hp = to_dense_gpu(Hp_cpu)

            slack_cols = cp.asarray(nnodephase + ref_bus * 3 + np.arange(3))
            Hx = cp.delete(J_dense, slack_cols, axis=1)

            dz = z_gpu[:, k] - hx

            Ht_W = Hx.T * var_inv
            Hp_t_W = Hp.T * var_inv

            rs = slice(s0, s1)
            rp = slice(scans * n_state, N)

            G_dense[rs, rs] += Ht_W @ Hx
            G_dense[rs, rp] += Ht_W @ Hp
            G_dense[rp, rs] += Hp_t_W @ Hx
            G_dense[rp, rp] += Hp_t_W @ Hp

            RHS[rs] += Ht_W @ dz
            RHS[rp] += Hp_t_W @ dz

        delta = cp.linalg.solve(G_dense, RHS)

        x_v[-12:] += 0.4 * delta[-12:]
        p_g = x_v[-12:]

        max_upd = float(cp.max(cp.abs(delta)).get())
        print(f"    iter {it + 1}: max Δ={max_upd:.2e}")

        if max_upd < tol:
            print("    converged")
            return cp.asnumpy(p_g) / miles, 1

    print(" did not converge")
    return cp.asnumpy(p_g) / miles, 0