import numpy as np
import cupy as cp
import scipy.sparse as sp
from opendssdirect import dss
from .mea_fun import measurement_function
from .make_jacobian import jacobian
from .make_jacobian_p import jacobian_line_params
from parse_opendss_file import build_global_y_per_unit

KERSTING_R_PER_MILE = np.array(
    [0.13018, -4.14314E-006, -8.11898E-006, 0.130177, -4.14314E-006,  0.13018]
)
KERSTING_X_PER_MILE = np.array(
    [0.0668195, -0.000224891, -4.2835E-005, 0.066985, -0.000224891, 0.0668195]
)

KERSTING_C_STR = "[566.228 | 0  566.228 | 0  0  566.228]"

# KERSTING_R_PER_MILE = np.array(
#     [0.457552 , 0.155951, 0.153485, 0.466628, 0.158007, 0.461473]
# )
#
# KERSTING_X_PER_MILE = np.array(
#     [1.07805, 0.501679, 0.384938, 10.4818, 0.423653, 10.6507]
# )

# KERSTING_C_STR = "[15.0675  |-4.86254  15.8754  |-1.85328  -3.09107  14.3258  ]"

def to_dense_gpu(mat):
    if sp.issparse(mat):
        mat = mat.toarray()              # NumPy ndarray, dtype unchanged
    else:
        mat = np.asarray(mat, dtype=np.float64, order='C')
    return cp.asarray(mat)               # safe: dtype is float64

def _build_dss_matrix_string(vals):
    return f"[{vals[0]} |{vals[1]} {vals[3]} |{vals[2]} {vals[4]} {vals[5]}]"

def _update_line_params_in_dss(line_name, new_R, new_X):
    temp = f"temp_{line_name}"
    dss.Command(f"New LineCode.{temp} nphases=3 Units=mi "
                f"Rmatrix={_build_dss_matrix_string(new_R)} "
                f"Xmatrix={_build_dss_matrix_string(new_X)} "
                f"Cmatrix={KERSTING_C_STR}")
    dss.Command(f"Edit Line.{line_name} LineCode={temp}")
    dss.Solution.Solve()

def calculate_param_jacobian_for_line(x, Ybus, mpc, busphase_map, line_idx):
    Hp = jacobian_line_params(x, Ybus, mpc, busphase_map)
    c0 = line_idx * 12
    return Hp[:, c0 : c0 + 12]

def correct_parameter_group_multi_scan(
        dss_filename, mpc_with_error, line_to_correct_idx,
        multi_scan_measurements_z, initial_states_multi_scan,
        R_variances_vec, busphase_map,
        max_iter=50, tol=1e-2):

    nnodephase = len(busphase_map)
    scans       = multi_scan_measurements_z.shape[1]

    line_row    = mpc_with_error["line3p"][line_to_correct_idx]
    line_name   = line_row[6]
    miles       = float(line_row[5] / 5280)

    # --- Per-unit line parameters ---
    line_row = mpc_with_error["line3p"][line_to_correct_idx]
    fbus_id = line_row[1]
    line_name = line_row[6]

    # --- Calculate Base Impedance (Z_base) for the line ---
    S_base = mpc_with_error["baseMVA"] * 1e6
    bus_mask = mpc_with_error["bus3p"][:, 0] == fbus_id
    V_base_LL = mpc_with_error["bus3p"][bus_mask, 2][0]
    V_base_LN = (V_base_LL * 1e3) / np.sqrt(3)
    Z_base = (V_base_LN ** 2) / S_base
    print(Z_base)

    # --- GPU copies ---
    p_g = cp.hstack([cp.asarray(KERSTING_R_PER_MILE) * miles,
                     cp.asarray(KERSTING_X_PER_MILE) * miles])

    z_gpu   = cp.asarray(multi_scan_measurements_z)
    x0_gpu  = cp.asarray(initial_states_multi_scan)
    var_inv = 1.0 / cp.asarray(R_variances_vec)          # weight vector

    n_state = 2 * nnodephase - 3
    N       = scans * n_state + 12

    ref_bus = int(np.where(mpc_with_error["bus3p"][:, 1] == 3)[0][0])

    # -------- build initial stacked state vector ----------
    x_states = cp.zeros(scans * n_state, dtype=cp.float64)
    for k in range(scans):
        vm = x0_gpu[:nnodephase, k]
        va = x0_gpu[nnodephase:, k]
        del_idx = ref_bus * 3 + cp.arange(3)
        va_sub  = cp.delete(va, del_idx)
        x_states[k*n_state:(k+1)*n_state] = cp.hstack([vm, va_sub])

    x_v = cp.concatenate([x_states, p_g])

    print(f'GPU correction for line "{line_name}" ({scans} scans)')

    for it in range(max_iter):
        G_dense = cp.zeros((N, N), dtype=cp.float64)
        RHS     = cp.zeros(N,       dtype=cp.float64)

        for k in range(scans):
            # ---------- sync parameters to OpenDSS (CPU) ----------
            p_cpu = cp.asnumpy(p_g)
            dss.Command("Clear")
            dss.Command(f'Redirect "{dss_filename}"')
            _update_line_params_in_dss(line_name,
                                       p_cpu[:6] / miles,
                                       p_cpu[6:] / miles)

            Ybus_cpu, _ = build_global_y_per_unit(mpc_with_error)

            # ---------- reconstruct full state for scan k (GPU) ----------
            s0, s1 = k*n_state, (k+1)*n_state
            x_k    = x_v[s0:s1]

            vm     = x_k[:nnodephase]
            va_sub = x_k[nnodephase:]
            ref_ang= cp.deg2rad(cp.array([0.0, -120.0, 120.0]))
            idx_ref= ref_bus * 3
            va     = cp.concatenate([va_sub[:idx_ref], ref_ang,
                                     va_sub[idx_ref:]])
            x_full_gpu = cp.hstack([vm, va])

            # ---------- CPU helpers ----------
            x_full_cpu = cp.asnumpy(x_full_gpu)
            hx_cpu     = measurement_function(x_full_cpu, Ybus_cpu,
                                              mpc_with_error, busphase_map)
            J_cpu      = jacobian        (x_full_cpu, Ybus_cpu,
                                          mpc_with_error, busphase_map)
            Hp_cpu     = calculate_param_jacobian_for_line(
                            x_full_cpu, Ybus_cpu, mpc_with_error,
                            busphase_map, line_to_correct_idx)

            # ---------- move back to GPU ----------
            hx      = cp.asarray(hx_cpu)
            J_dense = to_dense_gpu(J_cpu)
            Hp = to_dense_gpu(Hp_cpu)

            # delete slack‑angle columns on GPU
            slack_cols = cp.asarray(nnodephase + ref_bus*3 + np.arange(3))
            Hx = cp.delete(J_dense, slack_cols, axis=1)

            dz = z_gpu[:, k] - hx

            Ht_W   = Hx.T * var_inv                      # (n,m)
            Hp_t_W = Hp.T * var_inv

            # block indices
            rs = slice(s0, s1)
            rp = slice(scans*n_state, N)

            G_dense[rs, rs] += Ht_W @ Hx
            G_dense[rs, rp] += Ht_W @ Hp
            G_dense[rp, rs] += Hp_t_W @ Hx
            G_dense[rp, rp] += Hp_t_W @ Hp

            RHS[rs] += Ht_W @ dz
            RHS[rp] += Hp_t_W @ dz

        # ---------- solve ----------
        delta = cp.linalg.solve(G_dense, RHS)         # dense cuSOLVER
        print(x_v[-12:])
        print(delta[-12:])
        x_v[-12:]   += 0.4 * delta[-12:]                          # damping
        p_g    = x_v[-12:]

        max_upd = float(cp.max(cp.abs(delta)).get())
        print(f"    iter {it+1}: max Δ={max_upd:.2e}")

        if max_upd < tol:
            print("    converged")
            return cp.asnumpy(p_g) / miles, 1

    print(" did not converge")
    return cp.asnumpy(p_g) / miles, 0
