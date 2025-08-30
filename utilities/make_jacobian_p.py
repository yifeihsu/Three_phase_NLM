import numpy as np
from scipy.sparse import csr_matrix


def build_param_info(mpc):
    """
    Returns an array of shape (n_params, 3):
        [ line_id, parameter_name, parameter_value ]
    keeping exactly the same ordering as the legacy implementation
    so that downstream indexing (Mahalanobis test, etc.) is unaffected.
    """
    line3p = mpc["line3p"]
    lc     = mpc["lc"]

    # lcid ➜ primitive matrix entries
    lc_dict = {int(row[0]): row[1:13] for row in lc}

    param_names = [
        "Zaa_R", "Zab_R", "Zac_R", "Zbb_R", "Zbc_R", "Zcc_R",
        "Zaa_X", "Zab_X", "Zac_X", "Zbb_X", "Zbc_X", "Zcc_X",
    ]

    info = []
    for line_row in line3p:
        line_id  = int(line_row[-1])
        lcid     = int(line_row[4])
        length_mi = line_row[5] / 5280.0

        if lcid == 0 or lcid not in lc_dict:
            continue

        base_vals = lc_dict[lcid] * length_mi  # per‑phase R/X over full length
        for j, pname in enumerate(param_names):
            info.append((line_id, pname, base_vals[j]))

    return np.asarray(info, dtype=object)


def build_dYsub_symbolic(Ysub, param_type):
    """
    Same utility as in the legacy code – generates ∂Y/∂p for the
    6×6 ‘from‑to’ sub‑matrix of the line under study.
    """
    pmap = {
        "Zaa_R": (0, 0, 1.0), "Zaa_X": (0, 0, 1j),
        "Zab_R": (0, 1, 1.0), "Zab_X": (0, 1, 1j),
        "Zac_R": (0, 2, 1.0), "Zac_X": (0, 2, 1j),
        "Zbb_R": (1, 1, 1.0), "Zbb_X": (1, 1, 1j),
        "Zbc_R": (1, 2, 1.0), "Zbc_X": (1, 2, 1j),
        "Zcc_R": (2, 2, 1.0), "Zcc_X": (2, 2, 1j),
    }

    dZ3 = np.zeros((3, 3), dtype=complex)
    if param_type in pmap:
        i, j, val = pmap[param_type]
        dZ3[i, j] = val
        if i != j:
            dZ3[j, i] = val  # symmetric entry

    # ∂Y = -Y · ∂Z · Y  (Π‑model linearisation)
    Y3   = -Ysub[3:, 0:3]
    dY3  = -Y3 @ (dZ3 @ Y3)

    dY6 = np.zeros((6, 6), dtype=complex)
    dY6[0:3, 0:3] = dY3
    dY6[3:, 3:]   = dY3
    dY6[0:3, 3:]  = -dY3
    dY6[3:, 0:3]  = -dY3
    return dY6

# ---------------------------------------------------------------------------
#  ∂h/∂p for a single parameter (injection rows only)
# ---------------------------------------------------------------------------
def _compute_param_injection_partial(
    x, Ybus, mpc, busphase_map, line_id, param_name
):
    nnode = len(busphase_map)
    m_tot = 3 * nnode                      # new measurement length
    dh    = np.zeros(m_tot, dtype=float)   # will hold ∂h/∂p

    half = nnode
    Vm = x[:half]
    Va = x[half:]
    V  = Vm * np.exp(1j * Va)

    # indices of the six phase nodes affected by this line
    line_data = next(row for row in mpc["line3p"] if int(row[-1]) == line_id)
    fbus = int(line_data[1]) - 1
    tbus = int(line_data[2]) - 1
    f_idx = [fbus * 3 + k for k in range(3)]
    t_idx = [tbus * 3 + k for k in range(3)]
    rowcol = f_idx + t_idx                                # local ➜ global

    # 6×6 sub‑matrix of Ybus for the line
    Ysub = Ybus[np.ix_(rowcol, rowcol)]
    dY6  = build_dYsub_symbolic(Ysub, param_name)

    # ∂S_inj/∂p for each of the six involved node‑phases
    for local_i, g_idx in enumerate(rowcol):
        dI_param = np.sum(dY6[local_i, :] * V[rowcol])
        dSf      = V[g_idx] * np.conjugate(dI_param)
        dh[g_idx]             = dSf.real          # P row
        dh[g_idx + nnode]     = dSf.imag          # Q row
        # |V| rows (offset 2·N) stay zero

    return dh

def jacobian_line_params(x, Ybus, mpc, busphase_map):
    """
    Sparse Jacobian of size (3·N, n_params) wh  ere n_params = 12·n_lines
    (minus any lines skipped by build_param_info).
    """
    nnode = len(busphase_map)
    m_tot = 3 * nnode

    param_info = build_param_info(mpc)
    n_params   = len(param_info)

    data, rows, cols = [], [], []

    for c, (line_id, pname, _) in enumerate(param_info):
        dh_dp = _compute_param_injection_partial(
            x, Ybus, mpc, busphase_map, line_id, pname
        )
        nz = np.nonzero(dh_dp)[0]
        data.extend(dh_dp[nz])
        rows.extend(nz)
        cols.extend([c] * len(nz))

    return csr_matrix((data, (rows, cols)), shape=(m_tot, n_params))

def jacobian_shunt_params(x, mpc, busphase_map):
    nnode = len(busphase_map)
    m_tot = 3 * nnode                 # [P,Q,|V|] for every node‑phase
    n_par = 2 * nnode                 #  G_k  and  B_k

    Vm = x[:nnode]                    # per‑unit magnitudes
    V2 = Vm ** 2                      # |V_k|² used in every column

    data, rows, cols = [], [], []

    for k in range(nnode):
        # column order: [G1,B1, G2,B2, …]
        col_G = 2 * k
        col_B = 2 * k + 1

        # rows for this node‑phase
        rP = k                # P injection
        rQ = k + nnode        # Q injection

        # --------- conductance G_k -----------------
        data.append(V2[k])    # ∂P/∂G_k = |V|²
        rows.append(rP)
        cols.append(col_G)
        # Q‑row derivative is zero → skip

        # --------- susceptance B_k -----------------
        data.append(-V2[k])   # ∂Q/∂B_k = −|V|²
        rows.append(rQ)
        cols.append(col_B)
        # P‑row derivative is zero → skip

    return csr_matrix((data, (rows, cols)), shape=(m_tot, n_par))

#
# def build_param_info_admittance(mpc):
#     """
#     Returns an array of shape (n_params, 3) for admittance parameters:
#         [ line_id, parameter_name, parameter_value ]
#
#     *** CHANGED: The parameters now include all 12 unique terms of the
#     primitive admittance matrix (6 self- and 6 mutual-admittances). ***
#     """
#     line3p = mpc["line3p"]
#     lc = mpc["lc"]
#
#     # lcid ➜ primitive matrix entries (R and X values)
#     lc_dict = {int(row[0]): row[1:13] for row in lc}
#
#     # *** CHANGED: Expanded parameter names to include mutual admittances ***
#     param_names = [
#         "Gaa", "Gab", "Gac", "Gbb", "Gbc", "Gcc",
#         "Baa", "Bab", "Bac", "Bbb", "Bbc", "Bcc"
#     ]
#
#     info = []
#     for line_row in line3p:
#         line_id = int(line_row[-1])
#         lcid = int(line_row[4])
#         length_mi = line_row[5] / 5280.0
#
#         if lcid == 0 or lcid not in lc_dict:
#             continue
#
#         # Get primitive impedance values (Z = R + jX)
#         z_vals_per_mile = lc_dict[lcid]
#
#         # Form the 3x3 primitive impedance matrix Z_prim
#         R = z_vals_per_mile[0:6] * length_mi
#         X = z_vals_per_mile[6:12] * length_mi
#
#         Z3_prim = np.array([
#             [R[0] + 1j * X[0], R[1] + 1j * X[1], R[2] + 1j * X[2]],
#             [R[1] + 1j * X[1], R[3] + 1j * X[3], R[4] + 1j * X[4]],
#             [R[2] + 1j * X[2], R[4] + 1j * X[4], R[5] + 1j * X[5]]
#         ])
#
#         # Calculate primitive admittance matrix Y_prim = inv(Z_prim)
#         try:
#             Y3_prim = np.linalg.inv(Z3_prim)
#         except np.linalg.LinAlgError:
#             print(f"Warning: Singular impedance matrix for line ID {line_id}. Skipping.")
#             continue
#
#         # *** CHANGED: Extract all 12 G and B values ***
#         G = Y3_prim.real
#         B = Y3_prim.imag
#
#         param_values = {
#             "Gaa": G[0, 0], "Gab": G[0, 1], "Gac": G[0, 2],
#             "Gbb": G[1, 1], "Gbc": G[1, 2], "Gcc": G[2, 2],
#             "Baa": B[0, 0], "Bab": B[0, 1], "Bac": B[0, 2],
#             "Bbb": B[1, 1], "Bbc": B[1, 2], "Bcc": B[2, 2]
#         }
#
#         for pname in param_names:
#             info.append((line_id, pname, param_values[pname]))
#
#     return np.asarray(info, dtype=object)
#
#
# def build_dY_line_symbolic(param_type: str):
#     """
#     Generates ∂Y_line/∂p for the 6×6 'from-to' block of the line under study,
#     where p is a self- or mutual-admittance parameter.
#
#     *** CHANGED: Now handles mutual admittance parameters by updating the
#     symmetric entry in the derivative matrix. ***
#     """
#     # Map admittance parameter names to their location and complex value in dY_prim
#     # This map now includes the off-diagonal (mutual) terms.
#     pmap = {
#         "Gaa": (0, 0, 1.0), "Gab": (0, 1, 1.0), "Gac": (0, 2, 1.0),
#         "Gbb": (1, 1, 1.0), "Gbc": (1, 2, 1.0), "Gcc": (2, 2, 1.0),
#         "Baa": (0, 0, 1.0j), "Bab": (0, 1, 1.0j), "Bac": (0, 2, 1.0j),
#         "Bbb": (1, 1, 1.0j), "Bbc": (1, 2, 1.0j), "Bcc": (2, 2, 1.0j)
#     }
#
#     dY3 = np.zeros((3, 3), dtype=complex)
#     if param_type in pmap:
#         i, j, val = pmap[param_type]
#         dY3[i, j] = val
#         # *** CHANGED: If it's a mutual term, update the symmetric element ***
#         if i != j:
#             dY3[j, i] = val
#
#     # The derivative of the line's 6x6 admittance matrix (pi-model) is:
#     # dY_line = [[ dY_prim, -dY_prim],
#     #            [-dY_prim,  dY_prim]]
#     dY6 = np.zeros((6, 6), dtype=complex)
#     dY6[0:3, 0:3] = dY3
#     dY6[3:, 3:] = dY3
#     dY6[0:3, 3:] = -dY3
#     dY6[3:, 0:3] = -dY3
#     return dY6
#
#
# # ---------------------------------------------------------------------------
# #  ∂h/∂p for a single parameter (injection rows only)
# #  NOTE: This function and the main wrapper below required no changes,
# #  as their logic was already general enough.
# # ---------------------------------------------------------------------------
# def _compute_param_injection_partial_admittance(
#         x, mpc, busphase_map, line_id, param_name
# ):
#     nnode = len(busphase_map)
#     m_tot = 3 * nnode
#     dh = np.zeros(m_tot, dtype=float)
#
#     half = nnode
#     Vm = x[:half]
#     Va = x[half:]
#     V = Vm * np.exp(1j * Va)
#
#     line_data = mpc["line3p"][line_id - 1]
#     fbus = int(line_data[1]) - 1
#     tbus = int(line_data[2]) - 1
#     f_idx = [fbus * 3 + k for k in range(3)]
#     t_idx = [tbus * 3 + k for k in range(3)]
#     rowcol = f_idx + t_idx
#
#     dY6 = build_dY_line_symbolic(param_name)
#     dI_line = dY6 @ V[rowcol]
#
#     for local_i, g_idx in enumerate(rowcol):
#         dI_param = dI_line[local_i]
#         dS_param = V[g_idx] * np.conjugate(dI_param)
#
#         dh[g_idx] = dS_param.real
#         dh[g_idx + nnode] = dS_param.imag
#
#     return dh
#
#
# def jacobian_line_params(x, Ybus, mpc, busphase_map):
#     """
#     Sparse Jacobian of size (3·N, n_params) where n_params = 12·n_lines.
#     The parameters are the full set of self- and mutual-admittance
#     terms (G_ij, B_ij).
#     """
#     nnode = len(busphase_map)
#     m_tot = 3 * nnode
#
#     param_info = build_param_info_admittance(mpc)
#     n_params = len(param_info)
#
#     data, rows, cols = [], [], []
#
#     for c, (line_id, pname, _) in enumerate(param_info):
#         dh_dp = _compute_param_injection_partial_admittance(
#             x, mpc, busphase_map, line_id, pname
#         )
#         nz = np.nonzero(dh_dp)[0]
#         data.extend(dh_dp[nz])
#         rows.extend(nz)
#         cols.extend([c] * len(nz))
#
#     return csr_matrix((data, (rows, cols)), shape=(m_tot, n_params))