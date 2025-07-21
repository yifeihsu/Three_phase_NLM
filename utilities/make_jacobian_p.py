import numpy as np
from scipy.sparse import csr_matrix

def build_param_info(mpc):
    line3p = mpc["line3p"]
    lc = mpc["lc"]

    # Create a dictionary for quick lookup: lcid -> [R11, R21, ..., X33]
    lc_dict = {int(row[0]): row[1:13] for row in lc}

    n_lines = len(line3p)
    param_info = np.empty((n_lines * 12, 3), dtype=object)
    param_names = ['Zaa_R', 'Zab_R', 'Zac_R', 'Zbb_R', 'Zbc_R', 'Zcc_R',
                   'Zaa_X', 'Zab_X', 'Zac_X', 'Zbb_X', 'Zbc_X', 'Zcc_X']

    idx = 0
    for line_row in line3p:
        line_id = int(line_row[-1])
        lcid = int(line_row[4])
        length_mi = line_row[5] / 5280

        if lcid == 0 or lcid not in lc_dict:
            continue

        dat = lc_dict[lcid]  # [R11, R21, ..., X33]
        values = dat * length_mi
        for j, param_name in enumerate(param_names):
            param_info[idx + j] = (line_id, param_name, values[j])
        idx += 12

    return param_info[:idx]

def jacobian_line_params(x, Ybus, mpc, busphase_map):
    nnodephase = len(busphase_map)
    n_lines = len(mpc["line3p"])
    # Total number of measurements (assume full set)
    m_inj = 2 * nnodephase
    m_flow = 4 * (3 * n_lines)
    m_v = nnodephase
    m_total = m_inj + m_flow + m_v

    param_info = build_param_info(mpc)
    n_params = len(param_info) # Number of parameters
    # Use sparse matrix components
    data = []
    row_indices = []
    col_indices = []

    for c, (line_id, p_name, _) in enumerate(param_info):
        # Calculate the columm of the Jacobian matrix
        dh_dp = compute_param_flow_injection_partial(x, Ybus, mpc, busphase_map, line_id, p_name)
        # Only keep non-zero elements
        nonzero_idx = np.nonzero(dh_dp)[0]
        data.extend(dh_dp[nonzero_idx])
        row_indices.extend(nonzero_idx)
        col_indices.extend([c] * len(nonzero_idx))

    Hparam = csr_matrix((data, (row_indices, col_indices)), shape=(m_total, n_params))
    return Hparam


def compute_param_flow_injection_partial(x, Ybus, mpc, busphase_map, line_id, param_type):
    nnodephase = len(busphase_map)
    n_lines = len(mpc["line3p"])
    m_inj = 2 * nnodephase
    m_flow = 4 * (3 * n_lines)
    m_v = nnodephase
    m_total = m_inj + m_flow + m_v

    dh = np.zeros(m_total, dtype=float)
    half = nnodephase
    Vm = x[:half]
    Va = x[half:]
    V = Vm * np.exp(1j * Va)

    line_data = mpc["line3p"][line_id - 1]
    fbus = int(line_data[1]) - 1
    tbus = int(line_data[2]) - 1
    f_idx = [fbus * 3, fbus * 3 + 1, fbus * 3 + 2]
    t_idx = [tbus * 3, tbus * 3 + 1, tbus * 3 + 2]
    rowcol = f_idx + t_idx

    Ysub = Ybus[np.ix_(rowcol, rowcol)]
    dY6 = build_dYsub_symbolic(Ysub, param_type)

    # Bus injections
    for local_i in rowcol:
        local_idx = rowcol.index(local_i)
        dI_param = sum(dY6[local_idx, j] * V[rowcol[j]] for j in range(6))
        dSf = V[local_i] * np.conjugate(dI_param)
        dh[local_i] = dSf.real
        dh[local_i + nnodephase] = dSf.imag

    # Line flows
    offset_flow = 2 * nnodephase
    for alpha in range(3):
        dIf = sum(-dY6[alpha, 3 + xph] * (V[f_idx[xph]] - V[t_idx[xph]]) for xph in range(3))
        dSf = V[f_idx[alpha]] * np.conjugate(dIf)
        dIt = sum(-dY6[3 + alpha, xph] * (V[t_idx[xph]] - V[f_idx[xph]]) for xph in range(3))
        dSt = V[t_idx[alpha]] * np.conjugate(dIt)

        row_base = offset_flow + (line_id - 1) * 3 + alpha
        dh[row_base] = dSf.real
        dh[row_base + n_lines * 3] = dSt.real
        dh[row_base + n_lines * 6] = dSf.imag
        dh[row_base + n_lines * 9] = dSt.imag

    return dh

# def partial_p_bus_injection(node_i, param_type, line_id, V, Ybus, f_idx, t_idx, rowcol):
#     """
#     Symbolic partial derivative of (P_inj(node_i), Q_inj(node_i)) wrt param p
#     of line line_id in the upper-triangle 3x3 impedance.
#
#     If node_i not in rowcol => partial is 0.
#     Else we do:
#       P_inj(i) = real( V[i]* conj( I[i]) ),
#       I[i] = sum_j Ybus(i,j)*V[j].
#
#     partial wrt param =>
#       partial I[i] = sum_j partial(Ybus(i,j)) wrt param * V[j],
#       partial P_inj(i) = real( V[i]* conj( partial I[i] ) ),
#       partial Q_inj(i) = imag( V[i]* conj( partial I[i] ) ).
#
#     For demonstration, we only update the 6×6 sub-block of Ybus associated w/ line_id's from/to phases.
#     """
#     # Return (dPdp, dQdp)
#     dPdp = 0.0
#     dQdp = 0.0
#
#     # 1) Check if node_i is in rowcol => if not => partial=0
#     if node_i not in rowcol:
#         return (0.0, 0.0)
#
#     # 2) Build partial dYsub wrt param p (the 6×6 sub-block for from->to)
#     # rowcol = f_idx + t_idx => e.g. [fA,fB,fC, tA,tB,tC] => 6 nodes
#     # extract Ysub => shape(6,6)
#     Ysub = Ybus[np.ix_(rowcol, rowcol)]
#     # build dZsub => 3×3 for the line's primitive => embed in 6×6 => dZ6x6
#     dY6 = build_dYsub_symbolic(Ysub, param_type)  # shape(6,6) partial wrt param
#     # If your parameter doesn't affect diagonal => it might be 0 in some places
#
#     # 3) partial of I[node_i] wrt param => sum_j dYbus(node_i, j)*V[j]
#     #   but node_i local => local_i = rowcol.index(node_i)
#     local_i = rowcol.index(node_i)
#     # partial of Ybus(node_i, j) => dY6[ local_i, local_j ]
#     # I[node_i] = sum_j Ybus(node_i,j)*V[j], j in [0..N-1]
#     # but only j in rowcol matter => partial I => sum_{local_j=0..5} dY6(local_i, local_j)* V[rowcol[local_j]]
#     dI_param = 0+0j
#     for local_j in range(6):
#         j_global = rowcol[local_j]
#         dI_param += dY6[local_i, local_j] * V[j_global]
#
#     # 4) partial P_inj => real( V[node_i]* conj( dI_param ) )
#     # partial Q_inj => imag( V[node_i]* conj( dI_param ) )
#     Vi = V[node_i]
#     dSf = Vi * np.conjugate(dI_param)   # complex partial of S_inj
#     dPdp = dSf.real
#     dQdp = dSf.imag
#
#     return (dPdp, dQdp)
#
#
# def partial_p_line_flow(alpha, param_type, line_id, V, Ybus, f_idx, t_idx, rowcol):
#     """
#     Symbolic partial of Pf,Pt,Qf,Qt for the line line_id, phase alpha, wrt param p.
#
#     We do:
#       Pf = real( Vf(alpha)* conj(Ifrom(alpha)) ),
#       Ifrom(alpha) = sum_{xph} Ysub(alpha,xph)*(Vf[xph]) - ...
#     partial => partial Ifrom wrt param => ...
#     Similar for Pt => Vt(alpha)* conj(Ito(alpha)).
#     """
#     dPf = 0.0
#     dPt = 0.0
#     dQf = 0.0
#     dQt = 0.0
#
#     # 1) build dY6 = partial of 6×6 sub-block wrt param
#     Ysub = Ybus[np.ix_(rowcol, rowcol)]
#     dY6 = build_dYsub_symbolic(Ysub, param_type)
#
#     # (a) partial Pf => real( Vf_alpha * conj( Ifrom ) )
#     # => partial => real( Vf_alpha * conj( partial Ifrom ) ), ignoring partial of Vf_alpha wrt param
#     # similarly for Qf, Pt, Qt
#     # We'll define small helper:
#     dPf, dQf = partial_flow_from_side(alpha, dY6, V, f_idx, t_idx, param_type)
#     dPt, dQt = partial_flow_to_side(alpha, dY6, V, f_idx, t_idx, param_type)
#     return (dPf, dPt, dQf, dQt)


def build_dYsub_symbolic(Ysub, param_type):
    param_map = {
        'Zaa_R': (0, 0, 1.0), 'Zaa_X': (0, 0, 1j),
        'Zab_R': (0, 1, 1.0), 'Zab_X': (0, 1, 1j),
        'Zac_R': (0, 2, 1.0), 'Zac_X': (0, 2, 1j),
        'Zbb_R': (1, 1, 1.0), 'Zbb_X': (1, 1, 1j),
        'Zbc_R': (1, 2, 1.0), 'Zbc_X': (1, 2, 1j),
        'Zcc_R': (2, 2, 1.0), 'Zcc_X': (2, 2, 1j)
    }

    dZ3 = np.zeros((3, 3), dtype=complex)
    if param_type in param_map:
        i, j, val = param_map[param_type]
        dZ3[i, j] = val
        if i != j:
            dZ3[j, i] = val  # Symmetry

    Y3 = -Ysub[3:, 0:3]
    dY3 = -Y3 @ (dZ3 @ Y3)

    dY6 = np.zeros((6, 6), dtype=complex)
    dY6[0:3, 0:3] = dY3
    dY6[3:, 3:] = dY3
    dY6[0:3, 3:] = -dY3
    dY6[3:, 0:3] = -dY3

    return dY6

# def partial_flow_from_side(alpha, dY6, V, f_idx, t_idx, param_type):
#     """
#     partial (Pf, Qf) wrt param => 2 real numbers
#     Pf = real( Vf(alpha)* conj(Ifrom(alpha)) ),
#     Ifrom(alpha) = sum_{xph} Ysub(alpha,xph)*V[f_idx[xph]] - ...
#     => partial Ifrom = sum_j dY6(alpha, j)* V[rowcol[j]] ...
#     """
#     # compute partial Ifrom wrt param
#     # row=alpha => 0..2, col => 0..5
#     dIf = 0+0j
#     for xph in range(3):
#         # from-phase => col xph
#         iFx = f_idx[xph]
#         iTx = t_idx[xph]
#         dIf += -dY6[alpha, 3+xph]* (V[iFx] - V[iTx])
#     # partial Sf => V[f_idx[alpha]] * conj( dIf ), ignoring partial V wrt param
#     Vf_alpha = V[f_idx[alpha]]
#     dSf = Vf_alpha * np.conjugate(dIf)
#     dPf = dSf.real
#     dQf = dSf.imag
#     return (dPf, dQf)
#
# def partial_flow_to_side(alpha, dY6, V, f_idx, t_idx, param_type):
#     """
#     partial (Pt, Qt).
#     Pt= real( Vt(alpha)* conj(Ito(alpha)) ).
#     row= 3+alpha => etc.
#     """
#     dIt = 0+0j
#     for xph in range(3):
#         # row=3+alpha, col => xph => plus? minus?
#         dIt += -dY6[3+alpha, xph] * (V[t_idx[xph]] - V[f_idx[xph]])
#         # etc. depends on sign
#         # dIt += - dY6[3+alpha, 3+xph]* V[t_idx[xph]]
#     Vt_alpha = V[t_idx[alpha]]
#     dSt = Vt_alpha * np.conjugate(dIt)
#     dPt = dSt.real
#     dQt = dSt.imag
#     return (dPt, dQt)
