import numpy as np

def build_param_info(mpc):
    """
    Builds a list of (line_id, param_name, param_value) for each line in mpc["line3p"]
    using the data in mpc["lc"]. The 'param_name' can be e.g. 'Zaa_R', 'Zab_X', etc.
    and 'param_value' is the numerical value from the R, X in the primitive matrix.
    """
    line3p = mpc["line3p"]   # shape (#lines, 6) => [id, fbus, tbus, status, lcid, length_miles]
    lc = mpc["lc"]          # shape (#lcid_rows, 19?), e.g. [ [lcid, R11, R21, R31, R22, R32, R33, X11, X21,...], ...]

    lc_dict = {}
    for row in lc:
        lcid = int(row[0])
        R11 = row[1]; R21 = row[2]; R31 = row[3]; R22 = row[4]; R32 = row[5]; R33 = row[6]
        X11 = row[7]; X21 = row[8]; X31 = row[9]; X22 = row[10]; X32 = row[11]; X33 = row[12]
        # ignoring the rest if it's line-charging or other columns

        # store them
        lc_dict[lcid] = {
            'Raa': R11, 'Rab': R21, 'Rac': R31, 'Rbb': R22, 'Rbc': R32, 'Rcc': R33,
            'Xaa': X11, 'Xab': X21, 'Xac': X31, 'Xbb': X22, 'Xbc': X32, 'Xcc': X33
        }

    param_info = []
    # For each line, gather the 12 parameters from the associated lcid row
    for line_row in line3p:
        line_id = int(line_row[0])
        lcid = int(line_row[4])   # 5th column => lcid
        length_mi = line_row[5]   # if you want to scale R, X by length

        # retrieve the dictionary with the R/X for that line code
        # scale by 'length_mi' if you want the total R, X => R_total = R_ij * length
        # or if the data is already total, skip.
        lc_data = lc_dict[lcid]

        # build the 12 param_info for this line
        # e.g. (line_id, 'Zaa_R', value)
        param_info += [
          (line_id, 'Zaa_R', lc_data['Raa'] * length_mi),
          (line_id, 'Zab_R', lc_data['Rab'] * length_mi),
          (line_id, 'Zac_R', lc_data['Rac'] * length_mi),
          (line_id, 'Zbb_R', lc_data['Rbb'] * length_mi),
          (line_id, 'Zbc_R', lc_data['Rbc'] * length_mi),
          (line_id, 'Zcc_R', lc_data['Rcc'] * length_mi),

          (line_id, 'Zaa_X', lc_data['Xaa'] * length_mi),
          (line_id, 'Zab_X', lc_data['Xab'] * length_mi),
          (line_id, 'Zac_X', lc_data['Xac'] * length_mi),
          (line_id, 'Zbb_X', lc_data['Xbb'] * length_mi),
          (line_id, 'Zbc_X', lc_data['Xbc'] * length_mi),
          (line_id, 'Zcc_X', lc_data['Xcc'] * length_mi),
        ]

    return param_info


def jacobian_line_params(x, Ybus, mpc, busphase_map):
    """
    Returns partial of measurement h wrt each line param p in param_list.
    shape => (#meas, nparam)
    We do chain rule for each param:
      1) identify which line & which Z_{ab} => partial Y_sub wrt that param => partial flows
      2) place partial in final column
    """
    # #meas => same dimension as measurement_function => let's get m_inj + m_flow + m_v
    nnodephase = len(busphase_map)
    n_lines = len(mpc["line3p"])
    # compute total # of measurements
    m_inj = 2*nnodephase
    m_flow = 4*(3*n_lines)
    m_v = nnodephase
    m_total = m_inj + m_flow + m_v

    param_info = build_param_info(mpc)

    # We'll build Hparam => shape(m_total, nparam)
    n_params = len(param_info)
    Hparam = np.zeros((m_total, n_params), dtype=float)

    for c, (line_id, p_name, p_val) in enumerate(param_info):
        # compute partial wrt this param
        dh_dp = compute_param_flow_injection_partial(
            x, Ybus, mpc, busphase_map,
            line_id, p_name
        )
        Hparam[:, c] = dh_dp

    return Hparam

def compute_param_flow_injection_partial(
    x,
    Ybus,
    mpc,
    busphase_map,
    line_id,
    param_type
):
    """
    Symbolic partial derivative of ALL measurements h wrt a single line param p
    (which is param_type in line line_id).
    E.g. param_type could be 'Zaa_R', 'Zab_X', etc.

    We return dh/dp, shape (#meas, ), where #meas = 2*N + 4*(3*n_lines) + N.

    Steps:
      1) Identify which 6x6 sub-block of Ybus is impacted by param p.
      2) Derive partial of that sub-block wrt p. (Symbolic: Y' = -Y * E_{(a,b)} * Y if p is a self/mutual?)
      3) For each measurement:
         - If it's P_inj/Q_inj for a node i in from/to bus, or a line flow for line_id,
           do partial = chain rule wrt param p.
         - Otherwise partial = 0.

    We'll show the direct chain rule for line flows & bus injection
    referencing the sub-block.
    """

    #--- dimensions
    n_lines = len(mpc["line3p"])
    nbus = len(mpc["bus3p"])
    nnodephase = len(busphase_map)

    # number of measurements
    m_inj = 2*nnodephase
    m_flow = 4*(3*n_lines)
    m_v = nnodephase
    m_total = m_inj + m_flow + m_v

    # result
    dh = np.zeros(m_total, dtype=float)

    half = nnodephase
    Vm = x[:half]
    Va = x[half:]
    V = Vm * np.exp(1j*Va)

    # 1) figure out the from_bus, to_bus for 'line_id'
    # line => [id, from_bus, to_bus, ...], so let's find that in mpc["line3p"]

    line_data = mpc["line3p"][line_id - 1, :]
    fbus = int(line_data[1]) - 1
    tbus = int(line_data[2]) - 1

    # gather the from-phase indices & to-phase indices
    f_idx = [fbus*3, fbus*3+1, fbus*3+2]
    t_idx = [tbus*3, tbus*3+1, tbus*3+2]
    rowcol = f_idx + t_idx   # 6 node indices
    # A) partial of bus injection P_inj(i), Q_inj(i) for i in rowcol
    # B) partial of line flow Pf, Pt, Qf, Qt for line_id phases
    # C) partial of Vmag => 0 since we assume param doesn't change magnitude if

    for local_i in rowcol:
        dPdp, dQdp = partial_p_bus_injection(
            local_i, param_type, line_id, V, Ybus, f_idx, t_idx, rowcol
        )
        dh[local_i] = dPdp      # for P_inj
        dh[local_i + nnodephase] = dQdp  # for Q_inj

    #--- (B) partial of line flows => line_id
    # in the measurement vector, line flows are in block => offset_flow = 2*nnodephase
    # each line has 3 phases => we do for alpha in [0,1,2], each alpha => (Pf, Pt, Qf, Qt).
    # Suppose the line ordering is the same as the index in mpc["line3p"]. We find the
    # index_of_line in that list.
    # Then for alpha in [0..2], the row offset is offset_flow + 4*(line_id*3 + alpha)
    #  => Pf => row0, Pt => row1, Qf => row2, Qf => row3
    offset_flow = 2*nnodephase
    for alpha in range(3):
        dPf, dPt, dQf, dQt = partial_p_line_flow(
            alpha, param_type, line_id, V, Ybus, f_idx, t_idx, rowcol
        )
        row_base = offset_flow + (line_id -1)*3 + alpha
        dh[row_base] = dPf
        dh[row_base + (n_lines)*3] = dPt
        dh[row_base + (n_lines)*6] = dQf
        dh[row_base + (n_lines)*9] = dQt

    #--- (C) partial of Vmag => we can assume 0 if param changes don't alter
    #    distribution feeder's node voltage magnitude measurement directly
    return dh

def partial_p_bus_injection(node_i, param_type, line_id, V, Ybus, f_idx, t_idx, rowcol):
    """
    Symbolic partial derivative of (P_inj(node_i), Q_inj(node_i)) wrt param p
    of line line_id in the upper-triangle 3x3 impedance.

    If node_i not in rowcol => partial is 0.
    Else we do:
      P_inj(i) = real( V[i]* conj( I[i]) ),
      I[i] = sum_j Ybus(i,j)*V[j].

    partial wrt param =>
      partial I[i] = sum_j partial(Ybus(i,j)) wrt param * V[j],
      partial P_inj(i) = real( V[i]* conj( partial I[i] ) ),
      partial Q_inj(i) = imag( V[i]* conj( partial I[i] ) ).

    For demonstration, we only update the 6×6 sub-block of Ybus associated w/ line_id's from/to phases.
    """
    # Return (dPdp, dQdp)
    dPdp = 0.0
    dQdp = 0.0

    # 1) Check if node_i is in rowcol => if not => partial=0
    if node_i not in rowcol:
        return (0.0, 0.0)

    # 2) Build partial dYsub wrt param p (the 6×6 sub-block for from->to)
    # rowcol = f_idx + t_idx => e.g. [fA,fB,fC, tA,tB,tC] => 6 nodes
    # extract Ysub => shape(6,6)
    Ysub = Ybus[np.ix_(rowcol, rowcol)]
    # build dZsub => 3×3 for the line's primitive => embed in 6×6 => dZ6x6
    dY6 = build_dYsub_symbolic(Ysub, param_type)  # shape(6,6) partial wrt param
    # If your parameter doesn't affect diagonal => it might be 0 in some places

    # 3) partial of I[node_i] wrt param => sum_j dYbus(node_i, j)*V[j]
    #   but node_i local => local_i = rowcol.index(node_i)
    local_i = rowcol.index(node_i)
    # partial of Ybus(node_i, j) => dY6[ local_i, local_j ]
    # I[node_i] = sum_j Ybus(node_i,j)*V[j], j in [0..N-1]
    # but only j in rowcol matter => partial I => sum_{local_j=0..5} dY6(local_i, local_j)* V[rowcol[local_j]]
    dI_param = 0+0j
    for local_j in range(6):
        j_global = rowcol[local_j]
        dI_param += dY6[local_i, local_j] * V[j_global]

    # 4) partial P_inj => real( V[node_i]* conj( dI_param ) )
    # partial Q_inj => imag( V[node_i]* conj( dI_param ) )
    Vi = V[node_i]
    dSf = Vi * np.conjugate(dI_param)   # complex partial of S_inj
    dPdp = dSf.real
    dQdp = dSf.imag

    return (dPdp, dQdp)


def partial_p_line_flow(alpha, param_type, line_id, V, Ybus, f_idx, t_idx, rowcol):
    """
    Symbolic partial of Pf,Pt,Qf,Qt for the line line_id, phase alpha, wrt param p.

    We do:
      Pf = real( Vf(alpha)* conj(Ifrom(alpha)) ),
      Ifrom(alpha) = sum_{xph} Ysub(alpha,xph)*(Vf[xph]) - ...
    partial => partial Ifrom wrt param => ...
    Similar for Pt => Vt(alpha)* conj(Ito(alpha)).
    """
    dPf = 0.0
    dPt = 0.0
    dQf = 0.0
    dQt = 0.0

    # 1) build dY6 = partial of 6×6 sub-block wrt param
    Ysub = Ybus[np.ix_(rowcol, rowcol)]
    dY6 = build_dYsub_symbolic(Ysub, param_type)

    # (a) partial Pf => real( Vf_alpha * conj( Ifrom ) )
    # => partial => real( Vf_alpha * conj( partial Ifrom ) ), ignoring partial of Vf_alpha wrt param
    # similarly for Qf, Pt, Qt
    # We'll define small helper:
    dPf, dQf = partial_flow_from_side(alpha, dY6, V, f_idx, t_idx, param_type)
    dPt, dQt = partial_flow_to_side(alpha, dY6, V, f_idx, t_idx, param_type)
    return (dPf, dPt, dQf, dQt)

def build_dYsub_symbolic(Ysub, param_type):
    """
    Build partial of the 6×6 sub-block Ysub wrt param p, in symbolic form.
    E.g. param_type='Zab_R' => we set dZsub( a,b )=1. Then dYsub = -Ysub * dZsub * Ysub.
    We'll embed the 3×3 into top-left or do a more advanced approach.
    For demonstration, we just do top-left 3×3 => row0..2,col0..2 => from bus phases.
    Then row3..5 => to bus phases.
    """
    dY6 = np.zeros((6,6), dtype=complex)

    # parse param_type => which (a,b) in {0,1,2}, real or imag?
    # e.g. param_type='Zab_R' => (a=0,b=1, real=1). param_type='Zab_X' => (a=0,b=1, imag=1)...

    # minimal example:
    # 1) build dZ3 => 3×3
    dZ3 = np.zeros((3,3), dtype=complex)

    # parse which phases?
    if param_type == 'Zaa_R':
        dZ3[0, 0] = 1.0
    elif param_type == 'Zaa_X':
        dZ3[0, 0] = 1j

    elif param_type == 'Zab_R':
        dZ3[0, 1] = 1.0
        dZ3[1, 0] = 1.0
    elif param_type == 'Zab_X':
        dZ3[0, 1] = 1j
        dZ3[1, 0] = 1j

    elif param_type == 'Zac_R':
        dZ3[0, 2] = 1.0
        dZ3[2, 0] = 1.0
    elif param_type == 'Zac_X':
        dZ3[0, 2] = 1j
        dZ3[2, 0] = 1j

    elif param_type == 'Zbb_R':
        dZ3[1, 1] = 1.0
    elif param_type == 'Zbb_X':
        dZ3[1, 1] = 1j

    elif param_type == 'Zbc_R':
        dZ3[1, 2] = 1.0
        dZ3[2, 1] = 1.0
    elif param_type == 'Zbc_X':
        dZ3[1, 2] = 1j
        dZ3[2, 1] = 1j

    elif param_type == 'Zcc_R':
        dZ3[2, 2] = 1.0
    elif param_type == 'Zcc_X':
        dZ3[2, 2] = 1j

    else:
        # If we have an unexpected param name, leave dZ3 as all zeros
        pass
    # similarly handle 'Zaa_R','Zaa_X','Zac_R','Zbc_X', etc.

    # 2) compute dY3 => - Y3 * dZ3 * Y3, where Y3= top-left 3×3 of Ysub?
    # or we might embed the from bus in row0..2 => from bus, row3..5 => to bus.
    # Typically Yff => top-left 3×3, Yft => top-right 3×3, etc.
    # For demonstration, let's do partial for top-left block only:
    Y3 = -Ysub[3:, 0:3]
    dY3 = - Y3 @ (dZ3 @ Y3)

    # embed dY3 in dY6
    dY6[0:3, 0:3] = dY3
    dY6[3:, 3:] = dY3
    dY6[0:3, 3:] = -dY3
    dY6[3:, 0:3] = -dY3

    return dY6

def partial_flow_from_side(alpha, dY6, V, f_idx, t_idx, param_type):
    """
    partial (Pf, Qf) wrt param => 2 real numbers
    Pf = real( Vf(alpha)* conj(Ifrom(alpha)) ),
    Ifrom(alpha) = sum_{xph} Ysub(alpha,xph)*V[f_idx[xph]] - ...
    => partial Ifrom = sum_j dY6(alpha, j)* V[rowcol[j]] ...
    """
    # compute partial Ifrom wrt param
    # row=alpha => 0..2, col => 0..5
    dIf = 0+0j
    for xph in range(3):
        # from-phase => col xph
        iFx = f_idx[xph]
        iTx = t_idx[xph]
        dIf += -dY6[alpha, 3+xph]* (V[iFx] - V[iTx])
    # partial Sf => V[f_idx[alpha]] * conj( dIf ), ignoring partial V wrt param
    Vf_alpha = V[f_idx[alpha]]
    dSf = Vf_alpha * np.conjugate(dIf)
    dPf = dSf.real
    dQf = dSf.imag
    return (dPf, dQf)

def partial_flow_to_side(alpha, dY6, V, f_idx, t_idx, param_type):
    """
    partial (Pt, Qt).
    Pt= real( Vt(alpha)* conj(Ito(alpha)) ).
    row= 3+alpha => etc.
    """
    dIt = 0+0j
    for xph in range(3):
        # row=3+alpha, col => xph => plus? minus?
        dIt += -dY6[3+alpha, xph] * (V[t_idx[xph]] - V[f_idx[xph]])
        # etc. depends on sign
        # dIt += - dY6[3+alpha, 3+xph]* V[t_idx[xph]]
    Vt_alpha = V[t_idx[alpha]]
    dSt = Vt_alpha * np.conjugate(dIt)
    dPt = dSt.real
    dQt = dSt.imag
    return (dPt, dQt)
