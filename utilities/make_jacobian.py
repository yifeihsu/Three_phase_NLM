import numpy as np
def jacobian(x, Ybus, mpc, busphase_map):
    """
    Builds the Jacobian matrix wrt [Vm, Va] in polar coords,
    to match a measurement vector that includes:
      1) P_inj (N)
      2) Q_inj (N)
      3) P_f (3*n_lines), P_t(3*n_lines), Q_f(3*n_lines), Q_t(3*n_lines)
      4) Vmag (N)
    """
    # Basic shapes
    n_lines = len(mpc["line3p"])
    nbus = len(mpc["bus3p"])
    nnodephase = len(busphase_map)

    # separate x => (Vm, Va)
    half = nnodephase
    Vm = x[:half]
    Va = x[half:]
    V = Vm * np.exp(1j * Va)
    def dsbus_polar(Ybus, V, Vm):
        """
        Returns (dSbus_dVa, dSbus_dVm) for polar coords.
          Ibus = Ybus*V
          Sbus = diag(V) * conj(Ibus)
        Then in polar coords:
          dSbus/dVa = j * diag(V) * conj( Ibus - Ybus*diag(V) )
          dSbus/dVm = diag(V) * conj( Ybus * diag(V./abs(V)) ) + conj(diag(Ibus)) * diag(V./abs(V))
        Because we store V[i] = Vm[i]* exp(j Va[i]), and abs(V[i])=Vm[i].
        """
        Ibus = Ybus @ V
        diagV = np.diag(V)
        diagI = np.diag(Ibus)
        # dSbus_dVa = j * diag(V) * conj( diag(Ibus) - Ybus*diag(V) )
        # dSbus_dVm = diag(V) * conj( Ybus* diag(Vnorm) ) + conj(diag(Ibus)) * diag(Vnorm)
        #  where Vnorm = V/abs(V)
        Vnorm = V / Vm
        # partial w.r.t. Va
        term = diagI - (Ybus @ np.diag(V))
        dSbus_dVa = 1j * diagV @ np.conjugate(term)

        # partial w.r.t. Vm
        tmp1 = diagV @ np.conjugate(Ybus @ np.diag(Vnorm))
        tmp2 = np.conjugate(diagI) @ np.diag(Vnorm)
        dSbus_dVm = tmp1 + tmp2

        return dSbus_dVa, dSbus_dVm
    # A) partial derivatives for bus injections
    #    we have dsbus_polar => dSbus_dVa, dSbus_dVm
    dSbus_dVa, dSbus_dVm = dsbus_polar(Ybus, V, Vm)

    # We'll compute final #rows for the entire measurement set
    # 1) P_inj + Q_inj => 2*N
    # 2) P_f, P_t, Q_f, Q_t => 4*(3*n_lines)
    # 3) Vmag => N
    m_inj = 2*nnodephase
    m_flow = 4 * (3 * n_lines)
    m_v = nnodephase
    m_total = m_inj + m_flow + m_v

    # We'll build H => shape (m_total, 2*nnodephase)
    H = np.zeros((m_total, 2*nnodephase), dtype=float)

    #--------------------------------------------------------------------------
    # (1) partials for bus injection => first 2*N rows
    #  - row i => P_inj(i) = real( Sbus[i] )
    #  - row i+N => Q_inj(i) = imag( Sbus[i] )
    # columns => [d/dVm(0..N-1), d/dVa(0..N-1)]
    for i in range(nnodephase):
        rowP = i
        rowQ = i + nnodephase

        # partial wrt Vm => real(...) => 1..N
        H[rowP, :half] = np.real(dSbus_dVm[i,:])
        # partial wrt Va => real(...)
        H[rowP, half:] = np.real(dSbus_dVa[i,:])

        # partial wrt Vm => imag(...) => Q_inj
        H[rowQ, :half] = np.imag(dSbus_dVm[i,:])
        # partial wrt Va => imag(...)
        H[rowQ, half:] = np.imag(dSbus_dVa[i,:])

    #--------------------------------------------------------------------------
    # (2) partials for line flows => rows [2*N.. 2*N + 4*(3*n_lines) -1]
    # offset_flow => start row for flow partials
    offset_flow = 2 * nnodephase
    idx_flow = 0  # how many line-phase combos so far

    for line_id, line in enumerate(mpc["line3p"]):
        # line => [line_id, from_bus, to_bus, ...]
        fbus = int(line[1]) - 1  # 0-based
        tbus = int(line[2]) - 1

        # gather from-bus, to-bus node indices
        f_idx = [fbus*3, fbus*3 + 1, fbus*3 + 2]
        t_idx = [tbus*3, tbus*3 + 1, tbus*3 + 2]

        # For each phase alpha in [0,1,2]
        for alpha in [0,1,2]:
            # call the partial_line_flow_polar => shape(4, 2*N)
            block_4x = partial_line_flow_polar(
                x, Ybus, f_idx, t_idx, alpha, busphase_map
            )
            # block_4x => rows => [dPf, dPt, dQf, dQt], each shape(1 x 2N)

            # place them in the big H
            # we have 4 rows per line-phase
            row_start = offset_flow + idx_flow
            H[row_start, :] = block_4x[0, :]
            H[row_start + 3*n_lines, :] = block_4x[1, :]
            H[row_start + 6*n_lines, :] = block_4x[2, :]
            H[row_start + 9*n_lines, :] = block_4x[3, :]

            idx_flow += 1

    #--------------------------------------------------------------------------
    # (3) partials for Vmag => last N rows => offset_v = offset_flow + 4*(3*n_lines)
    offset_v = offset_flow + 4 * (3 * n_lines)
    for i in range(nnodephase):
        # partial of Vmag(i) wrt V_m(i) => 1
        # partial wrt V_a(i) => 0
        rowV = offset_v + i
        H[rowV, i] = 1.0

    return H

def partial_line_flow_polar(x, Ybus, f_idx, t_idx, alpha, busphase_map):
    """
    Returns partial derivatives wrt (Vm, Va) for the 4 measurements:
      [ Pf, Pt, Qf, Qt ] for ONE line-phase 'alpha'.

    shape => (4 x 2*N)

    Notation:
      - Pf = real( Sf ), Qf = imag( Sf ) where Sf = Vf(alpha) * conj(If(alpha))
      - Pt, Qt similarly for the 'to' side.

    Steps:
      1) Compute If, It from sub-block of Ybus (already done by measurement_function).
      2) Then do partial wrt each state's Vm(k), Va(k).

    We'll do an explicit chain rule. We sum over the
    relevant local xph in [0..2], but also must check how V[k] changes
    when we do partial wrt (Vm(k), Va(k)).
    """

    # number of total node-phases
    nnodephase = len(busphase_map)
    half = nnodephase

    # separate x => (Vm, Va)
    Vm = x[:half]
    Va = x[half:]
    V = Vm * np.exp(1j * Va)

    # We'll build a (4 x 2*N) block => first row => dPf/dx, second => dPt/dx,
    # third => dQf/dx, fourth => dQt/dx
    block = np.zeros((4, 2*nnodephase), dtype=float)

    #----------------------------------------------------------------------
    # 1) Compute the from-side current I_f(alpha) and to-side current I_t(alpha)
    #    (like in measurement_function, but just for alpha)
    # We'll do a local sub-block from Ybus
    rowcol = f_idx + t_idx  # [fA,fB,fC, tA,tB,tC]
    Ysub = Ybus[np.ix_(rowcol, rowcol)]  # shape(6,6)

    iF = f_idx[alpha]
    iT = t_idx[alpha]
    Vf_alpha = V[iF]
    Vt_alpha = V[iT]

    # Ifrom:
    Ifrom = 0+0j
    for xph in range(3):
        # from-phase row => alpha,
        # from-phase col => xph => Ysub[alpha, xph]
        # to-phase col => 3 + xph => Ysub[alpha, 3+xph]
        Y_ff = Ysub[alpha, xph]
        Y_ft = Ysub[alpha, 3 + xph]
        iFx = f_idx[xph]
        iTx = t_idx[xph]
        Ifrom += -Y_ft * (V[iFx] - V[iTx])

    # Ito:
    Ito = 0+0j
    for xph in range(3):
        Y_tf = Ysub[3+alpha, xph]
        Y_tt = Ysub[3+alpha, 3 + xph]
        iFx = f_idx[xph]
        iTx = t_idx[xph]
        Ito += Y_tf * (V[iFx] - V[iTx])

    # define S_from, S_to
    Sf = Vf_alpha * np.conjugate(Ifrom)
    St = Vt_alpha * np.conjugate(Ito)

    # we'll define real, imag
    Pf = Sf.real
    Qf = Sf.imag
    Pt = St.real
    Qt = St.imag

    #----------------------------------------------------------------------
    # 2) partial wrt each state: for k in [0..N-1], partial wrt Vm(k) or Va(k).
    # We'll do a loop over k => 0..(nnodephase-1). Then fill columns for
    # d/dVm(k) => col k, d/dVa(k) => col (k + half).
    #
    # Each partial is sum of partial wrt Vf_alpha plus partial wrt Ifrom, etc.
    # We do standard chain rule:
    #    Pf = Re{ Vf_alpha * conj(Ifrom) }
    #         => partial wrt V[i], Ifrom depends on V[i], ...
    #
    # We'll define small helpers to get dV/dVm, dV/dVa for node i.

    def dV_dVm(i):
        """ partial of V[i] wrt Vm(i). If 'k' != i => 0, else => e^(j*Va(i)). """
        return np.exp(1j * Va[i])

    def dV_dVa(i):
        """ partial of V[i] wrt Va(i). => j * V[i]. If k != i => 0. """
        return 1j * V[i]

    # We'll define partial of Ifrom wrt V[i], etc. Then partial of Sf wrt V[i].
    # Then take real part => dPf, imag => dQf.

    for k in range(nnodephase):
        # partial wrt Vm(k) => column k
        # partial wrt Va(k) => column (k+half)
        col_vm = k
        col_va = k + half

        #-------------- partial wrt Vm(k)
        # define dV_k = dV[i]/dVm(k). If i != k => 0
        # We'll compute partial of Pf => real( dSf/dVm(k) ), etc.

        # A) partial wrt Vm(k) of Pf = Re{ dSf/dVm(k) }
        dSf_dVm_k = compute_dSf_dVm_k(k, iF, iT, Ifrom, Vf_alpha, Vt_alpha, dV_dVm, V, Ysub, f_idx, t_idx, alpha)
        # Pf => real( Sf ), so dPf/dVm(k) => real( dSf/dVm(k) )
        dPf_dVm_k = dSf_dVm_k.real
        dQf_dVm_k = dSf_dVm_k.imag

        # B) partial wrt Vm(k) of Pt
        dSt_dVm_k = compute_dSt_dVm_k(k, iF, iT, Ito, Vf_alpha, Vt_alpha, dV_dVm, V, Ysub, f_idx, t_idx, alpha)
        dPt_dVm_k = dSt_dVm_k.real
        dQt_dVm_k = dSt_dVm_k.imag

        block[0, col_vm] = dPf_dVm_k   # row0 => Pf
        block[1, col_vm] = dPt_dVm_k   # row1 => Pt
        block[2, col_vm] = dQf_dVm_k   # row2 => Qf
        block[3, col_vm] = dQt_dVm_k   # row3 => Qt

        #-------------- partial wrt Va(k)
        dSf_dVa_k = compute_dSf_dVa_k(k, iF, iT, Ifrom, Vf_alpha, Vt_alpha, dV_dVa, V, Ysub, f_idx, t_idx, alpha)
        dPf_dVa_k = dSf_dVa_k.real
        dQf_dVa_k = dSf_dVa_k.imag

        dSt_dVa_k = compute_dSt_dVa_k(k, iF, iT, Ito, Vf_alpha, Vt_alpha, dV_dVa, V, Ysub, f_idx, t_idx, alpha)
        dPt_dVa_k = dSt_dVa_k.real
        dQt_dVa_k = dSt_dVa_k.imag

        block[0, col_va] = dPf_dVa_k
        block[1, col_va] = dPt_dVa_k
        block[2, col_va] = dQf_dVa_k
        block[3, col_va] = dQt_dVa_k

    return block

#------------------------------------------------------------------------------
# Helper subroutines for partial derivatives wrt Vm(k) or Va(k).
# For brevity, these are fairly verbose. Each returns a complex "delta" for dSf or dSt.
# The real part => dPf, imag part => dQf, etc.

def compute_dSf_dVm_k(k, iF, iT, Ifrom, Vf_alpha, Vt_alpha, dV_dVm, V, Ysub, f_idx, t_idx, alpha):
    """
    partial derivative of Sf(alpha) = V_f(alpha)*conj(Ifrom(alpha))
    wrt Vm(k). Returns a complex number dSf/dVm(k).
    """
    # dSf/dVm(k) = d(Vf_alpha)/dVm(k) * conj(Ifrom) + Vf_alpha* conj( dIfrom/dVm(k) )
    dSf = 0+0j
    # partial of Vf_alpha wrt Vm(k)
    if k == iF:
        # d(Vf_alpha)/dVm(k) => e^(j Va(iF))
        dVf = dV_dVm(iF)
        dSf += dVf * np.conjugate(Ifrom)

    # plus Vf_alpha * conj( dIfrom/dVm(k) )
    dIfrom_vm = compute_dIfrom_dVm(k, alpha, Ysub, f_idx, t_idx, V, dV_dVm)
    dSf += Vf_alpha * np.conjugate(dIfrom_vm)
    return dSf

def compute_dSt_dVm_k(k, iF, iT, Ito, Vf_alpha, Vt_alpha, dV_dVm, V, Ysub, f_idx, t_idx, alpha):
    """
    partial derivative of St(alpha) = Vt_alpha* conj(Ito(alpha))
    wrt Vm(k). Returns a complex.
    """
    dSt = 0+0j
    if k == iT:
        dVt = dV_dVm(iT)
        dSt += dVt * np.conjugate(Ito)

    dIto_vm = compute_dIto_dVm(k, alpha, Ysub, f_idx, t_idx, V, dV_dVm)
    dSt += Vt_alpha * np.conjugate(dIto_vm)
    return dSt

def compute_dSf_dVa_k(k, iF, iT, Ifrom, Vf_alpha, Vt_alpha, dV_dVa, V, Ysub, f_idx, t_idx, alpha):
    """
    partial derivative of Sf(alpha) = V_f(alpha)*conj(Ifrom(alpha))
    wrt Va(k).
    """
    dSf = 0+0j
    if k == iF:
        dVf = dV_dVa(iF)
        dSf += dVf * np.conjugate(Ifrom)

    dIfrom_va = compute_dIfrom_dVa(k, alpha, Ysub, f_idx, t_idx, V, dV_dVa)
    dSf += Vf_alpha * np.conjugate(dIfrom_va)
    return dSf

def compute_dSt_dVa_k(k, iF, iT, Ito, Vf_alpha, Vt_alpha, dV_dVa, V, Ysub, f_idx, t_idx, alpha):
    """
    partial derivative of St(alpha) = Vt_alpha* conj(Ito(alpha))
    wrt Va(k).
    """
    dSt = 0+0j
    if k == iT:
        dVt = dV_dVa(iT)
        dSt += dVt * np.conjugate(Ito)

    dIto_va = compute_dIto_dVa(k, alpha, Ysub, f_idx, t_idx, V, dV_dVa)
    dSt += Vt_alpha * np.conjugate(dIto_va)
    return dSt

#------------------------------------------------------------------------------
# partials of Ifrom wrt Vm(k) or Va(k)

def compute_dIfrom_dVm(k, alpha, Ysub, f_idx, t_idx, V, dV_dVm):
    """
    Ifrom(alpha) = sum_{xph} [ Ysub(alpha,xph)*V_f(xph) - Ysub(alpha,3+xph)*V_t(xph) ]
    partial wrt Vm(k)
    """
    dI = 0+0j
    for xph in range(3):
        Y_ff = Ysub[alpha, xph]
        Y_ft = Ysub[alpha, 3 + xph]
        iFx = f_idx[xph]
        iTx = t_idx[xph]

        if k == iFx:
            # partial of V[iFx] wrt Vm(k) => dV_dVm(iFx)
            dI += - Y_ft * dV_dVm(iFx)
        if k == iTx:
            # minus sign
            dI += Y_ft * dV_dVm(iTx)

    return dI

def compute_dIfrom_dVa(k, alpha, Ysub, f_idx, t_idx, V, dV_dVa):
    """
    partial Ifrom wrt Va(k)
    """
    dI = 0+0j
    for xph in range(3):
        Y_ff = Ysub[alpha, xph]
        Y_ft = Ysub[alpha, 3 + xph]
        iFx = f_idx[xph]
        iTx = t_idx[xph]

        if k == iFx:
            dI += -Y_ft * dV_dVa(iFx)
        if k == iTx:
            dI += Y_ft * dV_dVa(iTx)

    return dI

#------------------------------------------------------------------------------
# partials of Ito wrt Vm(k) or Va(k)

def compute_dIto_dVm(k, alpha, Ysub, f_idx, t_idx, V, dV_dVm):
    """
    Ito(alpha) = sum_{xph} Ysub(3+alpha,xph)*V_f(xph) - Ysub(3+alpha, 3+xph)*V_t(xph)
    partial wrt Vm(k).
    """
    dI = 0+0j
    for xph in range(3):
        Y_tf = Ysub[3+alpha, xph]
        Y_tt = Ysub[3+alpha, 3 + xph]
        iFx = f_idx[xph]
        iTx = t_idx[xph]

        if k == iFx:
            dI += Y_tf * dV_dVm(iFx)
        if k == iTx:
            dI += -Y_tf * dV_dVm(iTx)

    return dI

def compute_dIto_dVa(k, alpha, Ysub, f_idx, t_idx, V, dV_dVa):
    """
    partial Ito wrt Va(k).
    """
    dI = 0+0j
    for xph in range(3):
        Y_tf = Ysub[3+alpha, xph]
        Y_tt = Ysub[3+alpha, 3 + xph]
        iFx = f_idx[xph]
        iTx = t_idx[xph]

        if k == iFx:
            dI += Y_tf * dV_dVa(iFx)
        if k == iTx:
            dI += - Y_tf * dV_dVa(iTx)

    return dI
