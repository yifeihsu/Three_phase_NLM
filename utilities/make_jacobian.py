import numpy as np
import scipy.sparse as sps

def jacobian(x, Ybus, mpc, busphase_map):
    """
    Builds the Jacobian matrix wrt [Vm, Va] in polar coords,
    to match a measurement vector that includes:
      1) P_inj (N)
      2) Q_inj (N)
      3) P_f (3*n_lines), P_t(3*n_lines), Q_f(3*n_lines), Q_t(3*n_lines)
      4) Vmag (N)
    """
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
    n_lines = len(mpc["line3p"])
    nbus = len(mpc["bus3p"])
    nnodephase = len(busphase_map)

    half = nnodephase  # half = # of node-phases for magnitude
    Vm = x[:half]
    Va = x[half:]
    V = Vm * np.exp(1j * Va)

    # 1) We get the partial derivatives for bus injections in vectorized form.
    dSbus_dVa, dSbus_dVm = dsbus_polar(Ybus, V, Vm)
    # Pre-extract real and imag parts
    dSbus_dVm_real = dSbus_dVm.real
    dSbus_dVm_imag = dSbus_dVm.imag
    dSbus_dVa_real = dSbus_dVa.real
    dSbus_dVa_imag = dSbus_dVa.imag

    # Row counts for each measurement type:
    m_inj = 2 * nnodephase                  # P_inj + Q_inj
    m_flow = 4 * (3 * n_lines)             # P_f, P_t, Q_f, Q_t
    m_v = nnodephase                       # Vmag
    m_total = m_inj + m_flow + m_v

    # 2) Create a sparse LIL matrix to store the final Jacobian
    #    shape = (m_total, 2*nnodephase).
    H = sps.lil_matrix((m_total, 2*nnodephase), dtype=float)

    #--------------------------------------------------------------------------
    # (1) BUS INJECTION PART (first 2*N rows)
    #
    #  - rows 0..(nnodephase-1) => P_inj
    #  - rows nnodephase..(2*nnodephase-1) => Q_inj
    #
    #  columns => [ d/dVm(0..N-1), d/dVa(0..N-1) ]
    #
    # Instead of a for-loop over i, just assign the 2D blocks directly:
    H[0:nnodephase,        0:half] = dSbus_dVm_real
    H[0:nnodephase,        half:]  = dSbus_dVa_real
    H[nnodephase:2*nnodephase, 0:half] = dSbus_dVm_imag
    H[nnodephase:2*nnodephase, half:]  = dSbus_dVa_imag

    #--------------------------------------------------------------------------
    # (2) BRANCH FLOW PART (rows [2*N .. 2*N + 4*(3*n_lines)-1])
    # We'll place partials of [Pf, Pt, Qf, Qt]
    #
    # offset_flow = 2 * nnodephase  # row offset where flows start
    # idx_flow = 0                 # how many line-phase combos so far
    # #
    # # For each line, we call partial_line_flow_polar -> 4x(2*N) block
    # # Then we insert them into H in the correct row positions.
    #
    # for line_id, line in enumerate(mpc["line3p"]):
    #     fbus = int(line[1]) - 1  # 0-based
    #     tbus = int(line[2]) - 1
    #     f_idx = [fbus*3, fbus*3 + 1, fbus*3 + 2]
    #     t_idx = [tbus*3, tbus*3 + 1, tbus*3 + 2]
    #
    #     for alpha in [0,1,2]:
    #         block_4x = partial_line_flow_polar(x, Ybus, f_idx, t_idx,
    #                                            alpha, busphase_map)
    #         # block_4x: shape (4, 2*N) = [dPf; dPt; dQf; dQt]
    #         row_start = offset_flow + idx_flow
    #         # We have 4 rows per line-phase, but arranged in sets of
    #         #   + 0*n_lines => Pf
    #         #   + 3*n_lines => Pt
    #         #   + 6*n_lines => Qf
    #         #   + 9*n_lines => Qt
    #         #
    #         # so row_start is our "local offset", then we jump by
    #         # multiples of n_lines to place each row.
    #         H[row_start,                :] = block_4x[0, :]  # Pf
    #         H[row_start + 3*n_lines,    :] = block_4x[1, :]  # Pt
    #         H[row_start + 6*n_lines,    :] = block_4x[2, :]  # Qf
    #         H[row_start + 9*n_lines,    :] = block_4x[3, :]  # Qt
    #
    #         idx_flow += 1
    # Inside jacobian function, replace the branch flow part:
    offset_flow = 2 * nnodephase
    idx_flow = 0

    for line_id, line in enumerate(mpc["line3p"]):
        fbus = int(line[1]) - 1
        tbus = int(line[2]) - 1
        f_idx = [fbus * 3, fbus * 3 + 1, fbus * 3 + 2]
        t_idx = [tbus * 3, tbus * 3 + 1, tbus * 3 + 2]

        for alpha in [0, 1, 2]:
            # Get derivatives for relevant k's only
            (relevant_ks, dPf_dVm, dPf_dVa, dPt_dVm, dPt_dVa,
             dQf_dVm, dQf_dVa, dQt_dVm, dQt_dVa) = partial_line_flow_polar(
                x, Ybus, f_idx, t_idx, alpha, busphase_map
            )

            # Row indices for this line-phase
            row_Pf = offset_flow + idx_flow
            row_Pt = offset_flow + 3 * n_lines + idx_flow
            row_Qf = offset_flow + 6 * n_lines + idx_flow
            row_Qt = offset_flow + 9 * n_lines + idx_flow

            # Set non-zero entries directly in H
            for i, k in enumerate(relevant_ks):
                col_vm = k
                col_va = k + half
                H[row_Pf, col_vm] = dPf_dVm[i]
                H[row_Pf, col_va] = dPf_dVa[i]
                H[row_Pt, col_vm] = dPt_dVm[i]
                H[row_Pt, col_va] = dPt_dVa[i]
                H[row_Qf, col_vm] = dQf_dVm[i]
                H[row_Qf, col_va] = dQf_dVa[i]
                H[row_Qt, col_vm] = dQt_dVm[i]
                H[row_Qt, col_va] = dQt_dVa[i]

            idx_flow += 1
    #--------------------------------------------------------------------------
    # (3) VMAG PART (last N rows => offset_v)
    #
    offset_v = offset_flow + 4*(3*n_lines)
    for i in range(nnodephase):
        rowV = offset_v + i
        H[rowV, i] = 1.0

    return H.tocsr()


def partial_line_flow_polar(x, Ybus, f_idx, t_idx, alpha, busphase_map):
    """
    Returns partial derivatives of [Pf, Pt, Qf, Qt] for one line-phase 'alpha'
    with respect to Vm and Va of the relevant node-phases only.

    Returns:
    - relevant_ks: list of 6 indices [fA, fB, fC, tA, tB, tC]
    - dPf_dVm, dPf_dVa, dPt_dVm, dPt_dVa, dQf_dVm, dQf_dVa, dQt_dVm, dQt_dVa:
      arrays of shape (6,) containing derivatives for the relevant k's
    """
    nnodephase = len(busphase_map)
    half = nnodephase
    Vm = x[:half]
    Va = x[half:]
    V = Vm * np.exp(1j * Va)

    # Relevant k's: node-phases of from and to buses
    relevant_ks = f_idx + t_idx  # [fA, fB, fC, tA, tB, tC]

    # Initialize derivative arrays (4 measurements x 6 k's)
    dPf_dVm = np.zeros(6, dtype=float)
    dPf_dVa = np.zeros(6, dtype=float)
    dPt_dVm = np.zeros(6, dtype=float)
    dPt_dVa = np.zeros(6, dtype=float)
    dQf_dVm = np.zeros(6, dtype=float)
    dQf_dVa = np.zeros(6, dtype=float)
    dQt_dVm = np.zeros(6, dtype=float)
    dQt_dVa = np.zeros(6, dtype=float)

    # Local helpers
    def dV_dVm(i):
        return np.exp(1j * Va[i])

    def dV_dVa(i):
        return 1j * V[i]

    # Compute Ifrom and Ito
    rowcol = f_idx + t_idx
    Ysub = Ybus[np.ix_(rowcol, rowcol)]
    iF = f_idx[alpha]
    iT = t_idx[alpha]
    Vf_alpha = V[iF]
    Vt_alpha = V[iT]

    Ifrom = 0 + 0j
    Ito = 0 + 0j
    for xph in range(3):
        Y_ft = Ysub[alpha, 3 + xph]
        Y_tf = Ysub[3 + alpha, xph]
        iFx = f_idx[xph]
        iTx = t_idx[xph]
        Ifrom += -Y_ft * (V[iFx] - V[iTx])
        Ito += Y_tf * (V[iFx] - V[iTx])

    # Derivative computation functions (unchanged from original)
    def compute_dIfrom_both(k):
        dI_vm = 0 + 0j
        dI_va = 0 + 0j
        for xph in range(3):
            Y_ft = Ysub[alpha, 3 + xph]
            iFx = f_idx[xph]
            iTx = t_idx[xph]
            if k == iFx:
                dI_vm += -Y_ft * dV_dVm(iFx)
                dI_va += -Y_ft * dV_dVa(iFx)
            if k == iTx:
                dI_vm += Y_ft * dV_dVm(iTx)
                dI_va += Y_ft * dV_dVa(iTx)
        return dI_vm, dI_va

    def compute_dIto_both(k):
        dI_vm = 0 + 0j
        dI_va = 0 + 0j
        for xph in range(3):
            Y_tf = Ysub[3 + alpha, xph]
            iFx = f_idx[xph]
            iTx = t_idx[xph]
            if k == iFx:
                dI_vm += Y_tf * dV_dVm(iFx)
                dI_va += Y_tf * dV_dVa(iFx)
            if k == iTx:
                dI_vm += -Y_tf * dV_dVm(iTx)
                dI_va += -Y_tf * dV_dVa(iTx)
        return dI_vm, dI_va

    def compute_dSf_both(k):
        dI_vm, dI_va = compute_dIfrom_both(k)
        dSf_vm = Vf_alpha * np.conjugate(dI_vm)
        dSf_va = Vf_alpha * np.conjugate(dI_va)
        if k == iF:
            dVf_vm = dV_dVm(iF)
            dVf_va = dV_dVa(iF)
            dSf_vm += dVf_vm * np.conjugate(Ifrom)
            dSf_va += dVf_va * np.conjugate(Ifrom)
        return dSf_vm, dSf_va

    def compute_dSt_both(k):
        dI_vm, dI_va = compute_dIto_both(k)
        dSt_vm = Vt_alpha * np.conjugate(dI_vm)
        dSt_va = Vt_alpha * np.conjugate(dI_va)
        if k == iT:
            dVt_vm = dV_dVm(iT)
            dVt_va = dV_dVa(iT)
            dSt_vm += dVt_vm * np.conjugate(Ito)
            dSt_va += dVt_va * np.conjugate(Ito)
        return dSt_vm, dSt_va

    # Compute derivatives only for relevant k's
    for i, k in enumerate(relevant_ks):
        dSf_vm, dSf_va = compute_dSf_both(k)
        dSt_vm, dSt_va = compute_dSt_both(k)

        dPf_dVm[i] = dSf_vm.real
        dPf_dVa[i] = dSf_va.real
        dQf_dVm[i] = dSf_vm.imag
        dQf_dVa[i] = dSf_va.imag

        dPt_dVm[i] = dSt_vm.real
        dPt_dVa[i] = dSt_va.real
        dQt_dVm[i] = dSt_vm.imag
        dQt_dVa[i] = dSt_va.imag

    return (relevant_ks, dPf_dVm, dPf_dVa, dPt_dVm, dPt_dVa,
            dQf_dVm, dQf_dVa, dQt_dVm, dQt_dVa)

# def partial_line_flow_polar(x, Ybus, f_idx, t_idx, alpha, busphase_map):
#     """
#     Returns partial derivatives wrt (Vm, Va) for the 4 measurements:
#       [ Pf, Pt, Qf, Qt ] for ONE line-phase 'alpha'.
#
#     shape => (4 x 2*N)
#
#     Notation:
#       - Pf = real(Sf), Qf = imag(Sf), where Sf = Vf(alpha)*conjugate(Ifrom(alpha))
#       - Pt = real(St), Qt = imag(St), where St = Vt(alpha)*conjugate(Ito(alpha))
#
#     This revision returns the partial derivatives wrt Vm and Va from
#     single calls, to reduce overhead.
#     """
#
#     # --- 1) Basic setup ---
#     nnodephase = len(busphase_map)
#     half = nnodephase
#
#     # Separate x => (Vm, Va)
#     Vm = x[:half]
#     Va = x[half:]
#     V = Vm * np.exp(1j * Va)
#
#     # We'll build a (4 x 2*N) block
#     block = np.zeros((4, 2*nnodephase), dtype=float)
#
#     # Local helper: partial of V[i] wrt Vm(i) and Va(i)
#     def dV_dVm(i):
#         return np.exp(1j * Va[i])  # partial of V[i] wrt Vm(i)
#     def dV_dVa(i):
#         return 1j * V[i]           # partial of V[i] wrt Va(i)
#
#     # --- 2) Compute Ifrom(alpha) and Ito(alpha) from local sub-block ---
#     rowcol = f_idx + t_idx  # [fA, fB, fC, tA, tB, tC]
#     Ysub = Ybus[np.ix_(rowcol, rowcol)]  # shape(6,6)
#
#     iF = f_idx[alpha]
#     iT = t_idx[alpha]
#     Vf_alpha = V[iF]
#     Vt_alpha = V[iT]
#
#     # Ifrom, Ito
#     Ifrom = 0+0j
#     Ito   = 0+0j
#     for xph in range(3):
#         Y_ft = Ysub[alpha,     3 + xph]   # from-bus row => alpha
#         Y_tf = Ysub[3 + alpha,     xph]   # to-bus row   => (3+alpha)
#         iFx  = f_idx[xph]
#         iTx  = t_idx[xph]
#         Ifrom += -Y_ft * (V[iFx] - V[iTx])
#         Ito   +=  Y_tf * (V[iFx] - V[iTx])
#
#     Sf = Vf_alpha * np.conjugate(Ifrom)
#     St = Vt_alpha * np.conjugate(Ito)
#
#     # For reference: real & imag
#     Pf, Qf = Sf.real, Sf.imag
#     Pt, Qt = St.real, St.imag
#
#     # --- 3) Define unified derivative functions ---
#     def compute_dIfrom_both(k):
#         """
#         Returns (dI_vm, dI_va) for Ifrom wrt. Vm(k) and Va(k)
#         in a single pass over the 3 phases.
#         """
#         dI_vm = 0+0j
#         dI_va = 0+0j
#         for xph in range(3):
#             Y_ft = Ysub[alpha, 3 + xph]
#             iFx  = f_idx[xph]
#             iTx  = t_idx[xph]
#
#             if k == iFx:
#                 dI_vm += -Y_ft * dV_dVm(iFx)
#                 dI_va += -Y_ft * dV_dVa(iFx)
#             if k == iTx:
#                 dI_vm +=  Y_ft * dV_dVm(iTx)
#                 dI_va +=  Y_ft * dV_dVa(iTx)
#         return dI_vm, dI_va
#
#     def compute_dIto_both(k):
#         """
#         Returns (dI_vm, dI_va) for Ito wrt. Vm(k) and Va(k)
#         in a single pass.
#         """
#         dI_vm = 0+0j
#         dI_va = 0+0j
#         for xph in range(3):
#             Y_tf = Ysub[3 + alpha, xph]
#             iFx  = f_idx[xph]
#             iTx  = t_idx[xph]
#
#             if k == iFx:
#                 dI_vm +=  Y_tf * dV_dVm(iFx)
#                 dI_va +=  Y_tf * dV_dVa(iFx)
#             if k == iTx:
#                 dI_vm += -Y_tf * dV_dVm(iTx)
#                 dI_va += -Y_tf * dV_dVa(iTx)
#         return dI_vm, dI_va
#
#     def compute_dSf_both(k):
#         """
#         Returns (dSf_vm, dSf_va) = partial of Sf wrt Vm(k) and Va(k).
#         Sf = Vf_alpha * conj(Ifrom)
#         """
#         # Derivs of Ifrom wrt. Vm(k), Va(k)
#         dI_vm, dI_va = compute_dIfrom_both(k)
#
#         # dSf/dVm(k) = d(Vf_alpha)/dVm(k)*conj(Ifrom) + Vf_alpha*conj(dIfrom/dVm(k))
#         dSf_vm = 0+0j
#         if k == iF:
#             dVf_vm = dV_dVm(iF)
#             dSf_vm += dVf_vm * np.conjugate(Ifrom)
#         dSf_vm += Vf_alpha * np.conjugate(dI_vm)
#
#         # dSf/dVa(k) = d(Vf_alpha)/dVa(k)*conj(Ifrom) + Vf_alpha*conj(dIfrom/dVa(k))
#         dSf_va = 0+0j
#         if k == iF:
#             dVf_va = dV_dVa(iF)
#             dSf_va += dVf_va * np.conjugate(Ifrom)
#         dSf_va += Vf_alpha * np.conjugate(dI_va)
#
#         return dSf_vm, dSf_va
#
#     def compute_dSt_both(k):
#         """
#         Returns (dSt_vm, dSt_va) = partial of St wrt Vm(k) and Va(k).
#         St = Vt_alpha * conj(Ito)
#         """
#         # Derivs of Ito wrt. Vm(k), Va(k)
#         dI_vm, dI_va = compute_dIto_both(k)
#
#         # dSt/dVm(k)
#         dSt_vm = 0+0j
#         if k == iT:
#             dVt_vm = dV_dVm(iT)
#             dSt_vm += dVt_vm * np.conjugate(Ito)
#         dSt_vm += Vt_alpha * np.conjugate(dI_vm)
#
#         # dSt/dVa(k)
#         dSt_va = 0+0j
#         if k == iT:
#             dVt_va = dV_dVa(iT)
#             dSt_va += dVt_va * np.conjugate(Ito)
#         dSt_va += Vt_alpha * np.conjugate(dI_va)
#
#         return dSt_vm, dSt_va
#
#     # --- 4) Fill the Jacobian rows/columns ---
#     for k in range(nnodephase):
#         col_vm = k
#         col_va = k + half
#
#         # For Sf
#         dSf_vm, dSf_va = compute_dSf_both(k)
#         # For St
#         dSt_vm, dSt_va = compute_dSt_both(k)
#
#         # Place partials in block
#         # row 0 => Pf, row 1 => Pt, row 2 => Qf, row 3 => Qt
#         block[0, col_vm] = dSf_vm.real  # dPf/dVm
#         block[2, col_vm] = dSf_vm.imag  # dQf/dVm
#         block[0, col_va] = dSf_va.real  # dPf/dVa
#         block[2, col_va] = dSf_va.imag  # dQf/dVa
#
#         block[1, col_vm] = dSt_vm.real  # dPt/dVm
#         block[3, col_vm] = dSt_vm.imag  # dQt/dVm
#         block[1, col_va] = dSt_va.real  # dPt/dVa
#         block[3, col_va] = dSt_va.imag  # dQt/dVa
#
#     return block