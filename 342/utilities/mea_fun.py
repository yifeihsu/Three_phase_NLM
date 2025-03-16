import numpy as np

def measurement_function(x, Ybus, mpc, busphase_map):
    """
    Builds the measurement vector h(x) in the sequence:
       [ P_inj, Q_inj, Pf, Pt, Qf, Qt, Vmag ].

    Where:
      - P_inj, Q_inj : bus-level injections for each phase-node
      - Pf, Pt       : real power flow from- and to- side for each line-phase
      - Qf, Qt       : reactive power flow from- and to- side for each line-phase
      - Vmag         : voltage magnitudes at each phase-node

    Args:
      x : (2*N, ) state vector in polar form => [Vm(0..N-1), Va(0..N-1)]
      Ybus : (N,N) global bus admittance matrix (complex)
      mpc : dict with "line3p", "bus3p", etc.
      busphase_map : dict mapping (bus_id, phase_id) -> node index

    Returns:
      h : 1D numpy array, size = 2*N + 4*(3*n_lines) + N
          => (P_inj, Q_inj) => 2*N
             (Pf, Pt, Qf, Qt) => 4*(3*n_lines)
             (Vmag) => N
    """
    # --- 1) Basic shapes and initial data
    n_lines = len(mpc["line3p"])
    nbus = len(mpc["bus3p"])
    nnodephase = len(busphase_map)

    # Separate the state x into magnitude, angle
    half = nnodephase
    Vm = x[:half]         # voltage magnitudes
    Va = x[half:]         # voltage angles
    # Build complex voltages in polar form
    V = Vm * np.exp(1j * Va)   # shape (nnodephase,)

    # Dimensions of sub-blocks in h
    # P_injection + Q_injection => 2*N
    m_inj = 2*nnodephase
    # Flows => 4*(3*n_lines):
    #   for each line, we have 3 phases => 3 "phase flows" => each flow set is (Pf,Pt,Qf,Qt) => 4
    m_flow = 4 * (3 * n_lines)
    # Vmag => N
    m_v = nnodephase
    m_total = m_inj + m_flow + m_v
    h = np.zeros(m_total, dtype=float)

    # --- 2) Bus injections: P_inj, Q_inj
    # Compute Ibus = Ybus * V  => Nx1
    # Then Sbus[i] = V[i] * conj(Ibus[i]) => bus-level injection for node i
    Ibus = Ybus @ V
    Sbus = V * np.conjugate(Ibus)  # shape (nnodephase,)

    # Fill first 2*N entries => P_inj, Q_inj
    for i in range(nnodephase):
        h[i] = Sbus[i].real                # P_inj(i)
        h[i + nnodephase] = Sbus[i].imag   # Q_inj(i)

    # --- 3) Line flows: Pf, Pt, Qf, Qt for each line-phase
    # offset_flow => start index in h
    offset_flow = 2*nnodephase
    idx_flow = 0    # line-phase pointer

    for line in mpc["line3p"]:
        # line => [line_id, from_bus, to_bus, ...]
        fbus = int(line[1]) - 1   # zero-based
        tbus = int(line[2]) - 1

        # gather the 3 node indices for from-bus, to-bus
        f_idx = [fbus*3, fbus*3 + 1, fbus*3 + 2]
        t_idx = [tbus*3, tbus*3 + 1, tbus*3 + 2]

        # rowcol => the 6 node indices in [fA,fB,fC, tA,tB,tC]
        rowcol = f_idx + t_idx   # total 6
        # Extract the 6x6 sub-block from Ybus
        Ysub = Ybus[np.ix_(rowcol, rowcol)]  # shape (6,6)

        # For each phase alpha in [0,1,2], compute from- and to- flows
        for alpha in [0,1,2]:
            iF = f_idx[alpha]  # global node index
            iT = t_idx[alpha]
            Vf_alpha = V[iF]
            Vt_alpha = V[iT]

            # (a) from-side flow: Ifrom = sum_{xph} Ysub(alpha, xph)*(Vf[xph] - Vt[xph])
            # alpha row => 0..2 => from side block
            Ifrom = 0+0j
            for xph in [0,1,2]:
                # from-phase row => alpha
                # from-phase col => xph => Ysub[alpha, xph]
                # to-phase col => (3 + xph) => Ysub[alpha, 3+xph]
                # Y_ff = Ysub[alpha, xph]      # from-phase alpha row, from-phase xph col
                Y_ft = Ysub[alpha, 3 + xph]  # from-phase alpha row, to-phase xph col
                iFx = f_idx[xph]
                iTx = t_idx[xph]
                Ifrom +=  - Y_ft * (V[iFx] - V[iTx])

            S_from = Vf_alpha * np.conjugate(Ifrom)
            Pf = S_from.real
            Qf = S_from.imag

            # (b) to-side flow: Ito => row=3+alpha
            # Ito = sum_{xph} Ysub(3+alpha, xph)*(Vf[xph] - Vt[xph])
            Ito = 0+0j
            for xph in [0,1,2]:
                Y_tf = Ysub[3+alpha, xph]      # to-phase alpha row, from-phase xph col
                # Y_tt = Ysub[3+alpha, 3+xph]   # to-phase alpha row, to-phase xph col
                iFx = f_idx[xph]
                iTx = t_idx[xph]
                Ito += Y_tf * (V[iFx] -V[iTx])

            S_to = Vt_alpha * np.conjugate(Ito)
            Pt = S_to.real
            Qt = S_to.imag

            # Place them in the final measurement vector: [Pf, Pt, Qf, Qt]
            # offset_flow + 4*idx_flow => Pf
            # +1 => Pt, +2 => Qf, +3 => Qt
            h[offset_flow + idx_flow] = Pf
            h[offset_flow + idx_flow + 3*n_lines] = Pt
            h[offset_flow + idx_flow + 6*n_lines] = Qf
            h[offset_flow + idx_flow + 9*n_lines] = Qt

            idx_flow += 1

    # --- 4) Lastly, add Vmag
    # offset for Vmag is after 2*N + 4*(3*n_lines)
    offset_v = offset_flow + 4*(3*n_lines)
    for i in range(nnodephase):
        h[offset_v + i] = Vm[i]

    return h