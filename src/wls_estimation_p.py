import numpy as np

def run_wls_state_estimation_polar(z, x_init, busphase_map, Ybus, R, mpc, max_iter=20, tol=1e-6):
    """
    DSSE solver using polar coordinates for the state.
    The state vector x = [Vm(0..N-1), Va(0..N-1)] in p.u. and radians.
    Args:
      z : 1D numpy array of measurements (size m)
      x_init : initial guess (2*nnodephase),
               x_init[i]   = V_m,i
               x_init[i+N] = angle_i
      busphase_map : dictionary mapping (bus_id, phase_id) -> node index
      Ybus : NxN complex bus admittance matrix
      R : measurement covariance or weighting matrix, shape (m,m)
      max_iter, tol : iteration settings

    Returns:
      x_est : final estimated state in polar => [V_m(0..N-1), Va(0..N-1)]
      success : boolean
    """
    nbus = len(mpc["bus3p"])
    nnodephase = len(busphase_map)
    nstate = 2 * nnodephase
    x_init = np.zeros(nstate)
    # x_init[nnodephase:] angles in radians, three phases 0, -120, 120 --> 0, -2pi/3, 2pi/3
    for i in range(nbus):
        x_init[3*i] = 1.0
        x_init[3*i+nnodephase] = 0.0
        x_init[3*i+1] = 1.0
        x_init[3*i+nnodephase+1] = np.deg2rad(-120)
        x_init[3*i+2] = 1.0
        x_init[3*i+nnodephase+2] = np.deg2rad(120)
    x_est = x_init.copy()
    W = np.linalg.inv(R)

    def measurement_function(x):
        """
        Build measurement vector h(x) in the same order as z.
        """
        half = nnodephase
        Vm = x[:half]
        Va = x[half:]
        # Build complex voltages in polar
        V = Vm * np.exp(1j*Va)  # shape (N,)
        # Bus injection => S = V * conj(Ybus * V)
        Ibus = Ybus @ V
        Sbus = V * np.conjugate(Ibus)  # Nx1 complex
        # Now build h in the same 3*N layout
        m = 3*nnodephase
        h = np.zeros(m, dtype=float)
        for i in range(nnodephase):
            h[i] = Sbus[i].real
            h[i + nnodephase] = Sbus[i].imag
            h[i + 2*nnodephase] = Vm[i]
        return h

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

    def jacobian(x):
        """
        Analytical partial derivatives in *polar* coords.
        """
        half = nnodephase
        Vm = x[:half]
        Va = x[half:]
        V = Vm * np.exp(1j*Va)  # shape (N,)
        dSbus_dVa, dSbus_dVm = dsbus_polar(Ybus, V, Vm)

        # We'll fill H in shape (3N x 2N)
        m = 3*nnodephase
        H = np.zeros((m, 2*nnodephase), dtype=float)
        for i in range(nnodephase):
            # row P_inj
            rowP = i
            # partial wrt Va => real( dSbus_dVa[i,:] ), stored in columns [N..2N-1], offset by "phase" index
            # but we define columns: first half => d/dVm, second half => d/dVa
            # => So partial w.r.t. Va is in columns [half..2*half]
            # partial w.r.t. Vm is in columns [0..half]
            # => P_inj => real( dSbus_dVa[i,j] ), real(dSbus_dVm[i,j])
            H[rowP, :half] = np.real(dSbus_dVm[i, :])
            H[rowP, half:] = np.real(dSbus_dVa[i, :])

            # row Q_inj
            rowQ = i + nnodephase
            H[rowQ, :half] = np.imag(dSbus_dVm[i, :])
            H[rowQ, half:] = np.imag(dSbus_dVa[i, :])

            # row Vmag
            rowV = i + 2*nnodephase
            H[rowV, i] = 1.0
        return H
    # Start iteration
    success = True
    x_current = x_est.copy()
    for it in range(max_iter):
        hval = measurement_function(x_current)
        Hmat = jacobian(x_current)
        # Delete the columns corresponding to V_ang_{1,2,3} from H as they are set as the slack bus
        Hmat = np.delete(Hmat, [nnodephase, nnodephase+ 1, nnodephase + 2], axis=1)

        r = z - hval
        lhs = Hmat.T @ W @ Hmat
        rhs = Hmat.T @ W @ r
        try:
            dx = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            success = False
            break
        dx0 = np.zeros(2*nnodephase)
        dx0[:nnodephase] = dx[:nnodephase]
        dx0[nnodephase+3:] = dx[nnodephase:]
        x_current += dx0
        if np.max(np.abs(dx))<tol:
            print(f"DSSE polar converged in {it+1} iterations, max dx={np.max(np.abs(dx)):.3e}")
            break
    else:
        success = False
        print("DSSE polar did not converge in max_iter steps.")

    def print_estimation_results(x_est, mpc):
        """
        Prints the 3-phase bus data in a structured format.

        Args:
          x_est : Estimated state vector from the WLS estimation.
          mpc : Dictionary containing bus data.
        """
        nbus = len(mpc["bus3p"])
        nnodephase = len(x_est) // 2
        Vm = x_est[:nnodephase]
        Va = np.rad2deg(x_est[nnodephase:])  # Convert radians to degrees

        print("=" * 80)
        print("|     3-ph Bus Data                                                            |")
        print("=" * 80)
        print("  3-ph            Phase A Voltage    Phase B Voltage    Phase C Voltage")
        print(" Bus ID   Status   (kV)     (deg)     (kV)     (deg)     (kV)     (deg)")
        print("--------  ------  -------  -------   -------  -------   -------  -------")

        for i in range(nbus):
            bus_id = int(mpc["bus3p"][i][0])
            status = int(mpc["bus3p"][i][1])

            # Phase A
            Va_A = Va[3 * i]
            Vm_A = Vm[3 * i] * mpc["bus3p"][i][2]/np.sqrt(3)  # Assuming Vm is in p.u.

            # Phase B
            Va_B = Va[3 * i + 1]
            Vm_B = Vm[3 * i + 1] * mpc["bus3p"][i][2]/np.sqrt(3)

            # Phase C
            Va_C = Va[3 * i + 2]
            Vm_C = Vm[3 * i + 2] * mpc["bus3p"][i][2]/np.sqrt(3)

            print(f"{bus_id:>8}  {status:>6}  {Vm_A:>7.4f}  {Va_A:>7.2f}  ",
                  f"{Vm_B:>7.4f}  {Va_B:>7.2f}  {Vm_C:>7.4f}  {Va_C:>7.2f}")

    print_estimation_results(x_current, mpc)

    # --- Bad Data Processing (Using Residual Covariance)
    # Compute final measurement residual vector
    h_final = measurement_function(x_current)
    r_final = z - h_final

    # Build the Gain matrix from the last iteration: Gain = H^T * W * H
    Gain = Hmat.T @ W @ Hmat
    # Measurement covariance is R = inv(W)
    R_meas = np.linalg.inv(W)
    # Residual covariance: omega = R_meas - H * inv(Gain) * H^T
    omega = R_meas - Hmat @ np.linalg.inv(Gain) @ Hmat.T
    # Normalized residuals
    diag_omega = np.diag(omega)
    norm_resid = np.abs(r_final) / np.sqrt(diag_omega)
    max_norm = np.max(norm_resid)
    idx_max = np.argmax(norm_resid)
    bad_data_threshold = 3.0
    if max_norm > bad_data_threshold:
        print(f"Bad data detected at measurement index {idx_max}: normalized residual = {max_norm:.2f} exceeds threshold {bad_data_threshold}")
        success = False
    else:
        print(f"No bad data detected: max normalized residual = {max_norm:.2f} below threshold {bad_data_threshold}")

    return x_current, success
