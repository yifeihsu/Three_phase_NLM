import numpy as np

def run_wls_state_estimation(z, x_f, busphase_map, Ybus, R, mpc):
    """
    Args:
       z : 1D np.array of measurements
       meas_info : list of tuples describing each measurement (type, index, etc.)
       busphase_map : dictionary used in your PF code
    Returns:
       x_est : final estimated state (e.g. [Vr, Vi] for all nodes)
       success : boolean indicating if converged
    """
    # 1) Build initial guess for the state vector x
    #    For example, if we have nb buses * 3 phases => nnodephase
    #    Each node might have Vr, Vi => 2*nnodephase states
    # Use a flat start.
    nnodephase = len(busphase_map)
    nstate = 2 * nnodephase
    # 4) Build initial guess for V in rectangular form
    Vr0 = np.zeros(nnodephase)
    Vi0 = np.zeros(nnodephase)

    for row in mpc["bus3p"]:
        b = int(row[0])
        VmA, VmB, VmC = row[3], row[4], row[5]
        VaA, VaB, VaC = np.deg2rad(row[6]), np.deg2rad(row[7]), np.deg2rad(row[8])
        iA = busphase_map[(b, 0)]
        Vr0[iA] = VmA*np.cos(VaA)
        Vi0[iA] = VmA*np.sin(VaA)
        iB = busphase_map[(b, 1)]
        Vr0[iB] = VmB*np.cos(VaB)
        Vi0[iB] = VmB*np.sin(VaB)
        iC = busphase_map[(b, 2)]
        Vr0[iC] = VmC*np.cos(VaC)
        Vi0[iC] = VmC*np.sin(VaC)
    x_est = np.concatenate([Vr0, Vi0])
    x_est = np.concatenate([x_f.real, x_f.imag])

    def calc_S(Vc, Ybus):
        """
        Compute S_calc = V .* conj(I), where I = Ybus * V
        Vc is an array of complex voltages
        Returns S_calc, I
        """
        Ibus = Ybus @ Vc
        Sbus = Vc * np.conjugate(Ibus)  # elementwise
        return Sbus, Ibus

    def calc_dS_dVrVi(Ybus, Vc, Ibus):
        """
        Returns the NxN complex partial derivatives in the cartesian form:
          dS_dVr, dS_dVi,
        each NxN complex, where (i,j) = dS_i/dVr_j or dS_i/dVi_j.
        """
        # diag(V) and conj(Ybus) can be formed with
        n = len(Vc)
        diagV = np.diag(Vc)
        diagIbus = np.diag(Ibus)

        # Because Ybus is NxN complex, conj(Ybus) is just Ybus.conjugate()
        # dSbus_dVr = conj(diag(Ibus)) + diag(V)* conj(Ybus)
        dS_dVr = np.conjugate(diagIbus) + diagV @ np.conjugate(Ybus)

        # dSbus_dVi = j * ( conj(diag(Ibus)) - diag(V)* conj(Ybus) )
        dS_dVi = 1j * (np.conjugate(diagIbus) - diagV @ np.conjugate(Ybus))

        return dS_dVr, dS_dVi
    # 2) Build measurement function h(x), Jacobian H(x), etc.
    def measurement_function(x):
        """
        Compute expected measurement vector from state x.
        x: [Vr1, Vr2,..., VrN, Vi1, Vi2,..., ViN].
        """
        half = nnodephase
        Vr = x[:half]
        Vi = x[half:]
        Vc = Vr + 1j * Vi
        S_calc, _ = calc_S(Vc, Ybus)
        h_list = np.zeros(3 * nnodephase, dtype=float)
        for i in range(nnodephase):
            # P_inj row
            h_list[i] = S_calc[i].real
            # Q_inj row
            h_list[i + nnodephase] = S_calc[i].imag
            # Vmag row
            vm = Vr[i] ** 2 + Vi[i] ** 2
            h_list[i + 2 * nnodephase] = vm
        return np.array(h_list)

    def jacobian(x):
        """
        return the estimated measurement vector and the Jacobian matrix H
        """
        f0 = measurement_function(x)
        half = nnodephase
        Vr = x[:half]
        Vi = x[half:]
        Vc = Vr + 1j * Vi
        S_calc, Ibus = calc_S(Vc, Ybus)
        dS_dVr, dS_dVi = calc_dS_dVrVi(Ybus, Vc, Ibus)

        m = 3 * nnodephase
        H = np.zeros((m, 2 * nnodephase), dtype=float)

        for i in range(nnodephase):
            # partial of P_inj(i)
            rowP = i
            # partial wrt Vr => real(dS_dVr[i,:])
            H[rowP, :half] = np.real(dS_dVr[i, :])
            H[rowP, half:] = np.real(dS_dVi[i, :])

            # partial of Q_inj(i)
            rowQ = i + nnodephase
            H[rowQ, :half] = np.imag(dS_dVr[i, :])
            H[rowQ, half:] = np.imag(dS_dVi[i, :])

            # partial of Vmag(i) = sqrt(Vr[i]^2 + Vi[i]^2)
            rowV = i + 2 * nnodephase
            denom = np.sqrt(Vr[i] ** 2 + Vi[i] ** 2)
            if denom > 1e-9:
                # partial wrt Vr[i]
                H[rowV, i] = 2 * Vr[i]
                # partial wrt Vi[i]
                H[rowV, i + half] = Vi[i] * 2
            else:
                # if near zero magnitude, partial is ill-defined
                # you could set them to 0 or handle gracefully
                H[rowV, i] = 0.0
                H[rowV, i + half] = 0.0
        return f0, H

    def numeric_jacobian(x, measurement_function, eps=1e-6):
        """
        Computes a numeric (finite-difference) Jacobian of the measurement_function
        with respect to the state x.

        Args:
          x : 1D numpy array, shape (nstate,)
          measurement_function : a function f(x) -> 1D numpy array, shape (m,)
          eps : float, the small perturbation step

        Returns:
          f0 : the measurement_function evaluated at x  (shape (m,))
          J  : the numeric Jacobian matrix of shape (m, nstate), i.e.
               J[i,j] = ( f_pert[i] - f0[i] ) / eps
        """
        # Evaluate function at base point
        f0 = measurement_function(x)
        m = len(f0)
        n = len(x)
        J = np.zeros((m, n), dtype=float)

        # Finite difference each coordinate of x
        for j in range(n):
            x_pert = x.copy()
            x_pert[j] += eps
            f_pert = measurement_function(x_pert)
            J[:, j] = (f_pert - f0) / eps

        return f0, J

    # 3) Weighted Least Squares iteration
    max_iter = 20
    tol = 1e-4
    success = True

    for it in range(max_iter):
        h, H = jacobian(x_est)
        f0, J = numeric_jacobian(x_est, measurement_function)
        r = z - h  # residual
        W = np.linalg.inv(R)
        # Gauss-Newton step
        # J'J x = J'r
        JT = H.T
        G = JT @ W @ H
        RHS = JT @ W @ r
        try:
            dx = np.linalg.solve(G, RHS)
        except np.linalg.LinAlgError:
            success = False
            break
        x_est += dx
        if np.max(np.abs(dx)) < tol:
            break
    else:
        success = False

    return x_est, success