import numpy as np
import time
from parse_opendss_file import build_global_y_per_unit
import math

def run_newton_powerflow_3p(mpc, tol=1e-6, max_iter=20):
    """
    Full-Newton 3-phase PF with ratioed wye-wye transformer. Rectangular Coordinate.
    """
    t_start = time.time()
    # 1) Build Y-bus
    # Find the index of different nodes in global Ybus matrix
    bus3p = mpc["bus3p"]
    bus_ids = [int(row[0]) for row in bus3p]
    busphase_map = {}
    idx = 0
    for b in bus_ids:
        for ph in range(3):
            busphase_map[(b, ph)] = idx
            idx += 1
    Ybus, node_order = build_global_y_per_unit(mpc)

    nnodephase = Ybus.shape[0]

    # 2) Identify slack vs unknown node-phases
    slack_bus_ids = [int(row[0]) for row in bus3p if row[1] == 3]
    slack_indices = []
    unknown_indices = []
    for (b, ph), i in busphase_map.items():
        if b in slack_bus_ids:
            slack_indices.append(i)
        else:
            unknown_indices.append(i)

    # 3) Parse net injection (P_inj, Q_inj) in p.u.
    baseMVA = mpc["baseMVA"]
    P_inj = np.zeros(nnodephase)
    Q_inj = np.zeros(nnodephase)

    # dictionaries to accumulate load/generation per bus-phase
    bus_phase_load = {(b,ph):(0.0,0.0) for (b,ph) in busphase_map} # (P, Q)
    bus_phase_gen  = {(b,ph):(0.0,0.0) for (b,ph) in busphase_map}

    # 3.1) Parse loads
    for row in mpc["load3p"]:
        ldid, ldbus, status, PdA, PdB, PdC, QdA, QdB, QdC = row
        if status == 0:
            continue
        P_ph = np.array([PdA, PdB, PdC]) / 1000.0
        Q_ph = np.array([QdA, QdB, QdC]) / 1000.0
        for ph in range(3):
            oldP, oldQ = bus_phase_load[(int(ldbus), ph)]
            bus_phase_load[(int(ldbus), ph)] = (oldP + P_ph[ph], oldQ + Q_ph[ph])

    # 3.2) Parse gens
    for row in mpc["gen3p"]:
        genid, gbus, status, VgA, VgB, VgC, PgA, PgB, PgC, QgA, QgB, QgC = row
        if status == 0:
            continue
        P_ph = np.array([PgA, PgB, PgC]) / 1000.0  # MW
        Q_ph = np.array([QgA, QgB, QgC]) / 1000.0  # MVAR
        for ph in range(3):
            oldP, oldQ = bus_phase_gen[(int(gbus), ph)]
            bus_phase_gen[(int(gbus), ph)] = (oldP + P_ph[ph], oldQ + Q_ph[ph])

    # 3.3) Combine into net injection
    for (b,ph), i in busphase_map.items():
        loadP, loadQ = bus_phase_load[(b, ph)]
        genP, genQ   = bus_phase_gen[(b, ph)]
        P_inj[i] = (genP - loadP)/baseMVA
        Q_inj[i] = (genQ - loadQ)/baseMVA

    # 4) Build initial guess for V in rectangular form
    Vr0 = np.zeros(nnodephase)
    Vi0 = np.zeros(nnodephase)
    for row in bus3p:
        b = int(row[0])
        if row[1] == 3:
            VmA, VmB, VmC = row[3], row[4], row[5]
            VaA, VaB, VaC = row[6], row[7], row[8]
        else:
            VmA, VmB, VmC = 1.0, 1.0, 1.0
            VaA, VaB, VaC = 0.0, math.radians(-120.0), math.radians(120.0)
        # VmA, VmB, VmC = 1, 1, 1
        # VaA, VaB, VaC = np.deg2rad(0), np.deg2rad(-120), np.deg2rad(120)
        iA = busphase_map[(b, 0)]
        Vr0[iA] = VmA*np.cos(VaA)
        Vi0[iA] = VmA*np.sin(VaA)
        iB = busphase_map[(b, 1)]
        Vr0[iB] = VmB*np.cos(VaB)
        Vi0[iB] = VmB*np.sin(VaB)
        iC = busphase_map[(b, 2)]
        Vr0[iC] = VmC*np.cos(VaC)
        Vi0[iC] = VmC*np.sin(VaC)


    # We'll store unknown states in x = [Vr(unknowns), Vi(unknowns)]
    def pack_x(Vr, Vi):
        return np.concatenate([[Vr[i], Vi[i]] for i in unknown_indices])

    def unpack_x(x, Vr, Vi):
        idx = 0
        for i in unknown_indices:
            Vr[i] = x[idx]
            Vi[i] = x[idx+1]
            idx += 2

    x0 = pack_x(Vr0, Vi0)

    def calc_S(Vc, Ybus):
        """
        Compute S_calc = V .* conj(I), where I = Ybus * V
        Vc: complex voltages
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

    def mismatch_and_jacobian_compact(x):
        """
        Builds the mismatch f and the real-valued Jacobian

        f is 2*#unknowns in length (for real and imaginary mismatch).
        J is (2*#unknowns) x (2*#unknowns).
        """
        # 1) Reconstruct full V from x
        Vr = Vr0.copy()
        Vi = Vi0.copy()
        unpack_x(x, Vr, Vi)
        Vc = Vr + 1j*Vi

        # 2) Compute Sbus = Vc * conj(Ybus*Vc)
        S_calc, Ibus = calc_S(Vc, Ybus)

        # 3) Mismatch only for PQ nodes:
        #    f(2*i)   = real(S_calc[i]) - P_inj[i]
        #    f(2*i+1) = imag(S_calc[i]) - Q_inj[i]
        f_list = []
        for i in unknown_indices:
            f_list.append(S_calc[i].real - P_inj[i])  # dP
            f_list.append(S_calc[i].imag - Q_inj[i])  # dQ
        f = np.array(f_list)

        # 4) Build partial derivatives: dS_dVr, dS_dVi in complex NxN
        dS_dVr, dS_dVi = calc_dS_dVrVi(Ybus, Vc, Ibus)

        # We'll form a sub-Jacobian for unknown i, j only.
        # If we number unknown nodes as i_u, j_u in [0..(nU-1)],
        # then row for i_u => 2*i_u or 2*i_u+1
        # col for j_u => 2*j_u or 2*j_u+1
        nU = len(unknown_indices)
        J = np.zeros((2*nU, 2*nU), dtype=float)

        # fill J
        for r_idx, i_node in enumerate(unknown_indices):
            # row index for dP => 2*r_idx, dQ => 2*r_idx+1
            rowP = 2*r_idx
            rowQ = 2*r_idx + 1

            for c_idx, j_node in enumerate(unknown_indices):
                colVr = 2*c_idx
                colVi = 2*c_idx + 1

                dS_dVr_ij = dS_dVr[i_node, j_node]
                dS_dVi_ij = dS_dVi[i_node, j_node]

                J[rowP, colVr] = dS_dVr_ij.real
                J[rowQ, colVr] = dS_dVr_ij.imag
                J[rowP, colVi] = dS_dVi_ij.real
                J[rowQ, colVi] = dS_dVi_ij.imag

        return f, J

    # 5) Newton iteration with iteration table
    print(" it    max residual        max Δx")
    print("----  --------------  --------------")

    x_est = x0.copy()
    f, _ = mismatch_and_jacobian_compact(x_est)
    max_res = np.max(np.abs(f))
    print(f"  0      {max_res:1.3e}           -")

    for it in range(1, max_iter+1):
        f, J = mismatch_and_jacobian_compact(x_est)
        max_res = np.max(np.abs(f))
        if max_res < tol:
            break
        dx = np.linalg.solve(J, -f)
        max_dx = np.max(np.abs(dx))
        x_est += dx
        print(f" {it:2d}      {max_res:1.3e}       {max_dx:1.3e}")

    # final check
    f, _ = mismatch_and_jacobian_compact(x_est)
    max_res = np.max(np.abs(f))
    max_dx = np.max(np.abs(dx))
    if max_res < tol:
        print(f"Newton's method converged in {it} iterations.\nPF successful\n")
    else:
        print(f"Warning: Did not converge in {it} iterations (res={max_res:1.3e}).\n")

    t_elapsed = time.time() - t_start
    print(f"PF succeeded in {t_elapsed:.2f} seconds\n")

    # 6) Unpack final solution
    Vr_final = Vr0.copy()
    Vi_final = Vi0.copy()
    unpack_x(x_est, Vr_final, Vi_final)

    # # 7) Summarize the measurement function
    # def generate_measurements(Vr, Vi, Ybus):
    #     """
    #     Generate measurement data (z) from final power flow results.
    #     We can measure:
    #       1. Bus voltage magnitudes,
    #       2. Bus injection P, Q,
    #       3. Line flow P, Q,
    #       etc.
    #     Returns:
    #       z (np.array) : measurement vector
    #       measurement_info : info about each measurement (indices, types, etc.)
    #     """
    #     Vmag = np.sqrt(Vr**2 + Vi**2)
    #     z_list = []
    #     measurement_info = []
    #     Vc = Vr + 1j*Vi
    #     S_calc, _ = calc_S(Vc, Ybus)
    #     for i in range(nnodephase):
    #         z_list.append(S_calc[i].real)
    #         measurement_info.append(("P_inj", i))
    #     for i in range(nnodephase):
    #         z_list.append(S_calc[i].imag)
    #         measurement_info.append(("Q_inj", i+nnodephase))
    #     for i in range(nnodephase):
    #         z_list.append(Vmag[i])
    #         measurement_info.append(("Vmag", i+2*nnodephase))
    #     z = np.array(z_list)
    #     return z, measurement_info
    #
    # z, measurement_info = generate_measurements(Vr_final, Vi_final, Ybus)

    return Vr_final, Vi_final, busphase_map