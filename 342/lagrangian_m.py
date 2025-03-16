import numpy as np
from utilities.mea_fun import measurement_function
from utilities.make_jacobian import jacobian
from utilities.make_jacobian_p import jacobian_line_params

def find_zero_injection_nodes(mpc, busphase_map, tol=1.0):
    """
    Identify phase-level nodes (bus-phase) that have effectively zero net injection.

    Args:
        mpc : dict containing "bus3p", "load3p", "gen3p" data, etc.
        busphase_map : dict mapping (bus_id, phase_id) -> global node index
        tol : float, absolute threshold for net P,Q to consider zero injection (e.g. 1.0 kW)

    Returns:
        zero_inj_nodes : list of node indices (0-based) in the global 3-phase node ordering
                         that have net injection ~ 0.
    """
    # 1) Extract the relevant arrays
    bus3p  = mpc["bus3p"]   # columns: [bus_id, type, base_kV_LL, VmA, VmB, VmC, VaA, VaB, VaC]
    load3p = mpc["load3p"]  # [ldid, ldbus, status, PdA, PdB, PdC, pfA, pfB, pfC]
    gen3p  = mpc["gen3p"]   # [genid, gbus, status, VgA, VgB, VgC, PgA, PgB, PgC, QgA, QgB, QgC]

    # 2) Gather a list of bus IDs
    bus_ids = [int(row[0]) for row in bus3p]
    slack_id = [int(row[0]) for row in bus3p if row[1] == 3][0]

    # 3) Prepare storage for net injections at each (bus, phase)
    #    netPQ_phase[(bus, ph)] = [P_net(kW), Q_net(kVar)]
    netPQ_phase = {}
    for b in bus_ids:
        for ph in [0,1,2]:   # ph=0=>A, 1=>B, 2=>C (arbitrary naming)
            netPQ_phase[(b, ph)] = [0.0, 0.0]

    # 4) Process loads (subtract from net injection)
    #    load3p => [ldid, ldbus, status, PdA, PdB, PdC, pfA, pfB, pfC]
    for row in load3p:
        ldid, ldbus, status, PdA, PdB, PdC, QdA, QdB, QdC = row
        if status == 0:
            continue  # skip inactive loads
        bus_id = int(ldbus)
        # Phase A portion
        if PdA > 1e-9:  # Some small threshold
            # QdA = PdA * np.sqrt(1/(pfA**2) - 1) if pfA < 1.0 else 0.0
            # Subtract from net injection
            netPQ_phase[(bus_id, 0)][0] -= -PdA
            netPQ_phase[(bus_id, 0)][1] -= -QdA

        # Phase B
        if PdB > 1e-9:
            netPQ_phase[(bus_id, 1)][0] -= -PdB
            netPQ_phase[(bus_id, 1)][1] -= -QdB

        # Phase C
        if PdC > 1e-9:
            netPQ_phase[(bus_id, 2)][0] -= -PdC
            netPQ_phase[(bus_id, 2)][1] -= -QdC

    # 5) Process generations (add to net injection)
    #    gen3p => [genid, gbus, status, VgA, VgB, VgC, PgA, PgB, PgC, QgA, QgB, QgC]
    # for row in gen3p:
    #     genid, gbus, status, VgA, VgB, VgC, PgA, PgB, PgC, QgA, QgB, QgC = row
    #     if status == 0:
    #         continue  # skip inactive gens
    #
    #     bus_id = int(gbus)
    #     # Phase A
    #     if PgA > 1e-9 or abs(QgA) > 1e-9:
    #         netPQ_phase[(bus_id, 0)][0] += PgA
    #         netPQ_phase[(bus_id, 0)][1] += QgA
    #     # Phase B
    #     if PgB > 1e-9 or abs(QgB) > 1e-9:
    #         netPQ_phase[(bus_id, 1)][0] += PgB
    #         netPQ_phase[(bus_id, 1)][1] += QgB
    #     # Phase C
    #     if PgC > 1e-9 or abs(QgC) > 1e-9:
    #         netPQ_phase[(bus_id, 2)][0] += PgC
    #         netPQ_phase[(bus_id, 2)][1] += QgC

    # 6) Determine zero-injection nodes
    zero_inj_nodes = []
    for (b, ph) in netPQ_phase.keys():
        if b == slack_id:
            continue  # skip slack bus
        p_net, q_net = netPQ_phase[(b, ph)]
        # if near zero => mark node as zero injection
        if (abs(p_net) < tol) and (abs(q_net) < tol):
            # Convert (bus, phase) to global node index
            # e.g. busphase_map[(bus_id, 0..2)]
            node_idx = busphase_map[(b, ph)]
            zero_inj_nodes.append(node_idx)

    return zero_inj_nodes

def run_lagrangian_polar(z, x_init, busphase_map, Ybus, R, mpc, max_iter=20, tol=1e-6):
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
    # Number of parameters that needs to be monitored, assume only X and R for now
    fbus_self = []
    tbus_self = []
    fbus_mut  = []
    tbus_mut  = []
    for line in mpc["line3p"]:
        from_bus = int(line[1]) - 1  # convert to 0-based
        to_bus   = int(line[2]) - 1

        # self (phases 0..2 => A,B,C)
        for phase in range(3):
            fbus_self.append(from_bus*3 + phase)
            tbus_self.append(to_bus  *3 + phase)

        # mutual: ab=(0,1), bc=(1,2), ac=(0,2)
        fbus_mut.append(from_bus*3 + 0)
        tbus_mut.append(from_bus*3 + 1)

        fbus_mut.append(from_bus*3 + 1)
        tbus_mut.append(from_bus*3 + 2)

        fbus_mut.append(from_bus*3 + 0)
        tbus_mut.append(from_bus*3 + 2)

    fbus_self = np.array(fbus_self, dtype=int)
    tbus_self = np.array(tbus_self, dtype=int)
    fbus_mut  = np.array(fbus_mut,  dtype=int)
    tbus_mut  = np.array(tbus_mut,  dtype=int)

    # Combine into one index set
    fbus = np.concatenate([fbus_self, fbus_mut])
    tbus = np.concatenate([tbus_self, tbus_mut])
    npara_x = len(fbus)  # total X-parameters for all lines
    npara   = 2 * npara_x  # double to account for R-parameters
    # npara = len(mpc["line3p"]) * 3
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
    # Find the zero injection buses
    zi_nodes = find_zero_injection_nodes(mpc, busphase_map, tol=1.0)
    indices_to_remove = []
    for i_node in zi_nodes:
        # i_node is the node index in [0..nnodephase-1] for that phase
        # P row => i_node
        # Q row => i_node + nnodephase
        indices_to_remove.append(i_node)
        indices_to_remove.append(i_node + nnodephase)
    # Remove the zero injection buses from the W matrix
    W = np.delete(W, indices_to_remove, axis=0)
    W = np.delete(W, indices_to_remove, axis=1)
    # Remove the zero injection buses from the z vector
    z = np.delete(z, indices_to_remove)

    xl = np.zeros(2*nnodephase - 3 + 3 * nnodephase, dtype=float)
    for i in range(nnodephase):
        xl[3*i] = 1
        xl[3*i + nnodephase] = 0
        xl[3*i + 1] = 1
        xl[3*i + nnodephase + 1] = np.deg2rad(-120)
        xl[3*i + 2] = 1
        xl[3*i + nnodephase + 2] = np.deg2rad(120)
    ########################################################
    # Auxiliary functions
    # def measurement_function(x):
    #     """
    #     Build measurement vector h(x) in the same order as z.
    #     """
    #     half = nnodephase
    #     Vm = x[:half]
    #     Va = x[half:]
    #     # Build complex voltages in polar
    #     V = Vm * np.exp(1j*Va)  # shape (N,)
    #     # Bus injection => S = V * conj(Ybus * V)
    #     Ibus = Ybus @ V
    #     Sbus = V * np.conjugate(Ibus)  # Nx1 complex
    #     # Now build h in the same 3*N layout
    #     m = 3*nnodephase
    #     h = np.zeros(m, dtype=float)
    #     for i in range(nnodephase):
    #         h[i] = Sbus[i].real
    #         h[i + nnodephase] = Sbus[i].imag
    #         h[i + 2*nnodephase] = Vm[i]
    # #     return h
    #
    # def dsbus_polar(Ybus, V, Vm):
    #     """
    #     Returns (dSbus_dVa, dSbus_dVm) for polar coords.
    #       Ibus = Ybus*V
    #       Sbus = diag(V) * conj(Ibus)
    #     Then in polar coords:
    #       dSbus/dVa = j * diag(V) * conj( Ibus - Ybus*diag(V) )
    #       dSbus/dVm = diag(V) * conj( Ybus * diag(V./abs(V)) ) + conj(diag(Ibus)) * diag(V./abs(V))
    #     Because we store V[i] = Vm[i]* exp(j Va[i]), and abs(V[i])=Vm[i].
    #     """
    #     Ibus = Ybus @ V
    #     diagV = np.diag(V)
    #     diagI = np.diag(Ibus)
    #     # dSbus_dVa = j * diag(V) * conj( diag(Ibus) - Ybus*diag(V) )
    #     # dSbus_dVm = diag(V) * conj( Ybus* diag(Vnorm) ) + conj(diag(Ibus)) * diag(Vnorm)
    #     #  where Vnorm = V/abs(V)
    #     Vnorm = V / Vm
    #     # partial w.r.t. Va
    #     term = diagI - (Ybus @ np.diag(V))
    #     dSbus_dVa = 1j * diagV @ np.conjugate(term)
    #
    #     # partial w.r.t. Vm
    #     tmp1 = diagV @ np.conjugate(Ybus @ np.diag(Vnorm))
    #     tmp2 = np.conjugate(diagI) @ np.diag(Vnorm)
    #     dSbus_dVm = tmp1 + tmp2
    #
    #     return dSbus_dVa, dSbus_dVm
    #
    # def jacobian(x):
    #     """
    #     Analytical partial derivatives in *polar* coords.
    #     """
    #     half = nnodephase
    #     Vm = x[:half]
    #     Va = x[half:]
    #     V = Vm * np.exp(1j*Va)  # shape (N,)
    #     dSbus_dVa, dSbus_dVm = dsbus_polar(Ybus, V, Vm)
    #
    #     # We'll fill H in shape (3N x 2N)
    #     m = 3*nnodephase
    #     H = np.zeros((m, 2*nnodephase), dtype=float)
    #     for i in range(nnodephase):
    #         # row P_inj
    #         rowP = i
    #         # partial wrt Va => real( dSbus_dVa[i,:] ), stored in columns [N..2N-1], offset by "phase" index
    #         # but we define columns: first half => d/dVm, second half => d/dVa
    #         # => So partial w.r.t. Va is in columns [half..2*half]
    #         # partial w.r.t. Vm is in columns [0..half]
    #         # => P_inj => real( dSbus_dVa[i,j] ), real(dSbus_dVm[i,j])
    #         H[rowP, :half] = np.real(dSbus_dVm[i, :])
    #         H[rowP, half:] = np.real(dSbus_dVa[i, :])
    #
    #         # row Q_inj
    #         rowQ = i + nnodephase
    #         H[rowQ, :half] = np.imag(dSbus_dVm[i, :])
    #         H[rowQ, half:] = np.imag(dSbus_dVa[i, :])
    #
    #         # row Vmag
    #         rowV = i + 2*nnodephase
    #         H[rowV, i] = 1.0
    #     return H
    def print_estimation_results(x_est, mpc):
        """
        Prints the 3-phase bus data in a structured format.
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
    ########################################################
    # Start iteration
    success = True
    x_current = x_est.copy()
    for it in range(max_iter):
        hval = measurement_function(x_current, Ybus, mpc, busphase_map)
        print("Forming Jacobian")
        Hmat = jacobian(x_current, Ybus, mpc, busphase_map)
        print("Forming Jacobian Finished.")
        # Delete the columns corresponding to V_ang_{1,2,3} from H as they are set as the slack bus
        Hmat = np.delete(Hmat, [nnodephase, nnodephase+ 1, nnodephase + 2], axis=1)
        C = Hmat[indices_to_remove, :]
        Hmat = np.delete(Hmat, indices_to_remove, axis=0)
        cval = hval[indices_to_remove]
        hval = np.delete(hval, indices_to_remove)
        # Form the coefficient matrix
        #     Gain = [zeros(2*nnodephase-3), H'*W, C';
        #         H, eye(size(H, 1)), zeros(size(H, 1), size(C, 1));
        #         C, zeros(size(C, 1), size(H, 1)), zeros(size(C, 1))];
        # Form the Gain matrix
        nH, nX = Hmat.shape  # Dimensions of H
        nC = C.shape[0]  # Number of constraints

        # Define submatrices
        zero_block = np.zeros((2 * nnodephase - 3, 2 * nnodephase - 3))  # Zero matrix
        H_W = Hmat.T @ W  # H' * W
        identity_H = np.eye(nH)  # Identity matrix same size as H

        # Construct full Gain matrix
        Gain = np.block([
            [zero_block, H_W, C.T],  # First row
            [Hmat, identity_H, np.zeros((nH, nC))],  # Second row
            [C, np.zeros((nC, nH)), np.zeros((nC, nC))]  # Third row
        ])

        r = z - hval
        # r = [zeros(2*nb-1, 1); r; -cx];
        r = np.concatenate((np.zeros(2*nnodephase-3), r, -cval))
        # Solve the linear system
        lhs = Gain
        rhs = r
        try:
            dxl = np.linalg.solve(lhs, rhs)
            dx = dxl[:2*nnodephase-3]
        except np.linalg.LinAlgError:
            success = False
            break
        dx0 = np.zeros(2*nnodephase)
        dx0[:nnodephase] = dx[:nnodephase]
        dx0[nnodephase+3:] = dx[nnodephase:]
        x_current += dx0
        print_estimation_results(x_current, mpc)
        # Print the max dx
        print(f"DSSE polar iter {it+1}: max dx={np.max(np.abs(dx)):.3e}")
        if np.max(np.abs(dx))<tol:
            print(f"DSSE polar converged in {it+1} iterations, max dx={np.max(np.abs(dx)):.3e}")
            break
    else:
        success = False
        print("DSSE polar did not converge in max_iter steps.")

    print_estimation_results(x_current, mpc)

    lambdaN = 0
    # # Lagrangian multiplier calculation
    # # V, theta: the mag and angle for all nodes;
    # theta = x_current[nnodephase:]
    # V = x_current[:nnodephase]
    # Hp = jacobian_line_params(x_current, Ybus, mpc, busphase_map)
    # # Hp = np.zeros((3 * nnodephase, npara)) # N_mea * N_para
    # # for k in range(npara_x):
    # #     i = fbus[k]
    # #     j = tbus[k]
    # #
    # #     # Pinj w.r.t parameter k (reactance)
    # #     Hp[i, k] = - V[i] * V[j] * np.sin(theta[i] - theta[j])
    # #     Hp[j, k] = - V[i] * V[j] * np.sin(theta[j] - theta[i])
    # #     # Qinj w.r.t parameter k (reactance)
    # #     Hp[i + nnodephase, k] = -(V[i] ** 2) + V[i] * V[j] * np.cos(theta[i] - theta[j])
    # #     Hp[j + nnodephase, k] = -(V[j] ** 2) + V[i] * V[j] * np.cos(theta[j] - theta[i])
    # #
    # #     # Pinj w.r.t parameter k + npara_x (conductance)
    # #     Hp[i, k + npara_x] = V[i] ** 2 - V[i] * V[j] * np.cos(theta[i] - theta[j])
    # #     Hp[j, k + npara_x] = V[j] ** 2 - V[i] * V[j] * np.cos(theta[j] - theta[i])
    # #     # Qinj w.r.t parameter k + npara_x (conductance)
    # #     Hp[i + nnodephase, k + npara_x] = - V[i] * V[j] * np.sin(theta[i] - theta[j])
    # #     Hp[j + nnodephase, k + npara_x] = - V[i] * V[j] * np.sin(theta[j] - theta[i])
    #
    # Cp = Hp[indices_to_remove, :]
    # Hp = np.delete(Hp, indices_to_remove, axis=0)
    # S = -np.vstack((W @ Hp, Cp)).T
    # # Compute the Lagrangian multipliers
    # temp = np.linalg.inv(Gain)
    # # Determine a, b, c from the shapes of submatrices
    # a = zero_block.shape[0]  # zero_block is (a × a)
    # b = identity_H.shape[0]  # identity_H is (b × b)
    # c = C.shape[0]  # C is (c × a)
    # E5 = temp[a: a + b, a: a + b]
    # E8 = temp[a + b: a + b + c, a: a + b]
    #
    # phi = np.vstack((E5, E8))
    #
    # covu = phi @ np.linalg.inv(W) @ phi.T
    # ea = S @ covu @ S.T
    #
    # lambda_vec = S @ dxl[2 * nnodephase - 3:]
    #
    # tt = np.sqrt(np.diag(ea))
    #
    # lambdaN = lambda_vec / tt
    #
    # # --- Bad Data Processing (Using Residual Covariance)
    # # Compute final measurement residual vector
    # # h_final = measurement_function(x_current)
    # # r_final = z - h_final
    # #
    # # # Build the Gain matrix from the last iteration: Gain = H^T * W * H
    # # Gain = Hmat.T @ W @ Hmat
    # # # Measurement covariance is R = inv(W)
    # # R_meas = np.linalg.inv(W)
    # # # Residual covariance: omega = R_meas - H * inv(Gain) * H^T
    # # omega = R_meas - Hmat @ np.linalg.inv(Gain) @ Hmat.T
    # # # Normalized residuals
    # # diag_omega = np.diag(omega)
    # # norm_resid = np.abs(r_final) / np.sqrt(diag_omega)
    # # max_norm = np.max(norm_resid)
    # # idx_max = np.argmax(norm_resid)
    # # bad_data_threshold = 3.0
    # # if max_norm > bad_data_threshold:
    # #     print(f"Bad data detected at measurement index {idx_max}: normalized residual = {max_norm:.2f} exceeds threshold {bad_data_threshold}")
    # #     success = False
    # # else:
    # #     print(f"No bad data detected: max normalized residual = {max_norm:.2f} below threshold {bad_data_threshold}")

    return x_current, success, lambdaN
