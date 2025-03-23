import numpy as np
from utilities.mea_fun import measurement_function
from utilities.make_jacobian import jacobian
from utilities.make_jacobian_p import jacobian_line_params
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

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
            netPQ_phase[(bus_id, 0)][0] += PdA
            netPQ_phase[(bus_id, 0)][1] += QdA

        # Phase B
        if PdB > 1e-9:
            netPQ_phase[(bus_id, 1)][0] += PdB
            netPQ_phase[(bus_id, 1)][1] += QdB

        # Phase C
        if PdC > 1e-9:
            netPQ_phase[(bus_id, 2)][0] += PdC
            netPQ_phase[(bus_id, 2)][1] += QdC

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
            continue  # skip slack bus, we are loading original data rather than PF results
        p_net, q_net = netPQ_phase[(b, ph)]
        # Avoid floating point errors
        if (abs(p_net) < tol) and (abs(q_net) < tol):
            node_idx = busphase_map[(b, ph)]
            zero_inj_nodes.append(node_idx)
    return zero_inj_nodes

def run_lagrangian_polar(z, x_init, busphase_map, Ybus, R, mpc, max_iter=20, tol=1e-6):
    """
    DSSE solver using polar coordinates.
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
    ########################################################
    # Data Preparation
    ########################################################
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
    W = np.delete(W, indices_to_remove, axis=0)
    W = np.delete(W, indices_to_remove, axis=1)
    W_sp = sps.csr_matrix(W)
    z = np.delete(z, indices_to_remove)
    # Assuming only one slack bus (3-phases setting)
    xl = np.zeros(2*nnodephase - 3 + 3 * nnodephase, dtype=float)
    for i in range(nnodephase):
        xl[3*i] = 1
        xl[3*i + nnodephase] = 0
        xl[3*i + 1] = 1
        xl[3*i + nnodephase + 1] = np.deg2rad(-120)
        xl[3*i + 2] = 1
        xl[3*i + nnodephase + 2] = np.deg2rad(120)

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
    ########################################################
    success = True
    x_current = x_est.copy()
    for it in range(max_iter):
        # 1) Compute the measurement function and its Jacobian
        hval = measurement_function(x_current, Ybus, mpc, busphase_map)
        print("Start forming Jacobian")
        Hmat = jacobian(x_current, Ybus, mpc, busphase_map)
        print("Jacobian formed")

        # 2) Remove slack-bus angle columns (3 angles for 3-phase slack)
        slack_cols = [nnodephase, nnodephase + 1, nnodephase + 2]
        all_cols = np.arange(Hmat.shape[1])
        keep_cols = np.delete(all_cols, slack_cols)
        Hmat = Hmat[:, keep_cols]  # keep all rows, subset columns

        # 3) Remove zero-injection rows
        #    We'll first gather them in "C" for constraints:
        C = Hmat[indices_to_remove, :]

        #    Then keep the rest in Hmat
        all_rows = np.arange(Hmat.shape[0])
        keep_rows = np.delete(all_rows, indices_to_remove)
        Hmat = Hmat[keep_rows, :]

        # 4) Now Hmat and C are sparse. If you want them in CSR format:
        H_sp = Hmat.tocsr()  # ensure it's still CSR after slicing
        C_sp = C.tocsr()

        # 2) Remove columns for the slack bus angles (example: angles # nnodephase, nnodephase+1, nnodephase+2)
        cval = hval[indices_to_remove]
        hval = np.delete(hval, indices_to_remove)

        # 4) Form residual vector r = [zeros_for_lagrange; (z - h); -cval]
        r_meas = z - hval
        r = np.concatenate(( np.zeros(2*nnodephase - 3), r_meas, -cval ))

        # ---------------------------------------------------------------------
        #   SPARSE CHANGES HERE
        # ---------------------------------------------------------------------
        # Convert Hmat, C into sparse. "csr_matrix" is usually a good default.

        # We'll need an identity matrix matching Hmat's row count:
        nH = H_sp.shape[0]
        I_H_sp = sps.eye(nH, format='csr')

        # Make a zero block for the top-left corner: shape (2*nnodephase-3, 2*nnodephase-3)
        size_top = 2*nnodephase - 3
        zero_block_sp = sps.csr_matrix((size_top, size_top))

        # "H_W" = H' * W, but we need W in sparse as well
        # If W_sp is your weighting matrix, do:
        H_W_sp = H_sp.transpose().dot(W_sp)

        # For the third block dimension:
        nC = C_sp.shape[0]

        # Next we form the big "Gain" matrix in block-sparse form:
        #      [  0          H'W         C'
        #        H           I          0
        #        C           0          0  ]
        #
        # Use "scipy.sparse.bmat" for block assembly:
        Gain = sps.bmat([
            [ zero_block_sp, H_W_sp,     C_sp.transpose() ],
            [ H_sp,          I_H_sp,     None             ],
            [ C_sp,          None,       None             ]
        ], format='csr')

        # 5) Solve the linear system "Gain * dxl = r" via a sparse solver:
        try:
            print("Start solving the linear system")
            dxl = spsolve(Gain, r)  # dxl is 1D numpy array
            print("Linear system solved")
        except sps.linalg.MatrixRankWarning:
            # or sps.linalg.ArpackNoConvergence, or catch linalg.LinAlgError, etc.
            success = False
            print("Sparse solver encountered a rank or convergence issue.")
            break

        # 6) Extract dx from dxl
        # The first part of dxl corresponds to the "2*nnodephase - 3" states,
        # which we then map back into dx0 with zeros for the slack bus angles
        dx = dxl[:size_top]
        dx0 = np.zeros(2*nnodephase)
        # put dx into the correct spots for magnitude and angle (excluding the 3 slack angles)
        dx0[:nnodephase] = dx[:nnodephase]
        dx0[nnodephase+3:] = dx[nnodephase:]

        # Update x_current
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
    # For debug purpose
    lambdaN = 0
    # Lagrangian multiplier calculation
    # Hp = jacobian_line_params(x_current, Ybus, mpc, busphase_map)
    # Cp = Hp[indices_to_remove, :]
    # Hp = np.delete(Hp, indices_to_remove, axis=0)
    # S = -np.vstack((W @ Hp, Cp)).T
    # Compute the Lagrangian multipliers
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

    # --- Bad Data Processing (Using Residual Covariance)
    # Compute final measurement residual vector
    # h_final = measurement_function(x_current)
    # r_final = z - h_final
    #
    # # Build the Gain matrix from the last iteration: Gain = H^T * W * H
    # Gain = Hmat.T @ W @ Hmat
    # # Measurement covariance is R = inv(W)
    # R_meas = np.linalg.inv(W)
    # # Residual covariance: omega = R_meas - H * inv(Gain) * H^T
    # omega = R_meas - Hmat @ np.linalg.inv(Gain) @ Hmat.T
    # # Normalized residuals
    # diag_omega = np.diag(omega)
    # norm_resid = np.abs(r_final) / np.sqrt(diag_omega)
    # max_norm = np.max(norm_resid)
    # idx_max = np.argmax(norm_resid)
    # bad_data_threshold = 3.0
    # if max_norm > bad_data_threshold:
    #     print(f"Bad data detected at measurement index {idx_max}: normalized residual = {max_norm:.2f} exceeds threshold {bad_data_threshold}")
    #     success = False
    # else:
    #     print(f"No bad data detected: max normalized residual = {max_norm:.2f} below threshold {bad_data_threshold}")

    return x_current, success, lambdaN
