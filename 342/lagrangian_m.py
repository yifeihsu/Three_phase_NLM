import numpy as np
from utilities.mea_fun import measurement_function
from utilities.make_jacobian import jacobian
from utilities.make_jacobian_p import jacobian_line_params
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve, splu


def find_zero_injection_nodes(mpc, busphase_map, tol=1.0):
    """
    Identify phase-level nodes (bus-phase) that have effectively zero net injection,
    i.e. Generation - Load is within +/- tol (kW or kVar).

    Args:
        mpc : dict containing "bus3p", "load3p", "gen3p" data, etc.
        busphase_map : dict mapping (bus_id, phase_id) -> global node index
        tol : float, absolute threshold for net P,Q (kW,kVar) to consider zero injection

    Returns:
        zero_inj_nodes : list of node indices (0-based) in the global 3-phase node ordering
                         whose net injection magnitude is < tol.
    """

    bus3p = mpc["bus3p"]  # columns: [bus_id, type, base_kV_LL, VmA, VmB, VmC, VaA, VaB, VaC]
    load3p = mpc["load3p"]  # columns: [ldid, ldbus, status, PdA, PdB, PdC, QdA, QdB, QdC] (if truly P,Q)
    gen3p = mpc["gen3p"]  # columns: [genid, gbus, status, VgA, VgB, VgC, PgA, PgB, PgC, QgA, QgB, QgC]

    # 1) Gather bus IDs and identify the slack bus
    bus_ids = [int(row[0]) for row in bus3p]
    slack_buses = [int(row[0]) for row in bus3p if row[1] == 3]
    # If there's always exactly one slack bus in your system:
    # slack_id = slack_buses[0]

    # 2) Initialize netPQ_phase = Generation - Load
    #    We'll store netPQ_phase[(b, ph)] = [P_net(kW), Q_net(kVar)]
    netPQ_phase = {}
    for b in bus_ids:
        for ph in [0, 1, 2]:
            netPQ_phase[(b, ph)] = [0.0, 0.0]  # [P_net, Q_net]

    # 3) Process loads -> subtract from net injection
    #    Confirm that the last 3 columns are indeed QdA, QdB, QdC (not pfA,B,C).
    for row in load3p:
        ldid, ldbus, status, PdA, PdB, PdC, QdA, QdB, QdC = row
        if status == 0:
            continue

        bus_id = int(ldbus)
        # Phase A load
        if abs(PdA) > 1e-9 or abs(QdA) > 1e-9:
            netPQ_phase[(bus_id, 0)][0] -= PdA
            netPQ_phase[(bus_id, 0)][1] -= QdA
        # Phase B load
        if abs(PdB) > 1e-9 or abs(QdB) > 1e-9:
            netPQ_phase[(bus_id, 1)][0] -= PdB
            netPQ_phase[(bus_id, 1)][1] -= QdB
        # Phase C load
        if abs(PdC) > 1e-9 or abs(QdC) > 1e-9:
            netPQ_phase[(bus_id, 2)][0] -= PdC
            netPQ_phase[(bus_id, 2)][1] -= QdC

    # 4) Process generations -> add to net injection
    for row in gen3p:
        genid, gbus, status, VgA, VgB, VgC, PgA, PgB, PgC, QgA, QgB, QgC = row
        if status == 0:
            continue

        bus_id = int(gbus)
        # Phase A
        if abs(PgA) > 1e-9 or abs(QgA) > 1e-9:
            netPQ_phase[(bus_id, 0)][0] += PgA
            netPQ_phase[(bus_id, 0)][1] += QgA
        # Phase B
        if abs(PgB) > 1e-9 or abs(QgB) > 1e-9:
            netPQ_phase[(bus_id, 1)][0] += PgB
            netPQ_phase[(bus_id, 1)][1] += QgB
        # Phase C
        if abs(PgC) > 1e-9 or abs(QgC) > 1e-9:
            netPQ_phase[(bus_id, 2)][0] += PgC
            netPQ_phase[(bus_id, 2)][1] += QgC

    # 5) Determine which nodes have near-zero injection
    zero_inj_nodes = []
    for (b, ph), (p_net, q_net) in netPQ_phase.items():
        # Optionally skip the slack bus if you never want to classify it as zero injection
        if b in slack_buses:
            continue
        if (abs(p_net) < tol) and (abs(q_net) < tol):
            node_idx = busphase_map[(b, ph)]
            zero_inj_nodes.append(node_idx)

    return zero_inj_nodes


# def compute_line_mahalanobis_distances(lambdaN, EA, mpc):
#     """
#     Given the normalized Lagrange multipliers `lambdaN` of length npara,
#     and the full covariance submatrix EA of shape (npara, npara),
#     compute the group Mahalanobis distance for each line's 12 parameters.
#
#     Returns:
#       distances : list of (line_index, distance_value)
#     """
#     line_data = mpc["line3p"]
#     num_lines = len(line_data)
#     nparam_line = 12  # 6 for X + 6 for R per line
#
#     distances = []
#     for i_line in range(num_lines):
#         start_idx = i_line * nparam_line
#         end_idx = start_idx + nparam_line
#
#         # 1) Extract that line's multipliers
#         lam_vec = lambdaN[start_idx:end_idx]
#         # 2) Sub-cov
#         EA_sub = EA[start_idx:end_idx, start_idx:end_idx]
#
#         # 3) Invert sub-block
#         #    (Better to do a Cholesky factor if EA_sub is SPD, but np.linalg.inv is simpler for illustration.)
#         try:
#             EA_sub_inv = np.linalg.inv(EA_sub)
#             # 4) Mahalanobis distance
#             dist_line = np.sqrt(lam_vec @ EA_sub_inv @ lam_vec)
#         except np.linalg.LinAlgError:
#             dist_line = np.inf  # if singular, treat as large distance
#
#         distances.append((i_line, dist_line))
#
#     return distances

def compute_group_rms_dist(lambdaN, nparam_line=12):
    """
    If lambdaN is already individually normalized,
    then we can define a 'group distance' as RMS of each line's 12 parameters.
    """
    num_lines = len(lambdaN) // nparam_line
    distances = []
    for i_line in range(num_lines):
        start_idx = i_line * nparam_line
        end_idx   = start_idx + nparam_line
        lam_block = lambdaN[start_idx:end_idx]
        dist_line = np.sqrt(np.mean(lam_block**2)) * nparam_line
        distances.append((i_line, dist_line))
    return distances

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

    zi_nodes = []

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
        I_H_sp = sps.eye(nH, format='csc')

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
        ], format='csc')

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

    # print_estimation_results(x_current, mpc)

    # Lagrangian multiplier calculation
    # V, theta: the mag and angle for all nodes
    print("Forming Parameters Jacobian")
    # --- Form Hp as before ---
    Hp = jacobian_line_params(x_current, Ybus, mpc, busphase_map)
    print("Jacobian Formed")
    # print the shape of Hp
    print("Shape of Hp:", Hp.shape)

    if not sps.isspmatrix_csc(Hp):
        Hp = Hp.tocsc()
    if not sps.isspmatrix_csc(W_sp):
        W_sp = W_sp.tocsc()
    if not sps.isspmatrix_csc(Gain):
        Gain = Gain.tocsc()

    # Extract Cp and (reduced) Hp efficiently
    all_rows_hp = np.arange(Hp.shape[0])
    keep_rows_hp = np.delete(all_rows_hp, indices_to_remove)
    Cp = Hp[indices_to_remove, :]
    Hp = Hp[keep_rows_hp, :]

    # Compute WHp = W_sp * Hp
    WHp = W_sp.dot(Hp)

    # Stack and transpose in CSC format to form S_sp
    S_sp = sps.vstack([WHp, Cp], format='csc')
    S_sp = -S_sp.transpose(copy=True)

    # ----------------------------------------------------------------
    #       REPLACE FULL INVERSE WITH SPARSE FACTOR + PARTIAL SOLVES
    # ----------------------------------------------------------------

    print("Factorizing Gain for partial solves...")
    Gain_factor = splu(Gain)  # Instead of sps.linalg.inv(Gain)

    # Suppose you need to form E5, E8 = selected rows/cols of Gain_inv
    # Example: "a, b, c" are block sizes. Adjust as appropriate.
    a, b, c = zero_block_sp.shape[0], I_H_sp.shape[0], C_sp.shape[0]

    row_E5 = np.arange(a, a + b)
    col_E5 = row_E5

    row_E8 = np.arange(a + b, a + b + c)
    col_E8 = col_E5  # same columns as E5 in this scenario

    # Build E5 columns by solving Gain * x = e_j for each column j in col_E5
    E5_cols = []
    for j in col_E5:
        e_j = np.zeros(Gain.shape[0])
        e_j[j] = 1.0
        x = Gain_factor.solve(e_j)
        E5_cols.append(x[row_E5])

    # Stack columns into a 2D array, then convert to sparse if desired
    E5 = np.column_stack(E5_cols)
    E5_sp = sps.csc_matrix(E5)

    # Same procedure for E8
    E8_cols = []
    for j in col_E8:
        e_j = np.zeros(Gain.shape[0])
        e_j[j] = 1.0
        x = Gain_factor.solve(e_j)
        E8_cols.append(x[row_E8])

    E8 = np.column_stack(E8_cols)
    E8_sp = sps.csc_matrix(E8)

    # Stack to form phi_sp
    phi_sp = sps.vstack([E5_sp, E8_sp], format='csc')

    # ----------------------------------------------------------------
    #               REST OF THE COMPUTATION
    # ----------------------------------------------------------------

    # Invert diagonal of W_sp
    W_inv_sp = sps.diags(1.0 / W_sp.diagonal(), format='csc')

    # step1_sp = phi_sp * W_inv_sp * phi_sp^T
    print("Computing step1_sp")
    w_diag = W_inv_sp.diagonal()
    phi_scaled = phi_sp.dot(sps.diags(w_diag))
    step1_sp = phi_scaled.dot(phi_sp.transpose())
    covu_sp = step1_sp  # Keep it sparse as long as possible

    def check_PSD(covLa):
        """
        Check if the matrix is positive semi-definite (PSD).
        """
        try:
            # Compute eigenvalues for the symmetric matrix
            eigenvalues = np.linalg.eigvalsh(covLa)
            # Define a tolerance for floating-point comparisons
            tolerance = 1e-9  # Adjust tolerance if needed
            # Check if the minimum eigenvalue is non-negative within the tolerance
            is_psd = np.min(eigenvalues) >= -tolerance
            if is_psd:
                print(f"Matrix appears to be PSD (min eigenvalue: {np.min(eigenvalues):.2e})")
            else:
                print(f"Matrix is NOT PSD (min eigenvalue: {np.min(eigenvalues):.2e})")
        except np.linalg.LinAlgError:
            # Handle cases where eigenvalue computation fails
            print("Eigenvalue computation failed.")
            is_psd = False

    def regularization(covL):
        """
        Regularize the covariance matrix covL by adding a small value to its diagonal.
        """
        epsilon = 1e-12  # Small regularization parameter, adjust if needed

        # Ensure covL is available (Calculate S, R first if needed)
        # R_mat = np.linalg.inv(W) # Assuming W is dense here
        # S_mat = ... calculate S ... (May need G_inv = np.linalg.inv(H_sp.T @ W_sp @ H_sp) etc.)
        # covL = S_mat @ R_mat @ S_mat.T # This calculation might be needed if not already done
        if sps.issparse(covL):
            covL_reg = covL + epsilon * sps.eye(covL.shape[0], format=covL.format)
        else:  # Assuming covL is a NumPy array
            covL_reg = covL + epsilon * np.identity(covL.shape[0])
        return covL_reg
    print("Computing diagonal of EA = diag(S_sp * covu_sp * S_sp^T)")
    X_sp = S_sp.dot(covu_sp)  # both are sparse => X_sp also sparse (NxN)

    # 2) Elementwise multiply X_sp and S_sp, then sum each row.
    #    In SciPy, 'multiply()' is an elementwise product on sparse matrices;
    #    'sum(axis=1)' sums each row into a 1x1 result; we then flatten to 1D.
    ea_diag = X_sp.multiply(S_sp).sum(axis=1).A1

    # If you truly need the entire matrix EA in dense form, you can do:
    ea_sp = S_sp.dot(covu_sp).dot(S_sp.transpose())  # keep as sparse
    ea = ea_sp.toarray()                             # convert to dense if absolutely necessary


    print("EA diagonal computed.")

    # ----------------------------------------------------------------
    #               LAMBDA COMPUTATIONS
    # ----------------------------------------------------------------

    # Extract lam_part from your dxl array
    lam_part = dxl[2 * nnodephase - 3:]

    # Compute lambda_vec = S_sp @ lam_part in sparse form
    lambda_vec = S_sp @ lam_part

    # If using just the diagonal of EA:
    tt = np.sqrt(ea_diag)  # shape is (len(ea_diag), )
    lambdaN = lambda_vec / tt

    line_distances = compute_group_rms_dist(lambdaN, nparam_line=12)
    for i_line, dval in line_distances:
        print(f"Line {i_line + 1} => Mahalanobis distance = {dval:.3f}")

    # Done.  Now lambdaN, ea_diag, etc. are ready for further use.
    print("All done.")

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
