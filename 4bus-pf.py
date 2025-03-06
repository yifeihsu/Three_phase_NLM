import numpy as np
import time
from system_y import build_global_y_per_unit
from wls_estimation import run_wls_state_estimation
from wls_estimation_p import run_wls_state_estimation_polar
from lagrangian_m import run_lagrangian_polar
from utilities.mea_fun import measurement_function
from utilities.report_results import report_results

##############################################################################
# 1. Load the 4-bus test case (matpower-like data)
##############################################################################
def load_t_case3p_a():
    """
    Returns a Python 'dealmpc' dict for the 4-bus unbalanced distribution system
    with a wye-wye transformer that has ratio ~12.47/4.16 (3:1).
    """
    mpc = {}
    mpc["version"] = "2"
    mpc["baseMVA"] = 1   # for per-unit conversions

    mpc["freq"] = 60.0
    # bus3p columns: [bus_id, type, base_kV_LL, VmA, VmB, VmC, VaA, VaB, VaC]
    # type=3 => slack, type=1 => PQ, type=2 => PV
    # base_kV_LL: line-to-line base
    mpc["bus3p"] = np.array([
        [1, 3, 12.47, 1.0, 1.0, 1.0,   0.0,   -120.0, 120.0],
        [2, 1, 12.47, 1.0, 1.0, 1.0,   0.0,   -120.0, 120.0],
        [3, 1,  4.16, 1.0, 1.0, 1.0,   0.0,   -120.0, 120.0],
        [4, 1,  4.16, 1.0, 1.0, 1.0,   0.0,   -120.0, 120.0],
    ], dtype=float)
    # line3p columns: [id, fbus, tbus, status, lcid, length (miles)]
    mpc["line3p"] = np.array([
        [1, 1, 2, 1, 1, 2000.0/5280.0],
        [2, 3, 4, 1, 1, 2500.0/5280.0],
    ], dtype=float)
    # xfmr3p columns: [xfid, fbus, tbus, status, R, X, basekVA, basekV_LL(HV)]
    mpc["xfmr3p"] = np.array([
        [1, 2, 3, 1, 0.01, 0.06, 6000.0, 12.47, 4.16],
    ], dtype=float)
    # load3p columns: [ldid, ldbus, status, PdA, PdB, PdC, pfA, pfB, pfC]
    mpc["load3p"] = np.array([
        [1, 4, 1, 1275.0, 1800.0, 2375.0, 0.85, 0.90, 0.95],
    ], dtype=float)
    # gen3p columns: [genid, gbus, status, VgA, VgB, VgC, PgA, PgB, PgC, QgA, QgB, QgC]
    mpc["gen3p"] = np.array([
        [1, 1, 1, 1.0, 1.0, 1.0, 2000.0, 2000.0, 2000.0, 0.0, 0.0, 0.0],
    ], dtype=float)
    # line construction (3x3) data for lcid=1
    # [lcid, R11, R21, R31, R22, R32, R33, X11, X21, X31, X22, X32, X33, ...C11...]
    mpc["lc"] = np.array([
        [1,
         0.457541, 0.15594,  0.153474, 0.466617, 0.157996, 0.461462,
         1.078,    0.501648, 0.384909, 1.04813,  0.423624, 1.06502,
         15.0671, -4.86241, -1.85323, 15.875, -3.09098, 14.3254]
    ], dtype=float)

    return mpc
##############################################################################
# 2. Building the 3-phase Y-bus (Deprecated, now with OpenDSS directly)
##############################################################################
def build_ybus_3p(mpc):
    """
    Builds a global 3N x 3N Y-bus where each bus can have its own
    line-to-line base voltage. For lines, we use the from-bus base.
    For the wye-wye transformer, we apply a ratio a:1 with a = (HV_base / LV_base).
    """
    bus3p = mpc["bus3p"]    # columns: [bus_id, type, base_kV_LL, ...]
    # line3p = mpc["line3p"]  # columns: [id, fbus, tbus, status, lcid, length (miles)]
    # xfmr3p = mpc["xfmr3p"]  # columns: [xfid, fbus, tbus, status, R%, X%, basekVA, basekV(HV)]
    # lc = mpc["lc"]

    # 1) Identify buses, build index map
    bus_ids = [int(row[0]) for row in bus3p]
    # nbus = len(bus_ids)
    busphase_map = {}
    idx = 0
    for b in bus_ids:
        for ph in range(3):
            busphase_map[(b, ph)] = idx
            idx += 1
    return busphase_map
##############################################################################
# 3. Newton Power Flow in Rectangular Coordinates
##############################################################################
def run_newton_powerflow_3p(mpc, tol=1e-6, max_iter=20):
    """
    Full-Newton 3-phase PF with ratioed wye-wye transformer. Rectangular Coordinate.
    """
    t_start = time.time()

    # 1) Build Y-bus
    busphase_map = build_ybus_3p(mpc) # Find the index of different nodes in global Ybus matrix
    Ybus, node_order = build_global_y_per_unit()
    bus3p = mpc["bus3p"]
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
        ldid, ldbus, status, PdA, PdB, PdC, pfA, pfB, pfC = row
        if status == 0:
            continue
        # convert from kW to MW => /1000
        P_ph = np.array([PdA, PdB, PdC]) / 1000.0
        pf_ph = np.array([pfA, pfB, pfC])
        Q_ph = P_ph * np.sqrt(1/(pf_ph**2) - 1)
        # negative for load
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

    # 7) Summarize the measurement function
    def generate_measurements(Vr, Vi, Ybus):
        """
        Generate measurement data (z) from final power flow results.
        We can measure:
          1. Bus voltage magnitudes,
          2. Bus injection P, Q,
          3. Line flow P, Q,
          etc.
        Returns:
          z (np.array) : measurement vector
          measurement_info : info about each measurement (indices, types, etc.)
        """
        Vmag = np.sqrt(Vr**2 + Vi**2)
        z_list = []
        measurement_info = []
        Vc = Vr + 1j*Vi
        S_calc, _ = calc_S(Vc, Ybus)
        for i in range(nnodephase):
            z_list.append(S_calc[i].real)
            measurement_info.append(("P_inj", i))
        for i in range(nnodephase):
            z_list.append(S_calc[i].imag)
            measurement_info.append(("Q_inj", i+nnodephase))
        for i in range(nnodephase):
            z_list.append(Vmag[i])
            measurement_info.append(("Vmag", i+2*nnodephase))
        z = np.array(z_list)
        return z, measurement_info

    z, measurement_info = generate_measurements(Vr_final, Vi_final, Ybus)

    return Vr_final, Vi_final, busphase_map, z, measurement_info
##############################################################################
# 4. Reporting in MATPOWER-like format
##############################################################################
def print_lambdaN_mapping(lambdaN, mpc):
    """
    Prints the mapping of lambdaN values to line and phase indices.

    Parameters:
    lambdaN (np.array): The lambdaN vector.
    mpc (dict): The MATPOWER case dictionary.
    """
    n_lines = mpc["line3p"].shape[0]
    n_phases = 3
    npara = n_lines * n_phases  # For reactance; total lambdaN has 2*npara entries

    print("LambdaN mapping:")
    # First, print reactance sensitivities (first 6 entries)
    print("\nReactance sensitivities:")
    for line_idx in range(n_lines):
        # The first column of mpc["line3p"] defines the line ordering.
        line_number = int(mpc["line3p"][line_idx, 0])
        for phase in range(n_phases):
            index_reactance = line_idx * n_phases + phase
            print(f"  Line {line_number}, Phase {phase + 1} => lambdaN[{index_reactance}] = {lambdaN[index_reactance]}")

    # Next, print resistance sensitivities (last 6 entries)
    print("\nResistance sensitivities:")
    for line_idx in range(n_lines):
        line_number = int(mpc["line3p"][line_idx, 0])
        for phase in range(n_phases):
            index_resistance = npara + line_idx * n_phases + phase
            print(f"  Line {line_number}, Phase {phase + 1} => lambdaN[{index_resistance}] = {lambdaN[index_resistance]}")

# Example usage:
# print_lambdaN_mapping(lambdaN, mpc)

##############################################################################
# 5. Main driver
##############################################################################
def main():
    # 1) Load the 4-bus data
    mpc = load_t_case3p_a()
    from opendssdirect import dss
    dss.run_command('Redirect "4Bus-YY-Bal.DSS"')
    dss.Solution.Solve()
    # 2) Run Newton PF
    Vr, Vi, busphase_map, z, measurement_info = run_newton_powerflow_3p(mpc, tol=1e-6, max_iter=20)
    # 3) Print final PF results
    x_f = Vr + 1j * Vi
    x = np.concatenate([np.abs(x_f), np.angle(x_f)])
    report_results(Vr, Vi, busphase_map, mpc)
    Y_pu_s, _ = build_global_y_per_unit(line_changes=None)
    z = measurement_function(x, Y_pu_s, mpc, busphase_map)
    # 4) Generate the measurement data from PF results (Add noise)
    var_P = 0.001
    var_Q = 0.001
    var_V = 0.0001
    num_P_inj = len([1 for info in measurement_info if info[0] == "P_inj"])
    num_Q_inj = len([1 for info in measurement_info if info[0] == "Q_inj"])
    num_PQ_flow = 4 * 3 * len(mpc["line3p"])
    num_Vmag = len([1 for info in measurement_info if info[0] == "Vmag"])
    covariance_matrix = np.diag([var_P]*num_P_inj + [var_Q]*num_Q_inj + [var_P]*num_PQ_flow + [var_V]*num_Vmag)
    z_noisy = z + np.random.multivariate_normal(np.zeros(len(z)), covariance_matrix)
    x_est, success, lambdaN = run_lagrangian_polar(
        z_noisy, x_f, busphase_map, Y_pu_s, covariance_matrix, mpc
    )
    # Add parameter errors

    # # 1) Build the baseline Y with no changes
    # # 2) For each line in your circuit
    # for ln in dss.Lines.AllNames():
    #     ln_lower = ln.lower()
    #
    #     # 2.1) Iterate over all phase parameters for B first
    #     for param in phase_params:
    #         for fac in factors:
    #             # e.g. param='phase_a' => param_label='phase_a_b'
    #             param_label = param + '_b'
    #
    #             # Make a dictionary for the line changes:
    #             #   -> means scale the imaginary part of e.g. "phase_a" by 'fac'
    #             line_changes = {
    #                 ln_lower: {param_label: fac}
    #             }
    #
    #             # Build Y with this single B error
    #             Y_pu, nodes = build_global_y_per_unit(line_changes=line_changes)
    #
    #             # Run your Lagrangian-based detection
    #             x_est, success, lambdaN = run_lagrangian_polar(
    #                 z_noisy, x_f, busphase_map, Y_pu, covariance_matrix, mpc
    #             )
    #
    #             # Find the index of largest |lambdaN|
    #             largest_idx = np.argmax(np.abs(lambdaN))
    #             largest_val = lambdaN[largest_idx]
    #
    #             # Print info
    #             print(f"[B] line={ln_lower}, param={param}, factor={fac}: "
    #                   f"Largest idx={largest_idx}, |lambdaN|={abs(largest_val):.4g}, "
    #                   f"Val={largest_val:.4g}")
    #
    #             # print_lambdaN_mapping(lambdaN, mpc)
    #
    #     # 2.2) Iterate over all phase parameters for G next
    #     for param in phase_params:
    #         for fac in factors:
    #             param_label = param + '_g'
    #
    #             line_changes = {
    #                 ln_lower: {param_label: fac}
    #             }
    #
    #             # Build Y with this single G error
    #             Y_pu, nodes = build_global_y_per_unit(line_changes=line_changes)
    #
    #             x_est, success, lambdaN = run_lagrangian_polar(
    #                 z_noisy, x_f, busphase_map, Y_pu, covariance_matrix, mpc
    #             )
    #
    #             largest_idx = np.argmax(np.abs(lambdaN))
    #             largest_val = lambdaN[largest_idx]
    #
    #             print(f"[G] line={ln_lower}, param={param}, factor={fac}: "
    #                   f"Largest idx={largest_idx}, |lambdaN|={abs(largest_val):.4g}, "
    #                   f"Val={largest_val:.4g}")
    #
    #             # print_lambdaN_mapping(lambdaN, mpc)
    #
    #

    # 5) Run state estimation
    # print(x_f)
    # x_est, success = run_wls_state_estimation(z_noisy, x_f, busphase_map, Ybus, covariance_matrix, mpc)
    # x_est, success = run_wls_state_estimation_polar(z_noisy, x_f, busphase_map, Ybus, covariance_matrix, mpc)
    # Modify the Y matrix in SE program

    # matrix_variations = [
    #     {
    #         "Rmatrix": [0.457552, 0.155951, 0.466628, 0.153485, 0.158007, 0.461473],
    #         "Xmatrix": [100.805, 0.501679, 1.04, 0.384938, 0.423653, 1.06507],
    #         "Cmatrix": [15.0675, -4.86254, 15.8754, -1.85328, -3.09107, 14.3258]
    #     },
    #     # Add more variations as needed
    # ]
    # for idx, matrices in enumerate(matrix_variations):
    #     print(f"Running simulation with matrix set {idx + 1}")
    #
    #     # Create a new LineCode with the specified R, X, and C matrices
    #     dss.run_command(f"New LineCode.CustomCode nphases=3 units=mi "
    #                     f"Rmatrix={matrices['Rmatrix']} "
    #                     f"Xmatrix={matrices['Xmatrix']} "
    #                     f"Cmatrix={matrices['Cmatrix']}")
    #     # dss.run_command("Edit Line.line1 LineCode=CustomCode")
    #     # dss.run_command("Edit Line.line2 LineCode=CustomCode")
    #     dss.run_command("solve")
    # Ybus_err, node_order = build_global_y_per_unit()

    # 6) Print the results of the lambdaN
    n_lines = mpc["line3p"].shape[0]
    n_phases = 3
    npara = n_lines * n_phases  # For reactance; total lambdaN has 2*npara entries

    # For demonstration, assume lambdaN is computed and has 12 entries.
    # Here, we create a dummy lambdaN vector with values 1 to 12.
    print("LambdaN mapping:")
    # First, print reactance sensitivities (first 6 entries)
    print("\nReactance sensitivities:")
    for line_idx in range(n_lines):
        # The first column of mpc["line3p"] defines the line ordering.
        line_number = int(mpc["line3p"][line_idx, 0])
        for phase in range(n_phases):
            index_reactance = line_idx * n_phases + phase
            print(f"  Line {line_number}, Phase {phase + 1} => lambdaN[{index_reactance}] = {lambdaN[index_reactance]}")

    # Next, print resistance sensitivities (last 6 entries)
    print("\nResistance sensitivities:")
    for line_idx in range(n_lines):
        line_number = int(mpc["line3p"][line_idx, 0])
        for phase in range(n_phases):
            index_resistance = npara + line_idx * n_phases + phase
            print(
                f"  Line {line_number}, Phase {phase + 1} => lambdaN[{index_resistance}] = {lambdaN[index_resistance]}")

    if success:
        print("State Estimation converged successfully.")
    else:
        print("State Estimation failed or didn't converge.")
if __name__ == "__main__":
    main()
