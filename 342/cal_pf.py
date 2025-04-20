import numpy as np
import time
import math
from parse_opendss_file import build_global_y_per_unit

def run_newton_powerflow_3p(mpc, tol=1e-6, max_iter=20):
    """
    Full-Newton 3-phase PF with ratioed wye-wye transformer. Rectangular Coordinates.
    Incorporates PV buses by enforcing fixed voltage magnitude on them.
    """
    t_start = time.time()

    # 1) Build Y-bus
    Ybus, node_order = build_global_y_per_unit(mpc)
    nnodephase = Ybus.shape[0]

    bus3p = mpc["bus3p"]
    baseMVA = mpc["baseMVA"]

    # Build a lookup: (bus_id, phase) -> global node-phase index in Ybus
    bus_ids = [int(row[0]) for row in bus3p]
    busphase_map = {}
    idx = 0
    for b in bus_ids:
        for ph in range(3):
            busphase_map[(b, ph)] = idx
            idx += 1

    # 2) Identify slack, PV, PQ bus IDs
    slack_bus_ids = [int(row[0]) for row in bus3p if row[1] == 3]
    pv_bus_ids    = [int(row[0]) for row in bus3p if row[1] == 2]
    pq_bus_ids    = [int(row[0]) for row in bus3p if row[1] == 1]

    # 2a) Partition node-phases by type
    slack_indices = []
    pv_indices = []
    pq_indices = []
    for (b, ph), i in busphase_map.items():
        if b in slack_bus_ids:
            slack_indices.append(i)
        elif b in pv_bus_ids:
            pv_indices.append(i)
        else:
            pq_indices.append(i)

    # We'll define 'unknown_indices' = all node-phases that are either PQ or PV
    unknown_indices = pv_indices + pq_indices
    # We'll also need to know which subset each index belongs to so that
    # we form the mismatch properly (real for both, but imaginary vs. voltage-mag).
    # Let's build a small map from global index -> "PV" or "PQ"
    node_type_map = {}
    for i in pv_indices:
        node_type_map[i] = "PV"
    for i in pq_indices:
        node_type_map[i] = "PQ"

    # 3) Parse net injection (P_inj, Q_inj) in p.u.
    P_inj = np.zeros(nnodephase)
    Q_inj = np.zeros(nnodephase)

    # dictionaries to accumulate load/generation per bus-phase
    bus_phase_load = {(b, ph): (0.0, 0.0) for (b, ph) in busphase_map}
    bus_phase_gen  = {(b, ph): (0.0, 0.0) for (b, ph) in busphase_map}

    # 3.1) Parse loads
    for row in mpc["load3p"]:
        ldid, ldbus, status, PdA, PdB, PdC, QdA, QdB, QdC = row
        if status == 0:
            continue
        P_ph = np.array([PdA, PdB, PdC]) / 1000.0  # convert kW -> MW
        Q_ph = np.array([QdA, QdB, QdC]) / 1000.0  # convert kVar -> MVar
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
    for (b, ph), i in busphase_map.items():
        loadP, loadQ = bus_phase_load[(b, ph)]
        genP, genQ   = bus_phase_gen[(b, ph)]
        P_inj[i] = (genP - loadP) / baseMVA
        Q_inj[i] = (genQ - loadQ) / baseMVA

    # 4) Build initial guess for V in rectangular form
    #    Also store the "voltage magnitude setpoint" for PV buses if needed.
    Vr0 = np.zeros(nnodephase)
    Vi0 = np.zeros(nnodephase)
    Vset_pv = {}  # (global_index) -> setpoint magnitude

    for row in bus3p:
        b = int(row[0])
        btype = int(row[1])
        # row = [bus_id, bus_type, kvLL, VmA, VmB, VmC, VaA, VaB, VaC]
        _, _, _, VmA, VmB, VmC, VaA, VaB, VaC = row
        # Convert angles to rectangular for the initial guess
        # For Slack or PV, we'll use the actual V (from parse).
        # For PQ, we might just guess 1.0 per-phase with a typical 120-phase shift.

        for ph, (Vm, Va) in enumerate([(VmA, VaA), (VmB, VaB), (VmC, VaC)]):
            i_global = busphase_map[(b, ph)]
            if btype == 3:
                # Slack bus
                Vr0[i_global] = Vm * math.cos(Va)
                Vi0[i_global] = Vm * math.sin(Va)
            elif btype == 2:
                # PV bus, store setpoint magnitude = Vm
                Vr0[i_global] = Vm * math.cos(Va)
                Vi0[i_global] = Vm * math.sin(Va)
                Vset_pv[i_global] = Vm  ## [CHANGED] keep the setpoint
            else:
                # PQ bus, guess magnitude=1.0, angle per typical 3-phase shift
                if ph == 0:
                    angle0 = 0.0
                elif ph == 1:
                    angle0 = math.radians(-120.0)
                else:
                    angle0 = math.radians(120.0)
                Vr0[i_global] = 1.0 * math.cos(angle0)
                Vi0[i_global] = 1.0 * math.sin(angle0)

    # We'll store unknown states in x = [Vr(unknowns), Vi(unknowns)]
    def pack_x(Vr, Vi):
        arr = []
        for i in unknown_indices:
            arr.append(Vr[i])
            arr.append(Vi[i])
        return np.array(arr)

    def unpack_x(x, Vr, Vi):
        idx = 0
        for i in unknown_indices:
            Vr[i] = x[idx]
            Vi[i] = x[idx+1]
            idx += 2

    x0 = pack_x(Vr0, Vi0)

    # Basic Sbus calculation subroutine
    def calc_S(Vc, Ybus):
        Ibus = Ybus @ Vc
        Sbus = Vc * np.conjugate(Ibus)  # elementwise
        return Sbus, Ibus

    def calc_dS_dVrVi(Ybus, Vc, Ibus):
        """
        Returns (dS_dVr, dS_dVi),
        each NxN complex, partial derivatives in cartesian form.
        """
        n = len(Vc)
        diagV = np.diag(Vc)
        diagI = np.diag(Ibus)
        dS_dVr = np.conjugate(diagI) + diagV @ np.conjugate(Ybus)
        dS_dVi = 1j*(np.conjugate(diagI) - diagV @ np.conjugate(Ybus))
        return dS_dVr, dS_dVi

    def mismatch_and_jacobian(x):
        """
        Builds mismatch f and real-valued Jacobian J.
        For PQ node-phases:  dP=0, dQ=0.
        For PV node-phases:  dP=0, |V| - Vset=0.
        Slack is excluded from unknowns.
        """
        # 1) Reconstruct full V from x
        Vr = Vr0.copy()
        Vi = Vi0.copy()
        unpack_x(x, Vr, Vi)
        Vc = Vr + 1j*Vi

        # 2) Compute S_calc
        S_calc, Ibus = calc_S(Vc, Ybus)

        # 3) Prepare mismatch
        # Each unknown node-phase i gets 2 eqns, but differ by bus type
        f_list = []
        # We'll also build row -> global_i so we can easily find partials
        row_of_node = []  # each entry is (global_node, eqn_type="P" or "Q"/"V")
        for i_idx, i_node in enumerate(unknown_indices):
            # eqn for real power mismatch always
            f_list.append(S_calc[i_node].real - P_inj[i_node])  # dP
            row_of_node.append( (i_node, "P") )
            # eqn for Q or |V|-Vset
            if node_type_map[i_node] == "PQ":
                # Q mismatch
                f_list.append(S_calc[i_node].imag - Q_inj[i_node]) # dQ
                row_of_node.append( (i_node, "Q") )
            else:
                # PV bus => enforce voltage magnitude
                magV = math.sqrt(Vr[i_node]**2 + Vi[i_node]**2)
                f_list.append(magV - Vset_pv[i_node])
                row_of_node.append( (i_node, "V") )

        f = np.array(f_list, dtype=float)

        # 4) Build partial derivatives dS_dVr, dS_dVi
        dS_dVr, dS_dVi = calc_dS_dVrVi(Ybus, Vc, Ibus)

        # We form J by taking partials wrt [Vr_j, Vi_j], but only for j in unknown_indices
        nU = len(unknown_indices)
        # Each unknown node-phase has 2 eqns, so total eqn count:
        # eqn_count = sum(2 for PQ, or 2 for PV) = nU * 2, but actually
        # for each node-phase that is PV or PQ, we do 2 eqns. Good. So 2*nU total rows.
        # Actually note that for a PV node-phase, the second eqn is for |V|-Vset,
        # not Q mismatch, but we still have 2 eqns total.
        J = np.zeros((len(f_list), 2*nU), dtype=float)

        # Column index mapping: unknown j -> (colVr, colVi)
        col_index_map = {}
        for j_idx, j_node in enumerate(unknown_indices):
            colVr = 2*j_idx
            colVi = 2*j_idx + 1
            col_index_map[j_node] = (colVr, colVi)

        # Fill row by row
        for row_idx, (i_node, eqn_type) in enumerate(row_of_node):

            # partial wrt each unknown j_node
            for j_idx, j_node in enumerate(unknown_indices):
                cVr, cVi = col_index_map[j_node]

                if eqn_type == "P":
                    # d(Pcalc_i - Pinj_i)/dVr_j = Re(dS_dVr[i_node,j_node])
                    # d(Pcalc_i - Pinj_i)/dVi_j = Re(dS_dVi[i_node,j_node])
                    dSdVr_ij = dS_dVr[i_node, j_node]
                    dSdVi_ij = dS_dVi[i_node, j_node]
                    J[row_idx, cVr] = dSdVr_ij.real
                    J[row_idx, cVi] = dSdVi_ij.real

                elif eqn_type == "Q":
                    # d(Qcalc_i - Qinj_i)/dVr_j = Im(dS_dVr[i_node,j_node])
                    # d(Qcalc_i - Qinj_i)/dVi_j = Im(dS_dVi[i_node,j_node])
                    dSdVr_ij = dS_dVr[i_node, j_node]
                    dSdVi_ij = dS_dVi[i_node, j_node]
                    J[row_idx, cVr] = dSdVr_ij.imag
                    J[row_idx, cVi] = dSdVi_ij.imag

                else:  # eqn_type == "V" => mag(V_i) - Vset
                    # partial of (|V_i|) wrt Vr_j, Vi_j
                    if i_node == j_node:
                        # d/dVr_i of sqrt(Vr_i^2 + Vi_i^2) = Vr_i / |V_i|
                        # d/dVi_i of sqrt(Vr_i^2 + Vi_i^2) = Vi_i / |V_i|
                        magVi = math.sqrt(Vr[i_node]**2 + Vi[i_node]**2)
                        if magVi < 1e-12:
                            # avoid dividing by zero; just approximate
                            dVdVr = 0.0
                            dVdVi = 0.0
                        else:
                            dVdVr = Vr[i_node]/magVi
                            dVdVi = Vi[i_node]/magVi
                        J[row_idx, cVr] = dVdVr
                        J[row_idx, cVi] = dVdVi
                    else:
                        # derivative of bus i's magnitude w.r.t. bus j's voltage is 0
                        pass

        return f, J

    # 5) Newton iteration
    print(" it    max residual        max Δx")
    print("----  --------------  --------------")

    x_est = x0.copy()
    f, _ = mismatch_and_jacobian(x_est)
    max_res = np.max(np.abs(f))
    print(f"  0      {max_res:1.3e}           -")

    dx = None
    for it in range(1, max_iter+1):
        f, J = mismatch_and_jacobian(x_est)
        max_res = np.max(np.abs(f))
        dx = np.linalg.solve(J, -f)

        max_dx = np.max(np.abs(dx))
        if max_dx < tol:
            break
        x_est += dx
        print(f" {it:2d}      {max_res:1.3e}       {max_dx:1.3e}")

    # final check
    f, _ = mismatch_and_jacobian(x_est)
    max_res = np.max(np.abs(f))
    if dx is not None:
        max_dx = np.max(np.abs(dx))
    else:
        max_dx = 0.0

    if max_res < tol or max_dx < tol:
        print(f"Newton's method converged in {it} iterations.\nPF successful\n")
    else:
        print(f"Warning: Did not converge in {it} iterations (res={max_res:1.3e}).\n")

    t_elapsed = time.time() - t_start
    print(f"PF finished in {t_elapsed:.2f} seconds\n")

    # 6) Unpack final solution
    Vr_final = Vr0.copy()
    Vi_final = Vi0.copy()
    unpack_x(x_est, Vr_final, Vi_final)

    return Vr_final, Vi_final, busphase_map
