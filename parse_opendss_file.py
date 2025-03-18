# import opendssdirect as dss
# import numpy as np
# import math
# from scipy.sparse import csc_matrix, lil_matrix, diags
#
# def parse_opendss_to_mpc(dss_filename, baseMVA=1.0):
#     """
#     1) Clears and redirects the OpenDSS model in 'dss_filename'.
#     2) Solves the circuit (so we can read bus voltages, load powers, etc.).
#     3) Parses Buses, Lines, Transformers, Loads, Gens, storing in an 'mpc' dict
#        similar to a 'matpower-like' structure.
#
#     Additional:
#       - Stores 'bus_vbase_ln' dictionary with each bus's line-to-neutral base in volts.
#       - Optionally stores 'node_order' if you want a certain bus-phase listing.
#       - Stores a busname->ID map in mpc["busname_to_id"].
#
#     Returns:
#       mpc (dict)
#     """
#
#     # 1) Clear & Redirect
#     dss.Command("Clear")
#     dss.Command(f'Redirect "{dss_filename}"')
#     dss.Solution.Solve()
#
#     circuit = dss.Circuit
#     bus_names = circuit.AllBusNames()
#
#     mpc = {}
#     mpc["version"] = "2"
#     mpc["baseMVA"] = baseMVA
#     mpc["freq"] = 60.0
#
#     # We'll store a dictionary of bus LN base voltages
#     mpc["bus_vbase_ln"] = {}
#     # We'll store bus_id mapping
#     busname_to_id = {}
#     # A list for bus3p: [bus_id, type, base_kV_LL, VmA, VmB, VmC, VaA, VaB, VaC]
#     bus3p_list = []
#
#     for i, busname in enumerate(bus_names, start=1):
#         bus_lc = busname.lower()
#         busname_to_id[bus_lc] = i
#         dss.Circuit.SetActiveBus(busname)
#
#         # In many distribution feeders, Bus.kVBase() might be L-N or L-L depending on the .dss.
#         # We assume it's line-to-line for typical "kVBase=12.47" feeders.
#         # If it's actually L-N, you should adapt.
#         kv_ln = dss.Bus.kVBase()
#         kv_ll = kv_ln * math.sqrt(3)
#         if (kv_ll is None) or (kv_ll <= 1e-6):
#             kv_ll = 1.0  # fallback
#
#         # We'll assume the line-to-neutral base is kv_ll / sqrt(3):
#
#         # Store in volts:
#         mpc["bus_vbase_ln"][bus_lc] = kv_ln * 1e3
#
#         # Let's treat all buses as type=1 (PQ)
#         # If you detect a slack bus by name or an HV feed, you can do type=3, etc.
#         if bus_lc == "p1":
#             bus_type = 3
#         else:
#             bus_type = 1
#
#         # We read the post-solve bus voltages.
#         # We'll store them in per-unit (just for a guess).
#         nodes = dss.Bus.Nodes()  # e.g. [1,2,3]
#         mag_angle = dss.Bus.VMagAngle()  # e.g. [V1, ang1, V2, ang2, ...]
#
#         # create default arrays
#         vm = [0.0, 0.0, 0.0]
#         va = [0.0, 0.0, 0.0]
#         pair_list = list(zip(mag_angle[0::2], mag_angle[1::2]))
#
#         for idx_phase, node_number in enumerate(nodes):
#             if idx_phase < len(pair_list):
#                 # convert magnitude to p.u. wrt LN base
#                 # if Bus.kVBase() was line-to-line, LN base ~ kv_ll/sqrt(3)
#                 v_magnitude = pair_list[idx_phase][0]  # in volts
#                 v_angle_deg = pair_list[idx_phase][1]
#                 # convert to p.u. magnitude
#                 v_pu = 0.0
#                 if kv_ln > 1e-9:
#                     v_pu = (v_magnitude) / (kv_ln * 1e3)
#                 vm[idx_phase] = v_pu
#                 # convert angle to radians if you prefer
#                 va[idx_phase] = math.radians(v_angle_deg)
#
#         # bus3p row
#         row = [
#             i,            # bus_id
#             bus_type,     # type
#             kv_ll,        # base_kV_LL
#             vm[0], vm[1], vm[2],
#             va[0], va[1], va[2]
#         ]
#         bus3p_list.append(row)
#
#     mpc["bus3p"] = np.array(bus3p_list, dtype=float)
#     mpc["busname_to_id"] = busname_to_id
#
#     # 2) Lines
#     all_elems = circuit.AllElementNames()
#     line3p_list = []
#     line_counter = 0
#     for elem_name in all_elems:
#         if elem_name.lower().startswith("line."):
#             line_counter += 1
#             dss.Circuit.SetActiveElement(elem_name)
#
#             fb = dss.CktElement.BusNames()[0].split(".")[0].lower()
#             tb = dss.CktElement.BusNames()[1].split(".")[0].lower()
#             fbus_id = busname_to_id.get(fb, 0)
#             tbus_id = busname_to_id.get(tb, 0)
#
#             status = 1 if dss.CktElement.Enabled() else 0
#             line_name = elem_name.split(".", 1)[1]
#             dss.Lines.Name(line_name)
#
#             length_mi = dss.Lines.Length()
#             # line code name
#             lcode = dss.Lines.LineCode()
#             lcid = 0  # or parse line code map if needed
#
#             line3p_list.append([
#                 line_counter,
#                 fbus_id,
#                 tbus_id,
#                 status,
#                 lcid,
#                 length_mi
#             ])
#     mpc["line3p"] = np.array(line3p_list, dtype=float)
#
#     # 3) Transformers
#     xfmr3p_list = []
#     xfmr_counter = 0
#     for elem_name in all_elems:
#         if elem_name.lower().startswith("transformer."):
#             xfmr_counter += 1
#             dss.Circuit.SetActiveElement(elem_name)
#             xf = dss.Transformers
#
#             buses = dss.CktElement.BusNames()
#             fb = buses[0].split(".")[0].lower()
#             tb = buses[1].split(".")[0].lower()
#             fbus_id = busname_to_id.get(fb, 0)
#             tbus_id = busname_to_id.get(tb, 0)
#
#             status = 1 if dss.CktElement.Enabled() else 0
#             xf.First()  # if multi-winding, adapt logic
#
#             xf.Wdg(1)
#             w1_kv = xf.kV()
#             w1_kva = xf.kVA()
#
#             xf.Wdg(2)
#             w2_kv = xf.kV()
#             w2_kva = xf.kVA()
#
#             r = xf.Rneut()
#             x = xf.Xhl()
#
#             basekVA = max(w1_kva, w2_kva)
#             xfmr3p_list.append([
#                 xfmr_counter,
#                 fbus_id,
#                 tbus_id,
#                 status,
#                 r, x,
#                 basekVA,
#                 w1_kv,
#                 w2_kv
#             ])
#
#     mpc["xfmr3p"] = np.array(xfmr3p_list, dtype=float)
#
#     # 4) Loads (storing solved phase power usage)
#     load3p_list = []
#     load_id_counter = 0
#     for elem_name in all_elems:
#         if not elem_name.lower().startswith("load."):
#             continue
#         load_id_counter += 1
#         dss.Circuit.SetActiveElement(elem_name)
#         load_bus_full = dss.CktElement.BusNames()[0]
#         bus_only = load_bus_full.split(".")[0].lower()
#         bus_id = busname_to_id.get(bus_only, 0)
#         status = 1 if dss.CktElement.Enabled() else 0
#
#         powers = dss.CktElement.Powers()
#         n_conds = dss.CktElement.NumConductors()
#         nodes = dss.CktElement.NodeOrder()
#
#         pA = pB = pC = 0.0
#         qA = qB = qC = 0.0
#         for i_cond in range(n_conds):
#             p_cond = powers[2*i_cond]
#             q_cond = powers[2*i_cond + 1]
#             phase_node = 0
#             if i_cond < len(nodes):
#                 phase_node = nodes[i_cond]
#             if phase_node == 1:
#                 pA += p_cond
#                 qA += q_cond
#             elif phase_node == 2:
#                 pB += p_cond
#                 qB += q_cond
#             elif phase_node == 3:
#                 pC += p_cond
#                 qC += q_cond
#             else:
#                 pass
#
#         # Convert sign so we store positive consumption
#         pA_cons = -pA
#         pB_cons = -pB
#         pC_cons = -pC
#         qA_cons = -qA
#         qB_cons = -qB
#         qC_cons = -qC
#
#         load3p_list.append([
#             load_id_counter,
#             bus_id,
#             status,
#             pA_cons, pB_cons, pC_cons,
#             qA_cons, qB_cons, qC_cons
#         ])
#     mpc["load3p"] = np.array(load3p_list, dtype=float)
#
#     # 5) Generators (optional)
#     gen3p_list = []
#     gen_counter = 0
#     for elem_name in all_elems:
#         if elem_name.lower().startswith("generator."):
#             gen_counter += 1
#             dss.Circuit.SetActiveElement(elem_name)
#             gen_bus = dss.CktElement.BusNames()[0].split(".")[0].lower()
#             bus_id = busname_to_id.get(gen_bus, 0)
#             status = 1 if dss.CktElement.Enabled() else 0
#             dss.Generators.Name(elem_name.split(".")[1])
#             pg = dss.Generators.kW()
#             qg = dss.Generators.kvar()
#             vg = 1.0
#             pgA = pg/3.0
#             pgB = pg/3.0
#             pgC = pg/3.0
#             qgA = qg/3.0
#             qgB = qg/3.0
#             qgC = qg/3.0
#             gen3p_list.append([
#                 gen_counter, bus_id, status,
#                 vg, vg, vg,
#                 pgA, pgB, pgC,
#                 qgA, qgB, qgC
#             ])
#     mpc["gen3p"] = np.array(gen3p_list, dtype=float)
#
#     # 6) If you want a "node_order" that enumerates each bus-phase
#     #    (like "busX.1", "busX.2", "busX.3"), you can build it here
#     #    from the entire circuit or from AllNodeNames():
#     all_nodenames = dss.Circuit.AllNodeNames()  # e.g. ["bus1.1", "bus1.2", ...]
#     # but check if they are consistent with LN or LL naming
#     mpc["node_order"] = [nm.lower() for nm in all_nodenames]
#
#     return mpc
#
#
# def build_global_y_per_unit(mpc, dss_filename):
#     """
#     Builds the global Y matrix in per-unit by stamping lines/transformers YPrim
#     according to the node order and bus LN bases in the parsed 'mpc'.
#
#     Args:
#       mpc (dict): structure returned by parse_opendss_to_mpc
#       dss_filename (str): path to the same main .dss used in parse
#     Returns:
#       Y_pu (csc_matrix): system admittance matrix in p.u.
#       node_order (list[str]): the node names in the same row/col order as Y_pu
#     """
#     # 1) Reload the circuit
#     dss.Command("Clear")
#     dss.Command(f'Redirect "{dss_filename}"')
#     dss.Solution.Solve()
#
#     # 2) We'll rely on the same node_order from parse_mpc
#     if "node_order" not in mpc:
#         raise ValueError("mpc is missing 'node_order'; cannot build Y in consistent indexing.")
#
#     node_order = mpc["node_order"]
#     node_index_map = {nd: i for i, nd in enumerate(node_order)}
#     n_nodes = len(node_order)
#
#     # Global Y in LIL format
#     Y_global = lil_matrix((n_nodes, n_nodes), dtype=complex)
#
#     def get_local_node_names():
#         """
#         Returns a list of fully qualified node names for the active element.
#         We remove any extra dot-phases from the actual bus name.
#         """
#         phases = dss.CktElement.NodeOrder()  # e.g. [1,2,3,1,2,3]
#         bus_names = dss.CktElement.BusNames()  # e.g. ["p1.1.2.3", "p2.1.2.3"]
#         n_terminals = dss.CktElement.NumTerminals()
#         total_nodes = len(phases)
#         n_nodes_per_term = total_nodes // n_terminals
#
#         local_nodes = []
#         idx = 0
#         for t in range(n_terminals):
#             raw_bus_full = bus_names[t].lower()
#             # Split off any dot after the first
#             bus_split = raw_bus_full.split('.', 1)  # ["p1", "1.2.3"] if raw_bus_full="p1.1.2.3"
#             bus_base = bus_split[0]  # "p1"
#
#             for _ in range(n_nodes_per_term):
#                 ph = phases[idx]
#                 idx += 1
#                 if ph == 0:
#                     local_nodes.append(f"{bus_base}.0")
#                 else:
#                     local_nodes.append(f"{bus_base}.{ph}")
#         return local_nodes
#
#     def stamp_yprim(yprim_matrix, local_nodes):
#         dim = yprim_matrix.shape[0]
#         for r in range(dim):
#             nr = local_nodes[r].lower()
#             if nr not in node_index_map:
#                 continue
#             r_glob = node_index_map[nr]
#             for c in range(dim):
#                 nc = local_nodes[c].lower()
#                 if nc not in node_index_map:
#                     continue
#                 c_glob = node_index_map[nc]
#                 Y_global[r_glob, c_glob] += yprim_matrix[r, c]
#
#     # 3) Loop over Lines, Transformers (skip loads/gens)
#     # Lines
#     for ln_name in dss.Lines.AllNames():
#         dss.Lines.Name(ln_name)
#         yprim = dss.CktElement.YPrim()
#         cplx = []
#         for i in range(0, len(yprim), 2):
#             re = yprim[i]
#             im = yprim[i+1]
#             cplx.append(re + 1j*im)
#         n_loc = int(math.sqrt(len(cplx)))
#         y_mat = np.array(cplx).reshape((n_loc, n_loc))
#         loc_nodes = get_local_node_names()
#         stamp_yprim(y_mat, loc_nodes)
#
#     # Transformers
#     for xf_name in dss.Transformers.AllNames():
#         dss.Transformers.Name(xf_name)
#         yprim = dss.CktElement.YPrim()
#         cplx = []
#         for i in range(0, len(yprim), 2):
#             re = yprim[i]
#             im = yprim[i+1]
#             cplx.append(re + 1j*im)
#         n_loc = int(math.sqrt(len(cplx)))
#         y_mat = np.array(cplx).reshape((n_loc, n_loc))
#         loc_nodes = get_local_node_names()
#         stamp_yprim(y_mat, loc_nodes)
#
#     # Convert Y_global to csc
#     Y_phys = Y_global.tocsc()
#
#     # 4) Convert to per-unit: Y_pu = (1 / S_base) * D * Y_phys * D
#     # where D[i,i] = base voltage (LN) for that node, in volts.
#     S_base = mpc["baseMVA"] * 1e6
#     D_vec = np.ones(n_nodes)
#
#     # We'll parse node_order: each node is "bus.phase"
#     # We look up bus in mpc["bus_vbase_ln"], then store it in D_vec[i].
#     for i, nd in enumerate(node_order):
#         bus_only = nd.split(".")[0]  # "busx"
#         if bus_only in mpc["bus_vbase_ln"]:
#             ln_base = mpc["bus_vbase_ln"][bus_only]
#             D_vec[i] = ln_base  # in volts
#         else:
#             D_vec[i] = 1.0
#
#     D_mat = diags(D_vec, 0, shape=(n_nodes, n_nodes), dtype=complex)
#
#     Ytemp = (D_mat @ Y_phys) @ D_mat
#     Y_pu = (1.0 / S_base) * Ytemp
#
#     return Y_pu, node_order
#
#
# def main():
#     # Example usage with Master.dss:
#     dss_filename = "Master.DSS"
#
#     # 1) Parse
#     mpc = parse_opendss_to_mpc(dss_filename, baseMVA=1.0)
#     # You now have bus, line, xfmr, load data, plus bus_vbase_ln, node_order, etc.
#
#     # 2) Build Y in per-unit
#     Y_pu, node_ord = build_global_y_per_unit(mpc, dss_filename)
#
#     # 3) Print or debug
#     print("Node order from parse:", node_ord)
#     print("Per-unit Y matrix shape:", Y_pu.shape)
#     # e.g. convert small portion to dense for printing:
#     # Y_dense = Y_pu[:10,:10].toarray()
#     # print(Y_dense)
#
#
# if __name__ == "__main__":
#     main()

import opendssdirect as dss
import numpy as np
import math
from scipy.sparse import lil_matrix, csc_matrix, diags


def parse_opendss_to_mpc(dss_filename, baseMVA=1.0):
    """
    1) Clears and redirects the OpenDSS model in 'dss_filename'.
    2) Solves the circuit (so we can read bus voltages, load powers, etc.).
    3) Parses Buses, Lines, Transformers, Loads, Gens, storing in an 'mpc' dict
       with fields:
         mpc["bus3p"], mpc["line3p"], mpc["xfmr3p"], mpc["load3p"], mpc["gen3p"]
         mpc["bus_vbase_ln"], mpc["busname_to_id"], mpc["node_order"], ...
    """

    dss.Command("Clear")
    dss.Command(f'Redirect "{dss_filename}"')
    dss.Solution.Solve()

    circuit = dss.Circuit
    bus_names = circuit.AllBusNames()

    mpc = {}
    mpc["baseMVA"] = baseMVA
    mpc["freq"] = 60.0

    # We'll store each bus's LN base voltage in: mpc["bus_vbase_ln"][bus_lc] = <volts>
    mpc["bus_vbase_ln"] = {}

    # We'll store "bus_name -> integer bus_id"
    busname_to_id = {}
    # We'll store bus data: [bus_id, type, base_kV_LL, VmA, VmB, VmC, VaA, VaB, VaC]
    bus3p_list = []

    # 1) Build bus table
    for i, busname in enumerate(bus_names, start=1):
        bus_lc = busname.lower()
        busname_to_id[bus_lc] = i

        dss.Circuit.SetActiveBus(busname)
        kv_ln = dss.Bus.kVBase()  # LN base if the .dss is that way
        if kv_ln is None or kv_ln < 1e-9:
            kv_ln = 1.0
        kv_ll = kv_ln * math.sqrt(3)

        # store LN base in volts
        mpc["bus_vbase_ln"][bus_lc] = kv_ln * 1e3

        # example: bus_type = 3 if bus == "p1", else 1
        bus_type = 3 if (bus_lc == "sourcebus") else 1

        # post-solve bus voltages
        nodes = dss.Bus.Nodes()  # e.g. [1,2,3]
        magangle = dss.Bus.VMagAngle()  # e.g. [V1, ang1, V2, ang2, ...]
        pair_list = list(zip(magangle[0::2], magangle[1::2]))

        vm = [0.0, 0.0, 0.0]
        va = [0.0, 0.0, 0.0]
        for idx_phase, node_num in enumerate(nodes):
            if idx_phase < len(pair_list):
                vmag = pair_list[idx_phase][0]  # in volts
                vang_deg = pair_list[idx_phase][1]
                v_pu = (vmag)/(kv_ln*1e3) if kv_ln>1e-9 else 0.0
                vm[idx_phase] = v_pu
                va[idx_phase] = math.radians(vang_deg)

        bus3p_list.append([
            i,                     # bus_id
            bus_type,              # type
            kv_ll,                 # base_kV_LL
            vm[0], vm[1], vm[2],
            va[0], va[1], va[2]
        ])

    mpc["bus3p"] = np.array(bus3p_list, dtype=float)
    mpc["busname_to_id"] = busname_to_id

    # 2) Gather lines, xfmrs, loads, gens
    all_elems = circuit.AllElementNames()

    line_data = []
    line_counter = 0
    for elem in all_elems:
        if elem.lower().startswith("line."):
            line_counter += 1
            dss.Circuit.SetActiveElement(elem)
            fb = dss.CktElement.BusNames()[0].split(".")[0].lower()
            tb = dss.CktElement.BusNames()[1].split(".")[0].lower()
            fbid = busname_to_id.get(fb, 0)
            tbid = busname_to_id.get(tb, 0)

            st = 1 if dss.CktElement.Enabled() else 0
            ln_name = elem.split(".",1)[1]
            dss.Lines.Name(ln_name)
            ln_length = dss.Lines.Length()
            ln_code = dss.Lines.LineCode()
            lcid = 0

            line_data.append([
                line_counter,
                fbid,
                tbid,
                st,
                lcid,
                ln_length,
                ln_name.lower()
            ])
    mpc["line3p"] = np.array(line_data, dtype=object)

    xfmr_data = []
    xfmr_counter = 0
    for elem in all_elems:
        if elem.lower().startswith("transformer."):
            xfmr_counter += 1
            dss.Circuit.SetActiveElement(elem)
            xf = dss.Transformers

            buses = dss.CktElement.BusNames()
            fb = buses[0].split(".")[0].lower()
            tb = buses[1].split(".")[0].lower()
            fbid = busname_to_id.get(fb,0)
            tbid = busname_to_id.get(tb,0)

            st = 1 if dss.CktElement.Enabled() else 0
            xf.First()
            xf.Wdg(1)
            w1_kv = xf.kV()
            w1_kva = xf.kVA()
            xf.Wdg(2)
            w2_kv = xf.kV()
            w2_kva = xf.kVA()

            r = xf.Rneut()
            x = xf.Xhl()
            basekVA = max(w1_kva, w2_kva)

            xfmr_data.append([
                xfmr_counter,
                fbid,
                tbid,
                st,
                r, x,
                basekVA,
                w1_kv, w2_kv
            ])
    mpc["xfmr3p"] = np.array(xfmr_data, dtype=float)

    # loads
    load_data = []
    ld_counter=0
    for elem in all_elems:
        if not elem.lower().startswith("load."):
            continue
        ld_counter+=1
        dss.Circuit.SetActiveElement(elem)
        bus_full = dss.CktElement.BusNames()[0]
        bus_only = bus_full.split(".")[0].lower()
        bid = busname_to_id.get(bus_only, 0)
        st = 1 if dss.CktElement.Enabled() else 0

        pA=pB=pC=0.0
        qA=qB=qC=0.0
        powers = dss.CktElement.Powers()
        n_cond = dss.CktElement.NumConductors()
        nodes = dss.CktElement.NodeOrder()

        for i_cond in range(n_cond):
            p_cond = powers[2*i_cond]
            q_cond = powers[2*i_cond+1]
            ph_node = nodes[i_cond] if i_cond<len(nodes) else 0
            if ph_node==1:
                pA+=p_cond; qA+=q_cond
            elif ph_node==2:
                pB+=p_cond; qB+=q_cond
            elif ph_node==3:
                pC+=p_cond; qC+=q_cond
            else:
                pass
        load_data.append([
            ld_counter, bid, st,
            pA, pB, pC,
            qA, qB, qC
        ])
    mpc["load3p"] = np.array(load_data, dtype=float)

    # gens
    gen_data=[]
    gen_counter=0
    for elem in all_elems:
        if elem.lower().startswith("generator."):
            gen_counter+=1
            dss.Circuit.SetActiveElement(elem)
            gn_bus = dss.CktElement.BusNames()[0].split(".")[0].lower()
            bid = busname_to_id.get(gn_bus,0)
            st = 1 if dss.CktElement.Enabled() else 0
            dss.Generators.Name(elem.split(".",1)[1])
            pg = dss.Generators.kW()
            qg = dss.Generators.kvar()
            vg = 1.0
            # assume balanced
            pgA=pg/3.0; pgB=pg/3.0; pgC=pg/3.0
            qgA=qg/3.0; qgB=qg/3.0; qgC=qg/3.0
            gen_data.append([
                gen_counter, bid, st,
                vg,vg,vg,
                pgA,pgB,pgC,
                qgA,qgB,qgC
            ])
    mpc["gen3p"]= np.array(gen_data, dtype=float)

    # node_order from DSS (the original full circuit)
    # might or might not be used later
    all_nodes = dss.Circuit.AllNodeNames()
    mpc["node_order"] = [nm.lower() for nm in all_nodes]

    return mpc


def merge_closed_switches_in_mpc_and_dss(mpc, switch_threshold=1e-5):
    """
    Merges topologically redundant nodes in 'mpc' that are connected by
    near-zero-impedance lines, and also disables those lines in the OpenDSS
    environment so that we don't see them in YPrim stamping.

    Then it reassigns bus IDs in bus3p from 1..N. Also updates mpc["busname_to_id"]
    so the old bus names map to the final bus IDs.

    This ensures the final bus set & line set in 'mpc' is consistent with
    the updated DSS circuit from which we'll build Y.
    """
    line_arr = mpc["line3p"]  # shape [N,7], columns: [line_id, fbus, tbus, status, lcid, length, line_name]
    if line_arr.shape[1]<7:
        raise ValueError("mpc['line3p'] must have 7 columns: [line_id,fbus,tbus,status,lcid,length,line_name].")

    bus3p = mpc["bus3p"]
    bus_ids = bus3p[:,0].astype(int).tolist()

    parent={}
    def find(x):
        if parent[x]!=x:
            parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb:
            parent[rb]=ra
    for b in bus_ids:
        parent[b]=b

    import opendssdirect as dss
    # unify & disable lines under threshold
    for row in line_arr:
        _, fb, tb, st, _, length_mi, ln_name = row
        if st==1 and length_mi<switch_threshold:
            union(int(fb), int(tb))
            # disable in DSS
            dss.Lines.Name(ln_name)
            dss.Command(f"Edit line.{ln_name} enabled=no")

    rep_map={}
    for b in bus_ids:
        rep_map[b]=find(b)

    # rewrite bus, line, xfmr, load, gen
    for row in bus3p:
        old_id=int(row[0])
        row[0]=rep_map[old_id]
    for row in line_arr:
        row[1]=rep_map[int(row[1])]
        row[2]=rep_map[int(row[2])]
    if "xfmr3p" in mpc:
        for row in mpc["xfmr3p"]:
            row[1]=rep_map[int(row[1])]
            row[2]=rep_map[int(row[2])]
    if "load3p" in mpc:
        for row in mpc["load3p"]:
            row[1]=rep_map[int(row[1])]
    if "gen3p" in mpc:
        for row in mpc["gen3p"]:
            row[1]=rep_map[int(row[1])]

    # unify duplicates in bus3p
    unique_rows={}
    for row in bus3p:
        bid=int(row[0])
        if bid not in unique_rows:
            unique_rows[bid]=row
    new_bus_list=list(unique_rows.values())
    new_bus_arr=np.array(new_bus_list,dtype=float)
    new_bus_arr=new_bus_arr[new_bus_arr[:,0].argsort()]
    mpc["bus3p"] = new_bus_arr

    # reassign bus IDs => 1..N
    old_ids = new_bus_arr[:,0].astype(int)
    new_id_map={}
    for i,oid in enumerate(old_ids, start=1):
        new_id_map[oid]=i

    for row in mpc["bus3p"]:
        row[0]=new_id_map[int(row[0])]
    for row in line_arr:
        row[1]=new_id_map[int(row[1])]
        row[2]=new_id_map[int(row[2])]
    if "xfmr3p" in mpc:
        for row in mpc["xfmr3p"]:
            row[1]=new_id_map[int(row[1])]
            row[2]=new_id_map[int(row[2])]
    if "load3p" in mpc:
        for row in mpc["load3p"]:
            row[1]=new_id_map[int(row[1])]
    if "gen3p" in mpc:
        for row in mpc["gen3p"]:
            row[1]=new_id_map[int(row[1])]

    # remove lines that self-loop
    new_lines=[]
    for row in line_arr:
        if row[1]==row[2]:
            row[3]=0
            # skip
        else:
            new_lines.append(row)
    mpc["line3p"] = np.array(new_lines,dtype=object)

    # ------ Now update mpc["busname_to_id"] so old bus names point to final IDs ------
    old_dict = mpc["busname_to_id"]  # e.g. { "p2":2, "p3":3, ... }
    new_dict = {}
    for bus_nm, old_id in old_dict.items():
        # find rep => final => new
        rep_id = rep_map.get(old_id, old_id)
        final_id = new_id_map.get(rep_id, 0)
        new_dict[bus_nm] = final_id
    mpc["busname_to_id"] = new_dict

    print("merge_closed_switches_in_mpc_and_dss: done merging & disabling near-zero lines.")
    merge_parallel_transmission_lines_in_mpc(mpc)


def build_global_y_per_unit(mpc, dss_filename):
    """
    Build the global Y matrix in per-unit for the *merged* system.
    We iterate over the *original* DSS circuit lines/xfmrs, but map each bus name
    to the final bus ID in mpc.

    Steps:
      1) We'll create a 3-phase node for each bus_id in mpc["bus3p"] => (bus_id, ph=1..3).
      2) We'll stamp YPrim from each enabled line/xfmr into Y_global
         using the merged bus ID & phase for row/col.
      3) We'll convert Y_global to per-unit => Y_pu.

    Returns:
      Y_pu (csc_matrix),
      node_order (list of str) => e.g. ["bus1.1","bus1.2","bus1.3","bus2.1",...]
    """
    # We'll rely on the *current* DSS circuit in memory (after merges) but
    # the bus IDs are in mpc.

    # Build bus-phase indexing
    bus3p = mpc["bus3p"]  # shape Nx9
    bus_ids = bus3p[:,0].astype(int)
    # We'll do a stable sort
    sorted_buses = np.sort(bus_ids)

    # Build a list of (bus_id,phase) => global index
    node_phase_map = {}
    node_order_list = []
    idx=0
    for b in sorted_buses:
        for ph in (1,2,3):
            node_phase_map[(b,ph)] = idx
            node_order_list.append(f"bus{b}.{ph}")
            idx+=1
    n_nodes = idx

    # Make an empty Y in LIL
    Y_global = lil_matrix((n_nodes, n_nodes), dtype=complex)

    # We'll define a helper to map DSS bus -> final bus ID -> row index
    def map_local_nodes():
        """
        Returns a list of length local_dim for the active element,
        mapping each conductor to the final row index, or -1 if invalid.
        """
        phases = dss.CktElement.NodeOrder()    # e.g. [1,2,3,1,2,3]
        busnames = dss.CktElement.BusNames()   # e.g. ["p2.1.2.3","p4.1.2.3"]
        n_term = dss.CktElement.NumTerminals()
        total_conds = len(phases)
        npt = total_conds//n_term

        # We'll use the final "mpc['busname_to_id']" that was updated
        bus_dict = mpc["busname_to_id"]

        idx_list = []
        idx_ph=0
        for t in range(n_term):
            raw_bus = busnames[t].lower()
            base_bus = raw_bus.split('.',1)[0] # e.g. "p2"

            # final bus ID
            if base_bus in bus_dict:
                final_b = bus_dict[base_bus]
            else:
                final_b = -1

            for _ in range(npt):
                ph = phases[idx_ph]
                idx_ph+=1
                if final_b<1 or ph==0:
                    idx_list.append(-1)
                else:
                    # look up row
                    if (final_b, ph) in node_phase_map:
                        rowcol = node_phase_map[(final_b,ph)]
                    else:
                        rowcol = -1
                    idx_list.append(rowcol)
        return idx_list

    import opendssdirect as dss
    # 1) iterate lines
    for ln in dss.Lines.AllNames():
        dss.Lines.Name(ln)
        if not dss.CktElement.Enabled():
            continue
        # YPrim
        Yp = dss.CktElement.YPrim()
        cplx=[]
        for i in range(0,len(Yp),2):
            cplx.append(Yp[i]+1j*Yp[i+1])
        dim_local = int(math.sqrt(len(cplx)))
        y_mat = np.array(cplx).reshape((dim_local,dim_local))

        # map local
        loc_inds = map_local_nodes()
        for r in range(dim_local):
            nr = loc_inds[r]
            if nr<0:
                continue
            for c in range(dim_local):
                nc = loc_inds[c]
                if nc<0:
                    continue
                Y_global[nr,nc]+=y_mat[r,c]

    # 2) iterate transformers
    for xf in dss.Transformers.AllNames():
        dss.Transformers.Name(xf)
        if not dss.CktElement.Enabled():
            continue

        Yp = dss.CktElement.YPrim()
        cplx=[]
        for i in range(0,len(Yp),2):
            cplx.append(Yp[i]+1j*Yp[i+1])
        dim_local = int(math.sqrt(len(cplx)))
        y_mat = np.array(cplx).reshape((dim_local,dim_local))

        loc_inds = map_local_nodes()
        for r in range(dim_local):
            nr=loc_inds[r]
            if nr<0:
                continue
            for c in range(dim_local):
                nc=loc_inds[c]
                if nc<0:
                    continue
                Y_global[nr,nc]+=y_mat[r,c]

    # Convert to csc
    Y_phys = Y_global.tocsc()

    # 3) Convert to per-unit
    # Y_pu = (1/S_base)* D * Y_phys * D,
    # where D[i,i] = bus LN base for that node i
    S_base = mpc["baseMVA"]*1e6

    # build D
    D_vec = np.ones(n_nodes,dtype=float)
    for i,nd_name in enumerate(node_order_list):
        # e.g. "bus3.2" => bus3, phase2
        # parse "bus3" => "3"
        buslabel = nd_name.split('.',1)[0]   # "bus3"
        # remove leading "bus"
        busnum_str = buslabel[3:]   # "3"
        bus_id_int = int(busnum_str)
        # find LN base in mpc["bus_vbase_ln"]
        # we invented a new name "bus3",
        # so we need a mapping from bus_id->some old name or we can store in bus3p
        # but simpler: bus3p row => row[2..], or we can keep another dict
        # We'll do a pass that picks the old bus name from merges?
        # Alternatively, we store LN base in bus3p. Let's do that quickly:
        # => we can do a quick dictionary bus_id-> LN_base
        # We'll build that above:
        # for now, let's do it by scanning bus3p
        pass

    # A simpler approach: store LN base in bus3p
    # e.g. bus3p columns => [bus_id, type, base_kV_LL, ...]
    # => LN base = row[2]/sqrt(3)*1e3 if row[2] is in kV
    bus_id_to_Vln = {}
    for row in bus3p:
        b_id = int(row[0])
        kv_ll = float(row[2])  # e.g. 12.47
        # LN base in volts:
        vln = (kv_ll/math.sqrt(3))*1e3
        bus_id_to_Vln[b_id] = vln

    # now fill D_vec
    for i,nd_name in enumerate(node_order_list):
        buslab = nd_name.split('.',1)[0] # e.g. "bus3"
        bus_id_int = int(buslab[3:])     # 3
        if bus_id_int in bus_id_to_Vln:
            D_vec[i] = bus_id_to_Vln[bus_id_int]
        else:
            D_vec[i] = 1.0

    D_mat = diags(D_vec,0,shape=(n_nodes,n_nodes),dtype=complex)
    Ytemp = (D_mat@Y_phys)@D_mat
    Y_pu = (1.0/S_base)*Ytemp

    return Y_pu, node_order_list
def merge_parallel_transmission_lines_in_mpc(mpc):
    """
    Detects multiple parallel lines in mpc["line3p"] that have the same
    (fbus, tbus, line code (lcid), length, status=1), and merges them into a
    single equivalent line by dividing the length by the number of parallels.

    E.g., if there are 5 identical lines from busA to busB, each with length=10 ft,
    we keep exactly one line with length=10/5=2 ft in the final line3p array.

    NOTE: This approach also reduces the shunt admittance by factor of N, which
    may or may not be physically correct for parallel lines.
    """
    if "line3p" not in mpc:
        return
    lines = mpc["line3p"]  # shape [N, 7] => [line_id, fbus, tbus, status, lcid, length, line_name]
    if lines.shape[1] < 7:
        return  # or raise an error

    # We’ll group lines by (fbus, tbus, lcid, length, status)
    # If you only want to merge lines that are all 'enabled' => status=1 => skip status=0
    # If you also want to merge lines ignoring length differences, adapt as needed.
    from collections import defaultdict

    # key = (fbus, tbus, lcid, length, status)
    groups = defaultdict(list)
    for row in lines:
        line_id = row[0]
        fbus = row[1]
        tbus = row[2]
        status = row[3]
        lcid = row[4]
        length_mi = row[5]
        name = row[6]

        # Only group lines that are "enabled"
        if status == 1:
            key = (fbus, tbus, lcid, length_mi, status)
            groups[key].append(row)
        else:
            # keep disabled lines as-is in some "disabled_lines" bucket
            # or you can ignore them entirely
            pass

    merged_lines = []
    disabled_or_unique_lines = []

    # Now for each group, if it has N>1 lines,
    # we'll keep exactly one line with length=(original_length/N).
    for key, row_list in groups.items():
        if len(row_list) == 1:
            # no parallel line => keep as-is
            merged_lines.append(row_list[0])
        else:
            # multiple lines in parallel
            # pick the "first" line as a template
            # or you can create a brand-new line_id, line_name
            first = row_list[0].copy()
            N = len(row_list)

            # e.g. original length
            original_len = first[5]
            # scale by 1/N
            eq_len = original_len / N

            # update the first row’s length
            first[5] = eq_len

            # optionally rename the line or keep the original
            # e.g. first[6] = "merged_" + str(first[6])
            # or keep the original name
            # first[6] = first[6]  # do nothing

            merged_lines.append(first)

            # If you want to "disable" or remove all the others in parallel,
            # you can do so. For example, set row[3]=0 for them:
            for r in row_list[1:]:
                r[3] = 0  # status=0
                disabled_or_unique_lines.append(r)

    # Now we also want to keep lines that were “status=0” or not grouped
    # but not overshadow them if we already put them in disabled_or_unique_lines.
    # So we do a pass for all lines that are not in the big groups or are disabled:
    for row in lines:
        # If it was status=1 and grouped => we’ve handled it
        if row[3] == 0:
            # it’s disabled => keep it if you want
            # or skip if you don’t want them in final
            disabled_or_unique_lines.append(row)
        else:
            # If it’s status=1 but never ended up in groups => means it’s length=0?
            # Or we skip it. Usually, everything with status=1 was in a group
            pass

    # Combine them back: keep the merged lines + any disabled ones, etc.
    # final_list = merged_lines + disabled_or_unique_lines
    final_list = merged_lines

    # Convert to array, sorted by line_id if you want
    final_arr = np.array(final_list, dtype=object)
    # optionally sort by [line_id]
    final_arr = final_arr[final_arr[:,0].argsort()]

    mpc["line3p"] = final_arr

    print("merge_parallel_transmission_lines_in_mpc: finished merging parallel lines.\n")

def main():
    dss_filename="Master.DSS"

    # 1) parse
    mpc = parse_opendss_to_mpc(dss_filename, baseMVA=1.0)
    # 2) merges
    merge_closed_switches_in_mpc_and_dss(mpc, switch_threshold=2)
    # 3) build Y
    Y_pu, node_ord = build_global_y_per_unit(mpc, dss_filename)
    print("Final Y shape:", Y_pu.shape)
    print("Node order:", node_ord)


if __name__=="__main__":
    main()
