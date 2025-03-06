import numpy as np
import opendssdirect as dss
from scipy.sparse import csc_matrix, lil_matrix


def build_global_y_per_unit(line_changes=None):
    """
    Returns:
        Y_pu (csc_matrix): The system admittance matrix in per-unit
        node_order (list of str): the global node names in the same order
                                  as the rows/columns of Y_pu
    """
    if line_changes is None:
        line_changes = []
    line_changes_map = {}
    for item in line_changes:
        ln = item['line_name'].lower()
        # remove 'line_name' from dictionary
        dcopy = dict(item)
        dcopy.pop('line_name', None)
        line_changes_map[ln] = dcopy
    # 1) Gather all node names from the circuit
    all_nodes = dss.Circuit.AllNodeNames()  # e.g. ["bus1.1", "bus1.2", ...]
    # We'll make a dictionary mapping node_name -> row/col index
    node_index_map = {}
    for i, node in enumerate(all_nodes):
        node_index_map[node.lower()] = i
    n_nodes = len(all_nodes)

    # 2) Create a global Y in a sparse format, initially empty
    Y_global = lil_matrix((n_nodes, n_nodes), dtype=complex)
    def get_local_node_names():
        """
        Returns a list of fully qualified node names for the active circuit element.
        For example, if dss.CktElement.NodeOrder() returns [1,2,3,0,1,2,3,0] and
        dss.CktElement.BusNames() returns ["bus1", "bus2"], then the function returns:
          ['bus1.1', 'bus1.2', 'bus1.3', 'bus1.gnd',
           'bus2.1', 'bus2.2', 'bus2.3', 'bus2.gnd']
        """
        # 1) Retrieve the node order from the active element.
        phases = dss.CktElement.NodeOrder()  # e.g. [1, 2, 3, 0, 1, 2, 3, 0]

        # 2) Retrieve the bus names for the element's terminals.
        bus_names = dss.CktElement.BusNames()  # e.g. ["bus1", "bus2"]

        # 3) Get the number of terminals.
        n_terminals = dss.CktElement.NumTerminals()  # e.g. 2

        # 4) Determine the number of nodes per terminal.
        total_nodes = len(phases)
        n_nodes_per_terminal = total_nodes // n_terminals

        # 5) Build the list of fully qualified node names.
        local_node_names = []
        idx = 0
        for term_idx in range(n_terminals):
            bus_base = bus_names[term_idx].lower()  # e.g. 'bus1'
            for j in range(n_nodes_per_terminal):
                ph_number = phases[idx]
                idx += 1
                if ph_number == 0:
                    node_full = f"{bus_base}.0"
                else:
                    node_full = f"{bus_base}.{ph_number}"
                local_node_names.append(node_full)

        return local_node_names

    def modify_line_admittance_3x3(y3, param_label, factor):
        """
        y3: a 3x3 complex np.array with phases [a,b,c] in order
        param_label: which parameter to scale, e.g. 'phase_a', 'phase_ab', ...
        factor: how much to scale that parameter (e.g. 0.01, 10, 100, etc.)
        """
        y3_mod = y3.copy()
        if param_label.lower() == 'phase_a':
            y3_mod[0, 0] = y3_mod[0, 0] * factor

        elif param_label.lower() == 'phase_b':
            y3_mod[1, 1] = y3_mod[1, 1] * factor

        elif param_label.lower() == 'phase_c':
            y3_mod[2, 2] = y3_mod[2, 2] * factor

        elif param_label.lower() == 'phase_ab':
            y3_mod[0, 1] = y3_mod[0, 1] * factor
            y3_mod[1, 0] = y3_mod[1, 0] * factor  # keep symmetry

        elif param_label.lower() == 'phase_bc':
            y3_mod[1, 2] = y3_mod[1, 2] * factor
            y3_mod[2, 1] = y3_mod[2, 1] * factor

        elif param_label.lower() == 'phase_ac':
            y3_mod[0, 2] = y3_mod[0, 2] * factor
            y3_mod[2, 0] = y3_mod[2, 0] * factor
        else:
            pass
        return y3_mod
    def stamp_yprim(yprim_complex, node_order_local):
        """
        yprim_complex: 2D np.array of shape (2nphases, 2nphases)
        node_order_local: list of node names for those 2nph columns/rows
        We'll add yprim_complex to Y_global according to node_index_map
        """
        dim = yprim_complex.shape[0]
        for r in range(dim):
            node_r_name = node_order_local[r].lower()
            if node_r_name not in node_index_map:
                # possibly a node is unconnected or load side
                continue
            r_glob = node_index_map[node_r_name]
            for c in range(dim):
                node_c_name = node_order_local[c].lower()
                if node_c_name not in node_index_map:
                    continue
                c_glob = node_index_map[node_c_name]
                Y_global[r_glob, c_glob] += yprim_complex[r, c]

    # 3) We loop over Lines, Transformers and stamp them to the Y_global
    # Lines
    for line_name in dss.Lines.AllNames():
        dss.Lines.Name(line_name)
        yprim_flat = dss.CktElement.YPrim()
        local_nodes = get_local_node_names()
        # Convert the flat array
        cplx_list = []
        for i in range(0, len(yprim_flat), 2):
            cplx_list.append(yprim_flat[i] + 1j*yprim_flat[i+1])
        n_local = int(np.sqrt(len(cplx_list)))
        yprim_matrix = np.array(cplx_list).reshape((n_local, n_local))
        changes_for_line = line_changes.get(line_name.lower(), {})
        y3_modified = yprim_matrix.copy()
        for param_label, factor in changes_for_line.items():
            # e.g. param_label='phase_a', factor=100
            y3_modified = modify_line_admittance_3x3(y3_modified, param_label, factor)
        # Stamp into global
        stamp_yprim(y3_modified, local_nodes)

    # 3.2) Transformers
    for xfmr_name in dss.Transformers.AllNames():
        dss.Transformers.Name(xfmr_name)
        yprim_flat = dss.CktElement.YPrim()
        local_nodes = get_local_node_names()
        cplx_list = []
        for i in range(0, len(yprim_flat), 2):
            cplx_list.append(yprim_flat[i] + 1j * yprim_flat[i + 1])
        yprim_array = np.array(cplx_list)
        n_local = int(np.sqrt(len(cplx_list)))
        yprim_matrix = yprim_array.reshape((n_local, n_local))
        stamp_yprim(yprim_matrix, local_nodes)
    # (Optionally, we could also stamp capacitors, etc., if they exist
    #  but skip loads and generators as requested.)

    # 4) Now we have Y_global in physical units (Siemens).
    #    We want to convert it to per-unit. Each node may have a different base voltage.

    # Build a diagonal scaling vector for each node:
    #    V_base[node] in Volts or kV,
    #    (plus possibly 1/S_base to get a consistent power base).
    # For example, if each bus's line-to-neutral base is "bus_kV * 1e3/sqrt(3)",
    # we store D[i] = 1 / (V_base[i]) so that v_pu = D * v_phys.
    # Then Y_pu = D * Y_global * (1/D) = D * Y_global * D^{-1}  (if ignoring power base),
    # or Y_pu = (1/S_base) * (D * Y_global * D).

    # For demonstration, let's assume we do line-to-neutral base in V (not kV).
    # We'll store each bus's LN base voltage in a dictionary:
    bus_baseLN = {}
    # We can get bus base from the Bus kV. If we have 4 buses, or do a pass:
    for busname in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(busname)
        kv_ll = dss.Bus.kVBase()  # line-to-line base in kV
        if kv_ll <= 0:
            kv_ll = 12.47  # fallback if something is missing
        # kv_ln = kv_ll / np.sqrt(3)
        # store in volts
        bus_baseLN[busname.lower()] = kv_ll * 1e3

    # for xfmr_name in dss.Transformers.AllNames():
    #     dss.Transformers.Name(xfmr_name)
    #     xfmr_buses = dss.CktElement.BusNames() # e.g., ["bus2", "bus3"]
    #     if len(xfmr_buses) >= 2:
    #         hv_bus = xfmr_buses[0].lower()  # HV side
    #         lv_bus = xfmr_buses[1].lower()  # LV side
    #         if hv_bus in bus_baseLN:
    #             # Override the LV bus's LN base with the HV bus's LN base.
    #             bus_baseLN[lv_bus] = bus_baseLN[hv_bus]

    # build a vector diag for each node
    D_vec = np.ones(n_nodes, dtype=complex)
    S_base = 1e6  # 1 MVA, for example
    for i, nd in enumerate(all_nodes):
        # nd might be "busname.ph"
        bn = nd.split(".")[0].lower()
        if bn in bus_baseLN:
            V_ln = bus_baseLN[bn]  # in volts
            D_vec[i] = V_ln
        else:
            D_vec[i] = 0  # or 1.0 if unrecognized

    # we build a diagonal matrix from D_vec
    # for a full per-unit approach (with a 1 MVA base), we do:
    # Y_pu = (1/S_base) * D * Y_global * D
    # We'll keep it simpler in one pass:
    D_mat = lil_matrix((n_nodes, n_nodes), dtype=complex)
    for i in range(n_nodes):
        D_mat[i, i] = D_vec[i]

    # Convert Y_global to csc
    Y_global_csc = Y_global.tocsc()
    D_csc = D_mat.tocsc()

    # Y_pu = (1/S_base) * (D_csc * Y_global_csc * D_csc)
    # watch out for potential large matrix -> be sure the system is small
    Ytemp = D_csc @ Y_global_csc @ D_csc
    Y_pu_csc = (1 / S_base) * Ytemp

    return Y_pu_csc, all_nodes


def main():
    # 1) Load your DSS file
    dss.run_command('Redirect "Master.DSS"')
    dss.Solution.Solve()

    # 2) Build the per-unit Y matrix
    Y_pu, node_order = build_global_y_per_unit(ind)

    print("Global Y in p.u. shape =", Y_pu.shape)
    print("Node order =", node_order)

    # Example: convert to dense if small, or keep as csc
    Y_dense = Y_pu.toarray()
    print("Y_pu (dense) =", Y_dense)

# run the main function

main()
