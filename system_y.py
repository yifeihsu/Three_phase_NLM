import numpy as np
import opendssdirect as dss
from scipy.sparse import csc_matrix, lil_matrix


def build_global_y_per_unit(line_changes=None):
    """
    Builds the global Y matrix in per-unit for lines, transformers, 
    and voltage sources, but excludes loads/generators.
    Returns:
        Y_pu (csc_matrix): The system admittance matrix in per-unit
        node_order (list of str): the global node names in the same order 
                                  as the rows/columns of Y_pu
    """
    if line_changes is None:
        line_changes = {}
    # 1) Gather all node names from the circuit
    all_nodes = dss.Circuit.AllNodeNames()  # e.g. ["bus1.1", "bus1.2", ...]
    # We'll make a dictionary mapping node_name -> row/col index
    node_index_map = {}
    for i, node in enumerate(all_nodes):
        node_index_map[node.lower()] = i
    n_nodes = len(all_nodes)

    # 2) Create a global Y in a sparse format, initially empty
    #    We'll do a LIL (list-of-lists) matrix for easy incremental insertion
    Y_global = lil_matrix((n_nodes, n_nodes), dtype=complex)

    def extract_3x3_core_from_6x6(y6):
        """
        Given a 6x6 matrix representing a 2-terminal, 3-phase line,
        extract the top-left 3x3 block.  Typically, that block is repeated
        (plus a sign) in the other blocks if the line is standard.

        Returns y3 (3x3 complex array).
        """
        y3 = y6[0:3, 0:3].copy()  # top-left submatrix
        return y3

    def build_symmetric_6x6_from_3x3(y3):
        """
        Rebuild a 6x6 matrix in the standard form:
            [ y3,   -y3 ]
            [ -y3,   y3 ]
        This ensures consistent +/- stamping for a 2-terminal line.
        """
        y6 = np.zeros((6, 6), dtype=complex)
        # top-left
        y6[0:3, 0:3] = y3
        # top-right
        y6[0:3, 3:6] = -y3
        # bottom-left
        y6[3:6, 0:3] = -y3
        # bottom-right
        y6[3:6, 3:6] = y3
        return y6

    def modify_line_admittance_3x3(y3, param_label, factor):
        """
        Scale exactly one parameter in the 3x3 block by `factor`.
        The parameter label indicates:
          - which phase/phase-pair (phase_a, phase_b, phase_c, phase_ab, etc.)
          - which component: "_g" for real part (G), "_b" for imaginary part (B).

        e.g. 'phase_a_g' => scale only the real part of phase A self-admittance
             'phase_ab_b' => scale only the imaginary part of mutual A-B admittance
        """
        y3_mod = y3.copy()

        # A helper to scale only the real part for (r,c) and (c,r)
        def scale_symmetric_real(mat, r, c, fac):
            # top-left is mat[r,c]
            old_val = mat[r, c]
            new_val = complex(old_val.real * fac, old_val.imag)
            mat[r, c] = new_val
            # mirror, not necessary symmetric if transformers
            if r != c:
                old_val2 = mat[c, r]
                new_val2 = complex(old_val2.real * fac, old_val2.imag)
                mat[c, r] = new_val2
                dval = new_val2 - old_val2
                if abs(dval) > 1e-12:
                    mat[c, c] -= dval
                    mat[r, r] -= dval

        def scale_symmetric_imag(mat, r, c, fac):
            old_val = mat[r, c]
            new_val = complex(old_val.real, old_val.imag * fac)
            mat[r, c] = new_val
            # mirror
            if r != c:
                old_val2 = mat[c, r]
                new_val2 = complex(old_val2.real, old_val2.imag * fac)
                mat[c, r] = new_val2
                dval = new_val2 - old_val2
                if abs(dval) > 1e-12:
                    mat[c, c] -= dval
                    mat[r, r] -= dval

        # 1) Parse the label: e.g. "phase_ab_g" => location "phase_ab", component "g"
        plow = param_label.lower().strip()
        # separate the location part (phase_a / phase_ab / etc.) from _g or _b
        # For instance, "phase_ab_g" => location_str="phase_ab", comp_str="g"
        if plow.endswith('_g'):
            comp_str = 'g'
            location_str = plow[:-2]  # remove trailing "_g"
        elif plow.endswith('_b'):
            comp_str = 'b'
            location_str = plow[:-2]  # remove trailing "_b"
        else:
            # If it doesn't match the pattern, do nothing
            return y3_mod

        # 2) Identify which indices to apply
        # Diagonal indices for self phases:
        #   A->(0,0), B->(1,1), C->(2,2)
        # Off-diagonal pairs for mutual:
        #   AB->(0,1)/(1,0), BC->(1,2)/(2,1), AC->(0,2)/(2,0)
        def scale_self_phase(phase_idx):
            if comp_str == 'g':
                scale_symmetric_real(y3_mod, phase_idx, phase_idx, factor)
            else:  # comp_str == 'b'
                scale_symmetric_imag(y3_mod, phase_idx, phase_idx, factor)
        # Now assume that the parameters of two terminals will all change for the mutual impedance
        def scale_mutual_phases(r, c):
            if comp_str == 'g':
                scale_symmetric_real(y3_mod, r, c, factor)
            else:
                scale_symmetric_imag(y3_mod, r, c, factor)

        if location_str == 'phase_a':
            scale_self_phase(0)
        elif location_str == 'phase_b':
            scale_self_phase(1)
        elif location_str == 'phase_c':
            scale_self_phase(2)
        elif location_str == 'phase_ab':
            scale_mutual_phases(0, 1)
        elif location_str == 'phase_bc':
            scale_mutual_phases(1, 2)
        elif location_str == 'phase_ac':
            scale_mutual_phases(0, 2)
        else:
            # Unrecognized location => do nothing
            pass

        return y3_mod
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

    # Helper function to place a YPrim block into the global matrix
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

    # 3) We loop over Lines, Transformers, Vsources
    #    and skip Loads, Generators.
    #    For each element, parse YPrim and add it to Y_global.

    # 3.1) Lines
    for line_name in dss.Lines.AllNames():
        dss.Lines.Name(line_name)
        yprim_flat = dss.CktElement.YPrim()
        local_nodes = get_local_node_names()

        # Parse the 6x6
        cplx_list = []
        for i in range(0, len(yprim_flat), 2):
            cplx_list.append(yprim_flat[i] + 1j*yprim_flat[i+1])
        arr = np.array(cplx_list)
        n_local = int(np.sqrt(len(cplx_list)))  # should be 6
        y6_orig = arr.reshape((n_local, n_local))

        # Extract 3x3 top-left
        y3_tl = extract_3x3_core_from_6x6(y6_orig)

        # Check if we have changes for this line
        line_key = line_name.lower()
        if line_key in line_changes:
            for param_label, factor in line_changes[line_key].items():
                y3_tl = modify_line_admittance_3x3(y3_tl, param_label, factor)

        # Rebuild symmetrical 6x6
        y6_mod = build_symmetric_6x6_from_3x3(y3_tl)

        # stamp
        stamp_yprim(y6_mod, local_nodes)
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
    # Y_global_csc = csc_matrix(dss.YMatrix.getYsparse())

    # do matrix multiply: 
    # Y_pu = (1/S_base) * (D_csc * Y_global_csc * D_csc)
    # watch out for potential large matrix -> be sure the system is small
    Ytemp = D_csc @ Y_global_csc @ D_csc
    Y_pu_csc = (1 / S_base) * Ytemp

    return Y_pu_csc, all_nodes


def main():
    # 1) Load your DSS file
    dss.run_command('Redirect "4Bus-YY-Bal.DSS"')
    dss.Solution.Solve()

    # 2) Build the per-unit Y matrix
    Y_pu, node_order = build_global_y_per_unit()

    print("Global Y in p.u. shape =", Y_pu.shape)
    print("Node order =", node_order)

    # Example: convert to dense if small, or keep as csc
    Y_dense = Y_pu.toarray()
    print("Y_pu (dense) =", Y_dense)