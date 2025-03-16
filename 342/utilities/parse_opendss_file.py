import opendssdirect as dss
import numpy as np


def parse_opendss_to_mpc(dss_filename, baseMVA=1.0):
    # ----------------------------------------------------------------
    # 1) Run DSS file
    # ----------------------------------------------------------------
    dss.run_command("Clear")
    dss.run_command(f'Redirect "{dss_filename}"')
    # Force a Solve so that circuit elements are energized, etc.
    dss.Solution.Solve()

    # ----------------------------------------------------------------
    # 2) Initialize the mpc dict
    # ----------------------------------------------------------------
    mpc = {}
    mpc["version"] = "2"
    mpc["baseMVA"] = baseMVA
    mpc["freq"] = 60.0

    # You can store line-construction data if you like, or keep it empty initially
    mpc["lc"] = np.array([])

    # ----------------------------------------------------------------
    # 3) Extract Bus Data
    #    * In your 4-bus example, bus3p has columns:
    #      [bus_id, type, base_kV_LL, VmA, VmB, VmC, VaA, VaB, VaC]
    #    * We'll gather approximate equivalents from the DSS circuit.
    #    * For large systems, you may also want to keep track of #phases, etc.
    # ----------------------------------------------------------------
    circuit = dss.Circuit
    bus_names = circuit.AllBusNames()

    bus3p_list = []
    # Let us keep a mapping from bus_name -> integer bus_id for indexing
    busname_to_id = {}

    for i, busname in enumerate(bus_names, start=1):
        busname_to_id[busname.lower()] = i
        dss.Circuit.SetActiveBus(busname)

        # For distribution-level circuits in DSS, kVBase is typically line-to-neutral.
        # But you might want the line-to-line base for "12.47kV" systems, etc.
        # Check how you want to store it:
        base_kv_ll = dss.Bus.kVBase()  # This might be L-N or L-L depending on the model
        if base_kv_ll is None:
            base_kv_ll = 0.0

        # You can set all bus types = 1 (PQ) for now, or detect the Slack if needed
        # For instance, you might check if there's a PV or Slack element:
        bus_type = 1  # default PQ
        # If you have a known Slack bus in your .dss, you can do:
        # if "some_slack_bus_name" in busname:
        #     bus_type = 3

        # You can also read the DSS's bus voltage (post-solve) for initial guess
        voltages = dss.Bus.VMagAngle()  # [V1, angle1, V2, angle2, ...]
        # Typically you'd need 3 phases => so parse them carefully
        # If the bus is only single-phase or two-phase, you can decide how to pad it.

        # Let’s define a quick helper that picks out phase A,B,C from VMagAngle if present:
        # (This is a naive approach – you might need more robust logic for unbalanced circuits.)
        # Note that DSS doesn't always guarantee ordering A,B,C. You might have to look at
        # Bus.Nodes() to see which phases are actually connected.

        # For the sake of example, let's just do something simplistic:
        # We get the DSS magnitude/angle pairs, then we fill up (VA, VB, VC) or zeros:
        nodes = dss.Bus.Nodes()  # e.g. [1,2,3], or [1], etc.
        # Make default zero for all 3 phases
        vm = [0.0, 0.0, 0.0]
        va = [0.0, 0.0, 0.0]

        mag_angle = dss.Bus.VMagAngle()  # e.g. [mag1, ang1, mag2, ang2, ...]
        # pair them up
        pair_list = list(zip(mag_angle[0::2], mag_angle[1::2]))  # [(mag1,ang1), (mag2,ang2), ...]

        for idx_phase, node_number in enumerate(nodes):
            if idx_phase < len(pair_list):
                vm[idx_phase] = pair_list[idx_phase][0] / 1000.0  # Convert from volts -> kV maybe
                va[idx_phase] = pair_list[idx_phase][1]  # in degrees
            # else remain zero if not present

        # Now we assemble one row for bus3p:
        # bus3p row: [bus_id, type, base_kV_LL, VmA, VmB, VmC, VaA, VaB, VaC]
        row = [
            i,  # bus_id
            bus_type,  # type
            base_kv_ll,  # base_kV_LL
            vm[0], vm[1], vm[2],  # Vmag A,B,C
            va[0], va[1], va[2],  # Vang A,B,C (degrees)
        ]
        bus3p_list.append(row)

    mpc["bus3p"] = np.array(bus3p_list, dtype=float)

    # ----------------------------------------------------------------
    # 4) Extract Line Data
    #    * In your 4-bus example, line3p columns: [line_id, fbus, tbus, status, lcid, length]
    #    * Let’s read all Lines from DSS and build that
    # ----------------------------------------------------------------
    line3p_list = []
    # line_id -> an integer from 1..Nlines
    # we might do: dss.Lines.First(); while dss.Lines.Next() != 0: ...
    # or simply get a list from dss.Circuit.AllElementNames() and filter for "Line."

    all_elements = circuit.AllElementNames()
    line_counter = 0

    for elem_name in all_elements:
        if elem_name.lower().startswith("line."):
            line_counter += 1
            dss.Circuit.SetActiveElement(elem_name)

            # from/to buses
            fb = dss.CktElement.BusNames()[0].split(".")[0].lower()
            tb = dss.CktElement.BusNames()[1].split(".")[0].lower()
            fbus_id = busname_to_id.get(fb, 0)
            tbus_id = busname_to_id.get(tb, 0)

            # status: 1 if enabled, 0 if disabled
            status = 1 if dss.CktElement.Enabled() else 0

            # length (miles) from the DSS property
            length_mi = dss.Lines.Length()

            # line code ID (you might parse dss.Lines.LineCode() or
            # the geometry if needed)
            lcid_str = dss.Lines.LineCode()  # e.g. "switch", "trans", ...
            # You might keep an internal map of linecode strings -> integer IDs
            # For now, let's just store them as an integer or zero:
            # Or store them in some separate dictionary if you want
            # For demonstration:
            lcid = 0
            # (If you want to handle a library of line codes, fill that in here)

            line3p_list.append([
                line_counter,  # line_id
                fbus_id,  # from bus
                tbus_id,  # to bus
                status,
                lcid,  # line code ID
                length_mi
            ])

    mpc["line3p"] = np.array(line3p_list, dtype=float)

    # ----------------------------------------------------------------
    # 5) Extract Transformer Data
    #    * In your 4-bus example: xfmr3p cols:
    #      [xfid, fbus, tbus, status, R, X, basekVA, basekV_LL(HV), basekV_LL(LV)]
    #    * Let’s do something approximate for each Transformer in DSS
    # ----------------------------------------------------------------
    xfmr3p_list = []
    xfmr_counter = 0

    for elem_name in all_elements:
        if elem_name.lower().startswith("transformer."):
            xfmr_counter += 1
            dss.Circuit.SetActiveElement(elem_name)

            xf = dss.Transformers
            # We can parse buses
            buses = dss.CktElement.BusNames()
            fb = buses[0].split(".")[0].lower()
            tb = buses[1].split(".")[0].lower()
            fbus_id = busname_to_id.get(fb, 0)
            tbus_id = busname_to_id.get(tb, 0)

            status = 1 if dss.CktElement.Enabled() else 0

            # If you want R and X, or %loadloss, etc.:
            # The DSS API can read e.g. xf.Wdg() to pick which winding
            # Then do xf.Xhl(), xf.RdcOhms(), or xf.%LoadLoss() / baseMVA etc.

            # For demonstration, we do something simple:
            # basekVA = xf.kVA()   # might be for winding1 or sum
            # basekV_HV = xf.kV()  # for winding1
            # basekV_LV = xf.kV()  # for winding2, etc.
            xf.First()  # caution if bank has multiple windings
            # or parse each winding, etc.

            # Let’s assume 2-winding for demonstration:
            xf.Wdg(1)
            w1_kv = xf.kV()  # HV side
            w1_kva = xf.kVA()

            xf.Wdg(2)
            w2_kv = xf.kV()  # LV side
            w2_kva = xf.kVA()

            r = xf.Rneut()  # Not always reliable for 2-wdg
            x = xf.Xhl()  # or xf.Xscarray()

            if w1_kva < w2_kva:
                basekVA = w2_kva
            else:
                basekVA = w1_kva

            xfmr3p_list.append([
                xfmr_counter,
                fbus_id,
                tbus_id,
                status,
                r,  # R (pu or ohms, you must be consistent!)
                x,  # X (pu or ohms)
                basekVA,
                w1_kv,  # HV side base kV (approx)
                w2_kv,  # LV side base kV
            ])

    mpc["xfmr3p"] = np.array(xfmr3p_list, dtype=float)

    # ----------------------------------------------------------------
    # 6) Extract Load Data
    #    * In your 4-bus example: load3p columns:
    #      [ldid, ldbus, status, PdA, PdB, PdC, pfA, pfB, pfC]
    #    * In a big DSS circuit, many loads may be single-phase,
    #      or two-phase, etc. You will need a policy for how to store them.
    # ----------------------------------------------------------------
    load3p_list = []
    load_counter = 0

    for elem_name in all_elements:
        if elem_name.lower().startswith("load."):
            load_counter += 1
            dss.Circuit.SetActiveElement(elem_name)
            load_bus = dss.CktElement.BusNames()[0].split(".")[0].lower()
            bus_id = busname_to_id.get(load_bus, 0)
            status = 1 if dss.CktElement.Enabled() else 0

            # In DSS, the load has properties via dss.Loads
            # We can do something like:
            ld = dss.Loads
            # We need to move the "Load." pointer to the right name:
            # (since in the loop, we might not automatically move dss.Loads to that name)
            # A quick trick is:
            dss.Loads.Name(elem_name.split(".")[1])  # set active by name

            # Then read real power, etc. The total kW might be splitted among phases.
            # For demonstration, let's just store them as balanced 3-phase if they are 3-phase:
            phases = dss.CktElement.NumPhases()

            pdA = 0.0
            pdB = 0.0
            pdC = 0.0
            pfA = 0.0
            pfB = 0.0
            pfC = 0.0

            # e.g. total kW:
            p_tot = ld.kW()
            pf = ld.pf()  # single PF
            # If 3-phase, you might do p_tot/3 each phase, etc.
            # If 1-phase, then just put it in A-phase, etc.
            # Very rough approach:
            if phases == 3:
                pdA = p_tot / 3.0
                pdB = p_tot / 3.0
                pdC = p_tot / 3.0
                pfA = pf
                pfB = pf
                pfC = pf
            elif phases == 1:
                # We can see which phase from the bus connection, e.g. .1 or .2 or .3
                # For demonstration:
                pdA = p_tot
                pfA = pf
            else:
                # 2-phase. We’ll do a naive approach
                pdA = p_tot / 2.0
                pdB = p_tot / 2.0
                pfA = pf
                pfB = pf

            load3p_list.append([
                load_counter,
                bus_id,
                status,
                pdA, pdB, pdC,
                pfA, pfB, pfC
            ])

    mpc["load3p"] = np.array(load3p_list, dtype=float)

    # ----------------------------------------------------------------
    # 7) Extract Generators
    #    * In your 4-bus example: gen3p columns:
    #      [genid, gbus, status, VgA, VgB, VgC, PgA, PgB, PgC, QgA, QgB, QgC]
    #    * If the 390-bus circuit only has substation slack or a few gens,
    #      parse them similarly. For now we can do placeholders.
    # ----------------------------------------------------------------
    gen3p_list = []
    gen_counter = 0

    for elem_name in all_elements:
        if elem_name.lower().startswith("generator."):
            gen_counter += 1
            dss.Circuit.SetActiveElement(elem_name)
            gen_bus = dss.CktElement.BusNames()[0].split(".")[0].lower()
            bus_id = busname_to_id.get(gen_bus, 0)
            status = 1 if dss.CktElement.Enabled() else 0

            # For demonstration, put 0 or parse from dss.Generators
            dss.Generators.Name(elem_name.split(".")[1])
            pg = dss.Generators.kW()
            qg = dss.Generators.kvar()
            # Similarly parse voltage setpoints if relevant
            vg = 1.0

            # Suppose it’s 3-phase balanced:
            pgA = pg / 3.0
            pgB = pg / 3.0
            pgC = pg / 3.0
            qgA = qg / 3.0
            qgB = qg / 3.0
            qgC = qg / 3.0

            # For demonstration, store them:
            gen3p_list.append([
                gen_counter,
                bus_id,
                status,
                vg, vg, vg,  # VgA, VgB, VgC
                pgA, pgB, pgC,
                qgA, qgB, qgC
            ])

    # If you have a Slack source (Vsource, etc.), you might treat it as a special generator
    # or bus type=3, etc.

    mpc["gen3p"] = np.array(gen3p_list, dtype=float)

    # ----------------------------------------------------------------
    # 8) (Optional) If you want to store line construction data
    #    (mpc["lc"]) you can parse the different geometries, linecodes, etc.
    # ----------------------------------------------------------------

    return mpc

parse_opendss_to_mpc("Master.dss")
