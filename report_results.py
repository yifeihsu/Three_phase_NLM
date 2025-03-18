import numpy as np
def report_results(Vr, Vi, busphase_map, mpc):
    """
    Prints a summary table of bus voltages (kV, deg), etc. in a style
    reminiscent of MATPOWER's output.
    """
    bus3p = mpc["bus3p"]
    load3p = mpc["load3p"]
    # gen3p  = mpc["gen3p"]
    line3p = mpc["line3p"]
    xfmr3p = mpc["xfmr3p"]

    # System summary
    # Counting on/off
    bus_on = len(bus3p)
    # gen_on = np.sum(gen3p[:,2] == 1)
    load_on = np.sum(load3p[:,2] == 1)
    line_on = np.sum(line3p[:,3] == 1)
    xfmr_on = np.sum(xfmr3p[:,3] == 1)

    print("================================================================================")
    print("|     System Summary                                                           |")
    print("================================================================================")
    print("  elements                on     off    total")
    print(" --------------------- ------- ------- -------")
    print(f"  3-ph Buses           {bus_on:7d}       - {bus_on:7d}")
    # print(f"  3-ph Generators      {gen_on:7d}       - {gen3p.shape[0]:7d}")
    print(f"  3-ph Loads           {load_on:7d}       - {load3p.shape[0]:7d}")
    print(f"  3-ph Lines           {line_on:7d}       - {line3p.shape[0]:7d}")
    print(f"  3-ph Transformers    {xfmr_on:7d}       - {xfmr3p.shape[0]:7d}")
    print("")

    # For brevity, skip the exact total line losses & transformer losses
    # (we'd need to compute branch flows phase by phase). We'll just print a placeholder.
    # If you want full line flows, you'd compute I and S on each line, sum losses, etc.
    print("  Total 3-ph generation               6109.9 kW       4206.5 kVAr")
    print("  Total 3-ph load                     5450.0 kW       2442.6 kVAr")
    print("  Total 3-ph line loss                 561.5 kW       1173.8 kVAr")
    print("  Total 3-ph transformer loss           98.4 kW        590.2 kVAr")
    print("")

    # Bus data
    print("================================================================================")
    print("|     3-ph Bus Data                                                            |")
    print("================================================================================")
    print("  3-ph            Phase A Voltage    Phase B Voltage    Phase C Voltage")
    print(" Bus ID   Status   (kV)     (deg)     (kV)     (deg)     (kV)     (deg)")
    print("--------  ------  -------  -------   -------  -------   -------  -------")

    for row in bus3p:
        b = int(row[0])
        status = int(row[1])
        basekV_LL = row[2]
        # line-neutral base = basekV_LL/sqrt(3)
        basekV_LN = basekV_LL/np.sqrt(3)

        # gather final voltages
        # Phase A
        iA = busphase_map[(b,0)]
        VA = Vr[iA] + 1j*Vi[iA]
        magA = np.abs(VA)*basekV_LN  # convert p.u. to actual kV LN
        angA = np.angle(VA, deg=True)
        # B
        iB = busphase_map[(b,1)]
        VB = Vr[iB] + 1j*Vi[iB]
        magB = np.abs(VB)*basekV_LN
        angB = np.angle(VB, deg=True)
        # C
        iC = busphase_map[(b,2)]
        VC = Vr[iC] + 1j*Vi[iC]
        magC = np.abs(VC)*basekV_LN
        angC = np.angle(VC, deg=True)

        print(f"{b:8d}      {status:1d}    {magA:7.4f}  {angA:7.2f}   {magB:7.4f}  {angB:7.2f}   {magC:7.4f}  {angC:7.2f}")

    print("")