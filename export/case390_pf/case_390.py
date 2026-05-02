import os
import numpy as np
from utilities.report_results import report_results
from cal_pf import run_newton_powerflow_3p
from parse_opendss_file import parse_opendss_to_mpc, merge_closed_switches_in_mpc_and_dss
from config import DATA_DIR

def main():
    # 1) Load the data from OpenDSS, func parse_opendss_to_mpc will solve the circuit
    dss_filename = os.fspath(DATA_DIR / "Master.dss")
    mpc = parse_opendss_to_mpc(
        dss_filename, baseMVA=1.0,
        lc_filename=os.fspath(DATA_DIR / "LineConstantsCode.txt"),
        slack_bus="p1",
    )
    # 2) merges and Run Newton PF
    merge_closed_switches_in_mpc_and_dss(mpc, switch_threshold=2) ### Be careful with the switch threshold
    Vr, Vi, busphase_map = run_newton_powerflow_3p(mpc, tol=1e-6, max_iter=20)
    # 3) Print final PF results
    x_f = Vr + 1j * Vi
    x = np.concatenate([np.abs(x_f), np.angle(x_f)])
    report_results(Vr, Vi, busphase_map, mpc)
if __name__ == "__main__":
    main()
