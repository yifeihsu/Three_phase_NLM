import os
from opendssdirect import dss

from cal_pf import run_newton_powerflow_3p
from parse_opendss_file import parse_opendss_to_mpc
from utilities.report_results import report_results
from config import DATA_DIR, RESULTS_DIR


def main():
    # Paths
    dss_path = os.fspath(DATA_DIR / "4Bus-YY-Bal.DSS")
    lc_path = os.fspath(DATA_DIR / "LineConstantsCode4.txt")

    # Ensure results dir exists (for any downstream use)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load and solve the original 4-bus model
    print(f"Parsing OpenDSS model: {dss_path}")
    mpc = parse_opendss_to_mpc(
        dss_path,
        baseMVA=1.0,
        lc_filename=lc_path,
        slack_bus="sourcebus",
    )

    print("Running 3-phase Newton power flow...")
    Vr, Vi, busphase_map = run_newton_powerflow_3p(mpc, tol=1e-6, max_iter=20)

    print("\nPower flow results (original 4-bus case):")
    report_results(Vr, Vi, busphase_map, mpc)


if __name__ == "__main__":
    main()

