import os
import numpy as np
from opendssdirect import dss

from lagrangian_m import run_lagrangian_polar
from parse_opendss_file import (
    parse_opendss_to_mpc,
    build_global_y_per_unit,
)
from cal_pf import run_newton_powerflow_3p
from utilities.mea_fun import measurement_function
from utilities.report_results import report_results
from config import DATA_DIR, RESULTS_DIR

# --- Configuration ---
# Define the path to the original OpenDSS model file
PRISTINE_DSS = os.fspath(DATA_DIR / "4Bus-YY-Bal.DSS")
TEMP_PARAM_ERR_DSS = os.fspath(RESULTS_DIR / "4Bus-YY-Bal_param_error_temp.dss")

# A target line for parameter error injection (mirrors case_390 strategy)
TARGET_LINES = ["line.2"]  # edit this list to try other lines

# Example erroneous LineCode to simulate parameter mismatch
# Units consistent with the 4-bus test case (mi)
ERR_LINECODE_NAME = "NLM"
ERR_RMATRIX = "[0.5033  |0.1715  0.5133  |0.1688  0.1718  0.5076]"  # slightly perturbed from Kersting
ERR_XMATRIX = "[1.1859  |0.5518  1.1530  |0.4234  0.4660  1.1716]"
ERR_CMATRIX = "[16.57   |-5.35    17.46   |-2.04    -3.40    15.76 ]"


def write_param_error_dss(base_dss_path: str,
                          out_dss_path: str,
                          target_lines: list[str],
                          linecode_name: str,
                          rmatrix: str,
                          xmatrix: str,
                          cmatrix: str) -> str:
    """Create a small DSS that redirects the base model and applies a bad LineCode.

    Strategy mirrors src/case_390.py:
    - Redirect the pristine model
    - Define a new LineCode with erroneous parameters
    - Edit selected lines to use that LineCode
    - Solve
    Returns the output DSS path.
    """
    lines = []
    lines.append("Clear")
    lines.append(f'Redirect "{base_dss_path}"')
    lines.append(f"New LineCode.{linecode_name} nphases=3 units=mi")
    lines.append(f"~ Rmatrix = {rmatrix}")
    lines.append(f"~ Xmatrix = {xmatrix}")
    lines.append(f"~ Cmatrix = {cmatrix}")
    for ln in target_lines:
        # Support both 'line.2' and '2' style names
        ln_name = ln if ln.lower().startswith("line.") else f"line.{ln}"
        lines.append(f"Edit {ln_name} Linecode={linecode_name}")
    lines.append("Solve")

    os.makedirs(os.path.dirname(out_dss_path), exist_ok=True)
    with open(out_dss_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return out_dss_path


def main():
    """
    Demo that mirrors the parameter-error simulation strategy in src/case_390.py:
    1) Load and solve the pristine 4-bus model
    2) Generate measurements with noise from the pristine model
    3) Create a temporary DSS that applies an erroneous LineCode to selected lines
    4) Build Y from the modified model and run DSSE to highlight parameter errors
    """
    print("--- Starting 4-bus Parameter-Error Simulation ---")

    # 1) Parse & solve the pristine circuit (used for measurement generation)
    print(f"\n[Step 1] Parsing pristine model: {PRISTINE_DSS}")
    mpc = parse_opendss_to_mpc(
        PRISTINE_DSS,
        baseMVA=1.0,
        lc_filename=os.fspath(DATA_DIR / "LineConstantsCode4.txt"),
        slack_bus="sourcebus"
    )
    print("Parsing complete. Running Newton-Raphson PF...")
    Vr, Vi, bp = run_newton_powerflow_3p(mpc, 1e-6, 20)
    Vc = Vr + 1j * Vi
    print("PF successful.")

    print("\n[Step 2] Reporting PF results (pristine voltages):")
    report_results(Vr, Vi, bp, mpc)

    # 2) Measurement generation from pristine model
    print("\n[Step 3] Generating measurements from pristine Y-bus...")
    Y_pu_pristine, _ = build_global_y_per_unit(mpc)
    x_state = np.hstack([np.abs(Vc), np.angle(Vc)])
    z = measurement_function(x_state, Y_pu_pristine, mpc, bp)

    # Add measurement noise
    STD_P, STD_Q, STD_V = 1e-3, 1e-3, 1e-5
    bp_all = {(int(row[0]), ph): 3 * i + ph for i, row in enumerate(mpc["bus3p"]) for ph in range(3)}
    R = np.diag([STD_P ** 2] * len(bp_all) + [STD_Q ** 2] * len(bp_all) + [STD_V ** 2] * len(bp_all))
    z_noisy = z + np.random.normal(0.0, np.sqrt(np.diag(R)), size=len(z))

    # 3) Create and load the temporary parameter-error DSS
    print("\n[Step 4] Creating temporary parameter-error DSS and reloading circuit...")
    tmp_dss = write_param_error_dss(
        base_dss_path=PRISTINE_DSS,
        out_dss_path=TEMP_PARAM_ERR_DSS,
        target_lines=TARGET_LINES,
        linecode_name=ERR_LINECODE_NAME,
        rmatrix=ERR_RMATRIX,
        xmatrix=ERR_XMATRIX,
        cmatrix=ERR_CMATRIX,
    )
    print(f"Temporary DSS written: {tmp_dss}")

    dss.Command("Clear")
    dss.Command(f'Redirect "{tmp_dss}"')

    # Rebuild Y on the modified system (structure same as mpc)
    Y_pu_mod, _ = build_global_y_per_unit(mpc)

    # 4) Run DSSE with model mismatch to surface parameter errors
    print("\n[Step 5] Running DSSE with modified parameters...")
    x_est, converged, lambdaN, sorted_groups = run_lagrangian_polar(
        z_noisy,
        Vc,            # ok if ignored; solver will flat-start when size mismatches
        bp,
        Y_pu_mod,
        R,
        mpc,
    )
    print(z_noisy)
    print(np.diag(R))
    
    print("\n[Step 6] DSSE status and key outputs:")
    print(f"Converged: {converged}")
    if sorted_groups:
        best_idx, best_val = sorted_groups[0]
        print(f"Top suspect line group: index={best_idx}, chi2-k={best_val:.4f}")
    # Also report the single largest |λ_N|
    idx_max = int(np.argmax(np.abs(lambdaN))) if len(lambdaN) else -1
    if idx_max >= 0:
        print(f"Max |λ_N| at global index {idx_max}: {float(np.abs(lambdaN[idx_max])):.4f}")


# --- Main execution block ---
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main()
