import os
import numpy as np
from opendssdirect import dss
from lagrangian_m import run_lagrangian_polar
from utilities.mea_fun import measurement_function
from cal_pf import run_newton_powerflow_3p
from parse_opendss_file import parse_opendss_to_mpc
from parse_opendss_file import build_global_y_per_unit, merge_closed_switches_in_mpc_and_dss
from copy import deepcopy  # NEW
from config import DATA_DIR


def reorder_measurements_to_map(z, src_map, dst_map, fill_v=1.0):
    """
    Reorder measurement vector z=[P;Q;|V|] from src_map ordering to dst_map ordering.
    fill_v is used for |V| if a (bus,phase) key is missing in src_map.
    """
    N_dst = len(dst_map)
    # split source
    N_src = len(src_map)
    P_src = z[:N_src]
    Q_src = z[N_src:2*N_src]
    V_src = z[2*N_src:2*N_src + N_src]

    P_dst = np.zeros(N_dst, dtype=float)
    Q_dst = np.zeros(N_dst, dtype=float)
    V_dst = np.full(N_dst, float(fill_v))

    missing = []
    # map by keys
    for key, j_dst in dst_map.items():
        if key in src_map:
            i_src = src_map[key]
            P_dst[j_dst] = P_src[i_src]
            Q_dst[j_dst] = Q_src[i_src]
            V_dst[j_dst] = V_src[i_src]
        else:
            missing.append(key)
    if missing:
        print(f"[reorder_measurements_to_map] WARNING: {len(missing)} nodes exist in dst_map but not in src_map; "
              f"P/Q set to 0.0, |V| set to {fill_v}. Examples: {missing[:5]}{'...' if len(missing)>5 else ''}")

    return np.concatenate([P_dst, Q_dst, V_dst])

def build_mpc_with_measured_injections(mpc_base, busphase_map, P_meas_by_key, Q_meas_by_key):
    """
    Return a deep-copied MPC whose per-(bus,phase) net injections equal measured values.
    Positive => generator; negative => load. Always keep a generator row for slack buses.
    """
    mpc_pf = deepcopy(mpc_base)

    # bus type map
    bus_type = {int(row[0]): int(row[1]) for row in mpc_pf["bus3p"]}

    load3p_pf, gen3p_pf = [], []
    load_id, gen_id = 1, 1

    bus_ids = [int(row[0]) for row in mpc_pf["bus3p"]]
    for bid in bus_ids:
        # gather per-phase measured injections (defaults 0 if missing)
        def Pv(ph): return P_meas_by_key.get((bid, ph), 0.0)
        def Qv(ph): return Q_meas_by_key.get((bid, ph), 0.0)

        P_A, P_B, P_C = Pv(0), Pv(1), Pv(2)
        Q_A, Q_B, Q_C = Qv(0), Qv(1), Qv(2)

        # split into loads (negative part) and gens (positive part)
        PdA, PdB, PdC = max(-P_A, 0.0), max(-P_B, 0.0), max(-P_C, 0.0)
        QdA, QdB, QdC = max(-Q_A, 0.0), max(-Q_B, 0.0), max(-Q_C, 0.0)
        PgA, PgB, PgC = max(P_A, 0.0),  max(P_B, 0.0),  max(P_C, 0.0)
        QgA, QgB, QgC = max(Q_A, 0.0),  max(Q_B, 0.0),  max(Q_C, 0.0)

        if (PdA + PdB + PdC + QdA + QdB + QdC) > 0:
            load3p_pf.append([load_id, bid, 1, PdA, PdB, PdC, QdA, QdB, QdC])
            load_id += 1

        is_slack = (bus_type.get(bid, 1) == 3)
        if (PgA + PgB + PgC + QgA + QgB + QgC) > 0 or is_slack:
            # keep Vg placeholders at 1.0 pu
            gen3p_pf.append([gen_id, bid, 1, 1.0, 1.0, 1.0, PgA, PgB, PgC, QgA, QgB, QgC])
            gen_id += 1

    mpc_pf["load3p"] = load3p_pf
    mpc_pf["gen3p"]  = gen3p_pf
    return mpc_pf

def remap_by_busphase_map(vec_src, map_src, map_dst):
    """
    Remap a per-node vector from source map to destination map using (bus,phase) keys.
    """
    out = np.zeros(len(map_dst), dtype=float)
    missing = []
    for key, j_dst in map_dst.items():
        if key in map_src:
            out[j_dst] = vec_src[map_src[key]]
        else:
            missing.append(key)
    if missing:
        print(f"[remap_by_busphase_map] WARNING: {len(missing)} keys missing in src map. Examples: {missing[:5]}{'...' if len(missing)>5 else ''}")
    return out

def main():
    """
    This script simulates a cable burn-out scenario to test the parameter
    error detection capabilities of a Lagrangian-based state estimator.

    The process is as follows:
    1. Simulate Reality (Cable Burn-out): Modify the OpenDSS model to
       represent a "burn-out" of several parallel cables at Line 208 by
       disabling them. Run a power flow on this modified grid to get the
       "true" system state (voltages, currents, etc.).
    2. Generate Measurements: Based on this "true" state from the burnt-out
       model, generate a set of measurements (power injections, flows,
       voltage magnitudes) and add realistic noise.
    3. Assume Correct Model: Load the original, unmodified OpenDSS model,
       which represents the "assumed" or "correct" state of the grid from
       an operator's perspective (i.e., they don't know about the burn-out).
    4. Run State Estimation: Use the measurements from the burnt-out reality
       (Step 2) as input to the state estimator, which operates on the
       assumed correct model (Step 3).
    5. Detect Anomaly: Analyze the Lagrange multipliers from the state
       estimation. A large multiplier value is expected to correspond to the
       Kirchhoff's law constraints of the line that was burnt out, thus
       identifying the location of the modeling error.
    """
    np.random.seed(42)
    dss_filename = os.fspath(DATA_DIR / "Secondary_Equivalent.dss")
    dss_filename_ft = os.fspath(DATA_DIR / "Secondary_Equivalent_ft.dss")
    baseMVA = 1.0
    slack_bus = "s148"

    # --- 1) SIMULATE REALITY: CABLE BURN-OUT AND POWER FLOW ---
    print("--- Step 1: Simulating the true system with burnt-out cables ---")

    # Use opendssdirect to compile the master file and then disable the lines
    dss.Command("clear")
    dss.Command(f'Redirect "{dss_filename_ft}"')
    dss.Solution.Solve()

    print("Parsing the 'true' (burnt-out) model from OpenDSS...")

    mpc_true = parse_opendss_to_mpc(
        dss_filename_ft, baseMVA=baseMVA,
        lc_filename=os.fspath(DATA_DIR / "LineConstantsCode.txt"),
        slack_bus=slack_bus,
    )
    merge_closed_switches_in_mpc_and_dss(mpc_true, switch_threshold=2)

    Vr_true, Vi_true, busphase_map_true = run_newton_powerflow_3p(mpc_true, tol=1e-6, max_iter=20)
    x_f_true = Vr_true + 1j * Vi_true
    x_true = np.concatenate([np.abs(x_f_true), np.angle(x_f_true)])
    print("...Power flow for the burnt-out scenario is complete.")

    # --- 2) GENERATE NOISY MEASUREMENTS FROM THE "TRUE" STATE -----------------
    print("\n--- Step 2: Generating noisy measurements from the true state ---")

    Y_pu_true, _ = build_global_y_per_unit(mpc_true)
    z = measurement_function(x_true, Y_pu_true, mpc_true, busphase_map_true)

    num_bus_phases = len(busphase_map_true)
    num_P_inj = num_Q_inj = num_Vmag = num_bus_phases

    # standard deviations
    std_P, std_Q, std_V = 1e-4, 1e-4, 5e-4

    diag_R = (
            [std_P ** 2] * num_P_inj +
            [std_Q ** 2] * num_Q_inj +
            [std_V ** 2] * num_Vmag
    )
    R = np.diag(diag_R)

    z_noisy_true = z + np.random.normal(0.0, np.sqrt(diag_R))
    print("...Noisy measurements generated.")

    # --- 3) SETUP THE "ASSUMED CORRECT" (UNMODIFIED) MODEL ---
    print("\n--- Step 3: Setting up the assumed-correct (unmodified) model for estimation ---")

    # Re-compile the original master file to get the model WITHOUT the burn-out
    # This represents the model the operator believes to be correct.
    dss.Command("Clear")
    mpc_assumed = parse_opendss_to_mpc(
        dss_filename, baseMVA=baseMVA,
        lc_filename=os.fspath(DATA_DIR / "LineConstantsCode.txt"),
        slack_bus=slack_bus,
    )
    merge_closed_switches_in_mpc_and_dss(mpc_assumed, switch_threshold=2)

    # Build the admittance matrix for this assumed-correct model
    Y_pu_assumed, _ = build_global_y_per_unit(mpc_assumed)

    # Calculate a power flow solution for the assumed model. This will be used
    # as the initial guess (flat start) for the state estimator.
    Vr_assumed, Vi_assumed, busphase_map_assumed = run_newton_powerflow_3p(mpc_assumed, tol=1e-6, max_iter=20)
    x_f_assumed = Vr_assumed + 1j * Vi_assumed

    # --- 3a) Reorder measurements to the assumed ordering (critical!) ---  # NEW
    z_noisy = reorder_measurements_to_map(
        z_noisy_true,
        src_map=busphase_map_true,
        dst_map=busphase_map_assumed,
        fill_v=1.0
    )

    # Rebuild R for the assumed ordering (diagonal with same stds)          # NEW
    N_assumed = len(busphase_map_assumed)
    diag_R_assumed = (
        [std_P ** 2] * N_assumed +
        [std_Q ** 2] * N_assumed +
        [std_V ** 2] * N_assumed
    )
    R = np.diag(diag_R_assumed)

    # --- 3b) Build PF case on assumed network using measured P/Q injections ---  # NEW
    # Split the (reordered) measurement vector
    P_meas = z_noisy[:N_assumed]
    Q_meas = z_noisy[N_assumed:2 * N_assumed]
    # Map to dict keyed by (bus, phase) for the assumed network
    P_meas_by_key = {key: P_meas[idx] for key, idx in busphase_map_assumed.items()}
    Q_meas_by_key = {key: Q_meas[idx] for key, idx in busphase_map_assumed.items()}

    mpc_pf_from_meas = build_mpc_with_measured_injections(
        mpc_assumed, busphase_map_assumed, P_meas_by_key, Q_meas_by_key
    )
    Vr_pf_meas, Vi_pf_meas, busphase_map_pf = run_newton_powerflow_3p(mpc_pf_from_meas, tol=1e-6, max_iter=20)

    # If the PF routine returned a different internal ordering, remap to the estimator's order
    if busphase_map_pf != busphase_map_assumed:
        print("...Re-mapping PF solution from measured P/Q to estimator ordering.")
        Vr_pf_meas = remap_by_busphase_map(Vr_pf_meas, busphase_map_pf, busphase_map_assumed)
        Vi_pf_meas = remap_by_busphase_map(Vi_pf_meas, busphase_map_pf, busphase_map_assumed)

    # Initial guess for estimator (polar)
    x0_est = np.concatenate([np.abs(Vr_pf_meas + 1j * Vi_pf_meas),
                             np.angle(Vr_pf_meas + 1j * Vi_pf_meas)])
    print("...Initial guess for DSSE is the PF solution using measured P/Q on the assumed network.")

    print("...Assumed-correct model is ready.")

    # --- 4) RUN LAGRANGIAN ESTIMATOR TO FIND THE DISCREPANCY ---
    print("\n--- Step 4: Running Lagrangian Polar State Estimator ---")

    # We provide the estimator with:
    # - z_noisy: Measurements from the "real" (burnt-out) world.
    # - x_f_assumed: An initial state guess from the "assumed" world.
    # - Y_pu_assumed, mpc_assumed: The model of the "assumed" world.
    x_est, success, lambdaN, _ = run_lagrangian_polar(
        z_noisy, x0_est, busphase_map_assumed,
        Y_pu_assumed, R,
        mpc_assumed
    )

    if not success:
        print("!!! State estimation did not converge, which may be expected due to the model error.")
    else:
        print("...State estimation converged.")

    print("\n--- Step 5: Analyzing Lagrangian multipliers to locate the error ---")

    if lambdaN is None or len(lambdaN) == 0:
        print("Estimation did not produce Lagrange multipliers. Cannot locate error.")
        return

    largest_idx = np.argmax(np.abs(lambdaN))
    largest_val = lambdaN[largest_idx]

    # The structure of lambdaN is assumed to be 12 parameters per line (6 for R, 6 for X)
    line_index_in_mpc = largest_idx // 12
    param_index_within_line = largest_idx % 12

    # Determine if it's an R or X matrix error
    error_type = "X (reactance)" if param_index_within_line >= 6 else "R (resistance)"

    # The index within the 3x3 symmetric matrix (represented as a 6-element vector)
    param_index_in_matrix = param_index_within_line % 6

    print(f"Largest Lagrangian param => index={(largest_idx%12 + 1)%6}, value={largest_val:.4g}, "
          f"line={largest_idx//12 + 1}, error_type={error_type}")
    # Get the line name from the mpc_assumed structure for better reporting
    # try:
    #     line_info = mpc_assumed['line3p'][line_index_in_mpc]
    #     line_name = line_info['name']
    #     from_bus = line_info['from_bus_name']
    #     to_bus = line_info['to_bus_name']
    # except IndexError:
    #     print(f"Error: Cannot find line with MPC index {line_index_in_mpc}.")
    #     return
if __name__ == "__main__":
    main()
