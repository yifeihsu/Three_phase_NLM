import numpy as np
import random
import time
from opendssdirect import dss
from lagrangian_m import run_lagrangian_polar
from utilities.mea_fun import measurement_function
from cal_pf import run_newton_powerflow_3p
from parse_opendss_file import parse_opendss_to_mpc
from report_results import report_results
from parse_opendss_file import build_global_y_per_unit, merge_closed_switches_in_mpc_and_dss

def build_dss_matrix_string(r_vals):
    """
    Utility to transform a 1D list [a, b, c, d, e, f]
    into an OpenDSS triangular matrix string:
        [a | b  c | d  e  f]
    This is how OpenDSS expects Rmatrix or Xmatrix for 3-phase lines.
    """
    # For a 3x3 symmetrical matrix, OpenDSS uses row-wise triangular input:
    # row1: [r11]
    # row2: [r21, r22]
    # row3: [r31, r32, r33]
    # separated by "|"
    # r_vals is assumed to be [r11, r21, r22, r31, r32, r33]
    return f"[{r_vals[0]} | {r_vals[1]}  {r_vals[2]} | {r_vals[3]}  {r_vals[4]}  {r_vals[5]}]"

def random_scale_parameters(orig_vals, scale_count=2):
    """
    Randomly pick 'scale_count' indices in `orig_vals` and multiply
    by a factor between 10 and 100. Returns a new list of perturbed values.
    """
    new_vals = orig_vals[:]
    num_elements = len(new_vals)
    indices_to_scale = random.sample(range(num_elements), k=scale_count)  # pick distinct positions

    for idx in indices_to_scale:
        factor = random.uniform(10.0, 100.0)
        new_vals[idx] = new_vals[idx] * factor

    return new_vals

def test_iteration_wrapper(dss, mpc, z_noisy, x_f, busphase_map, Y_pu, covariance_matrix):
    """
    Iterates over line indices from 4..135 and for each line attempts
    up to 10 random 'big' perturbations of R or X to see if the
    run_lagrangian_polar incorrectly flags a different line.
    """

    # Original R and X from your snippet (adjust as needed)
    R_orig = [
        0.13018,
        -4.14314e-4, 0.130177,
        -8.11898e-4, -4.14314e-4, 0.13018
    ]
    X_orig = [
        0.0668195,
        -2.24891e-4, 0.066985,
        -4.2835e-5, -2.24891e-4, 0.0668195
    ]
    C_str = "[566.228 | 0  566.228 | 0  0  566.228]"

    for line_idx in range(4, 136):
        dss.Command('Redirect "Master.DSS"')
        for sub_iter in range(10):
            choice = random.choice(["R", "X"])
            if choice == "R":
                new_R = random_scale_parameters(R_orig, scale_count=2)  # pick how many to scale
                new_X = X_orig[:]
            else:
                new_R = R_orig[:]
                new_X = random_scale_parameters(X_orig, scale_count=2)

            R_str = build_dss_matrix_string(new_R)
            X_str = build_dss_matrix_string(new_X)

            dss.Command('New LineCode.NLM nphases=3 units=mi')
            dss.Command(f'~ Rmatrix = {R_str}')
            dss.Command(f'~ Xmatrix = {X_str}')
            dss.Command(f'~ Cmatrix = {C_str}')
            dss.Command(f'Edit line.{line_idx} Linecode=NLM')
            dss.Solution.Solve()

            Y_pu, _ = build_global_y_per_unit(mpc)

            x_est, success, lambdaN = run_lagrangian_polar(
                z_noisy, x_f, busphase_map, Y_pu, covariance_matrix, mpc
            )

            # Identify the line with largest Lagrangian multiplier
            largest_idx = np.argmax(np.abs(lambdaN))
            largest_val = lambdaN[largest_idx]
            identified_line = largest_idx // 12 + 1  # per your logic

            # Check your “failure” condition: if largest_val >= 4
            # AND the identified line != line_idx we just perturbed => test fails
            if (largest_val >= 4.5) and (identified_line != line_idx):
                # Store the info and break
                return {
                    "line_idx": line_idx,
                    "sub_iteration": sub_iter,
                    "which_matrix": choice,
                    "perturbed_values": (new_R if choice == "R" else new_X),
                    "largest_idx": largest_idx,
                    "largest_val": float(largest_val),
                    "identified_line": identified_line
                }
    return None

def main():
    # 1) Load the data from OpenDSS, parse to MPC
    dss_filename = "Master.DSS"
    mpc = parse_opendss_to_mpc(dss_filename, baseMVA=1.0, slack_bus="p1")

    # 2) merges and Run Newton PF
    merge_closed_switches_in_mpc_and_dss(mpc, switch_threshold=2)
    Vr, Vi, busphase_map = run_newton_powerflow_3p(mpc, tol=1e-6, max_iter=20)

    # 3) Build final PF results
    x_f = Vr + 1j * Vi
    x = np.concatenate([np.abs(x_f), np.angle(x_f)])

    # 4) Generate measurement data (z_noisy)
    Y_pu_s, _ = build_global_y_per_unit(mpc)
    z = measurement_function(x, Y_pu_s, mpc, busphase_map)
    std_P, std_Q, std_V = 0.0001, 0.0001, 0.00001
    num_P_inj = num_Q_inj = num_Vmag = len(busphase_map)
    num_PQ_flow = 4 * 3 * len(mpc["line3p"])
    covariance_matrix = np.diag(
        [std_P**2]*num_P_inj +
        [std_Q**2]*num_Q_inj +
        [std_P**2]*num_PQ_flow +
        [std_V**2]*num_Vmag
    )
    z_noisy = z + np.random.normal(0, np.sqrt(np.diag(covariance_matrix)), size=len(z))

    # 5) Solve once normally to set up references
    Y_pu, _ = build_global_y_per_unit(mpc)
    x_est_initial, success_initial, lambdaN_initial = run_lagrangian_polar(
        z_noisy, x_f, busphase_map, Y_pu, covariance_matrix, mpc
    )

    # 6) Now run the iteration wrapper to try to cause a mis‐identification
    fail_result = test_iteration_wrapper(dss, mpc, z_noisy, x_f, busphase_map, Y_pu, covariance_matrix)

    if fail_result is not None:
        print("Failure triggered with the following parameters:")
        for k, v in fail_result.items():
            print(f"  {k} => {v}")
    else:
        print("No mis‐identification found after all perturbations.")

if __name__ == "__main__":
    main()
