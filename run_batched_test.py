<<<<<<< HEAD:run_batched_test.py
import numpy as np
import re
import time
import os
from opendssdirect import dss
from lagrangian_m import run_lagrangian_polar
from utilities.mea_fun import measurement_function
from cal_pf import run_newton_powerflow_3p
from parse_opendss_file import parse_opendss_to_mpc, build_global_y_per_unit, merge_closed_switches_in_mpc_and_dss
from copy import deepcopy  # NEW


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

def create_faulted_dss_file(original_dss_path, line_to_burn_out, temp_dss_path):
    """
    Reads an original DSS file and writes a temporary version with one line commented out.

    Args:
        original_dss_path (str): Path to the source Master.DSS file.
        line_to_burn_out (str): The name of the line to disable (e.g., '136_2').
        temp_dss_path (str): The path where the temporary faulted DSS file will be written.
    """
    # Compile a regular expression to find the exact line definition.
    # It looks for "new Line.[line_name]" at the start of a line, ignoring case and whitespace.
    line_regex = re.compile(r"^\s*new\s+Line\." + re.escape(line_to_burn_out), re.IGNORECASE)

    with open(original_dss_path, 'r') as f_in, open(temp_dss_path, 'w') as f_out:
        for line in f_in:
            if line_regex.search(line):
                # If the line matches, comment it out with '!'
                f_out.write('! ' + line)
            else:
                # Otherwise, write the line as is.
                f_out.write(line)


def find_parallel_line_groups(dss_file_path):
    """
    Parses a DSS file to find groups of parallel lines, excluding switches.
    (This function is correct and remains unchanged)
    """
    try:
        with open(dss_file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: DSS file not found at {dss_file_path}")
        return {}
    all_line_names = re.findall(r"new\s+Line\.([\w_]+)", content, re.IGNORECASE)
    cable_names = [name for name in all_line_names if not name.lower().endswith('_sw')]
    groups = {}
    for name in cable_names:
        base_name = name.split('_')[0]
        if base_name not in groups:
            groups[base_name] = []
        groups[base_name].append(name)
    parallel_groups = {base: lines for base, lines in groups.items() if len(lines) > 1}
    return parallel_groups


def run_single_burnout_test(dss_filename, line_to_burn_out, true_base_name, mpc_assumed, x_f_assumed,
                            busphase_map_assumed, Y_pu_assumed):
    """
    Runs a single burnout test by generating a temporary faulted DSS file for parsing.
    """
    baseMVA = 1.0
    slack_bus = "s148"
    temp_dss_filename = f"temp_fault_{line_to_burn_out}.dss"

    try:
        # --- PRE-STEP: Generate the faulted DSS file ---
        create_faulted_dss_file(dss_filename, line_to_burn_out, temp_dss_filename)

        # --- 1) SIMULATE REALITY (using the temp file) ---

        # First, check if the power flow converges for this reality
        dss.Command("Clear")
        dss.Command(f'Redirect "{temp_dss_filename}"')
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            print(
                f"    - WARNING: Power flow for reality with '{line_to_burn_out}' burnt-out did not converge. Skipping test.")
            return None

        # Now parse the state from the faulted file
        mpc_true = parse_opendss_to_mpc(temp_dss_filename, baseMVA=baseMVA, slack_bus=slack_bus)

        # The merge function also modifies the live DSS object, which is fine here.
        merge_closed_switches_in_mpc_and_dss(mpc_true, switch_threshold=2)

        Vr_true, Vi_true, busphase_map_true = run_newton_powerflow_3p(mpc_true, tol=1e-6, max_iter=20)
        x_f_true = Vr_true + 1j * Vi_true
        x_true = np.concatenate([np.abs(x_f_true), np.angle(x_f_true)])

        # --- 2) GENERATE NOISY MEASUREMENTS ---
        Y_pu_true, _ = build_global_y_per_unit(mpc_true)
        z = measurement_function(x_true, Y_pu_true, mpc_true, busphase_map_true)
        num_bus_phases = len(busphase_map_true)
        std_P, std_Q, std_V = 1e-4, 1e-4, 1e-4
        diag_R = ([std_P ** 2] * num_bus_phases + [std_Q ** 2] * num_bus_phases + [std_V ** 2] * num_bus_phases)
        R = np.diag(diag_R)
        z_noisy_true = z + np.random.normal(0.0, np.sqrt(diag_R))
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
        # --- 3) RUN ESTIMATOR ---
        x_est, success, _, line_distances = run_lagrangian_polar(
            z_noisy, x0_est, busphase_map_assumed, Y_pu_assumed, R, mpc_assumed
        )

        # --- 4) ANALYZE RESULTS ---
        if not success or line_distances is None or len(line_distances) == 0:
            print(
                f"    - DETECTION FAILED: Estimator did not converge or produce distances for line {line_to_burn_out}.")
            return False

        valid_distances = [d for d in line_distances if np.isfinite(d[1])]
        if not valid_distances:
            print(f"    - DETECTION FAILED: All computed distances were NaN or Inf.")
            return False

        detected_line_index_mpc, max_dist = max(valid_distances, key=lambda item: item[1])
        detected_line_info = mpc_assumed['line3p'][detected_line_index_mpc]
        detected_line_name = detected_line_info[-1]

        is_successful = detected_line_name.lower() == true_base_name.lower()

        if is_successful:
            print(
                f"    - DETECTION SUCCESS: Fault on '{line_to_burn_out}' correctly identified as Line '{detected_line_name}' (Max Distance: {max_dist:.3f}).")
        else:
            print(
                f"    - DETECTION FAILED: Fault on '{line_to_burn_out}' was misidentified as Line '{detected_line_name}' (Max Distance: {max_dist:.3f}).")

        return is_successful

    except Exception as e:
        print(f"    - An unexpected error occurred during test for '{line_to_burn_out}': {e}")
        return False
    finally:
        # --- 5) CLEANUP: Ensure temporary file is always deleted ---
        if os.path.exists(temp_dss_filename):
            os.remove(temp_dss_filename)


def main():
    np.random.seed(42)
    dss_filename = "Secondary_Equivalent.dss"
    baseMVA = 1.0
    slack_bus = "s148"

    print("--- Step 1: Identifying all parallel line groups from DSS file ---")
    parallel_line_groups = find_parallel_line_groups(dss_filename)
    if not parallel_line_groups:
        print("No parallel line groups found to test. Exiting.")
        return
    print(f"Found {len(parallel_line_groups)} groups of parallel lines to be tested.")

    print("\n--- Step 2: Preparing the 'assumed-correct' base model ---")
    mpc_assumed = parse_opendss_to_mpc(dss_filename, baseMVA=baseMVA, slack_bus=slack_bus)
    merge_closed_switches_in_mpc_and_dss(mpc_assumed, switch_threshold=2)
    Vr_assumed, Vi_assumed, busphase_map_assumed = run_newton_powerflow_3p(mpc_assumed, tol=1e-6, max_iter=20)
    x_f_assumed = Vr_assumed + 1j * Vi_assumed
    Y_pu_assumed, _ = build_global_y_per_unit(mpc_assumed)
    print("...'Assumed-correct' model and initial state are ready.\n")

    print("--- Step 3: Running representative burnout simulation for each line group ---")

    successful_detections = 0
    total_tests = 0

    failed_line_ids = []   # <-- collect misdetections
    skipped_line_ids = []  # <-- collect PF-not-converged / skipped tests

    for base_name, cables in parallel_line_groups.items():
        total_tests += 1
        cable_to_burn = cables[1]

        print(f"\nTesting representative burnout for Line Group '{base_name}' (disabling '{cable_to_burn}')...")

        result = run_single_burnout_test(
            dss_filename=dss_filename,
            line_to_burn_out=cable_to_burn,
            true_base_name=base_name,
            mpc_assumed=mpc_assumed,
            x_f_assumed=x_f_assumed,
            busphase_map_assumed=busphase_map_assumed,
            Y_pu_assumed=Y_pu_assumed
        )

        if result is True:
            successful_detections += 1
        elif result is False:
            # Detection ran but was wrong -> record the actual burnt Line ID
            failed_line_ids.append((base_name, cable_to_burn))
        elif result is None:
            # Power flow for the "reality" didn't converge -> record as skipped
            skipped_line_ids.append((base_name, cable_to_burn))
            total_tests -= 1  # keep your success-rate denominator consistent

        time.sleep(0.1)

    print("\n" + "---" * 12)
    print("---   Representative Analysis Report   ---")
    print("---" * 12)

    if total_tests > 0:
        success_rate = (successful_detections / total_tests) * 100
        print(f"Total Groups Tested:    {total_tests}")
        print(f"Successful Detections:  {successful_detections}")
        print(f"Failed Detections:      {total_tests - successful_detections}")
        print(f"Detection Success Rate: {success_rate:.2f}%")
    else:
        print("No valid tests were completed.")

    if failed_line_ids:
        print("\nLine IDs with FAILED detection:")
        for base_name, line_id in failed_line_ids:
            print(f"  - Group '{base_name}': Line '{line_id}'")

    if skipped_line_ids:
        print("\nLine IDs SKIPPED (PF did not converge):")
        for base_name, line_id in skipped_line_ids:
            print(f"  - Group '{base_name}': Line '{line_id}'")

if __name__ == "__main__":
    main()
=======
import os
import re
import math
import shutil
import numpy as np
from typing import List, Dict, Tuple

from opendssdirect import dss

from midspan_hif_utils import make_midspan_hif_dss, kron_reduce
from parse_opendss_file import (
    parse_opendss_to_mpc, merge_closed_switches_in_mpc_and_dss,
    build_global_y_per_unit,
)
from cal_pf import run_newton_powerflow_3p
from utilities.mea_fun import measurement_function
# IMPORTANT: use your updated function that now returns group distances
from lagrangian_m import run_lagrangian_polar

# ------------------------------------------------------------------
# User-tunable constants
# ------------------------------------------------------------------
PRISTINE_DSS = "Master.DSS"
WORKDIR = "hif_batch_tmp"       # scratch dir for per-target files
SWITCH_THRES = 2                # Ω for switch merging
R_HIF = 1.0                     # Ω (LN) used for *all* tests
FAULT_BUS = "FaultBus"          # mid-span bus label created by make_midspan_hif_dss
STD_P, STD_Q, STD_V = 1e-4, 1e-4, 1e-5
RNG_SEED = 1234

# Group-distance detector settings
TOP_K_RANK = 1                  # detect if faulted line ranks in top-K (K=1 => top-1)
COUNT_DIVERGENCE_AS_DETECTION = True  # count estimator divergence as detection

# ------------------------------------------------------------------
# DSS helpers: parse & merge one "family" of parallel lines
# ------------------------------------------------------------------
_line_re = re.compile(r"^\s*(new|New)\s+Line\.([^\s]+)\s+(.*)$")
_kv_re = re.compile(r"(\w+)\s*=\s*([^ \t]+)")
_HIF_LOAD_RE = re.compile(r'^\s*(new|edit)\s+load\.HIF_Load\b', re.IGNORECASE)

def _parse_line_defs(dss_text: List[str]) -> Dict[str, Dict]:
    parsed = {}
    for raw in dss_text:
        m = _line_re.match(raw)
        if not m:
            continue
        name = m.group(2)
        kvs = {k.lower(): v for k, v in _kv_re.findall(m.group(3))}
        entry = {
            "raw": raw,
            "name": name,
            "base_name": re.sub(r"_\d+$", "", name),
            "bus1": kvs.get("bus1", ""),
            "bus2": kvs.get("bus2", ""),
            "phases": kvs.get("phases", ""),
            "length": float(kvs.get("length", "0")),
            "units": kvs.get("units", "ft"),
            "geometry": kvs.get("geometry", ""),
        }
        parsed[f"Line.{name}"] = entry
    return parsed

_UNIT_TO_FT = {"ft": 1.0, "m": 3.280839895, "mi": 5280.0, "km": 3280.839895}
def _to_feet(L: float, units: str) -> float: return L * _UNIT_TO_FT.get(units.lower(), 1.0)

def _rewrite_length_units(raw_line: str, new_length_ft: float) -> str:
    parts = raw_line.split()
    parts = [p for p in parts if not p.lower().startswith("length=") and not p.lower().startswith("units=")]
    parts.append(f"length={new_length_ft:.6g}")
    parts.append("units=ft")
    return " ".join(parts)

def _comment_out(raw_line: str) -> str: return "!  " + raw_line

def make_target_merged_master(pristine_path: str, target_fullname: str, out_path: str) -> Tuple[float, int, str, str]:
    """
    Merge only the parallel 'family' of target_fullname into a single equivalent line.
    Returns (L_eq_ft, N_parallel, base_bus1, base_bus2) where base_bus* are the *main* bus names (without .1.2.3).
    """
    txt = open(pristine_path, "r").read().splitlines()
    parsed = _parse_line_defs(txt)
    if target_fullname not in parsed:
        raise ValueError(f"{target_fullname} not found in {pristine_path}.")
    target = parsed[target_fullname]
    base = target["base_name"]

    # family: same base_name + same endpoints + same geometry
    family_keys = []
    for fullname, f in parsed.items():
        if f["base_name"] != base:
            continue
        if f["bus1"].lower() == target["bus1"].lower() and \
           f["bus2"].lower() == target["bus2"].lower() and \
           f["geometry"].lower() == target["geometry"].lower():
            family_keys.append(fullname)

    lengths_ft = [_to_feet(parsed[k]["length"], parsed[k]["units"]) for k in family_keys]
    L_eq_ft = 1.0 / sum(1.0 / L for L in lengths_ft)
    N_parallel = len(lengths_ft)

    # Rewrite target, comment out the rest of the family
    base_raw = target["raw"]
    base_new = _rewrite_length_units(base_raw, L_eq_ft)
    out_txt = []
    for line in txt:
        m = _line_re.match(line)
        if not m:
            out_txt.append(line); continue
        fullname = f"Line.{m.group(2)}"
        if fullname == target_fullname:
            out_txt.append(base_new)
        elif fullname in family_keys:
            out_txt.append(_comment_out(line))
        else:
            out_txt.append(line)
    open(out_path, "w").write("\n".join(out_txt))

    # Return endpoints without phase suffixes
    base_bus1 = target["bus1"].split(".")[0]
    base_bus2 = target["bus2"].split(".")[0]
    return L_eq_ft, N_parallel, base_bus1, base_bus2

def list_secondary_base_lines(pristine_path: str) -> List[str]:
    txt = open(pristine_path, "r").read().splitlines()
    parsed = _parse_line_defs(txt)
    cands = []
    for fullname, f in parsed.items():
        if f["geometry"].lower() != "sec":
            continue
        if re.search(r"_\d+$", f["name"]):   # keep only base names
            continue
        cands.append(fullname)
    def _key(fn):
        m = re.search(r"Line\.(\d+)$", fn)
        return (int(m.group(1)) if m else 1_000_000, fn)
    return sorted(cands, key=_key)

def _kv_ln_at_bus(dss_file: str, fault_bus: str, default_kv_ln: float = 0.120) -> float:
    """
    Returns the line-to-neutral base kV at `fault_bus` by loading the DSS file and
    calling CalcVoltageBases. Falls back to `default_kv_ln` if unavailable.
    """
    # Fresh engine state
    dss.Command("Clear")
    dss.Command(f'Redirect "{os.path.abspath(dss_file)}"')
    dss.Command("CalcVoltageBases")

    # Try exact, then lowercase (OpenDSS can be case-insensitive in practice)
    ok = dss.Circuit.SetActiveBus(fault_bus) or dss.Circuit.SetActiveBus(fault_bus.lower())
    if not ok:
        print(f'WARN: bus "{fault_bus}" not found in {dss_file}; using default {default_kv_ln:.3f} kV (LN).')
        return default_kv_ln

    kv_base = dss.Bus.kVBase()   # For 3φ buses: kV_LL; for 1φ/2φ buses: kV_LN (OpenDSS convention)
    if kv_base <= 0:
        print(f'WARN: bus "{fault_bus}" has kVBase=0; using default {default_kv_ln:.3f} kV (LN).')
        return default_kv_ln

    # Heuristic: if the bus looks 3φ, convert LL→LN; otherwise assume LN
    try:
        num_nodes = dss.Bus.NumNodes()
    except Exception:
        num_nodes = 0

    # kv_ln = kv_base / math.sqrt(3.0) if num_nodes >= 3 else kv_base
    # Return the first three digits (kV LN)
    return round(kv_base, 3)

def add_kw_for_constant_resistance(dss_file: str, r_ohm: float, fault_bus: str,
                                   default_kv_ln: float = 0.120) -> None:
    """
    Update (or create) the HIF load so it models a constant-R fault at the correct LN base.
    For Wye, 1φ constant-Z (Model=2): P_spec = V_ln^2 / R (at nominal V).
    """
    if r_ohm <= 0:
        raise ValueError("r_ohm must be positive.")

    # 1) Get LN base at the fault bus from the actual circuit
    kv_ln = _kv_ln_at_bus(dss_file, fault_bus, default_kv_ln=default_kv_ln)

    # 2) Compute the kW that reproduces R at nominal V (Model=2 is constant Z)
    p_kw = (kv_ln * 1e3)**2 / r_ohm / 1e3  # (kV→V)^2 / Ω → W → kW

    # 3) Rewrite the HIF load line in the file (create if missing)
    txt = open(dss_file, "r", encoding="utf-8").read().splitlines()
    rewritten = False
    for k, line in enumerate(txt):
        if _HIF_LOAD_RE.match(line):
            txt[k] = (f'New Load.HIF_Load Bus1={fault_bus}.1 Phases=1 Conn=Wye Model=2 '
                      f'kW={p_kw:.6g} kvar=0 kV={kv_ln:.6g}')
            rewritten = True
            break

    if not rewritten:
        # Append a new definition near the end to be safe
        txt.append((f'New Load.HIF_Load Bus1={fault_bus}.1 Phases=1 Conn=Wye Model=2 '
                    f'kW={p_kw:.6g} kvar=0 kV={kv_ln:.6g}'))

    open(dss_file, "w", encoding="utf-8").write("\n".join(txt))

    # Optional sanity message (can be removed to keep logs clean)
    print(f'HIF load set at {kv_ln:.3f} kV (LN) -> {p_kw:.3f} kW for R={r_ohm:g} Ω at {fault_bus}')

def _find_line_index_by_endpoints(mpc, bus1_name: str, bus2_name: str) -> int:
    """Find the line index in mpc['line3p'] whose endpoints match bus1/bus2 (either direction)."""
    bmap = mpc["busname_to_id"]
    b1 = bmap[bus1_name.lower()]
    b2 = bmap[bus2_name.lower()]
    for i, row in enumerate(mpc["line3p"]):
        f, t = int(row[1]), int(row[2])
        if (f == b1 and t == b2) or (f == b2 and t == b1):
            return i
    return -1

# ------------------------------------------------------------------
# One target run using GROUP DISTANCES as detector
# ------------------------------------------------------------------
def run_one_target(merged_master: str,
                   target_fullname: str,
                   target_bus1: str,
                   target_bus2: str,
                   r_hif_ohm: float,
                   rng: np.random.Generator) -> Tuple[bool, bool, int, float, int]:
    """
    Returns: (detected, converged, N_parallel, L_eq_ft, target_rank)
    Detection is Top-K (TOP_K_RANK) by *group distance* (shunt-terminals).
    """
    # -- Build a faulted master with a mid-span HIF on the target line
    fault_master = os.path.join(os.path.dirname(merged_master), "Master_HIF.dss")
    make_midspan_hif_dss(merged_master, fault_master, target_fullname)
    add_kw_for_constant_resistance(fault_master, r_hif_ohm, FAULT_BUS)

    # -- Parse pristine & faulted and merge closed switches
    mpc_orig = parse_opendss_to_mpc(merged_master, baseMVA=1.0,
                                    lc_filename="LineConstantsCode.txt", slack_bus="p1")
    merge_closed_switches_in_mpc_and_dss(mpc_orig, SWITCH_THRES)

    mpc_fault = parse_opendss_to_mpc(fault_master, baseMVA=1.0,
                                     lc_filename="LineConstantsCode.txt", slack_bus="p1")
    merge_closed_switches_in_mpc_and_dss(mpc_fault, SWITCH_THRES)

    # -- PF on faulted to get "truth" voltages
    Vr, Vi, bp_full = run_newton_powerflow_3p(mpc_fault, 1e-6, 20)
    Vc_full = Vr + 1j * Vi

    # -- Global Y (faulted) + shunt for HIF
    dss.Command("Clear")
    dss.Command(f'Redirect "{fault_master}"')
    Y_full, node_order_list = build_global_y_per_unit(mpc_fault)

    hif_admittance = 1.0 / r_hif_ohm
    fault_bus_id = mpc_fault["busname_to_id"][FAULT_BUS.lower()]
    node_phase_map = {node_name: i for i, node_name in enumerate(node_order_list)}
    hif_node_name = f"bus{fault_bus_id}.1"
    if hif_node_name in node_phase_map:
        hif_idx = node_phase_map[hif_node_name]
        v_base_ln = (1.2e2 / math.sqrt(3))
        s_base = mpc_fault["baseMVA"] * 1e6
        z_base = v_base_ln ** 2 / s_base
        y_base = 1.0 / z_base
        y_hif_pu = hif_admittance / y_base
        Y_full = Y_full.tolil()
        Y_full[hif_idx, hif_idx] += y_hif_pu
        Y_full = Y_full.tocsc()

    # -- Kron-reduce the 3 phases of FaultBus
    bp_red = {(int(row[0]), ph): 3 * i + ph for i, row in enumerate(mpc_orig["bus3p"]) for ph in range(3)}
    elim_idx = [bp_full[(fault_bus_id, ph)] for ph in range(3)]
    Y_red = kron_reduce(Y_full, elim_idx)

    V_red = np.array([Vc_full[bp_full[(int(row[0]), ph)]] for row in mpc_orig["bus3p"] for ph in range(3)])
    x_red = np.hstack([np.abs(V_red), np.angle(V_red)])

    # -- Measurements with Gaussian noise
    z = measurement_function(x_red, Y_red, mpc_orig, bp_red)
    mP = len(bp_red)
    R_diag = np.array([STD_P**2]*mP + [STD_Q**2]*mP + [STD_V**2]*mP)
    noise = rng.normal(0.0, np.sqrt(R_diag))
    z_noisy = z + noise
    R = np.diag(R_diag)

    # -- Build Y of pristine (merged) for the estimator
    dss.Command("Clear")
    dss.Command(f'Redirect "{merged_master}"')
    Y_orig, _ = build_global_y_per_unit(mpc_orig)

    # -- RUN DSSE: must be the updated function returning sorted group distances
    x0 = x_red
    x_est, converged, lambdaN, sorted_group_distances = run_lagrangian_polar(
        z_noisy, x0, bp_red, Y_orig, R, mpc_orig
    )

    # -- If divergence, apply policy
    if not converged:
        detected = COUNT_DIVERGENCE_AS_DETECTION
        return detected, False, None, None, -1

    # -- Target line index in mpc_orig
    target_idx = _find_line_index_by_endpoints(mpc_orig, target_bus1, target_bus2)
    if target_idx < 0:
        # can't map: treat as miss but keep data
        return False, True, None, None, -1

    # -- Rank of the target by group distance (0=best)
    rank = -1
    for r, (i_line, _) in enumerate(sorted_group_distances):
        if i_line == target_idx:
            rank = r
            break

    detected = (rank >= 0 and rank < TOP_K_RANK)
    return detected, True, None, None, rank

# ------------------------------------------------------------------
# Batch driver
# ------------------------------------------------------------------
def main():
    rng = np.random.default_rng(RNG_SEED)
    # if os.path.exists(WORKDIR):
    #     shutil.rmtree(WORKDIR)
    # os.makedirs(WORKDIR, exist_ok=True)

    candidates = list_secondary_base_lines(PRISTINE_DSS)
    print(f"Found {len(candidates)} secondary base lines to test.")

    n_total = 0
    n_detect = 0
    n_conv = 0
    rows = []

    for target_fullname in candidates:
        n_total += 1
        tag = target_fullname.replace(".", "_")
        merged_master = os.path.join(WORKDIR, f"Master_merged_{tag}.dss")
        try:
            L_eq_ft, N_parallel, base_bus1, base_bus2 = make_target_merged_master(
                PRISTINE_DSS, target_fullname, merged_master
            )
        except Exception as e:
            print(f"[{target_fullname}] merge failed: {e}")
            rows.append((target_fullname, 0, 0, -1, np.nan, 0, 0.0))
            continue

        detected, converged, _, _, rank = run_one_target(
                merged_master, target_fullname, base_bus1, base_bus2, R_HIF, rng
            )
        n_conv += int(converged)
        n_detect += int(detected)
        status = "DETECTED" if detected else "miss"
        print(f"[{target_fullname}] {status} (rank={rank if rank>=0 else 'NA'})")
        rows.append((target_fullname, int(detected), int(converged), rank, np.nan, N_parallel, L_eq_ft))

    rate = (n_detect / n_total) if n_total else 0.0
    print("\n=== Batch HIF (Group Distance) detection summary ===")
    print(f"Targets tested : {n_total}")
    print(f"Converged      : {n_conv}")
    print(f"Detections     : {n_detect}")
    print(f"Detection rate : {rate:.3%}   (Top-K={TOP_K_RANK}, divergence counts as detection={COUNT_DIVERGENCE_AS_DETECTION})")

    # Write CSV
    out_csv = os.path.join(WORKDIR, "hif_batch_results_group.csv")
    with open(out_csv, "w") as f:
        f.write("line,detected,converged,rank,placeholder,max_abs_lambdaN,N_parallel,L_eq_ft\n")
        for (line, det, conv, rank, _, npar, leq) in rows:
            f.write(f"{line},{det},{conv},{rank},{''},{''},{npar},{leq}\n")
    print(f"Per-line results → {out_csv}")

if __name__ == "__main__":
    main()
>>>>>>> group-ind:342/run_batched_test.py
