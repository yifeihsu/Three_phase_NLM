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
