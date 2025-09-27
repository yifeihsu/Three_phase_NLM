import os
import re
import math
import numpy as np
from typing import List, Dict, Tuple, Optional

from opendssdirect import dss

from midspan_hif_utils import make_midspan_hif_dss, kron_reduce
from parse_opendss_file import (
    parse_opendss_to_mpc, merge_closed_switches_in_mpc_and_dss,
    build_global_y_per_unit,
)
from cal_pf import run_newton_powerflow_3p
from utilities.mea_fun import measurement_function
from lagrangian_m import run_lagrangian_polar

# ------------------------------------------------------------------
# User-tunable constants
# ------------------------------------------------------------------
PRISTINE_DSS = "Secondary_Equivalent.dss"
WORKDIR = "hif_batch_tmp"
SWITCH_THRES = 2
R_HIF = 1
FAULT_BUS = "FaultBus"

# Measurement settings
STD_P, STD_Q, STD_V = 1e-4, 1e-4, 2e-4
USE_MEAS_NOISE = True
RNG_SEED = 1234

# Group-distance detector settings
TOP_K_RANK = 1
COUNT_DIVERGENCE_AS_DETECTION = True

# ------------------------------------------------------------------
# DSS line parsing helpers
# ------------------------------------------------------------------
_line_re = re.compile(r"^\s*(new|New)\s+Line\.([^\s]+)\s+(.*)$")
_kv_re   = re.compile(r"(\w+)\s*=\s*([^ \t]+)")
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
def _to_feet(L: float, units: str) -> float:
    return L * _UNIT_TO_FT.get(units.lower(), 1.0)

def list_secondary_base_lines(pristine_path: str) -> List[str]:
    """
    Return the list of *base family* identifiers (without trailing _N) for geometry=sec.
    E.g., returns ['Line.123', 'Line.207', ...]; these are actual line names in the file.
    """
    txt = open(pristine_path, "r", encoding="utf-8").read().splitlines()
    parsed = _parse_line_defs(txt)
    cands = []
    for fullname, f in parsed.items():
        if f["geometry"].lower() != "sec":
            continue
        if re.search(r"_\d+$", f["name"]):   # keep only base names (no suffix)
            continue
        cands.append(fullname)
    def _key(fn):
        m = re.search(r"Line\.(\d+)$", fn)
        return (int(m.group(1)) if m else 1_000_000, fn)
    return sorted(cands, key=_key)

def get_parallel_family(pristine_path: str, base_fullname: str) -> Tuple[List[str], str, str, str, Dict[str, float]]:
    """
    For a given base family 'Line.X', return:
      (members_fullnames, base_bus1, base_bus2, geometry, length_ft_by_member)
    Only members with same endpoints and same geometry are included.
    """
    txt = open(pristine_path, "r", encoding="utf-8").read().splitlines()
    parsed = _parse_line_defs(txt)
    if base_fullname not in parsed:
        raise ValueError(f"{base_fullname} not found in {pristine_path}.")
    target = parsed[base_fullname]
    base = target["base_name"]

    members = []
    length_ft_by_member = {}
    for fullname, f in parsed.items():
        if f["base_name"] != base:
            continue
        if f["bus1"].lower() == target["bus1"].lower() and \
           f["bus2"].lower() == target["bus2"].lower() and \
           f["geometry"].lower() == target["geometry"].lower():
            members.append(fullname)
            length_ft_by_member[fullname] = _to_feet(f["length"], f["units"])
    # Prefer the no-suffix member; else pick the first in sorted order
    members_sorted = sorted(members, key=lambda s: (0 if s == base_fullname else 1, s))
    return members_sorted, target["bus1"].split(".")[0], target["bus2"].split(".")[0], target["geometry"], length_ft_by_member

# ------------------------------------------------------------------
# Voltage base + HIF load helpers
# ------------------------------------------------------------------
def _kv_ln_at_bus(dss_file: str, fault_bus: str, default_kv_ln: float = 0.120) -> float:
    """
    Returns the *line-to-neutral* base kV at `fault_bus` by loading the DSS file and
    calling CalcVoltageBases. If not found, returns default_kv_ln.
    """
    dss.Command("Clear")
    dss.Command(f'Redirect "{os.path.abspath(dss_file)}"')
    dss.Command("CalcVoltageBases")

    ok = dss.Circuit.SetActiveBus(fault_bus) or dss.Circuit.SetActiveBus(fault_bus.lower())
    if not ok:
        print(f'WARN: bus "{fault_bus}" not found in {dss_file}; using default {default_kv_ln:.3f} kV (LN).')
        return default_kv_ln

    kv_base = dss.Bus.kVBase()  # OpenDSS: 3φ buses -> kV_LL; 1φ/2φ buses -> kV_LN
    if kv_base <= 0:
        print(f'WARN: bus "{fault_bus}" has kVBase=0; using default {default_kv_ln:.3f} kV (LN).')
        return default_kv_ln

    try:
        num_nodes = dss.Bus.NumNodes()
    except Exception:
        num_nodes = 0

    kv_ln = kv_base / math.sqrt(3.0) if num_nodes >= 3 else kv_base
    return round(float(kv_base), 3)

def add_kw_for_constant_resistance(dss_file: str, r_ohm: float, fault_bus: str,
                                   default_kv_ln: float = 0.120) -> float:
    """
    Update (or create) Load.HIF_Load so it models a constant-R fault at the correct LN base.
    For Wye, 1φ constant-Z (Model=2): P_spec = V_ln^2 / R  (at nominal V).
    Returns the kV (LN) used.
    """
    if r_ohm <= 0:
        raise ValueError("r_ohm must be positive.")

    kv_ln = _kv_ln_at_bus(dss_file, fault_bus, default_kv_ln=default_kv_ln)
    p_kw = (kv_ln * 1e3)**2 / r_ohm / 1e3  # (kV→V)^2 / Ω → kW

    txt = open(dss_file, "r", encoding="utf-8").read().splitlines()
    rewritten = False
    for k, line in enumerate(txt):
        if _HIF_LOAD_RE.match(line):
            txt[k] = (f'New Load.HIF_Load Bus1={fault_bus}.1 Phases=1 Conn=Wye Model=2 '
                      f'kW={p_kw:.6g} kvar=0 kV={kv_ln:.6g}')
            rewritten = True
            break

    if not rewritten:
        txt.append((f'New Load.HIF_Load Bus1={fault_bus}.1 Phases=1 Conn=Wye Model=2 '
                    f'kW={p_kw:.6g} kvar=0 kV={kv_ln:.6g}'))

    open(dss_file, "w", encoding="utf-8").write("\n".join(txt))
    print(f'HIF load set at {kv_ln:.3f} kV (LN) -> {p_kw:.3f} kW for R={r_ohm:g} Ω at {fault_bus}')
    return kv_ln

# ------------------------------------------------------------------
# Mapping helpers
# ------------------------------------------------------------------
def _find_line_indices_by_endpoints(mpc, bus1_name: str, bus2_name: str) -> List[int]:
    """
    Return **all** line indices in mpc['line3p'] whose endpoints match bus1/bus2 (either direction).
    This handles parallel lines that share endpoints.
    """
    bmap = mpc["busname_to_id"]
    b1 = bmap[bus1_name.lower()]
    b2 = bmap[bus2_name.lower()]
    idxs = []
    for i, row in enumerate(mpc["line3p"]):
        f, t = int(row[1]), int(row[2])
        if (f == b1 and t == b2) or (f == b2 and t == b1):
            idxs.append(i)
    return idxs

# ------------------------------------------------------------------
# One target run using GROUP DISTANCES as detector
# ------------------------------------------------------------------
def run_one_target(pristine_master: str,
                   target_line_fullname: str,
                   family_bus1: str,
                   family_bus2: str,
                   r_hif_ohm: float,
                   rng: np.random.Generator) -> Tuple[bool, bool, int]:
    """
    Create a faulted copy by injecting a midspan HIF on *one* member (target_line_fullname).
    Returns: (detected, converged, best_rank_in_family)
    """
    # -- Build a faulted master with a midspan HIF on the target line (no cable merging)
    tag = target_line_fullname.replace(".", "_")
    os.makedirs(WORKDIR, exist_ok=True)
    fault_master = os.path.join(WORKDIR, f"Secondary_Equivalent_faulted_{tag}.dss")

    make_midspan_hif_dss(pristine_master, fault_master, target_line_fullname)
    kv_ln_used = add_kw_for_constant_resistance(fault_master, r_hif_ohm, FAULT_BUS)

    # -- Parse pristine & faulted and merge closed switches
    mpc_orig = parse_opendss_to_mpc(pristine_master, baseMVA=1.0,
                                    lc_filename="LineConstantsCode.txt", slack_bus="s148")
    merge_closed_switches_in_mpc_and_dss(mpc_orig, SWITCH_THRES)

    mpc_fault = parse_opendss_to_mpc(fault_master, baseMVA=1.0,
                                     lc_filename="LineConstantsCode.txt", slack_bus="s148")
    merge_closed_switches_in_mpc_and_dss(mpc_fault, SWITCH_THRES)

    # -- PF on faulted to get "truth" voltages
    Vr, Vi, bp_full = run_newton_powerflow_3p(mpc_fault, 1e-6, 20)
    Vc_full = Vr + 1j * Vi

    # -- Global Y (faulted) + shunt for HIF
    dss.Command("Clear")
    dss.Command(f'Redirect "{fault_master}"')
    Y_full, node_order_list = build_global_y_per_unit(mpc_fault)

    # Append HIF shunt admittance in per unit at FAULT_BUS.1
    hif_admittance = 1.0 / r_hif_ohm  # S (physical units)
    node_phase_map = {node_name: i for i, node_name in enumerate(node_order_list)}
    # Base conversion to per-unit using kV (LN) at FaultBus and system base MVA (3φ)
    s_base_3ph = mpc_fault["baseMVA"] * 1e6
    z_base = (kv_ln_used * 1e3) ** 2 / s_base_3ph  # note: same convention as build_global_y_per_unit
    y_base = 1.0 / z_base
    y_hif_pu = hif_admittance / y_base

    # Index of FAULT_BUS phase-1 in the Y ordering
    dss_fault_id = mpc_fault["busname_to_id"].get(FAULT_BUS.lower(), None)
    if dss_fault_id is not None:
        hif_node_name = f"bus{dss_fault_id}.1"
        if hif_node_name in node_phase_map:
            hif_idx = node_phase_map[hif_node_name]
            Y_full = Y_full.tolil()
            Y_full[hif_idx, hif_idx] += y_hif_pu
            Y_full = Y_full.tocsc()

    # -- Kron-reduce the (up to) 3 phases of FaultBus in Y_full
    elim_idx = [
        node_phase_map[nm] for nm in
        [f"bus{dss_fault_id}.{ph}" for ph in (1, 2, 3)]
        if nm in node_phase_map
    ]
    if elim_idx:
        Y_red = kron_reduce(Y_full, elim_idx)
    else:
        Y_red = Y_full

    # -- Build reduced state vector x_red from the PF truth (faulted)
    V_red = np.array([Vc_full[bp_full[(int(row[0]), ph)]]
                      for i, row in enumerate(mpc_orig["bus3p"]) for ph in range(3)])
    x_red = np.hstack([np.abs(V_red), np.angle(V_red)])

    # -- Measurements (with optional Gaussian noise)
    z = measurement_function(x_red, Y_red, mpc_orig,
                             {(int(row[0]), ph): 3 * i + ph
                              for i, row in enumerate(mpc_orig["bus3p"]) for ph in range(3)})
    mP = len(mpc_orig["bus3p"]) * 3
    R_diag = np.array([STD_P**2]*mP + [STD_Q**2]*mP + [STD_V**2]*mP)
    z = z + rng.normal(0.0, np.sqrt(R_diag))
    R = np.diag(R_diag)


    # -- Build Y of pristine (unfaulted) for the estimator
    dss.Command("Clear")
    dss.Command(f'Redirect "{pristine_master}"')
    Y_orig, _ = build_global_y_per_unit(mpc_orig)

    # -- RUN DSSE: must be the updated function returning sorted group distances
    x0 = x_red
    x_est, converged, lambdaN, sorted_group_distances = run_lagrangian_polar(
        z, x0, {(int(row[0]), ph): 3 * i + ph
                for i, row in enumerate(mpc_orig["bus3p"]) for ph in range(3)},
        Y_orig, R, mpc_orig
    )

    if not converged:
        detected = COUNT_DIVERGENCE_AS_DETECTION
        return detected, False, -1

    # -- Rank among all parallel members that share the family endpoints
    family_indices = _find_line_indices_by_endpoints(mpc_orig, family_bus1, family_bus2)
    ranks = {i: r for r, (i, _) in enumerate(sorted_group_distances)}
    present_ranks = [ranks[i] for i in family_indices if i in ranks]
    best_rank = min(present_ranks) if present_ranks else -1

    detected = (best_rank >= 0 and best_rank < TOP_K_RANK)
    return detected, True, best_rank

# ------------------------------------------------------------------
# Batch driver
# ------------------------------------------------------------------
def main():
    rng = np.random.default_rng(RNG_SEED)
    os.makedirs(WORKDIR, exist_ok=True)

    families = list_secondary_base_lines(PRISTINE_DSS)
    print(f"Found {len(families)} secondary base *families* to test (no cable merging).")

    n_total = 0
    n_detect = 0
    n_conv = 0
    rows = []

    for base_fullname in families:
        n_total += 1
        try:
            members, base_bus1, base_bus2, geom, length_ft_by_member = get_parallel_family(
                PRISTINE_DSS, base_fullname
            )
            # Choose the target member to fault: prefer the no-suffix member, else the first
            target_fullname = members[0]
            N_parallel = len(members)
            L_member_ft = length_ft_by_member[target_fullname]
        except Exception as e:
            print(f"[{base_fullname}] family parse failed: {e}")
            rows.append((base_fullname, 0, 0, -1, np.nan, 0, 0.0))
            continue

        detected, converged, best_rank = run_one_target(
            PRISTINE_DSS, target_fullname, base_bus1, base_bus2, R_HIF, rng
        )
        n_conv += int(converged)
        n_detect += int(detected)
        status = "DETECTED" if detected else "miss"
        print(f"[{target_fullname}] {status} (best rank among family={best_rank if best_rank>=0 else 'NA'})")
        # Keep CSV schema: L_eq_ft now holds the chosen member's length (ft)
        rows.append((target_fullname, int(detected), int(converged), best_rank, np.nan, N_parallel, L_member_ft))

    rate = (n_detect / n_total) if n_total else 0.0
    print("\n=== Batch HIF (Group Distance) detection summary ===")
    print(f"Families tested : {n_total}")
    print(f"Converged       : {n_conv}")
    print(f"Detections      : {n_detect}")
    print(f"Detection rate  : {rate:.3%}   (Top-K={TOP_K_RANK}, divergence counts as detection={COUNT_DIVERGENCE_AS_DETECTION})")

    # Write CSV
    out_csv = os.path.join(WORKDIR, "hif_batch_results_group.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("line,detected,converged,rank,placeholder,max_abs_lambdaN,N_parallel,L_eq_ft\n")
        for (line, det, conv, rank, _, npar, lft) in rows:
            f.write(f"{line},{det},{conv},{rank},{''},{''},{npar},{lft}\n")
    print(f"Per-line results → {out_csv}")

if __name__ == "__main__":
    main()
