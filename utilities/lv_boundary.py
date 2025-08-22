import math
from copy import deepcopy
import opendssdirect as dss
import numpy as np

def _base_name(b):
    return (b or "").split(".", 1)[0].split(":", 1)[0].lower()

def buses_120_208_v(kvll_tol=0.03):
    """
    Return set of base bus names whose base kV_LL is ~0.208 kV (±kvll_tol kV).
    """
    out = set()
    for b in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(b)
        kv_ln = dss.Bus.kVBase() or 0.0
        kv_ll = kv_ln * math.sqrt(3.0)
        if abs(kv_ll - 0.208) <= kvll_tol:
            out.add(_base_name(b))
    return out

def secondary_heads(lv_buses, kvll_lv=0.208, kvll_tol=0.03):
    """
    Return LV 'head' buses:
      any 120/208‑V bus that is connected (enabled, not open) through a *Transformer*
      to at least one non‑LV bus. Also checks Lines as a fallback.

    Args
    ----
    lv_buses : Iterable[str]
        Set of base bus names that are 120/208‑V (from buses_120_208_v()).
    kvll_lv : float
        Target LL base kV for the secondary (default 0.208 kV).
    kvll_tol : float
        Tolerance around kvll_lv for classification.

    Notes
    -----
    - We prefer membership in `lv_buses` to classify an LV terminal.
      If a terminal's bus isn't in that set, we fall back to checking Bus.kVBase().
    - A terminal is considered 'closed' if *any* conductor on that terminal is not open.
    """
    import math
    lv = set(lv_buses)
    heads = set()

    # ---------- 1) Detect LV heads via TRANSFORMERS ----------
    for xf in dss.Transformers.AllNames():
        dss.Transformers.Name(xf)
        if not dss.CktElement.Enabled():
            continue

        buses = dss.CktElement.BusNames()  # one per terminal
        if not buses:
            continue

        nterm = dss.CktElement.NumTerminals()
        # OpenDSS: NumConductors is per terminal
        nc = max(1, dss.CktElement.NumConductors())

        # terminal closed flags
        term_closed = [False] * nterm
        for t in range(1, nterm + 1):
            # consider terminal t 'closed' if any conductor is not open
            for c in range(1, nc + 1):
                try:
                    if not dss.CktElement.IsOpen(t, c):
                        term_closed[t - 1] = True
                        break
                except Exception:
                    # Some builds can throw on IsOpen for certain conductor indices.
                    # Be conservative: treat as closed if API misreports.
                    term_closed[t - 1] = True
                    break

        # classify terminals as LV / non‑LV
        term_is_lv = []
        base_names = []
        for t in range(nterm):
            bn = _base_name(buses[t]) if t < len(buses) else ""
            base_names.append(bn)

            if bn in lv:
                term_is_lv.append(True)
            else:
                # fall back to voltage base if not in lv set
                try:
                    dss.Circuit.SetActiveBus(bn)
                    kvln = dss.Bus.kVBase() or 0.0
                    kvll = kvln * math.sqrt(3.0)
                    term_is_lv.append(abs(kvll - kvll_lv) <= kvll_tol)
                except Exception:
                    term_is_lv.append(False)

        # If an LV terminal is closed and *any* other closed terminal is not LV, mark LV bus as a head
        for t in range(nterm):
            if not term_closed[t] or not term_is_lv[t]:
                continue
            for u in range(nterm):
                if u == t or not term_closed[u]:
                    continue
                if not term_is_lv[u]:
                    heads.add(base_names[t])
                    break

    # ---------- 2) Fallback: detect LV↔non‑LV via LINES (rare for prim/secondary, but safe to keep) ----------
    for ln in dss.Lines.AllNames():
        dss.Lines.Name(ln)
        if not dss.CktElement.Enabled():
            continue
        buses = dss.CktElement.BusNames()
        if len(buses) < 2:
            continue

        fb = _base_name(buses[0]); tb = _base_name(buses[1])
        nc = max(1, dss.CktElement.NumConductors())

        closed = any(
            (not dss.CktElement.IsOpen(1, c)) and (not dss.CktElement.IsOpen(2, c))
            for c in range(1, nc + 1)
        )
        if not closed:
            continue

        in_lv = (fb in lv, tb in lv)
        if in_lv == (True, False):
            heads.add(fb)
        elif in_lv == (False, True):
            heads.add(tb)

    return heads


def snapshot_head_setpoints(head_buses):
    """
    Read |V| and angle at each head bus from the *current full* DSS solution.
    Returns dict:
      head -> {"vm_pu": [va, vb, vc], "va_rad": [aa, ab, ac], "kv_ln": base_kV_LN}
    NOTE: vm_pu is per-unit on each bus's own L-N base.
    """
    out = {}
    for bn in head_buses:
        dss.Circuit.SetActiveBus(bn)
        kv_ln = dss.Bus.kVBase() or 0.120   # kV L-N
        mags_angles = dss.Bus.VMagAngle()   # [Vmag1, ang1, Vmag2, ang2, ...] (volts, degrees)
        nodes = dss.Bus.Nodes()
        # pack phase-indexed arrays; default zeros for missing phases
        vm = [0.0, 0.0, 0.0]; va = [0.0, 0.0, 0.0]
        for i in range(0, len(mags_angles), 2):
            vmag = mags_angles[i]
            vang_deg = mags_angles[i+1]
            ph = nodes[i//2] if (i//2) < len(nodes) else 0
            if ph in (1,2,3):
                vm[ph-1] = (vmag / (kv_ln*1e3))  # per-unit on L-N base
                va[ph-1] = math.radians(vang_deg)
        out[bn] = {"vm_pu": vm, "va_rad": va, "kv_ln": kv_ln}
    return out

def disable_outside_dss(keep_buses):
    """
    In DSS, disable any element whose any terminal bus is not in keep_buses.
    """
    keep = set(keep_buses)
    # Lines
    for ln in dss.Lines.AllNames():
        dss.Lines.Name(ln)
        buses = dss.CktElement.BusNames()
        fb = _base_name(buses[0]) if len(buses)>0 else ""
        tb = _base_name(buses[1]) if len(buses)>1 else ""
        if fb not in keep or tb not in keep:
            dss.Command(f"Edit Line.{ln} enabled=no")
    # Transformers
    for xf in dss.Transformers.AllNames():
        dss.Transformers.Name(xf)
        buses = dss.CktElement.BusNames()
        fb = _base_name(buses[0]) if len(buses)>0 else ""
        tb = _base_name(buses[1]) if len(buses)>1 else ""
        if fb not in keep or tb not in keep:
            dss.Command(f"Edit Transformer.{xf} enabled=no")
    # Loads
    for ld in dss.Loads.AllNames():
        dss.Loads.Name(ld)
        b = _base_name(dss.CktElement.BusNames()[0])
        if b not in keep:
            dss.Command(f"Edit Load.{ld} enabled=no")
    # Generators
    for gn in dss.Generators.AllNames():
        dss.Generators.Name(gn)
        b = _base_name(dss.CktElement.BusNames()[0])
        if b not in keep:
            dss.Command(f"Edit Generator.{gn} enabled=no")

def filter_mpc_to_buses(mpc_in, keep_buses):
    """
    Deep-copy MPC and keep only buses in keep_buses; reindex to 1..N.
    Does NOT set any PV/slack; that's handled by apply_boundary_pv().
    """
    mpc = deepcopy(mpc_in)
    keep = set(keep_buses)
    old_map = mpc["busname_to_id"]
    inv_old = {v:k for k,v in old_map.items()}
    # Build new bus sets
    kept_names = [bn for bn in old_map.keys() if bn in keep]
    if not kept_names:
        raise RuntimeError("filter_mpc_to_buses: no buses to keep.")
    # new bus3p
    id_to_row = {int(r[0]): r for r in mpc["bus3p"]}
    new_busname_to_id = {}
    new_bus_rows = []
    for nid, bn in enumerate(kept_names, start=1):
        old_id = int(old_map[bn])
        row = id_to_row[old_id].copy()
        row[0] = nid
        # reset all to PQ for now; boundary setup comes later
        row[1] = 1
        new_bus_rows.append(row)
        new_busname_to_id[bn] = nid
    mpc["bus3p"] = np.array(new_bus_rows, dtype=float)
    mpc["busname_to_id"] = new_busname_to_id
    # lines
    if "line3p" in mpc and mpc["line3p"].size:
        new_lines = []
        for row in mpc["line3p"]:
            f_old = int(row[1]); t_old = int(row[2])
            fb = inv_old.get(f_old); tb = inv_old.get(t_old)
            if fb in keep and tb in keep:
                r2 = list(row)
                r2[1] = new_busname_to_id[fb]
                r2[2] = new_busname_to_id[tb]
                new_lines.append(r2)
        mpc["line3p"] = np.array(new_lines, dtype=object) if new_lines else np.zeros((0,7), dtype=object)
    # xfmr
    if "xfmr3p" in mpc and mpc["xfmr3p"].size:
        new_x = []
        for row in mpc["xfmr3p"]:
            f_old = int(row[1]); t_old = int(row[2])
            fb = inv_old.get(f_old); tb = inv_old.get(t_old)
            if fb in keep and tb in keep:
                r2 = list(row); r2[1] = new_busname_to_id[fb]; r2[2] = new_busname_to_id[tb]
                new_x.append(r2)
        mpc["xfmr3p"] = np.array(new_x, dtype=float) if new_x else np.zeros((0,9), dtype=float)
    # loads
    if "load3p" in mpc and mpc["load3p"].size:
        new_ld = []
        for row in mpc["load3p"]:
            bn = inv_old.get(int(row[1]))
            if bn in keep:
                r2 = list(row); r2[1] = new_busname_to_id[bn]; new_ld.append(r2)
        mpc["load3p"] = np.array(new_ld, dtype=float) if new_ld else np.zeros((0,9), dtype=float)
    # gens
    mpc["gen3p"] = np.zeros((0,12), dtype=float)  # we will recreate boundary gens
    # node order
    if "node_order" in mpc and mpc["node_order"]:
        mpc["node_order"] = [nm for nm in mpc["node_order"] if _base_name(nm) in keep]
    return mpc

def apply_boundary_pv(mpc, head_setpoints, slack_bus=None):
    """
    Insert PV rows (and one slack) at head buses using |V| from head_setpoints (per-unit L-N).
    head_setpoints: dict bus -> {"vm_pu":[va,vb,vc], "va_rad":[aa,ab,ac]}
    slack_bus: if None, pick the first head; otherwise name (case-insensitive).
    """
    bn2id = mpc["busname_to_id"]
    # 1) mark PV/slack in bus3p, seed angles
    if slack_bus is None:
        slack_bus = next(iter(head_setpoints.keys()))
    slack_bus = _base_name(slack_bus)
    for i in range(len(mpc["bus3p"])):
        # default PQ
        mpc["bus3p"][i,1] = 1
    for bn, sp in head_setpoints.items():
        bn_l = _base_name(bn)
        if bn_l not in bn2id:  # head may have been pruned out (shouldn't happen)
            continue
        idx = None
        # find row index in bus3p
        for r, row in enumerate(mpc["bus3p"]):
            if int(row[0]) == bn2id[bn_l]:
                idx = r; break
        if idx is None:
            continue
        if bn_l == slack_bus:
            mpc["bus3p"][idx,1] = 3  # slack
        else:
            mpc["bus3p"][idx,1] = 2  # PV
        # seed angles to measured (helps convergence)
        vaA, vaB, vaC = sp["va_rad"]
        mpc["bus3p"][idx,6] = vaA
        mpc["bus3p"][idx,7] = vaB
        mpc["bus3p"][idx,8] = vaC
    # 2) create/append gen3p rows with Vg setpoints
    gen_rows = []
    gid = 1
    for bn, sp in head_setpoints.items():
        bn_l = _base_name(bn)
        if bn_l not in bn2id:
            continue
        bid = bn2id[bn_l]
        VgA, VgB, VgC = sp["vm_pu"]
        # Pg/Qg initial guesses 0; PF will solve Q for PV; slack will balance P/Q
        gen_rows.append([gid, bid, 1, VgA, VgB, VgC, 0.0,0.0,0.0, 0.0,0.0,0.0])
        gid += 1
    mpc["gen3p"] = np.array(gen_rows, dtype=float) if gen_rows else np.zeros((0,12), dtype=float)
    return mpc
