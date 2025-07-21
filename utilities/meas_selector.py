from collections import defaultdict, deque
import numpy as np


def _buses_within_hops(mpc, target_line_idx, hops=2):

    ln = mpc["line3p"][target_line_idx]
    fbus, tbus = int(ln[1]), int(ln[2])

    # adjacency list
    adj = defaultdict(set)
    for row in mpc["line3p"]:
        if int(row[3]) == 0:          # disabled line
            continue
        a, b = int(row[1]), int(row[2])
        adj[a].add(b)
        adj[b].add(a)

    visited = {fbus, tbus}
    q = deque([(fbus, 0), (tbus, 0)])
    while q:
        cur, d = q.popleft()
        if d == hops:
            continue
        for nb in adj[cur]:
            if nb not in visited:
                visited.add(nb)
                q.append((nb, d + 1))
    return visited

def build_meas_mask(mpc, busphase_map, target_line_idx, hops=2):
    """
    Produce ONE numpy array of row‑indices to keep from the full measurement
    vector `z` and the corresponding model output `h(x)`.

    Measurement ordering must match `measurement_function()`:

        [ P‑inj (3φ buses)
          Q‑inj
          P/Q flows (per line, 3φ, Pfrom Qfrom Pto Qto)   ← 12 per line
          |V|  ]

    The mask contains
      • all P‑inj, Q‑inj, |V| rows whose buses are within `hops` of the line ends
      • all branch‑flow rows for lines whose *both* ends are in that bus set
    """
    bus_set = _buses_within_hops(mpc, target_line_idx, hops=hops)

    nnodeph   = len(busphase_map)
    nlines    = len(mpc["line3p"])
    flow_blk  = 12                       # rows per line (3φ × P,Q at both ends)

    # ---------- injections -------------------------------------------------
    pinj_idx  = [i for i, (b, _) in enumerate(busphase_map) if b in bus_set]
    qinj_idx  = [i + nnodeph for i in pinj_idx]

    # ---------- branch flows -----------------------------------------------
    flow_start = 2 * nnodeph
    flow_idx   = []
    for li, row in enumerate(mpc["line3p"]):
        if int(row[1]) in bus_set and int(row[2]) in bus_set:
            base = flow_start + li * flow_blk
            flow_idx.extend(range(base, base + flow_blk))

    # ---------- magnitudes --------------------------------------------------
    vmag_start = flow_start + nlines * flow_blk
    vmag_idx   = [vmag_start + i for i, (b, _) in enumerate(busphase_map)
                  if b in bus_set]

    mask = np.array(sorted(pinj_idx + qinj_idx + flow_idx + vmag_idx),
                    dtype=int)
    return mask

def build_state_col_mask(busphase_map, ref_bus_idx, bus_set):
    """
    Return a list of column indices (after slack‑angle removal) that
    correspond to state variables (|V| and δ) whose *bus* is in `bus_set`.

    Parameters
    ----------
    busphase_map : list[(bus_id, phase)]
    ref_bus_idx  : int     – index of the *bus* that is the angle reference
    bus_set      : set[int]

    Notes
    -----
    •  Column ordering of |V| block:     0 … nnodeφ−1   (exactly busphase_map order)
    •  Column ordering of δ  block:     nnodeφ … 2 nnodeφ−3,
       **but δ‑columns for the three reference phases are skipped**.
    """
    nnodeph = len(busphase_map)

    # map busphase index → δ column (after slack removal)
    delta_cols = {}
    dcol = 0
    for i, (b, _) in enumerate(busphase_map):
        if b == ref_bus_idx:          # skip slack angles
            continue
        delta_cols[i] = nnodeph + dcol
        dcol += 1

    cols = []
    for i, (b, _) in enumerate(busphase_map):
        if b not in bus_set:
            continue
        # |V|
        cols.append(i)
        # δ  (unless slack)
        if i in delta_cols:
            cols.append(delta_cols[i])

    return np.array(sorted(cols), dtype=int)