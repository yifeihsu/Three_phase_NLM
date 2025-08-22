import os
import re
from collections import defaultdict, deque
import opendssdirect as dss

# --- Configuration ---
master_dss_file_path = r"C:\Users\Holiday\OneDrive\NYU Projects\Three_ph_NLM\Master.dss"
output_dss_file = "Secondary_Equivalent.dss"

# Heuristic: explicitly exclude “spot network” buses named like 'S123' (uppercase S)
EXCLUDE_SPOT_BY_NAME = True
SPOT_NAME_REGEX = re.compile(r"^S\d+\b")

# --- Small data class ---
class TheveninSource:
    """Hold equivalent source data for a boundary bus."""
    def __init__(self, name, bus, kv_ll, pu, angle_deg, r1, x1, r0, x0, r2, x2):
        self.name = name
        self.bus = bus
        self.kv_ll = kv_ll          # base LL kV for 3φ sources (0.208 for 120/208 V)
        self.pu = pu                # pre-fault pu magnitude (phase A)
        self.angle_deg = angle_deg  # pre-fault angle of phase A (deg)
        self.r1, self.x1 = r1, x1
        self.r0, self.x0 = r0, x0
        self.r2, self.x2 = r2, x2

# --------- Text parsing helpers (robust to '~' continuations) ----------
def _collect_blocks(full_text: str, header_regex: str):
    """
    Collect DSS object definitions that match header_regex on the first line,
    including any subsequent continuation lines beginning with '~'.
    """
    blocks = []
    for m in re.finditer(header_regex, full_text, flags=re.IGNORECASE | re.MULTILINE):
        start = m.start()
        cur = []
        for line in full_text[start:].splitlines():
            if not cur:
                cur.append(line)
            else:
                if line.lstrip().startswith("~"):
                    cur.append(line)
                else:
                    break
        blocks.append("\n".join(cur))
    return blocks

def get_prop(block: str, key: str):
    """
    Read a property value from a 'New ...' block.
    Returns the raw token including [] or quotes where applicable.
    """
    m = re.search(
        rf"(?i)\b{re.escape(key)}\s*=\s*"
        r"(\[[^\]]*\]|'[^']*'|\"[^\"]*\"|[^\s!]+)", block
    )
    return m.group(1).strip() if m else None

def strip_quotes_brackets(val: str) -> str:
    return (val or "").strip().strip("[]\"'")

def base_bus(bus_or_term: str) -> str:
    """Strip .node/.phase suffix from a bus reference."""
    return (bus_or_term or "").split(".")[0]

def canon(busname: str) -> str:
    """Canonical form for bus names: base bus, lower-case."""
    return base_bus(strip_quotes_brackets(busname)).lower()

def normalize_bus_token_with_phases(token: str) -> str:
    """
    Lower-case only the base bus name and preserve any .node/.phase suffix.
    Examples:
      'S182.1.2.3' -> 's182.1.2.3'
      's1.1'       -> 's1.1'
      's99'        -> 's99'
    """
    t = strip_quotes_brackets(token or "")
    m = re.match(r"^([^.\s]+)(.*)$", t)  # base + suffix (suffix may be empty)
    if not m:
        return t.lower()
    base, suffix = m.group(1), m.group(2)
    return base.lower() + suffix

# --------- Support definition resolver (transitive over LineGeometry deps) ----------
def resolve_support_definitions(full_text: str, needed_names):
    """
    needed_names: dict { 'LineGeometry': set(...), 'LineCode': set(...),
                         'LineSpacing': set(...), 'WireData': set(...) }
    Returns list of full 'New ...' text blocks for all referenced/required defs,
    including spacing/wiredatas found inside LineGeometry blocks.
    """
    support_blocks = []
    names = {
        "LineGeometry": set(needed_names.get("LineGeometry", set())),
        "LineCode": set(needed_names.get("LineCode", set())),
        "LineSpacing": set(needed_names.get("LineSpacing", set())),
        "WireData": set(needed_names.get("WireData", set())),
    }

    # 1) LineGeometry blocks + their deps
    for geom in sorted(names["LineGeometry"]):
        pat = rf"(?im)^\s*New\s+LineGeometry\.{re.escape(geom)}\b.*$"
        blocks = _collect_blocks(full_text, pat)
        support_blocks.extend(b.strip() for b in blocks)
        for b in blocks:
            spacing = strip_quotes_brackets(get_prop(b, "spacing") or "")
            if spacing:
                names["LineSpacing"].add(spacing)
            wires = get_prop(b, "wires")
            if wires:
                for w in re.findall(r"([\w.\-]+)", wires.strip("[]")):
                    names["WireData"].add(w)

    # 2) LineSpacing blocks
    for sp in sorted(names["LineSpacing"]):
        pat = rf"(?im)^\s*New\s+LineSpacing\.{re.escape(sp)}\b.*$"
        blocks = _collect_blocks(full_text, pat)
        support_blocks.extend(b.strip() for b in blocks)

    # 3) WireData blocks
    for wd in sorted(names["WireData"]):
        pat = rf"(?im)^\s*New\s+WireData\.{re.escape(wd)}\b.*$"
        blocks = _collect_blocks(full_text, pat)
        support_blocks.extend(b.strip() for b in blocks)

    # 4) LineCode blocks
    for lc in sorted(names["LineCode"]):
        pat = rf"(?im)^\s*New\s+LineCode\.{re.escape(lc)}\b.*$"
        blocks = _collect_blocks(full_text, pat)
        support_blocks.extend(b.strip() for b in blocks)

    # De-dupe preserving order
    seen = set()
    dedup = []
    for b in support_blocks:
        if b not in seen:
            dedup.append(b)
            seen.add(b)
    return dedup

# --------- Build LV (120/208 V) bus set by graph walk ----------
def build_lv_bus_set(full_text: str, boundary_buses, kv_threshold=0.24):
    """
    1) Extract ALL New Line.* blocks (we preserve parallels).
    2) Build an undirected graph on canonical bus names (lower-case).
    3) BFS from boundary_buses (canonical), but only traverse buses with kVBase ≤ kv_threshold.
       Uses actual bus names from the compiled circuit via a lower->actual mapping.
    4) Optionally, drop “spot network” buses by name pattern.
    """
    # Map canonical -> actual from the compiled circuit
    actual_buses = dss.Circuit.AllBusNames() or []
    canon_to_actual = {b.lower(): b for b in actual_buses}

    # All line blocks
    line_blocks_all = _collect_blocks(full_text, r"(?im)^\s*New\s+Line\.")
    # Build graph (canonical bus names)
    graph = defaultdict(set)
    for blk in line_blocks_all:
        b1 = canon(get_prop(blk, "bus1") or "")
        b2 = canon(get_prop(blk, "bus2") or "")
        if b1 and b2:
            graph[b1].add(b2)
            graph[b2].add(b1)

    # kVBase query helper (accepts canonical)
    _kv_cache = {}
    def is_lv_bus(canon_bus: str) -> bool:
        # Exclude by name (spot network heuristic)
        actual_name = canon_to_actual.get(canon_bus, canon_bus)
        if EXCLUDE_SPOT_BY_NAME and SPOT_NAME_REGEX.match(actual_name):
            return False
        if canon_bus not in _kv_cache:
            ok = dss.Circuit.SetActiveBus(actual_name)
            if not ok:
                _kv_cache[canon_bus] = float("inf")
            else:
                kvb = dss.Bus.kVBase()
                try:
                    _kv_cache[canon_bus] = float(kvb) if kvb is not None else float("inf")
                except Exception:
                    _kv_cache[canon_bus] = float("inf")
        return _kv_cache[canon_bus] <= kv_threshold

    # BFS from canonical boundary buses
    start = [canon(b) for b in boundary_buses if is_lv_bus(canon(b))]
    dq = deque(start)
    seen = set(start)
    while dq:
        u = dq.popleft()
        for v in graph.get(u, []):
            if v not in seen and is_lv_bus(v):
                seen.add(v)
                dq.append(v)
    return seen  # canonical bus names

# --------- Filter components by LV bus set ----------
def filter_secondary_components_by_bus(full_text: str, allowed_buses_canon):
    """
    Keep only components whose buses lie entirely inside allowed_buses_canon (canonical lower-case).
    - Lines: both bus1 and bus2 ∈ allowed (preserve parallels). **Preserve phase/node suffixes.**
    - Switches: subset of lines with switch=yes or name contains '_sw'.
    - Loads: bus/bus1 ∈ allowed. **Preserve phase/node suffixes.**
    Also accumulate supporting definitions referenced directly by lines (LineCode/LineGeometry)
    and expand the LineGeometry dependencies (spacing/wires) via resolve_support_definitions().
    """
    kept_lines = []
    kept_switches = []
    needed_names = {"LineCode": set(), "LineSpacing": set(), "WireData": set(), "LineGeometry": set()}

    # Lines (include everything; filter by bus set)
    all_line_blocks = _collect_blocks(full_text, r"(?im)^\s*New\s+Line\.")
    for blk in all_line_blocks:
        raw_b1 = get_prop(blk, "bus1") or ""
        raw_b2 = get_prop(blk, "bus2") or ""
        b1c = canon(raw_b1)
        b2c = canon(raw_b2)
        if not b1c or not b2c:
            continue
        if b1c in allowed_buses_canon and b2c in allowed_buses_canon:
            # --- CHANGED: preserve node/phase suffixes ---
            new_b1 = normalize_bus_token_with_phases(raw_b1)
            new_b2 = normalize_bus_token_with_phases(raw_b2)
            blk = re.sub(r"(?i)(\bbus1\s*=\s*)(\S+)", rf"\1{new_b1}", blk, count=1)
            blk = re.sub(r"(?i)(\bbus2\s*=\s*)(\S+)", rf"\1{new_b2}", blk, count=1)

            # Classification as switch or normal line
            is_switch = False
            sw_prop = get_prop(blk, "switch")
            if sw_prop and strip_quotes_brackets(sw_prop).lower() in {"yes", "true", "1"}:
                is_switch = True
            elif re.search(r"(?i)\bNew\s+Line\.[^\s]*_sw\b", blk):
                is_switch = True

            # Support defs referenced directly by this line
            geom = strip_quotes_brackets(get_prop(blk, "geometry") or "")
            code = strip_quotes_brackets(get_prop(blk, "linecode") or "")
            if geom:
                needed_names["LineGeometry"].add(geom)
            if code:
                needed_names["LineCode"].add(code)

            if is_switch:
                kept_switches.append(blk)
            else:
                kept_lines.append(blk)

    # Loads: keep if bus/bus1 in allowed set (by base bus), but preserve suffixes
    kept_loads = []
    all_load_blocks = _collect_blocks(full_text, r"(?im)^\s*New\s+Load\.")
    for blk in all_load_blocks:
        rb = get_prop(blk, "bus1") or get_prop(blk, "bus") or ""
        if canon(rb) in allowed_buses_canon:
            # --- CHANGED: preserve node/phase suffixes ---
            new_bus = normalize_bus_token_with_phases(rb)
            if re.search(r"(?i)\bbus1\s*=", blk):
                blk = re.sub(r"(?i)(\bbus1\s*=\s*)(\S+)", rf"\1{new_bus}", blk, count=1)
            elif re.search(r"(?i)\bbus\s*=", blk):
                blk = re.sub(r"(?i)(\bbus\s*=\s*)(\S+)", rf"\1{new_bus}", blk, count=1)
            kept_loads.append(blk)

    # Expand supporting definitions (LineGeometry + LineSpacing + WireData + LineCode)
    support_defs = resolve_support_definitions(full_text, needed_names)

    return {
        "lines": kept_lines,
        "switches": kept_switches,
        "loads": kept_loads,
        "support_defs": support_defs,
    }

# -------------------- Main Execution --------------------
print("--- Starting OpenDSS Secondary Network Extraction (OpenDSSDirect) ---")

# 1) Compile master and solve normal (snapshot) to cache pu/angle
print("1) Compiling & solving SNAPSHOT to cache nominal pu and angle...")
dss.Basic.ClearAll()
dss.Command(f"compile [{master_dss_file_path}]")
dss.Command("set mode=snapshot")
dss.Solution.Solve()

# 2) Identify boundary transformers whose low-side is ~0.208 kV (LL) and collect boundary buses
print("2) Scanning for 120/208 V boundary transformers...")
boundary = []   # list of dict(name, bus_low)
xf_idx = dss.Transformers.First()
while xf_idx:
    xf_name = dss.Transformers.Name()
    dss.Transformers.Wdg(2)  # low-side winding
    kv_ll = dss.Transformers.kV()
    if kv_ll is not None and abs(kv_ll - 0.208) < 0.02:
        buses = dss.CktElement.BusNames() or []
        if len(buses) >= 2:
            bus2 = base_bus(buses[1])  # keep original case for querying
            boundary.append({"name": xf_name, "bus": bus2})
    xf_idx = dss.Transformers.Next()
print(f"   Found {len(boundary)} candidate boundary transformers.")

# 3) Cache pre-fault pu magnitude and angle at those buses
print("3) Caching pre-fault pu and angle at boundary buses...")
prefault = {}  # bus (original case) -> (pu_mag, angle_deg)
for b in boundary:
    dss.Circuit.SetActiveBus(b["bus"])
    pu_va = dss.Bus.puVmagAngle()  # [mag1, ang1, mag2, ang2, ...]
    if pu_va and len(pu_va) >= 2:
        prefault[b["bus"]] = (float(pu_va[0]), float(pu_va[1]))
    else:
        prefault[b["bus"]] = (1.0, 0.0)

# 4) Switch to FaultStudy to compute sequence impedances at boundary buses
print("4) Running FAULTSTUDY and refreshing Zsc...")
dss.Command("set mode=faultstudy")
dss.Solution.Solve()

# Refresh Zsc cache from current system Y
for b in boundary:
    dss.Circuit.SetActiveBus(b["bus"])
    dss.Bus.ZscRefresh()

# 5) Build TheveninSource objects using sequence impedances
print("5) Building Thevenin sources from Zsc (seq impedances) + pre-fault setpoints...")
vsource_data = []
for b in boundary:
    bus = b["bus"]
    pu_mag, ang_deg = prefault[bus]
    dss.Circuit.SetActiveBus(bus)

    # Positive- and zero-sequence Thevenin at this bus (Ohms). API returns [R, X].
    z1 = dss.Bus.Zsc1() or [0.0, 0.0]
    z0 = dss.Bus.Zsc0() or [0.0, 0.0]
    r1, x1 = float(z1[0]), float(z1[1])
    r0, x0 = float(z0[0]), float(z0[1])

    # Negative-sequence: use R2/X2 = R1/X1 unless Z2 explicitly available
    r2, x2 = r1, x1

    vsource_data.append(
        TheveninSource(
            name=f"Veq_{b['name']}",
            bus=canon(bus), kv_ll=0.208,
            pu=pu_mag, angle_deg=ang_deg,
            r1=r1, x1=x1, r0=r0, x0=x0, r2=r2, x2=x2,
        )
    )
print(f"   Built {len(vsource_data)} Thevenin sources.")

# 6) Parse full text and build LV bus set to filter components
print("6) Building the 120/208-V bus set and filtering components...")
with open(master_dss_file_path, "r", encoding="utf-8", errors="ignore") as f:
    full_text = f.read()

boundary_buses = [b["bus"] for b in boundary]
allowed_canon = build_lv_bus_set(full_text, boundary_buses, kv_threshold=0.24)
print(f"   LV bus count (after BFS & filters): {len(allowed_canon)}")

components = filter_secondary_components_by_bus(full_text, allowed_canon)
print(f"   Kept lines: {len(components['lines'])} (incl. parallels), "
      f"switches: {len(components['switches'])}, loads: {len(components['loads'])}, "
      f"support defs: {len(components['support_defs'])}")

# 7) Write the new DSS file
print(f"7) Writing new equivalent DSS file: {output_dss_file}")
with open(output_dss_file, "w", encoding="utf-8") as f:
    f.write(f"! Equivalent Secondary Network (120/208V) from {os.path.basename(master_dss_file_path)}\n")
    f.write("Clear\nSet DefaultBaseFrequency=60\n\n")

    # Use the first boundary bus as the slack bus
    if not vsource_data:
        raise RuntimeError("No boundary transformers found at 0.208 kV; cannot build sources.")

    main = vsource_data[0]
    f.write("! ------ CIRCUIT & SLACK SOURCE ------\n")
    f.write(f"New Circuit.SecNet bus1={main.bus} basekv={main.kv_ll:.3f} phases=3\n")
    f.write(
        "Edit Vsource.Source "
        f"bus1={main.bus} basekv={main.kv_ll:.3f} phases=3 "
        f"pu={main.pu:.6f} angle={main.angle_deg:.6f} "
        f"R1={main.r1:.8f} X1={main.x1:.8f} R0={main.r0:.8f} X0={main.x0:.8f}\n\n"
    )

    # Additional boundary sources
    if len(vsource_data) > 1:
        f.write("! ------ ADDITIONAL THEVENIN SOURCES ------\n")
        for src in vsource_data[1:]:
            f.write(
                f"New Vsource.{src.name} "
                f"bus1={src.bus} basekv={src.kv_ll:.3f} phases=3 "
                f"pu={src.pu:.6f} angle={src.angle_deg:.6f} "
                f"R1={src.r1:.8f} X1={src.x1:.8f} R0={src.r0:.8f} X0={src.x0:.8f}\n"
            )
        f.write("\n")

    # Supporting definitions before elements that reference them
    f.write("! ------ SUPPORTING DEFINITIONS ------\n")
    for defn in components["support_defs"]:
        f.write(defn + "\n")
    f.write("\n")

    # Secondary network pieces
    f.write("! ------ SECONDARY LINES (INCLUDING PARALLELS) ------\n")
    for line in components["lines"]:
        f.write(line + "\n")

    f.write("\n! ------ SECONDARY SWITCHES ------\n")
    for sw in components["switches"]:
        f.write(sw + "\n")

    f.write("\n! ------ SECONDARY LOADS ------\n")
    for ld in components["loads"]:
        f.write(ld + "\n")

    # Final setup and solve
    f.write("\n! ------ SOLVER COMMANDS ------\n")
    f.write("Set VoltageBases=[0.208]\nCalcV\nset mode=snapshot\nsolve\n")

print("\n--- Success ---")
print(f"Created: {output_dss_file}")
