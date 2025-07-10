import re, shutil, pathlib
from typing import List
import numpy as np
from scipy.sparse import csc_matrix

def make_midspan_hif_dss(
        pristine_path : str,
        new_path      : str,
        line_name     : str = "line.2",
        split_ratio   : float = 0.5,
        fault_bus     : str = "FaultBus",
        hif_load_name : str = "Load.HIF_Load",
        encoding      : str = "utf-8"        # change if you want cp1252, …
    ) -> None:
    """
    Creates *new_path* by copying *pristine_path* and injecting a block that
    1.  disables <line_name>,
    2.  creates 2 half‑length replacements,
    3.  adds a 1‑phase HIF load tied to the new mid‑span bus.

    The block is inserted **before** the first  `set voltagebases`  or, if that
    tag is absent, before the first  `solve`  so that the original `solve`
    operates on the modified circuit.
    """
    src = pathlib.Path(pristine_path).resolve()
    dst = pathlib.Path(new_path).resolve()
    if src == dst:
        raise ValueError("new_path must differ from pristine_path")

    # 0) read pristine text (we keep it in memory so we can re‑write later)
    with src.open("r", encoding=encoding, errors="replace") as f:
        lines = f.readlines()

    # 1) capture the original line definition so we can clone its length/unit
    patt = re.compile(rf"^\s*new\s+{re.escape(line_name)}\b.+$", re.I)
    for ln in lines:
        if patt.match(ln):
            tokens = ln.split()
            toks   = {kv.split("=", 1)[0].lower():kv.split("=", 1)[1]
                      for kv in tokens if "=" in kv}
            length = float(toks["length"])
            units  = toks.get("units", "ft")
            geom   = toks.get("geometry")
            lcode  = toks.get("linecode")
            bus1, bus2 = toks["bus1"], toks["bus2"]
            break
    else:
        raise ValueError(f"Line '{line_name}' not found in {pristine_path}")

    len_a = split_ratio * length
    len_b = length - len_a

    # 2) build the ASCII, Windows‑friendly block (no fancy Unicode)
    block = []
    block.append("")
    block.append("! ------------- Mid‑span HIF auto‑generated block -----------")
    block.append(f"Edit {line_name} enabled=no")
    base = f"length={{}} units={units} "
    if geom:  base += f"geometry={geom} "
    if lcode: base += f"linecode={lcode} "
    block.append(f"New {line_name}_a bus1={bus1} bus2={fault_bus} " +
                 base.format(f'{len_a:.4f}'))
    block.append(f"New {line_name}_b bus1={fault_bus} bus2={bus2} " +
                 base.format(f'{len_b:.4f}'))
    block.append(f"New {hif_load_name} Bus1={fault_bus}.1 Phases=1 "
                 "Conn=Wye Model=1 kW=0 kvar=0")
    block.append("! -----------------------------------------------------------")
    block.append("")

    # 3) insert block before first 'set voltagebases'  or  'solve'
    insert_at = None
    for idx, ln in enumerate(lines):
        low = ln.lower()
        if "set voltagebases" in low or re.match(r"\s*solve\b", low):
            insert_at = idx
            break
    if insert_at is None:
        insert_at = len(lines)

    new_lines = lines[:insert_at] + [b + "\n" for b in block] + lines[insert_at:]

    # 4) write to destination (explicit encoding = ASCII‑safe or UTF‑8)
    with dst.open("w", encoding=encoding, errors="replace") as f:
        f.writelines(new_lines)

def kron_reduce(Y: csc_matrix, eliminate: List[int]) -> csc_matrix:
    keep = np.setdiff1d(np.arange(Y.shape[0]), eliminate)
    Yaa  = Y[keep, :][:, keep].toarray()
    Yab  = Y[keep, :][:, eliminate].toarray()
    Yba  = Y[eliminate, :][:, keep].toarray()
    Ybb  = Y[eliminate, :][:, eliminate].toarray()
    return csc_matrix(Yaa - Yab @ np.linalg.inv(Ybb) @ Yba)
