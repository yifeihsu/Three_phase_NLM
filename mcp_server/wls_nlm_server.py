# wls_nlm_server_fastmcp.py
from __future__ import annotations

import os
import sys
import io
import json
import contextlib
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# ---------------------
# Path setup (same as your original)
# ---------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if os.fspath(SRC_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(SRC_DIR))

# ---------------------
# MCP server
# ---------------------
mcp = FastMCP(name="WLS-NLM", mask_error_details=True)

# ---------------------
# Helpers
# ---------------------
def _import_runner():
    # Lazy import: solver deps might include GPU libs; import at call-time.
    from lagrangian_m import run_lagrangian_polar  # type: ignore
    return run_lagrangian_polar

def _to_numpy_array(obj: Any, dtype: Optional[Any] = None) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj.astype(dtype) if dtype is not None else obj
    return np.array(obj, dtype=dtype)

def _load_mpc(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a JSON-serializable MPC dict into the numpy-backed structure.
    Arrays are accepted as lists and converted to np.array.
    Unknown keys are passed through untouched.
    """
    mpc = dict(obj)
    for key in ("bus3p", "line3p", "load3p", "gen3p", "lc", "xfmr3p"):
        if key in mpc and mpc[key] is not None:
            if key == "line3p":
                mpc[key] = np.array(mpc[key], dtype=object)
            else:
                mpc[key] = np.array(mpc[key], dtype=float)
    return mpc

def _assemble_R(R: Optional[List[List[float]]], R_diag: Optional[List[float]]) -> np.ndarray:
    if R is not None:
        return _to_numpy_array(R, dtype=float)
    if R_diag is not None:
        d = _to_numpy_array(R_diag, dtype=float)
        return np.diag(d)
    raise ValueError("One of R (full) or R_diag must be provided")

# ---------------------
# Result schemas (dataclasses, JSON-friendly)
# ---------------------
@dataclass
class TopGroup:
    line_index: int
    score: Optional[float]

@dataclass
class GroupScore:
    line_index: int
    score: Optional[float]

@dataclass
class NlmResult:
    success: bool
    n_state: int
    n_lambda: int
    max_abs_lambdaN: Optional[float]
    top_groups: List[TopGroup]
    # Optional payloads:
    x_est: Optional[List[float]] = None
    lambdaN: Optional[List[float]] = None
    group_scores: Optional[List[GroupScore]] = None

# ---------------------
# Tool
# ---------------------
@mcp.tool
def run_wls_nlm_from_dss(
    dss_master_path: str,
    lc_filename: Optional[str] = None,
    baseMVA: float = 1.0,
    slack_bus: str = "p1",
    # Measurements: either provide full z, or provide component dicts keyed by "bus.phase"
    z: Optional[List[float]] = None,
    P_by_busphase: Optional[Dict[str, float]] = None,
    Q_by_busphase: Optional[Dict[str, float]] = None,
    V_by_busphase: Optional[Dict[str, float]] = None,
    # Covariance: full or diagonal
    R: Optional[List[List[float]]] = None,
    R_diag: Optional[List[float]] = None,
    # Algorithm options
    max_iter: int = 20,
    tol: float = 1e-4,
    zi_enable: bool = False,
    zi_tol_kw: float = 1.0,
    alpha: float = 1e-8,
    gpu_permc: str = "COLAMD",
    gpu_diag_pivot_thresh: float = 1.0,
    debug: bool = False,
    return_full: bool = False,
) -> NlmResult:
    """Parse an OpenDSS model, assemble measurements & covariance, and run WLS + NLM."""
    try:
        import opendssdirect as dss  # noqa: F401
    except Exception as e:
        raise ToolError("opendssdirect is required to parse .dss files. pip install opendssdirect") from e

    try:
        # Utility imports from your src/
        from parse_opendss_file import (
            parse_opendss_to_mpc,
            merge_closed_switches_in_mpc_and_dss,
            build_global_y_per_unit,
        )
        from cal_pf import run_newton_powerflow_3p
    except Exception as e:
        raise ToolError(f"Failed to import project utilities from {SRC_DIR}: {e}") from e

    if lc_filename is None:
        lc_filename = os.fspath(REPO_ROOT / "data" / "LineConstantsCode.txt")

    try:
        # 1) Parse DSS ➜ mpc and merge zero-impedance switches
        mpc = parse_opendss_to_mpc(
            dss_master_path, baseMVA=baseMVA, lc_filename=lc_filename, slack_bus=slack_bus
        )
        merge_closed_switches_in_mpc_and_dss(mpc, switch_threshold=2)

        # 2) Build Ybus (per-unit) and busphase_map
        Ybus, _ = build_global_y_per_unit(mpc)
        pf_buf = io.StringIO()
        with contextlib.redirect_stdout(pf_buf):
            _, _, busphase_map = run_newton_powerflow_3p(mpc, tol=1e-6, max_iter=20)
        # (optional) PF logs not returned; stdout is suppressed to keep MCP stdio clean.

        nnode = len(busphase_map)

        # 3) Assemble measurement vector z
        if z is not None:
            z_arr = _to_numpy_array(z, dtype=float)
            if len(z_arr) != 3 * nnode:
                raise ToolError(f"z length must be 3*N ({3*nnode}), got {len(z_arr)}")
        else:
            # Build P/Q/V from dicts keyed as "bus.phase" with phase in {1,2,3} or {A,B,C}
            def parse_phase_token(tok: str) -> int:
                t = tok.strip().upper()
                if t in ("1", "A"): return 0
                if t in ("2", "B"): return 1
                if t in ("3", "C"): return 2
                raise ToolError(f"Unrecognized phase token '{tok}' (use 1|2|3 or A|B|C)")

            idx_map = dict(busphase_map)  # (bus_id, phase_idx) -> node_idx
            busname_to_id = mpc.get("busname_to_id", {})

            def collect(component: Optional[Dict[str, float]]) -> np.ndarray:
                vec = np.zeros(nnode, dtype=float)
                if not component:
                    return vec
                for key, val in component.items():
                    if not isinstance(key, str) or "." not in key:
                        raise ToolError("Measurement keys must be 'bus.phase' strings (e.g., 'p2.1')")
                    bus_tok, ph_tok = key.split(".", 1)
                    try:
                        bus_id = int(bus_tok)
                    except ValueError:
                        bus_id = int(busname_to_id.get(bus_tok.lower(), 0))
                        if bus_id <= 0:
                            raise ToolError(f"Unknown bus name '{bus_tok}' in measurement key '{key}'")
                    ph_idx = parse_phase_token(ph_tok)
                    node_idx = idx_map.get((bus_id, ph_idx))
                    if node_idx is None:
                        raise ToolError(f"No node index for bus {bus_id} phase {ph_idx} (key '{key}')")
                    vec[node_idx] = float(val)
                return vec

            P = collect(P_by_busphase)
            Q = collect(Q_by_busphase)
            V = collect(V_by_busphase)
            z_arr = np.concatenate([P, Q, V])

        # 4) Covariance
        R_arr = _assemble_R(R, R_diag)
        if R_arr.shape != (len(z_arr), len(z_arr)):
            raise ToolError(f"R shape must be {(len(z_arr), len(z_arr))}, got {R_arr.shape}")

        # 5) Run solver
        run = _import_runner()
        solver_buf = io.StringIO()
        with contextlib.redirect_stdout(solver_buf):
            x_est, success, lambdaN, sorted_groups = run(
                z_arr, None, busphase_map, Ybus, R_arr, mpc,
                max_iter=max_iter, tol=tol,
                zi_enable=zi_enable, zi_tol_kw=zi_tol_kw,
                alpha=alpha,
                gpu_permc=gpu_permc, gpu_diag_pivot_thresh=gpu_diag_pivot_thresh,
                debug=debug,
            )

        # Package result with JSON-friendly shapes
        top_groups: List[TopGroup] = [
            TopGroup(line_index=int(i), score=(None if v is None else float(v)))
            for (i, v) in (sorted_groups[:5] if sorted_groups else [])
        ]

        result = NlmResult(
            success=bool(success),
            n_state=int(len(x_est)) if x_est is not None else 0,
            n_lambda=int(len(lambdaN)) if (lambdaN is not None) else 0,
            max_abs_lambdaN=(
                float(np.max(np.abs(lambdaN))) if (lambdaN is not None and len(lambdaN) > 0) else None
            ),
            top_groups=top_groups,
        )

        if return_full:
            result.x_est = (None if x_est is None else _to_numpy_array(x_est, dtype=float).tolist())
            result.lambdaN = (None if lambdaN is None else _to_numpy_array(lambdaN, dtype=float).tolist())
            result.group_scores = [
                GroupScore(line_index=int(i), score=(None if v is None else float(v)))
                for (i, v) in (sorted_groups or [])
            ]

        return result

    except ToolError:
        raise
    except Exception as e:
        # Surface a concise error message to the client
        raise ToolError(f"run_wls_nlm_from_dss failed: {e}")

# ---------------------
# Entrypoint
# ---------------------
if __name__ == "__main__":
    mcp.run()
