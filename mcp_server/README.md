WLS+NLM MCP Server (local)

Overview

- Exposes `src/lagrangian_m.py:run_lagrangian_polar` via a local MCP stdio server.
- Server name: `wls-nlm` with a single tool:
  - `run_wls_nlm_from_dss`: ingest `.dss` + named measurements; builds Ybus internally.
- Returns compact summaries by default; full arrays optional via `return_full`.

Requirements

- Python 3.9+
- Packages: `mcp`, `numpy`, `scipy`, `opendssdirect`, and a CuPy build matching your CUDA.
  - Install: `pip install mcp numpy scipy opendssdirect`
  - For CuPy: choose a build matching your CUDA (e.g. `pip install cupy-cuda12x`).

Run the server

- From repo root:
  - `python mcp_server/wls_nlm_server.py`

Register with an MCP client

- Example Claude Desktop `claude_desktop_config.json` snippet:

  {
    "mcpServers": {
      "wls-nlm": {
        "command": "python",
        "args": ["mcp_server/wls_nlm_server.py"],
        "transport": "stdio"
      }
    }
  }

Tool

- run_wls_nlm_from_dss
  - `dss_master_path` (string): Path to OpenDSS master/circuit.
  - `lc_filename` (string, optional): Path to LineConstantsCode.txt. Defaults to `data/LineConstantsCode.txt`.
  - `baseMVA` (number, default 1.0), `slack_bus` (string, default `p1`).
  - Provide either:
    - `z` (number[]) of length 3*N in order `[P(0..N-1), Q(0..N-1), |V|(0..N-1)]`, or
    - Named components (recommended):
      - `P_by_busphase`, `Q_by_busphase`, `V_by_busphase` (objects). Keys are `"bus.phase"` strings where phase is `1|2|3` or `A|B|C`. Example: `{ "p2.1": 0.001, "p3.B": 0.0 }`.
- Covariance: `R` (matrix) or `R_diag` (diagonal list).
  - Length/order must match `z`: first N are P, next N are Q, last N are |V|.
  - Options: `max_iter`, `tol`, `zi_enable`, `zi_tol_kw`, `alpha`, `gpu_permc`, `gpu_diag_pivot_thresh`, `debug`.
  - `return_full` (bool): Include `x_est`, `lambdaN`, and all group scores.


Output

- `success` (bool)
- `n_state` (int)
- `n_lambda` (int)
- `max_abs_lambdaN` (number | null)
- `top_groups` ({line_index:number, score:number|null}[])
- If `return_full=true`: `x_est` (number[]), `lambdaN` (number[]), `group_scores` ([line_index, score][])

Notes

- The solver uses CuPy for multiplier/covariance math; ensure a working CUDA+CuPy setup. The linear solve may fall back to CPU, but CuPy is still used.
- `run_wls_nlm_from_dss` parses and merges zero-impedance switches and builds Ybus internally using `parse_opendss_file.py` utilities.

Examples

- From an MCP client, call `run_wls_nlm_from_dss` with named measurements:

  {
    "tool": "run_wls_nlm_from_dss",
    "arguments": {
      "dss_master_path": "/abs/path/to/Master.dss",
      "lc_filename": "/abs/path/to/LineConstantsCode.txt",
      "slack_bus": "p1",
      "P_by_busphase": { "p2.1": 0.001, "p2.2": 0.000, "p2.3": -0.002 },
      "Q_by_busphase": { "p2.1": 0.000, "p2.2": 0.000, "p2.3": 0.000 },
      "V_by_busphase": { "p2.1": 1.00,  "p2.2": 1.00,  "p2.3": 1.00  },
      "R_diag": [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, ...],
      "max_iter": 20,
      "tol": 1e-4,
      "return_full": false
    }
  }

- If you have `z` already assembled, pass it directly and omit the P/Q/V objects.

Covariance tips

- If you only have P/Q (no |V|), you can still call the tool:
  - Provide `P_by_busphase` and `Q_by_busphase` and omit `V_by_busphase`.
  - Build `R_diag` with very large variances for the |V| block (last N entries), e.g., `1e6`, to effectively ignore |V| rows.
