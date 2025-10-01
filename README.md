# Three-phase Normalized Lagrangian Multiplier State Estimation

Mahalanobis-distance based grouped indices for three-phase, unbalanced distribution system state estimation. The method uses normalized Lagrangian multipliers (NLMs) to quantify constraint sensitivity and ranks likely parameter/location errors at scale.

The estimator and the three-phase power-flow routines interface with OpenDSS via OpenDSSDirect.py, enabling end-to-end studies on included primary and secondary feeders.

## Highlights

- Three-phase DSSE with polar states and zero-injection constraints
- Normalized Lagrangian multipliers (NLM) for parameter and constraint saliency
- Grouped Mahalanobis distance per line: line parameters plus both terminal shunts
- OpenDSS-backed parsing and power flow; automatic switch merging for near-zero-impedance switches
- Optional GPU acceleration via CuPy with CPU fallback
- Built-in scenarios: 4-bus demo, primary feeder runs, high-impedance fault (HIF), cable burnout, open-main

## Repository Structure

```
.
├── data/                  # DSS models, line constants, coordinates
├── docs/                  # Notes and experiment summaries
├── notebooks/             # Optional interactive exploration
├── results/               # Output figures and generated DSS variants
├── src/                   # Estimator, PF, parsers, scenarios, and utilities
│   ├── cal_pf.py          # Three‑phase Newton power‑flow
│   ├── case_4.py          # 4‑bus demo incl. Kron reduction checks
│   ├── case_390.py        # Primary feeder example (Master.dss)
│   ├── case390_burnout.py # Cable burnout scenario
│   ├── case390_open_main.py # Open‑main scenario
│   ├── case390_HIF.py     # One‑off HIF scenario
│   ├── run_batched_HIF.py # Batch HIF detection across families (group index)
│   ├── run_batched_test.py      # Representative burnout tests over groups
│   ├── run_two_cables_batch_test.py # Two‑cable simultaneous burnout tests
│   ├── lagrangian_m.py    # DSSE + NLM and grouped χ² indices
│   ├── wls_estimation_p.py# WLS state estimator (polar)
│   ├── utilities/         # Jacobians, measurement model, reporting helpers
│   └── vis/               # Network visualization utilities
└── README.md
```

## Requirements

- Python 3.9+
- Packages (install via pip):
  - Core: `numpy`, `scipy`, `opendssdirect`
  - Visualization: `matplotlib`, `networkx` (optional for `src/vis/visualization.py`)
  - GPU (optional): `cupy` wheels matched to your CUDA, e.g. `cupy-cuda12x` or `cupy-cuda11x`
  - Graphviz (optional): for nicer layouts in visualization; falls back to spring layout if missing

Quick install example (via requirements file):

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Quick install example (CPU-only, without requirements file):

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy scipy opendssdirect matplotlib networkx
```

Optional GPU acceleration with CuPy (choose the right wheel for your CUDA):

```bash
pip install cupy-cuda12x  # or cupy-cuda11x, or plain cupy for ROCm/CPU
```

Note: CuPy must be installed for the current code to import. When GPU factorization is unavailable at runtime, the solver falls back to SciPy CPU factorization, but the CuPy import is still required.

## Mathematical Notes

State estimation is posed as the weighted least-squares problem with equality constraints:

J(x) = (z − h(x))ᵀ W (z − h(x))  s.t.  g(x) = 0

with Lagrangian L(x, λ) = J(x) + λᵀ g(x). The normalized multiplier for constraint i is

λ̃ᵢ = λᵢ / sqrt(Var(λᵢ)).

For a set of related constraints/parameters g, we compute a group index via a Mahalanobis distance:

D_g = sqrt( λ̃_gᵀ Σ_g⁻¹ λ̃_g ).

In code, line-group indices combine a line’s impedance parameters with shunt admittances at both terminals, improving localization robustness.

Measurement model (per-node ordering): h(x) = [P_inj, Q_inj, |V|] with length 3·N; see `src/utilities/mea_fun.py:27`.

## Quick Start

From the repository root, run any scenario. Paths are resolved through the central config.

```bash
python src/case_4.py           # 4‑bus demo + Kron reduction checks
python src/case_390.py         # Primary feeder baseline
python src/case390_HIF.py      # One‑off HIF on primary/secondary
python src/case390_burnout.py  # Burnout scenario on primary
python src/case390_open_main.py# Open‑main scenario on primary
```

Each script runs a three‑phase power flow, builds the measurement vector, performs DSSE, and prints NLM/χ²‑group summaries.

## Batch Experiments

- Batch HIF detection across secondary families using grouped indices:

  ```bash
  python src/run_batched_HIF.py
  ```

  Outputs a summary to stdout and writes CSV results to `src/hif_batch_tmp/hif_batch_results_group.csv`.

- Representative cable burnout across parallel line groups:

  ```bash
  python src/run_batched_test.py
  ```

  Prints per‑group outcomes and a final success‑rate report.

- Two‑cable simultaneous burnout stress test:

  ```bash
  python src/run_two_cables_batch_test.py
  ```

Experiment notes and detection-rate snapshots are in `docs/` (e.g., `docs/New Text Document (6).md`).

## Paths and Config

- Centralized path configuration: `src/config.py:1`
  - `DATA_DIR` → `data/`
  - `RESULTS_DIR` → `results/`
  - Helpers: `data_file(name)`, `result_file(name)`
- Example: `os.fspath(DATA_DIR / "Master.dss")` converts a `pathlib.Path` to a string for OpenDSS.
- If your DSS models live elsewhere, edit `src/config.py` and the rest of the code will follow.

## Visualization

Render and save a simple network overview from the OpenDSS model:

```bash
python src/vis/visualization.py
```

This compiles `data/Master.dss`, builds a graph (buses + lines/transformers), colors buses by voltage category (HV/MV/SN/LV), highlights bridging nodes, and saves `results/vis_overview.png`.

## Tips and Troubleshooting

- Switch merging threshold: adjust `switch_threshold` in scripts that call `merge_closed_switches_in_mpc_and_dss(...)` to reflect your model’s near‑zero‑ohm switches.
- GPU optional: CuPy acceleration is used when available; otherwise, a SciPy CPU solve is used automatically.
- Measurement noise: batch/HIF scripts expose `STD_P`, `STD_Q`, and `STD_V` at the top for quick sensitivity studies.
- Graphviz optional: visualization falls back to a spring layout if pygraphviz/graphviz is not installed.
- Reproducibility: batch HIF experiments set a fixed RNG seed; see `src/run_batched_HIF.py:31`.

## Acknowledgments

This work builds on OpenDSS and the dss-extensions OpenDSSDirect.py interface. If you use this repository in academic work, please cite your usage accordingly and reference OpenDSS.

## How to Contribute

- Open an issue to discuss bugs, ideas, or feature requests.
- Fork the repo and create a branch for your change (`feat/...`, `fix/...`).
- Keep changes focused; follow existing code style and structure.
- Test your scenario locally (e.g., `case_4.py`, `case_390.py`, batch scripts) and include a short summary in the PR description.
- Submit a Pull Request referencing the related issue when applicable.

## License

MIT License. You are free to use, modify, and distribute this software with attribution and without warranty.
