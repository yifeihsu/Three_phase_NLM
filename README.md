# Three Phase NLM

This repository contains research scripts for three‑phase power flow analysis and state estimation based on the Newton–Lagrangian Method (NLM). The code primarily operates on OpenDSS feeder models and includes utilities for building a per‑unit admittance matrix, generating measurement data and locating parameter errors.

## Repository layout

- `*.dss` files – sample OpenDSS networks such as the 4‑Bus and IEEE 390‑bus feeders.
- `parse_opendss_file.py` – converts OpenDSS cases to a MATPOWER‑style data structure and builds the Y‑bus.
- `cal_pf.py` – Newton three‑phase power flow solver.
- `lagrangian_m.py`, `lagrangian_m_gpu.py` – NLM based state estimators (CPU and optional GPU version).
- `wls_estimation_p.py` – weighted least squares state estimator for comparison.
- Example scripts (`case4_correction.py`, `case390_correction.py`, `case390_burnout.py`, etc.) demonstrate measurement generation, parameter error detection and correction.

## Requirements

Python 3 with the following packages:

```
numpy
scipy
opendssdirect.py
matplotlib (for plotting scripts)
```

`cupy` can be installed to accelerate the GPU version of the estimator but is optional.

## Quick start

1. Install the dependencies, for example using pip:

```bash
pip install numpy scipy opendssdirect.py matplotlib
```

2. Run a small test case:

```bash
python case4_correction.py
```

This loads the 4‑bus system, generates noisy measurements and applies the Lagrangian estimator to identify incorrect line parameters.

For larger networks, `case390_correction.py` operates on the IEEE 390‑bus feeder and will take longer to run.

**Note:** when switching between different OpenDSS cases, update the slack bus identifier accordingly. Failing to do so may lead to inconsistent results.

## Output

The scripts print diagnostic information to the console. Some cases save measurement data in `.npz` files. Voltage plots and subgraphs produced by the visualization utilities can be found under the `plots` directory.

## License

No explicit license has been provided. Use at your own discretion.
