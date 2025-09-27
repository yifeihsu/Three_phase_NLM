# Three-phase Normalized Lagrangian Multiplier State Estimation

This repository implements the Mahalanobis-distance based grouped index for
three-phase, unbalanced distribution system state estimation. The methodology
relies on **normalized Lagrangian multipliers (NLM)** to quantify constraint
sensitivity, enabling efficient detection of abnormal conditions in large-scale
networks.

The codebase couples the state estimator and three-phase power-flow solver with
[OpenDSS](https://github.com/dss-extensions/OpenDSSDirect.py) via the
`opendssdirect.py` interface, allowing validation on both the IEEE 4-node feeder
and the low-voltage IEEE 342-node test system.

## Repository structure

```
.
├── data/                  # Measurement data and configuration files
├── docs/                  # Project documentation and supplementary material
├── notebooks/             # Interactive exploration and validation notebooks
├── results/               # Saved experiment results
├── src/                   # Source code for estimation and diagnostics
│   ├── cal_pf.py          # Three-phase power-flow calculation
│   ├── case_4.py          # IEEE 4-node validation example
│   ├── case_390.py        # IEEE 342-node low-voltage study
│   ├── lagrangian_m.py    # Normalized Lagrangian multiplier utilities
│   ├── wls_estimation_p.py# Weighted least-squares state estimator
│   ├── utilities/         # Helper functions for OpenDSS interaction
│   └── vis/               # Plotting utilities and figure scripts
└── README.md              # Project overview (this file)
```

## Mathematical formulation

State estimation is posed as the weighted least-squares problem

\[
J(x) = (z - h(x))^{\mathsf{T}} W (z - h(x))
\]

subject to a set of equality constraints $g(x) = 0$. The associated Lagrangian
is

\[
\mathcal{L}(x, \lambda) = J(x) + \lambda^{\mathsf{T}} g(x)
\]

The sensitivity of each constraint is evaluated through normalized Lagrangian
multipliers

\[
\tilde{\lambda}_i = \frac{\lambda_i}{\sqrt{\mathrm{Var}(\lambda_i)}}
\]

For a group of related constraints, the Mahalanobis-distance based index is
computed as

\[
D_g = \sqrt{\tilde{\boldsymbol{\lambda}}_g^{\mathsf{T}} \Sigma_g^{-1}
\tilde{\boldsymbol{\lambda}}_g }
\]

where $\tilde{\boldsymbol{\lambda}}_g$ collects the normalized multipliers for
group $g$ and $\Sigma_g$ is their covariance matrix. High values of $D_g$
indicate anomalous behavior in the corresponding parameter set.

## Getting started

1. Install dependencies (Python 3.9+, `opendssdirect.py`).
2. Run an example study from the `src` directory:
   ```bash
   python case_4.py   # IEEE 4-node test system
   python case_390.py # IEEE 342-node low-voltage network
   ```

These scripts perform a three-phase power flow, execute the state estimator, and
compute grouped indices for constraint monitoring.

