# Three-phase Normalized Lagrangian Multiplier State Estimation

This repository implements the Mahalanobis-distance based grouped index for
three-phase, unbalanced distribution system state estimation.  The methodology
relies on **normalized Lagrangian multipliers (NLM)** to quantify the sensitivity
of parameter constraints, enabling efficient detection of abnormal conditions in
large-scale networks.

The codebase couples the state estimator and three-phase power-flow solver with
[OpenDSS](https://github.com/dss-extensions/OpenDSSDirect.py) through
`opendssdirect.py`, allowing validation on both the IEEE 4-Node feeder and the
low-voltage IEEE 342-Node test system.

## Repository structure

```
.
├── Source/                 # Core algorithm components
├── utilities/              # Helper functions for OpenDSS interaction
├── plots/                  # Plotting scripts and generated figures
├── case_4.py               # Example: IEEE 4-Node validation
├── case_390.py             # Example: large low-voltage feeder study
├── cal_pf.py               # Three-phase power-flow calculation
├── lagrangian_m.py         # Normalized Lagrangian multiplier utilities
├── wls_estimation_p.py     # Weighted least-squares state estimator
├── *.dss                   # OpenDSS circuit descriptions
└── README.md               # Project documentation
```

## Mathematical formulation

State estimation is posed as the weighted least-squares problem

\[
J(x) = (z - h(x))^T W (z - h(x))
\]

subject to a set of equality constraints \(g(x) = 0\).  The associated
Lagrangian is

\[
\mathcal{L}(x,\lambda) = J(x) + \lambda^T g(x)
\]

The sensitivity of each constraint is evaluated through normalized Lagrangian
multipliers

\[
\tilde{\lambda}_i = \frac{\lambda_i}{\sqrt{\operatorname{Var}(\lambda_i)}}
\]

For a group of related constraints, the Mahalanobis-distance based index is
computed as

\[
D_g = \sqrt{\tilde{\boldsymbol{\lambda}}_g^{\mathsf{T}}\, \Sigma_g^{-1}
\tilde{\boldsymbol{\lambda}}_g }
\]

where \(\tilde{\boldsymbol{\lambda}}_g\) collects the normalized multipliers for
group \(g\) and \(\Sigma_g\) is their covariance matrix.  High values of \(D_g\)
indicate anomalous behavior in the corresponding parameter set.

## Getting started

1. Install dependencies (Python 3.9+, `opendssdirect.py`).
2. Run an example study:
   ```bash
   python case_4.py   # IEEE 4-Node test system
   python case_390.py # IEEE 342-Node low-voltage network
   ```

These scripts perform a three-phase power flow, execute the state estimator, and
compute grouped indices for constraint monitoring.

