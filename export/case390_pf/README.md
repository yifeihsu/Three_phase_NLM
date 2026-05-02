# IEEE 342-Node System Power Flow

This folder is a self-contained bundle to run the IEEE 342-node (This is the case included with OpenDSS with the name LVTestCaseNorthAmerican. In OpenDSS it actually has 390 nodes.) primary feeder power flow using OpenDSSDirect.py.

## Contents
- `case_390.py` — main script that parses the IEEE 342(390)-node OpenDSS model, runs the 3-phase Newton power flow, and prints a voltage summary.
- `cal_pf.py` — 3-phase Newton power flow solver (rectangular coordinates, PV/PQ/Slack handling).
- `parse_opendss_file.py` — OpenDSS-to-MPC parser, Y-bus builder, and switch-merging helper.
- `utilities/` — measurement model, Jacobian builders, and reporting helper (`report_results.py`).
- `config.py` — local paths for `data/` and `results/`.
- `data/Master.dss`, `data/LineConstantsCode.txt`, `data/buscoords.dat` — IEEE 342-node test case and line constants.
- `requirements.txt` — minimal dependencies.

## Quick start
1) Use Python 3.9+ (recommended).
2) Create and activate a virtual environment (optional but recommended):

   python -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate

3) Install dependencies:
Note that for opendssdirect.py you may need to follow the official installation guide.

   pip install -r requirements.txt

4) Run the power flow script:

   python case_390.py

The script will parse `data/Master.dss`, build the three-phase network model, run the Newton power flow, and print a concise bus voltage report.

## Notes
- Requires OpenDSSDirect; the `OpenDSSDirect.py` package provides the `opendssdirect` import.
