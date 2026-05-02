# 4-Bus Power Flow (Export Bundle)

This folder is a self-contained bundle to compute and report the original 4-bus case power flow results using OpenDSSDirect.

## Contents
- `run_4bus_pf.py` — main script to run the 3-phase Newton power flow and print a report
- `cal_pf.py` — Newton power flow solver (rectangular coordinates, PV/PQ/Slack)
- `parse_opendss_file.py` — OpenDSS-to-MPC parser and Y-bus builder
- `utilities/report_results.py` — simple voltage report printer
- `config.py` — paths for `data/` and `results/`
- `data/4Bus-YY-Bal.DSS`, `data/LineConstantsCode4.txt` — test case
- `requirements.txt` — minimal dependencies

## Quick start
1) Python 3.9+ recommended
2) Create and activate a virtual environment (optional but recommended)

   python -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate

3) Install dependencies

   pip install -r requirements.txt

4) Run the script

   python run_4bus_pf.py

The script will parse the 4-bus DSS model, run the power flow, and print a concise bus voltage report.

## Notes
- Requires OpenDSSDirect; the `OpenDSSDirect.py` package provides the `opendssdirect` import.
- Results are printed to stdout. A `results/` folder is included if you want to save outputs later.
