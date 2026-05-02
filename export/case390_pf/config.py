from __future__ import annotations

import os
from pathlib import Path

# Export bundle root (directory of this file)
REPO_ROOT: Path = Path(__file__).resolve().parent

# Common directories
DATA_DIR: Path = REPO_ROOT / "data"
RESULTS_DIR: Path = REPO_ROOT / "results"


def data_file(name: str) -> str:
    """Return absolute path to a file under the data directory."""
    return os.fspath(DATA_DIR / name)


def result_file(name: str) -> str:
    """Return absolute path to a file under the results directory."""
    return os.fspath(RESULTS_DIR / name)

