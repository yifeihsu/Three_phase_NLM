from __future__ import annotations

import os
from pathlib import Path

# Repository root (../ from this file's directory)
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Common directories
DATA_DIR: Path = REPO_ROOT / "data"
RESULTS_DIR: Path = REPO_ROOT / "results"
SRC_DIR: Path = REPO_ROOT / "src"


def data_file(name: str) -> str:
    """Return absolute path to a file under the data directory."""
    return os.fspath(DATA_DIR / name)


def result_file(name: str) -> str:
    """Return absolute path to a file under the results directory."""
    return os.fspath(RESULTS_DIR / name)

