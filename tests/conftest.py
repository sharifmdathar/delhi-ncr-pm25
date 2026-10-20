"""Shared test fixtures.

The full frozen dataset is not in the repository (it is 36 MB and
regenerable), so a fresh clone has only the small sample dataset.  The tests
run against whichever is available, so a reviewer who clones and runs
``pytest`` gets real coverage rather than skips.
"""
from pathlib import Path

import pytest

FULL = Path("data/processed/delhi_ncr_pm25_2019_2025.npz")
SAMPLE = Path("data/sample/sample_dataset.npz")


@pytest.fixture(scope="module")
def data_path():
    """Path to the dataset the tests should run against."""
    if FULL.exists():
        return FULL
    if SAMPLE.exists():
        return SAMPLE
    pytest.skip(
        f"no dataset found: neither {FULL} nor {SAMPLE}. Build one with "
        "scripts/build_dataset.py, or run scripts/make_sample_dataset.py.")


@pytest.fixture(scope="module")
def ds(data_path):
    from src.data.dataset import load_dataset
    return load_dataset(str(data_path))
