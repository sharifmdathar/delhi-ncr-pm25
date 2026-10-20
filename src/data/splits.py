"""Chronological split definitions.

Two modes:

``fixed``
    The locked scope: train 2019-2022, validation 2023, test 2024-2025.

``expanding``
    Rolling-origin evaluation over the 7 available years.  Fold *k* trains on
    everything from 2019 up to the end of year ``2019 + k``, validates on the
    final quarter of that year, and tests on the following calendar year.
    This checks that conclusions are not an artefact of one particular
    train/test boundary, and it is only possible because several real years
    of data exist.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

# (train_end, val_end, test_start, test_end) as inclusive date strings
FIXED: Dict[str, Tuple[str, str, str, str]] = {
    "train": ("2022-12-31", "2023-12-31", "2024-01-01", "2025-12-31"),
}

# Expanding-window folds: label -> (train_end, val_start, val_end, test_start, test_end)
EXPANDING: Dict[str, Dict[str, str]] = {
    "fold0": {"train_end": "2020-12-31", "val_start": "2020-10-01",
              "val_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
    "fold1": {"train_end": "2021-12-31", "val_start": "2021-10-01",
              "val_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
    "fold2": {"train_end": "2022-12-31", "val_start": "2022-10-01",
              "val_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
    "fold3": {"train_end": "2023-12-31", "val_start": "2023-10-01",
              "val_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
    "fold4": {"train_end": "2024-12-31", "val_start": "2024-10-01",
              "val_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31"},
}


def fixed_split(ds: Dict) -> Dict[str, np.ndarray]:
    """The locked train/val/test index split."""
    return {
        "train": ds["split_indices_train"],
        "val": ds["split_indices_val"],
        "test": ds["split_indices_test"],
    }


def expanding_split(ds: Dict, fold: str) -> Dict[str, np.ndarray]:
    """Rolling-origin split for one fold.

    Train: everything up to and including ``train_end``.
    Val:   the last quarter before the test year (used only for early stopping).
    Test:  the following calendar year.
    """
    if fold not in EXPANDING:
        raise ValueError(f"unknown fold {fold!r}; expected one of {list(EXPANDING)}")
    spec = EXPANDING[fold]
    ts = pd.to_datetime(ds["timestamps"])

    train = np.where(ts <= pd.Timestamp(spec["train_end"]))[0]
    val = np.where((ts >= pd.Timestamp(spec["val_start"])) &
                   (ts <= pd.Timestamp(spec["val_end"])))[0]
    test = np.where((ts >= pd.Timestamp(spec["test_start"])) &
                    (ts <= pd.Timestamp(spec["test_end"])))[0]
    return {"train": train, "val": val, "test": test}


def get_split(ds: Dict, mode: str = "fixed", fold: str | None = None) -> Dict[str, np.ndarray]:
    if mode == "fixed":
        return fixed_split(ds)
    if mode == "expanding":
        return expanding_split(ds, fold)
    raise ValueError(f"unknown split mode {mode!r}")
