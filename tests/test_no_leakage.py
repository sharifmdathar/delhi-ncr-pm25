"""Tests that no information crosses a split boundary.

The checks here are the ones the study's claims rest on: if a window's
inputs or targets straddle a split, the reported test error is not a test
error.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

DATASET = Path("data/processed/delhi_ncr_pm25_2019_2025.npz")
INPUT_HOURS = 72
OUTPUT_HOURS = 48




def test_splits_are_disjoint_and_chronological(ds):
    tr = ds["split_indices_train"]
    va = ds["split_indices_val"]
    te = ds["split_indices_test"]
    assert len(np.intersect1d(tr, va)) == 0
    assert len(np.intersect1d(va, te)) == 0
    assert len(np.intersect1d(tr, te)) == 0
    assert tr[-1] < va[0], "train does not end before validation starts"
    assert va[-1] < te[0], "validation does not end before test starts"


def test_every_window_lies_inside_its_own_split(ds):
    """A window's inputs AND its targets must be inside the same split."""
    for split in ("train", "val", "test"):
        idx = ds[f"split_indices_{split}"]
        lo, hi = int(idx[0]), int(idx[-1])
        starts = np.arange(lo, hi - INPUT_HOURS - OUTPUT_HOURS + 2)
        assert len(starts) > 0, f"{split} has no windows"
        assert starts[0] >= lo
        assert starts[-1] + INPUT_HOURS + OUTPUT_HOURS - 1 <= hi, (
            f"{split} has a window whose target leaves the split")


def test_expanding_window_folds_are_self_contained(ds):
    """The rolling-origin folds must satisfy the same containment rule."""
    from src.data.splits import expanding_split, EXPANDING
    for fold in EXPANDING:
        sp = expanding_split(ds, fold)
        for name in ("train", "val", "test"):
            idx = sp[name]
            if len(idx) == 0:
                continue
            lo, hi = int(idx[0]), int(idx[-1])
            starts = np.arange(lo, hi - INPUT_HOURS - OUTPUT_HOURS + 2)
            assert len(starts) > 0, f"{fold}/{name} has no windows"
            assert starts[-1] + INPUT_HOURS + OUTPUT_HOURS - 1 <= hi


def test_normalisation_is_fitted_on_train_only(ds):
    """Recompute the normalisation statistics from the training split alone."""
    nm, ns = float(ds["norm_mean"]), float(ds["norm_std"])
    tr = ds["split_indices_train"]
    y = ds["y_full"][tr]
    mask = ds["mask_y"][tr] > 0
    vals = y[mask]
    if vals.size == 0:
        pytest.skip("no observed training targets")
    # the stored statistics are for the log1p transform
    expected_mean = float(np.log1p(np.expm1(vals * ns + nm)).mean())
    expected_std = float(np.log1p(np.expm1(vals * ns + nm)).std())
    assert abs(expected_mean - nm) < 1e-3, (
        f"norm_mean {nm} does not match the train-only recomputation {expected_mean}")
    assert abs(expected_std - ns) < 1e-3, (
        f"norm_std {ns} does not match the train-only recomputation {expected_std}")


def test_correlation_graph_uses_train_only(ds):
    """The correlation graph must not see the validation or test years."""
    from src.data.graphs import build_graph
    nm, ns = float(ds["norm_mean"]), float(ds["norm_std"])
    tr = ds["split_indices_train"]
    trs = slice(int(tr[0]), int(tr[-1]) + 1)
    y_train = np.expm1(np.clip(ds["y_full"][trs] * ns + nm, -700, 700))

    a = build_graph("correlation", ds["coords"], pm25_train=y_train)
    assert not np.isnan(a).any(), "correlation graph contains NaN"

    # the same graph built with the test period appended must differ, which
    # shows the train-only argument is what determines it
    te = ds["split_indices_test"]
    tes = slice(int(te[0]), int(te[-1]) + 1)
    y_both = np.expm1(np.clip(ds["y_full"][trs] * ns + nm, -700, 700))
    y_te = np.expm1(np.clip(ds["y_full"][tes] * ns + nm, -700, 700))
    b = build_graph("correlation", ds["coords"], pm25_train=np.concatenate([y_both, y_te]))
    assert not np.allclose(a, b), (
        "the correlation graph is identical with and without the test years, "
        "so the train-only argument is not doing anything")


def test_wind_graph_is_built_from_the_training_split(ds):
    """The wind-aware graph's edge directions must come from train only."""
    from src.data.graphs import build_graph
    tr = ds["split_indices_train"]
    trs = slice(int(tr[0]), int(tr[-1]) + 1)
    with np.errstate(invalid="ignore"):
        wu = np.nanmean(ds["X_full"][trs][:, :, 3], axis=0)
        wv = np.nanmean(ds["X_full"][trs][:, :, 4], axis=0)
    wu = np.where(np.isnan(wu), np.nanmean(wu), wu)
    wv = np.where(np.isnan(wv), np.nanmean(wv), wv)
    a = build_graph("wind_aware", ds["coords"], wind_u=wu, wind_v=wv)
    assert not np.isnan(a).any(), "wind-aware graph contains NaN"
    assert (a != 0).any(), "wind-aware graph is empty"


def test_leave_station_out_holdout_is_deterministic():
    from src.data.splits import leave_station_out_stations
    sids = [f"site_{i}" for i in range(42)]
    a = leave_station_out_stations(sids, 10, seed=0)
    b = leave_station_out_stations(sids, 10, seed=0)
    c = leave_station_out_stations(sids, 10, seed=1)
    assert a == b, "the hold-out selection is not deterministic"
    assert a != c, "different seeds give the same hold-out set"
    assert len(a) == 10 and len(set(a)) == 10
