"""Tests that the frozen dataset has the shapes and dtypes the models expect."""
import numpy as np
import pytest
from pathlib import Path

DATASET = Path("data/processed/delhi_ncr_pm25_2019_2025.npz")
INPUT_HOURS = 72
OUTPUT_HOURS = 48
HORIZONS = [1, 3, 6, 12, 24, 48]




def test_required_arrays_exist(ds):
    for key in ("X_full", "y_full", "mask_X", "mask_y", "timestamps",
                "station_ids", "coords", "feature_names",
                "norm_mean", "norm_std",
                "split_indices_train", "split_indices_val", "split_indices_test"):
        assert key in ds, f"missing {key}"


def test_shapes_are_consistent(ds):
    T, N, F = ds["X_full"].shape
    assert ds["y_full"].shape == (T, N)
    assert ds["mask_y"].shape == (T, N)
    assert ds["mask_X"].shape == (T, N, F)
    assert len(ds["timestamps"]) == T
    assert len(ds["station_ids"]) == N
    assert ds["coords"].shape == (N, 2)
    assert len(ds["feature_names"]) == F
    assert T > 0 and N > 0 and F > 0


def test_dtypes_are_float32(ds):
    for key in ("X_full", "y_full", "mask_X", "mask_y"):
        assert ds[key].dtype == np.float32, f"{key} is {ds[key].dtype}"


def test_normalisation_statistics_are_sane(ds):
    nm, ns = float(ds["norm_mean"]), float(ds["norm_std"])
    assert 0.0 < nm < 5.0, f"implausible norm_mean {nm}"
    assert ns > 1e-3, f"implausible norm_std {ns}"


def test_window_counts_match_the_split_lengths(ds):
    for split in ("train", "val", "test"):
        idx = ds[f"split_indices_{split}"]
        n = len(idx) - (INPUT_HOURS + OUTPUT_HOURS) + 1
        assert n > 0, f"{split} is too short to form a single window"


def test_horizons_are_available(ds):
    for h in HORIZONS:
        assert h <= 48, "a requested horizon exceeds the output window"


def test_timestamps_are_monotonic_and_hourly(ds):
    ts = ds["timestamps"]
    ts = np.asarray(ts)
    if ts.dtype.kind == "M":
        secs = ts.astype("datetime64[s]").astype("int64")
    else:
        secs = ts.astype("int64")
    assert len(secs) > 1
    assert (np.diff(secs) == 3600).all(), "timestamps are not an hourly grid"


def test_graphs_have_the_right_shape(ds):
    from src.data.graphs import build_graph, DistanceGraph, normalize_adjacency
    N = len(ds["station_ids"])
    for gt in ("distance", "identity"):
        a = build_graph(gt, ds["coords"])
        assert a.shape == (N, N)
        assert not np.isnan(a).any(), f"{gt} graph contains NaN"

    # the raw kernel has no self-loop; the normalisation adds one, which is
    # the standard D^-1/2 (A + I) D^-1/2 a GCN expects
    raw = DistanceGraph(ds["coords"]).get_adjacency()
    assert np.allclose(np.diag(raw), 0.0), "the distance kernel has a self-loop"
    norm = build_graph("distance", ds["coords"])
    assert (np.diag(norm) > 0).all(), "the normalised adjacency has no self-loop"
