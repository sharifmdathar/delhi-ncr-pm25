"""Tests that the frozen dataset is real, and that every value in it is
traceable to an actual observation.

These are the mechanical version of the checks behind the provenance table in
``DATA_CARD.md``.  They exist because an earlier version of this study used
a synthetic fallback, and a promise that the data is real is worth much less
than a test that fails if it stops being true.
"""
import numpy as np
import pytest
from pathlib import Path

DATASET = Path("data/processed/delhi_ncr_pm25_2019_2025.npz")


@pytest.fixture(scope="module")
def ds():
    from src.data.dataset import load_dataset
    if not DATASET.exists():
        pytest.skip(f"{DATASET} not built; run scripts/build_dataset.py")
    return load_dataset(DATASET)


def test_no_value_is_fabricated(ds):
    """Every stored PM2.5 value is either observed or missing.

    A value that is present while the observation mask says the hour was not
    observed would be a fabricated (imputed or synthetic) value.
    """
    pm25 = list(ds["feature_names"]).index("pm25_norm")
    x = ds["X_full"][:, :, pm25]
    mx = ds["mask_X"][:, :, pm25] > 0

    assert np.nansum(np.isfinite(x) & ~mx) == 0, (
        "PM2.5 values exist for hours flagged as not observed")
    assert np.nansum(~np.isfinite(x) & mx) == 0, (
        "hours flagged as observed have no PM2.5 value")

    y = ds["y_full"]
    my = ds["mask_y"] > 0
    assert np.nansum(np.isfinite(y) & ~my) == 0, (
        "target values exist for hours flagged as not observed")
    assert np.nansum(~np.isfinite(y) & my) == 0, (
        "hours flagged as observed have no target value")


def test_mask_is_binary_and_consistent(ds):
    for key in ("mask_y", "mask_X"):
        m = ds[key]
        assert set(np.unique(m)) <= {0.0, 1.0}, f"{key} is not binary"
    # the target mask and the PM2.5 feature mask must agree hour by hour
    pm25 = list(ds["feature_names"]).index("pm25_norm")
    assert np.array_equal(ds["mask_y"] > 0, ds["mask_X"][:, :, pm25] > 0), (
        "mask_y and the PM2.5 feature mask disagree")


def test_every_station_meets_the_coverage_bar(ds):
    """Each station must clear the real-coverage bar in every split."""
    from src.constants import MIN_REAL_DATA_FRACTION
    mask = ds["mask_y"] > 0
    for split in ("train", "val", "test"):
        idx = ds[f"split_indices_{split}"]
        frac = mask[idx].mean(axis=0)
        bad = [(str(s), float(f)) for s, f in zip(ds["station_ids"], frac)
               if f < MIN_REAL_DATA_FRACTION]
        assert not bad, (
            f"stations below {MIN_REAL_DATA_FRACTION:.0%} real coverage in {split}: {bad}")


def test_coverage_report_matches_the_frozen_dataset(ds):
    """The per-station-year coverage report must agree with the dataset."""
    import pandas as pd
    path = Path("data/processed/coverage_report.csv")
    if not path.exists():
        pytest.skip("coverage_report.csv not built")
    cov = pd.read_csv(path)
    mask = ds["mask_y"] > 0
    ts = pd.to_datetime(ds["timestamps"], unit="s")
    sids = [str(s) for s in ds["station_ids"]]
    for _, row in cov.iterrows():
        if row["station_id"] not in sids:
            continue
        j = sids.index(row["station_id"])
        sel = np.asarray(ts.year == int(row["year"]))
        frac = mask[sel, j].mean()
        assert abs(frac - row["fraction"]) < 0.01, (
            f"coverage mismatch for {row['station_id']} {row['year']}: "
            f"report {row['fraction']:.4f} vs dataset {frac:.4f}")


def test_targets_are_finite_where_observed(ds):
    """De-normalised targets must be finite and non-negative where observed."""
    nm, ns = float(ds["norm_mean"]), float(ds["norm_std"])
    y = np.expm1(np.clip(ds["y_full"] * ns + nm, -700, 700))
    obs = ds["mask_y"] > 0
    vals = y[obs]
    assert np.all(np.isfinite(vals)), "non-finite target where observed"
    assert np.all(vals >= 0), "negative target where observed"
    assert vals.max() < 2000, "implausible target value"
