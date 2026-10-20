"""Tests that the observation mask behaves correctly.

The mask is what separates "the model was wrong" from "there was no reading".
Every loss and every metric in this study honours it, so it gets its own
checks.
"""
import numpy as np
import pytest
from pathlib import Path

DATASET = Path("data/processed/delhi_ncr_pm25_2019_2025.npz")




def test_mask_is_binary(ds):
    assert set(np.unique(ds["mask_y"])) <= {0.0, 1.0}
    assert set(np.unique(ds["mask_X"])) <= {0.0, 1.0}


def test_observed_hours_are_a_large_majority(ds):
    """If the mask were mostly zero the metrics would be computed on scraps."""
    for split in ("train", "val", "test"):
        idx = ds[f"split_indices_{split}"]
        frac = (ds["mask_y"][idx] > 0).mean()
        assert frac > 0.5, (
            f"only {frac:.1%} of {split} is observed")


def test_mask_agrees_with_the_feature_mask(ds):
    pm25 = list(ds["feature_names"]).index("pm25_norm")
    assert np.array_equal(ds["mask_y"] > 0, ds["mask_X"][:, :, pm25] > 0)


def test_no_forward_looking_imputation(ds):
    """A missing hour must stay missing rather than be filled from the future.

    We check this behaviourally: the masked MAE over a window must be
    computed only on the observed entries, and a filled-in value would show
    up as a finite target where the mask says the hour was not observed.
    """
    y = ds["y_full"]
    m = ds["mask_y"] > 0
    assert np.isfinite(y[m]).all()
    assert np.isnan(y[~m]).all(), (
        "a masked-out hour carries a value, which means it was imputed")


def test_split_observed_fractions_are_reported(ds):
    """The observed fraction per split must be in the provenance artefact."""
    path = Path("outputs/provenance.json")
    if not path.exists():
        pytest.skip("outputs/provenance.json not generated")
    import json
    prov = json.loads(path.read_text())
    mask = ds["mask_y"] > 0
    for split in ("train", "val", "test"):
        idx = ds[f"split_indices_{split}"]
        expected = float(mask[idx].mean())
        got = prov["mask_audit"][split]["observed_fraction"]
        assert abs(expected - got) < 1e-6, (
            f"{split} observed fraction {got} does not match the dataset {expected}")
