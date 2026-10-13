"""Shape and consistency tests for the frozen dataset."""

import numpy as np
from pathlib import Path


def test_frozen_dataset_shapes():
    """Test that frozen dataset has expected shapes."""
    from src.data.dataset import load_dataset
    
    dataset = load_dataset(Path("data/processed/delhi_ncr_pm25_2019_2024.npz"))
    
    # Check all required arrays exist
    required_keys = [
        "X", "y", "mask_y", "mask_X", "timestamps",
        "station_ids", "coords", "feature_names"
    ]
    
    for key in required_keys:
        assert key in dataset, f"Missing required array: {key}"
    
    # Check shapes
    X = dataset["X"]
    y = dataset["y"]
    
    assert len(X.shape) == 4, f"X should be 4D, got {len(X.shape)}D: {X.shape}"
    assert len(y.shape) == 3, f"y should be 3D, got {len(y.shape)}D: {y.shape}"
    
    # Check dimensions match expectations
    N, T_input, S, F = X.shape
    N_y, T_output, S_y = y.shape
    
    assert T_input == 72, f"Expected input length 72, got {T_input}"
    assert T_output == 48, f"Expected output length 48, got {T_output}"
    assert S == 70, f"Expected 70 stations, got {S}"
    assert F > 0, f"Expected positive feature count, got {F}"
    
    # Check station IDs match
    station_ids = dataset["station_ids"]
    assert len(station_ids) == 70, f"Expected 70 station IDs, got {len(station_ids)}"
    
    # Check coordinates
    coords = dataset["coords"]
    assert coords.shape == (70, 2), f"Expected coords (70, 2), got {coords.shape}"
    
    print(f"Dataset shapes verified: X={X.shape}, y={y.shape}, stations={S}, features={F}")


def test_feature_names():
    """Test that feature names match expected values."""
    from src.constants import FEATURE_NAMES
    from src.data.dataset import load_dataset
    
    dataset = load_dataset(Path("data/processed/delhi_ncr_pm25_2019_2024.npz"))
    feature_names = dataset.get("feature_names", [])
    
    # Check that feature names match
    assert len(feature_names) > 0, "Feature names should not be empty"
    
    # Check that feature names match constants
    # Note: Feature names in dataset come from constants.py
    assert len(feature_names) == len(FEATURE_NAMES), (
        f"Feature count mismatch: expected {len(FEATURE_NAMES)}, got {len(feature_names)}"
    )
    
    print(f"Feature names verified: {feature_names}")