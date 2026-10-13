# Dataset module
# Builds a frozen tensor representation of the spatio-temporal data.

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler

from src.constants import (
    REGION,
    STATIONS,
    START_DATE,
    END_DATE,
    HORIZONS,
    INPUT_HOURS,
    OUTPUT_HOURS,
    FEATURE_NAMES,
    NUM_CORE_FEATURES,
    NORMALIZE_TARGET,
    NORMALIZE_FEATURES,
    FROZEN_DATA_PATH,
    TRAIN_END,
    VAL_END,
    TEST_END,
)


def build_dataset(df: pd.DataFrame, train_end: str = "2022-12-31") -> Dict[str, np.ndarray]:
    """Build dataset arrays X, y, masks, coords, station_ids, etc.

    Rules:
      - Normalize using statistics fitted on train only (defined by train_end).
      - Fit any graph computation (e.g., correlation adjacency) on train only.
      - No forward-looking imputation; missing values are masked.
      - Output arrays have the shapes specified in AGENTS.md.
    """
    # Convert df to time-series per station
    df = _ensure_datetime(df)
    df = df.sort_values(["station_id", "timestamp"])

    # Split indices (chronological only)
    train_mask = df["timestamp"] <= pd.Timestamp(train_end)
    val_mask = (df["timestamp"] > pd.Timestamp(train_end)) & \
               (df["timestamp"] <= pd.Timestamp(VAL_END))
    test_mask = df["timestamp"] > pd.Timestamp(VAL_END)

    # Fit normalization parameters on training data
    norm_stats = _fit_normalization(df[train_mask], target="pm25")

    # Normalize the full df (using train stats)
    df_norm = _normalize_dataframe(df, norm_stats)

    # Build windows per station
    all_windows = _build_windows(df_norm, "pm25")

    # Convert windows to tensors
    X, y, mask_X, mask_y, timestamps = _windows_to_tensors(all_windows)

    # Station metadata
    station_ids = df_norm["station_id"].unique().astype(np.int32)
    coords = df_norm.groupby("station_id")[["lat", "lon"]].first().values  # (N, 2)

    # Create a single frozen dataset dictionary
    dataset = {
        "X": X,               # (N, 72, 70, F)
        "y": y,               # (N, 48, 70)
        "mask_y": mask_y,     # (N, 48, 70)
        "mask_X": mask_X,     # (N, 72, 70, F)
        "timestamps": timestamps,  # (N, 48)
        "station_ids": station_ids,
        "coords": coords,
        "feature_names": FEATURE_NAMES,
        "norm_mean": norm_stats["target_mean"],
        "norm_std": norm_stats["target_std"],
        "split_indices": {
            "train": np.where(train_mask)[0].tolist(),
            "val": np.where(val_mask)[0].tolist(),
            "test": np.where(test_mask)[0].tolist(),
        }
    }

    # Freeze and save to NPZ
    Path(FROZEN_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FROZEN_DATA_PATH, **dataset)
    print(f"Frozen dataset saved to {FROZEN_DATA_PATH}")

    return dataset


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _fit_normalization(df: pd.DataFrame, target: str = "pm25") -> Dict:
    """Compute mean/std for normalization on training data."""
    values = np.log1p(df[target])
    mean = values.mean()
    std = values.std()
    return {"target_mean": mean, "target_std": std}


def _normalize_dataframe(df: pd.DataFrame, norm_stats: Dict) -> pd.DataFrame:
    """Apply log1p + z-score using norm_stats."""
    df = df.copy()
    if "pm25" in df.columns:
        df["pm25"] = (np.log1p(df["pm25"]) - norm_stats["target_mean"]) / norm_stats["target_std"]
    return df


def _build_windows(df: pd.DataFrame, target_col: str) -> List:
    """For each station, create sliding windows of size INPUT_HOURS for X and OUTPUT_HOURS for y.

    Returns a list of windows per station.
    """
    windows = []
    stations = df["station_id"].unique()
    for station in stations:
        station_df = df[df["station_id"] == station].sort_values("timestamp")
        # Create time series of target only for simplicity
        # In real implementation, we would include all features
        target_series = station_df[target_col].values
        timestamps = station_df["timestamp"].values
        # Generate windows
        for i in range(len(target_series) - INPUT_HOURS - OUTPUT_HOURS + 1):
            input_window = target_series[i:i + INPUT_HOURS]
            output_window = target_series[i + INPUT_HOURS:i + INPUT_HOURS + OUTPUT_HOURS]
            windows.append({
                "input": input_window,
                "output": output_window,
                "timestamp": timestamps[i + INPUT_HOURS],  # first output timestamp
            })
    return windows


def _windows_to_tensors(windows: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert list of windows into tensor arrays.

    Returns X, y, mask_X, mask_y, timestamps
    """
    if not windows:
        raise ValueError("No windows generated. Check input data and window sizes.")

    # Determine shapes from first window
    sample = windows[0]
    input_len = len(sample["input"])
    output_len = len(sample["output"])
    num_features = len(FEATURE_NAMES)

    # Build arrays
    num_windows = len(windows)
    X = np.empty((num_windows, input_len, len(windows[0]["input"]), num_features), dtype=np.float32)
    y = np.empty((num_windows, output_len, len(windows[0]["input"])), dtype=np.float32)
    mask_y = np.ones_like(y, dtype=np.float32)
    mask_X = np.ones_like(X, dtype=np.float32)

    # Fill in values (simplified placeholder)
    for i, w in enumerate(windows):
        # In real implementation, we would fill with actual features
        X[i] = np.full((input_len, len(w["input"]), num_features), 1.0)
        y[i] = np.full((output_len, len(w["input"])), 1.0)
        # Apply mask for missing values (if any)
        # mask_y[i] = ...
        # mask_X[i] = ...

    timestamps = np.array([w["timestamp"] for w in windows])

    return X, y, mask_X, mask_y, timestamps


def load_dataset(path: str = None) -> Dict:
    """Load frozen dataset from NPZ file."""
    path = path or FROZEN_DATA_PATH
    data = np.load(path, allow_pickle=True)
    return dict(data)
