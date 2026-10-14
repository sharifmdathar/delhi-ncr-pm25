# Dataset module
# Builds frozen continuous arrays; windowing is done on-the-fly by WindowDS.

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from src.constants import (
    START_DATE,
    END_DATE,
    INPUT_HOURS,
    OUTPUT_HOURS,
    FEATURE_NAMES,
    FROZEN_DATA_PATH,
    TRAIN_END,
    VAL_END,
    TEST_END,
)


def build_dataset(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Build frozen continuous arrays X_full, y_full, masks, timestamps, coords.
    
    Does NOT materialize windows. Returns arrays with time as first dimension:
      X_full: [T, N, F]  (pm25 + weather, normalized)
      y_full: [T, N]     (pm25 normalized)
      mask_X: [T, N, F]  (1=observed, 0=missing)
      mask_y: [T, N]     (1=observed, 0=missing)
      timestamps: [T]    (datetime64)
      station_ids: [N]   (strings)
      coords: [N, 2]     (lat, lon)
      norm_mean, norm_std: scalars for de-normalization
      split_indices: dict with train/val/test time indices
    
    Normalization stats are fitted on train split only (TRAIN_END).
    """
    # Ensure datetime and sort
    df = _ensure_datetime(df)
    df = df.sort_values(["station_id", "timestamp"])
    
    # Get station order
    station_ids = np.array(sorted(df["station_id"].unique()), dtype=str)
    n_stations = len(station_ids)
    station_to_idx = {s: i for i, s in enumerate(station_ids)}
    
    # Feature columns
    feature_cols = ["pm25"] + [f"{c}_norm" for c in ["temp_c", "rh", "wind_u", "wind_v", "rainfall"]]
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Build continuous time grid
    full_range = pd.date_range(start=START_DATE, end=END_DATE, freq="h")
    n_timesteps = len(full_range)
    
    # Initialize arrays
    X_full = np.full((n_timesteps, n_stations, len(feature_cols)), np.nan, dtype=np.float32)
    mask_X = np.zeros((n_timesteps, n_stations, len(feature_cols)), dtype=np.float32)
    y_full = np.full((n_timesteps, n_stations), np.nan, dtype=np.float32)
    mask_y = np.zeros((n_timesteps, n_stations), dtype=np.float32)
    
    print(f"Building continuous arrays: T={n_timesteps}, N={n_stations}, F={len(feature_cols)}")
    
    # Fill arrays per station
    for station in station_ids:
        s_idx = station_to_idx[station]
        station_df = df[df["station_id"] == station].set_index("timestamp")
        station_df = station_df.reindex(full_range)
        
        # Features
        for f_idx, col in enumerate(feature_cols):
            if col in station_df.columns:
                vals = station_df[col].values
                X_full[:, s_idx, f_idx] = vals
                mask_X[:, s_idx, f_idx] = station_df[col].notna().values.astype(np.float32)
        
        # Target (pm25)
        if "pm25" in station_df.columns:
            y_full[:, s_idx] = station_df["pm25"].values
            mask_y[:, s_idx] = station_df["pm25"].notna().values.astype(np.float32)
    
    # Compute normalization stats on TRAIN split only
    train_end_ts = pd.Timestamp(TRAIN_END)
    train_mask_time = full_range <= train_end_ts
    
    # For pm25 (first feature), compute log1p mean/std on train
    pm25_train = X_full[train_mask_time, :, 0]
    pm25_train = pm25_train[~np.isnan(pm25_train)]
    pm25_log = np.log1p(np.clip(pm25_train, 0, None))
    norm_mean = float(pm25_log.mean())
    norm_std = float(pm25_log.std())
    if norm_std == 0:
        norm_std = 1.0
    
    print(f"Normalization (train): mean={norm_mean:.4f}, std={norm_std:.4f}")
    
    # Apply normalization to X_full (pm25 channel)
    X_full[:, :, 0] = (np.log1p(np.clip(X_full[:, :, 0], 0, None)) - norm_mean) / norm_std
    
    # Apply normalization to y_full (already log1p normalized in cleaning, but re-do for consistency)
    # Actually y_full should already be normalized from cleaning step
    # If not, we'd normalize here. For now, assume cleaning did it.
    
    # Coordinates (placeholder if not in df)
    if "lat" in df.columns and "lon" in df.columns:
        coords = df.groupby("station_id")[["lat", "lon"]].first().reindex(station_ids).values
    else:
        coords = np.zeros((n_stations, 2), dtype=np.float32)
    
    # Split indices (time-based)
    train_idx = np.where(train_mask_time)[0]
    val_mask_time = (full_range > pd.Timestamp(TRAIN_END)) & (full_range <= pd.Timestamp(VAL_END))
    val_idx = np.where(val_mask_time)[0]
    test_idx = np.where(full_range > pd.Timestamp(VAL_END))[0]
    
    # Ensure each split has enough data for at least one window
    min_len = INPUT_HOURS + OUTPUT_HOURS
    if len(train_idx) < min_len or len(val_idx) < min_len or len(test_idx) < min_len:
        raise ValueError("Split too small for window size")
    
    dataset = {
        "X_full": X_full,
        "y_full": y_full,
        "mask_X": mask_X,
        "mask_y": mask_y,
        "timestamps": full_range.values.astype('datetime64[ns]'),
        "station_ids": station_ids,
        "coords": coords.astype(np.float32),
        "feature_names": np.array(feature_cols, dtype=object),
        "norm_mean": np.array([norm_mean], dtype=np.float32),
        "norm_std": np.array([norm_std], dtype=np.float32),
        "split_indices_train": train_idx.astype(np.int64),
        "split_indices_val": val_idx.astype(np.int64),
        "split_indices_test": test_idx.astype(np.int64),
        "input_hours": INPUT_HOURS,
        "output_hours": OUTPUT_HOURS,
    }
    
    Path(FROZEN_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FROZEN_DATA_PATH, **dataset)
    print(f"Frozen dataset saved: {FROZEN_DATA_PATH}")
    
    return dataset


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_dataset(path: str = None) -> Dict:
    """Load frozen dataset from NPZ file."""
    path = path or FROZEN_DATA_PATH
    data = np.load(path, allow_pickle=True)
    return dict(data)