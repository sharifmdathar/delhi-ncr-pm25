# Dataset module
# Builds a frozen tensor representation of the spatio-temporal data.

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

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


def build_dataset(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Build dataset arrays X, y, masks, coords, station_ids, etc.

    Rules:
      - Normalize using statistics fitted on train only (defined by TRAIN_END).
      - Fit any graph computation (e.g., correlation adjacency) on train only.
      - No forward-looking imputation; missing values are masked.
      - Output arrays have the shapes specified in AGENTS.md.
    """
    # Convert df to time-series per station
    df = _ensure_datetime(df)
    df = df.sort_values(["station_id", "timestamp"])

    # Split indices (chronological only)
    train_mask = df["timestamp"] <= pd.Timestamp(TRAIN_END)
    val_mask = (df["timestamp"] > pd.Timestamp(TRAIN_END)) & \
               (df["timestamp"] <= pd.Timestamp(VAL_END))
    test_mask = df["timestamp"] > pd.Timestamp(VAL_END)

    # Fit normalization parameters on training data
    norm_stats = _fit_normalization(df[train_mask], target="pm25")

    # Normalize the full df (using train stats)
    df_norm = _normalize_dataframe(df, norm_stats)

    # Normalize weather features on train stats
    df_norm = _normalize_weather(df_norm, train_mask)

    # Build windows per station
    all_windows = _build_windows(df_norm)

    # Convert windows to tensors
    X, y, mask_X, mask_y, timestamps = _windows_to_tensors(all_windows)

    # Station metadata
    station_ids = np.array(sorted(df_norm["station_id"].unique()), dtype=str)
    # Get coordinates from station metadata if available, else use placeholder
    if "lat" in df_norm.columns and "lon" in df_norm.columns:
        coords = df_norm.groupby("station_id")[["lat", "lon"]].first().reindex(station_ids).values
    else:
        # Placeholder coordinates for synthetic data
        n_stations = len(station_ids)
        coords = np.zeros((n_stations, 2), dtype=np.float32)

    # Create a single frozen dataset dictionary
    dataset = {
        "X": X,               # (N, 72, S, F)
        "y": y,               # (N, 48, S)
        "mask_y": mask_y,     # (N, 48, S)
        "mask_X": mask_X,     # (N, 72, S, F)
        "timestamps": timestamps,  # (N, 48)
        "station_ids": station_ids,
        "coords": coords,
        "feature_names": np.array(FEATURE_NAMES, dtype=object),
        "norm_mean": np.array([norm_stats["target_mean"]], dtype=np.float32),
        "norm_std": np.array([norm_stats["target_std"]], dtype=np.float32),
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
    if target not in df.columns:
        return {"target_mean": 0.0, "target_std": 1.0}
    values = np.log1p(df[target].clip(lower=0))
    mean = values.mean()
    std = values.std()
    if std == 0:
        std = 1.0
    return {"target_mean": float(mean), "target_std": float(std)}


def _normalize_dataframe(df: pd.DataFrame, norm_stats: Dict) -> pd.DataFrame:
    """Apply log1p + z-score using norm_stats."""
    df = df.copy()
    if "pm25" in df.columns:
        df["pm25"] = (np.log1p(df["pm25"].clip(lower=0)) - norm_stats["target_mean"]) / norm_stats["target_std"]
    return df


def _normalize_weather(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Normalize weather features using train statistics."""
    df = df.copy()
    weather_cols = ["temp_c", "rh", "wind_u", "wind_v", "rainfall"]
    
    for col in weather_cols:
        if col in df.columns:
            train_values = df.loc[train_mask, col]
            mean = train_values.mean()
            std = train_values.std()
            if std > 0:
                df[f"{col}_norm"] = (df[col] - mean) / std
            else:
                df[f"{col}_norm"] = 0.0
    
    return df


def _build_windows(df: pd.DataFrame) -> List[Dict]:
    """For each station, create sliding windows of size INPUT_HOURS for X and OUTPUT_HOURS for y.
    
    Returns a list of dicts with input features, target, masks, and timestamps.
    Uses stride of 24 hours (daily samples) to keep dataset manageable.
    """
    windows = []
    stations = sorted(df["station_id"].unique())
    station_to_idx = {s: i for i, s in enumerate(stations)}
    num_stations = len(stations)
    
    # Feature columns (pm25 + weather normalized)
    feature_cols = ["pm25"] + [f"{c}_norm" for c in ["temp_c", "rh", "wind_u", "wind_v", "rainfall"]]
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"Building windows with {len(feature_cols)} features: {feature_cols}")
    print(f"Stations: {num_stations}")
    
    STRIDE = 24  # Daily samples to keep dataset size manageable (~150K windows vs 3.6M)
    
    for station in stations:
        station_df = df[df["station_id"] == station].sort_values("timestamp").reset_index(drop=True)
        n = len(station_df)
        
        if n < INPUT_HOURS + OUTPUT_HOURS:
            continue
        
        # Extract feature matrix and target
        features = station_df[feature_cols].values  # (n, F)
        target = station_df["pm25"].values  # (n,) - already normalized
        timestamps = station_df["timestamp"].values
        
        # Build masks (1 for observed, 0 for missing/imputed)
        feature_masks = station_df[feature_cols].notna().values.astype(np.float32)
        target_mask = station_df["pm25"].notna().values.astype(np.float32)
        
        # Generate sliding windows with stride
        for i in range(0, n - INPUT_HOURS - OUTPUT_HOURS + 1, STRIDE):
            input_feat = features[i:i + INPUT_HOURS]  # (72, F)
            input_mask = feature_masks[i:i + INPUT_HOURS]  # (72, F)
            
            output_target = target[i + INPUT_HOURS:i + INPUT_HOURS + OUTPUT_HOURS]  # (48,)
            output_mask = target_mask[i + INPUT_HOURS:i + INPUT_HOURS + OUTPUT_HOURS]  # (48,)
            
            windows.append({
                "input": input_feat,
                "input_mask": input_mask,
                "output": output_target,
                "output_mask": output_mask,
                "station_idx": station_to_idx[station],
                "timestamp": timestamps[i + INPUT_HOURS],  # first output timestamp
            })
    
    print(f"Total windows (stride={STRIDE}h): {len(windows)}")
    return windows


def _windows_to_tensors(windows: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert list of windows into tensor arrays.

    Returns X, y, mask_X, mask_y, timestamps
    """
    if not windows:
        raise ValueError("No windows generated. Check input data and window sizes.")

    # Get dimensions
    num_windows = len(windows)
    sample = windows[0]
    input_len = sample["input"].shape[0]  # 72
    output_len = sample["output"].shape[0]  # 48
    num_features = sample["input"].shape[1]  # F
    num_stations = max(w["station_idx"] for w in windows) + 1
    
    # Initialize arrays
    X = np.zeros((num_windows, input_len, num_stations, num_features), dtype=np.float32)
    mask_X = np.zeros((num_windows, input_len, num_stations, num_features), dtype=np.float32)
    y = np.zeros((num_windows, output_len, num_stations), dtype=np.float32)
    mask_y = np.zeros((num_windows, output_len, num_stations), dtype=np.float32)
    
    # Fill arrays
    for i, w in enumerate(windows):
        s = w["station_idx"]
        X[i, :, s, :] = w["input"]
        mask_X[i, :, s, :] = w["input_mask"]
        y[i, :, s] = w["output"]
        mask_y[i, :, s] = w["output_mask"]
    
    timestamps = np.array([w["timestamp"] for w in windows], dtype='datetime64[ns]')
    
    print(f"Tensor shapes: X={X.shape}, y={y.shape}, mask_X={mask_X.shape}, mask_y={mask_y.shape}")
    return X, y, mask_X, mask_y, timestamps


def load_dataset(path: str = None) -> Dict:
    """Load frozen dataset from NPZ file."""
    path = path or FROZEN_DATA_PATH
    data = np.load(path, allow_pickle=True)
    return dict(data)