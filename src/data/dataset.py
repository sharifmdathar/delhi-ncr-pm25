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
    TEST_START,
    TEST_END,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
COORD_CSV = PROJECT_ROOT / "data" / "raw" / "station_coords.csv"


def build_dataset(df: pd.DataFrame, save_path: str = None) -> Dict[str, np.ndarray]:
    """Build frozen continuous arrays X_full, y_full, masks, timestamps, coords.

    Does NOT materialize windows. Arrays have time as the first dimension:
      X_full:   [T, N, F]  (pm25 + weather, normalized)
      y_full:   [T, N]     (pm25 normalized)
      mask_X:   [T, N, F]  (1 = the underlying value was really observed)
      mask_y:   [T, N]     (1 = really observed)
      timestamps: [T]       (datetime64, naive IST)
      station_ids: [N]      (CPCB site ids)
      coords:   [N, 2]      (lat, lon)
      norm_mean, norm_std: for de-normalization
      split_indices_*: time indices per split

    Normalization statistics are fitted on the train split only.
    """
    df = _ensure_datetime(df)
    df = df.sort_values(["station_id", "timestamp"])

    station_ids = np.array(sorted(df["station_id"].unique()), dtype=str)
    n_stations = len(station_ids)
    station_to_idx = {s: i for i, s in enumerate(station_ids)}

    feature_cols = ["pm25_norm"] + [f"{c}_norm" for c in
                                    ["temp_c", "rh", "wind_u", "wind_v", "rainfall"]]
    feature_cols = [c for c in feature_cols if c in df.columns]

    full_range = pd.date_range(start=START_DATE, end=END_DATE, freq="h")
    n_timesteps = len(full_range)

    X_full = np.full((n_timesteps, n_stations, len(feature_cols)), np.nan, dtype=np.float32)
    mask_X = np.zeros((n_timesteps, n_stations, len(feature_cols)), dtype=np.float32)
    y_full = np.full((n_timesteps, n_stations), np.nan, dtype=np.float32)
    mask_y = np.zeros((n_timesteps, n_stations), dtype=np.float32)

    print(f"Building continuous arrays: T={n_timesteps}, N={n_stations}, F={len(feature_cols)}")

    for station in station_ids:
        s_idx = station_to_idx[station]
        sdf = df[df["station_id"] == station].set_index("timestamp").reindex(full_range)

        # The observation mask is per (time, station): an hour counts as observed
        # if the source contained at least one real pm25 reading for that hour.
        obs = sdf["observed"].fillna(0.0).values.astype(np.float32) if "observed" in sdf.columns \
            else sdf["pm25"].notna().values.astype(np.float32)

        for f_idx, col in enumerate(feature_cols):
            if col in sdf.columns:
                X_full[:, s_idx, f_idx] = sdf[col].values
                mask_X[:, s_idx, f_idx] = obs

        if "pm25_norm" in sdf.columns:
            y_full[:, s_idx] = sdf["pm25_norm"].values
            mask_y[:, s_idx] = obs

    # Normalization stats (train only) - recomputed from the raw pm25 column
    train_mask_time = full_range <= pd.Timestamp(TRAIN_END)
    pm25_train = df[(df["timestamp"] <= pd.Timestamp(TRAIN_END)) & (df["observed"] > 0)]["pm25"]
    pm25_log = np.log1p(np.clip(pm25_train.to_numpy(), 0, None))
    norm_mean = float(pm25_log.mean())
    norm_std = float(pm25_log.std()) or 1.0
    print(f"Normalization (train): mean={norm_mean:.4f}, std={norm_std:.4f}")

    # Coordinates
    if COORD_CSV.exists():
        coords_df = pd.read_csv(COORD_CSV)
        coords = coords_df.set_index("station_id")[["lat", "lon"]].reindex(station_ids).values
        coords = np.nan_to_num(coords.astype(np.float32), nan=0.0)
    else:
        coords = np.zeros((n_stations, 2), dtype=np.float32)

    # Split indices (time-based, non-overlapping)
    train_idx = np.where(train_mask_time)[0]
    val_mask_time = (full_range > pd.Timestamp(TRAIN_END)) & (full_range <= pd.Timestamp(VAL_END))
    val_idx = np.where(val_mask_time)[0]
    test_mask_time = (full_range >= pd.Timestamp(TEST_START)) & (full_range <= pd.Timestamp(TEST_END))
    test_idx = np.where(test_mask_time)[0]

    min_len = INPUT_HOURS + OUTPUT_HOURS
    if len(train_idx) < min_len or len(val_idx) < min_len or len(test_idx) < min_len:
        raise ValueError("Split too small for window size")

    # Guard against leakage: no split may overlap another
    if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
        raise ValueError("Split overlap detected - leakage")

    dataset = {
        "X_full": X_full,
        "y_full": y_full,
        "mask_X": mask_X,
        "mask_y": mask_y,
        "timestamps": full_range.values.astype("datetime64[ns]"),
        "station_ids": station_ids,
        "coords": coords.astype(np.float32),
        "feature_names": np.array(feature_cols, dtype=object),
        "norm_mean": np.float32(norm_mean),
        "norm_std": np.float32(norm_std),
        "split_indices_train": train_idx.astype(np.int64),
        "split_indices_val": val_idx.astype(np.int64),
        "split_indices_test": test_idx.astype(np.int64),
        "input_hours": INPUT_HOURS,
        "output_hours": OUTPUT_HOURS,
    }

    # Save to the frozen path, unless the caller asked for a different one
    # (the sample-dataset builder uses this to avoid clobbering the full
    # dataset).
    out_path = Path(save_path or FROZEN_DATA_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dataset)
    print(f"Frozen dataset saved: {out_path}")

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
    d = dict(data)
    # Normalise norm_mean/norm_std to Python floats so both the old
    # (1-element array) and new (0-d scalar) frozen formats work.
    if "norm_mean" in d:
        d["norm_mean"] = float(np.asarray(d["norm_mean"]).flat[0])
    if "norm_std" in d:
        d["norm_std"] = float(np.asarray(d["norm_std"]).flat[0])
    return d
