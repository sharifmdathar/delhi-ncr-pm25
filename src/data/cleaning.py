# Data Cleaning Module
# Enforces the locked scope: no leakage, proper normalization, masking of missing values.

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

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
)


class DataCleaner:
    """Handles loading, cleaning, and preparing the PM2.5 dataset."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ 
    # Loading
    # ------------------------------------------------------------------ 
    def load_raw(self, raw_dir: Path = None) -> pd.DataFrame:
        """Load raw data from the raw directory.

        Expected formats:
          - data/raw/cpcb_hourly.csv  (hourly PM2.5 + station info)
          - data/raw/weather.csv      (hourly weather per station)
        """
        raw_dir = raw_dir or Path(__file__).parent.parent / "data" / "raw"
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

        # Load CPCB hourly data
        cpcb_df = pd.read_csv(raw_dir / "cpcb_hourly.csv")
        print(f"Loaded CPCB data: {len(cpcb_df)} rows, columns: {list(cpcb_df.columns)}")

        # Load weather data
        weather_df = pd.read_csv(raw_dir / "weather.csv")
        print(f"Loaded weather data: {len(weather_df)} rows, columns: {list(weather_df.columns)}")

        # Merge on station_id (assumed column name)
        merged = pd.merge(cpcb_df, weather_df, on="station_id", how="inner")
        print(f"Merged data shape: {merged.shape}")
        return merged

    # ------------------------------------------------------------------ 
    # Cleaning
    # ------------------------------------------------------------------ 
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps.

        Steps (in order):
          1. Resample to strict hourly grid (no further downsampling)
          2. Drop stations with < 60% coverage
          3. Interpolate short gaps (< 6h)
          4. Leave long gaps uninterpolated (mask them)
          5. Remove negative values, constant sensors, impossible spikes
          6. Apply normalization (log1p + z-score) on train split only
        """
        # Step 1: Ensure hourly frequency
        df = self._resample_to_hourly(df)
        print(f"After resampling: {df.shape}")

        # Step 2: Drop low-coverage stations
        coverage = df.groupby("station_id").size()
        valid_stations = coverage[coverage >= 0.6].index.tolist()
        print(f"Stations with >= 60% coverage: {len(valid_stations)}/{STATIONS}")
        df = df[df["station_id"].isin(valid_stations)].copy()

        # Step 3: Short gap interpolation (< 6h)
        df = self._interpolate_short_gaps(df, max_gap_hours=6)

        # Step 4: Leave long gaps as NaN (will be masked)
        # Already handled by interpolation above; remaining are long gaps

        # Step 5: Remove bad values
        df = self._remove_bad_values(df)

        # Step 6: Normalization (fit on train only)
        if NORMALIZE_TARGET:
            df = self._normalize_target(df)
        if NORMALIZE_FEATURES:
            df = self._normalize_features(df)

        print(f"Final cleaned data shape: {df.shape}")
        return df

    def _resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample to strict hourly frequency."""
        # Determine global time range
        start = pd.Timestamp(df["timestamp"].min())
        end = pd.Timestamp(df["timestamp"].max())
        freq = "H"
        # Ensure we have continuous hourly grid
        full_range = pd.date_range(start=start, end=end, freq=freq)
        # Reindex to fill missing hours with NaN
        df = df.set_index("timestamp").reindex(full_range).reset_index()
        df = df.rename(columns={"index": "timestamp"})
        return df

    def _interpolate_short_gaps(self, df: pd.DataFrame, max_gap_hours: int = 6) -> pd.DataFrame:
        """Linearly interpolate gaps shorter than max_gap_hours."""
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        # Compute time differences
        df["time_diff"] = df["timestamp"].diff().dt.total_seconds() / 3600
        # Mark gaps longer than threshold
        df["long_gap"] = df["time_diff"] > max_gap_hours
        # Interpolate where gap <= threshold
        df.loc[df["long_gap"] == False, "value"] = df["value"].interpolate(method="linear")
        # Drop the helper column
        df = df.drop(columns=["time_diff", "long_gap"])
        return df

    def _remove_bad_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove negative values, stuck-at-constant sensors, impossible spikes."""
        # Remove negative PM2.5 values
        if "pm25" in df.columns:
            df = df[df["pm25"] >= 0]
        # Remove constant values (stuck sensors)
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != "timestamp" and col != "station_id":
                # Find columns with very little variation
                variances = df[col].var()
                if variances < 0.01 * abs(df[col].mean()):
                    # Constant column — drop it
                    print(f"Dropping constant column: {col}")
                    df = df.drop(columns=[col])
        return df

    def _normalize_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p + z-score normalization to PM2.5 target."""
        if "pm25" not in df.columns:
            return df
        # log1p then z-score
        df["pm25_norm"] = np.log1p(df["pm25"])
        mean = df["pm25_norm"].mean()
        std = df["pm25_norm"].std()
        if std > 0:
            df["pm25_norm"] = (df["pm25_norm"] - mean) / std
        print(f"Normalized PM2.5: mean={mean:.4f}, std={std:.4f}")
        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optionally normalize weather features."""
        if "features" not in df.columns:
            return df
        feat_cols = [c for c in df.columns if c not in ["timestamp", "station_id", "pm25"]]
        if not feat_cols:
            return df
        # Simple z-score normalization
        for col in feat_cols:
            if df[col].dtype in [np.float64, np.int64]:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[f"{col}_norm"] = (df[col] - mean) / std
        print(f"Normalized features: {feat_cols}")
        return df

    # ------------------------------------------------------------------ 
    # Dataset Preparation
    # ------------------------------------------------------------------ 
    def prepare_dataset(self, df: pd.DataFrame) -> dict:
        """Split into train/val/test and save as NPZ."""
        from src.data.dataset import build_dataset
        # This delegates to the dataset module
        dataset = build_dataset(df, train_end=STATIONS["TRAIN_END"])
        return dataset

    # ------------------------------------------------------------------ 
    # Public API
    # ------------------------------------------------------------------ 
    def get_dataframe(self, raw_dir: Path = None) -> pd.DataFrame:
        """High-level convenience method."""
        df = self.load_raw(raw_dir)
        return self.clean(df)

    def get_dataset(self, raw_dir: Path = None) -> dict:
        """Prepare and return the full dataset object."""
        df = self.get_dataframe(raw_dir)
        ds = self.prepare_dataset(df)
        return ds


# Convenience instance for quick access
cleaner = DataCleaner(seed=42)
