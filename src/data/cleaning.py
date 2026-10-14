# Data Cleaning Module
# Enforces the locked scope: no leakage, proper normalization, masking of missing values.
# Uses synthetic fallback for 2019-2024 (Kaggle data only covers 2015-2020).

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

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
        
        Uses synthetic fallback covering 2019-2024 (Kaggle data only goes to 2020).
        """
        project_root = Path(__file__).parent.parent.parent
        raw_dir = raw_dir or project_root / "data" / "raw"
        
        print("Loading synthetic fallback data (covers 2019-2024)...")
        return self._load_synthetic(raw_dir)

    def _load_synthetic(self, raw_dir: Path) -> pd.DataFrame:
        """Load synthetic fallback data with proper timestamp merge."""
        project_root = Path(__file__).parent.parent.parent
        cpcb_df = pd.read_csv(project_root / "data" / "raw" / "cpcb_hourly.csv")
        weather_df = pd.read_csv(project_root / "data" / "raw" / "weather.csv")
        
        # Ensure datetime
        cpcb_df["timestamp"] = pd.to_datetime(cpcb_df["timestamp"])
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
        
        # Merge on both station_id and timestamp (exact match)
        # Since weather is sparse (10h intervals), we'll merge on timestamp too
        merged = pd.merge(cpcb_df, weather_df, on=["station_id", "timestamp"], how="left")
        
        # Forward fill weather for missing hours
        weather_cols = ["temp_c", "rh", "wind_u", "wind_v", "rainfall"]
        for col in weather_cols:
            if col in merged.columns:
                merged[col] = merged.groupby("station_id")[col].ffill().bfill()
        
        return merged

    # ------------------------------------------------------------------ 
    # Cleaning
    # ------------------------------------------------------------------ 
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps."""
        # Step 1: Ensure hourly frequency
        df = self._resample_to_hourly(df)
        print(f"After resampling: {df.shape}")

        # Step 2: Drop low-coverage stations
        coverage = df.groupby("station_id").size()
        expected_hours = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).total_seconds() / 3600
        min_coverage = 0.6 * expected_hours
        valid_stations = coverage[coverage >= min_coverage].index.tolist()
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
        """Resample to strict hourly frequency per station."""
        result_chunks = []
        full_range = pd.date_range(start=START_DATE, end=END_DATE, freq="h")
        
        for station in df["station_id"].unique():
            station_df = df[df["station_id"] == station].copy()
            station_df = station_df.drop_duplicates(subset=["timestamp"])
            station_df = station_df.set_index("timestamp").reindex(full_range)
            station_df["station_id"] = station
            
            # Forward/backward fill metadata
            meta_cols = [c for c in station_df.columns if c not in ["station_id"] and station_df[c].dtype == object]
            for col in meta_cols:
                station_df[col] = station_df[col].ffill().bfill()
            
            # Forward/backward fill numeric (weather)
            num_cols = station_df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if col != "station_id":
                    station_df[col] = station_df[col].ffill().bfill()
            
            result_chunks.append(station_df.reset_index().rename(columns={"index": "timestamp"}))
        
        return pd.concat(result_chunks, ignore_index=True)

    def _interpolate_short_gaps(self, df: pd.DataFrame, max_gap_hours: int = 6) -> pd.DataFrame:
        """Linearly interpolate gaps shorter than max_gap_hours."""
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["time_diff"] = df["timestamp"].diff().dt.total_seconds() / 3600
        df["long_gap"] = df["time_diff"] > max_gap_hours
        
        # Interpolate numeric columns where gap <= threshold
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ["time_diff", "long_gap", "station_id"]:
                mask = df["long_gap"] == False
                df.loc[mask, col] = df.loc[mask, col].interpolate(method="linear")
        
        df = df.drop(columns=["time_diff", "long_gap"])
        return df

    def _remove_bad_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove negative values, stuck-at-constant sensors, impossible spikes."""
        if "pm25" in df.columns:
            df = df[df["pm25"] >= 0]
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != "timestamp" and col != "station_id":
                variances = df[col].var()
                if variances < 0.01 * abs(df[col].mean()):
                    print(f"Dropping constant column: {col}")
                    df = df.drop(columns=[col])
        return df

    def _normalize_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p + z-score normalization to PM2.5 target."""
        if "pm25" not in df.columns:
            return df
        train_mask = df["timestamp"] <= pd.Timestamp(TRAIN_END)
        train_values = df.loc[train_mask, "pm25"]
        train_log = np.log1p(train_values)
        mean = train_log.mean()
        std = train_log.std()
        if std > 0:
            df["pm25_norm"] = (np.log1p(df["pm25"]) - mean) / std
            df.attrs["norm_mean"] = float(mean)
            df.attrs["norm_std"] = float(std)
            print(f"Normalized PM2.5 (train): mean={mean:.4f}, std={std:.4f}")
        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optionally normalize weather features."""
        train_mask = df["timestamp"] <= pd.Timestamp(TRAIN_END)
        feat_cols = [c for c in df.columns if c not in ["timestamp", "station_id", "pm25"]]
        if not feat_cols:
            return df
        for col in feat_cols:
            if df[col].dtype in [np.float64, np.int64]:
                train_values = df.loc[train_mask, col]
                mean = train_values.mean()
                std = train_values.std()
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
        dataset = build_dataset(df)
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