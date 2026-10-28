# Data Cleaning Module
# Enforces the locked scope: no leakage, proper normalization, masking of missing values.
# REAL DATA ONLY - no synthetic fallback in production path.
#
# Source: CPCB 15-minute observations (Vonter/india-cpcb-aqi GitHub releases),
# fetched per station into data/raw/cpcb_station_csv/.

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
    TEST_START,
    TEST_END,
    MIN_REAL_DATA_FRACTION,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
STATION_CSV_DIR = PROJECT_ROOT / "data" / "raw" / "cpcb_station_csv"
COORD_CSV = PROJECT_ROOT / "data" / "raw" / "station_coords.csv"


class DataCleaner:
    """Loads CPCB 15-min station data, aggregates to hourly, and prepares frames.

    REAL DATA ONLY. No synthetic fallback in the production path.
    """

    def __init__(self, seed: int = 42, stations=None):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.coverage_report = None
        # Optional station subset.  The sample-dataset builder uses this so
        # that the coverage filter, the normalisation statistics and the
        # frozen arrays all describe the same set of stations.  Filtering
        # after the cleaning would leave the normalisation fitted on a
        # different set of stations from the one in the dataset.
        self.stations = list(stations) if stations is not None else None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_raw(self, raw_dir: Path = None) -> pd.DataFrame:
        """Load all Delhi-NCR CPCB station CSVs (15-minute resolution)."""
        src = Path(raw_dir) if raw_dir else STATION_CSV_DIR
        if not src.exists():
            raise FileNotFoundError(
                f"Station CSVs not found in {src}.\n"
                f"Run: python scripts/fetch_cpcb_years.py --years 2019 2020 2021 2022 2023 2024 2025"
            )

        files = sorted(src.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No station CSVs in {src}")

        print(f"Loading {len(files)} CPCB station files from {src} ...")
        frames = []
        for f in files:
            df = pd.read_csv(f)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            # Keep only the reading at the top of each hour.  This reproduces the
            # published construction exactly: the hourly-grid reindex in clean()
            # keeps the :00 reading rather than the mean of the four 15-minute
            # values (see EXPERIMENTS.md "Data-construction detail"), and 99.99% of
            # station-hours have a :00 reading, so no data is lost.  Pre-filtering
            # here cuts the rows that reach the groupby 4x (13.5M -> 3.4M), which
            # prevents the OOM that concat-all-then-groupby triggers on a ~4 GB-RAM
            # machine, without changing the frozen arrays.
            df = df[df["timestamp"].dt.minute == 0]
            frames.append(df)

        df = pd.concat(frames, ignore_index=True)
        if self.stations is not None:
            keep = set(self.stations)
            df = df[df["station_id"].isin(keep)].copy()
            print(f"Restricted to {len(keep)} requested station(s); "
                  f"{df['station_id'].nunique()} present")
        print(f"Loaded {len(df):,} rows, {df['station_id'].nunique()} stations, "
              f"{df['timestamp'].min()} -> {df['timestamp'].max()}")
        return df

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 15-min CPCB readings to hourly, preserving the observation mask.

        The mask records whether *any* real observation existed in the hour, so
        imputed hours stay distinguishable from observed ones downstream.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Restrict to the locked period
        df = df[(df["timestamp"] >= pd.Timestamp(START_DATE)) &
                (df["timestamp"] <= pd.Timestamp(END_DATE) + pd.Timedelta(hours=23, minutes=59))]

        # Numeric coercion + sentinel handling
        for c in ["pm25", "temp_c", "rh", "wind_speed", "wind_dir", "rainfall"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df["pm25"] < 0, "pm25"] = np.nan
        df.loc[df["pm25"] > 2000, "pm25"] = np.nan
        df.loc[df["rh"] > 100, "rh"] = np.nan
        df.loc[df["rh"] < 0, "rh"] = np.nan

        # Meteorological wind -> u/v components.
        # WD is the direction the wind blows FROM; u = -speed*sin(wd), v = -speed*cos(wd)
        wd_rad = np.deg2rad(df["wind_dir"])
        df["wind_u"] = (-df["wind_speed"] * np.sin(wd_rad)).astype("float32")
        df["wind_v"] = (-df["wind_speed"] * np.cos(wd_rad)).astype("float32")

        # Aggregate to hourly. min_count=1 means a value is only produced from
        # real observations; the observed-count column becomes the mask.
        agg = df.groupby(["station_id", "timestamp"]).agg(
            pm25=("pm25", "mean"),
            temp_c=("temp_c", "mean"),
            rh=("rh", "mean"),
            wind_u=("wind_u", "mean"),
            wind_v=("wind_v", "mean"),
            rainfall=("rainfall", "mean"),
            n_obs=("pm25", "count"),
        ).reset_index()

        # Keep the identity columns from the first observation of each hour
        meta = (df.groupby(["station_id", "timestamp"])
                  .agg(state=("state", "first"),
                       city=("city", "first"),
                       station_name=("station_name", "first"))
                  .reset_index())
        agg = agg.merge(meta, on=["station_id", "timestamp"], how="left")

        # Observed = at least one real pm25 reading in the hour
        agg["observed"] = (agg["n_obs"] > 0).astype("float32")
        agg = agg.drop(columns=["n_obs"])

        # Full hourly grid per station (so gaps are explicit NaN, not missing rows)
        full_range = pd.date_range(start=START_DATE, end=END_DATE, freq="h")
        out = []
        for sid, g in agg.groupby("station_id"):
            g = g.set_index("timestamp").reindex(full_range)
            g["station_id"] = sid
            for c in ["state", "city", "station_name"]:
                g[c] = g[c].ffill().bfill()
            g["observed"] = g["observed"].fillna(0.0)
            out.append(g.reset_index().rename(columns={"index": "timestamp"}))

        df = pd.concat(out, ignore_index=True)
        df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

        # Drop stations that cannot meet the coverage bar in every split
        df = self._filter_low_coverage_stations(df)

        # --- Coverage verification (fails fast) ---
        self._verify_coverage(df)

        # Normalization (fit on train only)
        if NORMALIZE_TARGET:
            df = self._normalize_target(df)
        if NORMALIZE_FEATURES:
            df = self._normalize_features(df)

        print(f"Cleaned data: {df.shape}, stations={df['station_id'].nunique()}")
        return df

    def _filter_low_coverage_stations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop stations whose real-observation fraction is below the bar in any split.

        This keeps the modelling set homogeneous: every retained station has
        genuine measurements in train, val and test.
        """
        splits = {
            "train": (pd.Timestamp(START_DATE), pd.Timestamp(TRAIN_END)),
            "val": (pd.Timestamp(TRAIN_END) + pd.Timedelta(hours=1), pd.Timestamp(VAL_END)),
            "test": (pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)),
        }
        min_frac = MIN_REAL_DATA_FRACTION
        keep = []
        drop = []
        for sid, g in df.groupby("station_id"):
            worst = 1.0
            for lo, hi in splits.values():
                sub = g[(g["timestamp"] >= lo) & (g["timestamp"] <= hi)]
                frac = float(sub["observed"].mean()) if len(sub) else 0.0
                worst = min(worst, frac)
            if worst >= min_frac:
                keep.append(sid)
            else:
                drop.append((sid, worst))

        if drop:
            print(f"\nDropping {len(drop)} station(s) below {min_frac:.0%} real coverage "
                  f"in at least one split:")
            for sid, w in sorted(drop, key=lambda x: x[1]):
                print(f"  {sid}: {w:.0%}")
            df = df[df["station_id"].isin(keep)].copy()

        print(f"Retained {df['station_id'].nunique()} stations")
        return df

    def _verify_coverage(self, df: pd.DataFrame):
        """Per-station, per-year real-observation fractions; fails if a split is thin."""
        df = df.copy()
        df["year"] = df["timestamp"].dt.year

        hours_per_year = {2019: 8760, 2020: 8784, 2021: 8760, 2022: 8760,
                          2023: 8760, 2024: 8784, 2025: 8760}
        rows = []
        for sid, g in df.groupby("station_id"):
            for y in sorted(df["year"].unique()):
                gy = g[g["year"] == y]
                obs = float(gy["observed"].sum())
                exp = hours_per_year.get(int(y), 8760)
                rows.append({"station_id": sid, "year": int(y),
                             "observed_hours": int(obs), "expected_hours": exp,
                             "fraction": obs / exp if exp else 0.0})
        cov = pd.DataFrame(rows)
        self.coverage_report = cov

        print("\n=== REAL DATA COVERAGE (observed hours / expected) ===")
        for sid, g in cov.groupby("station_id"):
            s = ", ".join(f"{int(r['year'])}:{r['fraction']:.0%}" for _, r in g.iterrows())
            print(f"  {sid}: {s}")

        # Per-split coverage: every station must have >= MIN_REAL_DATA_FRACTION observed
        splits = {
            "train": (pd.Timestamp(START_DATE), pd.Timestamp(TRAIN_END)),
            "val": (pd.Timestamp(TRAIN_END) + pd.Timedelta(hours=1), pd.Timestamp(VAL_END)),
            "test": (pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)),
        }
        min_frac = MIN_REAL_DATA_FRACTION
        for split, (lo, hi) in splits.items():
            sub = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]
            if len(sub) == 0:
                raise ValueError(f"split '{split}' is empty")
            per_station = sub.groupby("station_id")["observed"].mean()
            bad = per_station[per_station < min_frac]
            if len(bad):
                raise ValueError(
                    f"Insufficient real data in {split} for {len(bad)} station(s): "
                    + ", ".join(f"{s}={v:.0%}" for s, v in bad.items())
                    + f" (need >= {min_frac:.0%})"
                )
            print(f"  {split}: {len(per_station)} stations, "
                  f"min observed fraction {per_station.min():.0%} (>= {min_frac:.0%}) OK")

    def _normalize_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """log1p + z-score for PM2.5, fit on train split only."""
        train = df["timestamp"] <= pd.Timestamp(TRAIN_END)
        vals = df.loc[train & (df["observed"] > 0), "pm25"]
        log = np.log1p(vals)
        mean, std = float(log.mean()), float(log.std())
        if std == 0:
            std = 1.0
        df["pm25_norm"] = ((np.log1p(df["pm25"].clip(lower=0)) - mean) / std).astype("float32")
        df.attrs["norm_mean"] = mean
        df.attrs["norm_std"] = std
        print(f"Normalized PM2.5 (train): mean={mean:.4f}, std={std:.4f}")
        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """z-score weather features, fit on train split only."""
        train = df["timestamp"] <= pd.Timestamp(TRAIN_END)
        feats = ["temp_c", "rh", "wind_u", "wind_v", "rainfall"]
        for c in feats:
            if c not in df.columns:
                continue
            tv = df.loc[train, c]
            m, s = float(tv.mean()), float(tv.std())
            if s > 0:
                df[f"{c}_norm"] = ((df[c] - m) / s).astype("float32")
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_dataframe(self, raw_dir: Path = None) -> pd.DataFrame:
        df = self.load_raw(raw_dir)
        return self.clean(df)

    def get_dataset(self, raw_dir: Path = None) -> dict:
        from src.data.dataset import build_dataset
        return build_dataset(self.get_dataframe(raw_dir))


# Convenience instance
cleaner = DataCleaner(seed=42)
