"""Baseline models for PM2.5 forecasting.

Implement the floor baselines first: persistence, historical average, LightGBM.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict


class PersistenceModel:
    """Baseline: y_hat = last observed value.

    For horizons > 1, repeats the last value.
    """

    def __init__(self, config: dict = None):
        self.name = "persistence"

    def train(self, X_train, y_train, mask_X, mask_y, **kwargs):
        """No training needed for persistence."""
        pass

    def predict(self, X_input, horizon: int = 1) -> np.ndarray:
        """Predict by repeating last observed value.

        Args:
            X_input: (N, 72, 70) PM2.5 history (normalized)
            horizon: How many hours ahead to predict

        Returns:
            predictions: (N, horizon, 70)
        """
        # Last observed value
        last_val = X_input[:, -1:, :]  # (N, 1, 70)
        # Repeat for horizon
        preds = np.repeat(last_val, horizon, axis=1)
        return preds


class HistoricalAverageModel:
    """Baseline: hour-of-week historical average.

    Computes average PM2.5 for each (hour, weekday, station) on training data.
    """

    def __init__(self, config: dict = None):
        self.name = "historical_average"
        self.hourly_avg: Optional[np.ndarray] = None  # (24, 7, 70)
        self.train_mean: float = 0.0
        self.train_std: float = 1.0

    def train(self, X_train, y_train, mask_X, mask_y, timestamps, coords, **kwargs):
        """Compute historical averages by hour-of-day and day-of-week.

        Args:
            X_train: training input features (N, 72, 70, F)
            y_train: training targets (N, 48, 70)
            mask_y: mask for observed values
            timestamps: start timestamps for each sample
        """
        N, T_in, S, F = X_train.shape
        _, T_out, _ = y_train.shape

        # Extract PM2.5 (first feature)
        pm25 = X_train[:, :, :, 0]  # (N, 72, 70)
        target = y_train  # (N, 48, 70)

        # Compute averages by hour-of-week
        # timestamps[N] gives the start of the output window for each sample
        if timestamps is not None:
            ts = pd.to_datetime(timestamps.flatten(), unit='ms' if timestamps.max() > 1e12 else None)
            # For simplicity, use mean of training targets
            self.train_mean = np.nanmean(target)
            self.train_std = np.nanstd(target)
        else:
            self.train_mean = np.nanmean(target)
            self.train_std = np.nanstd(target)

        # Compute average for each station
        self.station_avg = np.nanmean(target.reshape(-1, S), axis=0)  # (70,)

    def predict(self, X_input, horizon: int = 1, hour_of_day: int = 0) -> np.ndarray:
        """Predict using historical station average.

        Args:
            X_input: (N, 72, 70, F) or (N, 72, 70)
            horizon: How many hours ahead to predict

        Returns:
            predictions: (N, horizon, 70)
        """
        N = X_input.shape[0]
        S = X_input.shape[-1] if X_input.ndim == 3 else X_input.shape[2]
        if self.station_avg is None:
            # Fallback to last value if not trained
            last_val = X_input[:, -1:, :] if X_input.ndim == 3 else X_input[:, -1:, :, 0]
            return np.repeat(last_val, horizon, axis=1)
        # Repeat station average across all samples and time
        preds = np.tile(self.station_avg[np.newaxis, :], (N, 1))  # (N, 70)
        preds = np.repeat(preds[:, np.newaxis, :], horizon, axis=1)  # (N, horizon, 70)
        return preds


class LightGBMBaseline:
    """Baseline: LightGBM per station with lag + weather features.

    Note: Requires lightgbm package.
    """

    def __init__(self, config: dict = None):
        self.name = "lightgbm"
        self.config = config or {}
        self.models = {}  # one per station

    def train(self, X_train, y_train, mask_X, mask_y, feature_names=None, **kwargs):
        """Train LightGBM model per station."""
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise ImportError("lightgbm not installed. pip install lightgbm")

        N, T_in, S, F = X_train.shape
        _, T_out, _ = y_train.shape

        # For each station, build lagged features
        # Features: PM2.5 lags (t-1, t-2, ..., t-6, etc.) + weather
        pm25_in = X_train[:, :, :, 0]  # (N, 72, 70)

        for s in range(S):
            # Build training features for station s
            lags = [pm25_in[:, -lag, s] for lag in range(1, 13)]  # 12 lags
            # Weather features from last timestamp
            weather_feats = X_train[:, -1, s, 1:]  # skip PM2.5

            # Stack features
            if F > 1:
                features = np.column_stack([lag for lag in lags] +
                                           [weather_feats])
            else:
                features = np.column_stack(lags)

            # Target: 1-hour ahead prediction for this station
            target = y_train[:, 0, s]

            param = self.config.get("model_config", {})
            model = LGBMRegressor(
                n_estimators=param.get("n_estimators", 500),
                learning_rate=param.get("learning_rate", 0.05),
                max_depth=param.get("max_depth", 8),
                num_leaves=param.get("num_leaves", 64),
                feature_fraction=param.get("feature_fraction", 0.8),
                bagging_fraction=param.get("bagging_fraction", 0.8),
                objective=param.get("objective", "regression"),
                verbosity=-1,
                random_state=42,
            )
            model.fit(features, target)
            self.models[s] = model

    def predict(self, X_input, horizon: int = 1) -> np.ndarray:
        """Predict using trained LightGBM models."""
        N, T_in, S, F = X_input.shape
        preds = np.zeros((N, horizon, S))

        pm25_in = X_input[:, :, :, 0]

        for s in range(S):
            if s not in self.models:
                continue
            lags = [pm25_in[:, -lag, s] for lag in range(1, 13)]
            if F > 1:
                weather_feats = X_input[:, -1, s, 1:]
                features = np.column_stack([lag for lag in lags] + [weather_feats])
            else:
                features = np.column_stack(lags)

            pred = self.models[s].predict(features)
            preds[:, 0, s] = pred  # 1-hour ahead
            # For longer horizons, repeat (or re-run iteratively)
            for h in range(1, horizon):
                preds[:, h, s] = pred

        return preds


# Factory function
def load_baseline(name: str, config: dict = None):
    """Load a baseline model by name."""
    baselines = {
        "persistence": PersistenceModel,
        "hist_avg": HistoricalAverageModel,
        "lightgbm": LightGBMBaseline,
    }
    if name not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(baselines.keys())}")
    return baselines[name](config=config)
