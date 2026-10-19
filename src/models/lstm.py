"""Univariate and Multivariate LSTM models for PM2.5 forecasting.

Both variants use the shared output head from ``base.ResidualForecaster``:
the network predicts the change from the last observed value and a linear
autoregressive skip (the ridge solution) is added, so the model starts at
Persistence and is measured relative to a strong linear predictor.

Univariate: one LSTM applied to each station separately with *shared* weights
(so per-station history is preserved rather than averaged away).
Multivariate: one LSTM over all stations concatenated as channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from src.models.base import ResidualForecaster, DeltaLinearSkip, zero_init


class UnivariateLSTM(ResidualForecaster):
    """Shared-weight LSTM applied per station.

    Input: (batch, seq_len, num_stations, num_features)
    Output: (batch, output_horizons, num_stations)
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.name = "lstm_univariate"

        hidden_size = self.config.get("hidden_size", 64)
        num_layers = self.config.get("num_layers", 2)
        dropout = self.config.get("dropout", 0.3)
        input_size = self.config.get("input_size", 1)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        # delta head: hidden -> H, zero-initialised so the model starts at Persistence
        self.fc = zero_init(nn.Linear(hidden_size, self.output_horizons))

        if self.use_lag_skip:
            self.lag_linear = DeltaLinearSkip(
                self.num_stations, self.lag_skip, self.output_horizons)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, n_feat = x.shape
        # (B, T, N, F) -> (B*N, T, F): shared LSTM across stations
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, n_feat)
        out, _ = self.lstm(x_flat)            # (B*N, T, H)
        last = out[:, -1, :]                  # (B*N, H)
        delta = self.fc(last)                 # (B*N, H)
        return delta.reshape(B, N, self.output_horizons)


class MultivariateLSTM(ResidualForecaster):
    """LSTM over all stations jointly (stations as channels).

    Input: (batch, seq_len, num_stations, num_features)
    Output: (batch, output_horizons, num_stations)
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.name = "lstm_multivariate"

        hidden_size = self.config.get("hidden_size", 64)
        num_layers = self.config.get("num_layers", 2)
        dropout = self.config.get("dropout", 0.3)
        input_size = self.config.get("input_size", 1)

        self.lstm = nn.LSTM(
            input_size=input_size * self.num_stations,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = zero_init(nn.Linear(hidden_size, self.num_stations * self.output_horizons))

        if self.use_lag_skip:
            self.lag_linear = DeltaLinearSkip(
                self.num_stations, self.lag_skip, self.output_horizons)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, n_feat = x.shape
        x_flat = x.reshape(B, T, N * n_feat)      # (B, T, N*F)
        out, _ = self.lstm(x_flat)            # (B, T, H)
        last = out[:, -1, :]                  # (B, H)
        delta = self.fc(last)                 # (B, N*H)
        return delta.reshape(B, N, self.output_horizons)


def build_lstm_model(model_type: str, config: dict = None, multivariate: bool = False):
    """Build an LSTM model.

    Args:
        model_type: "lstm_univariate" or "lstm_multivariate"
        config: model configuration
        multivariate: if True, force the multivariate variant
    """
    if model_type not in ["lstm_univariate", "lstm_multivariate"]:
        raise ValueError(f"Unknown LSTM model type: {model_type}")

    if model_type == "lstm_multivariate" or multivariate:
        return MultivariateLSTM(config)
    return UnivariateLSTM(config)
