"""Univariate and Multivariate LSTM models for PM2.5 forecasting.

Implements shared LSTM weights across stations (univariate).
Multivariate version concatenates all stations as input channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class UnivariateLSTM(nn.Module):
    """Shared LSTM across stations (univariate baseline).

    Input: (batch, seq_len, num_stations, num_features)
    Uses a single LSTM cell shared across stations, returns 48-horizon predictions per station.
    """

    def __init__(self, config: dict = None):
        super().__init__()
        self.name = "lstm_univariate"
        self.config = config or {}
        
        hidden_size = self.config.get("hidden_size", 128)
        num_layers = self.config.get("num_layers", 2)
        dropout = self.config.get("dropout", 0.3)
        input_size = self.config.get("input_size", 1)
        num_stations = self.config.get("num_stations", 70)
        output_horizons = self.config.get("output_horizons", 48)

        # Shared LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # Output layer: hidden -> 48 horizons * num_stations
        self.fc = nn.Linear(hidden_size, num_stations * output_horizons)
        self.output_horizons = output_horizons
        self.num_stations = num_stations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, num_stations, input_size)

        Returns:
            output: (batch, output_horizons, num_stations)
        """
        batch_size, seq_len, num_stations, _ = x.shape
        # Reshape for shared LSTM: treat stations as separate input dimensions
        # Use mean across stations before LSTM
        x_shared = x.mean(dim=2, keepdim=False)  # (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x_shared)  # (batch, seq_len, hidden_size)

        # Take last hidden state for prediction
        preds = self.fc(lstm_out[:, -1:, :])  # (batch, 1, stations*horizons)
        preds = preds.reshape(batch_size, self.output_horizons, self.num_stations)  # (batch, 48, stations)
        return preds


class MultivariateLSTM(nn.Module):
    """Multivariate LSTM that processes all stations jointly.

    Input: (batch, seq_len, num_stations, num_features)
    Treats each station as a separate channel.
    """

    def __init__(self, config: dict = None):
        super().__init__()
        self.name = "lstm_multivariate"
        self.config = config or {}

        hidden_size = self.config.get("hidden_size", 128)
        num_layers = self.config.get("num_layers", 2)
        dropout = self.config.get("dropout", 0.3)
        input_size = self.config.get("input_size", 1)
        num_stations = self.config.get("num_stations", 70)
        output_horizons = self.config.get("output_horizons", 48)

        # LSTM for each station channel (shared layers, separate input projection)
        self.lstm = nn.LSTM(
            input_size=input_size * num_stations,  # flatten stations
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, num_stations * output_horizons)
        self.num_stations = num_stations
        self.output_horizons = output_horizons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, num_stations, input_size)

        Returns:
            output: (batch, 48, num_stations)
        """
        batch_size, seq_len, num_stations, input_size = x.shape
        # Flatten stations into feature dimension
        x_flat = x.reshape(batch_size, seq_len, -1)  # (batch, seq_len, stations*input_size)
        lstm_out, _ = self.lstm(x_flat)  # (batch, seq_len, hidden_size)

        # Take last hidden state for predictions
        preds = self.fc(lstm_out[:, -1:, :])  # (batch, 1, stations*48)
        preds = preds.reshape(batch_size, self.output_horizons, num_stations)  # (batch, 48, stations)
        return preds


# Factory function
def build_lstm_model(model_type: str, config: dict = None, multivariate: bool = False):
    """Build LSTM model.

    Args:
        model_type: "lstm_univariate" or "lstm_multivariate"
        config: model configuration
        multivariate: if True, use multivariate LSTM

    Returns:
        model: torch.nn.Module
    """
    if model_type not in ["lstm_univariate", "lstm_multivariate"]:
        raise ValueError(f"Unknown LSTM model type: {model_type}")

    if model_type == "lstm_univariate":
        return UnivariateLSTM(config)
    elif model_type == "lstm_multivariate":
        if multivariate:
            return MultivariateLSTM(config)
        else:
            return UnivariateLSTM(config)
