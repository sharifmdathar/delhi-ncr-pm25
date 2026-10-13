"""Plain Transformer for temporal PM2.5 forecasting.

Input: (batch, seq_len, num_stations, num_features)
Treats stations as channels with temporal attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalTransformer(nn.Module):
    """Plain Transformer for temporal forecasting.

    Architecture:
    - Input projection to hidden_size
    - Positional encoding (learned)
    - Stacked Transformer encoder layers
    - Output projection to stations × horizons
    """

    def __init__(self, config: dict = None):
        super().__init__()
        self.name = "transformer"
        self.config = config or {}

        hidden_size = self.config.get("hidden_size", 128)
        num_layers = self.config.get("num_layers", 2)
        num_heads = self.config.get("num_heads", 4)
        dropout = self.config.get("dropout", 0.3)
        input_size = self.config.get("input_size", 1)
        num_stations = 70  # hardcoded
        seq_len = 72
        pred_horizons = 48

        # Input projection
        self.input_proj = nn.Linear(input_size * num_stations, hidden_size)

        # Positional encoding (learned)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_size) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, num_stations * pred_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, num_stations, input_size)

        Returns:
            output: (batch, pred_horizons, num_stations)
        """
        batch_size, seq_len, num_stations, input_size = x.shape

        # Flatten stations into feature dimension
        x = x.reshape(batch_size, seq_len, -1)  # (batch, seq_len, stations*input_size)

        # Project to hidden size
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)

        # Add positional encoding
        x = x + self.pos_embed[:, :seq_len, :]

        # Transformer encoder
        x = self.transformer(x)  # (batch, seq_len, hidden_size)

        # Use last token for prediction (or pool)
        x = x[:, -1, :]  # (batch, hidden_size)

        # Project to output
        preds = self.output_proj(x)  # (batch, stations * pred_horizons)
        preds = preds.reshape(batch_size, 48, num_stations)  # (batch, 48, stations)

        return preds


def build_transformer_model(config: dict = None):
    """Build Transformer model from config."""
    return TemporalTransformer(config)