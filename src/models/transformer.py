"""Plain Transformer for temporal PM2.5 forecasting.

Input: (batch, seq_len, num_stations, num_features)
Treats stations as channels with temporal attention.

Uses the shared residual output head (delta prediction + linear
autoregressive skip) from ``base.ResidualForecaster``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.base import ResidualForecaster, DeltaLinearSkip, zero_init


class TemporalTransformer(ResidualForecaster):
    """Plain Transformer for temporal forecasting.

    Architecture:
    - Input projection to hidden_size
    - Learned positional encoding
    - Stacked Transformer encoder layers
    - Output projection to stations x horizons (delta)
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.name = "transformer"

        hidden_size = self.config.get("hidden_size", 64)
        num_layers = self.config.get("num_layers", 2)
        num_heads = self.config.get("num_heads", 4)
        dropout = self.config.get("dropout", 0.3)
        input_size = self.config.get("input_size", 1)
        seq_len = self.config.get("seq_len", 72)

        self.input_proj = nn.Linear(input_size * self.num_stations, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_size) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = zero_init(
            nn.Linear(hidden_size, self.num_stations * self.output_horizons))

        if self.use_lag_skip:
            self.lag_linear = DeltaLinearSkip(
                self.num_stations, self.lag_skip, self.output_horizons)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, n_feat = x.shape
        x_flat = x.reshape(B, T, N * n_feat)
        h = self.input_proj(x_flat)
        h = h + self.pos_embed[:, :T, :]
        h = self.transformer(h)
        h = h[:, -1, :]                       # last token
        delta = self.output_proj(h)           # (B, N*H)
        return delta.reshape(B, N, self.output_horizons)


def build_transformer_model(config: dict = None):
    """Build Transformer model from config."""
    return TemporalTransformer(config)
