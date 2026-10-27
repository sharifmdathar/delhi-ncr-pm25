"""GCN+GRU Spatiotemporal model for PM2.5 forecasting.

A clean implementation of the tiny GCN+GRU from scratch, inheriting from
ResidualForecaster (delta prediction + linear autoregressive skip).

Architecture:
- Input projection to hidden size
- Graph convolution: H' = act(A @ (H W)) -- matmul against normalised adjacency
- GRU temporal refinement
- Delta output head via ResidualForecaster

Input: (B, T, N, F)
Output: (B, H, N) -- delta predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.base import ResidualForecaster, DeltaLinearSkip, zero_init


class GCN_GRU(ResidualForecaster):
    """Graph Convolution + GRU temporal model."""

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.name = "gcn_gru"

        hidden_size = self.config.get("hidden_size", 64)
        dropout = self.config.get("dropout", 0.3)
        input_size = self.config.get("input_size", 1)
        num_layers = int(self.config.get("num_layers", 1))

        self.register_buffer("adj", torch.eye(self.num_stations))

        # Input projection + GCN layers
        # 1x1 conv = linear transform on the channel axis
        self.gc1 = nn.Conv1d(input_size, hidden_size, 1)
        self.gc2 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = zero_init(nn.Linear(hidden_size, self.output_horizons))

        if self.use_lag_skip:
            self.lag_linear = DeltaLinearSkip(
                self.num_stations, self.lag_skip, self.output_horizons)

    def set_adjacency(self, adj: torch.Tensor):
        """Install a fixed adjacency matrix (built on train data only)."""
        self.adj = adj.to(self.adj.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F)
        B, T, N, F = x.shape

        # (B, T, N, F) -> (B*N, F, T)
        xt = x.permute(0, 2, 3, 1).reshape(B * N, F, T)

        # GCN layers: graph conv as matmul against A
        h = self.act(self.gc1(xt))              # (B*N, H, T)
        # One graph conv step: A @ (B, N, H*T)
        hm = h.view(B, N, -1)
        hm = torch.matmul(self.adj, hm)
        h = self.dropout(hm).view(B * N, -1, T)
        h = self.act(self.gc2(h))

        # GRU over time: (B*N, T, H)
        h = h.permute(0, 2, 1).contiguous()
        out, _ = self.gru(h)                    # (B*N, T, H)
        last = out[:, -1, :]                    # (B*N, H)

        delta = self.fc(last)                   # (B*N, H)
        return delta.reshape(B, N, self.output_horizons)


def build_gcn_gru_model(config: dict = None):
    """Build GCN+GRU model from config."""
    return GCN_GRU(config)