"""STGCN-style spatiotemporal graph convolution for PM2.5 forecasting.

The graph convolution is a dense matmul against the (fixed) adjacency matrix,
so no external graph library is needed:

    H' = act( A_hat @ (X W) )

``A_hat`` is a symmetrically normalised adjacency built once from the training
split (distance / correlation / wind-aware), or the identity for the no-graph
ablation.

The model uses the shared residual output head (delta prediction + linear
autoregressive skip) from ``base.ResidualForecaster``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.base import ResidualForecaster, DeltaLinearSkip, zero_init


class STGCN(ResidualForecaster):
    """Spatio-temporal graph convolution network.

    Input: (batch, seq_len, num_stations, num_features)
    Output: (batch, output_horizons, num_stations)
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.name = "stgcn"

        hidden_size = self.config.get("hidden_size", 64)
        dropout = self.config.get("dropout", 0.3)
        input_size = self.config.get("input_size", 1)

        self.register_buffer("adj", torch.eye(self.num_stations))

        # Temporal encoder: per-station 1D conv over time (shared across stations)
        self.temporal_conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)

        # Graph convolution layers
        # 1x1 convs = linear transforms on the channel axis
        self.gc1 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.gc2 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Recurrent refinement over the encoded sequence
        self.gru = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)

        self.fc = zero_init(nn.Linear(hidden_size, self.output_horizons))

        if self.use_lag_skip:
            self.lag_linear = DeltaLinearSkip(
                self.num_stations, self.lag_skip, self.output_horizons)

    def set_adjacency(self, adj: torch.Tensor):
        """Install a fixed adjacency matrix (built on train data only)."""
        self.adj = adj.to(self.adj.dtype)

    # Alias kept for backwards compatibility with the training script.
    set_adjacency.__name__ = "set_adjacency"

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, n_feat = x.shape

        # Temporal conv: (B, T, N, F) -> (B*N, F, T) -> (B*N, H, T)
        xt = x.permute(0, 2, 3, 1).reshape(B * N, n_feat, T)
        h = self.act(self.temporal_conv(xt))               # (B*N, H, T)

        # Graph convolutions over the station axis.
        # h: (B*N, H, T) -> 1x1 conv -> (B, N, H*T) -> A @ -> back.
        A = self.adj
        h = self.act(self.gc1(h))                          # (B*N, H, T)
        hm = h.view(B, N, -1)                              # (B, N, H*T)
        hm = torch.matmul(A, hm)                           # mix over stations
        h = self.dropout(hm).view(B * N, -1, T)            # (B*N, H, T)
        h = self.act(self.gc2(h))
        hm = h.view(B, N, -1)
        hm = torch.matmul(A, hm)
        h = self.dropout(hm).view(B * N, -1, T)            # (B*N, H, T)

        # Recurrent refinement over time: (B*N, H, T) -> (B*N, T, H)
        h = h.permute(0, 2, 1).contiguous()
        out, _ = self.gru(h)                                # (B*N, T, H)
        last = out[:, -1, :]                                # (B*N, H)

        delta = self.fc(last)                               # (B*N, H)
        return delta.reshape(B, N, self.output_horizons)


def build_stgcn_model(config: dict = None):
    """Build STGCN model from config."""
    return STGCN(config)
