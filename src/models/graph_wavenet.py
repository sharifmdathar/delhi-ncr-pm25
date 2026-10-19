"""Graph WaveNet for PM2.5 forecasting.

Simplified but faithful re-implementation of the Graph WaveNet idea
(Zhan et al., IJCAI 2020):

- Dilated causal temporal convolutions over the time axis.
- Graph convolution over the station axis, implemented as a dense matmul
  against a fixed adjacency (distance / correlation / wind-aware), or a
  *learned adaptive* adjacency ``softmax(relu(E1 E2^T))``.
- Residual + skip connections across dilated blocks.

Uses the shared residual output head (delta prediction + linear
autoregressive skip) from ``base.ResidualForecaster``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.base import ResidualForecaster, DeltaLinearSkip, zero_init


class GraphWaveNet(ResidualForecaster):
    """Graph WaveNet.

    Input: (batch, seq_len, num_stations, num_features)
    Output: (batch, output_horizons, num_stations)
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.name = "graph_wavenet"

        hidden_size = self.config.get("hidden_size", 64)
        dropout = self.config.get("dropout", 0.3)
        input_size = self.config.get("input_size", 1)
        self.graph_type = self.config.get("graph_type", "wind_aware")
        self.blocks = int(self.config.get("blocks", 2))
        self.layers = int(self.config.get("layers", 4))
        self.adapt_dim = int(self.config.get("adapt_dim", 10))
        # Width of the dilated temporal convolutions. The original paper uses
        # residual/dilation channels of 32 with a wider skip and end path; we
        # keep that here because the convs dominate the cost.
        self.conv_channels = int(self.config.get("conv_channels", 32))

        self.register_buffer("adj", torch.eye(self.num_stations))

        C = self.conv_channels
        self.start_conv = nn.Conv1d(input_size, C, kernel_size=1)

        # Dilated causal convolutions
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        receptive_field = 1
        for _ in range(self.blocks):
            dilation = 1
            for _ in range(self.layers):
                self.filter_convs.append(nn.Conv1d(C, C, 2, dilation=dilation))
                self.gate_convs.append(nn.Conv1d(C, C, 2, dilation=dilation))
                self.residual_convs.append(nn.Conv1d(C, C, 1))
                self.skip_convs.append(nn.Conv1d(C, C, 1))
                # 1x1 conv = linear transform on the channel axis, which is
                # the layout we already have (B*N, H, T).
                self.gconv_convs.append(nn.Conv1d(C, C, 1))
                self.bn.append(nn.BatchNorm1d(C))
                receptive_field += dilation
                dilation *= 2
        self.receptive_field = receptive_field

        self.end_conv_1 = nn.Conv1d(C, hidden_size, 1)
        self.end_conv_2 = zero_init(nn.Conv1d(hidden_size, self.output_horizons, 1))
        self.dropout = nn.Dropout(dropout)

        # Learned adaptive adjacency
        if self.graph_type == "learned_adaptive":
            self.nodevec1 = nn.Parameter(torch.randn(self.num_stations, self.adapt_dim) * 0.1)
            self.nodevec2 = nn.Parameter(torch.randn(self.adapt_dim, self.num_stations) * 0.1)

        if self.use_lag_skip:
            self.lag_linear = DeltaLinearSkip(
                self.num_stations, self.lag_skip, self.output_horizons)

    def set_adjacency(self, adj: torch.Tensor):
        """Install a fixed adjacency matrix (built on train data only)."""
        self.adj = adj.to(self.adj.dtype)

    def _effective_adj(self) -> torch.Tensor:
        if self.graph_type == "learned_adaptive":
            return F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)
        return self.adj

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, n_feat = x.shape

        # (B, T, N, F) -> (B*N, F, T)
        xt = x.permute(0, 2, 3, 1).reshape(B * N, n_feat, T)
        h = self.start_conv(xt)  # (B*N, H, T)

        # Left-pad so the causal convs see a consistent length
        if h.size(2) < self.receptive_field:
            h = F.pad(h, (self.receptive_field - h.size(2), 0))

        A = self._effective_adj()
        skip = 0.0
        for i in range(self.blocks * self.layers):
            residual = h
            filt = torch.tanh(self.filter_convs[i](h))
            gate = torch.sigmoid(self.gate_convs[i](h))
            h = filt * gate

            # Graph convolution over the station axis, in (B, N, H, T) layout.
            # The linear transform is applied on the H axis with an einsum so
            # no permute copies are needed.
            # Mix over stations as a BLAS matmul on the N axis:
            # (B, N, H*T) -> A @ that -> back to (B*N, H, T)
            hg = h.view(B, N, -1)                          # (B, N, H*T)
            hg = torch.matmul(A, hg)                       # (B, N, H*T)
            h = hg.view(B * N, -1, h.size(2))              # (B*N, H, T)
            h = self.gconv_convs[i](h)                     # 1x1 conv on channels

            s = self.skip_convs[i](h)
            skip = s if not isinstance(skip, torch.Tensor) else skip[..., -s.size(2):] + s

            h = self.residual_convs[i](h)
            h = self.bn[i](h)
            if h.size(2) != residual.size(2):
                residual = residual[..., -h.size(2):]
            h = self.dropout(h + residual)

        out = F.relu(skip)
        out = F.relu(self.end_conv_1(out))
        out = self.end_conv_2(out)          # (B*N, H_out, T)
        out = out[..., -1]                  # (B*N, H_out)

        return out.reshape(B, N, self.output_horizons)


def build_graph_wavenet_model(config: dict = None):
    """Build Graph WaveNet model from config."""
    return GraphWaveNet(config)
