"""GATv2 Spatiotemporal Graph Convolutional Network for PM2.5 forecasting.

Implements GATv2 (Brody et al., "How Attentive are Graph Attention Networks?")
with a temporal encoder, replacing the fixed adjacency matmul with learned
attention over neighbor features. This tests whether learned spatial attention
beats the static/wind-aware/distance graphs in the main ablation.

Architecture:
- Temporal 1D conv per station (shared)
- GATv2 attention layers with multi-head attention
- GRU temporal refinement
- Residual output head (delta prediction + linear skip)

Input: (B, T, N, F)
Output: (B, H, N) -- delta predictions via ResidualForecaster.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.base import ResidualForecaster, DeltaLinearSkip, zero_init


class GATv2Layer(nn.Module):
    """GATv2 attention layer with multi-head support.

    Computes attention: alpha_ij = softmax_j( a^T LeakyReLU(W [h_i || h_j]) )
    Then aggregates: h_i' = sum_j alpha_ij h_j

    Multi-head: concat heads on output, or average for last layer.
    """

    def __init__(self, in_features: int, out_features: int, num_heads: int = 1,
                 dropout: float = 0.3, negative_slope: float = 0.2, concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout

        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.head_dim = out_features // num_heads

        # Per-head linear projections
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention vector for each head: (num_heads, 2 * head_dim, 1)
        self.a = nn.Parameter(torch.randn(num_heads, 2 * self.head_dim, 1) * 0.1)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, H_in) - node features
        adj: (N, N) - adjacency matrix (1 where edge exists, 0 otherwise)
        Returns: (B, N, H_out)
        """
        B, N, H_in = x.shape
        H_out = self.out_features
        hd = self.head_dim

        # Linear projection: (B, N, H_out)
        Wh = self.W(x)  # (B, N, H_out)

        # Reshape to multi-head: (B, N, num_heads, head_dim)
        Wh = Wh.view(B, N, self.num_heads, hd)

        # For each head, compute attention over neighbors
        # Wh: (B, N, h, d) -> expand for pairwise: (B, N, N, h, d)
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1, -1)  # (B, N, N, h, d)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1, -1)  # (B, N, N, h, d)
        Wh_cat = torch.cat([Wh_i, Wh_j], dim=-1)  # (B, N, N, h, 2*d)

        # Wh_cat: (B, N, N, h, 2*d) -> (B, h, N, N, 2*d) @ (h, 2*d) -> (B, h, N, N)
        Wh_cat = Wh_cat.permute(0, 3, 1, 2, 4)  # (B, h, N, N, 2*d)
        # Per-head attention: e[b,h,i,j] = sum_d Wh_cat[b,h,i,j,d] * a[h,d]
        e = torch.einsum("bhijd,hd->bhij", Wh_cat, self.a.squeeze(-1))  # (B, h, N, N)
        e = self.leaky_relu(e)

        # Mask by adjacency: -inf where no edge, then softmax over neighbors
        e = e.masked_fill(adj.unsqueeze(0).unsqueeze(0) == 0, -1e9)
        alpha = F.softmax(e, dim=-1)  # (B, h, N, N)

        # Dropout on attention weights
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # Aggregate: (B, h, N, N) @ (B, N, h, d) -> (B, h, N, d)
        Wh_for_agg = Wh.permute(0, 2, 1, 3)  # (B, h, N, d)
        h_out = torch.matmul(alpha, Wh_for_agg)  # (B, h, N, d)
        h_out = h_out.permute(0, 2, 1, 3).reshape(B, N, H_out)  # (B, N, H_out)

        if self.concat:
            h_out = F.elu(h_out)

        return h_out


class GATv2STGCN(ResidualForecaster):
    """GATv2-based spatiotemporal model.

    Temporal conv -> GATv2 attention -> GRU temporal refinement -> delta head
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.name = "gatv2_stgcn"

        hidden_size = self.config.get("hidden_size", 64)
        dropout = self.config.get("dropout", 0.3)
        input_size = self.config.get("input_size", 1)
        num_layers = int(self.config.get("num_gat_layers", 2))
        num_heads = int(self.config.get("num_heads", 4))
        negative_slope = self.config.get("negative_slope", 0.2)

        self.register_buffer("adj", torch.eye(self.num_stations))

        # Temporal encoder: 1D conv over time (shared across stations)
        self.temporal_conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)

        # GATv2 layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_feat = hidden_size if i == 0 else hidden_size
            self.gat_layers.append(GATv2Layer(
                in_features=in_feat,
                out_features=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                negative_slope=negative_slope,
                concat=(i < num_layers - 1)
            ))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Recurrent refinement over the encoded sequence
        self.gru = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)

        self.fc = zero_init(nn.Linear(hidden_size, self.output_horizons))

        if self.use_lag_skip:
            self.lag_linear = DeltaLinearSkip(
                self.num_stations, self.lag_skip, self.output_horizons)

    def set_adjacency(self, adj: torch.Tensor):
        """Install a fixed adjacency matrix (used as sparsity mask for attention)."""
        self.adj = adj.to(self.adj.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, n_feat = x.shape

        # Temporal conv: (B, T, N, F) -> (B*N, F, T) -> (B*N, H, T)
        xt = x.permute(0, 2, 3, 1).reshape(B * N, n_feat, T)
        h = self.act(self.temporal_conv(xt))               # (B*N, H, T)

        # Use the full temporal sequence for GRU: (B*N, H, T) -> (B*N, T, H)
        h_seq = h.permute(0, 2, 1).contiguous()  # (B*N, T, H)

        # GATv2 attention over spatial graph at each time step
        # We apply attention at the last time step, then GRU over full sequence
        # Actually, better: apply GAT at each time step, then GRU
        # But GAT is expensive - apply at last step then GRU for temporal
        
        # Last time step features: (B*N, H) -> (B, N, H)
        h_last = h_seq[:, -1, :].view(B, N, -1)

        # GATv2 attention layers over the station graph
        A = self.adj
        h = h_last
        for layer in self.gat_layers:
            h = layer(h, A)

        h = self.dropout(h)

        # GRU over time with GAT output as initial hidden state? 
        # Simpler: just use the temporal conv sequence for GRU
        # and the GAT output as an additional feature
        # Actually, let's run GRU over the temporal sequence, 
        # initialized with GAT output at last step
        
        # h_seq: (B*N, T, H)
        out, _ = self.gru(h_seq)        # (B*N, T, H)
        last = out[:, -1, :]            # (B*N, H)

        # Combine GAT spatial features with GRU temporal features
        # h: (B, N, H) -> (B*N, H)
        h_flat = h.view(B * N, -1)
        combined = last + h_flat  # residual connection

        delta = self.fc(combined)           # (B*N, H)
        return delta.reshape(B, N, self.output_horizons)


def build_gatv2_stgcn_model(config: dict = None):
    """Build GATv2-STGCN model from config."""
    return GATv2STGCN(config)