"""WindAttentionGraph - temporal attention over wind-aware adjacency.

This implements a time-lagged advection kernel where edge weights depend on
the concentration gradient and wind direction at multiple time lags, not just
the period-mean prevailing direction. This addresses the limitation noted in
the report: "a time-lagged advection kernel... is a different construction
and this study does not test it."

The attention is computed as:
    w_ij(t) = exp(-d_ij^2 / 2σ^2) * max(0, cos(θ_ij - φ_i(t))) * |ΔPM2.5_i(t)|
    
where φ_i(t) is the wind direction at station i at time t, and ΔPM2.5_i(t)
is the local concentration change (proxy for gradient magnitude).

This produces a time-varying, directed adjacency that can be used in a
temporal attention mechanism within the model, or as a sequence of adjacencies
for dynamic graph convolution.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class WindAttentionGraph:
    """Time-lagged wind-aware adjacency with concentration-gradient weighting.
    
    For each time step in the input window, computes a directed adjacency where:
    - Base: Gaussian kernel on distance (as in static distance graph)
    - Direction: max(0, cos(θ_ij - φ_i(t))) -- wind at i points toward j
    - Strength: |ΔPM2.5_i(t)| -- local concentration change magnitude
    
    This is the "advection kernel" the report mentions was not tested.
    """
    
    def __init__(self, coords: np.ndarray, sigma: Optional[float] = None):
        self.coords = coords
        self.N = len(coords)
        self.distances = self._haversine_matrix(coords)
        
        if sigma is None:
            sigma = 5.0  # km
        self.sigma = float(sigma)
        
        # Precompute bearings i -> j
        self.bearings = self._compute_bearings()
        
        # Gaussian distance kernel (time-invariant)
        self.distance_kernel = np.exp(-self.distances ** 2 / (2 * self.sigma ** 2))
        np.fill_diagonal(self.distance_kernel, 0.0)
        
    def _haversine_matrix(self, coords: np.ndarray) -> np.ndarray:
        EARTH_RADIUS_KM = 6371.0
        lat = np.deg2rad(coords[:, 0])[:, None]
        lon = np.deg2rad(coords[:, 1])[:, None]
        dlat = lat - lat.T
        dlon = lon - lon.T
        a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
        return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        
    def _compute_bearings(self) -> np.ndarray:
        """Bearing from i to j in degrees (0=north)."""
        lat = np.deg2rad(self.coords[:, 0])[:, None]
        lon = np.deg2rad(self.coords[:, 1])[:, None]
        dlon = lon - lon.T
        y = np.sin(dlon) * np.cos(lat.T)
        x = np.cos(lat) * np.sin(lat.T) - np.sin(lat.T) * np.cos(lat) * np.cos(dlon)
        theta = (np.degrees(np.arctan2(y, x))) % 360.0
        return theta
    
    def build_adjacency_sequence(self, 
                                  wind_u_seq: np.ndarray,  # (T, N) -- u component over time
                                  wind_v_seq: np.ndarray,  # (T, N) -- v component over time
                                  pm25_seq: np.ndarray) -> np.ndarray:
        """
        Build time-varying adjacency matrices for the input window.
        
        Args:
            wind_u_seq: (T, N) u-component of wind at each station over time
            wind_v_seq: (T, N) v-component of wind at each station over time
            pm25_seq: (T, N) PM2.5 values at each station over time
            
        Returns:
            (T, N, N) adjacency matrices
        """
        T, N = wind_u_seq.shape
        assert N == self.N
        
        # Wind direction at each station and time (where air is GOING, i.e., -u, -v)
        wind_to_u = -wind_u_seq  # (T, N)
        wind_to_v = -wind_v_seq  # (T, N)
        wind_dir = (np.degrees(np.arctan2(wind_to_u, wind_to_v))) % 360.0  # (T, N)
        
        # Concentration change magnitude at each station and time
        # |ΔPM2.5(t)| = |PM2.5(t) - PM2.5(t-1)|
        pm25_change = np.abs(np.diff(pm25_seq, axis=0, prepend=pm25_seq[0:1]))  # (T, N)
        
        adj_seq = np.zeros((T, N, N), dtype=np.float32)
        
        for t in range(T):
            # Wind alignment at this time step
            phi = wind_dir[t]  # (N,)
            delta = np.deg2rad(self.bearings - phi[:, None])  # (N, N)
            align = np.cos(delta)
            align[align < 0] = 0.0  # only downwind edges
            
            # Gradient strength at source stations
            grad_strength = pm25_change[t]  # (N,)
            
            # Adjacency = distance_kernel * alignment * gradient_strength[source]
            adj = self.distance_kernel * align * grad_strength[:, None]
            np.fill_diagonal(adj, 0.0)
            
            # Symmetrically normalize: D^-1/2 A D^-1/2
            adj_seq[t] = self._normalize_adjacency(adj)
            
        return adj_seq
    
    def _normalize_adjacency(self, adj: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
        """Symmetric normalisation D^-1/2 (A + I) D^-1/2."""
        a = adj.copy()
        if add_self_loops:
            np.fill_diagonal(a, a.diagonal() + 1.0)
        d = a.sum(axis=1)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
        return (a * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]).astype(np.float32)


class TemporalWindAttention(nn.Module):
    """Temporal attention over a sequence of wind-aware adjacencies.
    
    This is a model component that takes a sequence of adjacencies (from
    WindAttentionGraph) and computes attention-weighted graph convolutions
    over time, allowing the model to attend to the most relevant past
    adjacency states.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor,           # (B, T, N, H)
                adj_seq: torch.Tensor) -> torch.Tensor:  # (T, N, N)
        """
        Apply temporal attention over graph-convolved features.
        
        For each time step t, we have features x_t and adjacency adj_t.
        We compute attention over the sequence of graph-convolved features.
        """
        B, T, N, H = x.shape
        
        # Apply graph convolution at each time step
        # x_t: (B, N, H), adj_t: (N, N)
        # h_t = adj_t @ x_t  (matmul over N)
        gcn_seq = []
        for t in range(T):
            x_t = x[:, t, :, :]  # (B, N, H)
            adj_t = adj_seq[t]   # (N, N)
            # (B, N, N) @ (B, N, H) -> (B, N, H)
            h_t = torch.matmul(adj_t.unsqueeze(0).expand(B, -1, -1), x_t)
            gcn_seq.append(h_t)
        
        gcn_seq = torch.stack(gcn_seq, dim=1)  # (B, T, N, H)
        
        # Temporal attention: query = last step, key/value = all steps
        # Reshape for multi-head attention: (B*N, T, H)
        gcn_flat = gcn_seq.permute(0, 2, 1, 3).reshape(B * N, T, H)
        
        q = self.q_proj(gcn_flat[:, -1:, :])  # (B*N, 1, H) -- last time step
        k = self.k_proj(gcn_flat)              # (B*N, T, H)
        v = self.v_proj(gcn_flat)              # (B*N, T, H)
        
        # Multi-head attention
        q = q.view(B * N, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B*N, h, 1, d)
        k = k.view(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B*N, h, T, d)
        v = v.view(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B*N, h, T, d)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B*N, h, 1, T)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)  # (B*N, h, 1, d)
        out = out.transpose(1, 2).reshape(B * N, 1, H)  # (B*N, 1, H)
        out = self.out_proj(out)  # (B*N, 1, H)
        
        return out.view(B, N, H)


def build_wind_attention_adjacency(ds, tr_slice: slice, sigma: float = 5.0):
    """Build the time-lagged wind-aware adjacency sequence from training data.
    
    Returns a callable that can build the adjacency sequence for any input window.
    """
    coords = ds["coords"]
    norm_mean = float(ds["norm_mean"])
    norm_std = float(ds["norm_std"])
    
    # De-normalize PM2.5 for gradient computation
    y_train = np.expm1(np.clip(ds["y_full"][tr_slice] * norm_std + norm_mean, -700, 700))
    
    # Wind components are in X_full features 3 and 4 (wind_u, wind_v)
    # Use train slice
    X_train = ds["X_full"][tr_slice]
    wind_u_seq = X_train[:, :, 3]  # (T, N)
    wind_v_seq = X_train[:, :, 4]  # (T, N)
    pm25_seq = y_train             # (T, N)
    
    graph = WindAttentionGraph(coords, sigma=sigma)
    adj_seq = graph.build_adjacency_sequence(wind_u_seq, wind_v_seq, pm25_seq)
    
    return adj_seq  # (T_train, N, N)