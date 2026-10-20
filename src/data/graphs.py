"""Graph construction utilities for PM2.5 forecasting.

Implements the graph types compared in this work:
- ``distance``       - Gaussian kernel on great-circle distance
- ``correlation``    - pairwise PM2.5 correlation, computed on TRAIN data only
- ``wind_aware``     - directed edges weighted by how well the prevailing
                       wind at station i points towards station j
- ``learned_adaptive`` - Graph WaveNet's softmax(relu(E1 E2^T)), learned in-model
- ``identity``       - no graph (ablation baseline)

Every graph is built from train-split information only.
"""

import numpy as np
import torch
from typing import Optional, Tuple

EARTH_RADIUS_KM = 6371.0


def haversine_matrix(coords: np.ndarray) -> np.ndarray:
    """Great-circle distance matrix in km. coords: (N, 2) as (lat, lon) in degrees."""
    lat = np.deg2rad(coords[:, 0])[:, None]
    lon = np.deg2rad(coords[:, 1])[:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def normalize_adjacency(adj: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """Symmetric normalisation D^-1/2 (A + I) D^-1/2."""
    a = np.asarray(adj, dtype=np.float64).copy()
    if add_self_loops:
        np.fill_diagonal(a, a.diagonal() + 1.0)
    d = a.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    return (a * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]).astype(np.float32)


class DistanceGraph:
    """Gaussian kernel on great-circle distance.

    W_ij = exp(-d_ij^2 / (2 sigma^2)).

    ``sigma`` defaults to 5 km: a physically meaningful pollution-transport
    scale for the Delhi-NCR network (which spans ~40 km).  Using the median
    pairwise distance instead would be dominated by the outlying Gurugram /
    Ghaziabad / Faridabad stations and produce an almost flat kernel.
    """

    DEFAULT_SIGMA_KM = 5.0

    def __init__(self, coords: np.ndarray, sigma: Optional[float] = None):
        self.coords = coords
        self.distances = haversine_matrix(coords)
        if sigma is None:
            sigma = self.DEFAULT_SIGMA_KM
        self.sigma = float(sigma)
        adj = np.exp(-self.distances ** 2 / (2 * self.sigma ** 2))
        np.fill_diagonal(adj, 0.0)
        self.adjacency = adj

    def get_adjacency(self) -> np.ndarray:
        return self.adjacency


class CorrelationGraph:
    """Correlation of PM2.5 across stations, computed on TRAIN data only.

    Missing values are handled by pairwise-complete correlation.
    """

    def __init__(self, data: np.ndarray, threshold: float = 0.3):
        """data: (T, S) train-split PM2.5 in original units."""
        data = np.asarray(data, dtype=np.float64)
        T, S = data.shape
        obs = np.isfinite(data)
        corr = np.eye(S)
        for i in range(S):
            for j in range(i + 1, S):
                both = obs[:, i] & obs[:, j]
                if both.sum() < 24:          # need at least a day of overlap
                    r = 0.0
                else:
                    xi, xj = data[both, i], data[both, j]
                    if xi.std() < 1e-9 or xj.std() < 1e-9:
                        r = 0.0
                    else:
                        r = float(np.corrcoef(xi, xj)[0, 1])
                corr[i, j] = corr[j, i] = r
        adj = np.abs(corr)
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        self.adjacency = adj

    def get_adjacency(self) -> np.ndarray:
        return self.adjacency


class WindAwareGraph:
    """Directed graph weighted by wind alignment.

    For each ordered pair (i, j) the weight is

        w_ij = exp(-d_ij^2 / (2 sigma^2)) * max(0, cos(theta_ij - phi_i))

    where ``theta_ij`` is the bearing from i to j and ``phi_i`` is the
    prevailing wind direction *at station i* (the direction the wind blows
    towards, i.e. the meteorological "to" direction).  The cosine term is
    clipped at zero, so an edge only exists when the wind actually carries
    air from i towards j.

    ``wind_u``/``wind_v`` are the u/v components (m/s) of the prevailing
    wind, computed on the TRAIN split only.
    """

    def __init__(self, coords: np.ndarray, wind_u: np.ndarray, wind_v: np.ndarray,
                 sigma: Optional[float] = None, exponent: float = 2.0):
        self.coords = coords
        N = len(coords)
        self.distances = haversine_matrix(coords)

        if sigma is None:
            sigma = DistanceGraph.DEFAULT_SIGMA_KM
        self.sigma = float(sigma)

        # Prevailing wind direction at each station: the "to" direction.
        # Meteorological u/v point *from* the origin, so the flow direction
        # (where the air is heading) is the opposite of the wind vector.
        speed = np.hypot(wind_u, wind_v)
        # Direction the air travels, as a bearing in degrees (0 = north).
        to_u, to_v = -np.asarray(wind_u), -np.asarray(wind_v)
        phi = (np.degrees(np.arctan2(to_u, to_v))) % 360.0
        self.wind_to_deg = phi
        self.wind_speed = speed

        lat = np.deg2rad(coords[:, 0])[:, None]
        lon = np.deg2rad(coords[:, 1])[:, None]
        dlon = lon - lon.T
        y = np.sin(dlon) * np.cos(lat.T)
        x = np.cos(lat) * np.sin(lat.T) - np.sin(lat.T) * np.cos(lat) * np.cos(dlon)
        theta = (np.degrees(np.arctan2(y, x))) % 360.0     # bearing i -> j

        delta = np.deg2rad(theta - phi[:, None])            # (N, N)
        align = np.cos(delta)
        align[align < 0] = 0.0                              # only downwind edges

        kernel = np.exp(-self.distances ** 2 / (2 * self.sigma ** 2))
        adj = kernel * align
        np.fill_diagonal(adj, 0.0)
        self.adjacency = adj
        self.alignment = align

    def get_adjacency(self) -> np.ndarray:
        return self.adjacency


class IdentityGraph:
    """No graph: identity (used for the no-graph ablation)."""

    def __init__(self, num_nodes: int):
        self.adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    def get_adjacency(self) -> np.ndarray:
        return self.adjacency


def build_graph(graph_type: str,
                coords: np.ndarray,
                pm25_train: Optional[np.ndarray] = None,
                wind_u: Optional[np.ndarray] = None,
                wind_v: Optional[np.ndarray] = None,
                normalize: bool = True,
                **kwargs) -> np.ndarray:
    """Build a graph adjacency matrix.

    Args:
        graph_type: "distance" | "correlation" | "wind_aware" | "learned_adaptive" | "identity"
        coords: (N, 2) lat/lon in degrees
        pm25_train: (T, S) train-split PM2.5 in original units (for correlation)
        wind_u, wind_v: (S,) train-split prevailing wind components (for wind_aware)
        normalize: apply symmetric normalisation

    Returns:
        (N, N) float32 adjacency
    """
    N = len(coords)
    if graph_type == "distance":
        adj = DistanceGraph(coords, sigma=kwargs.get("sigma")).get_adjacency()
    elif graph_type == "correlation":
        if pm25_train is None:
            raise ValueError("pm25_train required for the correlation graph")
        adj = CorrelationGraph(pm25_train, threshold=kwargs.get("threshold", 0.3)).get_adjacency()
    elif graph_type == "wind_aware":
        if wind_u is None or wind_v is None:
            raise ValueError("wind_u/wind_v required for the wind-aware graph")
        adj = WindAwareGraph(coords, wind_u, wind_v, sigma=kwargs.get("sigma")).get_adjacency()
    elif graph_type == "wind_aware_era5":
        # Same construction as wind_aware, but the edge directions come from
        # ERA5 reanalysis instead of the stations' own anemometers.  The two
        # sources sit at opposite ends of the trade-off: ERA5 is complete for
        # every station and spatially smooth (mean cross-station correlation
        # of the u-component 0.98), while the station wind is missing for 9 of
        # 42 stations and noisy where present (cross-station correlation 0.11).
        if wind_u is None or wind_v is None:
            raise ValueError("wind_u/wind_v required for the ERA5 wind-aware graph")
        adj = WindAwareGraph(coords, wind_u, wind_v, sigma=kwargs.get("sigma")).get_adjacency()
    elif graph_type == "learned_adaptive":
        # The adjacency is learned inside GraphWaveNet; start from identity.
        adj = np.eye(N)
    elif graph_type == "identity":
        adj = IdentityGraph(N).get_adjacency()
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    adj = np.asarray(adj, dtype=np.float64)
    if normalize and graph_type != "learned_adaptive":
        adj = normalize_adjacency(adj)
    return adj.astype(np.float32)
