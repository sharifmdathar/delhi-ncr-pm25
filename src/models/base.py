"""Shared building blocks for spatiotemporal forecasters.

Two ideas are central here, both motivated by the failure analysis in
EXPERIMENTS.md:

1. **Delta prediction.**  The network predicts the *change* from the last
   observed value, ``y(t+h) - y(t)``, and the last value is added back at the
   output.  At initialisation the output layer is zero, so the model starts
   exactly at Persistence and can only improve from there.  This removes the
   "collapse to the training mean" failure mode, where a model that must
   predict an absolute level has to learn the marginal distribution before it
   can learn any dynamics.

2. **Linear autoregressive skip.**  A per-station linear map from the last K
   lags to the H horizons is added to the network output.  This is the ridge
   regression solution expressed inside the model, so the network's
   contribution is measured *relative to* a strong linear predictor rather
   than from scratch.  If the network learns nothing useful, the model
   degenerates to ridge, not to a constant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ResidualForecaster(nn.Module):
    """Mixin providing the delta + linear-skip output head.

    Sub-classes implement ``encode(x) -> (B, N, H)`` returning a *delta* per
    station and horizon.  This class turns that into absolute predictions.
    """

    def __init__(self, config: dict = None):
        super().__init__()
        self.config = config or {}
        self.num_stations = self.config.get("num_stations", 42)
        self.output_horizons = self.config.get("output_horizons", 48)
        self.lag_skip = int(self.config.get("lag_skip", 24))
        self.use_lag_skip = bool(self.config.get("use_lag_skip", True))

    # -- helpers ---------------------------------------------------------
    def _last_values(self, x: torch.Tensor) -> torch.Tensor:
        """Last observed value per station: (B, N, 1)."""
        # x: (B, T, N, F) -> take the PM2.5 channel (index 0) at t = T-1
        return x[:, -1, :, 0].unsqueeze(-1)  # (B, N, 1)

    def _lag_window(self, x: torch.Tensor) -> torch.Tensor:
        """Last K values per station: (B, N, K), oldest first."""
        T = x.shape[1]
        K = min(self.lag_skip, T)
        vals = x[:, -K:, :, 0].transpose(1, 2)  # (B, N, K)
        return vals

    # -- to be provided by sub-classes -----------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return delta predictions of shape (B, N, H)."""
        raise NotImplementedError

    # -- output head -----------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, T, N, F).  Returns (B, H, N)."""
        delta = self.encode(x)  # (B, N, H)

        last = self._last_values(x)  # (B, N, 1)
        pred = last + delta  # (B, N, H)

        if self.use_lag_skip:
            lags = self._lag_window(x)  # (B, N, K)
            skip = self.lag_linear(lags)  # (B, N, H)
            pred = pred + skip

        return pred.transpose(1, 2)  # (B, H, N)


class DeltaLinearSkip(nn.Module):
    """Per-station linear map from the last K lags to H horizons (the ridge floor)."""

    def __init__(self, num_stations: int, lag_skip: int, output_horizons: int):
        super().__init__()
        self.lin = nn.Linear(lag_skip, output_horizons)
        # Small init: the linear skip should start near zero so the network
        # output dominates early in training, but it can still learn.
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        self.station_bias = nn.Parameter(torch.zeros(num_stations, output_horizons))

    def forward(self, lags: torch.Tensor) -> torch.Tensor:
        # lags: (B, N, K)
        out = self.lin(lags)  # (B, N, H)
        return out + self.station_bias.unsqueeze(0)


def zero_init(layer: nn.Linear):
    """Zero-initialise a Linear layer's weight and bias."""
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer
