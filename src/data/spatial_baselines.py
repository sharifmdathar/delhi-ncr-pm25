#!/usr/bin/env python3
"""Simple spatial baselines, for the leave-station-out comparison.

The leave-station-out result compares the graph models against a no-graph
model that has nothing to work with, which is a strawman.  These baselines
use the same information the graph has -- the *other* stations' recent
observations -- with no training and no model, and they are the honest
comparison for "can a graph help a station the model has never seen?".

Three predictors, all using only data available at forecast time (the input
window), none of them looking at the target hours:

  Network mean (persistence)
      the network-wide mean PM2.5 at the last input hour, held over the
      horizon.  The crudest way to use the regional signal.

  IDW neighbour (persistence)
      the inverse-distance-weighted mean of the OTHER stations' last observed
      values, held over the horizon.

  IDW neighbour (seasonal)
      the same, but each neighbour contributes its own value from 24 hours
      before the target -- so the neighbours' diurnal structure is carried
      over as well as their level.  This is the strongest of the three.

These are deliberately simple.  If the graph models cannot beat them on a
station they have never seen, then whatever the graph appears to be doing is
interpolation that any of these would do.
"""
import numpy as np

from src.data.graphs import haversine_matrix


def _last_observed(y, mask, m, n_st, lookback=200):
    """Most recent observed value at or before hour m-1, per station."""
    last = np.full(n_st, np.nan)
    for back in range(lookback):
        idx = m - 1 - back
        if idx < 0:
            break
        cand = y[idx].astype(np.float64)
        obs = mask[idx] > 0
        take = obs & ~np.isfinite(last)
        last = np.where(take, cand, last)
    return last


def _idw_weights(coords, exclude):
    """Inverse-distance-squared weights over the other stations.

    Returns (W, valid) where W[j, i] is the weight of station i for
    predicting station j, and valid[j] is False if station j has no usable
    neighbour (or if the distances are degenerate).
    """
    D = haversine_matrix(coords)
    n = len(coords)
    W = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            d = D[j, i]
            if d < 1e-6:                     # coincident stations
                d = 1e-6
            W[j, i] = 1.0 / d ** 2
    valid = W.sum(axis=1) > 0
    return W, valid


def spatial_baselines(ds, starts, coords=None, predict_idx=None):
    """Return the three spatial predictors for the given windows.

    ``predict_idx`` restricts which stations are predicted (the held-out ones
    in the leave-station-out experiment); the other stations are used as
    neighbours only.

    Shapes: each predictor returns (n_windows, 48, n_stations).
    """
    y = ds["y_full"]
    mask = ds["mask_y"]
    n_st = y.shape[1]
    if coords is None:
        coords = ds["coords"]
    W, w_valid = _idw_weights(coords, predict_idx)

    n_win = len(starts)
    net_mean = np.zeros((n_win, 48, n_st))
    idw_pers = np.zeros((n_win, 48, n_st))
    idw_seas = np.zeros((n_win, 48, n_st))

    for k, s in enumerate(starts):
        s = int(s)
        m, e = s + 72, s + 72 + 48

        # --- network mean at the last input hour, held over the horizon ---
        obs_last = mask[m - 1] > 0
        vals_last = np.where(obs_last, y[m - 1].astype(np.float64), np.nan)
        nm = np.nanmean(vals_last) if np.isfinite(vals_last).any() else 0.0
        net_mean[k, :, :] = nm

        # --- IDW over the neighbours' last observed values ---
        last = _last_observed(y, mask, m, n_st)
        for j in range(n_st):
            w = W[j]
            num = np.nansum(w * np.nan_to_num(last, nan=0.0))
            den = np.nansum(w * np.isfinite(last))
            idw_pers[k, :, j] = num / den if den > 0 else np.nan

        # --- IDW over the neighbours' value 24 h before each target ---
        for h in range(48):
            tgt = m + h
            src = tgt - 24
            if src < 0:
                continue
            obs = mask[src] > 0
            vals = np.where(obs, y[src].astype(np.float64), np.nan)
            for j in range(n_st):
                w = W[j]
                num = np.nansum(w * np.nan_to_num(vals, nan=0.0))
                den = np.nansum(w * np.isfinite(vals))
                idw_seas[k, h, j] = num / den if den > 0 else np.nan

    return {
        "Network mean (persistence)": net_mean,
        "IDW neighbour (persistence)": idw_pers,
        "IDW neighbour (seasonal)": idw_seas,
    }


def network_mean_series(ds):
    """Network-wide mean PM2.5 at every hour (observed hours only)."""
    y = ds["y_full"]
    mask = ds["mask_y"] > 0
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.where(mask, y, np.nan), axis=1)


def network_mean_channel(ds, feature_index=0):
    """Network-mean PM2.5 per hour in the dataset's normalised units.

    The mean is taken over the stations that had an observation in that hour,
    so an hour where only one station reported does not drag the mean toward
    zero.  Hours with no observation anywhere are set to 0, which is the
    training-split mean in normalised space.

    This is the channel appended to the inputs by the "network mean feature"
    diagnostic: a no-graph model told the regional signal explicitly.
    """
    x = ds["X_full"][:, :, feature_index]
    m = ds["mask_X"][:, :, feature_index] > 0
    with np.errstate(invalid="ignore"):
        num = np.nansum(np.where(m, x, np.nan), axis=1)
        den = m.sum(axis=1)
    return np.where(den > 0, num / np.maximum(den, 1), 0.0).astype(np.float32)
