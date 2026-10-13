"""Metrics for evaluating PM2.5 forecasting models."""

import numpy as np
from typing import Dict, Tuple, Optional


def masked_mae(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray = None) -> float:
    """Compute masked MAE."""
    if mask is None:
        mask = np.ones_like(targets)
    
    errors = np.abs(preds - targets)
    masked_errors = errors * mask
    total_error = np.sum(masked_errors)
    total_mask = np.sum(mask)
    
    return total_error / (total_mask + 1e-8)


def masked_rmse(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray = None) -> float:
    """Compute masked RMSE."""
    if mask is None:
        mask = np.ones_like(targets)
    
    errors = (preds - targets) ** 2
    masked_errors = errors * mask
    total_error = np.sum(masked_errors)
    total_mask = np.sum(mask)
    
    return np.sqrt(total_error / (total_mask + 1e-8))


def masked_mape(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray = None) -> float:
    """Compute masked MAPE."""
    if mask is None:
        mask = np.ones_like(targets)
    
    # Avoid division by zero
    errors = np.abs((preds - targets) / (targets + 1e-8))
    masked_errors = errors * mask
    total_error = np.sum(masked_errors)
    total_mask = np.sum(mask)
    
    return total_error / (total_mask + 1e-8) * 100.0


def masked_r2(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray = None) -> float:
    """Compute masked R-squared."""
    if mask is None:
        mask = np.ones_like(targets)
    
    # Mean of targets (masked)
    target_sum = np.sum(targets * mask)
    mask_sum = np.sum(mask)
    target_mean = target_sum / (mask_sum + 1e-8)
    
    # Total sum of squares
    tss = np.sum(((targets - target_mean) ** 2) * mask)
    # Residual sum of squares
    rss = np.sum(((preds - targets) ** 2) * mask)
    
    return 1.0 - (rss / (tss + 1e-8))


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray = None,
) -> Dict[str, float]:
    """Compute all metrics and return as dictionary."""
    return {
        "mae": masked_mae(preds, targets, mask),
        "rmse": masked_rmse(preds, targets, mask),
        "mape": masked_mape(preds, targets, mask),
        "r2": masked_r2(preds, targets, mask),
    }


def compute_horizon_metrics(
    preds: np.ndarray,      # (batch, horizons, stations)
    targets: np.ndarray,    # (batch, horizons, stations)
    mask: np.ndarray = None, # (batch, horizons, stations)
) -> Dict[str, Dict[str, float]]:
    """Compute metrics per horizon."""
    horizons = preds.shape[1]
    results = {}
    
    for h in range(horizons):
        pred_h = preds[:, h, :]
        target_h = targets[:, h, :]
        mask_h = mask[:, h, :] if mask is not None else None
        
        metrics = compute_metrics(pred_h, target_h, mask_h)
        results[f"horizon_{h+1}h"] = metrics
    
    return results