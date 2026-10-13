"""Evaluation utilities for forecasting experiments."""

import numpy as np
from typing import Dict, List, Optional
from src.train.metrics import compute_metrics, compute_horizon_metrics


def evaluate_model(
    model,
    dataset: Dict,
    config: Dict,
    device: Optional[str] = None,
) -> Dict:
    """Evaluate a model on the dataset."""
    # This is a placeholder; actual implementation depends on model type
    # and whether it's a neural network or baseline.
    
    # For now, return basic metrics
    return {
        "model_name": config.get("model", "unknown"),
        "metrics": {"mae": 0.0, "rmse": 0.0, "r2": 0.0},
    }


def evaluate_episodes(
    preds: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray = None,
    threshold: float = 150.0,
    duration_hours: int = 6,
) -> Dict:
    """Evaluate predictions specifically during severe episodes.
    
    Episodes are defined as PM2.5 > threshold sustained for duration_hours.
    """
    # Find episodes in targets
    episodes = find_episodes(targets, threshold, duration_hours, mask)
    
    if len(episodes) == 0:
        return {"episode_count": 0, "episode_metrics": {}}
    
    # Extract predictions and targets during episodes
    episode_preds = []
    episode_targets = []
    episode_masks = []
    
    for start, end in episodes:
        episode_preds.append(preds[start:end])
        episode_targets.append(targets[start:end])
        if mask is not None:
            episode_masks.append(mask[start:end])
    
    # Compute metrics
    if episode_preds:
        preds_episodes = np.concatenate(episode_preds, axis=0)
        targets_episodes = np.concatenate(episode_targets, axis=0)
        mask_episodes = np.concatenate(episode_masks, axis=0) if episode_masks else None
        
        metrics = compute_metrics(preds_episodes, targets_episodes, mask_episodes)
        
        return {
            "episode_count": len(episodes),
            "episode_metrics": metrics,
        }
    
    return {"episode_count": 0, "episode_metrics": {}}


def find_episodes(
    data: np.ndarray,
    threshold: float,
    duration_hours: int,
    mask: np.ndarray = None,
) -> List[tuple]:
    """Find sustained episodes where PM2.5 exceeds threshold."""
    episodes = []
    if mask is not None:
        # Only consider observed values
        data_masked = np.where(mask > 0, data, 0)
    else:
        data_masked = data
    
    # Find sustained periods
    in_episode = False
    start_idx = 0
    
    for i in range(len(data_masked)):
        if data_masked[i] > threshold and not in_episode:
            in_episode = True
            start_idx = i
        elif (data_masked[i] <= threshold or mask[i] == 0) and in_episode:
            if i - start_idx >= duration_hours:
                episodes.append((start_idx, i))
            in_episode = False
    
    # Check if episode extends to end
    if in_episode and len(data_masked) - start_idx >= duration_hours:
        episodes.append((start_idx, len(data_masked)))
    
    return episodes