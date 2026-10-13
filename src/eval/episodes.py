"""Utilities for episode-based analysis."""

import numpy as np
from typing import Dict, List, Optional, Tuple


def find_episodes(
    data: np.ndarray,
    threshold: float,
    duration_hours: int,
    mask: np.ndarray = None,
) -> List[Tuple[int, int]]:
    """Find sustained episodes where data exceeds threshold for at least duration_hours.
    
    Args:
        data: 1D array of PM2.5 values
        threshold: PM2.5 threshold for defining an episode (µg/m³)
        duration_hours: Minimum episode duration in hours
        mask: Optional mask indicating observed values (1=observed, 0=missing)
        
    Returns:
        List of (start_idx, end_idx) tuples for each episode
    """
    episodes = []
    
    # Ensure data is 1D
    data = data.flatten()
    
    # If mask is provided, only consider observed values
    if mask is not None:
        mask = mask.flatten()
        data_masked = np.where(mask > 0, data, 0)
    else:
        data_masked = data
    
    # Find episodes
    in_episode = False
    start_idx = 0
    
    for i in range(len(data_masked)):
        if data_masked[i] > threshold and not in_episode:
            in_episode = True
            start_idx = i
        elif (data_masked[i] <= threshold or (mask is not None and mask[i] == 0)) and in_episode:
            # End of episode
            if i - start_idx >= duration_hours:
                episodes.append((start_idx, i))
            in_episode = False
    
    # Check if episode extends to end
    if in_episode and len(data_masked) - start_idx >= duration_hours:
        episodes.append((start_idx, len(data_masked)))
    
    return episodes


def extract_episodes(
    preds: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 150.0,
    duration_hours: int = 6,
) -> Dict:
    """Extract episodes from predictions and targets for analysis.
    
    Returns:
        Dictionary with episode information and metrics
    """
    # Flatten arrays if needed
    if preds.ndim > 1:
        preds = preds.flatten()
    if targets.ndim > 1:
        targets = targets.flatten()
    if mask is not None and mask.ndim > 1:
        mask = mask.flatten()
    
    # Find episodes in targets
    episodes = find_episodes(targets, threshold, duration_hours, mask)
    
    if len(episodes) == 0:
        return {
            "episode_count": 0,
            "episode_metrics": {},
            "episode_times": [],
        }
    
    # Extract episode data
    episode_preds = []
    episode_targets = []
    episode_masks = []
    episode_times = []
    
    for start, end in episodes:
        episode_preds.append(preds[start:end])
        episode_targets.append(targets[start:end])
        if mask is not None:
            episode_masks.append(mask[start:end])
        # Simulate time indices (would use real timestamps in production)
        episode_times.append((start, end))
    
    # Compute metrics
    from src.train.metrics import compute_metrics
    
    if episode_preds:
        preds_episodes = np.concatenate(episode_preds, axis=0)
        targets_episodes = np.concatenate(episode_targets, axis=0)
        mask_episodes = np.concatenate(episode_masks, axis=0) if episode_masks else None
        
        metrics = compute_metrics(preds_episodes, targets_episodes, mask_episodes)
        
        return {
            "episode_count": len(episodes),
            "episode_metrics": metrics,
            "episode_times": episode_times,
            "avg_pm25": np.mean(targets_episodes[mask_episodes] if mask_episodes is not None else targets_episodes),
            "max_pm25": np.max(targets_episodes[mask_episodes] if mask_episodes is not None else targets_episodes),
        }
    
    return {
        "episode_count": 0,
        "episode_metrics": {},
        "episode_times": [],
    }