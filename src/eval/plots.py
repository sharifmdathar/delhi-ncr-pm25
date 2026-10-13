"""Plotting utilities for PM2.5 forecasting evaluation."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Optional, List


def plot_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    station_ids: Optional[List[int]] = None,
    timestamp_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """Plot predictions vs actuals for selected stations.
    
    Args:
        preds: (N, 48, 70) predictions
        targets: (N, 48, 70) targets
        station_ids: List of station IDs to plot
        timestamp_labels: Labels for time axis
        save_path: Path to save plot
    """
    if station_ids is None:
        station_ids = list(range(min(5, preds.shape[2])))
    
    fig, axes = plt.subplots(len(station_ids), 1, figsize=(12, 3 * len(station_ids)))
    if len(station_ids) == 1:
        axes = [axes]
    
    for idx, station in enumerate(station_ids):
        ax = axes[idx]
        
        # Take first sample for visualization
        pred_first = preds[0, :, station]
        target_first = targets[0, :, station]
        
        time_steps = np.arange(len(pred_first))
        
        ax.plot(time_steps, target_first, label="Actual", marker="o", markersize=3)
        ax.plot(time_steps, pred_first, label="Predicted", linestyle="--", marker="x", markersize=3)
        
        ax.set_title(f"Station {station_ids[idx]} - Predictions vs Actual")
        ax.set_xlabel("Time Step (hours ahead)")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_spatial_errors(
    errors: np.ndarray,  # (70,) per-station errors
    coords: np.ndarray,  # (70, 2) lat/lon
    save_path: Optional[str] = None,
):
    """Create a scatter plot of spatial errors."""
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(
        coords[:, 1], coords[:, 0],  # lon, lat
        c=errors,
        cmap="Reds",
        s=100,
        alpha=0.8,
        edgecolors="black",
    )
    
    plt.colorbar(scatter, label="MAE (µg/m³)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Spatial Error Distribution - Per Station MAE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Spatial plot saved to {save_path}")
    
    return plt.gcf()


def plot_adjacency(
    adj: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot adjacency matrix."""
    plt.figure(figsize=(8, 8))
    plt.imshow(adj, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Edge Weight")
    plt.title("Graph Adjacency Matrix")
    plt.xlabel("Station Index")
    plt.ylabel("Station Index")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Adjacency plot saved to {save_path}")
    
    return plt.gcf()