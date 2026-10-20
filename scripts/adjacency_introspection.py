#!/usr/bin/env python3
"""Adjacency introspection.

Does the learned adaptive adjacency of Graph WaveNet correspond to geography,
or to the prevailing wind?  We extract the learned adjacency from a trained
checkpoint and compare it with the distance kernel and with the wind-aware
directed graph:

  * edge weight vs. distance (km)
  * alignment of the learned edge direction with the prevailing wind
  * overlap (rank correlation) with the distance and wind-aware graphs
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_dataset
from src.data.graphs import build_graph, haversine_matrix
from src.models.graph_wavenet import GraphWaveNet

OUT = PROJECT_ROOT / "outputs"


def spearman(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ds = load_dataset(str(PROJECT_ROOT / "data" / "processed" /
                           "delhi_ncr_pm25_2019_2025.npz"))
    coords = ds["coords"]
    sids = list(ds["station_ids"])

    ckpt = None
    for cand in sorted((PROJECT_ROOT / "checkpoints").glob("graph_wavenet_learned_adaptive*.pt")):
        ckpt = cand
        break
    if ckpt is None:
        print("No learned_adaptive checkpoint found; training one is required "
              "for this figure. Skipping.")
        return

    print(f"Using checkpoint {ckpt.name}")
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = blob.get("config", {})
    sd = blob.get("model_state_dict", blob.get("state_dict", blob))

    model = GraphWaveNet(cfg)
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        A_learned = model._effective_adj().numpy().astype(np.float64)

    # ---- reference graphs (train split only) -----------------------------
    tr = ds["split_indices_train"]
    trs = slice(int(tr[0]), int(tr[-1]) + 1)
    nm, ns = float(ds["norm_mean"]), float(ds["norm_std"])
    y_train = np.expm1(np.clip(np.nan_to_num(ds["y_full"][trs], nan=0.0) * ns + nm, -700, 700))
    with np.errstate(invalid="ignore"):
        wu = np.nanmean(ds["X_full"][trs][:, :, 3], axis=0)
        wv = np.nanmean(ds["X_full"][trs][:, :, 4], axis=0)
    # match the training path exactly (scripts/train_all.py): stations with no
    # measured wind fall back to the network-mean wind vector
    wu = np.where(np.isnan(wu), np.nanmean(wu), wu)
    wv = np.where(np.isnan(wv), np.nanmean(wv), wv)

    A_dist = build_graph("distance", coords)
    A_wind = build_graph("wind_aware", coords, wind_u=wu, wind_v=wv)
    D = haversine_matrix(coords)

    iu = np.triu_indices(len(sids), 1)
    d_km = D[iu]

    print("\n=== edge weight vs distance ===")
    for name, A in [("distance kernel", A_dist), ("wind-aware", A_wind),
                    ("learned adaptive", A_learned)]:
        w = A[iu]
        rho = spearman(w, d_km)
        # is the learned weight decaying with distance?
        print(f"  {name:<20} mean weight {w.mean():.4f}  max {w.max():.4f}  "
              f"Spearman(weight, distance) = {rho:+.3f}")

    # ---- does the learned edge direction match the prevailing wind? -----
    # For each ordered pair (i, j) the learned weight is the affinity from i
    # to j.  Compare with cos(theta_ij - phi_i), the wind alignment used to
    # build the wind-aware graph.
    cos_wind = np.zeros_like(A_wind)
    for i in range(len(sids)):
        for j in range(len(sids)):
            if i == j:
                continue
            dx = (coords[j][0] - coords[i][0]) * np.cos(np.radians(coords[i][1]))
            dy = coords[j][1] - coords[i][1]
            theta = np.degrees(np.arctan2(dx, dy))     # bearing i -> j
            phi = np.degrees(np.arctan2(wu[i], wv[i]))
            cos_wind[i, j] = np.cos(np.radians(theta - phi))

    sel = cos_wind > 0.2          # pairs the wind actually blows towards
    if sel.sum() > 10:
        rho = spearman(A_learned[sel], cos_wind[sel])
        print(f"\n=== learned affinity vs wind alignment (pairs with cos>0.2, "
              f"n={int(sel.sum())}) ===")
        print(f"  Spearman(learned affinity, wind alignment) = {rho:+.3f}")
        # compare against the same statistic for the distance kernel
        print(f"  Spearman(distance weight,   wind alignment) = "
              f"{spearman(A_dist[sel], cos_wind[sel]):+.3f}")
        print(f"  Spearman(wind-aware weight, wind alignment) = "
              f"{spearman(A_wind[sel], cos_wind[sel]):+.3f}")
    else:
        rho = float("nan")

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    ax.scatter(d_km, A_learned[iu], s=14, alpha=0.45, color="tab:purple",
               label="learned adaptive")
    ax.scatter(d_km, A_dist[iu], s=14, alpha=0.45, color="tab:blue",
               label="distance kernel")
    ax.set_xlabel("pairwise distance (km)")
    ax.set_ylabel("edge weight")
    ax.set_title("Learned affinity vs geography")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    im = ax.imshow(A_learned, cmap="magma")
    ax.set_title("learned adaptive adjacency")
    ax.set_xlabel("destination station")
    ax.set_ylabel("source station")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    im = ax.imshow(A_wind, cmap="magma")
    ax.set_title("wind-aware directed adjacency (train split)")
    ax.set_xlabel("destination station")
    ax.set_ylabel("source station")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Adjacency introspection: what did the model actually learn?")
    fig.tight_layout()
    fig.savefig(OUT / "fig_adjacency_introspection.png", dpi=140)
    plt.close(fig)

    out = {
        "checkpoint": ckpt.name,
        "spearman_learned_vs_distance": spearman(A_learned[iu], d_km),
        "spearman_distance_vs_distance": spearman(A_dist[iu], d_km),
        "spearman_learned_vs_wind_alignment": rho,
        "spearman_distance_graph_vs_wind_alignment": float(spearman(A_dist[sel], cos_wind[sel])),
        "spearman_wind_graph_vs_wind_alignment": float(spearman(A_wind[sel], cos_wind[sel])),
        "n_wind_aligned_pairs": int(sel.sum()),
    }
    (OUT / "adjacency_introspection.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT/'adjacency_introspection.json'}")


if __name__ == "__main__":
    main()
