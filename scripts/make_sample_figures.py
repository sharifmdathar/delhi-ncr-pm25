#!/usr/bin/env python3
"""Generate the four figures the LaTeX report references, from the sample dataset.

The full figures (42 stations, 7 years) live in outputs/ and are regenerable
with scripts/analyze.py and scripts/case_studies.py, which need the frozen
full dataset.  A fresh clone has only the 8-station sample, so this script
produces the same four filenames from the sample so report/report.pdf compiles
without the 36 MB dataset.  The figures are honest about their provenance --
they show the sample, not the full experiment.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_dataset
from src.data.graphs import build_graph

DATA = PROJECT_ROOT / "data" / "sample" / "sample_dataset.npz"
OUT = PROJECT_ROOT / "outputs"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(str(DATA))
    nm, ns = float(ds["norm_mean"]), float(ds["norm_std"])
    y = ds["y_full"]
    mask = ds["mask_y"] > 0
    ts = pd.to_datetime(ds["timestamps"], unit="s")
    vals = np.expm1(np.clip(y * ns + nm, -700, 700))
    net = pd.Series(np.nanmean(np.where(mask, vals, np.nan), axis=1), index=ts)

    # ---- 1-3: three case studies (network-mean traces) -------------------
    events = [
        ("Winter inversion", net[net.index.month.isin([12, 1])].idxmax(),
         "Winter inversion -- sample network mean (8 stations, Nov 2024 - Jan 2025)"),
        ("Diwali", net[(net.index >= "2024-10-28") & (net.index <= "2024-10-31")].idxmax()
         if not net[(net.index >= "2024-10-28") & (net.index <= "2024-10-31")].empty
         else net.index[len(net) // 2],
         "Diwali -- sample network mean (8 stations, Nov 2024 - Jan 2025)"),
        ("Dust", net[(net.index.month.isin([5, 6]))].idxmax()
         if not net[net.index.month.isin([5, 6])].empty
         else net.index[len(net) // 2],
         "Dust episode -- sample network mean (8 stations, Nov 2024 - Jan 2025)"),
    ]
    for name, pk, title in events:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        win = net[max(0, ts.get_loc(pk) - 36): ts.get_loc(pk) + 60]
        ax.plot(win.index, win.values, lw=2.0, color="tab:blue", label="network mean PM2.5")
        ax.axhline(150, color="tab:red", lw=0.8, ls=":", alpha=0.7, label="severe threshold")
        ax.set_title(f"{name} -- sample data (8 stations, Nov 2024 - Jan 2025)")
        ax.set_ylabel("PM2.5 (ug/m3)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        kind = name.lower().replace(" ", "_")
        fig.savefig(OUT / f"fig_case_{kind}.png", dpi=140)
        plt.close(fig)
        print(f"  wrote fig_case_{kind}.png")

    # ---- 4: adjacency introspection ---------------------------------------
    tr = ds["split_indices_train"]
    trs = slice(int(tr[0]), int(tr[-1]) + 1)
    coords = ds["coords"]
    with np.errstate(invalid="ignore"):
        wu = np.nanmean(ds["X_full"][trs][:, :, 3], axis=0)
        wv = np.nanmean(ds["X_full"][trs][:, :, 4], axis=0)
    wu = np.where(np.isnan(wu), np.nanmean(wu), wu)
    wv = np.where(np.isnan(wv), np.nanmean(wv), wv)

    graphs = {
        "distance": build_graph("distance", coords),
        "wind_aware": build_graph("wind_aware", coords, wind_u=wu, wind_v=wv),
        "correlation": build_graph("correlation", coords,
                                   pm25_train=np.expm1(np.clip(
                                       ds["y_full"][trs] * ns + nm, -700, 700))),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (gname, A) in zip(axes, graphs.items()):
        im = ax.imshow(A, cmap="magma")
        ax.set_title(gname)
        ax.set_xlabel("destination station")
        ax.set_ylabel("source station")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Fixed graph structures (sample train split only)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_adjacency_introspection.png", dpi=140)
    plt.close(fig)
    print("  wrote fig_adjacency_introspection.png")

    print(f"\nWrote 4 figures to {OUT}")


if __name__ == "__main__":
    main()