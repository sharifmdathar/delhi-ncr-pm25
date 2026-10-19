#!/usr/bin/env python3
"""Generate an animated AQI map from the trained model's forecasts.

Produces outputs/aqi_map.gif: the network's stations on a Delhi map, with
observed values as filled circles and predicted values as crosses, for a
48-hour forecast window.
"""

import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.data.dataset import load_dataset
from scripts.evaluate import denorm, model_predictions
from scripts.train_all import _to_hn

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "delhi_ncr_pm25_2019_2025.npz"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
OUT = PROJECT_ROOT / "outputs"


def value_to_color(v: float):
    """CPCB-ish scale: green -> yellow -> orange -> red -> maroon."""
    if v <= 50:
        r = int(255 * v / 50)
        return (0, 150 + r // 3, 60 + r)
    if v <= 100:
        r = int(255 * (v - 50) / 50)
        return (255, 190 - r, 60)
    if v <= 200:
        r = int(255 * (v - 100) / 100)
        return (255, 90, 60 - r // 2)
    if v <= 300:
        r = int(120 * (v - 200) / 100)
        return (200, r, 0)
    if v <= 400:
        r = int(120 * (v - 300) / 100)
        return (200, 0, r)
    return (150, 0, 150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint stem; default = best MAE checkpoint")
    ap.add_argument("--window", type=int, default=0,
                    help="index of the test window to animate")
    ap.add_argument("--out", default="outputs/aqi_map.gif")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset(str(DATA_PATH))
    nm, ns = float(ds["norm_mean"]), float(ds["norm_std"])
    n_st = ds["X_full"].shape[1]
    coords = ds["coords"]
    station_ids = ds["station_ids"]

    # pick a checkpoint
    ck_path = None
    if args.checkpoint:
        cands = list(CKPT_DIR.glob(f"{args.checkpoint}*.pt"))
        if cands:
            ck_path = cands[0]
    if ck_path is None:
        cands = [p for p in sorted(CKPT_DIR.glob("*.pt"))
                 if "2019_2024" not in p.stem and "2024_2025" not in p.stem]
        if not cands:
            raise SystemExit("No checkpoints found - train the models first")
        ck_path = cands[0]
    print(f"Using checkpoint {ck_path.name}")

    idx = ds["split_indices_test"]
    starts = np.arange(int(idx[0]), int(idx[-1]) - 72 - 48 + 2)
    # pick a window in the winter peak so the map is interesting
    if args.window == 0:
        s = int(idx[0]) + 24 * 300
    else:
        s = int(starts[args.window])

    p, t, m = model_predictions(ds, str(ck_path), [s], device)
    pred = denorm(np.nan_to_num(p[0], nan=0.0), nm, ns)      # (48, N)
    truth = denorm(np.nan_to_num(t[0], nan=0.0), nm, ns)

    lats, lons = coords[:, 0], coords[:, 1]
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    W, H, M = 1000, 820, 90

    def to_px(lat, lon):
        x = M + (lon - lon_min) / max(lon_max - lon_min, 1e-9) * (W - 2 * M)
        y = M + (lat_max - lat) / max(lat_max - lat_min, 1e-9) * (H - 2 * M)
        return int(x), int(y)

    frames = []
    for tstep in range(0, 48, 3):
        img = Image.new("RGB", (W, H), (250, 250, 250))
        dr = ImageDraw.Draw(img)
        for gx in range(0, W, 50):
            dr.line([(gx, 0), (gx, H)], fill=(225, 225, 225))
        for gy in range(0, H, 50):
            dr.line([(0, gy), (W, gy)], fill=(225, 225, 225))
        for j in range(n_st):
            x, y = to_px(lats[j], lons[j])
            pc = value_to_color(float(pred[tstep, j]))
            tc = value_to_color(float(truth[tstep, j]))
            dr.ellipse([x - 11, y - 11, x + 11, y + 11], fill=tc, outline=(0, 0, 0), width=2)
            dr.line([x - 8, y - 8, x + 8, y + 8], fill=pc, width=3)
            dr.line([x - 8, y + 8, x + 8, y - 8], fill=pc, width=3)
            dr.text((x + 13, y - 6), str(station_ids[j]), fill=(0, 0, 0))
        dr.text((12, 10), f"Delhi-NCR PM2.5 forecast  |  hour +{tstep}", fill=(0, 0, 0))
        dr.text((12, 30), "circle = observed,  cross = predicted", fill=(0, 0, 0))
        dr.text((12, 50), "green<50  yellow<100  orange<200  red<300  dark red  purple>400  µg/m³",
                fill=(0, 0, 0))
        frames.append(img)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   optimize=False, duration=900, loop=0)
    print(f"Wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
