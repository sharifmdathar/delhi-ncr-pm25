#!/usr/bin/env python3
"""Train all spatiotemporal forecasters on the frozen 2019-2025 dataset.

Design notes (see EXPERIMENTS.md for the failure analysis that motivated them):

* Targets are **deltas** from the last observed value, and the model adds the
  last observed value back at the output.  The model therefore starts at
  Persistence and can only improve.
* Every model carries a **linear autoregressive skip** (lag -> horizon), so
  the network's contribution is measured relative to ridge, not from scratch.
* The observation **mask** is appended as an input channel so the network can
  tell a real reading from an imputed one.
* Loss is masked MAE in *normalised* space, but the checkpoint is selected on
  masked MAE in *denormalised* µg/m³, which is the metric we report.

Usage:
    python scripts/train_all.py --models lstm_univariate stgcn
    python scripts/train_all.py --graph wind_aware --models stgcn graph_wavenet
    python scripts/train_all.py --seeds 42 123 456
"""

import sys
from pathlib import Path
import argparse
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import load_dataset
from src.data.graphs import build_graph
from src.models.lstm import build_lstm_model
from src.models.transformer import build_transformer_model
from src.models.stgcn import build_stgcn_model
from src.models.graph_wavenet import build_graph_wavenet_model

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "delhi_ncr_pm25_2019_2025.npz"
CKPT_DIR = PROJECT_ROOT / "checkpoints"


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denorm_np(a, mean, std):
    return np.expm1(np.clip(a * std + mean, -700, 700))


def build_model(name, cfg, n_stations):
    cfg = dict(cfg)
    cfg["num_stations"] = n_stations
    cfg["output_horizons"] = 48
    if name.startswith("lstm_"):
        return build_lstm_model(name, cfg)
    if name == "transformer":
        return build_transformer_model(cfg)
    if name == "stgcn":
        return build_stgcn_model(cfg)
    if name == "graph_wavenet":
        return build_graph_wavenet_model(cfg)
    raise ValueError(name)


def make_graph(ds, graph_type, n_stations):
    """Build the adjacency from TRAIN data only."""
    if graph_type in ("identity", "learned_adaptive"):
        return torch.eye(n_stations)

    coords = ds["coords"]
    train_idx = ds["split_indices_train"]
    tr = slice(int(train_idx[0]), int(train_idx[-1]) + 1)

    norm_mean = float(ds["norm_mean"])
    norm_std = float(ds["norm_std"])
    pm25_train = denorm_np(ds["y_full"][tr], norm_mean, norm_std)  # (T, N)

    wind_u = wind_v = None
    if graph_type == "wind_aware":
        # wind_u / wind_v are feature columns 3 and 4 of X_full.
        # Use nanmean: the raw series has NaN in hours where the station
        # reported no weather.
        with np.errstate(invalid="ignore"):
            wind_u = np.nanmean(ds["X_full"][tr][:, :, 3], axis=0)
            wind_v = np.nanmean(ds["X_full"][tr][:, :, 4], axis=0)
        # Replace any all-NaN station with the network mean
        if np.all(np.isnan(wind_u)):
            wind_u = np.zeros(n_stations)
        else:
            wind_u = np.where(np.isnan(wind_u), np.nanmean(wind_u), wind_u)
        if np.all(np.isnan(wind_v)):
            wind_v = np.zeros(n_stations)
        else:
            wind_v = np.where(np.isnan(wind_v), np.nanmean(wind_v), wind_v)

    adj = build_graph(graph_type, coords, pm25_train=pm25_train,
                      wind_u=wind_u, wind_v=wind_v)
    return torch.from_numpy(adj)


def _to_hn(pred, horizons: int, n_stations: int) -> torch.Tensor:
    """Normalise a model output to (H, N).

    Accepts (B, H, N), (B, N, H) or (H, N).
    """
    if pred.dim() == 4:
        pred = pred.squeeze(0)
    if pred.dim() == 3:
        if pred.shape[0] == horizons and pred.shape[2] == n_stations:
            pass
        elif pred.shape[0] == n_stations and pred.shape[2] == horizons:
            pred = pred.permute(2, 0, 1)
        elif pred.shape[0] == 1:
            pred = pred.squeeze(0)
    return pred


def run(model_name, ds, cfg, args, seed, device):
    set_seed(seed)
    X = torch.from_numpy(ds["X_full"])                 # (T, N, F)
    y = torch.from_numpy(ds["y_full"])                 # (T, N)
    mask_y = torch.from_numpy(ds["mask_y"])            # (T, N)
    norm_mean = float(ds["norm_mean"])
    norm_std = float(ds["norm_std"])
    n_stations = X.shape[1]
    n_feat = X.shape[2]

    tr = ds["split_indices_train"]
    va = ds["split_indices_val"]
    train_starts = np.arange(int(tr[0]), int(tr[-1]) - 72 - 48 + 2)
    # Evenly spaced validation windows so checkpoint selection is not biased
    # towards the start of the split (which would be the winter peak).
    val_all = np.arange(int(va[0]), int(va[-1]) - 72 - 48 + 2)
    if len(val_all) > args.val_windows:
        sel = np.linspace(0, len(val_all) - 1, args.val_windows).astype(int)
        val_starts = val_all[sel]
    else:
        val_starts = val_all

    model = build_model(model_name, cfg, n_stations).to(device)
    if model_name in ("stgcn", "graph_wavenet"):
        model.set_adjacency(make_graph(ds, args.graph, n_stations).to(device))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{model_name} seed={seed} graph={args.graph}] params={n_params/1e3:.1f}K")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    def make_batch(starts):
        """Stack a list of window starts into one batch tensor.

        Returns xb (B, 72, N, F+1), yb (B, 48, N), ymask (B, 48, N).
        """
        idx = np.asarray(starts, dtype=np.int64)
        b = len(idx)
        xs, ys, ms = [], [], []
        for s in idx:
            s = int(s)
            m, e = s + 72, s + 72 + 48
            xb = X[s:m]                            # (72, N, F)
            mb = mask_y[s:m]                       # (72, N)
            yb = y[m:e]                            # (48, N)
            ymask = mask_y[m:e]                    # (48, N)
            # Missing hours are NaN; replace with 0 (= the train mean in
            # normalised space) and let the mask channel carry the
            # "was this observed?" signal.
            xb = torch.nan_to_num(xb, nan=0.0)
            yb = torch.nan_to_num(yb, nan=0.0)
            xs.append(torch.cat([xb, mb.unsqueeze(-1)], dim=-1))
            ys.append(yb)
            ms.append(ymask)
        xb = torch.stack(xs).to(device)
        yb = torch.stack(ys).to(device)
        ymask = torch.stack(ms).to(device)
        return xb, yb, ymask

    @torch.no_grad()
    def evaluate(starts, max_windows=None):
        model.eval()
        se = ae = 0.0
        n = 0
        sel = starts if max_windows is None else starts[:max_windows]
        for i in range(0, len(sel), args.batch_size):
            chunk = sel[i:i + args.batch_size]
            xb, yb, ymask = make_batch(chunk)
            with torch.no_grad():
                pred = _to_hn(model(xb), 48, n_stations)
            if pred.shape[1:] != (48, n_stations):
                continue
            p = denorm_np(np.nan_to_num(pred.detach().cpu().numpy(), nan=0.0), norm_mean, norm_std)
            t = denorm_np(np.nan_to_num(yb.detach().cpu().numpy(), nan=0.0), norm_mean, norm_std)
            m = ymask.detach().cpu().numpy() > 0
            err = np.abs(p - t)[m]
            ae += float(err.sum())
            se += float((err ** 2).sum())
            n += err.size
        model.train()
        return (ae / max(n, 1)), (se / max(n, 1)) ** 0.5

    best = float("inf")
    best_epoch = -1
    patience = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        perm = np.random.permutation(train_starts)
        model.train()
        tot = 0.0
        nb = 0
        for i in range(0, len(perm), args.batch_size):
            batch_starts = perm[i:i + args.batch_size]
            opt.zero_grad()
            # Each window is a separate forward (variable station count is fixed,
            # so we could batch, but the loss is masked and windows are short)
            xb, yb, ymask = make_batch(batch_starts)
            pred = _to_hn(model(xb), 48, n_stations)
            # Masked MAE in normalised space
            diff = torch.abs(pred - yb) * ymask
            loss = diff.sum() / (ymask.sum() + 1e-8)
            loss.backward()
            tot += float(loss.detach())
            nb += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_mae, val_rmse = evaluate(val_starts, max_windows=args.val_windows)
        sched.step(val_mae)
        print(f"  epoch {epoch+1:3d}  train(norm MAE)={tot/max(nb,1):.4f}  "
              f"val(µg/m³ MAE)={val_mae:.2f}  RMSE={val_rmse:.2f}  "
              f"[{time.time()-t0:.0f}s]")

        if val_mae < best - 1e-4:
            best = val_mae
            best_epoch = epoch + 1
            patience = 0
            tag = f"{model_name}_{args.graph}_s{seed}"
            torch.save({"state_dict": model.state_dict(),
                        "model": model_name,
                        "config": cfg, "graph": args.graph, "seed": seed,
                        "n_stations": n_stations, "val_mae_ugm3": best,
                        "epoch": best_epoch},
                       CKPT_DIR / f"{tag}.pt")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop at epoch {epoch+1}")
                break

    print(f"  BEST {model_name} seed={seed} graph={args.graph}: "
          f"val MAE={best:.2f} µg/m³ @ epoch {best_epoch}")
    return {"model": model_name, "graph": args.graph, "seed": seed,
            "val_mae_ugm3": best, "epoch": best_epoch, "params": n_params}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["lstm_univariate", "lstm_multivariate",
                             "transformer", "stgcn", "graph_wavenet"])
    ap.add_argument("--graph", default="distance",
                    choices=["distance", "correlation", "wind_aware",
                             "learned_adaptive", "identity"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--val_windows", type=int, default=400)
    ap.add_argument("--out", default="outputs/train_log.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(str(DATA_PATH))
    print(f"Dataset: X={ds['X_full'].shape}, y={ds['y_full'].shape}, "
          f"stations={ds['station_ids'].shape[0]}")

    results = []
    for seed in args.seeds:
        for name in args.models:
            cfg = {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.3,
                "input_size": ds["X_full"].shape[2] + 1,   # +1 for the mask channel
                "lag_skip": 24,
                "use_lag_skip": True,
            }
            if name == "transformer":
                cfg["num_heads"] = 4
                cfg["seq_len"] = 72
                cfg["pred_horizons"] = 48
            if name == "graph_wavenet":
                cfg["graph_type"] = args.graph
                cfg["blocks"] = 2
                cfg["layers"] = 4
            print(f"\n=== {name} | graph={args.graph} | seed={seed} ===")
            try:
                results.append(run(name, ds, cfg, args, seed, device))
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  FAILED: {e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = []
    if out.exists():
        try:
            prev = json.loads(out.read_text())
        except Exception:
            prev = []
    out.write_text(json.dumps(prev + results, indent=2))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
