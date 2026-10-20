#!/usr/bin/env python3
"""Add a network-mean PM2.5 input feature (a diagnostic, not a product).

The leave-station-out experiment showed that a graph neural network is beaten
by a simple inverse-distance-weighted average of the neighbours when it has
to predict a station it has never seen.  The obvious explanation is that the
only useful spatial information is the regional signal, and a graph is a
blunt way to get at it.

This diagnostic tests that directly.  A no-graph model is given the
network-wide mean PM2.5 as an extra input channel.  The network mean is
computed from the dataset's own PM2.5 feature, so no rebuild is needed; it is
appended to the input at batch-construction time.

  * if a no-graph model with the network mean matches the graph models, then
    the graph contributes nothing beyond the regional signal;
  * if it does not, the graph is contributing something the regional mean
    does not, and the leave-station-out result has a different explanation.

Usage:
    python scripts/train_all.py --models stgcn --graph identity \
        --network_mean_feature --seeds 42 123 456
"""
import sys
from pathlib import Path
import argparse

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def network_mean_channel(ds, feature_index=0):
    """Network-mean PM2.5 per hour, in the dataset's normalised units.

    Returns (T,) -- the mean over the stations that had an observation in
    that hour, so a hour where only one station reported does not drag the
    mean toward zero.  Missing hours are filled with the train-split mean,
    which is 0 in normalised space.
    """
    x = ds["X_full"][:, :, feature_index]
    m = ds["mask_X"][:, :, feature_index] > 0
    with np.errstate(invalid="ignore"):
        num = np.nansum(np.where(m, x, np.nan), axis=1)
        den = m.sum(axis=1)
    out = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["stgcn"])
    ap.add_argument("--graph", default="identity")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--val_windows", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--split_mode", default="fixed")
    ap.add_argument("--fold", default=None)
    ap.add_argument("--network_mean_feature", action="store_true")
    ap.add_argument("--out", default="outputs/train_log.json")
    args = ap.parse_args()

    from src.data.dataset import load_dataset
    from src.data.graphs import build_graph
    from src.data.splits import get_split
    from src.train.metrics import compute_metrics
    from scripts.train_all import (build_model, set_seed, denorm_np, _to_hn)

    ds = load_dataset(str(PROJECT_ROOT / "data" / "processed" /
                           "delhi_ncr_pm25_2019_2025.npz"))
    norm_mean = float(ds["norm_mean"])
    norm_std = float(ds["norm_std"])
    n_st = ds["X_full"].shape[1]
    n_feat = ds["X_full"].shape[2]

    X = torch.from_numpy(ds["X_full"])
    y = torch.from_numpy(ds["y_full"])
    mask_y = torch.from_numpy(ds["mask_y"])

    net_mean = None
    if args.network_mean_feature:
        net_mean = torch.from_numpy(network_mean_channel(ds))
        print(f"network-mean PM2.5 feature: {net_mean.shape}, "
              f"range [{float(net_mean.min()):.3f}, {float(net_mean.max()):.3f}]")

    split = get_split(ds, args.split_mode, args.fold)
    tr, va = split["train"], split["val"]
    train_starts = np.arange(int(tr[0]), int(tr[-1]) - 72 - 48 + 2)
    val_all = np.arange(int(va[0]), int(va[-1]) - 72 - 48 + 2)
    if len(val_all) > args.val_windows:
        sel = np.linspace(0, len(val_all) - 1, args.val_windows).astype(int)
        val_starts = val_all[sel]
    else:
        val_starts = val_all

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    for seed in args.seeds:
        set_seed(seed)
        cfg = {
            "hidden_size": 64, "dropout": 0.3, "lag_skip": 24,
            "use_lag_skip": True,
            "input_size": n_feat + 1 + (1 if args.network_mean_feature else 0),
            "graph_type": args.graph, "blocks": 2, "layers": 4, "adapt_dim": 10,
        }
        model = build_model(args.models[0], cfg, n_st).to(device)
        if args.models[0] in ("stgcn", "graph_wavenet"):
            if args.graph in ("identity", "learned_adaptive"):
                adj = np.eye(n_st, dtype=np.float32)
            else:
                trs = slice(int(tr[0]), int(tr[-1]) + 1)
                pm25_train = denorm_np(ds["y_full"][trs], norm_mean, norm_std)
                wind_u = wind_v = None
                if args.graph == "wind_aware":
                    with np.errstate(invalid="ignore"):
                        wind_u = np.nanmean(ds["X_full"][trs][:, :, 3], axis=0)
                        wind_v = np.nanmean(ds["X_full"][trs][:, :, 4], axis=0)
                    wind_u = np.where(np.isnan(wind_u), np.nanmean(wind_u), wind_u)
                    wind_v = np.where(np.isnan(wind_v), np.nanmean(wind_v), wind_v)
                adj = build_graph(args.graph, ds["coords"], pm25_train=pm25_train,
                                  wind_u=wind_u, wind_v=wind_v)
            model.set_adjacency(torch.from_numpy(adj).to(device))

        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

        def make_batch(starts):
            xs, ys, ms = [], [], []
            for s in np.asarray(starts, dtype=np.int64):
                s = int(s)
                m, e = s + 72, s + 72 + 48
                xb = torch.nan_to_num(X[s:m], nan=0.0)
                mb = mask_y[s:m]
                if net_mean is not None:
                    nm = net_mean[s:m].view(-1, 1, 1).expand(-1, n_st, 1)
                    xb = torch.cat([xb, nm], dim=-1)
                xs.append(torch.cat([xb, mb.unsqueeze(-1)], dim=-1))
                ys.append(torch.nan_to_num(y[m:e], nan=0.0))
                ms.append(mask_y[m:e])
            return (torch.stack(xs).to(device), torch.stack(ys).to(device),
                    torch.stack(ms).to(device))

        def evaluate(starts):
            model.eval()
            ae = se = 0.0
            n = 0
            with torch.no_grad():
                for i in range(0, len(starts), args.batch_size):
                    chunk = starts[i:i + args.batch_size]
                    xb, yb, ymask = make_batch(chunk)
                    pred = _to_hn(model(xb), 48, n_st)
                    p = denorm_np(np.nan_to_num(pred.cpu().numpy(), nan=0.0),
                                  norm_mean, norm_std)
                    t = denorm_np(np.nan_to_num(yb.cpu().numpy(), nan=0.0),
                                  norm_mean, norm_std)
                    mm = ymask.cpu().numpy() > 0
                    err = np.abs(p - t)[mm]
                    ae += float(err.sum())
                    se += float((err ** 2).sum())
                    n += err.size
            model.train()
            return ae / max(n, 1), (se / max(n, 1)) ** 0.5

        best = float("inf")
        best_epoch = -1
        patience = 0
        tag = f"{args.models[0]}_{args.graph}_netmean_s{seed}"
        for epoch in range(args.epochs):
            perm = np.random.permutation(train_starts)
            tot = 0.0
            nb = 0
            for i in range(0, len(perm), args.batch_size):
                batch_starts = perm[i:i + args.batch_size]
                opt.zero_grad()
                xb, yb, ymask = make_batch(batch_starts)
                pred = _to_hn(model(xb), 48, n_st)
                diff = torch.abs(pred - yb) * ymask
                loss = diff.sum() / (ymask.sum() + 1e-8)
                loss.backward()
                tot += float(loss.detach())
                nb += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            val_mae, val_rmse = evaluate(val_starts)
            print(f"  epoch {epoch+1:3d}  train={tot/max(nb,1):.4f}  "
                  f"val(µg/m³ MAE)={val_mae:.2f}  RMSE={val_rmse:.2f}")
            if val_mae < best - 1e-4:
                best = val_mae
                best_epoch = epoch + 1
                patience = 0
                torch.save({"dataset_shape": tuple(ds["X_full"].shape),
                        "state_dict": model.state_dict(),
                            "model": args.models[0], "config": cfg,
                            "graph": args.graph, "seed": seed,
                            "n_stations": n_st, "val_mae_ugm3": best,
                            "epoch": best_epoch,
                            "network_mean_feature": bool(args.network_mean_feature)},
                           PROJECT_ROOT / "checkpoints" / f"{tag}.pt")
            else:
                patience += 1
                if patience >= args.patience:
                    break
        print(f"  BEST {tag}: val MAE={best:.2f} @ epoch {best_epoch}")
        results.append({"model": args.models[0], "graph": args.graph,
                        "seed": seed, "val_mae_ugm3": best, "epoch": best_epoch,
                        "network_mean_feature": bool(args.network_mean_feature)})

    import json
    (PROJECT_ROOT / args.out).write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
