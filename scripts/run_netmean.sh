#!/usr/bin/env bash
# Diagnostic: a no-graph model given the network-mean PM2.5 as an extra input
# feature.  If it matches the graph models, the graph contributes nothing
# beyond the regional signal.
set -u
cd "$(dirname "$0")/.."
for s in 42 123 456; do
  .venv/bin/python -u scripts/train_netmean.py --models stgcn --graph identity \
      --seeds "$s" --epochs 8 --patience 3 --val_windows 400 --batch_size 32 \
      > "outputs/netmean_stgcn_identity_s${s}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE netmean stgcn identity s${s}"
done
