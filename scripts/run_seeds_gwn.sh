#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
GWN="--epochs 6 --patience 3 --val_windows 400 --batch_size 32"
for s in 7 13 21 34 55; do
  for g in distance wind_aware identity; do
    .venv/bin/python -u scripts/train_all.py --models graph_wavenet --graph "$g" --seeds "$s" $GWN \
      > "outputs/gwn_${g}_s${s}.log" 2>&1
    echo "[$(date +%H:%M:%S)] DONE gwn_${g}_s${s}"
  done
done
