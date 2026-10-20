#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
GWN="--epochs 6 --patience 3 --val_windows 400 --batch_size 32"
for f in fold0 fold2 fold4; do
  for g in identity wind_aware distance; do
    .venv/bin/python -u scripts/train_all.py --models graph_wavenet --graph "$g" --seeds 42 \
        --split_mode expanding --fold "$f" $GWN > "outputs/gwn_${g}_s42_${f}.log" 2>&1
    echo "[$(date +%H:%M:%S)] DONE gwn_${g}_s42_${f}"
  done
done
