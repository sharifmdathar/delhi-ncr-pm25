#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
STGCN="--epochs 8 --patience 3 --val_windows 400 --batch_size 32"
for f in fold0 fold1 fold2 fold3 fold4; do
  for g in identity wind_aware distance; do
    .venv/bin/python -u scripts/train_all.py --models stgcn --graph "$g" --seeds 42 \
        --split_mode expanding --fold "$f" $STGCN > "outputs/stgcn_${g}_s42_${f}.log" 2>&1
    echo "[$(date +%H:%M:%S)] DONE stgcn_${g}_s42_${f}"
  done
done
