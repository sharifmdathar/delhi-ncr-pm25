#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
GWN="--epochs 6 --patience 3 --val_windows 400 --batch_size 32"
for s in 42 123 456; do
  .venv/bin/python -u scripts/train_all.py --models graph_wavenet --graph learned_adaptive \
      --seeds "$s" $GWN > "outputs/gwn_learned_adaptive_s${s}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE gwn_learned_adaptive_s${s}"
done
