#!/usr/bin/env bash
# Rolling-origin (expanding-window) evaluation over the 7 available years.
# For each fold the model trains on 2019..Y, validates on Q4 of Y, and is
# tested on Y+1 -- so the conclusion is not tied to one train/test boundary.
set -u
cd "$(dirname "$0")/.."
STGCN="--epochs 8 --patience 3 --val_windows 400 --batch_size 32"
GWN="--epochs 6 --patience 3 --val_windows 400 --batch_size 32"

run() {
  local name="$1"; shift
  .venv/bin/python -u scripts/train_all.py "$@" > "outputs/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $name"
}

while pgrep -f "bin/python -u scripts/train_all.py" > /dev/null; do sleep 20; done

# Stream A: STGCN, all folds, three graph types
(
  for f in fold0 fold1 fold2 fold3 fold4; do
    for g in identity wind_aware distance; do
      run "stgcn_${g}_s42_${f}" --models stgcn --graph "$g" --seeds 42 \
          --split_mode expanding --fold "$f" $STGCN
    done
  done
) &

sleep 15

# Stream B: GraphWaveNet, all folds, three graph types
(
  for f in fold0 fold1 fold2 fold3 fold4; do
    for g in identity wind_aware distance; do
      run "gwn_${g}_s42_${f}" --models graph_wavenet --graph "$g" --seeds 42 \
          --split_mode expanding --fold "$f" $GWN
    done
  done
) &

wait
echo "[$(date +%H:%M:%S)] ALL FOLDS COMPLETE"
