#!/usr/bin/env bash
# Node-dropout leave-station-out.
#
# The plain leave-station-out experiment is a handicap: the models are trained
# to forecast a station from its own history and are then asked to forecast a
# station whose own history is missing -- a situation they never practised.
# Here the models are trained with node dropout (each station's own history is
# hidden with probability p for a training window), so the held-out condition
# is one they have seen, and the comparison against the spatial baselines is
# fair.
set -u
cd "$(dirname "$0")/.."
STGCN="--epochs 8 --patience 3 --val_windows 400 --batch_size 32"
GWN="--epochs 6 --patience 3 --val_windows 400 --batch_size 32"
ND="--node_dropout 0.3 --holdout_k 10 --holdout_seed 0"

run() {
  local name="$1"; shift
  if [ -f "checkpoints/$name.pt" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $name"
    return
  fi
  .venv/bin/python -u scripts/train_all.py "$@" > "outputs/${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $name"
}

(
  for s in 42 123 456; do
    for g in identity wind_aware distance; do
      run "nd_stgcn_${g}_s${s}_lso10_both" --models stgcn --graph "$g" --seeds "$s" \
          --holdout_mode both $ND $STGCN
    done
  done
) &

sleep 20

(
  for s in 42 123 456; do
    for g in identity wind_aware distance; do
      run "nd_gwn_${g}_s${s}_lso10_both" --models graph_wavenet --graph "$g" --seeds "$s" \
          --holdout_mode both $ND $GWN
    done
  done
) &

wait
echo "[$(date +%H:%M:%S)] NODE DROPOUT COMPLETE"
