#!/usr/bin/env bash
# Leave-station-out experiment.
#   mode=target : the station's targets never enter the loss, its own history
#                 is visible  -> "generalise to an unseen monitor?"
#   mode=both   : targets AND its own history are hidden
#                 -> "can the graph substitute for a station's own history?"
set -u
cd "$(dirname "$0")/.."
STGCN="--epochs 8 --patience 3 --val_windows 400 --batch_size 32"
GWN="--epochs 6 --patience 3 --val_windows 400 --batch_size 32"

run() {
  local name="$1"; shift
  .venv/bin/python -u scripts/train_all.py "$@" > "outputs/${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $name"
}

# Stream A: STGCN, both modes, four graph types, three seeds
(
  for mode in target both; do
    for s in 42 123 456; do
      for g in identity wind_aware wind_aware_era5 distance; do
        run "lso_stgcn_${g}_s${s}_${mode}" --models stgcn --graph "$g" --seeds "$s" \
            --holdout_k 10 --holdout_seed 0 --holdout_mode "$mode" $STGCN
      done
    done
  done
) &

sleep 20

# Stream B: GraphWaveNet, both modes, three graph types, two seeds
(
  for mode in target both; do
    for s in 42 123; do
      for g in identity wind_aware distance; do
        run "lso_gwn_${g}_s${s}_${mode}" --models graph_wavenet --graph "$g" --seeds "$s" \
            --holdout_k 10 --holdout_seed 0 --holdout_mode "$mode" $GWN
      done
    done
  done
) &

wait
echo "[$(date +%H:%M:%S)] LSO COMPLETE"
