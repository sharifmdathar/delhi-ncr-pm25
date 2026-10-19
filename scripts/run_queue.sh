#!/usr/bin/env bash
# Queue the remaining training runs.
# Memory-aware settings: the GPU has 3.7 GB, so graph models use batch 32 and
# the LSTM stream uses batch 24 (B*N sequences).
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs

GWN="--epochs 6 --patience 3 --val_windows 400 --batch_size 32"
STGCN="--epochs 8 --patience 3 --val_windows 400 --batch_size 32"
SMALL="--epochs 8 --patience 3 --val_windows 400 --batch_size 24"

run() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name"
  .venv/bin/python -u scripts/train_all.py "$@" > "outputs/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $name"
}

# Wait for any currently running training to finish
while pgrep -f "bin/python -u scripts/train_all.py" > /dev/null; do sleep 20; done

# Distance-graph STGCN / GraphWaveNet checkpoints produced before the
# station-coordinate fix must be discarded and retrained.
rm -f checkpoints/stgcn_distance_s*.pt checkpoints/graph_wavenet_distance_s*.pt

# Stream A: seed 456 (all models) then all STGCN graph variants
(
  run s456 --seeds 456 $SMALL
  for g in distance correlation wind_aware identity; do
    for s in 42 123 456; do
      run "stgcn_${g}_s${s}" --models stgcn --graph "$g" --seeds "$s" $STGCN
    done
  done
) &

sleep 15

# Stream B: GraphWaveNet, seeds 42 and 123
(
  for g in distance correlation wind_aware identity; do
    for s in 42 123; do
      run "gwn_${g}_s${s}" --models graph_wavenet --graph "$g" --seeds "$s" $GWN
    done
  done
) &

sleep 15

# Stream C: GraphWaveNet, seed 456
(
  for g in distance correlation wind_aware identity; do
    run "gwn_${g}_s456" --models graph_wavenet --graph "$g" --seeds 456 $GWN
  done
) &

wait
echo "[$(date +%H:%M:%S)] ALL QUEUED RUNS COMPLETE"
