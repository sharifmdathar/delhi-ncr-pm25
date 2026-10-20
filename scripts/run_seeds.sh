#!/usr/bin/env bash
# Extra seeds for the six key graph configurations (identity / wind_aware / distance
# for STGCN and GraphWaveNet), so the graph comparison can be judged with a
# paired bootstrap over test weeks rather than a per-seed standard deviation.
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs

GWN="--epochs 6 --patience 3 --val_windows 400 --batch_size 32"
STGCN="--epochs 8 --patience 3 --val_windows 400 --batch_size 32"
SEEDS="7 13 21 34 55"

run() {
  local name="$1"; shift
  .venv/bin/python -u scripts/train_all.py "$@" > "outputs/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $name"
}

while pgrep -f "bin/python -u scripts/train_all.py" > /dev/null; do sleep 20; done

# Stream A: STGCN variants
(
  for s in $SEEDS; do
    for g in distance wind_aware identity; do
      run "stgcn_${g}_s${s}" --models stgcn --graph "$g" --seeds "$s" $STGCN
    done
  done
) &

sleep 15

# Stream B: GraphWaveNet, first half of the seeds
(
  for s in $SEEDS; do
    for g in distance wind_aware identity; do
      run "gwn_${g}_s${s}" --models graph_wavenet --graph "$g" --seeds "$s" $GWN
    done
  done
) &

wait
echo "[$(date +%H:%M:%S)] EXTRA SEEDS COMPLETE"
