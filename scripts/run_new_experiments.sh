#!/usr/bin/env bash
# Remaining training for the two new experiments.
#   Stream A: the ERA5-wind ablation on the fixed split (3 seeds each)
#   Stream B: the leave-station-out experiment (trimmed to a time box)
# Both skip runs whose checkpoint already exists, so the script is resumable.
set -u
cd "$(dirname "$0")/.."

have() { [ -f "checkpoints/$1.pt" ]; }

run() {
  local tag="$1"; shift
  if have "$tag"; then
    echo "[$(date +%H:%M:%S)] SKIP $tag (exists)"
    return
  fi
  if .venv/bin/python -u scripts/train_all.py "$@" > "outputs/${tag}.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] DONE $tag"
  else
    echo "[$(date +%H:%M:%S)] FAILED $tag  (see outputs/${tag}.log)"
  fi
}

GWN="--epochs 6 --patience 3 --val_windows 400 --batch_size 32"
STGCN="--epochs 8 --patience 3 --val_windows 400 --batch_size 32"
LSO_STGCN="--epochs 4 --patience 2 --val_windows 400 --batch_size 32"
LSO_GWN="--epochs 4 --patience 2 --val_windows 400 --batch_size 32"

# ---- Stream A: ERA5 wind ablation, fixed split ----
(
  for s in 42 123 456; do
    run "stgcn_wind_aware_era5_s${s}" --models stgcn --graph wind_aware_era5 --seeds "$s" $STGCN
  done
  for s in 42 123 456; do
    run "graph_wavenet_wind_aware_era5_s${s}" --models graph_wavenet --graph wind_aware_era5 --seeds "$s" $GWN
  done
) &

sleep 20

# ---- Stream B: leave-station-out ----
(
  for mode in target both; do
    for s in 42 123; do
      for g in identity wind_aware distance; do
        run "lso_stgcn_${g}_s${s}_${mode}" --models stgcn --graph "$g" --seeds "$s" \
            --holdout_k 10 --holdout_seed 0 --holdout_mode "$mode" $LSO_STGCN
      done
    done
  done
  for mode in target both; do
    for s in 42 123; do
      for g in identity wind_aware distance; do
        run "lso_gwn_${g}_s${s}_${mode}" --models graph_wavenet --graph "$g" --seeds "$s" \
            --holdout_k 10 --holdout_seed 0 --holdout_mode "$mode" $LSO_GWN
      done
    done
  done
) &

wait
echo "[$(date +%H:%M:%S)] ALL TRAINING COMPLETE"
