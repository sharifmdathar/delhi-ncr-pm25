#!/usr/bin/env bash
# ERA5-wind ablation: same wind-aware construction, but edge directions from
# ERA5 reanalysis instead of the stations' own anemometers.
set -u
cd "$(dirname "$0")/.."
GWN="--epochs 6 --patience 3 --val_windows 400 --batch_size 32"
STGCN="--epochs 8 --patience 3 --val_windows 400 --batch_size 32"
for s in 42 123 456; do
  .venv/bin/python -u scripts/train_all.py --models stgcn --graph wind_aware_era5 --seeds "$s" $STGCN \
      > "outputs/stgcn_era5_s${s}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE stgcn_era5_s${s}"
done
for s in 42 123 456; do
  .venv/bin/python -u scripts/train_all.py --models graph_wavenet --graph wind_aware_era5 --seeds "$s" $GWN \
      > "outputs/gwn_era5_s${s}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE gwn_era5_s${s}"
done
