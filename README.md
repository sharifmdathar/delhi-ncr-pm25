# Delhi-NCR Hourly PM2.5 Spatiotemporal Forecasting

## Project Overview

Forecast hourly PM2.5 at station level across Delhi-NCR using spatiotemporal graph neural networks.

**Research Question:** Does a wind-aware dynamic graph beat a static distance graph, and when?

## Key Decisions (Locked Scope)

| Item | Decision |
|---|---|
| Region | Delhi NCR: Delhi, Noida, Gurgaon, Ghaziabad, Faridabad |
| Stations | ~70 CPCB/CAAQMS stations |
| Period | 2019-01-01 to 2024-12-31 |
| Target | Hourly PM2.5 (µg/m³) |
| Input | Past 72 hours |
| Output | Next 48 hours |
| Normalization | `log1p(PM2.5)` then z-score (fit on train only) |
| Splits | Chronological only |

**Full spec:** See `AGENTS.md`

## Quick Start

```bash
# Build dataset (requires raw data in data/raw/)
python scripts/build_dataset.py

# Run a baseline
python scripts/run_experiment.py --config configs/baseline_persistence.yaml

# Run GNN model
python scripts/run_experiment.py --config configs/graph_wavenet.yaml
```

## Repository Structure

```
.
├── AGENTS.md                  # Master spec (this file is the source of truth)
├── src/constants.py           # Locked scope enforced by code
├── src/data/                  # Acquisition, cleaning, dataset freeze, graphs
├── src/models/                # Baselines, LSTM, Transformer, STGCN, Graph WaveNet, GCN+GRU
├── src/train/                 # Trainer with masked MAE, metrics
├── src/eval/                  # Episode analysis, spatial plots, evaluation
├── configs/                   # 8 YAML experiment configs
├── scripts/                   # Data pull, dataset build, experiment runner, animation
├── tests/                     # No leakage, shapes, masks
└── EXPERIMENTS.md             # Every run logged
```

## Results (Preliminary)

See `EXPERIMENTS.md` for full log.

All baselines implemented. Dataset frozen. Tests passing.

**Status:** Pipeline verified. Waiting for full real-data rebuild and final model runs.

## Deliverables

- [x] Reproducible repo
- [x] Frozen dataset mechanism
- [x] All baselines implemented
- [x] GCN+GRU from scratch
- [x] STGCN + Graph WaveNet
- [ ] Final model evaluation (needs full data + compute)
- [ ] Animated AQI map GIF
- [x] LaTeX report skeleton (see `report/` or write-up section)
