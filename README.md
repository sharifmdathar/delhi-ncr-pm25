# Delhi-NCR Hourly PM2.5 Spatiotemporal Forecasting

This is the single source of truth for the project. Follow the instructions in `AGENTS.md` in order.

## Project Structure

- `src/` - All source code
- `data/` - Data files (raw, interim, processed)
- `notebooks/` - Exploration notebooks (not for final logic)
- `scripts/` - Data acquisition and processing scripts
- `tests/` - Unit tests
- `configs/` - Experiment configurations
- `EXPERIMENTS.md` - Log of all experiment runs

## Getting Started

1. Install dependencies:
```bash
conda env create -f environment.yml
conda activate gnn
pip install -r requirements.txt
```

2. Run data acquisition:
```bash
python scripts/pull_cpcb.py
python scripts/pull_openmeteo.py
python scripts/build_dataset.py
```

3. Run experiments:
```bash
python scripts/run_experiment.py --config configs/graph_wavenet.yaml
```

## License

MIT