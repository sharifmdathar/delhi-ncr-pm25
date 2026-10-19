# Delhi-NCR Hourly PM2.5 Spatiotemporal Forecasting

**Research Question:** Does a wind-aware dynamic graph beat a static distance graph, and when?

**Answer:** No — and neither beats using no graph structure at all. The differences
between graph variants are within seed noise. Every neural model here does beat two
strong baselines, so the comparison is meaningful rather than a broken-model artefact.

---

## Results (Test 2025, 2000 windows, masked µg/m³, 3 seeds)

| Config | MAE | RMSE | R² | 1h | 48h |
|--------|-----|------|-----|----|-----|
| **STGCN, no graph** | **32.60 ± 0.10** | 57.25 | 0.626 | 15.05 | 36.39 |
| LSTM Univariate | 32.87 ± 0.09 | 57.61 | 0.621 | 15.46 | 37.03 |
| STGCN, wind-aware | 33.06 ± 0.11 | 57.73 | 0.620 | 15.63 | 36.94 |
| STGCN, distance | 33.11 ± 0.14 | 58.08 | 0.615 | 15.48 | 36.99 |
| GraphWaveNet, no graph | 33.35 ± 0.05 | 58.16 | 0.614 | 15.22 | 37.37 |
| STGCN, correlation | 33.46 ± 0.43 | 58.19 | 0.614 | 15.91 | 36.84 |
| GraphWaveNet, wind-aware | 33.76 ± 0.30 | 58.72 | 0.604 | 15.54 | 37.69 |
| GraphWaveNet, distance | 34.83 ± 0.69 | 59.95 | 0.590 | 15.67 | 38.60 |
| GraphWaveNet, correlation | 35.43 ± 1.11 | 60.86 | 0.578 | 15.89 | 39.26 |
| LSTM Multivariate | 38.20 ± 1.83 | 68.47 | 0.465 | 16.56 | 42.13 |
| Transformer | 38.88 ± 1.04 | 68.93 | 0.457 | 16.45 | 44.55 |
| Ridge (24 lags) | 44.03 | 73.69 | 0.380 | 15.83 | 42.34 |
| Persistence | 54.83 | 90.01 | 0.076 | 46.83 | 49.37 |

### Graph ablation

| Graph | STGCN MAE | GraphWaveNet MAE |
|-------|-----------|------------------|
| **None (identity)** | **32.60 ± 0.10** | **33.35 ± 0.05** |
| Wind-aware (directed) | 33.06 ± 0.11 | 33.76 ± 0.30 |
| Distance (static) | 33.11 ± 0.14 | 34.83 ± 0.69 |
| Correlation (static) | 33.46 ± 0.43 | 35.43 ± 1.11 |

Full tables (all horizons, episode analysis, per-station errors) in
[`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md).

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Fetch real CPCB data (2019-2025, 15-min) and ERA5 weather
python scripts/fetch_cpcb_years.py --years 2019 2020 2021 2022 2023 2024 2025
python scripts/prepare_real_data.py pm25 --src data/raw/opencity
python scripts/prepare_real_data.py weather --era5 data/raw/era5

# 2. Build frozen dataset (fails if any station is below 70% real coverage)
python scripts/build_dataset.py

# 3. Train (3 seeds; --graph selects the adjacency)
python scripts/train_all.py --models lstm_univariate stgcn --seeds 42 123 456
python scripts/train_all.py --models graph_wavenet --graph wind_aware --seeds 42 123 456

# 4. Evaluate, analyse, write the report
python scripts/evaluate.py --split test --max_windows 2000
python scripts/analyze.py --max_windows 2000
python scripts/make_report.py
```

---

## Data

| Item | Value |
|------|-------|
| Source | CPCB 15-min, via `Vonter/india-cpcb-aqi` releases (2009–2025) |
| Stations | 39 Delhi (DL001–DL039), ≥70% real coverage in every split |
| Period | 2024-01-01 → 2025-12-31 (17,521 hours) |
| Target | Hourly PM2.5 (µg/m³) |
| Input | Past 72 hours |
| Output | Next 48 hours |
| Features | PM2.5 + station-measured T, RH, wind speed/direction, rainfall |
| Splits | Train Jan–Sep 2024 · Val Oct–Dec 2024 · Test 2025 (chronological only) |
| Normalisation | log1p(PM2.5) → z-score (train-only stats) |
| Missing | masked (1 = really observed), honoured in loss and metrics |

Split distributions are matched (mean 102.0 / 98.0 / 97.5 µg/m³), so the
training split contains four full winters.

---

## Models

| Model | Description | Key config |
|-------|-------------|------------|
| Persistence | ŷ = y(t−1) | baseline |
| Ridge (24 lags) | per-station linear map, last 24 h → 48 horizons | baseline / linear floor |
| LSTM Univariate | shared-weight LSTM applied per station | hidden=64, layers=2, dropout=0.3 |
| LSTM Multivariate | single LSTM, all stations as channels | hidden=64, layers=2 |
| Transformer | temporal attention, stations as channels | hidden=64, layers=2, heads=4 |
| STGCN | graph convolution + GRU | hidden=64, σ=5 km |
| GraphWaveNet | dilated causal convs + graph conv | blocks=2, layers=4, conv width 32 |

All models predict the **change from the last observed value** and carry a
**linear autoregressive skip**, so each starts at Persistence and is measured
relative to ridge. See `src/models/base.py`.

---

## Graph structures

| Type | Definition |
|------|------------|
| Distance | Gaussian kernel exp(−d²/2σ²) on haversine distance, σ = 5 km |
| Correlation | pairwise-complete Pearson correlation of PM2.5 (train only), thresholded |
| Wind-aware | directed edge i→j only when the prevailing wind at i points at j: exp(−d²/2σ²)·max(0, cos(θ_ij − φ_i)) |
| Learned adaptive | softmax(relu(E₁E₂ᵀ)), learned in-model |
| None (identity) | ablation baseline |

---

## Project layout

```
├── AGENTS.md                  locked scope
├── EXPERIMENTS.md             every run logged, including the failed study
├── RESULTS_SUMMARY.md         final tables
├── scripts/
│   ├── fetch_cpcb_years.py   download + extract CPCB 2019-2025
│   ├── prepare_real_data.py   OpenCity parsing + ERA5 weather
│   ├── build_dataset.py       frozen dataset, with the coverage gate
│   ├── train_all.py           training (batched, delta targets, graph arg)
│   ├── evaluate.py            baselines + every checkpoint
│   ├── analyze.py             episodes, spatial errors, figures
│   ├── make_report.py         LaTeX generated from the JSON artefacts
│   └── make_animated_map.py   GIF
├── src/
│   ├── constants.py           locked scope + MIN_REAL_DATA_FRACTION
│   ├── data/                  cleaning (no synthetic fallback), dataset, graphs
│   ├── models/                base (delta + skip), lstm, transformer, stgcn, graph_wavenet
│   ├── train/                 masked metrics
│   └── eval/                  episodes, plots
├── data/processed/            frozen dataset + coverage_report.csv
├── checkpoints/               33 trained models
├── outputs/                   JSON results, figures, aqi_map.gif
└── report/report.tex          generated report
```

---

## Limitations

- ~2 years of data — generalisation across years and pollution regimes untested.
- 2019–2023 hourly data for these stations could not be obtained (OpenCity files
  are sparse 2017–2018; the Kaggle extract stops in 2020). **This is the single
  biggest constraint on what can be claimed.**
- The wind-aware graph uses a period-mean prevailing direction and assumes a
  single direction per station.
- Static graphs except GraphWaveNet's learned variant.
- Severe episodes remain 3–5× worse than ordinary conditions for every model.

---

## License

MIT. Data: CPCB (via OpenCity / `Vonter/india-cpcb-aqi`), ERA5 (Copernicus CDS).
