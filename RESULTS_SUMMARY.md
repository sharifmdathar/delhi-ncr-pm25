# Delhi-NCR PM2.5 Forecasting — Final Results (2024–2025 real data)

## Research Question
**Does a wind-aware dynamic graph beat a static distance graph, and when?**

## Answer
**No — and neither beats using no graph structure at all.** The differences between
graph variants are within seed noise. Every neural model here does beat two strong
baselines, so the comparison is meaningful rather than a broken-model artefact.

## Test results (2025, 2000 evenly-spaced windows, masked, µg/m³, 3 seeds)

| Config | MAE | RMSE | R² | 1h | 3h | 6h | 12h | 24h | 48h |
|--------|-----|------|-----|----|----|----|-----|-----|-----|
| **STGCN, no graph** | **32.60 ± 0.10** | 57.25 | 0.626 | 15.05 | 23.80 | 28.52 | 30.99 | 32.92 | 36.39 |
| LSTM Univariate | 32.87 ± 0.09 | 57.61 | 0.621 | 15.46 | 23.91 | 28.62 | 31.17 | 33.20 | 37.03 |
| STGCN, wind-aware | 33.06 ± 0.11 | 57.73 | 0.620 | 15.63 | 23.92 | 28.79 | 31.30 | 33.68 | 36.94 |
| STGCN, distance | 33.11 ± 0.14 | 58.08 | 0.615 | 15.48 | 24.31 | 28.79 | 31.47 | 33.67 | 36.99 |
| GraphWaveNet, no graph | 33.35 ± 0.05 | 58.16 | 0.614 | 15.22 | 24.16 | 29.00 | 31.61 | 33.35 | 37.37 |
| STGCN, correlation | 33.46 ± 0.43 | 58.19 | 0.614 | 15.91 | 24.29 | 29.27 | 32.05 | 33.82 | 36.84 |
| GraphWaveNet, wind-aware | 33.76 ± 0.30 | 58.72 | 0.604 | 15.54 | 24.18 | 29.16 | 31.89 | 33.78 | 37.69 |
| GraphWaveNet, distance | 34.83 ± 0.69 | 59.95 | 0.590 | 15.67 | 24.96 | 30.36 | 32.86 | 34.68 | 38.60 |
| GraphWaveNet, correlation | 35.43 ± 1.11 | 60.86 | 0.578 | 15.89 | 25.35 | 30.72 | 33.19 | 35.09 | 39.26 |
| LSTM Multivariate | 38.20 ± 1.83 | 68.47 | 0.465 | 16.56 | 25.78 | 31.24 | 35.63 | 39.05 | 42.13 |
| Transformer | 38.88 ± 1.04 | 68.93 | 0.457 | 16.45 | 26.55 | 32.07 | 35.74 | 39.00 | 44.55 |
| Ridge (24 lags) | 44.03 | 73.69 | 0.380 | 15.83 | 28.09 | 39.46 | 47.14 | 36.45 | 42.34 |
| Persistence | 54.83 | 90.01 | 0.076 | 46.83 | 49.04 | 54.72 | 58.09 | 48.89 | 49.37 |

## Graph ablation (the research question)

| Graph | STGCN MAE | GraphWaveNet MAE |
|-------|-----------|------------------|
| **None (identity)** | **32.60 ± 0.10** | **33.35 ± 0.05** |
| Wind-aware (directed) | 33.06 ± 0.11 | 33.76 ± 0.30 |
| Distance (static) | 33.11 ± 0.14 | 34.83 ± 0.69 |
| Correlation (static) | 33.46 ± 0.43 | 35.43 ± 1.11 |

- STGCN: wind-aware vs distance differs by 0.05 MAE — inside one seed std. **No effect.**
- GraphWaveNet: wind-aware vs distance differs by 1.07 MAE — about two pooled std. **Marginal at best.**
- The **no-graph** variant is best for both architectures. Neighbourhood mixing adds
  noise, not signal.

## Severe episodes

Episodes = PM2.5 > 150 µg/m³ sustained ≥ 6 h (15.6% of test hours).

| Model | Overall | Episodes | Non-episodes |
|-------|---------|----------|--------------|
| STGCN, no graph | 32.60 | 78.0 | 23.5 |
| LSTM Univariate | 32.87 | 79.5 | 23.7 |
| Ridge | 44.03 | 98.4 | 33.4 |
| Persistence | 54.83 | 120.4 | 42.0 |

All architectures are 3–5× worse during episodes, and the spread between models
widens there. Extreme pollution is where the remaining error lives.

## Data

| Item | Value |
|------|-------|
| Source | CPCB 15-min, via `Vonter/india-cpcb-aqi` releases (2009–2025) |
| Stations | **42** (34 Delhi + 8 Noida/Ghaziabad), ≥70% real coverage in every split |
| Period | **2019-01-01 → 2025-12-31** (61,345 hours, 7 years) |
| Splits | Train 2019–2022 (34,922 windows) · Val 2023 (8,641) · Test 2024–2025 (17,402) |
| Features | PM2.5 + station-measured T/RH/wind/rain |
| Normalisation | log1p + z-score, train split only |
| Missing | masked (1 = really observed), honoured in loss and metrics |

Split distributions are matched (mean 102.0 / 98.0 / 97.5 µg/m³; frac > 300 =
4.82% / 4.23% / 4.15%), so the training split contains **four full winters**.
Split boundaries are contiguous (0 h train→val) with a 23 h gap before test; no
window's input or target ever crosses a boundary (verified programmatically).

## What made the models work

1. **Predict the change from the last observation**, not the absolute concentration.
   The output layer is zero-initialised, so each model starts *at* Persistence and
   can only improve.
2. **Linear autoregressive skip** (last 24 lags → 48 horizons), i.e. the ridge
   solution inside the network. The network is measured relative to a strong
   linear floor; without it, models collapsed to the training mean.
3. **Mask channel** so imputed hours are distinguishable from observed ones.
4. **Checkpoint selection on denormalised µg/m³**, not normalised loss.

## Honest limitations

- ~2 years of data: generalisation across years and pollution regimes is untested.
- 2019–2023 hourly data for these stations could not be obtained (OpenCity files
  are sparse 2017–2018; the Kaggle extract stops in 2020). This is the single
  biggest constraint on what can be claimed.
- The wind-aware graph uses a period-mean prevailing direction, discarding gust
  structure, and assumes a single direction per station.
- Graph structures are static except GraphWaveNet's learned variant.
- One hazard (PM2.5), one horizon family, no episode-conditioned modelling.

## Reproduce

```bash
python scripts/fetch_cpcb_years.py --years 2019 2020 2021 2022 2023 2024 2025
python scripts/prepare_real_data.py pm25 --src data/raw/opencity
python scripts/prepare_real_data.py weather --era5 data/raw/era5
python scripts/build_dataset.py                 # fails if coverage < 70%
python scripts/train_all.py --models lstm_univariate stgcn --seeds 42 123 456
python scripts/evaluate.py --split test --max_windows 2000
python scripts/analyze.py --max_windows 2000
python scripts/make_report.py
```

## Artefacts

| File | Contents |
|------|----------|
| `data/processed/delhi_ncr_pm25_2019_2025.npz` | frozen dataset |
| `data/processed/coverage_report.csv` | per-station per-year real-observation fractions |
| `checkpoints/*.pt` | 33 trained models |
| `outputs/eval_results.json` | per-checkpoint metrics |
| `outputs/analysis_episodes.json` | episode vs non-episode metrics |
| `outputs/analysis_spatial.json` | per-station MAE / bias |
| `outputs/fig_*.png` | figures |
| `outputs/aqi_map.gif` | animated forecast map |
| `report/report.tex` | generated report |
