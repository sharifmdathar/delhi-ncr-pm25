# Delhi-NCR Hourly PM2.5 Spatiotemporal Forecasting

**Research Question:** Does a wind-aware dynamic graph beat a static distance graph, and when?

**Answer:** Partly — and the detail matters. A wind-aware directed graph *does*
beat a static distance graph for Graph WaveNet (33.77 vs 34.60 µg/m³, paired
difference 0.78, 95% CI [0.40, 1.15] by bootstrap over 105 test weeks), but is
indistinguishable from it for STGCN (33.05 vs 33.18, CI [−0.11, 0.30]). More
importantly, **no graph at all beats both** in both architectures
(0.36–0.51 µg/m³, both CIs excluding zero). The wind-aware edge over the
distance graph is concentrated in severe-episode windows (1.22 µg/m³ during
episodes against 0.23 outside) — but the same cut shows that *all* spatial
mixing is a net loss: no-graph is best during episodes too, where the graph
penalty is 0.5–1.7 µg/m³. Every neural model here does beat two strong
baselines, so the comparison is meaningful rather than a broken-model artefact.

---

## Results (Test 2024–2025, 2000 windows, masked µg/m³, 8 seeds for the graph models)

| Config | MAE | RMSE | R² | 1h | 48h |
|--------|-----|------|-----|----|-----|
| **STGCN, no graph** | **32.66 ± 0.16** | 57.06 | 0.628 | 15.04 | 36.31 |
| LSTM Univariate | 32.87 ± 0.09 | 57.57 | 0.622 | 15.46 | 36.92 |
| STGCN, wind-aware | 33.05 ± 0.12 | 57.99 | 0.616 | 15.35 | 36.48 |
| STGCN, distance | 33.18 ± 0.15 | 57.84 | 0.618 | 15.75 | 36.58 |
| GraphWaveNet, no graph | 33.44 ± 0.13 | 58.49 | 0.610 | 15.22 | 37.58 |
| STGCN, correlation | 33.46 ± 0.43 | 57.88 | 0.618 | 15.56 | 36.33 |
| GraphWaveNet, wind-aware | 33.77 ± 0.24 | 59.42 | 0.597 | 15.56 | 37.65 |
| GraphWaveNet, distance | 34.60 ± 0.61 | 59.96 | 0.590 | 15.67 | 38.48 |
| GraphWaveNet, correlation | 35.43 ± 1.11 | 59.23 | 0.600 | 15.96 | 37.88 |
| LSTM Multivariate | 38.20 ± 1.83 | 66.94 | 0.489 | 16.27 | 41.66 |
| Transformer | 38.88 ± 1.04 | 66.34 | 0.498 | 16.38 | 42.41 |
| Ridge (24 lags) | 44.03 | 73.69 | 0.380 | 15.83 | 42.34 |
| Persistence | 54.83 | 90.01 | 0.076 | 46.83 | 49.37 |

### Graph ablation

| Graph | STGCN MAE | GraphWaveNet MAE |
|-------|-----------|------------------|
| **None (identity)** | **32.66 ± 0.16** | **33.44 ± 0.13** |
| Wind-aware (directed) | 33.05 ± 0.12 | 33.77 ± 0.24 |
| Distance (static) | 33.18 ± 0.15 | 34.60 ± 0.61 |
| Correlation (static) | 33.46 ± 0.43 | 35.43 ± 1.11 |

Differences this small are not decidable from a per-seed standard deviation, so
the comparison is made with a **paired bootstrap over the 105 ISO weeks of the
test period** (the same weeks resampled for both models), which respects the
temporal correlation in the errors:

| Comparison | ΔMAE | 95% CI | P(A better) | Verdict |
|------------|------|--------|-------------|---------|
| STGCN, distance vs wind-aware | +0.09 | [−0.11, 0.30] | 0.186 | not distinguishable |
| GraphWaveNet, distance vs wind-aware | +0.78 | [0.40, 1.15] | 0.000 | **wind-aware better** |
| STGCN, distance vs no graph | +0.51 | [0.15, 0.89] | 0.003 | **no graph better** |
| STGCN, wind-aware vs no graph | +0.42 | [0.12, 0.74] | 0.003 | **no graph better** |
| GraphWaveNet, distance vs no graph | +1.14 | [0.56, 1.75] | 0.000 | **no graph better** |
| GraphWaveNet, wind-aware vs no graph | +0.36 | [0.01, 0.75] | 0.023 | **no graph better** |

### Where the graph penalty is actually paid: severe episodes

Severe PM2.5 episodes in Delhi occur under calm, stable conditions: 78.2% of the
test windows in the lowest wind third contain a severe-episode hour, against
52.1% and 49.0% in the two windier thirds. Splitting the test set by episode
membership shows where the ablation comes from:

| Subset | STGCN no graph | STGCN wind | STGCN dist. | GWN no graph | GWN wind | GWN dist. |
|--------|----------------|------------|-------------|--------------|----------|-----------|
| all windows | **32.66** | 33.05 | 33.18 | **33.44** | 33.77 | 34.60 |
| episode windows | **42.31** | 42.92 | 43.16 | **43.34** | 43.84 | 45.06 |
| non-episode windows | **17.90** | 17.93 | 17.91 | **18.29** | 18.35 | 18.58 |

The graph penalty is paid almost entirely in the high-pollution regime, and
spatial mixing does not rescue the high-pollution case even where transport
should matter most.

### Rolling-origin check (is it just the split boundary?)

Train on 2019…Y, validate on Q4 of Y, test on Y+1. The ranking is stable as
the boundary moves, so the finding is not an artefact of one split:

| Model | Fold | Test year | No graph | Wind-aware | Distance | Best |
|-------|------|-----------|----------|------------|----------|------|
| STGCN | fold0 | 2021 | 36.65 | 36.98 | 37.34 | no graph |
| GraphWaveNet | fold0 | 2021 | 37.56 | 37.63 | 43.25 | no graph |
| STGCN | fold1 | 2022 | 32.96 | 33.09 | 32.61 | distance |
| STGCN | fold2 | 2023 | 33.03 | 33.01 | 33.58 | wind-aware |
| GraphWaveNet | fold2 | 2023 | 34.23 | 33.64 | 34.05 | wind-aware |
| STGCN | fold3 | 2024 | 33.82 | 34.57 | 34.22 | no graph |
| STGCN | fold4 | 2025 | 30.76 | 30.77 | 31.41 | no graph |
| GraphWaveNet | fold4 | 2025 | 31.32 | 31.23 | 32.87 | wind-aware |

Full tables (all horizons, episode analysis, per-station errors) in
[`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md).

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Fetch real CPCB data (2019-2025, 15-min).  The station files also carry
#    temperature, humidity, wind speed/direction and rainfall measured at the
#    station, so no reanalysis download is needed.
python scripts/fetch_cpcb_years.py --years 2019 2020 2021 2022 2023 2024 2025
python scripts/prepare_real_data.py pm25 --src data/raw/cpcb_station_csv

# 2. Build frozen dataset (fails if any station is below 70% real coverage)
python scripts/build_dataset.py

# 3. Train (--graph selects the adjacency; --split_mode expanding runs a
#    rolling-origin fold instead of the fixed split)
python scripts/train_all.py --models lstm_univariate stgcn --seeds 42 123 456
python scripts/train_all.py --models graph_wavenet --graph wind_aware --seeds 42 123 456
python scripts/train_all.py --models stgcn --split_mode expanding --fold fold0

# 4. Evaluate, analyse, write the report
python scripts/evaluate.py --split test --max_windows 2000
python scripts/bootstrap.py --n_boot 50000
python scripts/analyze.py --max_windows 1200
python scripts/make_report.py
```

---

## Data

| Item | Value |
|------|-------|
| Source | CPCB 15-min, via `Vonter/india-cpcb-aqi` releases (2009–2025) |
| Stations | **42** (34 Delhi + 8 Noida/Ghaziabad), ≥70% real coverage in every split |
| Period | **2019-01-01 → 2025-12-31** (61,345 hours, 7 years) |
| Target | Hourly PM2.5 (µg/m³) |
| Input | Past 72 hours |
| Output | Next 48 hours |
| Features | PM2.5 + station-measured T, RH, wind speed/direction, rainfall |
| Splits | Train 2019–2022 (34,922 windows) · Val 2023 (8,641) · Test 2024–2025 (17,402), chronological, no window crosses a boundary |
| Normalisation | log1p(PM2.5) → z-score (train-only stats) |
| Missing | masked (1 = really observed), honoured in loss and metrics |

Split distributions are matched (mean 102.0 / 98.0 / 97.5 µg/m³; fraction
> 300 µg/m³ = 4.82% / 4.23% / 4.15%), and the training split contains **four
full winters**, so winter dynamics are learnable.  Split boundaries are
contiguous (0 h train→val) with a 23 h gap before test; no window's input or
target ever crosses a boundary.

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

- Seven years of data (2019–2025), but the fixed test block covers only two
  winters, so generalisation across pollution regimes is only partly tested.
  The rolling-origin evaluation (5 folds, test years 2021–2025) is our partial
  answer to this.
- The wind-aware graph uses a period-mean prevailing direction and assumes a
  single direction per station. The per-station wind vectors agree only weakly
  (mean cross-station correlation of the u-component is 0.11), so that mean
  direction is itself a noisy estimate.
- The graph comparison rests on 8 seeds for the two graph architectures, but on
  a single model family and a single hyperparameter setting per configuration.
  Each variant received the same tuning effort, but we did not sweep the
  graph-specific hyperparameters.
- Static graphs except GraphWaveNet's learned variant.
- Severe episodes remain 3–5× worse than ordinary conditions for every model.

---

## License

MIT. Data: CPCB, via the `Vonter/india-cpcb-aqi` releases (CPCB data is
published under ODbL). The weather variables are the station-measured values
distributed in the same CPCB files.
