# Master Spec / AGENTS.md  
## Delhi-NCR Hourly PM2.5 Spatiotemporal Forecasting

This is the single source of truth. Follow it in order. Do not skip gates. Do not start GNN work until baselines are logged. When a choice is not specified, choose the simplest option, document it in `EXPERIMENTS.md`, and continue.

---

## 0. Agent Operating Principles

1. **One repo, one pipeline, one frozen dataset.**
2. **Commit after every completed task.** Small, descriptive commits.
3. **Update `EXPERIMENTS.md` after every run** with config, seed, metrics, notes.
4. **Never leak.** No random splits. No test-period normalization. No forward-looking imputation.
5. **Baselines before GNNs.** No exceptions.
6. **Depth over breadth.** Two well-trained models beat five half-trained ones.
7. **Compute is not the bottleneck.** Data engineering and thinking are.
8. **Target hardware:** 4 GB VRAM (e.g., 3050 Ti laptop) or free Colab/Kaggle. Use mixed precision, gradient accumulation, checkpointing if needed.
9. **Plain PyTorch for core models.** Skip PyTorch Geometric unless you need it only as a reference. Implement graph convolution as matmuls with adjacency matrices.
10. **If OOM:** reduce hidden size, batch size, use gradient accumulation, enable `torch.autocast`. Do not reduce scope.

---

## 1. Project Summary

**Goal:** Forecast hourly PM2.5 at station level across Delhi-NCR.  
**Research question:** Does a wind-aware dynamic graph beat a static distance graph, and when?  
**Deliverable:** Reproducible repo + 10–12 page LaTeX report + animated AQI map GIF.  
**Timeline:** 12 weeks.

**Compute reality:**  
- ~70 stations × 5 years hourly × ~10 features ≈ 70 MB in memory.  
- Graph WaveNet ≈ 300K params. Trains in 20–60 min on a 3050 Ti using <2 GB VRAM.  
- Bottleneck: data cleaning, splits, graph construction, evaluation.

---

## 2. Locked Scope

| Item | Decision |
|---|---|
| Region | Delhi NCR: Delhi, Noida, Gurgaon, Ghaziabad, Faridabad |
| Stations | ~70 CPCB/CAAQMS stations |
| Period | 2019-01-01 to 2024-12-31 |
| Target | Hourly PM2.5 (µg/m³) |
| Input | Past 72 hours |
| Output | Next 48 hours |
| Horizons | 1, 3, 6, 12, 24, 48 hours |
| Features | PM2.5 history + weather: wind u/v, temperature, humidity, rainfall |
| Optional feature | Boundary layer height, NASA FIRMS fire counts |
| Train | 2019–2022 |
| Val | 2023 |
| Test | 2024 |
| Splits | Chronological only. Never shuffle. |
| Normalization | `log1p(PM2.5)` then z-score. Fit on train only. |
| Missing values | Mask imputed entries. Never impute using future values. |

Write these into `src/constants.py` so they are enforced by code, not memory.

---

## 3. Repo Structure

```
.
├── AGENTS.md                  # this file
├── README.md
├── requirements.txt
├── environment.yml
├── Makefile
├── configs/
│   ├── baseline_persistence.yaml
│   ├── baseline_hist_avg.yaml
│   ├── baseline_lightgbm.yaml
│   ├── lstm_univariate.yaml
│   ├── lstm_multivariate.yaml
│   ├── transformer.yaml
│   ├── stgcn.yaml
│   └── graph_wavenet.yaml
├── data/
│   ├── raw/                   # gitignored
│   ├── interim/               # gitignored
│   └── processed/             # gitignored, frozen npz
├── notebooks/                 # exploration only, no real logic
├── scripts/
│   ├── pull_cpcb.py
│   ├── pull_openmeteo.py
│   ├── build_dataset.py
│   └── run_experiment.py
├── src/
│   ├── constants.py
│   ├── data/
│   │   ├── acquisition.py
│   │   ├── cleaning.py
│   │   ├── dataset.py
│   │   └── graphs.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── lstm.py
│   │   ├── transformer.py
│   │   ├── stgcn.py
│   │   └── graph_wavenet.py
│   ├── train/
│   │   ├── trainer.py
│   │   └── metrics.py
│   ├── eval/
│   │   ├── evaluate.py
│   │   ├── episodes.py
│   │   └── plots.py
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── checkpoint.py
├── tests/
│   ├── test_no_leakage.py
│   ├── test_shapes.py
│   └── test_masks.py
└── EXPERIMENTS.md
```

All real logic in `src/`. Notebooks only for exploration.

---

## 4. Data Acquisition

### Air Quality
- **Primary:** CPCB via data.gov.in API (free key, hourly station data).
- **Backfill:** Kaggle CPCB historical dumps. Check station ID consistency.
- **Optional:** OpenAQ.
- Log station IDs, lat/lon, timestamps, source.

### Weather
- **Primary:** Open-Meteo historical API. Serves ERA5-derived hourly data per lat/lon. Free, no registration.
- Get wind u/v, temperature, humidity, rainfall.
- **Optional:** ERA5 from Copernicus CDS only if a required variable like boundary layer height is missing from Open-Meteo.
- **Optional:** NASA FIRMS fire counts for stubble-burning season.

### Cleaning Checklist
- Resample to strict hourly grid. Log every gap.
- Drop stations below 60% coverage over period. Expect losses.
- Short gaps <6h: linear interpolation.
- Long gaps: leave missing, mask them.
- Never forward-looking interpolation.
- Remove negative values, stuck-at-constant sensors, impossible spikes.
- Unit consistency: µg/m³.
- Keep Diwali/winter spikes. Do not winsorize away episodes.
- Reuse prior AQ pipeline if available.

---

## 5. Dataset Freeze

Build one tensor and freeze it as `.npz` before modeling.

**Arrays:**
- `X`: `[N, 72, 70, F]` input features
- `y`: `[N, 48, 70]` target PM2.5
- `mask_y`: `[N, 48, 70]` observed=1, imputed=0
- `mask_X`: `[N, 72, 70, F]`
- `timestamps`: `[N, 48]`
- `station_ids`: `[70]`
- `coords`: `[70, 2]` lat/lon
- `feature_names`: list

**Rules:**
- Normalization stats fit on train only.
- Any correlation-based adjacency computed on train only.
- Evaluation only on observed values. Mask imputed entries in loss and metrics.
- Freeze as `data/processed/delhi_ncr_pm25_2019_2024.npz`.
- Do not rebuild after modeling starts.

**Tests:**
- `test_no_leakage.py`: check stats, adjacency, imputation use train only.
- `test_shapes.py`: check tensor shapes.
- `test_masks.py`: check masks align with missing values.

---

## 6. Graph Construction

Implement all as the same interface: dense `70×70` matrix or sparse edge list.

| Graph | Description |
|---|---|
| Distance | Gaussian kernel `exp(-d²/σ²)`, tune σ, also try kNN thresholding |
| Correlation | Pairwise correlation of historical PM2.5 on train only, thresholded |
| Wind-aware directed | Edge i→j weighted by wind alignment from i to j, historical or seasonal prevailing winds. **Core contribution.** |
| Learned adaptive | Graph WaveNet self-adaptive adjacency: `A = softmax(ReLU(E₁E₂ᵀ))` |

The comparison across these is the **main ablation table** and the main claim to contribution. Report honestly if wind-aware edges do not beat distance.

---

## 7. Models — Exact Order

Do not skip. Log every model.

1. **Persistence** — `ŷ = last value`.
2. **Hour-of-week historical average** — floor baseline.
3. **LightGBM/XGBoost per station** — lag + weather features. Often surprisingly strong.
4. **Univariate LSTM per station** — shared weights.
5. **Multivariate LSTM** — all 70 stations as input channels, no graph. **Critical baseline.** It sees the same information as the GNN. Answers: does graph structure add value, or just more inputs?
6. **Plain transformer** — temporal attention, stations as channels.
7. **STGCN** — on fixed adjacencies.
8. **Graph WaveNet** — with adaptive adjacency. Flagship. Adapt official repo `nnzhan/Graph-WaveNet` (plain PyTorch, simple npz format).
9. **Optional:** DCRNN if time allows.

Also implement one tiny GCN+GRU from scratch (~150 lines) for understanding. You will be interviewed on what message passing does.

---

## 8. Training Protocol

| Setting | Value |
|---|---|
| Loss | Masked MAE. Huber variant if MAE underfits peaks. |
| Optimizer | Adam |
| Learning rate | 2e-4 – 3e-3 |
| Batch size | 64 (reduce if OOM) |
| Dropout | 0.3 |
| Early stopping | Val MAE, patience 15 |
| Gradient clipping | 1.0 |
| Precision | Mixed precision if memory tight |
| Seeds | 3 random seeds. Report mean ± std. |
| Checkpoint | Every epoch. Make runs resumable. |
| Evaluation space | Original µg/m³ (de-logged, de-normalized) |
| Config | YAML per experiment |

Develop and debug locally. Run full experiments on laptop or Colab/Kaggle. Checkpoint to Google Drive if on Colab.

---

## 9. Evaluation & Analysis

### Main Table
All models × horizons. MAE, RMSE, R². Mean ± std over 3 seeds.

### Episode Analysis
Define episodes: PM2.5 > 150 µg/m³ sustained ≥ 6h.  
Report metrics **during episodes separately**.  
Headline: “GNN improves X% overall but Y% during severe episodes” or the reverse.

### Case Studies
Pick 2–3 real events:
- Winter inversion week
- Diwali
- Dust storm

Plot predictions vs truth per station.

### Spatial Error Maps
Use folium/plotly. Per-station error. Does GNN help peripheral stations most? That is a mechanism claim.

### Adjacency Introspection
Plot learned adaptive adjacency vs distance. Scatter edge weight vs km.  
Do learned edges align with prevailing wind direction? This figure makes the report feel like research.

### Ablations
- Static vs wind-aware graph
- With vs without meteorological features
- Different graph thresholds
- Correlation vs distance
- Optional: leave-station-out evaluation
- Optional: uncertainty estimates (quantile loss)

### Failure Analysis
Where does it miss, and why? Sudden spikes, sensor outages.

---

## 10. Deliverables

| Item | Details |
|---|---|
| LaTeX report | 10–12 pages. Related work, data, methods, results, honest discussion, conclusion. Target arXiv/workshop. |
| Repo README | Results table, one-command reproduce, configs. |
| Animated map | Predicted vs actual PM2.5 across stations. GIF. |
| Optional demo | Small Streamlit app. |
| EXPERIMENTS.md | Every run logged. |

Report structure: related work (STGCN, DCRNN, Graph WaveNet, PM2.5-GNN, Ada-MSTNet), data, methods, results, failures, conclusion.

---

## 11. Timeline — 12 Weeks

| Weeks | Work |
|---|---|
| 1–2 | Data pipeline end-to-end. Dataset frozen. Tests pass. |
| 3 | Baselines 1–3 logged. |
| 4 | LSTM baselines 4–5 logged. |
| 5 | Transformer baseline. |
| 6–7 | STGCN + Graph WaveNet adapted and training. |
| 8 | Adjacency ablations. Seeds. Final tables. |
| 9 | Episode analysis, figures, maps. |
| 10 | Failure analysis, adjacency introspection. |
| 11–12 | LaTeX report, README, animated GIF. |

Write the report while final experiments run. Do not leave writing to the end.

---

## 12. Classic Ways This Goes Wrong

- Random splits or leakage via normalization/imputation → invalid results.
- Data pipeline rebuilt three times → freeze dataset early.
- Five half-trained models → depth over breadth.
- Zero baselines → GNN RMSE means nothing.
- Running out of time before writing → the write-up is the product.
- Comparing models with different input features.
- Overfitting to one winter.
- MAPE on low values → blows up. Avoid or handle carefully.

---

## 13. First Tasks for Coding Agent

1. Create repo structure and `AGENTS.md` (this file).
2. Write `src/constants.py` with splits, horizons, paths, features.
3. Write `scripts/pull_cpcb.py` and `scripts/pull_openmeteo.py`.
4. Write `src/data/cleaning.py` and `src/data/dataset.py`.
5. Build `data/processed/delhi_ncr_pm25_2019_2024.npz`.
6. Write and run tests: no leakage, shapes, masks.
7. Implement baselines 1–3. Log results in `EXPERIMENTS.md`.
8. Implement LSTM baselines 4–5.
9. Implement STGCN and Graph WaveNet.
10. Run ablations, evaluation, plots.
11. Write report and README.

**Stop and document** if a data source is unavailable. Use fallback. Do not change scope without updating this file.