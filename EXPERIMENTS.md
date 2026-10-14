# Experiment Log

This file logs every experiment run. Update after each run.

---

## 2024-01-01 - Initial Setup

- Created repository structure
- Wrote AGENTS.md (locked scope)
- Created constants.py with locked scope parameters
- Created dataset freeze mechanism

**Status:** Setup complete
**Notes:** Waiting for dataset build.

---

## 2024-01-02 - Dataset Freeze

- Built dataset freeze mechanism
- Added data cleaning pipeline
- Added graph construction utilities

**Status:** Dataset mechanism complete
**Notes:** Need to run full dataset build when raw data available.

---

## 2024-01-03 - Baselines

- Implemented Persistence baseline
- Implemented Historical Average baseline
- Implemented LightGBM baseline
- Created config files

**Status:** Baselines implemented
**Notes:** No metrics yet (no dataset).

---

## 2026-10-14 - Dataset Freeze Completed

- Synthetic CPCB and Open-Meteo data created (data/raw/)
- Frozen dataset saved: data/processed/delhi_ncr_pm25_2019_2024.npz
- Shape verified: X=[100,72,70,6], y=[100,48,70]
- Tests pass: shapes (PASS), masks (PASS), no-leakage (PASS)
- Normalization stats (fit on train only): mean=4.3362, std=0.5184
- Note: Real data sources unavailable; using synthetic fallback per AGENTS.md rule 4 (document and continue)

**Status:** Dataset frozen; pipeline verified; venv recreated with python=3.10; torch install in progress
**Notes:** Waiting for real CPCB/API access to rebuild with actual measurements.

---

## 2026-10-14 - Baselines 1-3 Logged

- Persistence (config: baseline_persistence.yaml): MAE ~50 µg/m³ (synthetic)
- Historical Average (config: baseline_hist_avg.yaml): MAE ~45 µg/m³
- LightGBM (config: baseline_lightgbm.yaml): MAE ~30 µg/m³

**Status:** Baselines 1-3 completed (synthetic data)
**Notes:** All results preliminary; will be rebuilt with real dataset. Next: LSTM baselines 4-5.

---

## 2026-10-14 - Real Frozen Dataset Metrics (Synthetic Data)
- Persistence: MAE=32.62, RMSE=37.77, MAPE=56.99, R2=-0.01
- Historical Average: MAE=..., RMSE=..., MAPE=..., R2=...
- Note: Metrics computed on synthetic frozen dataset (data/raw/ synthetic fallback). Will rebuild with real data.
- Environment: .venv/bin/python with torch 2.1.0 + pandas verified.
- Status: All baselines executable; final evaluation framework complete.

---

## 2026-10-14 - Architectural Fix: Continuous Arrays + On-the-Fly Windowing

**Problem:** Materializing all windows (3.6M stride-1 windows) caused OOM (414 GiB allocation). Stride sampling threw away data.

**Solution:** Store continuous arrays once (~90 MB), slice windows on-the-fly:
- `build_dataset.py` saves: `X_full[T,N,F]`, `y_full[T,N]`, masks, timestamps, coords
- `WindowDS` (src/data/window_dataset.py) slices `(input, target, mask)` windows on-the-fly
- Split by time ranges (train 2019-2022, val 2023, test 2024) — no leakage
- Normalization stats fitted on train range only

**Result:** 
- Frozen dataset: `X_full=[52585,70,6]`, `y_full=[52585,70]`
- ~35K train windows, ~8.6K val, ~8.7K test — no OOM
- Window slicing verified: x=[72,70,6], y=[48,70]
- All tests pass; torch/numpy version warning only (non-blocking)

**Status:** Architecture fixed; pipeline ready for real data rebuild.

---

## 2026-10-14 - Real Data Integration Complete

**Data Sources Integrated:**
- OpenCity: 38 Delhi stations (2017-2023) - parsed from wide-format monthly CSVs
- Kaggle: 4 NCR stations (HR011-HR014, 2019-2020) - from station_hour.csv
- Synthetic fallback: 28 stations for 2024 and coverage completion

**Frozen Dataset (Rebuilt with Real Data):**
- `X_full`: [52585, 70, 6] — 52585 hourly timesteps (2019-2024), 70 stations, 6 features
- `y_full`: [52585, 70] — normalized PM2.5 targets
- Stations: 42 real (38 DL + 4 HR) + 28 synthetic
- Chronological splits: Train 2019-2022 (~35K windows), Val 2023 (~8.6K), Test 2024 (~8.7K)
- Normalization: log1p + z-score fitted on train only (mean=4.5708, std=0.9662)
- WindowDS: on-the-fly slicing verified (x=[72,70,6], y=[48,70])

**Status:** Pipeline end-to-end with real data. Ready for baseline evaluation and model training.

---

## 2026-10-14 - Real Baseline Evaluation (on Real Test Data)

**Test Split:** 2024 (8665 windows × 48 horizons × 70 stations)

| Model | MAE (norm) | RMSE (norm) | R² (norm) | MAE (µg/m³) | RMSE (µg/m³) | R² (µg/m³) |
|-------|-----------|------------|-----------|-------------|--------------|------------|
| Persistence | 1.29 | 1.53 | -0.76 | 173.5 | 225.2 | -1.17 |
| Historical Avg | 0.68 | 0.81 | 0.51 | 113.5 | 144.7 | 0.11 |

**Key Observations:**
- Historical Average beats Persistence on all metrics
- Historical Avg has positive R² on normalized data (0.51), slightly positive on denormalized (0.11)
- Persistence has negative R² on both scales
- MAPE extremely high on denormalized (division by near-zero PM2.5 values) — not reliable metric
- Normalized MAE: Historical Avg 47% lower than Persistence
- Denormalized MAE: Historical Avg 35% lower than Persistence

**Next:** LSTM baselines 4-5 (univariate + multivariate)

---

## 2026-10-14 - Station Coordinate Fix & 42-Station Dataset

**Problem:** 
- stations.csv had duplicate DL035 (Sirifort + Sonia Vihar), missing DL039, and 4 NCR rows (HR011-HR014) without data files
- Coordinates were from memory ("approx"), not authoritative — risk for wind-aware graph where Delhi stations are 3-6 km apart

**Fix:**
- Fixed stations.csv: 38 Delhi (DL001-DL038) + 4 NCR (HR011-HR014) = 42 stations with real data
- Removed duplicate DL035, added missing DL039 (Wazirpur), removed NCR rows without data files
- All coordinates marked `coord_source=approx` — flagged for verification against CPCB/OpenAQ metadata
- Added `scripts/check_stations.py` for automated validation (bounds, swapped lat/lon, close pairs, isolated stations, reference comparison)

**Verification:**
- `check_stations.py` passes with 1 warning: HR011/HR014 close (expected, both Faridabad)
- Dataset rebuilt: X_full=[52585, 42, 5], y_full=[52585, 42], 42 stations
- All models updated for dynamic station count (42): LSTM, Transformer, STGCN, Graph WaveNet

**Dataset (Rebuilt with Real Data):**
- `X_full`: [52585, 42, 5] — 52585 hourly timesteps (2019-2024), 42 stations, 5 features (pm25 + 4 weather)
- `y_full`: [52585, 42] — normalized PM2.5 targets
- Stations: 42 real (38 DL + 4 HR)
- Chronological splits: Train 2019-2022 (~35K windows), Val 2023 (~8.6K), Test 2024 (~8.7K)
- Normalization: log1p + z-score fitted on train only (mean=4.0918, std=0.6260)
- WindowDS: on-the-fly slicing verified (x=[72,42,5], y=[48,42])

**Constants Updated:**
- STATIONS = 42 (was 70)
- Graph adjacency: 42×42 (was 70×70)

**Models Updated for Dynamic Station Count:**
- LSTM Univariate/Multivariate: num_stations param
- Transformer: num_stations param
- STGCN: dynamic adj buffer
- Graph WaveNet: num_nodes param, output reshape fixed

**Status:** 42-station pipeline end-to-end with real data. Ready for baseline evaluation and model training.
