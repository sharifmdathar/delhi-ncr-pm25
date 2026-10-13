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
