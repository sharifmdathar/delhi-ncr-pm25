#!/usr/bin/env python3
"""Build a tiny sample dataset, so a reviewer can run the pipeline in minutes.

The full dataset needs a 745 MB download per year, which is a real barrier to
someone who just wants to see the code work.  This builds a miniature version
of the same pipeline: eight stations, one winter month, the same cleaning,
the same train-only normalisation, the same split logic.

    python scripts/make_sample_dataset.py

writes

    data/sample/sample_dataset.npz      the frozen arrays
    data/sample/README.md                what it is and how to use it

The sample is a Derived Database of the CPCB records and is offered under the
Open Database Licence 1.0, with attribution, as the licence requires.  See
LICENCE.md.
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "sample"
OUT_NPZ = OUT_DIR / "sample_dataset.npz"

# Eight stations spread over Delhi, Noida and Ghaziabad, chosen for good
# coverage of the sample month.  The network has to be wide enough for the
# distance and wind-aware graphs to be meaningful.
SAMPLE_STATIONS = [
    "site_103",   # CRRI Mathura Road, Delhi
    "site_106",   # IGI Airport, Delhi
    "site_111",   # Sector 62, Noida
    "site_117",   # ITO, Delhi
    "site_124",   # R K Puram, Delhi
    "site_1421",  # Dr. Karni Singh Shooting Range, Delhi
    "site_144",   # Vasundhara, Ghaziabad
    "site_153",   # Sector 125, Noida
]

START = "2024-11-01"
END = "2025-01-31 23:59:59"

# The sample is split the same way as the full dataset, scaled down: 57 days
# of training, 4 days of validation, 31 days of test.  The test month is
# January 2025, which contains the study's most severe episode.
TRAIN_END = "2024-12-25 23:59:59"
VAL_END = "2024-12-31 23:59:59"
TEST_START = "2025-01-01 00:00:00"


def main():
    src_dir = RAW / "cpcb_station_csv"
    if not src_dir.exists():
        raise SystemExit(
            f"{src_dir} not found. Run scripts/fetch_cpcb_years.py first, or "
            "see data/sample/README.md for the pre-built sample.")

    frames = []
    for sid in SAMPLE_STATIONS:
        p = src_dir / f"{sid}.csv"
        if not p.exists():
            print(f"  (skipping {sid}: {p.name} not found)")
            continue
        d = pd.read_csv(p, parse_dates=["timestamp"])
        d = d[(d["timestamp"] >= START) & (d["timestamp"] <= END)]
        if len(d):
            frames.append(d)
    if not frames:
        raise SystemExit("no sample stations found")
    df = pd.concat(frames, ignore_index=True)
    print(f"Raw sample: {len(df):,} rows, {df['station_id'].nunique()} stations, "
          f"{df['timestamp'].min()} .. {df['timestamp'].max()}")

    # ---- the same cleaning, with the sample's own period and split -------
    # NB: src.data.cleaning and src.data.dataset import the constants BY
    # VALUE (`from src.constants import ...`), so patching src.constants is
    # not enough -- every module that holds a copy has to be patched.
    import src.constants as C
    from src.data import cleaning as CL
    from src.data import dataset as DS

    saved = (C.START_DATE, C.END_DATE, C.TRAIN_END, C.VAL_END, C.TEST_START, C.TEST_END)
    new = (START, END, TRAIN_END, VAL_END, TEST_START, END)
    for mod in (C, CL, DS):
        for name, val in zip(
                ("START_DATE", "END_DATE", "TRAIN_END", "VAL_END",
                 "TEST_START", "TEST_END"), new):
            if hasattr(mod, name):
                setattr(mod, name, val)
    try:
        # Restrict to the sample stations BEFORE cleaning, so the coverage
        # filter, the normalisation statistics and the frozen arrays all
        # describe the same eight stations.
        cleaner = CL.DataCleaner(seed=42, stations=SAMPLE_STATIONS)
        df_clean = cleaner.get_dataframe()
        print(f"\nSample stations after the coverage filter: "
              f"{df_clean['station_id'].nunique()} "
              f"({sorted(df_clean['station_id'].unique())})")
        ds = DS.build_dataset(df_clean, save_path=str(OUT_NPZ))
    finally:
        for mod in (C, CL, DS):
            for name, val in zip(
                    ("START_DATE", "END_DATE", "TRAIN_END", "VAL_END",
                     "TEST_START", "TEST_END"), saved):
                if hasattr(mod, name):
                    setattr(mod, name, val)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nm, ns = float(ds["norm_mean"]), float(ds["norm_std"])
    y = np.expm1(np.clip(ds["y_full"] * ns + nm, -700, 700))
    obs = ds["mask_y"] > 0
    print(f"\nWrote {OUT_NPZ.relative_to(PROJECT_ROOT)}")
    print(f"  X_full {ds['X_full'].shape}   y_full {ds['y_full'].shape}")
    for split in ("train", "val", "test"):
        idx = ds[f"split_indices_{split}"]
        v = y[idx][obs[idx]]
        print(f"  {split:5s} {len(idx):5d} h  observed {obs[idx].mean():.1%}  "
              f"mean PM2.5 {v.mean():.1f}")

    (OUT_DIR / "README.md").write_text(f"""# Sample dataset

A miniature version of the full dataset, for reviewers who want to run the
pipeline without a 745 MB-per-year download.

- {ds['X_full'].shape[1]} stations, {ds['X_full'].shape[0]} hourly timestamps
- {START} to 2025-01-31
- split: train to {TRAIN_END[:10]}, validation to {VAL_END[:10]}, test from {TEST_START[:10]}

Built by `scripts/make_sample_dataset.py`, which runs the same cleaning,
normalisation (train-only statistics) and split logic as the full build.

## Licence and attribution

This is a **Derived Database** of the Central Pollution Control Board (CPCB)
continuous monitoring records, Ministry of Environment, Forest and Climate
Change, Government of India (<https://cpcb.gov.in/>), packaged by the
`Vonter/india-cpcb-aqi` project, and is **offered under the Open Database
Licence 1.0** (<http://opendatacommons.org/licenses/odbl/1.0/>) as that
licence requires for adapted versions.

## Using it

```bash
python scripts/evaluate.py --split test --data data/sample/sample_dataset.npz
python scripts/train_all.py --data data/sample/sample_dataset.npz \\
    --models stgcn --graph identity --seeds 42 --epochs 4
```

The full dataset is built with `scripts/fetch_cpcb_years.py` and
`scripts/build_dataset.py`; see the top-level README.
""")

    meta = {
        "stations": [str(s) for s in ds["station_ids"]],
        "start": START, "end": END,
        "train_end": TRAIN_END, "val_end": VAL_END, "test_start": TEST_START,
        "licence": "ODbL 1.0 (Derived Database of the CPCB records)",
    }
    (OUT_DIR / "sample_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {OUT_DIR/'README.md'}")


if __name__ == "__main__":
    main()
