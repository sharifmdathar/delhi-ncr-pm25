# Sample dataset

A miniature version of the full dataset, for reviewers who want to run the
pipeline without a 745 MB-per-year download.

- 8 stations, 2208 hourly timestamps
- 2024-11-01 to 2025-01-31
- split: train to 2024-12-25, validation to 2024-12-31, test from 2025-01-01

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
python scripts/train_all.py --data data/sample/sample_dataset.npz \
    --models stgcn --graph identity --seeds 42 --epochs 4
```

The full dataset is built with `scripts/fetch_cpcb_years.py` and
`scripts/build_dataset.py`; see the top-level README.
