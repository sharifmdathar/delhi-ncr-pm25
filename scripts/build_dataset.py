#!/usr/bin/env python3
"""Build and freeze the continuous dataset.

Usage:
    python scripts/build_dataset.py

This will:
  1. Load raw CPCB and weather data (REAL DATA ONLY)
  2. Clean the data (follows AGENTS.md rules) - fails if coverage < 70%
  3. Build continuous arrays (no window materialization)
  4. Save as compressed NPZ file + coverage report
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

# Add the project root to sys.path so `src.*` imports resolve
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import cleaning, dataset


def main():
    print("Building Delhi-NCR PM2.5 dataset (REAL DATA ONLY)...")
    
    # Load and clean data - fails if any station is below MIN_REAL_DATA_FRACTION
    cleaner = cleaning.DataCleaner(seed=42)
    df_clean = cleaner.get_dataframe()
    
    # Print and save coverage report
    if hasattr(cleaner, 'coverage_report') and cleaner.coverage_report is not None:
        import pandas as pd
        cov_df = pd.DataFrame(cleaner.coverage_report)
        if not cov_df.empty:
            cov_path = PROJECT_ROOT / "data" / "processed" / "coverage_report.csv"
            cov_path.parent.mkdir(parents=True, exist_ok=True)
            cov_df.to_csv(cov_path, index=False)
            print(f"Coverage report saved: {cov_path}")
    
    # Build dataset (saves frozen NPZ)
    ds = dataset.build_dataset(df_clean)
    
    print("Dataset built successfully!")
    print(f"  X_full: {ds['X_full'].shape}")
    print(f"  y_full: {ds['y_full'].shape}")
    # Estimate windows per split
    for split in ["train", "val", "test"]:
        key = f"split_indices_{split}"
        if key in ds:
            n = len(ds[key]) - 120
            print(f"  {split.capitalize()} windows: ~{max(0, n)}")


if __name__ == "__main__":
    main()