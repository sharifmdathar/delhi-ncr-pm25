#!/usr/bin/env python3
"""Build and freeze the continuous dataset.

Usage:
    python scripts/build_dataset.py

This will:
  1. Load raw CPCB and weather data
  2. Clean the data (follows AGENTS.md rules)
  3. Build continuous arrays (no window materialization)
  4. Save as compressed NPZ file
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

# Add src to path so we can import modules
sys.path.append(str(SRC_DIR))

from src.data import cleaning, dataset


def main():
    print("Building Delhi-NCR PM2.5 dataset...")
    
    # Load and clean data
    df = cleaning.cleaner.get_dataframe()
    df_clean = cleaning.cleaner.clean(df)
    
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