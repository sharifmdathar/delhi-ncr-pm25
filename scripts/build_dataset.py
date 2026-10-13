#!/usr/bin/env python3
"""Build and freeze the dataset.

Usage:
    python scripts/build_dataset.py

This will:
  1. Load raw CPCB and weather data
  2. Clean the data (follows AGENTS.md rules)
  3. Build dataset windows
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
    df = cleaning.clener.load_dataframe()
    df_clean = cleaning.clener.clean(df)
    
    # Build dataset
    ds = dataset.build_dataset(df_clean)
    
    print("Dataset built successfully!")


if __name__ == "__main__":
    main()