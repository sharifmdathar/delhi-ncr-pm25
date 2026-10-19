# Delhi-NCR PM2.5 Forecasting — Constants
# Locked scope. Do not change without updating AGENTS.md and EXPERIMENTS.md.

# --- Regions & Stations ---
REGION = "delhi_ncr"
STATIONS = 54  # Delhi-NCR stations from CPCB (site_* ids); filtered by coverage at build time
# Data source: Vonter/india-cpcb-aqi GitHub releases (CPCB 15-min, 2009-2025)

# --- Period ---
START_DATE = "2019-01-01"
END_DATE = "2025-12-31"
TRAIN_END = "2022-12-31"      # inclusive (4 years, 4 winters)
VAL_END = "2023-12-31"        # inclusive (1 year, 1 winter)
TEST_START = "2024-01-01"     # inclusive (2 years, 2 winters)
TEST_END = "2025-12-31"       # inclusive (2025)

# --- Horizons ---
HORIZONS = [1, 3, 6, 12, 24, 48]  # hours ahead to predict

# --- Input/Output windows ---
INPUT_HOURS = 72   # past context
OUTPUT_HOURS = 48  # future target

# --- Features ---
# Core features: PM2.5 history + weather
FEATURE_NAMES = [
    "pm25",           # target, historical
    "temp_c",         # temperature (°C)
    "rh",             # relative humidity (%)
    "wind_u",         # wind u component (m/s)
    "wind_v",         # wind v component (m/s)
    "rainfall",       # rainfall (mm)
]

NUM_CORE_FEATURES = len(FEATURE_NAMES)  # = 5

# --- Normalization ---
# log1p then z-score, fit on train only
NORMALIZE_TARGET = True  # apply log1p + z-score to PM2.5
NORMALIZE_FEATURES = True  # optionally normalize weather features too

# --- Graph Construction ---
# Graph adjacency matrices are N×N (one per station)
# All graphs built on train-split data only — never use val/test info.

# Distance graph: Gaussian kernel on station coordinates
DISTANCE_SIGMA = None  # auto-tune (median distance) or set explicit value

# Correlation graph: threshold on pairwise PM2.5 correlation (train only)
CORRELATION_THRESHOLD = 0.3  # keep edges with |corr| >= threshold

# Wind-aware directed graph: edge i→j weighted by wind alignment
# Will be computed from historical wind vectors

# Learned adaptive: Graph WaveNet adjacency E₁E₂ᵀ inside the model

# --- Data Freeze ---
FROZEN_DATA_PATH = "data/processed/delhi_ncr_pm25_2019_2025.npz"

# --- Data Quality ---
# Minimum fraction of real (non-imputed) observations required in each split
# per station. If any station in train/val/test falls below, build fails.
MIN_REAL_DATA_FRACTION = 0.70

# --- Training ---
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 15
GRAD_CLIP = 1.0
LR = 3e-4
DROPOUT = 0.3
SEEDS = [42, 123, 456]  # 3 random seeds for reporting mean ± std

# --- Evaluation ---
HORIZON_NAMES = {1: "1h", 3: "3h", 6: "6h", 12: "12h", 24: "24h", 48: "48h"}

# --- Episode Analysis ---
EPISODE_PM25_THRESHOLD = 150  # µg/m³
EPISODE_DURATION_HOURS = 6     # sustained ≥ this many hours

# --- Paths ---
PROJECT_ROOT = "/home/doormat/repos/gnn"