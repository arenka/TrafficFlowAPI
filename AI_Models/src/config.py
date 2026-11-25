"""
Configuration constants for the METR-LA Dynamic Traffic Flow Prediction project.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"

for _d in (MODELS_DIR, PLOTS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

H5_PATH = DATA_DIR / "metr-la.h5"
LOCATIONS_PATH = DATA_DIR / "graph_sensor_locations.csv"
DISTANCES_PATH = DATA_DIR / "distances_la_2012.csv"

# Temporal framing (5-minute intervals)
INPUT_SEQ_LEN = 12          # 60 minutes of historical observations
HORIZONS = [3, 6, 12]       # 15 min, 30 min, 60 min ahead
HORIZON_LABELS = {3: "15 min", 6: "30 min", 12: "60 min"}

# Chronological split (standard METR-LA benchmark protocol)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Speed unit in METR-LA is mph (typical range 0 - 70 mph)
CONGESTION_SEVERE_MPH = 25.0
CONGESTION_MODERATE_MPH = 45.0

RANDOM_SEED = 42
