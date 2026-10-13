from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

CERRA_MULTI_LEVEL_DIR = Path("/Data/gfi/share/Reanalyses/CERRA")
WIND_TRACKER_CSV = DATA_DIR / "Global-Wind-Power-Tracker-February-2026.csv"
WIND_TRACKER_BELOW_CSV = DATA_DIR / "Global-Wind-Power-Tracker-February-2026-Below_Threshold.csv"
WIND_OUTPUT_CSV = REPO_ROOT / "output" / "norway_wind_power_2025_weather_2026_capacity.csv"
