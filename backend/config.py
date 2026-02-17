import os
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
LIDAR_DIR = BASE_DIR / "lidar"
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = DATA_DIR / "cities.db"

NOMINATIM_USER_AGENT = "citybuilder-app/1.0"
NOMINATIM_TIMEOUT = 30

# Set CITYBUILDER_NO_CACHE=1 to bypass the processed-feature cache
USE_CACHE = os.environ.get("CITYBUILDER_NO_CACHE", "").strip() not in ("1", "true", "yes")
