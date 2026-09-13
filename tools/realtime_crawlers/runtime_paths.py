"""Shared runtime-data paths for real-time crawlers.

Keep generated data outside the Git repository. Source code lives under
E:\AAAqian\code\storm_surge_clean, while crawler outputs and databases live
under E:\AAAqian\storm_surge_runtime_data.
"""

from __future__ import annotations

import os
from pathlib import Path

RUNTIME_ROOT = Path(os.environ.get("STORM_SURGE_RUNTIME_DATA", r"E:\AAAqian\storm_surge_runtime_data"))

ZHEJIANG_WATER_DATA_DIR = RUNTIME_ROOT / "zhejiang_water_data"
ZHEJIANG_WATER_DB_PATH = ZHEJIANG_WATER_DATA_DIR / "zhejiang_water_data.db"
ZHEJIANG_WATER_EXPORTS_DIR = ZHEJIANG_WATER_DATA_DIR / "exports"
ZHEJIANG_WATER_DEBUG_DIR = ZHEJIANG_WATER_DATA_DIR / "debug"
ZHEJIANG_WATER_LOGS_DIR = ZHEJIANG_WATER_DATA_DIR / "logs"
ZHEJIANG_WATER_BACKUPS_DIR = ZHEJIANG_WATER_DATA_DIR / "backups"

TYPHOON_DATA_DIR = RUNTIME_ROOT / "typhoon_data"
NMC_OUTPUT_DIR = TYPHOON_DATA_DIR / "nmc"
GFS_OUTPUT_DIR = TYPHOON_DATA_DIR / "gfs"
NMEFC_OUTPUT_DIR = TYPHOON_DATA_DIR / "nmefc"
CWA_OUTPUT_DIR = TYPHOON_DATA_DIR / "cwa"
WENZHOU_OUTPUT_DIR = RUNTIME_ROOT / "wenzhou_open_data"
TIDE_SOURCE_MATCHES_PATH = TYPHOON_DATA_DIR / "tide_source_matches.csv"
BAVI_TIDE_CANDIDATES_PATH = TYPHOON_DATA_DIR / "bavi_nearby_tide_station_candidates.csv"
