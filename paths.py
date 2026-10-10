"""Paths/configuration for the official OFFSET production pipeline.

The network inventories are read-only.  Prefer copies in
MOE using offsets/Input/Network. For backwards compatibility, when they
have not yet been copied, use the original verified inventory location.
No GPS calculation is part of this production workflow.
"""

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent  # MOE using offsets
OFFSET_ROOT = CODE_DIR                    # MOE using offsets
PROJECT_ROOT = OFFSET_ROOT.parent            # Streetlight-Data-Processing
LEGACY_NETWORK_DIR = PROJECT_ROOT / "MOE for selected links" / "Results" / "Network_MOE"
NETWORK_INPUT_DIR = OFFSET_ROOT / "Input" / "Network"
OUTPUT_DIR = OFFSET_ROOT / "Results" / "Network_MOE"


def network_source(filename):
    """Use local verified network inventory when present, otherwise historic saved copy."""
    local = NETWORK_INPUT_DIR / filename
    return local if local.is_file() else LEGACY_NETWORK_DIR / filename


CV_FILE = Path(os.environ.get(
    "OFFSET_MOE_CV_FILE", str(PROJECT_ROOT / "Output" / "Final" / "9_20_2025" / "9_20_2025_hr=19.csv")
))
PARENT_FILE = Path(os.environ.get("OFFSET_MOE_PARENT_FILE", str(network_source("parent_links.parquet"))))
SEGMENT_FILE = Path(os.environ.get("OFFSET_MOE_SEGMENT_FILE", str(network_source("moe_segments_500ft.parquet"))))
MOE_FILE = OUTPUT_DIR / "segment_moe_500ft_15min_offset.parquet"
JOURNEY_PARTITION_DIR = OUTPUT_DIR / "CV_Journey_Partitions"
MOE_PARTIAL_DIR = OUTPUT_DIR / "MOE_Parallel_Parts"
# Historical reference, used only when the optional comparison is run by hand.
GPS_MOE_FILE = LEGACY_NETWORK_DIR / "segment_moe_500ft_15min_gps.parquet"

SEGMENT_LENGTH_FT = 500.0
TIME_BIN_MINUTES = 15
SLOW_SPEED_THRESHOLD_MPH = 5.0
MAX_WAYPOINT_GAP_SECONDS = 120
CSV_CHUNK_SIZE = 500_000
PARQUET_COMPRESSION = "snappy"
REPORT_INTERVAL_SECONDS = 5
JOURNEY_PARTITIONS = 32
PARALLEL_WORKERS = min(12, os.cpu_count() or 1)
FINAL_SEGMENT_BATCH_SIZE = 100_000


def source_summary():
    return {"matched_cv": CV_FILE, "parent_links": PARENT_FILE,
            "segment_inventory": SEGMENT_FILE, "offset_parquet": MOE_FILE,
            "output_dir": OUTPUT_DIR}


def check_inputs():
    for label, file in source_summary().items():
        if label in ("matched_cv", "parent_links", "segment_inventory") and not file.is_file():
            raise FileNotFoundError(f"Required offset MOE source missing ({label}): {file}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
