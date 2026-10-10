"""Compare GPS and offset MOEs on the identical segment-time inventory."""

import json
from itertools import zip_longest

import numpy as np
import pyarrow.parquet as pq

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths


METRICS = [
    "avg_speed_mph", "speed_reduction_pct", "slow_movement_pct", "DSH",
    "avg_travel_time_s", "speed_estimated_travel_time_s",
]
COLUMNS = ["moe_segment_id", "time_bin", "waypoint_count"] + METRICS


# Compare cells without reading the two network-sized tables into memory.
def compare():
    sources = [paths.GPS_MOE_FILE, paths.MOE_FILE]
    for file in sources:
        if not file.is_file():
            raise FileNotFoundError(file)
    left, right = (pq.ParquetFile(file) for file in sources)
    if left.metadata.num_rows != right.metadata.num_rows:
        raise ValueError("Position methods have different segment-time inventories")
    summary = {
        "segment_time_rows": 0,
        "occupied_gps": 0,
        "occupied_offset": 0,
        "different_observation_counts": 0,
        "gps_waypoints": 0,
        "offset_waypoints": 0,
        "metrics": {name: {"availability_differs": 0, "value_differs_over_0_001": 0,
                           "mean_absolute_difference": 0.0} for name in METRICS},
    }
    differences = dict.fromkeys(METRICS, 0.0)
    comparisons = dict.fromkeys(METRICS, 0)
    batches = zip_longest(
        left.iter_batches(batch_size=100_000, columns=COLUMNS),
        right.iter_batches(batch_size=100_000, columns=COLUMNS),
    )
    for gps_batch, offset_batch in batches:
        if gps_batch is None or offset_batch is None:
            raise ValueError("Position methods produced different batch counts")
        gps, offset = gps_batch.to_pandas(), offset_batch.to_pandas()
        if not gps[["moe_segment_id", "time_bin"]].equals(
            offset[["moe_segment_id", "time_bin"]]
        ):
            raise ValueError("Position methods have different cell ordering")
        summary["segment_time_rows"] += len(gps)
        a = gps["waypoint_count"].fillna(0).to_numpy(float)
        b = offset["waypoint_count"].fillna(0).to_numpy(float)
        summary["gps_waypoints"] += int(a.sum())
        summary["offset_waypoints"] += int(b.sum())
        summary["occupied_gps"] += int(np.count_nonzero(a > 0))
        summary["occupied_offset"] += int(np.count_nonzero(b > 0))
        summary["different_observation_counts"] += int(np.count_nonzero(a != b))
        for name in METRICS:
            a, b = gps[name].to_numpy(float), offset[name].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            detail = summary["metrics"][name]
            detail["availability_differs"] += int(np.count_nonzero(np.isfinite(a) != np.isfinite(b)))
            delta = np.abs(a[both] - b[both])
            detail["value_differs_over_0_001"] += int(np.count_nonzero(delta > .001))
            differences[name] += float(delta.sum())
            comparisons[name] += int(len(delta))
    for name in METRICS:
        summary["metrics"][name]["mean_absolute_difference"] = (
            differences[name] / comparisons[name] if comparisons[name] else None
        )
    output = paths.OUTPUT_DIR / "moe_position_method_comparison.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    compare()
