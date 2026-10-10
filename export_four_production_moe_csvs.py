"""Export the FOUR production-wide compact CSVs from the completed OFFSET MOE Parquet.

This is an export-only stage: it does not reassign CV points, recompute DSH,
recompute slow movement, or alter the saved offset MOE Parquet.

A distinct journey_id in a link x 500-ft section x 15-min cell counts as
one vehicle passage (unique_veh_count = saved journey_count). If the same
physical vehicle returns as a new journey, it counts again.

Run on the original Windows project (defaults resolve relative to this script):
  python export_four_production_moe_csvs.py \
    --expected-total 18869088 --expected-observed 9777233 \
    --expected-waypoints 44150021
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "Results" / "Network_MOE" / "segment_moe_500ft_15min_offset.parquet"
DEFAULT_OUTPUT = ROOT / "Results" / "Network_MOE" / "Compact_CSV_500ft_15min_offset"

COMMON = [
    "link_key", "segment_section_ft", "segment_length_ft", "start_time",
    "end_time", "waypoint_count", "unique_veh_count",
]
OUTPUTS = {
    "speed_reduction.csv": ["avg_speed_mph", "posted_speed_limit_mph", "speed_reduction_pct"],
    "slow_movement.csv": ["slow_movement_count", "slow_movement_pct"],
    "dsh.csv": ["dsh_trip_count", "DSH"],
    "travel_time.csv": ["avg_speed_mph", "speed_estimated_travel_time_s"],
}
INPUT = [
    "link_key", "parent_start_ft", "parent_end_ft", "segment_length_ft",
    "time_bin", "waypoint_count", "journey_count", "avg_speed_mph",
    "posted_speed_limit_mph", "slow_movement_count", "slow_movement_pct",
    "dsh_trip_count", "DSH", "speed_estimated_travel_time_s",
]


def ft_label(x: float) -> str:
    """Match the reviewed corridor's human-readable section label, 0-500 etc."""
    return f"{x:.4f}".rstrip("0").rstrip(".")


def pick_fresh_directory(base: Path) -> Path:
    """Never overwrite an earlier production export (or an Excel-open CSV)."""
    for index in range(0, 1000):
        proposed = base if index == 0 else base.with_name(f"{base.name}_{index:03d}")
        staging = proposed.with_name(proposed.name + ".building")
        if not proposed.exists() and not staging.exists():
            return proposed
    raise RuntimeError(f"Cannot find an unused output folder based on {base}")


def normalized_count(series: pd.Series, field: str, batch_no: int, *, null_as_zero: bool) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    if null_as_zero:
        result = result.fillna(0)
    invalid = result.isna() | ~np.isfinite(result) | (result < 0) | (result % 1 != 0)
    if invalid.any():
        raise ValueError(f"Batch {batch_no}: invalid {field} on {int(invalid.sum()):,} rows")
    return result.astype("int64")


def export(
    source: Path,
    base_output: Path,
    *,
    batch_size: int = 100_000,
    expected_total: int | None = None,
    expected_observed: int | None = None,
    expected_waypoints: int | None = None,
    reader_factory=None,  # Enables an in-memory streaming fixture test.
) -> Path:
    source = Path(source)
    base_output = Path(base_output)
    if not source.is_file():
        raise FileNotFoundError(f"Saved OFFSET MOE Parquet not found: {source}")
    if batch_size < 1_000:
        raise ValueError("batch-size must be at least 1000")
    if reader_factory is None:
        import pyarrow.parquet as pq
        reader_factory = pq.ParquetFile
    parquet = reader_factory(source)
    missing = sorted(set(INPUT) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"Source Parquet is missing fields: {missing}")
    total_cells = parquet.metadata.num_rows
    if expected_total is not None and total_cells != expected_total:
        raise ValueError(f"Input row count {total_cells:,} differs from expected {expected_total:,}")
    print(f"Source: {source}", flush=True)
    print(f"Full network grid: {total_cells:,} segment-time cells", flush=True)
    print("Output policy: observed cells only; zero waypoint cells excluded", flush=True)
    print("unique_veh_count = saved journey_count (distinct journeys per cell)", flush=True)

    base_output.parent.mkdir(parents=True, exist_ok=True)
    output = pick_fresh_directory(base_output)
    stage = output.with_name(output.name + ".building")
    stage.mkdir(exist_ok=False)
    observed_total = waypoint_total = journey_total = 0
    batches_done = 0
    scanned = 0
    next_progress = 1_000_000
    handles = {}
    try:
        try:
            for filename, measures in OUTPUTS.items():
                handle = (stage / filename).open("w", encoding="utf-8-sig", newline="")
                handles[filename] = handle
                handle.write(",".join(COMMON + measures) + "\n")

            for batches_done, record_batch in enumerate(
                parquet.iter_batches(batch_size=batch_size, columns=INPUT), start=1
            ):
                data = record_batch.to_pandas()
                scanned += len(data)
                counts = normalized_count(data["waypoint_count"], "waypoint_count", batches_done, null_as_zero=True)
                mask = counts.gt(0)
                if not mask.any():
                    continue
                data = data.loc[mask].copy()
                data["waypoint_count"] = counts.loc[mask].to_numpy()

                # Each observed cell must have a valid count of its distinct journey IDs.
                data["unique_veh_count"] = normalized_count(
                    data["journey_count"], "journey_count", batches_done, null_as_zero=False
                ).to_numpy()
                invalid_journeys = ((data["unique_veh_count"] < 1) |
                                    (data["unique_veh_count"] > data["waypoint_count"]))
                if invalid_journeys.any():
                    row = data.loc[invalid_journeys].iloc[0]
                    raise ValueError(
                        f"Batch {batches_done}: inconsistent distinct journeys at "
                        f"link_key={row['link_key']}, time_bin={row['time_bin']}: "
                        f"waypoint_count={row['waypoint_count']}, "
                        f"journey_count={row['unique_veh_count']}"
                    )

                start = pd.to_numeric(data["parent_start_ft"], errors="coerce")
                end = pd.to_numeric(data["parent_end_ft"], errors="coerce")
                length = pd.to_numeric(data["segment_length_ft"], errors="coerce")
                valid_sections = (np.isfinite(start) & np.isfinite(end) & np.isfinite(length)
                                  & start.ge(0) & end.gt(start) & length.gt(0))
                if not valid_sections.all():
                    raise ValueError(f"Batch {batches_done}: {int((~valid_sections).sum())} observed cells have invalid section boundaries/lengths")
                if data["link_key"].isna().any() or data["link_key"].astype(str).str.strip().eq("").any():
                    raise ValueError(f"Batch {batches_done}: observed cell has missing link_key")

                data["segment_section_ft"] = start.map(ft_label) + "-" + end.map(ft_label)
                data["segment_length_ft"] = length
                stamp = pd.to_datetime(data["time_bin"], errors="coerce")
                if stamp.isna().any():
                    raise ValueError(f"Batch {batches_done}: observed cell has invalid local time_bin")
                if not (stamp.dt.minute.mod(15).eq(0) & stamp.dt.second.eq(0)).all():
                    raise ValueError(f"Batch {batches_done}: observed cell has a non-15-minute local time_bin")
                data["start_time"] = stamp.dt.strftime("%Y-%m-%d %H:%M:%S")
                data["end_time"] = (stamp + pd.Timedelta(minutes=15)).dt.strftime("%Y-%m-%d %H:%M:%S")

                # No eligible slow readings or DSH pairs -> zero occurrences.
                # Keep the previously calculated slow_movement_pct and DSH unchanged.
                for col in ("slow_movement_count", "dsh_trip_count"):
                    data[col] = normalized_count(data[col], col, batches_done, null_as_zero=True)

                speed = pd.to_numeric(data["avg_speed_mph"], errors="coerce")
                posted = pd.to_numeric(data["posted_speed_limit_mph"], errors="coerce")
                valid_speed_limit = np.isfinite(speed) & np.isfinite(posted) & posted.gt(0)
                reduction = pd.Series(np.nan, index=data.index, dtype="float64")
                reduction.loc[valid_speed_limit] = np.maximum(
                    0.0,
                    100.0 * (posted.loc[valid_speed_limit] - speed.loc[valid_speed_limit])
                    / posted.loc[valid_speed_limit],
                )
                data["speed_reduction_pct"] = reduction.round(3)

                # Preserve other saved MOEs; NaN/infinity becomes blank in CSV.
                for col in ("avg_speed_mph", "posted_speed_limit_mph", "slow_movement_pct",
                            "DSH", "speed_estimated_travel_time_s"):
                    value = pd.to_numeric(data[col], errors="coerce")
                    data[col] = value.where(np.isfinite(value), np.nan)

                for filename, metrics in OUTPUTS.items():
                    data[COMMON + metrics].to_csv(
                        handles[filename], index=False, header=False, na_rep="", float_format="%.4f"
                    )

                observed_total += len(data)
                waypoint_total += int(data["waypoint_count"].sum())
                journey_total += int(data["unique_veh_count"].sum())
                if observed_total >= next_progress:
                    print(f"  Exported {observed_total:,} observed cells / scanned {scanned:,} grid rows", flush=True)
                    next_progress = ((observed_total // 1_000_000) + 1) * 1_000_000
        finally:
            for handle in handles.values():
                handle.close()

        if scanned != total_cells:
            raise RuntimeError(f"Scanned {scanned:,} grid rows but Parquet metadata reports {total_cells:,}")
        if expected_observed is not None and observed_total != expected_observed:
            raise ValueError(f"Observed cell count {observed_total:,} != expected {expected_observed:,}")
        if expected_waypoints is not None and waypoint_total != expected_waypoints:
            raise ValueError(f"Waypoint sum {waypoint_total:,} != expected {expected_waypoints:,}")

        manifest = {
            "source_parquet": str(source),
            "export_generated": datetime.now().astimezone().isoformat(timespec="seconds"),
            "full_grid_cells_scanned": scanned,
            "observed_cells_per_csv": observed_total,
            "sum_waypoint_count": waypoint_total,
            "sum_unique_veh_count_over_cells": journey_total,
            "time_bin_minutes": 15,
            "section_size_ft": 500,
            "unique_veh_count_definition": "COUNT(DISTINCT journey_id) per link_key x section x 15-minute local-time bin. Re-entry on a new journey counts again.",
            "output_columns": {key: COMMON + value for key, value in OUTPUTS.items()},
            "decisions": {
                "observed_cells_only": True,
                "speed_reduction_floor_zero": True,
                "slow_movement_definition": "provisional, unchanged from saved Parquet",
                "DSH": "provisional, unchanged from saved Parquet",
                "travel_time": "saved speed_estimated_travel_time_s; no interpolation or traversal count",
                "GPS_baseline_modified": False,
                "offset_parquet_modified": False,
            },
        }
        (stage / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.rename(stage, output)  # Output becomes visible only after all four CSVs have been written.
        print("\nEXPORT SUCCESSFUL", flush=True)
        for filename in OUTPUTS:
            print(f"  {filename}: {observed_total:,} rows", flush=True)
        print(f"  Total saved waypoints: {waypoint_total:,}", flush=True)
        print(f"  Summed distinct journeys across cells: {journey_total:,}", flush=True)
        print(f"  Output folder: {output}", flush=True)
        return output
    except BaseException:
        # Only delete the staging directory created by this invocation.
        if stage.exists():
            try:
                shutil.rmtree(stage)
            except OSError:
                print(f"Incomplete export directory could not be deleted: {stage}", flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--expected-observed", type=int)
    parser.add_argument("--expected-waypoints", type=int)
    args = parser.parse_args()
    export(args.source, args.output, batch_size=args.batch_size,
           expected_total=args.expected_total,
           expected_observed=args.expected_observed,
           expected_waypoints=args.expected_waypoints)


if __name__ == "__main__":
    main()
