"""Production CSV stage for the existing offset MOE workflow.

The original run_offset_moe.py already invokes:
    python 10_export_moe_csvs.py --method offset

This drop-in replacement exports the four agreed compact files, using the
EXACT SAME implementation as the checked standalone exporter.  It does not
alter the saved MOE Parquet, the GPS baseline, the comparison script, or the
calculation stage. Each run writes to a new folder if an earlier export exists.

unique_veh_count = saved journey_count = distinct journey_id values per cell;
a re-entry recorded as another journey counts as another vehicle passage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import paths
from export_four_production_moe_csvs import export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Preserve the CLI used by the unmodified run_offset_moe.py runner.
    parser.add_argument("--method", choices=["offset"], default="offset")
    parser.add_argument("--source", type=Path,
                        default=paths.MOE_FILE,
                        help="Saved offset MOE Parquet (default from paths.py)")
    parser.add_argument("--output", type=Path,
                        default=paths.OUTPUT_DIR / "Compact_CSV_500ft_15min_offset",
                        help="Base folder for the four compact CSVs")
    parser.add_argument("--batch-size", type=int, default=100_000)
    # Optional for validation of a known finished dataset. DO NOT hardcode
    # these numbers: future inputs may yield different row and waypoint totals.
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--expected-observed", type=int)
    parser.add_argument("--expected-waypoints", type=int)
    args = parser.parse_args()

    print("\nSTAGE 10: FOUR COMPACT OFFSET MOE CSVs", flush=True)
    print("  The original five-wide-CSV exporter is no longer used.", flush=True)
    export(
        source=args.source,
        base_output=args.output,
        batch_size=args.batch_size,
        expected_total=args.expected_total,
        expected_observed=args.expected_observed,
        expected_waypoints=args.expected_waypoints,
    )


if __name__ == "__main__":
    main()
