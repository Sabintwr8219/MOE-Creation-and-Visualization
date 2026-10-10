"""OFFICIAL production runner: offset assignment -> MOE Parquet -> four compact CSVs.

Historical GPS comparisons are stored in Utility Scripts/ and NOT run by default.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import paths

HERE = Path(__file__).resolve().parent


def run_script(path: Path, *arguments: str) -> None:
    subprocess.run([sys.executable, "-u", str(path), *arguments], check=True, cwd=HERE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=paths.PARALLEL_WORKERS)
    parser.add_argument("--rebuild-partitions", action="store_true")
    parser.add_argument("--export-only", action="store_true",
                        help="Re-export the existing Parquet; NO MOE recomputation")
    parser.add_argument("--check-inputs", action="store_true",
                        help="Check input locations without doing calculation or export")
    parser.add_argument("--compare-only", action="store_true",
                        help="Optional HISTORICAL GPS/offset comparison only (not production)")
    args = parser.parse_args()

    if args.compare_only:
        run_script(HERE / "Utility Scripts" / "11_compare_moe_methods.py")
        return
    if args.check_inputs:
        print("Official OFFSET pipeline paths:")
        for label, path in paths.source_summary().items():
            print(f"  {label}: {path}")
        paths.check_inputs()
        print("Input check successful; no MOE computation or CSV export ran.")
        return
    if not args.export_only:
        paths.check_inputs()
        command = ["--workers", str(args.workers)]
        if args.rebuild_partitions:
            command.append("--rebuild-partitions")
        print("STAGE 1: OFFSETS -> 500-FT x 15-MIN MOE PARQUET", flush=True)
        run_script(paths.CODE_DIR / "03_calculate_segment_moes.py", *command)
    else:
        print("Skipping calculation; using saved offset MOE Parquet.", flush=True)
    print("STAGE 2: FOUR COMPACT PRODUCTION CSVs", flush=True)
    run_script(paths.CODE_DIR / "10_export_moe_csvs.py", "--method", "offset")


if __name__ == "__main__":
    main()
