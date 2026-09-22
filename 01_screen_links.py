import time
from pathlib import Path

import numpy as np
import pandas as pd


# Paths and settings
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "Output"
    / "Final"
    / "9_20_2025"
    / "9_20_2025_hr=19.csv"
)

RESULTS_DIR = Path(__file__).resolve().parent / "Results"
OUTPUT_FILE = RESULTS_DIR / "link_screening_summary.csv"

CHUNK_SIZE = 1_000_000

USE_COLUMNS = [
    "journey_id",
    "link_key",
    "speed_mph",
]


# Check input file and create results folder
def validate_paths():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Calculate waypoint count and speed sum for each link in one chunk
def calculate_link_stats(df):
    return (
        df.groupby("link_key", sort=False)["speed_mph"]
        .agg(
            waypoint_count="size",
            speed_sum="sum",
        )
        .reset_index()
    )


# Count unique journeys while carrying the last journey across chunk boundaries
def calculate_journey_counts(df, carry_journey=None, carry_links=None):

    if carry_links is None:
        carry_links = set()

    pairs = (
        df[["journey_id", "link_key"]]
        .dropna()
        .drop_duplicates()
    )

    if carry_journey is not None and carry_links:
        carry_df = pd.DataFrame({
            "journey_id": [carry_journey] * len(carry_links),
            "link_key": list(carry_links),
        })

        pairs = pd.concat(
            [carry_df, pairs],
            ignore_index=True,
        ).drop_duplicates()

    if pairs.empty:
        return None, carry_journey, carry_links

    last_journey = pairs["journey_id"].iloc[-1]

    completed = pairs[
        pairs["journey_id"] != last_journey
    ]

    journey_counts = (
        completed.groupby("link_key")
        .size()
        .rename("journey_count")
        .reset_index()
    )

    last_links = set(
        pairs.loc[
            pairs["journey_id"] == last_journey,
            "link_key",
        ]
    )

    return journey_counts, last_journey, last_links


# Combine chunk results into the final link screening table
def combine_results(link_parts, journey_parts, carry_links):

    link_summary = (
        pd.concat(link_parts, ignore_index=True)
        .groupby("link_key", as_index=False)
        .agg(
            waypoint_count=("waypoint_count", "sum"),
            speed_sum=("speed_sum", "sum"),
        )
    )

    if carry_links:
        journey_parts.append(
            pd.DataFrame({
                "link_key": list(carry_links),
                "journey_count": 1,
            })
        )

    journey_summary = (
        pd.concat(journey_parts, ignore_index=True)
        .groupby("link_key", as_index=False)["journey_count"]
        .sum()
    )

    summary = link_summary.merge(
        journey_summary,
        on="link_key",
        how="left",
    )

    summary["journey_count"] = (
        summary["journey_count"]
        .fillna(0)
        .astype(int)
    )

    summary["mean_speed_mph"] = (
        summary["speed_sum"]
        / summary["waypoint_count"]
    )

    summary = summary.drop(columns="speed_sum")

    summary["mean_speed_mph"] = (
        summary["mean_speed_mph"].round(3)
    )

    return summary.sort_values(
        ["waypoint_count", "journey_count"],
        ascending=False,
    ).reset_index(drop=True)


def main():

    # Step 1: Check paths and prepare output folder
    validate_paths()

    print("LINK SCREENING")
    print("=" * 70)
    print("Input:", INPUT_FILE)

    start_time = time.perf_counter()

    link_parts = []
    journey_parts = []

    carry_journey = None
    carry_links = set()

    total_rows = 0
    valid_rows = 0

    # Step 2: Read the large hourly CSV in chunks
    reader = pd.read_csv(
        INPUT_FILE,
        usecols=USE_COLUMNS,
        chunksize=CHUNK_SIZE,
        dtype={
            "journey_id": "string",
            "link_key": "string",
            "speed_mph": "float64",
        },
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        total_rows += len(chunk)

        # Step 3: Keep matched links with valid non-negative speeds
        valid = chunk[
            chunk["link_key"].notna()
            & chunk["speed_mph"].notna()
            & np.isfinite(chunk["speed_mph"])
            & (chunk["speed_mph"] >= 0)
        ].copy()

        valid_rows += len(valid)

        # Step 4: Calculate waypoint and speed statistics
        if not valid.empty:
            link_parts.append(
                calculate_link_stats(valid)
            )

        # Step 5: Calculate unique journey counts
        journey_counts, carry_journey, carry_links = (
            calculate_journey_counts(
                valid,
                carry_journey,
                carry_links,
            )
        )

        if journey_counts is not None and not journey_counts.empty:
            journey_parts.append(journey_counts)

        print(
            f"Chunk {chunk_number:>3} | "
            f"Rows: {total_rows:,} | "
            f"Valid matched: {valid_rows:,}"
        )

    # Step 6: Combine all chunk-level results
    summary = combine_results(
        link_parts,
        journey_parts,
        carry_links,
    )

    # Step 7: Save final screening table
    summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    runtime = time.perf_counter() - start_time

    print()
    print("=" * 70)
    print("SCREENING COMPLETE")
    print(f"Total rows:       {total_rows:,}")
    print(f"Valid rows:       {valid_rows:,}")
    print(f"Total links:      {len(summary):,}")
    print(f"Runtime:          {runtime / 60:.2f} min")
    print(f"Output:           {OUTPUT_FILE}")

    print()
    print("TOP 20 LINKS")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()