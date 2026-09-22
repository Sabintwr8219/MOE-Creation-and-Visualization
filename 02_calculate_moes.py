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

SELECTED_LINKS_FILE = RESULTS_DIR / "selected_links.csv"
OUTPUT_FILE = RESULTS_DIR / "link_moe_5min.csv"

TIME_BIN_MINUTES = 5
SLOW_SPEED_THRESHOLD_MPH = 5.0
MAX_WAYPOINT_GAP_SECONDS = 120
CHUNK_SIZE = 1_000_000

USE_COLUMNS = [
    "journey_id",
    "capture_time",
    "local_time",
    "link_key",
    "speed_mph",
]


# Load links selected only from CV data availability
def load_selected_links():

    selected = pd.read_csv(
        SELECTED_LINKS_FILE,
        dtype={"link_key": "string"},
    )

    return selected, set(selected["link_key"])


# Split journeys at gaps greater than 2 minutes and identify link runs
def prepare_link_sequence(df):

    data = df[
        df["journey_id"].notna()
    ].copy()

    data = data.sort_values(
        ["journey_id", "capture_time"]
    ).reset_index(drop=True)

    data["local_time"] = pd.to_datetime(
        data["local_time"]
    )

    new_journey = (
        data["journey_id"]
        .ne(data["journey_id"].shift())
    )

    time_gap = (
        data.groupby(
            "journey_id",
            sort=False,
        )["capture_time"]
        .diff()
    )

    new_trip = (
        new_journey
        | (time_gap > MAX_WAYPOINT_GAP_SECONDS)
    )

    data["trip_segment_id"] = np.cumsum(
        new_trip.to_numpy(
            dtype=bool,
            na_value=True,
        )
    )

    link_for_compare = (
        data["link_key"]
        .fillna("__UNMATCHED__")
    )

    new_run = (
        new_trip
        | link_for_compare.ne(
            link_for_compare.shift()
        )
    )

    data["run_id"] = np.cumsum(
        new_run.to_numpy(
            dtype=bool,
            na_value=True,
        )
    )

    data["time_bin"] = (
        data["local_time"]
        .dt.floor(f"{TIME_BIN_MINUTES}min")
    )

    return data


# Calculate slow movements and DSH for each time bin
def calculate_speed_moes(data, selected_links):

    selected = data[
        data["link_key"].isin(selected_links)
        & data["speed_mph"].notna()
        & np.isfinite(data["speed_mph"])
        & (data["speed_mph"] >= 0)
    ].copy()

    if selected.empty:
        return None, None

    selected = selected.sort_values(
        ["journey_id", "capture_time"]
    )

    selected["is_slow"] = (
        selected["speed_mph"]
        < SLOW_SPEED_THRESHOLD_MPH
    )

    selected["slow_speed"] = np.where(
        selected["is_slow"],
        selected["speed_mph"],
        0.0,
    )

    # Basic statistics for each link and time bin
    basic = (
        selected.groupby(
            ["link_key", "time_bin"],
            as_index=False,
        )
        .agg(
            waypoint_count=("speed_mph", "size"),
            journey_count=("journey_id", "nunique"),
            total_speed_sum=("speed_mph", "sum"),
            slow_movement_count=("is_slow", "sum"),
            slow_speed_sum=("slow_speed", "sum"),
        )
    )

    # Calculate consecutive speed changes independently within each time bin
    selected["speed_change"] = (
        selected.groupby(
            ["run_id", "time_bin"],
            sort=False,
        )["speed_mph"]
        .diff()
        .abs()
    )

    # Summarize continuous link visits within each time bin
    run_dsh = (
        selected.groupby(
            [
                "trip_segment_id",
                "journey_id",
                "link_key",
                "run_id",
                "time_bin",
            ],
            as_index=False,
        )
        .agg(
            waypoint_count=("speed_mph", "size"),
            speed_change_sum=("speed_change", "sum"),
        )
    )

    # DSH requires at least two waypoint speeds
    run_dsh = run_dsh[
        run_dsh["waypoint_count"] >= 2
    ].copy()

    # Combine repeated visits within the same trip and time bin
    trip_dsh = (
        run_dsh.groupby(
            [
                "trip_segment_id",
                "journey_id",
                "link_key",
                "time_bin",
            ],
            as_index=False,
        )
        .agg(
            waypoint_count=("waypoint_count", "sum"),
            speed_change_sum=("speed_change_sum", "sum"),
        )
    )

    # Paper definition uses n waypoint speeds as the denominator
    trip_dsh["trip_dsh"] = (
        trip_dsh["speed_change_sum"]
        / trip_dsh["waypoint_count"]
    )

    # Average trip-level DSH for each link and time bin
    dsh = (
        trip_dsh.groupby(
            ["link_key", "time_bin"],
            as_index=False,
        )
        .agg(
            dsh_sum=("trip_dsh", "sum"),
            dsh_trip_count=("trip_dsh", "size"),
        )
    )

    return basic, dsh


# Calculate complete link travel times and assign them to entry time bins
def calculate_travel_times(data, selected_links):

    if data.empty:
        return None

    runs = (
        data.groupby(
            "run_id",
            as_index=False,
            sort=False,
        )
        .agg(
            trip_segment_id=("trip_segment_id", "first"),
            journey_id=("journey_id", "first"),
            link_key=("link_key", "first"),
            entry_time=("capture_time", "first"),
            entry_local_time=("local_time", "first"),
        )
    )

    runs = runs.sort_values(
        "run_id"
    ).reset_index(drop=True)

    runs["previous_trip"] = (
        runs["trip_segment_id"].shift()
    )

    runs["next_trip"] = (
        runs["trip_segment_id"].shift(-1)
    )

    runs["previous_link"] = (
        runs["link_key"].shift()
    )

    runs["next_link"] = (
        runs["link_key"].shift(-1)
    )

    runs["next_entry_time"] = (
        runs["entry_time"].shift(-1)
    )

    # Keep only fully observed traversals within the same trip
    complete = runs[
        runs["link_key"].isin(selected_links)
        & (
            runs["trip_segment_id"]
            == runs["previous_trip"]
        )
        & (
            runs["trip_segment_id"]
            == runs["next_trip"]
        )
        & runs["previous_link"].notna()
        & runs["next_link"].notna()
    ].copy()

    if complete.empty:
        return None

    complete["travel_time_s"] = (
        complete["next_entry_time"]
        - complete["entry_time"]
    )

    complete = complete[
        complete["travel_time_s"] > 0
    ].copy()

    # Travel time belongs to the time period when the vehicle entered the link
    complete["time_bin"] = (
        pd.to_datetime(
            complete["entry_local_time"]
        )
        .dt.floor(f"{TIME_BIN_MINUTES}min")
    )

    travel = (
        complete.groupby(
            ["link_key", "time_bin"],
            as_index=False,
        )
        .agg(
            travel_time_sum=("travel_time_s", "sum"),
            complete_traversal_count=("travel_time_s", "size"),
        )
    )

    return travel


# Combine chunk results into final 5-minute MOEs
def combine_results(
    basic_parts,
    dsh_parts,
    travel_parts,
):

    basic = (
        pd.concat(
            basic_parts,
            ignore_index=True,
        )
        .groupby(
            ["link_key", "time_bin"],
            as_index=False,
        )
        .agg(
            waypoint_count=("waypoint_count", "sum"),
            journey_count=("journey_count", "sum"),
            total_speed_sum=("total_speed_sum", "sum"),
            slow_movement_count=("slow_movement_count", "sum"),
            slow_speed_sum=("slow_speed_sum", "sum"),
        )
    )

    # Slow movement percentage from Eq. 6
    basic["slow_movement_pct"] = np.where(
        basic["total_speed_sum"] > 0,
        (
            100
            * basic["slow_speed_sum"]
            / basic["total_speed_sum"]
        ),
        np.nan,
    )

    # Combine DSH
    if dsh_parts:

        dsh = (
            pd.concat(
                dsh_parts,
                ignore_index=True,
            )
            .groupby(
                ["link_key", "time_bin"],
                as_index=False,
            )
            .agg(
                dsh_sum=("dsh_sum", "sum"),
                dsh_trip_count=("dsh_trip_count", "sum"),
            )
        )

        dsh["DSH"] = (
            dsh["dsh_sum"]
            / dsh["dsh_trip_count"]
        )

        dsh = dsh.drop(
            columns=["dsh_sum"]
        )

        basic = basic.merge(
            dsh,
            on=["link_key", "time_bin"],
            how="left",
        )

    else:

        basic["dsh_trip_count"] = 0
        basic["DSH"] = np.nan

    # Combine travel time
    if travel_parts:

        travel = (
            pd.concat(
                travel_parts,
                ignore_index=True,
            )
            .groupby(
                ["link_key", "time_bin"],
                as_index=False,
            )
            .agg(
                travel_time_sum=("travel_time_sum", "sum"),
                complete_traversal_count=(
                    "complete_traversal_count",
                    "sum",
                ),
            )
        )

        travel["avg_travel_time_s"] = (
            travel["travel_time_sum"]
            / travel["complete_traversal_count"]
        )

        travel = travel.drop(
            columns=["travel_time_sum"]
        )

        basic = basic.merge(
            travel,
            on=["link_key", "time_bin"],
            how="left",
        )

    else:

        basic["complete_traversal_count"] = 0
        basic["avg_travel_time_s"] = np.nan

    basic = basic.drop(
        columns=[
            "total_speed_sum",
            "slow_speed_sum",
        ]
    )

    return basic.sort_values(
        ["link_key", "time_bin"]
    ).reset_index(drop=True)


def main():

    start_time = time.perf_counter()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Step 1: Load coverage-selected links
    selected, selected_links = load_selected_links()

    print("5-MINUTE LINK MOE PROCESSING")
    print("=" * 70)
    print(f"Selected links:     {len(selected):,}")
    print(f"Time bin:           {TIME_BIN_MINUTES} minutes")
    print(f"Slow threshold:     < {SLOW_SPEED_THRESHOLD_MPH} mph")
    print(f"Maximum trip gap:   {MAX_WAYPOINT_GAP_SECONDS} seconds")
    print()

    basic_parts = []
    dsh_parts = []
    travel_parts = []

    carry = pd.DataFrame()
    total_rows = 0

    # Step 2: Read the complete hourly CV file in chunks
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

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):

        total_rows += len(chunk)

        # Step 3: Add unfinished journey from previous chunk
        if not carry.empty:

            chunk = pd.concat(
                [carry, chunk],
                ignore_index=True,
            )

        last_journey = (
            chunk["journey_id"].iloc[-1]
        )

        # Step 4: Carry final journey into next chunk
        carry = chunk[
            chunk["journey_id"]
            == last_journey
        ].copy()

        complete_chunk = chunk[
            chunk["journey_id"]
            != last_journey
        ].copy()

        if complete_chunk.empty:
            continue

        # Step 5: Split trips and build link sequences
        sequence = prepare_link_sequence(
            complete_chunk
        )

        # Step 6: Calculate slow movement and DSH
        basic, dsh = calculate_speed_moes(
            sequence,
            selected_links,
        )

        if basic is not None:
            basic_parts.append(basic)

        if dsh is not None and not dsh.empty:
            dsh_parts.append(dsh)

        # Step 7: Calculate complete link travel times
        travel = calculate_travel_times(
            sequence,
            selected_links,
        )

        if travel is not None and not travel.empty:
            travel_parts.append(travel)

        print(
            f"Chunk {chunk_number:>3} | "
            f"Rows processed: {total_rows:,}"
        )

    # Step 8: Process final journey
    if not carry.empty:

        sequence = prepare_link_sequence(
            carry
        )

        basic, dsh = calculate_speed_moes(
            sequence,
            selected_links,
        )

        if basic is not None:
            basic_parts.append(basic)

        if dsh is not None and not dsh.empty:
            dsh_parts.append(dsh)

        travel = calculate_travel_times(
            sequence,
            selected_links,
        )

        if travel is not None and not travel.empty:
            travel_parts.append(travel)

    # Step 9: Combine 5-minute results
    result = combine_results(
        basic_parts,
        dsh_parts,
        travel_parts,
    )

    # Step 10: Round final MOE values
    result["slow_movement_pct"] = (
        result["slow_movement_pct"].round(3)
    )

    result["DSH"] = (
        result["DSH"].round(3)
    )

    result["avg_travel_time_s"] = (
        result["avg_travel_time_s"].round(3)
    )

    # Step 11: Save the 5-minute MOE table
    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    print()
    print("=" * 70)
    print("5-MINUTE MOE PROCESSING COMPLETE")
    print(f"Selected links: {len(selected):,}")
    print(f"MOE rows:       {len(result):,}")
    print(f"Runtime:        {runtime / 60:.2f} min")
    print(f"Output:         {OUTPUT_FILE}")

    print()
    print("FIRST 20 RESULTS")
    print(
        result.head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()