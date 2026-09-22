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

CORRIDOR_FILE = RESULTS_DIR / "corridor_links.csv"
OUTPUT_FILE = RESULTS_DIR / "corridor_moe_5min.csv"

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


# Load corridor metadata
def load_corridor():

    if not CORRIDOR_FILE.exists():
        raise FileNotFoundError(
            f"Missing corridor file:\n{CORRIDOR_FILE}"
        )

    corridor = pd.read_csv(
        CORRIDOR_FILE,
        dtype={
            "link_key": "string",
        },
    )

    return corridor, set(corridor["link_key"])


# Build trip segments and continuous link runs
def prepare_link_sequence(df):

    data = df[
        df["journey_id"].notna()
    ].copy()

    if data.empty:
        return data

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

    link_compare = (
        data["link_key"]
        .fillna("__UNMATCHED__")
    )

    link_change = (
        link_compare
        .ne(link_compare.shift())
        .fillna(True)
    )

    new_run = (
        new_trip
        | link_change
    )

    data["run_id"] = np.cumsum(
        new_run.to_numpy(
            dtype=bool,
            na_value=True,
        )
    )

    data["time_bin"] = (
        data["local_time"]
        .dt.floor(
            f"{TIME_BIN_MINUTES}min"
        )
    )

    return data


# Calculate waypoint speed, slow movement, and DSH measures
def calculate_speed_moes(
    data,
    corridor_links,
):

    selected = data[
        data["link_key"].isin(corridor_links)
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

    selected["speed_change"] = (
        selected.groupby(
            ["run_id", "time_bin"],
            sort=False,
        )["speed_mph"]
        .diff()
        .abs()
    )

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

    run_dsh = run_dsh[
        run_dsh["waypoint_count"] >= 2
    ].copy()

    if run_dsh.empty:
        return basic, None

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

    trip_dsh["trip_dsh"] = (
        trip_dsh["speed_change_sum"]
        / trip_dsh["waypoint_count"]
    )

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


# Calculate complete link traversal travel times
def calculate_travel_times(
    data,
    corridor_links,
):

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

    complete = runs[
        runs["link_key"].isin(corridor_links)
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

    if complete.empty:
        return None

    complete["time_bin"] = (
        pd.to_datetime(
            complete["entry_local_time"]
        )
        .dt.floor(
            f"{TIME_BIN_MINUTES}min"
        )
    )

    return (
        complete.groupby(
            ["link_key", "time_bin"],
            as_index=False,
        )
        .agg(
            travel_time_sum=("travel_time_s", "sum"),
            complete_traversal_count=("travel_time_s", "size"),
        )
    )


# Combine chunk-level results and calculate final MOEs
def combine_results(
    basic_parts,
    dsh_parts,
    travel_parts,
    corridor,
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

    basic["avg_speed_mph"] = (
        basic["total_speed_sum"]
        / basic["waypoint_count"]
    )

    basic["slow_movement_pct"] = np.where(
        basic["total_speed_sum"] > 0,
        (
            100
            * basic["slow_speed_sum"]
            / basic["total_speed_sum"]
        ),
        np.nan,
    )

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

    metadata = corridor[
        [
            "corridor_order",
            "link_key",
            "link_length_ft",
            "distance_start_ft",
            "distance_end_ft",
            "posted_speed_limit_mph",
        ]
    ].copy()

    result = basic.merge(
        metadata,
        on="link_key",
        how="left",
    )

    # Speed reduction relative to actual posted speed
    result["speed_reduction_pct"] = np.where(
        result["posted_speed_limit_mph"] > 0,
        (
            (
                result["posted_speed_limit_mph"]
                - result["avg_speed_mph"]
            )
            / result["posted_speed_limit_mph"]
            * 100
        ),
        np.nan,
    )

    result = result.drop(
        columns=[
            "total_speed_sum",
            "slow_speed_sum",
        ]
    )

    for column in [
        "dsh_trip_count",
        "DSH",
        "complete_traversal_count",
        "avg_travel_time_s",
    ]:

        if column not in result.columns:
            result[column] = np.nan

    columns = [
        "corridor_order",
        "link_key",
        "time_bin",
        "distance_start_ft",
        "distance_end_ft",
        "link_length_ft",
        "posted_speed_limit_mph",
        "avg_speed_mph",
        "speed_reduction_pct",
        "waypoint_count",
        "journey_count",
        "slow_movement_count",
        "slow_movement_pct",
        "dsh_trip_count",
        "DSH",
        "complete_traversal_count",
        "avg_travel_time_s",
    ]

    return (
        result[columns]
        .sort_values(
            ["corridor_order", "time_bin"]
        )
        .reset_index(drop=True)
    )


def main():

    start_time = time.perf_counter()

    # Step 1: Load corridor metadata
    corridor, corridor_links = load_corridor()

    print("CORRIDOR 5-MINUTE MOE PROCESSING")
    print("=" * 70)
    print(
        f"Corridor links: {len(corridor):,}"
    )

    basic_parts = []
    dsh_parts = []
    travel_parts = []

    carry = pd.DataFrame()
    total_rows = 0

    # Step 2: Process the complete CV hour
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

        chunk = chunk[
            chunk["journey_id"].notna()
        ].copy()

        if chunk.empty:
            continue

        if not carry.empty:

            chunk = pd.concat(
                [carry, chunk],
                ignore_index=True,
            )

        last_journey = (
            chunk["journey_id"].iloc[-1]
        )

        carry = chunk[
            chunk["journey_id"]
            == last_journey
        ].copy()

        complete = chunk[
            chunk["journey_id"]
            != last_journey
        ].copy()

        if complete.empty:
            continue

        sequence = prepare_link_sequence(
            complete
        )

        basic, dsh = calculate_speed_moes(
            sequence,
            corridor_links,
        )

        if basic is not None:
            basic_parts.append(basic)

        if dsh is not None:
            dsh_parts.append(dsh)

        travel = calculate_travel_times(
            sequence,
            corridor_links,
        )

        if travel is not None:
            travel_parts.append(travel)

        print(
            f"Chunk {chunk_number:>3} | "
            f"Rows processed: {total_rows:,}"
        )

    # Step 3: Process the final carried journey
    if not carry.empty:

        sequence = prepare_link_sequence(
            carry
        )

        basic, dsh = calculate_speed_moes(
            sequence,
            corridor_links,
        )

        if basic is not None:
            basic_parts.append(basic)

        if dsh is not None:
            dsh_parts.append(dsh)

        travel = calculate_travel_times(
            sequence,
            corridor_links,
        )

        if travel is not None:
            travel_parts.append(travel)

    # Step 4: Combine and calculate Speed Reduction Percentage
    result = combine_results(
        basic_parts,
        dsh_parts,
        travel_parts,
        corridor,
    )

    for column in [
        "link_length_ft",
        "posted_speed_limit_mph",
        "avg_speed_mph",
        "speed_reduction_pct",
        "slow_movement_pct",
        "DSH",
        "avg_travel_time_s",
    ]:

        result[column] = (
            result[column]
            .round(3)
        )

    # Step 5: Save final 5-minute corridor MOEs
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
    print("MOE PROCESSING COMPLETE")
    print(
        f"Rows:     {len(result):,}"
    )
    print(
        f"Runtime:  {runtime / 60:.2f} min"
    )
    print(
        f"Output:   {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()