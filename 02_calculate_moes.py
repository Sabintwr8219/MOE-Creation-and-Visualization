from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"D:\Sabin\Streetlight-Data-Processing")
MOE_DIR = PROJECT_ROOT / "MOE for selected links"
RESULTS_DIR = MOE_DIR / "Results"

INPUT_FILE = PROJECT_ROOT / "Output" / "Final" / "9_20_2025" / "9_20_2025_hr=19.csv"
SELECTED_LINKS_FILE = RESULTS_DIR / "selected_links.csv"
MOE_OUTPUT_FILE = RESULTS_DIR / "link_moe_5min.csv"

CHUNK_SIZE = 1_000_000
TIME_BIN_MINUTES = 5
TRIP_GAP_SECONDS = 120
SLOW_SPEED_THRESHOLD_MPH = 5.0

USE_COLUMNS = ["journey_id", "capture_time", "local_time", "speed_mph", "link_key"]


def load_selected_links():
    df = pd.read_csv(SELECTED_LINKS_FILE, dtype={"link_key": "string"})
    return set(df["link_key"].dropna().astype(str))


def prepare_batch(batch):
    batch = batch.copy()
    batch["link_key"] = batch["link_key"].astype("string")
    batch["speed_mph"] = pd.to_numeric(batch["speed_mph"], errors="coerce")
    batch["event_time"] = pd.to_datetime(batch["local_time"], errors="coerce")
    batch = batch.sort_values(["journey_id", "event_time"], kind="mergesort").reset_index(drop=True)

    same_journey = batch["journey_id"].eq(batch["journey_id"].shift())
    gap_s = (batch["event_time"] - batch["event_time"].shift()).dt.total_seconds()
    matched = batch["link_key"].notna()
    previous_matched = matched.shift(fill_value=False)

    new_trip = (
        ~same_journey
        | gap_s.gt(TRIP_GAP_SECONDS).fillna(False)
        | ~matched
        | ~previous_matched
    )
    batch["trip_segment"] = np.cumsum(new_trip.to_numpy(dtype=bool, na_value=True))

    new_run = new_trip | batch["link_key"].ne(batch["link_key"].shift()).fillna(True)
    batch["run_id"] = np.cumsum(new_run.to_numpy(dtype=bool, na_value=True))
    batch["time_bin"] = batch["event_time"].dt.floor(f"{TIME_BIN_MINUTES}min")
    return batch


def waypoint_metrics(batch, selected_links):
    valid = batch[
        batch["link_key"].isin(selected_links)
        & batch["event_time"].notna()
        & batch["speed_mph"].notna()
        & batch["speed_mph"].ge(0)
    ].copy()
    if valid.empty:
        return pd.DataFrame()

    valid["slow"] = valid["speed_mph"] < SLOW_SPEED_THRESHOLD_MPH
    valid["slow_speed"] = valid["speed_mph"].where(valid["slow"], 0.0)

    return (
        valid.groupby(["link_key", "time_bin"], observed=True)
        .agg(
            waypoint_count=("speed_mph", "size"),
            journey_count=("journey_id", "nunique"),
            total_speed=("speed_mph", "sum"),
            slow_movement_count=("slow", "sum"),
            slow_speed_sum=("slow_speed", "sum"),
        )
        .reset_index()
    )


def dsh_metrics(batch, selected_links):
    valid = batch[
        batch["link_key"].isin(selected_links)
        & batch["event_time"].notna()
        & batch["speed_mph"].notna()
        & batch["speed_mph"].ge(0)
    ].copy()
    if valid.empty:
        return pd.DataFrame()

    groups = ["link_key", "time_bin", "journey_id", "trip_segment", "run_id"]
    valid["speed_change"] = valid.groupby(groups, observed=True)["speed_mph"].diff().abs()

    trip = (
        valid.groupby(groups, observed=True)
        .agg(n_speeds=("speed_mph", "size"), speed_change_sum=("speed_change", "sum"))
        .reset_index()
    )
    # Paper Eq. 5: denominator is n waypoint speeds, not n - 1.
    trip["trip_DSH"] = trip["speed_change_sum"] / trip["n_speeds"]

    return (
        trip.groupby(["link_key", "time_bin"], observed=True)
        .agg(dsh_trip_count=("trip_DSH", "size"), dsh_sum=("trip_DSH", "sum"))
        .reset_index()
    )


def travel_time_metrics(batch, selected_links):
    matched = batch[batch["link_key"].notna() & batch["event_time"].notna()].copy()
    if matched.empty:
        return pd.DataFrame()

    runs = (
        matched.groupby(["journey_id", "trip_segment", "run_id", "link_key"], observed=True)
        .agg(entry_time=("event_time", "min"))
        .reset_index()
        .sort_values(["journey_id", "trip_segment", "entry_time", "run_id"], kind="mergesort")
    )

    group = ["journey_id", "trip_segment"]
    runs["previous_link"] = runs.groupby(group, observed=True)["link_key"].shift(1)
    runs["next_link"] = runs.groupby(group, observed=True)["link_key"].shift(-1)
    runs["next_entry_time"] = runs.groupby(group, observed=True)["entry_time"].shift(-1)

    complete = runs[
        runs["link_key"].isin(selected_links)
        & runs["previous_link"].notna()
        & runs["next_link"].notna()
        & runs["next_entry_time"].notna()
    ].copy()
    if complete.empty:
        return pd.DataFrame()

    complete["travel_time_s"] = (complete["next_entry_time"] - complete["entry_time"]).dt.total_seconds()
    complete = complete[complete["travel_time_s"].gt(0)].copy()
    if complete.empty:
        return pd.DataFrame()

    complete["time_bin"] = complete["entry_time"].dt.floor(f"{TIME_BIN_MINUTES}min")
    return (
        complete.groupby(["link_key", "time_bin"], observed=True)
        .agg(
            complete_traversal_count=("travel_time_s", "size"),
            travel_time_sum_s=("travel_time_s", "sum"),
        )
        .reset_index()
    )


def combine(frames, values):
    frames = [x for x in frames if not x.empty]
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .groupby(["link_key", "time_bin"], observed=True)[values]
        .sum()
        .reset_index()
    )


def process_batch(batch, selected_links):
    batch = prepare_batch(batch)
    return (
        waypoint_metrics(batch, selected_links),
        dsh_metrics(batch, selected_links),
        travel_time_metrics(batch, selected_links),
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selected_links = load_selected_links()

    waypoint_parts, dsh_parts, travel_parts = [], [], []
    carry = pd.DataFrame()

    reader = pd.read_csv(
        INPUT_FILE,
        usecols=USE_COLUMNS,
        chunksize=CHUNK_SIZE,
        dtype={"journey_id": "string", "link_key": "string"},
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = pd.DataFrame()
        if chunk.empty:
            continue

        last_journey = chunk["journey_id"].iloc[-1]
        carry = chunk[chunk["journey_id"] == last_journey].copy()
        complete = chunk[chunk["journey_id"] != last_journey].copy()

        if not complete.empty:
            waypoint, dsh, travel = process_batch(complete, selected_links)
            waypoint_parts.append(waypoint)
            dsh_parts.append(dsh)
            travel_parts.append(travel)

        print(f"Chunk {chunk_number:,}: rows={len(chunk):,}, carry={len(carry):,}")

    if not carry.empty:
        waypoint, dsh, travel = process_batch(carry, selected_links)
        waypoint_parts.append(waypoint)
        dsh_parts.append(dsh)
        travel_parts.append(travel)

    waypoint = combine(
        waypoint_parts,
        ["waypoint_count", "journey_count", "total_speed", "slow_movement_count", "slow_speed_sum"],
    )
    dsh = combine(dsh_parts, ["dsh_trip_count", "dsh_sum"])
    travel = combine(travel_parts, ["complete_traversal_count", "travel_time_sum_s"])

    result = waypoint.merge(dsh, on=["link_key", "time_bin"], how="left").merge(
        travel, on=["link_key", "time_bin"], how="left"
    )

    for column in ["dsh_trip_count", "dsh_sum", "complete_traversal_count", "travel_time_sum_s"]:
        if column not in result.columns:
            result[column] = 0.0

    result["slow_movement_pct"] = np.where(
        result["total_speed"].gt(0), result["slow_speed_sum"] / result["total_speed"] * 100.0, np.nan
    )
    result["DSH"] = np.where(
        result["dsh_trip_count"].gt(0), result["dsh_sum"] / result["dsh_trip_count"], np.nan
    )
    result["avg_travel_time_s"] = np.where(
        result["complete_traversal_count"].gt(0),
        result["travel_time_sum_s"] / result["complete_traversal_count"],
        np.nan,
    )

    result = result[
        [
            "link_key",
            "time_bin",
            "waypoint_count",
            "journey_count",
            "slow_movement_count",
            "slow_movement_pct",
            "dsh_trip_count",
            "DSH",
            "complete_traversal_count",
            "avg_travel_time_s",
        ]
    ].sort_values(["link_key", "time_bin"])

    result.to_csv(MOE_OUTPUT_FILE, index=False)
    print(f"Rows written: {len(result):,}")
    print(f"Saved: {MOE_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
