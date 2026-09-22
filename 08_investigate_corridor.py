from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkt


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

NETWORK_FILE = (
    PROJECT_ROOT
    / "Initial Input Files"
    / "OSM_Short_Level_CSV"
    / "Complete_OSM_Short_Level_Link_List.csv"
)

TXDOT_FILE = (
    PROJECT_ROOT
    / "Initial Input Files"
    / "TxDOT"
    / "link_list_with_speed_NOL.csv"
)

RESULTS_DIR = Path(__file__).resolve().parent / "Results"

CORRIDOR_FILE = RESULTS_DIR / "corridor_links.csv"
MOE_FILE = RESULTS_DIR / "corridor_moe_5min.csv"

OUTPUT_FILE = RESULTS_DIR / "corridor_diagnostic.csv"
MAP_FILE = RESULTS_DIR / "corridor_map_points.csv"

NETWORK_CHUNK_SIZE = 500_000
TXDOT_CHUNK_SIZE = 500_000


# Load corridor and MOE results
def load_results():

    corridor = pd.read_csv(
        CORRIDOR_FILE,
        dtype={
            "link_key": "string",
            "Matched_GID": "string",
        },
    )

    moe = pd.read_csv(
        MOE_FILE,
        dtype={
            "link_key": "string",
        },
        parse_dates=["time_bin"],
    )

    return corridor, moe


# Extract start, midpoint, and end coordinates from WGS84 geometry
def extract_coordinates(geometry_text):

    if pd.isna(geometry_text):
        return pd.Series(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        )

    try:

        geometry = wkt.loads(
            str(geometry_text)
        )

        if geometry.is_empty:
            raise ValueError

        start = geometry.coords[0]
        end = geometry.coords[-1]

        midpoint = geometry.interpolate(
            0.5,
            normalized=True,
        )

        return pd.Series(
            [
                start[1],
                start[0],
                midpoint.y,
                midpoint.x,
                end[1],
                end[0],
            ]
        )

    except Exception:

        return pd.Series(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        )


# Load original roadway metadata for the selected corridor
# Load original roadway metadata for the selected corridor
def load_network_metadata(corridor):

    target_links = set(
        corridor["link_key"]
    )

    header = pd.read_csv(
        NETWORK_FILE,
        nrows=0,
    )

    available = set(
        header.columns
    )

    desired = [
        "link_key",
        "name",
        "osm_way_id",
        "from_node_id",
        "to_node_id",
        "geometry",
        "facility_type",
        "RDBD_TYPE",
        "Functional Class",
        "heading",
        "length",
        "Matched_GID",
        "TXDOT_DI_2",
    ]

    usecols = [
        column
        for column in desired
        if column in available
    ]

    parts = []
    total_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            NETWORK_FILE,
            usecols=usecols,
            chunksize=NETWORK_CHUNK_SIZE,
            dtype="string",
        ),
        start=1,
    ):

        total_rows += len(chunk)

        matched = chunk[
            chunk["link_key"]
            .isin(target_links)
        ].copy()

        if not matched.empty:
            parts.append(matched)

        if chunk_number % 10 == 0:

            found = (
                pd.concat(parts)["link_key"].nunique()
                if parts
                else 0
            )

            print(
                f"Network chunk {chunk_number:>3} | "
                f"Rows scanned: {total_rows:,} | "
                f"Corridor links found: {found:,}/"
                f"{len(target_links):,}"
            )

        if parts:

            found_links = set(
                pd.concat(parts)[
                    "link_key"
                ]
                .dropna()
                .tolist()
            )

            if target_links.issubset(
                found_links
            ):
                break

    if not parts:

        raise RuntimeError(
            "No corridor links were found "
            "in the short-link network."
        )

    metadata = pd.concat(
        parts,
        ignore_index=True,
    )

    metadata = metadata.drop_duplicates(
        subset="link_key"
    )

    coordinates = metadata[
        "geometry"
    ].apply(
        extract_coordinates
    )

    coordinates.columns = [
        "start_lat",
        "start_lon",
        "mid_lat",
        "mid_lon",
        "end_lat",
        "end_lon",
    ]

    metadata = pd.concat(
        [
            metadata.reset_index(
                drop=True
            ),
            coordinates.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    return metadata

# Load exact TxDOT segment information used for posted-speed matching
# Load exact TxDOT segment information used for posted-speed matching
def load_txdot_metadata(corridor):

    target_ids = set(
        pd.to_numeric(
            corridor["txdot_link_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    desired = [
        "LinkID",
        "GID",
        "RDBD_TYPE",
        "DES_DRCT",
        "COUNTY",
        "MAP_LBL",
        "SpeedLimit",
        "NumberOfLanes",
        "Ramp_type",
        "Functional Class",
    ]

    header = pd.read_csv(
        TXDOT_FILE,
        nrows=0,
    )

    usecols = [
        column
        for column in desired
        if column in header.columns
    ]

    parts = []
    total_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            TXDOT_FILE,
            usecols=usecols,
            chunksize=TXDOT_CHUNK_SIZE,
            dtype="string",
        ),
        start=1,
    ):

        total_rows += len(chunk)

        ids = pd.to_numeric(
            chunk["LinkID"],
            errors="coerce",
        )

        matched = chunk[
            ids.isin(target_ids)
        ].copy()

        if not matched.empty:
            parts.append(matched)

        if chunk_number % 5 == 0:

            found = (
                pd.concat(parts)["LinkID"].nunique()
                if parts
                else 0
            )

            print(
                f"TxDOT chunk {chunk_number:>3} | "
                f"Rows scanned: {total_rows:,} | "
                f"Segments found: {found:,}/"
                f"{len(target_ids):,}"
            )

        if parts:

            found_ids = set(
                pd.to_numeric(
                    pd.concat(parts)["LinkID"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
            )

            if target_ids.issubset(
                found_ids
            ):
                break

    if not parts:
        return pd.DataFrame()

    data = pd.concat(
        parts,
        ignore_index=True,
    )

    data = data.rename(
        columns={
            "LinkID": "txdot_link_id",
            "GID": "txdot_source_gid",
            "RDBD_TYPE": "txdot_roadbed_type",
            "DES_DRCT": "txdot_direction",
            "COUNTY": "txdot_county",
            "MAP_LBL": "txdot_road_label",
            "SpeedLimit": "txdot_speed_limit",
            "NumberOfLanes": "txdot_lanes",
            "Ramp_type": "txdot_ramp_type",
            "Functional Class": "txdot_functional_class",
        }
    )

    data["txdot_link_id"] = pd.to_numeric(
        data["txdot_link_id"],
        errors="coerce",
    )

    data["txdot_speed_limit"] = pd.to_numeric(
        data["txdot_speed_limit"],
        errors="coerce",
    )

    data["txdot_lanes"] = pd.to_numeric(
        data["txdot_lanes"],
        errors="coerce",
    )

    return data.drop_duplicates(
        subset="txdot_link_id"
    )


# Summarize hourly performance of each corridor link
def summarize_moes(moe):

    return (
        moe.groupby(
            "link_key",
            as_index=False,
        )
        .agg(
            total_waypoints=(
                "waypoint_count",
                "sum",
            ),
            mean_speed_mph=(
                "avg_speed_mph",
                "mean",
            ),
            minimum_speed_mph=(
                "avg_speed_mph",
                "min",
            ),
            mean_speed_reduction_pct=(
                "speed_reduction_pct",
                "mean",
            ),
            max_speed_reduction_pct=(
                "speed_reduction_pct",
                "max",
            ),
            mean_travel_time_s=(
                "avg_travel_time_s",
                "mean",
            ),
            max_travel_time_s=(
                "avg_travel_time_s",
                "max",
            ),
            mean_DSH=(
                "DSH",
                "mean",
            ),
            max_DSH=(
                "DSH",
                "max",
            ),
            slow_movement_count=(
                "slow_movement_count",
                "sum",
            ),
            mean_slow_movement_pct=(
                "slow_movement_pct",
                "mean",
            ),
        )
    )


# Combine spatial, roadway, and MOE information
def build_diagnostic(
    corridor,
    network,
    txdot,
    moe_summary,
):

    result = corridor.merge(
        network,
        on="link_key",
        how="left",
        suffixes=("", "_network"),
    )

    result["txdot_link_id"] = pd.to_numeric(
        result["txdot_link_id"],
        errors="coerce",
    )

    if not txdot.empty:

        result = result.merge(
            txdot,
            on="txdot_link_id",
            how="left",
        )

    result = result.merge(
        moe_summary,
        on="link_key",
        how="left",
    )

    # Coordinate string that can be pasted directly into Google Maps
    result["google_maps_coordinate"] = (
        result["mid_lat"]
        .round(7)
        .astype(str)
        + ", "
        + result["mid_lon"]
        .round(7)
        .astype(str)
    )

    result["google_maps_url"] = (
        "https://www.google.com/maps/search/?api=1&query="
        + result["mid_lat"].astype(str)
        + ","
        + result["mid_lon"].astype(str)
    )

    return result.sort_values(
        "corridor_order"
    )


# Save a compact map-tracing table
def save_map_points(result):

    columns = [
        "corridor_order",
        "link_key",
        "distance_start_ft",
        "distance_end_ft",
        "link_length_ft",
        "name",
        "facility_type",
        "txdot_road_label",
        "txdot_roadbed_type",
        "txdot_ramp_type",
        "txdot_direction",
        "posted_speed_limit_mph",
        "start_lat",
        "start_lon",
        "mid_lat",
        "mid_lon",
        "end_lat",
        "end_lon",
        "google_maps_coordinate",
        "google_maps_url",
    ]

    columns = [
        column
        for column in columns
        if column in result.columns
    ]

    result[
        columns
    ].to_csv(
        MAP_FILE,
        index=False,
    )


# Print corridor identity and worst-performing links
def print_summary(result):

    print()
    print("=" * 100)
    print("CORRIDOR SEQUENCE")
    print("=" * 100)

    columns = [
        "corridor_order",
        "link_key",
        "distance_start_ft",
        "distance_end_ft",
        "name",
        "txdot_road_label",
        "txdot_roadbed_type",
        "txdot_ramp_type",
        "posted_speed_limit_mph",
        "google_maps_coordinate",
    ]

    columns = [
        column
        for column in columns
        if column in result.columns
    ]

    print(
        result[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("WORST SPEED-REDUCTION LINKS")
    print("=" * 100)

    columns = [
        "corridor_order",
        "link_key",
        "distance_start_ft",
        "distance_end_ft",
        "name",
        "txdot_road_label",
        "txdot_roadbed_type",
        "txdot_ramp_type",
        "posted_speed_limit_mph",
        "mean_speed_mph",
        "mean_speed_reduction_pct",
        "max_speed_reduction_pct",
        "total_waypoints",
        "google_maps_coordinate",
    ]

    columns = [
        column
        for column in columns
        if column in result.columns
    ]

    print(
        result.sort_values(
            "mean_speed_reduction_pct",
            ascending=False,
        )[
            columns
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("WORST TRAVEL-TIME LINKS")
    print("=" * 100)

    columns = [
        "corridor_order",
        "link_key",
        "distance_start_ft",
        "distance_end_ft",
        "name",
        "txdot_road_label",
        "txdot_roadbed_type",
        "txdot_ramp_type",
        "link_length_ft",
        "mean_travel_time_s",
        "max_travel_time_s",
        "mean_speed_mph",
        "total_waypoints",
        "google_maps_coordinate",
    ]

    columns = [
        column
        for column in columns
        if column in result.columns
    ]

    print(
        result.sort_values(
            "mean_travel_time_s",
            ascending=False,
        )[
            columns
        ]
        .head(15)
        .to_string(
            index=False
        )
    )


def main():

    # Step 1: Load current corridor and MOEs
    corridor, moe = load_results()

    print("CORRIDOR TRACEBACK")
    print("=" * 100)
    print(
        f"Links:  {len(corridor):,}"
    )
    print(
        f"Length: "
        f"{corridor['link_length_ft'].sum():,.1f} ft"
    )

    # Step 2: Extract original roadway metadata and geographic coordinates
    network = load_network_metadata(
        corridor
    )

    # Step 3: Retrieve matched TxDOT attributes
    txdot = load_txdot_metadata(
        corridor
    )

    # Step 4: Summarize link performance
    moe_summary = summarize_moes(
        moe
    )

    # Step 5: Combine everything
    result = build_diagnostic(
        corridor,
        network,
        txdot,
        moe_summary,
    )

    # Step 6: Save full diagnostic file
    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # Step 7: Save compact Google Maps traceback table
    save_map_points(
        result
    )

    # Step 8: Print important links
    print_summary(
        result
    )

    print()
    print("=" * 100)
    print(
        f"Diagnostic: {OUTPUT_FILE}"
    )
    print(
        f"Map points: {MAP_FILE}"
    )


if __name__ == "__main__":
    main()