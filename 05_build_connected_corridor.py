import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt
from shapely.ops import transform as shapely_transform


# Paths and settings
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CV_FILE = (
    PROJECT_ROOT
    / "Output"
    / "Final"
    / "9_20_2025"
    / "9_20_2025_hr=19.csv"
)

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

CV_CHUNK_SIZE = 1_000_000
NETWORK_CHUNK_SIZE = 500_000
TXDOT_CHUNK_SIZE = 500_000

MAX_WAYPOINT_GAP_SECONDS = 120

# None = automatically choose a high-support starting link
START_LINK_KEY = None

# Display extent only
MAX_CORRIDOR_LINKS = 40

# Conservative spatial matching checks
MAX_TXDOT_MATCH_DISTANCE_FT = 100.0
MAX_TXDOT_HEADING_DIFFERENCE_DEG = 35.0

SOURCE_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:3083"

M_TO_FT = 3.280839895

TRANSFORMER = Transformer.from_crs(
    SOURCE_CRS,
    PROJECTED_CRS,
    always_xy=True,
)


# Split CV journeys and identify consecutive roadway-link runs
def prepare_link_runs(df):

    data = df[
        df["journey_id"].notna()
    ].copy()

    if data.empty:
        return pd.DataFrame()

    data = data.sort_values(
        ["journey_id", "capture_time"]
    ).reset_index(drop=True)

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

    runs = (
        data.groupby(
            "run_id",
            as_index=False,
            sort=False,
        )
        .agg(
            trip_segment_id=("trip_segment_id", "first"),
            link_key=("link_key", "first"),
        )
    )

    return runs


# Count observed directional transitions between matched CV links
def calculate_transition_counts(df):

    runs = prepare_link_runs(df)

    if runs.empty:
        return None

    runs["next_trip"] = (
        runs["trip_segment_id"]
        .shift(-1)
    )

    runs["next_link"] = (
        runs["link_key"]
        .shift(-1)
    )

    transitions = runs[
        (runs["trip_segment_id"] == runs["next_trip"])
        & runs["link_key"].notna()
        & runs["next_link"].notna()
        & (runs["link_key"] != runs["next_link"])
    ].copy()

    if transitions.empty:
        return None

    return (
        transitions.groupby(
            ["link_key", "next_link"],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "transition_count"
            }
        )
    )


# Read the hourly CV file and aggregate link-to-link transition support
def build_transition_table():

    parts = []

    carry = pd.DataFrame()
    total_rows = 0

    reader = pd.read_csv(
        CV_FILE,
        usecols=[
            "journey_id",
            "capture_time",
            "link_key",
        ],
        chunksize=CV_CHUNK_SIZE,
        dtype={
            "journey_id": "string",
            "link_key": "string",
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

        if not complete.empty:

            result = calculate_transition_counts(
                complete
            )

            if result is not None:
                parts.append(result)

        print(
            f"CV chunk {chunk_number:>3} | "
            f"Rows processed: {total_rows:,}"
        )

    if not carry.empty:

        result = calculate_transition_counts(
            carry
        )

        if result is not None:
            parts.append(result)

    transitions = (
        pd.concat(
            parts,
            ignore_index=True,
        )
        .groupby(
            ["link_key", "next_link"],
            as_index=False,
        )["transition_count"]
        .sum()
    )

    return transitions


# Load lightweight OSM short-link topology for network-connected path building
def load_network_topology():

    use_columns = [
        "link_key",
        "from_node_id",
        "to_node_id",
        "heading",
        "length",
    ]

    parts = []
    total_rows = 0

    reader = pd.read_csv(
        NETWORK_FILE,
        usecols=use_columns,
        chunksize=NETWORK_CHUNK_SIZE,
        dtype={
            "link_key": "string",
        },
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):

        total_rows += len(chunk)

        parts.append(chunk)

        if chunk_number % 10 == 0:

            print(
                f"Network topology chunk {chunk_number:>3} | "
                f"Rows loaded: {total_rows:,}"
            )

    topology = pd.concat(
        parts,
        ignore_index=True,
    )

    topology["from_node_id"] = pd.to_numeric(
        topology["from_node_id"],
        errors="coerce",
    )

    topology["to_node_id"] = pd.to_numeric(
        topology["to_node_id"],
        errors="coerce",
    )

    topology["heading"] = pd.to_numeric(
        topology["heading"],
        errors="coerce",
    )

    topology["length"] = pd.to_numeric(
        topology["length"],
        errors="coerce",
    )

    topology = topology[
        topology["link_key"].notna()
        & topology["from_node_id"].notna()
        & topology["to_node_id"].notna()
    ].copy()

    print()
    print(
        f"Network topology links loaded: "
        f"{len(topology):,}"
    )

    return topology


# Return the smallest circular difference between two headings
def heading_difference(
    heading_a,
    heading_b,
):

    if (
        pd.isna(heading_a)
        or pd.isna(heading_b)
    ):
        return np.nan

    difference = abs(
        float(heading_a)
        - float(heading_b)
    )

    return min(
        difference,
        360.0 - difference,
    )


# Return the first two pieces of a short-link key
def get_link_family(link_key):

    parts = str(link_key).split("_")

    if len(parts) >= 2:
        return "_".join(parts[:2])

    return str(link_key)


# Choose a well-supported observed starting link
def choose_start_link(
    transitions,
    topology_by_link,
):

    if START_LINK_KEY is not None:

        start_link = str(
            START_LINK_KEY
        )

        if start_link not in topology_by_link.index:

            raise ValueError(
                f"START_LINK_KEY not found: "
                f"{start_link}"
            )

        return start_link

    support = (
        transitions.groupby(
            "link_key"
        )["transition_count"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    for link_key in support.index:

        link_key = str(link_key)

        if link_key in topology_by_link.index:
            return link_key

    raise RuntimeError(
        "No CV transition link was found "
        "in the roadway topology."
    )


# Build a physically connected directional corridor from network topology
def build_connected_corridor(
    transitions,
    topology,
):

    topology_by_link = (
        topology
        .drop_duplicates(
            subset=["link_key"],
            keep="first",
        )
        .set_index(
            "link_key",
            drop=False,
        )
    )

    outgoing = (
        topology
        .set_index(
            "from_node_id",
            drop=False,
        )
        .sort_index()
    )

    transition_lookup = (
        transitions
        .set_index(
            ["link_key", "next_link"]
        )["transition_count"]
    )

    start_link = choose_start_link(
        transitions,
        topology_by_link,
    )

    path = [
        start_link
    ]

    transition_support = [
        np.nan
    ]

    selection_method = [
        "start"
    ]

    visited = {
        start_link
    }

    current_link = start_link

    while len(path) < MAX_CORRIDOR_LINKS:

        current = topology_by_link.loc[
            current_link
        ]

        current_from = current[
            "from_node_id"
        ]

        current_to = current[
            "to_node_id"
        ]

        current_heading = current[
            "heading"
        ]

        current_family = get_link_family(
            current_link
        )

        if current_to not in outgoing.index:
            break

        candidates = outgoing.loc[
            [current_to]
        ].copy()

        candidates = candidates[
            ~candidates["link_key"]
            .isin(visited)
        ].copy()

        if candidates.empty:
            break

        # Avoid immediately reversing onto the previous node when alternatives exist
        non_reverse = candidates[
            candidates["to_node_id"]
            != current_from
        ].copy()

        if not non_reverse.empty:
            candidates = non_reverse

        candidates["transition_support"] = [
            int(
                transition_lookup.get(
                    (
                        current_link,
                        str(candidate),
                    ),
                    0,
                )
            )
            for candidate in candidates[
                "link_key"
            ]
        ]

        candidates[
            "heading_difference"
        ] = candidates[
            "heading"
        ].apply(
            lambda value:
            heading_difference(
                current_heading,
                value,
            )
        )

        candidates[
            "same_link_family"
        ] = candidates[
            "link_key"
        ].apply(
            lambda value:
            get_link_family(value)
            == current_family
        )

        observed = candidates[
            candidates[
                "transition_support"
            ] > 0
        ].copy()

        if not observed.empty:

            observed = observed.sort_values(
                by=[
                    "transition_support",
                    "same_link_family",
                    "heading_difference",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
                na_position="last",
            )

            selected = observed.iloc[0]

            method = (
                "observed_cv_transition"
            )

        else:

            candidates = candidates.sort_values(
                by=[
                    "same_link_family",
                    "heading_difference",
                ],
                ascending=[
                    False,
                    True,
                ],
                na_position="last",
            )

            selected = candidates.iloc[0]

            method = (
                "network_continuity"
            )

        next_link = str(
            selected["link_key"]
        )

        path.append(
            next_link
        )

        transition_support.append(
            int(
                selected[
                    "transition_support"
                ]
            )
        )

        selection_method.append(
            method
        )

        visited.add(
            next_link
        )

        current_link = next_link

    corridor = pd.DataFrame(
        {
            "corridor_order": np.arange(
                1,
                len(path) + 1,
            ),
            "link_key": path,
            "transition_support": transition_support,
            "selection_method": selection_method,
        }
    )

    return corridor


# Read full metadata only for corridor links
def load_corridor_network_metadata(
    corridor,
):

    corridor_set = set(
        corridor["link_key"]
    )

    use_columns = [
        "link_key",
        "from_node_id",
        "to_node_id",
        "geometry",
        "heading",
        "length",
        "Matched_GID",
    ]

    parts = []
    found = set()

    reader = pd.read_csv(
        NETWORK_FILE,
        usecols=use_columns,
        chunksize=NETWORK_CHUNK_SIZE,
        dtype={
            "link_key": "string",
            "Matched_GID": "string",
        },
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):

        matched = chunk[
            chunk["link_key"]
            .isin(corridor_set)
        ].copy()

        if not matched.empty:

            parts.append(matched)

            found.update(
                matched["link_key"]
                .dropna()
                .tolist()
            )

        if corridor_set.issubset(found):
            break

        if chunk_number % 10 == 0:

            print(
                f"Network metadata chunk {chunk_number:>3} | "
                f"Found: {len(found):,}/"
                f"{len(corridor_set):,}"
            )

    metadata = pd.concat(
        parts,
        ignore_index=True,
    )

    metadata = metadata.drop_duplicates(
        subset=["link_key"],
        keep="first",
    )

    return metadata


# Convert a valid Matched_GID to the integer TxDOT GID
def normalize_matched_gid(value):

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if not re.fullmatch(
        r"\d+(?:\.0+)?",
        text,
    ):
        return np.nan

    gid = int(
        float(text)
    )

    if gid <= 0:
        return np.nan

    return gid


# Convert WKT geometry from WGS84 to the project CRS
def project_wkt_geometry(
    geometry_text,
):

    if pd.isna(geometry_text):
        return None

    try:

        geometry = wkt.loads(
            str(geometry_text)
        )

        return shapely_transform(
            TRANSFORMER.transform,
            geometry,
        )

    except Exception:

        return None


# Read only TxDOT segments belonging to corridor GIDs
def load_txdot_candidates(
    corridor_metadata,
):

    valid_gids = set(
        corridor_metadata[
            "txdot_gid"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    if not valid_gids:
        return pd.DataFrame()

    use_columns = [
        "LinkID",
        "FromNode",
        "ToNode",
        "GID",
        "Geometry",
        "Heading",
        "SpeedLimit",
        "length",
    ]

    parts = []
    total_rows = 0
    matched_rows = 0

    reader = pd.read_csv(
        TXDOT_FILE,
        usecols=use_columns,
        chunksize=TXDOT_CHUNK_SIZE,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):

        total_rows += len(chunk)

        chunk["GID"] = pd.to_numeric(
            chunk["GID"],
            errors="coerce",
        )

        matched = chunk[
            chunk["GID"]
            .isin(valid_gids)
        ].copy()

        if not matched.empty:

            parts.append(matched)

            matched_rows += len(
                matched
            )

        if chunk_number % 5 == 0:

            print(
                f"TxDOT chunk {chunk_number:>3} | "
                f"Rows scanned: {total_rows:,} | "
                f"Candidate rows: {matched_rows:,}"
            )

    if not parts:
        return pd.DataFrame()

    candidates = pd.concat(
        parts,
        ignore_index=True,
    )

    candidates["SpeedLimit"] = (
        pd.to_numeric(
            candidates["SpeedLimit"],
            errors="coerce",
        )
    )

    candidates["Heading"] = (
        pd.to_numeric(
            candidates["Heading"],
            errors="coerce",
        )
    )

    candidates[
        "_geometry_projected"
    ] = candidates[
        "Geometry"
    ].apply(
        project_wkt_geometry
    )

    return candidates


# Match each OSM short link to the correct TxDOT speed-limit segment
def assign_posted_speed_limits(
    corridor_metadata,
    txdot_candidates,
):

    result = corridor_metadata.copy()

    result[
        "_geometry_projected"
    ] = result[
        "geometry"
    ].apply(
        project_wkt_geometry
    )

    result[
        "posted_speed_limit_mph"
    ] = np.nan

    result[
        "txdot_link_id"
    ] = pd.NA

    result[
        "txdot_match_distance_ft"
    ] = np.nan

    result[
        "txdot_heading_difference_deg"
    ] = np.nan

    result[
        "txdot_candidate_count"
    ] = 0

    result[
        "txdot_valid_speed_candidate_count"
    ] = 0

    result[
        "speed_match_status"
    ] = "unmatched"

    if txdot_candidates.empty:
        return result

    for index, row in result.iterrows():

        gid = row[
            "txdot_gid"
        ]

        osm_geometry = row[
            "_geometry_projected"
        ]

        if pd.isna(gid):

            result.at[
                index,
                "speed_match_status",
            ] = "invalid_matched_gid"

            continue

        if (
            osm_geometry is None
            or osm_geometry.is_empty
        ):

            result.at[
                index,
                "speed_match_status",
            ] = "invalid_osm_geometry"

            continue

        candidates = txdot_candidates[
            txdot_candidates["GID"]
            == int(gid)
        ].copy()

        result.at[
            index,
            "txdot_candidate_count",
        ] = len(candidates)

        candidates = candidates[
            candidates[
                "_geometry_projected"
            ].notna()
        ].copy()

        valid_speed = candidates[
            candidates["SpeedLimit"].notna()
            & np.isfinite(
                candidates["SpeedLimit"]
            )
            & (
                candidates["SpeedLimit"]
                > 0
            )
        ].copy()

        result.at[
            index,
            "txdot_valid_speed_candidate_count",
        ] = len(valid_speed)

        if valid_speed.empty:

            result.at[
                index,
                "speed_match_status",
            ] = "no_valid_speed_for_gid"

            continue

        midpoint = osm_geometry.interpolate(
            0.5,
            normalized=True,
        )

        valid_speed[
            "_distance_ft"
        ] = valid_speed[
            "_geometry_projected"
        ].apply(
            lambda geometry:
            midpoint.distance(
                geometry
            )
            * M_TO_FT
        )

        valid_speed[
            "_heading_difference"
        ] = valid_speed[
            "Heading"
        ].apply(
            lambda value:
            heading_difference(
                row["heading"],
                value,
            )
        )

        heading_compatible = valid_speed[
            valid_speed[
                "_heading_difference"
            ].isna()
            | (
                valid_speed[
                    "_heading_difference"
                ]
                <= MAX_TXDOT_HEADING_DIFFERENCE_DEG
            )
        ].copy()

        if heading_compatible.empty:

            result.at[
                index,
                "speed_match_status",
            ] = "no_heading_compatible_speed"

            continue

        heading_compatible = (
            heading_compatible.sort_values(
                by=[
                    "_distance_ft",
                    "_heading_difference",
                ],
                ascending=[
                    True,
                    True,
                ],
                na_position="last",
            )
        )

        best = (
            heading_compatible.iloc[0]
        )

        distance_ft = float(
            best["_distance_ft"]
        )

        heading_diff = (
            best[
                "_heading_difference"
            ]
        )

        result.at[
            index,
            "txdot_match_distance_ft",
        ] = distance_ft

        result.at[
            index,
            "txdot_heading_difference_deg",
        ] = heading_diff

        result.at[
            index,
            "txdot_link_id",
        ] = best["LinkID"]

        if (
            distance_ft
            > MAX_TXDOT_MATCH_DISTANCE_FT
        ):

            result.at[
                index,
                "speed_match_status",
            ] = "txdot_segment_too_far"

            continue

        result.at[
            index,
            "posted_speed_limit_mph",
        ] = float(
            best["SpeedLimit"]
        )

        result.at[
            index,
            "speed_match_status",
        ] = "matched"

    return result


# Calculate actual geometry endpoint gaps between consecutive links
def add_connection_diagnostics(
    corridor,
):

    result = corridor.sort_values(
        "corridor_order"
    ).reset_index(drop=True)

    gaps = [
        0.0
    ]

    topology_checks = [
        True
    ]

    for index in range(
        1,
        len(result),
    ):

        previous = result.iloc[
            index - 1
        ]

        current = result.iloc[
            index
        ]

        topology_connected = (
            previous["to_node_id"]
            == current["from_node_id"]
        )

        topology_checks.append(
            bool(topology_connected)
        )

        previous_geometry = previous[
            "_geometry_projected"
        ]

        current_geometry = current[
            "_geometry_projected"
        ]

        if (
            previous_geometry is None
            or current_geometry is None
            or previous_geometry.is_empty
            or current_geometry.is_empty
        ):

            gaps.append(
                np.nan
            )

            continue

        previous_end = list(
            previous_geometry.coords
        )[-1]

        current_start = list(
            current_geometry.coords
        )[0]

        gap_ft = (
            np.hypot(
                previous_end[0]
                - current_start[0],
                previous_end[1]
                - current_start[1],
            )
            * M_TO_FT
        )

        gaps.append(
            gap_ft
        )

    result[
        "topology_connected_from_previous"
    ] = topology_checks

    result[
        "geometry_connection_gap_ft"
    ] = gaps

    return result


# Add actual roadway length and cumulative corridor distance
def add_spatial_extent(
    corridor,
):

    result = corridor.copy()

    result[
        "link_length_ft"
    ] = pd.to_numeric(
        result["length"],
        errors="coerce",
    )

    if result[
        "link_length_ft"
    ].isna().any():

        missing = result[
            result[
                "link_length_ft"
            ].isna()
        ]["link_key"]

        raise RuntimeError(
            "Missing short-link length for:\n"
            + "\n".join(
                missing.astype(str)
            )
        )

    result[
        "distance_start_ft"
    ] = (
        result["link_length_ft"]
        .cumsum()
        .shift(
            fill_value=0.0
        )
    )

    result[
        "distance_end_ft"
    ] = (
        result["distance_start_ft"]
        + result["link_length_ft"]
    )

    return result


def main():

    start_time = time.perf_counter()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("BUILDING CONNECTED CORRIDOR")
    print("=" * 75)

    # Step 1: Build observed CV transition support
    transitions = build_transition_table()

    # Step 2: Load actual network topology
    print()
    print("LOADING NETWORK TOPOLOGY")
    print("=" * 75)

    topology = load_network_topology()

    # Step 3: Build a strictly connected directional path
    print()
    print("BUILDING TOPOLOGY-CONNECTED PATH")
    print("=" * 75)

    corridor = build_connected_corridor(
        transitions,
        topology,
    )

    print(
        f"Connected links selected: "
        f"{len(corridor):,}"
    )

    # Release the full topology before geometry / TxDOT work
    del topology

    # Step 4: Load full metadata for only those links
    print()
    print("LOADING CORRIDOR ROADWAY METADATA")
    print("=" * 75)

    metadata = (
        load_corridor_network_metadata(
            corridor
        )
    )

    metadata[
        "txdot_gid"
    ] = metadata[
        "Matched_GID"
    ].apply(
        normalize_matched_gid
    )

    corridor = corridor.merge(
        metadata,
        on="link_key",
        how="left",
    )

    # Step 5: Load only TxDOT rows belonging to corridor GIDs
    print()
    print("LOADING TXDOT SPEED CANDIDATES")
    print("=" * 75)

    txdot_candidates = (
        load_txdot_candidates(
            corridor
        )
    )

    print()
    print(
        f"TxDOT candidate segments loaded: "
        f"{len(txdot_candidates):,}"
    )

    # Step 6: Spatially assign the actual posted speed limit
    print()
    print("MATCHING POSTED SPEED LIMITS")
    print("=" * 75)

    corridor = (
        assign_posted_speed_limits(
            corridor,
            txdot_candidates,
        )
    )

    del txdot_candidates

    # Step 7: Verify physical connectivity
    corridor = (
        add_connection_diagnostics(
            corridor
        )
    )

    # Step 8: Use the existing short-link length for spatial extent
    corridor = (
        add_spatial_extent(
            corridor
        )
    )

    # Step 9: Keep final diagnostics and save
    output_columns = [
        "corridor_order",
        "link_key",
        "from_node_id",
        "to_node_id",
        "Matched_GID",
        "txdot_gid",
        "heading",
        "transition_support",
        "selection_method",
        "link_length_ft",
        "distance_start_ft",
        "distance_end_ft",
        "topology_connected_from_previous",
        "geometry_connection_gap_ft",
        "posted_speed_limit_mph",
        "txdot_link_id",
        "txdot_match_distance_ft",
        "txdot_heading_difference_deg",
        "txdot_candidate_count",
        "txdot_valid_speed_candidate_count",
        "speed_match_status",
    ]

    corridor = corridor[
        output_columns
    ].copy()

    corridor.to_csv(
        CORRIDOR_FILE,
        index=False,
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    topology_breaks = (
        ~corridor[
            "topology_connected_from_previous"
        ]
    ).sum()

    speed_matched = (
        corridor[
            "speed_match_status"
        ]
        .eq("matched")
        .sum()
    )

    missing_speed = (
        corridor[
            "posted_speed_limit_mph"
        ]
        .isna()
        .sum()
    )

    print()
    print("=" * 75)
    print("CORRIDOR PREPARATION COMPLETE")
    print(
        f"Links:                 "
        f"{len(corridor):,}"
    )
    print(
        f"Corridor length:       "
        f"{corridor['link_length_ft'].sum():,.1f} ft"
    )
    print(
        f"Topology breaks:       "
        f"{topology_breaks:,}"
    )
    print(
        f"Maximum geometry gap:  "
        f"{corridor['geometry_connection_gap_ft'].max():,.2f} ft"
    )
    print(
        f"Posted speeds matched: "
        f"{speed_matched:,}/{len(corridor):,}"
    )
    print(
        f"Missing posted speeds: "
        f"{missing_speed:,}"
    )
    print(
        f"Runtime:               "
        f"{runtime / 60:.2f} min"
    )
    print(
        f"Output:                "
        f"{CORRIDOR_FILE}"
    )

    print()
    print("SPEED MATCH STATUS")
    print(
        corridor[
            "speed_match_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print("FINAL CORRIDOR")
    print(
        corridor[
            [
                "corridor_order",
                "link_key",
                "link_length_ft",
                "posted_speed_limit_mph",
                "txdot_match_distance_ft",
                "txdot_heading_difference_deg",
                "selection_method",
                "geometry_connection_gap_ft",
                "speed_match_status",
            ]
        ]
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()