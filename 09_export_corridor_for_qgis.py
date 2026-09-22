from pathlib import Path

import pandas as pd
import geopandas as gpd
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

OUTPUT_GPKG = RESULTS_DIR / "corridor_qgis.gpkg"

NETWORK_CHUNK_SIZE = 500_000
TXDOT_CHUNK_SIZE = 500_000


# Read the 40 selected corridor links
def load_corridor():

    corridor = pd.read_csv(
        CORRIDOR_FILE,
        dtype={
            "link_key": "string",
        },
    )

    return corridor


# Extract only the selected 40 links from the huge short-link network
def extract_corridor_network(corridor):

    target_links = set(
        corridor["link_key"]
    )

    desired = [
        "link_key",
        "name",
        "osm_way_id",
        "from_node_id",
        "to_node_id",
        "geometry",
        "facility_type",
        "link_type",
        "heading",
        "length",
        "Matched_GID",
        "RDBD_TYPE",
        "Functional Class",
    ]

    header = pd.read_csv(
        NETWORK_FILE,
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
            NETWORK_FILE,
            usecols=usecols,
            chunksize=NETWORK_CHUNK_SIZE,
            dtype="string",
        ),
        start=1,
    ):

        total_rows += len(chunk)

        selected = chunk[
            chunk["link_key"].isin(
                target_links
            )
        ].copy()

        if not selected.empty:
            parts.append(selected)

        found = (
            pd.concat(parts)["link_key"].nunique()
            if parts
            else 0
        )

        print(
            f"Network chunk {chunk_number:>3} | "
            f"Rows: {total_rows:,} | "
            f"Found: {found}/"
            f"{len(target_links)}"
        )

        if found == len(target_links):
            break

    network = pd.concat(
        parts,
        ignore_index=True,
    )

    network = network.drop_duplicates(
        subset="link_key"
    )

    network = network.merge(
        corridor[
            [
                "corridor_order",
                "link_key",
                "distance_start_ft",
                "distance_end_ft",
                "posted_speed_limit_mph",
            ]
        ],
        on="link_key",
        how="left",
    )

    network["geometry"] = (
        network["geometry"]
        .apply(wkt.loads)
    )

    return gpd.GeoDataFrame(
        network,
        geometry="geometry",
        crs="EPSG:4326",
    )


# Extract TxDOT segments belonging to corridor GIDs
def extract_txdot_segments(corridor):

    gid_column = None

    for candidate in [
        "txdot_gid",
        "Matched_GID",
    ]:

        if candidate in corridor.columns:
            gid_column = candidate
            break

    if gid_column is None:
        return None

    target_gids = set(
        pd.to_numeric(
            corridor[gid_column],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    if not target_gids:
        return None

    desired = [
        "LinkID",
        "GID",
        "Geometry",
        "RDBD_TYPE",
        "DES_DRCT",
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

        gids = pd.to_numeric(
            chunk["GID"],
            errors="coerce",
        )

        selected = chunk[
            gids.isin(
                target_gids
            )
        ].copy()

        if not selected.empty:
            parts.append(selected)

        print(
            f"TxDOT chunk {chunk_number:>3} | "
            f"Rows: {total_rows:,}"
        )

    if not parts:
        return None

    txdot = pd.concat(
        parts,
        ignore_index=True,
    )

    txdot["geometry"] = (
        txdot["Geometry"]
        .apply(wkt.loads)
    )

    return gpd.GeoDataFrame(
        txdot,
        geometry="geometry",
        crs="EPSG:4326",
    )


def main():

    # Step 1: Load the visualized corridor
    corridor = load_corridor()

    print(
        f"Corridor links: {len(corridor)}"
    )

    # Step 2: Extract only those links
    network = extract_corridor_network(
        corridor
    )

    # Step 3: Save tiny corridor layer
    network.to_file(
        OUTPUT_GPKG,
        layer="corridor_links",
        driver="GPKG",
    )

    print(
        f"Saved corridor layer: "
        f"{len(network)} links"
    )

    # Step 4: Extract relevant TxDOT segments
    txdot = extract_txdot_segments(
        corridor
    )

    if txdot is not None:

        txdot.to_file(
            OUTPUT_GPKG,
            layer="txdot_segments",
            driver="GPKG",
        )

        print(
            f"Saved TxDOT layer: "
            f"{len(txdot):,} segments"
        )

    print()
    print(
        f"QGIS file:\n{OUTPUT_GPKG}"
    )


if __name__ == "__main__":
    main()