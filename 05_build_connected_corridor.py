from pathlib import Path
from collections import Counter, defaultdict
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import shape


PROJECT_ROOT = Path(r"D:\Sabin\Streetlight-Data-Processing")
MOE_DIR = PROJECT_ROOT / "MOE for selected links"
RESULTS_DIR = MOE_DIR / "Results"

CV_FILE = PROJECT_ROOT / "Output" / "Final" / "9_20_2025" / "9_20_2025_hr=19.csv"
NETWORK_FILE = PROJECT_ROOT / "Initial Input Files" / "OSM_Short_Level_CSV" / "Complete_OSM_Short_Level_Link_List.csv"
TXDOT_FILE = PROJECT_ROOT / "Initial Input Files" / "TxDOT" / "link_list_with_speed_NOL.csv"
OUTPUT_FILE = RESULTS_DIR / "corridor_links.csv"

CHUNK_SIZE = 1_000_000
TRIP_GAP_SECONDS = 120
MAX_CORRIDOR_LINKS = 40
MAX_TXDOT_MATCH_DISTANCE_FT = 100.0
MAX_TXDOT_HEADING_DIFFERENCE_DEG = 35.0
MANUAL_START_LINK = None


def normalize_gid(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return None if text in {"", "nan", "None", "-1"} else text


def parse_geometry(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    try:
        return shape(json.loads(text)) if text.startswith("{") else wkt.loads(text)
    except Exception:
        return None


def angular_difference(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)


def link_family(link_key):
    parts = str(link_key).split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else str(link_key)


def count_transitions(batch, counts, outgoing_total):
    batch = batch.copy()
    batch["capture_time"] = pd.to_datetime(batch["capture_time"], errors="coerce")
    batch = batch.sort_values(["journey_id", "capture_time"], kind="mergesort")

    previous_journey = batch["journey_id"].shift()
    previous_time = batch["capture_time"].shift()
    previous_link = batch["link_key"].shift()
    gap = (batch["capture_time"] - previous_time).dt.total_seconds()

    valid = (
        batch["journey_id"].eq(previous_journey)
        & gap.le(TRIP_GAP_SECONDS)
        & batch["link_key"].notna()
        & previous_link.notna()
        & batch["link_key"].ne(previous_link)
    )

    transitions = pd.DataFrame(
        {
            "from_link": previous_link[valid].astype(str),
            "to_link": batch.loc[valid, "link_key"].astype(str),
        }
    )
    grouped = transitions.value_counts().rename("count").reset_index()

    for row in grouped.itertuples(index=False):
        counts[(row.from_link, row.to_link)] += int(row.count)
        outgoing_total[row.from_link] += int(row.count)


def build_transition_support():
    counts = Counter()
    outgoing_total = Counter()
    carry = pd.DataFrame()

    reader = pd.read_csv(
        CV_FILE,
        usecols=["journey_id", "capture_time", "link_key"],
        chunksize=CHUNK_SIZE,
        dtype={"journey_id": "string", "link_key": "string"},
        low_memory=False,
    )

    for number, chunk in enumerate(reader, start=1):
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = pd.DataFrame()
        if chunk.empty:
            continue

        last_journey = chunk["journey_id"].iloc[-1]
        carry = chunk[chunk["journey_id"] == last_journey].copy()
        complete = chunk[chunk["journey_id"] != last_journey].copy()
        if not complete.empty:
            count_transitions(complete, counts, outgoing_total)
        print(f"Transition chunk {number:,}")

    if not carry.empty:
        count_transitions(carry, counts, outgoing_total)

    return counts, outgoing_total


def load_network():
    requested = [
        "link_key", "from_node_id", "to_node_id", "geometry", "heading", "length",
        "Matched_GID", "name", "facility_type", "RDBD_TYPE", "Functional Class",
    ]
    try:
        network = pd.read_csv(NETWORK_FILE, usecols=requested, dtype="string", low_memory=False)
    except ValueError:
        fallback = [
            "link_key", "from_node_id", "to_node_id", "geometry", "heading", "length",
            "matched_gid", "name", "facility_type", "rdbd_type", "functional_class",
        ]
        network = pd.read_csv(NETWORK_FILE, usecols=fallback, dtype="string", low_memory=False)
        network = network.rename(
            columns={
                "matched_gid": "Matched_GID",
                "rdbd_type": "RDBD_TYPE",
                "functional_class": "Functional Class",
            }
        )

    for column in ["from_node_id", "to_node_id", "heading", "length"]:
        network[column] = pd.to_numeric(network[column], errors="coerce")

    network["link_key"] = network["link_key"].astype(str)
    network["Matched_GID"] = network["Matched_GID"].map(normalize_gid)
    network["geometry_obj"] = network["geometry"].map(parse_geometry)
    return network.drop_duplicates("link_key").set_index("link_key", drop=False)


def choose_next_link(current_key, previous_key, network, outgoing_by_node, transitions):
    current = network.loc[current_key]
    candidates = outgoing_by_node.get(current["to_node_id"], [])
    choices = []

    for candidate_key in candidates:
        candidate = network.loc[candidate_key]
        if previous_key is not None and candidate["to_node_id"] == current["from_node_id"]:
            continue

        choices.append(
            {
                "link_key": candidate_key,
                "transition_support": transitions.get((current_key, candidate_key), 0),
                "same_family": int(link_family(current_key) == link_family(candidate_key)),
                "heading_difference": angular_difference(current["heading"], candidate["heading"]),
            }
        )

    if not choices:
        return None, 0, "dead_end"

    table = pd.DataFrame(choices)
    table["heading_difference"] = table["heading_difference"].fillna(999.0)
    best = table.sort_values(
        ["transition_support", "same_family", "heading_difference"],
        ascending=[False, False, True],
    ).iloc[0]

    if best["transition_support"] > 0:
        method = "topology_cv_transition"
    elif best["same_family"] == 1:
        method = "topology_same_family"
    else:
        method = "topology_heading"

    return str(best["link_key"]), int(best["transition_support"]), method


def build_corridor(network, transitions, outgoing_total):
    outgoing_by_node = defaultdict(list)
    for row in network.itertuples():
        outgoing_by_node[row.from_node_id].append(row.link_key)

    if MANUAL_START_LINK is not None:
        start_link = str(MANUAL_START_LINK)
    else:
        eligible = [(key, value) for key, value in outgoing_total.items() if key in network.index]
        if not eligible:
            raise RuntimeError("No transition-supported start link found.")
        start_link = max(eligible, key=lambda item: item[1])[0]

    rows = []
    visited = set()
    current = start_link
    previous = None
    method = "start_highest_transition_support"

    for order in range(1, MAX_CORRIDOR_LINKS + 1):
        if current in visited:
            break
        visited.add(current)
        support = outgoing_total.get(current, 0) if order == 1 else transitions.get((previous, current), 0)
        rows.append(
            {
                "corridor_order": order,
                "link_key": current,
                "transition_support": support,
                "selection_method": method,
            }
        )

        next_link, _, next_method = choose_next_link(
            current, previous, network, outgoing_by_node, transitions
        )
        if next_link is None:
            break
        previous, current, method = current, next_link, next_method

    corridor = pd.DataFrame(rows).merge(network.reset_index(drop=True), on="link_key", how="left")
    corridor["link_length_ft"] = pd.to_numeric(corridor["length"], errors="coerce")
    corridor["distance_start_ft"] = corridor["link_length_ft"].fillna(0).cumsum().shift(fill_value=0)
    corridor["distance_end_ft"] = corridor["link_length_ft"].fillna(0).cumsum()

    connected = [np.nan]
    geometry_gap = [np.nan]
    for i in range(1, len(corridor)):
        previous_row = corridor.iloc[i - 1]
        current_row = corridor.iloc[i]
        connected.append(previous_row["to_node_id"] == current_row["from_node_id"])

        a = previous_row["geometry_obj"]
        b = current_row["geometry_obj"]
        if a is None or b is None:
            geometry_gap.append(np.nan)
        else:
            try:
                a_gdf = gpd.GeoSeries([a], crs="EPSG:4326").to_crs("EPSG:3083")
                b_gdf = gpd.GeoSeries([b], crs="EPSG:4326").to_crs("EPSG:3083")
                geometry_gap.append(a_gdf.iloc[0].boundary.geoms[-1].distance(b_gdf.iloc[0].boundary.geoms[0]))
            except Exception:
                geometry_gap.append(np.nan)

    corridor["topology_connected_from_previous"] = connected
    corridor["geometry_connection_gap_ft"] = geometry_gap
    return corridor


def collect_txdot_candidates(gids):
    gids = {normalize_gid(x) for x in gids if normalize_gid(x) is not None}
    parts = []
    usecols = ["LinkID", "GID", "Geometry", "Heading", "SpeedLimit", "RDBD_TYPE", "MAP_LBL", "Ramp_type"]

    for chunk in pd.read_csv(TXDOT_FILE, usecols=usecols, chunksize=500_000, dtype="string", low_memory=False):
        chunk["GID_norm"] = chunk["GID"].map(normalize_gid)
        keep = chunk[chunk["GID_norm"].isin(gids)].copy()
        if not keep.empty:
            parts.append(keep)

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    result["Heading"] = pd.to_numeric(result["Heading"], errors="coerce")
    result["SpeedLimit"] = pd.to_numeric(result["SpeedLimit"], errors="coerce")
    result["geometry_obj"] = result["Geometry"].map(parse_geometry)
    return result


def assign_txdot_speed(corridor):
    candidates = collect_txdot_candidates(corridor["Matched_GID"])
    output = []

    if candidates.empty:
        candidates = pd.DataFrame(columns=["GID_norm", "SpeedLimit", "geometry_obj", "Heading", "LinkID"])

    txdot = candidates[candidates.get("geometry_obj", pd.Series(dtype=object)).notna()].copy()
    if not txdot.empty:
        txdot_gdf = gpd.GeoDataFrame(txdot, geometry="geometry_obj", crs="EPSG:4326").to_crs("EPSG:3083")
    else:
        txdot_gdf = gpd.GeoDataFrame(txdot, geometry=[], crs="EPSG:3083")

    osm_valid = corridor[corridor["geometry_obj"].notna()].copy()
    osm_gdf = gpd.GeoDataFrame(osm_valid, geometry="geometry_obj", crs="EPSG:4326").to_crs("EPSG:3083")
    osm_geometry = dict(zip(osm_gdf["link_key"], osm_gdf.geometry))

    for row in corridor.itertuples(index=False):
        gid = normalize_gid(row.Matched_GID)
        subset = txdot_gdf[txdot_gdf.get("GID_norm", pd.Series(index=txdot_gdf.index, dtype=object)) == gid].copy()
        candidate_count = len(subset)
        valid_speed = subset[subset.get("SpeedLimit", pd.Series(index=subset.index, dtype=float)).gt(0)].copy()
        valid_speed_count = len(valid_speed)
        geometry = osm_geometry.get(row.link_key)

        if geometry is None or valid_speed.empty:
            output.append(
                {
                    "link_key": row.link_key,
                    "posted_speed_limit_mph": np.nan,
                    "txdot_link_id": pd.NA,
                    "txdot_gid": gid,
                    "txdot_match_distance_ft": np.nan,
                    "txdot_heading_difference_deg": np.nan,
                    "txdot_candidate_count": candidate_count,
                    "txdot_valid_speed_candidate_count": valid_speed_count,
                    "speed_match_status": "no_valid_speed_candidate" if valid_speed.empty else "missing_osm_geometry",
                }
            )
            continue

        midpoint = geometry.interpolate(0.5, normalized=True)
        valid_speed["match_distance_ft"] = valid_speed.geometry.distance(midpoint)
        valid_speed["heading_difference"] = valid_speed["Heading"].apply(lambda x: angular_difference(row.heading, x))

        eligible = valid_speed[
            valid_speed["match_distance_ft"].le(MAX_TXDOT_MATCH_DISTANCE_FT)
            & (
                valid_speed["heading_difference"].isna()
                | valid_speed["heading_difference"].le(MAX_TXDOT_HEADING_DIFFERENCE_DEG)
            )
        ].copy()

        if eligible.empty:
            best = valid_speed.sort_values("match_distance_ft").iloc[0]
            status = "outside_threshold"
        else:
            best = eligible.sort_values(["match_distance_ft", "heading_difference"]).iloc[0]
            status = "matched"

        output.append(
            {
                "link_key": row.link_key,
                "posted_speed_limit_mph": best["SpeedLimit"],
                "txdot_link_id": best["LinkID"],
                "txdot_gid": best["GID_norm"],
                "txdot_match_distance_ft": best["match_distance_ft"],
                "txdot_heading_difference_deg": best["heading_difference"],
                "txdot_candidate_count": candidate_count,
                "txdot_valid_speed_candidate_count": valid_speed_count,
                "speed_match_status": status,
            }
        )

    return corridor.merge(pd.DataFrame(output), on="link_key", how="left")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    transitions, outgoing_total = build_transition_support()
    network = load_network()
    corridor = build_corridor(network, transitions, outgoing_total)
    corridor = assign_txdot_speed(corridor)

    columns = [
        "corridor_order", "link_key", "from_node_id", "to_node_id", "Matched_GID", "txdot_gid",
        "heading", "transition_support", "selection_method", "link_length_ft", "distance_start_ft",
        "distance_end_ft", "topology_connected_from_previous", "geometry_connection_gap_ft",
        "posted_speed_limit_mph", "txdot_link_id", "txdot_match_distance_ft",
        "txdot_heading_difference_deg", "txdot_candidate_count", "txdot_valid_speed_candidate_count",
        "speed_match_status", "name", "facility_type", "RDBD_TYPE", "Functional Class", "geometry",
    ]
    corridor[[c for c in columns if c in corridor.columns]].to_csv(OUTPUT_FILE, index=False)
    print(f"Corridor links: {len(corridor):,}")
    print(f"Corridor length: {corridor['link_length_ft'].sum():,.1f} ft")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
