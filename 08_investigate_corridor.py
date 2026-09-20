from pathlib import Path
import json

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import shape


PROJECT_ROOT = Path(r"D:\Sabin\Streetlight-Data-Processing")
RESULTS_DIR = PROJECT_ROOT / "MOE for selected links" / "Results"
CORRIDOR_FILE = RESULTS_DIR / "corridor_links.csv"
MOE_FILE = RESULTS_DIR / "corridor_moe_5min.csv"
NETWORK_FILE = PROJECT_ROOT / "Initial Input Files" / "OSM_Short_Level_CSV" / "Complete_OSM_Short_Level_Link_List.csv"

DIAGNOSTIC_FILE = RESULTS_DIR / "corridor_diagnostic.csv"
MAP_POINTS_FILE = RESULTS_DIR / "corridor_map_points.csv"


def parse_geometry(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    try:
        return shape(json.loads(text)) if text.startswith("{") else wkt.loads(text)
    except Exception:
        return None


def geometry_points(geometry):
    if geometry is None or geometry.is_empty:
        return {key: np.nan for key in ["start_lon", "start_lat", "mid_lon", "mid_lat", "end_lon", "end_lat"]}
    start = geometry.coords[0]
    end = geometry.coords[-1]
    midpoint = geometry.interpolate(0.5, normalized=True)
    return {
        "start_lon": start[0],
        "start_lat": start[1],
        "mid_lon": midpoint.x,
        "mid_lat": midpoint.y,
        "end_lon": end[0],
        "end_lat": end[1],
    }


def google_maps_url(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return ""
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"


def fill_network_attributes(corridor):
    target_links = set(corridor["link_key"].dropna().astype(str))
    needed = ["link_key", "geometry", "name", "Matched_GID", "RDBD_TYPE", "Functional Class", "heading", "length"]
    parts = []

    for chunk in pd.read_csv(NETWORK_FILE, usecols=needed, chunksize=500_000, dtype="string", low_memory=False):
        keep = chunk[chunk["link_key"].isin(target_links)].copy()
        if not keep.empty:
            parts.append(keep)

    if not parts:
        return corridor

    network = pd.concat(parts, ignore_index=True).drop_duplicates("link_key")
    return corridor.merge(network, on="link_key", how="left", suffixes=("", "_network"))


def build_moe_summary():
    moe = pd.read_csv(MOE_FILE, dtype={"link_key": "string"})
    numeric = [
        "avg_speed_mph", "speed_reduction_pct", "avg_travel_time_s", "waypoint_count",
        "journey_count", "slow_movement_count", "slow_movement_pct", "DSH", "complete_traversal_count",
    ]
    for column in numeric:
        if column in moe.columns:
            moe[column] = pd.to_numeric(moe[column], errors="coerce")

    return (
        moe.groupby("link_key", observed=True)
        .agg(
            mean_avg_speed_mph=("avg_speed_mph", "mean"),
            mean_speed_reduction_pct=("speed_reduction_pct", "mean"),
            max_speed_reduction_pct=("speed_reduction_pct", "max"),
            mean_avg_travel_time_s=("avg_travel_time_s", "mean"),
            total_waypoints=("waypoint_count", "sum"),
            mean_slow_movement_pct=("slow_movement_pct", "mean"),
            mean_DSH=("DSH", "mean"),
            total_complete_traversals=("complete_traversal_count", "sum"),
        )
        .reset_index()
    )


def main():
    corridor = pd.read_csv(CORRIDOR_FILE, dtype="string")
    corridor = fill_network_attributes(corridor)

    for column in ["corridor_order", "distance_start_ft", "distance_end_ft", "link_length_ft", "posted_speed_limit_mph"]:
        if column in corridor.columns:
            corridor[column] = pd.to_numeric(corridor[column], errors="coerce")

    geometry_column = "geometry" if "geometry" in corridor.columns else "geometry_network"
    corridor["geometry_obj"] = corridor[geometry_column].map(parse_geometry)
    points = corridor["geometry_obj"].apply(geometry_points).apply(pd.Series)
    corridor = pd.concat([corridor.reset_index(drop=True), points.reset_index(drop=True)], axis=1)

    for prefix in ["start", "mid", "end"]:
        corridor[f"{prefix}_google_maps_url"] = corridor.apply(
            lambda row, p=prefix: google_maps_url(row[f"{p}_lat"], row[f"{p}_lon"]), axis=1
        )

    diagnostic = corridor.merge(build_moe_summary(), on="link_key", how="left").sort_values("corridor_order")
    diagnostic.drop(columns=["geometry_obj"], errors="ignore").to_csv(DIAGNOSTIC_FILE, index=False)

    map_columns = [
        "corridor_order", "link_key", "distance_start_ft", "distance_end_ft",
        "start_lat", "start_lon", "mid_lat", "mid_lon", "end_lat", "end_lon",
        "start_google_maps_url", "mid_google_maps_url", "end_google_maps_url",
    ]
    diagnostic[[c for c in map_columns if c in diagnostic.columns]].to_csv(MAP_POINTS_FILE, index=False)

    print("\nCORRIDOR SEQUENCE")
    print("=" * 120)
    show = [
        "corridor_order", "link_key", "distance_start_ft", "distance_end_ft", "name", "RDBD_TYPE",
        "posted_speed_limit_mph", "mid_lat", "mid_lon", "mean_avg_speed_mph",
        "mean_speed_reduction_pct", "max_speed_reduction_pct", "mean_avg_travel_time_s", "total_waypoints",
    ]
    show = [c for c in show if c in diagnostic.columns]
    print(diagnostic[show].to_string(index=False))

    print("\nWORST SPEED REDUCTION")
    print("=" * 120)
    print(diagnostic.sort_values("mean_speed_reduction_pct", ascending=False).head(10)[show].to_string(index=False))

    print("\nWORST TRAVEL TIME")
    print("=" * 120)
    print(diagnostic.sort_values("mean_avg_travel_time_s", ascending=False).head(10)[show].to_string(index=False))

    print(f"\nSaved: {DIAGNOSTIC_FILE}")
    print(f"Saved: {MAP_POINTS_FILE}")


if __name__ == "__main__":
    main()
