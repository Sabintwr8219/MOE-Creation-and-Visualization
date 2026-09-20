from pathlib import Path
import json

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import shape


PROJECT_ROOT = Path(r"D:\Sabin\Streetlight-Data-Processing")
RESULTS_DIR = PROJECT_ROOT / "MOE for selected links" / "Results"
CORRIDOR_FILE = RESULTS_DIR / "corridor_links.csv"
NETWORK_FILE = PROJECT_ROOT / "Initial Input Files" / "OSM_Short_Level_CSV" / "Complete_OSM_Short_Level_Link_List.csv"
TXDOT_FILE = PROJECT_ROOT / "Initial Input Files" / "TxDOT" / "link_list_with_speed_NOL.csv"
OUTPUT_GPKG = RESULTS_DIR / "corridor_qgis.gpkg"


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


def collect_network_links(link_keys):
    usecols = [
        "link_key", "geometry", "name", "from_node_id", "to_node_id", "heading", "length",
        "Matched_GID", "RDBD_TYPE", "Functional Class", "facility_type",
    ]
    parts = []
    for chunk in pd.read_csv(NETWORK_FILE, usecols=usecols, chunksize=500_000, dtype="string", low_memory=False):
        keep = chunk[chunk["link_key"].isin(link_keys)].copy()
        if not keep.empty:
            parts.append(keep)
    return pd.concat(parts, ignore_index=True).drop_duplicates("link_key") if parts else pd.DataFrame()


def collect_txdot_segments(gids):
    usecols = [
        "LinkID", "GID", "Geometry", "RDBD_TYPE", "DES_DRCT", "MAP_LBL", "Heading",
        "SpeedLimit", "Ramp_type", "length", "Functional Class",
    ]
    parts = []
    for chunk in pd.read_csv(TXDOT_FILE, usecols=usecols, chunksize=500_000, dtype="string", low_memory=False):
        chunk["GID_norm"] = chunk["GID"].map(normalize_gid)
        keep = chunk[chunk["GID_norm"].isin(gids)].copy()
        if not keep.empty:
            parts.append(keep)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    corridor = pd.read_csv(CORRIDOR_FILE, dtype="string")
    link_keys = set(corridor["link_key"].dropna().astype(str))
    network = collect_network_links(link_keys)
    if network.empty:
        raise RuntimeError("No corridor links found in the short-link network.")

    network["geometry_obj"] = network["geometry"].map(parse_geometry)
    network = network[network["geometry_obj"].notna()].copy()
    attributes = corridor.drop(columns=["geometry"], errors="ignore")
    network = network.merge(attributes, on="link_key", how="left", suffixes=("", "_corridor"))

    corridor_gdf = gpd.GeoDataFrame(network, geometry="geometry_obj", crs="EPSG:4326").sort_values("corridor_order")
    gids = {normalize_gid(x) for x in corridor["Matched_GID"] if normalize_gid(x) is not None}
    txdot = collect_txdot_segments(gids)

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()

    corridor_gdf.to_file(OUTPUT_GPKG, layer="corridor_links", driver="GPKG")

    if not txdot.empty:
        txdot["geometry_obj"] = txdot["Geometry"].map(parse_geometry)
        txdot = txdot[txdot["geometry_obj"].notna()].copy()
        txdot_gdf = gpd.GeoDataFrame(txdot, geometry="geometry_obj", crs="EPSG:4326")
        txdot_gdf.to_file(OUTPUT_GPKG, layer="txdot_segments", driver="GPKG")

    print(f"Saved: {OUTPUT_GPKG}")
    print(f"corridor_links: {len(corridor_gdf):,}")
    print(f"txdot_segments: {len(txdot):,}")


if __name__ == "__main__":
    main()
