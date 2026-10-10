"""Review the SAME validated Dallas corridor from saved GPS and offset MOEs.

- Read-only on the production and offset Parquets.
- NULL/blank waypoint_count means an empty cell (0), for BOTH GPS and offset.
- Empty cells have unique_veh_count=0 even if journey_count is NULL.
- Observed cells use saved journey_count (distinct journeys in that cell).
- No GPS positioning is used to calculate offset MOEs: saved road geometries
  are transformed to WGS84 solely to make a QGIS-readable GeoJSON.
- Every run chooses a NEW output folder, avoiding locked Excel CSV files.

Requires pyarrow, shapely, pyproj from the existing MOE Python environment.
"""
import argparse
import csv
import json
import math
from collections import Counter
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
OFFSET_ROOT = HERE.parent
PROJECT_ROOT = OFFSET_ROOT.parent
GPS = PROJECT_ROOT / 'MOE for selected links' / 'Results' / 'Network_MOE'
OFFSET = OFFSET_ROOT / 'Results' / 'Network_MOE'

METRICS = [
    'waypoint_count', 'avg_speed_mph', 'posted_speed_limit_mph',
    'speed_reduction_pct', 'slow_movement_count', 'slow_movement_pct',
    'dsh_trip_count', 'DSH', 'speed_estimated_travel_time_s',
]
ROW_COLS = [
    'link_key', 'segment_index', 'parent_start_ft', 'parent_end_ft',
    'segment_length_ft', 'time_bin', 'journey_count', *METRICS,
]
CELL_FIELDS = [
    'corridor_order', 'link_key', 'segment_index', 'segment_section_ft',
    'segment_length_ft', 'start_time', 'end_time', 'observed',
    'waypoint_count', 'unique_veh_count', 'gps_waypoint_count',
    'offset_waypoint_count', 'avg_speed_mph', 'gps_avg_speed_mph',
    'speed_reduction_pct', 'posted_speed_limit_mph', 'slow_movement_count',
    'slow_movement_pct', 'dsh_trip_count', 'DSH',
    'speed_estimated_travel_time_s',
]
DIFF_FIELDS = [
    'corridor_order', 'link_key', 'segment_index', 'start_time',
    'metric', 'gps_value', 'offset_value',
]


def selected_rows(file, columns, keys):
    """Read just selected columns/keys in bounded-memory batches."""
    pf = pq.ParquetFile(file)
    missing = set(columns) - set(pf.schema_arrow.names)
    if missing:
        raise ValueError(f'{file.name} missing columns: {sorted(missing)}')
    value_set = pa.array(sorted(keys), type=pa.string())
    for batch in pf.iter_batches(batch_size=100_000, columns=columns):
        match = pc.is_in(pc.cast(batch.column('link_key'), pa.string()), value_set=value_set)
        filtered = batch.filter(match)
        if filtered.num_rows:
            yield from filtered.to_pylist()


def finite_number(value):
    if value is None:
        return None
    try:
        v = float(value)
    except (ValueError, TypeError):
        return None
    return v if math.isfinite(v) else None


def valid_count(value, field, key, label, null_is_zero=False):
    """Normalize absent waypoint counts to 0; never invent observed journeys."""
    v = finite_number(value)
    if v is None:
        if null_is_zero:
            return 0
        raise ValueError(f'{label} {field} missing in OBSERVED cell {key}')
    if v < 0 or not v.is_integer():
        raise ValueError(f'{label} invalid {field}={value!r} in cell {key}')
    return int(v)


def normalized_cell_counts(row, key, label):
    wp = valid_count(row.get('waypoint_count'), 'waypoint_count', key, label, null_is_zero=True)
    if wp == 0:
        return 0, 0
    journeys = valid_count(row.get('journey_count'), 'journey_count', key, label)
    if journeys > wp:
        raise ValueError(f'{label} journey_count {journeys} exceeds {wp} waypoints in cell {key}')
    return wp, journeys


def label_ft(value):
    v = finite_number(value)
    if v is None:
        raise ValueError(f'Invalid section offset: {value!r}')
    return f'{v:.4f}'.rstrip('0').rstrip('.')


def eq_moe(a, b, tol=0.001):
    if a is None or b is None:
        return a is None and b is None
    return abs(a-b) <= tol


def read_corridor_links(file):
    if not file.is_file():
        raise FileNotFoundError('Original production corridor inventory missing: '+str(file))
    with file.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'link_key', 'corridor_order'}.issubset(reader.fieldnames or []):
            raise ValueError('Corridor link inventory lacks link_key or corridor_order')
        rows = list(reader)
    if not rows:
        raise ValueError('Original corridor file has no links')
    rows.sort(key=lambda r:int(r['corridor_order']))
    keys = [str(r['link_key']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Repeated link_key in corridor link inventory')
    return rows, keys


def reserve_unique_output(base):
    """Reserve a fresh directory; never overwrite or collide with an open CSV."""
    base.parent.mkdir(parents=True, exist_ok=True)
    for number in range(1, 10000):
        folder = base.parent / f'{base.name}_{number:03d}'
        try:
            folder.mkdir(exist_ok=False)
            return folder
        except FileExistsError:
            continue
    raise RuntimeError('No new available output folder name')


def extract(production, offset, corridor_links, output_base):
    gps_file = production / 'segment_moe_500ft_15min_gps.parquet'
    offset_file = offset / 'segment_moe_500ft_15min_offset.parquet'
    geometry_file = production / 'moe_segments_500ft.parquet'
    for file in (gps_file, offset_file, geometry_file):
        if not file.is_file():
            raise FileNotFoundError('Required completed result missing: '+str(file))
    links, path = read_corridor_links(corridor_links)
    ids = set(path)
    order = {key:i+1 for i, key in enumerate(path)}
    print('Original production corridor links:', len(path), flush=True)

    geoms = {}
    for row in selected_rows(geometry_file, ['link_key', 'segment_index', 'geometry_wkb'], ids):
        key = (str(row['link_key']), int(row['segment_index']))
        if key in geoms:
            raise ValueError('Repeated geometry segment: '+str(key))
        if row['geometry_wkb'] is None:
            raise ValueError('Missing saved geometry for segment: '+str(key))
        geoms[key] = row['geometry_wkb']
    print('Corridor segments with geometry:', len(geoms), flush=True)
    if not geoms:
        raise ValueError('No saved corridor section geometry')

    method_cells = {}
    for label, file in [('GPS', gps_file), ('offset', offset_file)]:
        cells = {}
        for row in selected_rows(file, ROW_COLS, ids):
            key = (str(row['link_key']), int(row['segment_index']), row['time_bin'])
            if key in cells:
                raise ValueError(f'Repeated {label} MOE cell: {key}')
            cells[key] = row
        method_cells[label] = cells
        print(label, 'cells:', len(cells), flush=True)

    gps_cells = method_cells['GPS']
    offset_cells = method_cells['offset']
    if gps_cells.keys() != offset_cells.keys():
        raise ValueError(f'GPS/offset grids differ: GPS-only {len(gps_cells.keys()-offset_cells.keys())}, '
                         f'offset-only {len(offset_cells.keys()-gps_cells.keys())}')
    if {(link, segment) for link, segment, _ in gps_cells} != set(geoms):
        raise ValueError('Saved geometry inventory does not match the corridor MOE segments')

    ordered = sorted(gps_cells, key=lambda x:(order[x[0]], x[1], x[2]))
    transform = Transformer.from_crs('EPSG:3083','EPSG:4326',always_xy=True)
    comparison_counts = Counter()
    sums = {'GPS': 0, 'offset': 0}
    observed = {'GPS': 0, 'offset': 0}
    journey_sum = 0
    speed_available = 0
    slow_cells = 0
    posted_unavailable = 0
    full_rows = []
    diff_rows = []
    features = []

    # Validate and build ALL output records before writing anything.
    for key in ordered:
        link, segment, stamp = key
        gps = gps_cells[key]
        off = offset_cells[key]
        gps_wp, gps_journeys = normalized_cell_counts(gps, key, 'GPS')
        off_wp, off_journeys = normalized_cell_counts(off, key, 'offset')
        sums['GPS'] += gps_wp
        sums['offset'] += off_wp
        observed['GPS'] += int(gps_wp > 0)
        observed['offset'] += int(off_wp > 0)
        journey_sum += off_journeys
        start_time = stamp.strftime('%Y-%m-%d %H:%M:%S')
        end_time = (stamp + timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')
        for metric in METRICS:
            gps_value = gps_wp if metric == 'waypoint_count' else finite_number(gps.get(metric))
            off_value = off_wp if metric == 'waypoint_count' else finite_number(off.get(metric))
            if not eq_moe(gps_value, off_value):
                diff_rows.append(dict(corridor_order=order[link],link_key=link,
                                      segment_index=segment,start_time=start_time,
                                      metric=metric,gps_value=gps_value,offset_value=off_value))
                comparison_counts[metric] += 1
        if gps_journeys != off_journeys:
            diff_rows.append(dict(corridor_order=order[link],link_key=link,
                                  segment_index=segment,start_time=start_time,
                                  metric='journey_count',gps_value=gps_journeys,
                                  offset_value=off_journeys))
            comparison_counts['journey_count'] += 1
        segment_start = finite_number(off['parent_start_ft'])
        segment_end = finite_number(off['parent_end_ft'])
        segment_length = finite_number(off['segment_length_ft'])
        if (segment_start is None or segment_end is None or segment_end <= segment_start
            or segment_length is None or segment_length <= 0):
            raise ValueError(f'Invalid saved section bounds or length in {key}')
        gps_start = finite_number(gps['parent_start_ft'])
        gps_end = finite_number(gps['parent_end_ft'])
        if (gps_start is None or gps_end is None
            or abs(gps_start-segment_start)>0.01 or abs(gps_end-segment_end)>0.01):
            raise ValueError(f'GPS and offset section boundaries mismatch in {key}')
        avg_speed = finite_number(off['avg_speed_mph'])
        posted = finite_number(off['posted_speed_limit_mph'])
        reduction = (round(max(0.,100*(posted-avg_speed)/posted),3)
                     if avg_speed is not None and posted is not None and posted>0 else None)
        speed_available += int(off_wp>0 and avg_speed is not None)
        slow_cells += int(off_wp>0 and (finite_number(off['slow_movement_count']) or 0)>0)
        posted_unavailable += int(off_wp>0 and (posted is None or posted<=0))
        row = dict(
            corridor_order=order[link],link_key=link,segment_index=segment,
            segment_section_ft=f'{label_ft(segment_start)}-{label_ft(segment_end)}',
            segment_length_ft=segment_length,start_time=start_time,end_time=end_time,
            observed=off_wp>0,waypoint_count=off_wp,unique_veh_count=off_journeys,
            gps_waypoint_count=gps_wp,offset_waypoint_count=off_wp,
            avg_speed_mph=avg_speed,gps_avg_speed_mph=finite_number(gps['avg_speed_mph']),
            speed_reduction_pct=reduction,posted_speed_limit_mph=posted,
            slow_movement_count=int(finite_number(off['slow_movement_count']) or 0),
            slow_movement_pct=finite_number(off['slow_movement_pct']),
            dsh_trip_count=int(finite_number(off['dsh_trip_count']) or 0),
            DSH=finite_number(off['DSH']),
            speed_estimated_travel_time_s=finite_number(off['speed_estimated_travel_time_s']),
        )
        road_line = shapely.from_wkb(geoms[(link,segment)])
        wgs84_line = shapely.transform(road_line, transform.transform, interleaved=False)
        feature = dict(type='Feature',geometry=json.loads(shapely.to_geojson(wgs84_line)),
                       properties=row)
        full_rows.append(row)
        features.append(feature)

    # A fresh output path avoids Windows permission errors from CSVs open in Excel.
    folder = reserve_unique_output(output_base)
    with (folder/'corridor_links.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(links[0]));writer.writeheader();writer.writerows(links)
    with (folder/'corridor_cells.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=CELL_FIELDS);writer.writeheader();writer.writerows(full_rows)
    with (folder/'corridor_differences.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=DIFF_FIELDS);writer.writeheader();writer.writerows(diff_rows)
    (folder/'corridor_cells.geojson').write_text(
        json.dumps(dict(type='FeatureCollection',features=features),allow_nan=False),encoding='utf-8')
    summary = dict(
        corridor_source=str(corridor_links),gps_source=str(gps_file),offset_source=str(offset_file),
        links=len(path),distinct_500ft_sections=len(geoms),segment_time_cells=len(ordered),
        gps_observed_cells=observed['GPS'],offset_observed_cells=observed['offset'],
        gps_summed_waypoints=sums['GPS'],offset_summed_waypoints=sums['offset'],
        cells_with_valid_offset_speed=speed_available,
        cells_with_at_least_one_below_5mph_reading=slow_cells,
        observed_cells_without_usable_posted_speed=posted_unavailable,
        gps_offset_field_differences_over_0_001=dict(comparison_counts),
        total_field_differences=len(diff_rows),summed_unique_veh_count_across_cells=journey_sum,
        gps_offset_journey_count_differing_cells=comparison_counts['journey_count'],
        unique_veh_count_definition='Distinct journey_id in each link x section x 15-minute cell; a new journey on re-entry counts again',
        note='No MOE recalculation. Provisional slow movement and DSH copied unchanged; speed reduction nonnegative in output.',
    )
    (folder/'corridor_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'Validation complete: {len(ordered):,} cells, {sums["offset"]:,} offset waypoints, '
          f'{journey_sum:,} summed cell-level unique journeys',flush=True)
    print('Five files saved in:',folder,flush=True)
    return folder, summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production',type=Path,default=GPS)
    parser.add_argument('--offset',type=Path,default=OFFSET)
    parser.add_argument('--corridor-links',type=Path,
                        default=GPS/'Production_Corridor_Review'/'production_corridor_links.csv')
    parser.add_argument('--output-base',type=Path,
                        default=OFFSET/'Offset_Corridor_Review_CORRECTED')
    args=parser.parse_args()
    extract(args.production,args.offset,args.corridor_links,args.output_base)


if __name__=='__main__':
    main()
