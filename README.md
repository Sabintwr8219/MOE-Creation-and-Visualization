# StreetLight MOE Calculation and Visualization

Python scripts for calculating traffic Measures of Effectiveness (MOEs)
from processed StreetLight connected-vehicle observations and visualizing
their variation across road links, time periods, and connected corridors.

## Process

The workflow uses observations already assigned to network links.

Two analysis paths are provided:

1. Selected-link analysis: calculate five-minute and hourly MOEs for
   a supplied set of links, then create comparison heatmaps.
2. Corridor analysis: build a connected sequence of network links,
   attach TxDOT posted-speed information, calculate five-minute MOEs,
   and create corridor time-space visualizations.

Additional scripts inspect corridor results and export spatial layers
for review in QGIS.

## Scripts

| Script | Role |
|---|---|
| `02_calculate_moes.py` | Calculate five-minute MOEs for links listed in `selected_links.csv`. |
| `03_calculate_hourly_moes.py` | Calculate link MOEs across the configured hourly observation file. |
| `04_visualize_moes.py` | Create MOE heatmaps for the links with the most observations. |
| `05_build_connected_corridor.py` | Build a connected corridor using network topology, observed vehicle transitions, link-family continuity, and heading; attach TxDOT posted-speed information. |
| `06_calculate_corridor_moes.py` | Calculate five-minute MOEs for corridor links, including average speed and speed reduction percentage. |
| `07_visualize_corridor_timespace.py` | Plot corridor MOEs against time and cumulative corridor distance. |
| `08_investigate_corridor.py` | Summarize corridor attributes and MOEs, identify links with high travel time or speed reduction, and generate map-reference points. |
| `09_export_corridor_for_qgis.py` | Export corridor links and associated TxDOT segments to a GeoPackage. |

## Measures of Effectiveness

| Measure | Calculation |
|---|---|
| Waypoint count | Number of valid speed observations. |
| Journey count | Number of distinct original journey IDs represented. |
| Slow movement count | Number of valid observations with speed below 5 mph. |
| Slow movement percentage | Sum of speeds below 5 mph divided by the sum of all valid speeds, multiplied by 100. |
| Degree of Speed Harmonization (DSH) | Sum of consecutive absolute speed changes within each continuous run and aggregation period, divided by its number of speed observations; averaged across those runs. |
| Average travel time | Average next-link entry time minus current-link entry time for eligible complete link traversals. |
| Average speed | Arithmetic mean of valid waypoint speeds; included in corridor results. |
| Speed reduction percentage | `(posted_speed_limit_mph - avg_speed_mph) / posted_speed_limit_mph × 100`; included in corridor results. |

Slow movement percentage is speed-weighted; it is not the percentage
of observations below the speed threshold.

Speed reduction uses TxDOT posted-speed information. Negative values
indicate an average observed speed above the posted limit. The CSV
retains these values, while the speed-reduction plot displays a
0–100% range.

## Journey and Traversal Processing

- A time gap greater than 120 seconds starts a new trip segment.
- Unmatched observations form sequence boundaries.
- Link changes define separate continuous runs.
- Travel-time calculations require preceding and following matched
  links within the same trip segment.
- Positive travel times are retained without an upper outlier cutoff.
- CSV observations are read in chunks of 1,000,000 rows.
- Chunk processing expects each journey's records to be contiguous
  in the input file.

## Inputs

The scripts use:

- Processed StreetLight observations containing `journey_id`,
  `capture_time`, `local_time`, `speed_mph`, and `link_key`.
- `selected_links.csv` for selected-link analysis.
- An OSM short-link network containing link keys, node connectivity,
  geometry, headings, lengths, and matched TxDOT identifiers.
- A TxDOT link dataset containing geometry, headings, and `SpeedLimit`.

Input datasets are supplied separately. Configure the paths at the
top of each script before running it.

The hourly calculation script expects an input file covering one hour.

## Execution Order

### Selected-link analysis

1. Supply `selected_links.csv`.
2. Run `02_calculate_moes.py`.
3. Run `03_calculate_hourly_moes.py`.
4. Run `04_visualize_moes.py`.

### Corridor analysis

1. Run `05_build_connected_corridor.py`.
2. Run `06_calculate_corridor_moes.py`.
3. Run `07_visualize_corridor_timespace.py`.

For additional inspection, run `08_investigate_corridor.py`.
For spatial export, run `09_export_corridor_for_qgis.py`.

## Outputs

- `link_moe_5min.csv`: selected-link five-minute MOEs.
- `link_moe_hourly.csv`: selected-link hourly MOEs.
- `corridor_links.csv`: ordered corridor links and supporting attributes.
- `corridor_moe_5min.csv`: corridor link MOEs by five-minute period.
- `MOE Heatmaps/`: selected-link comparison figures.
- `Corridor Time-Space Visualizations/`: corridor figures.
- `corridor_diagnostic.csv`: corridor inspection summary.
- `corridor_map_points.csv`: geographic reference points and map links.
- `corridor_qgis.gpkg`: corridor and associated TxDOT spatial layers.

## Python Dependencies

- pandas
- numpy
- matplotlib
- geopandas
- shapely
- pyproj