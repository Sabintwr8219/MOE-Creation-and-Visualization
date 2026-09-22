# Network-wide 200-ft MOE workflow

This workflow calculates five-minute measures of effectiveness for every OSM
link represented in the connected-vehicle input. It retrieves those parent
links from the complete OSM network, attaches posted speed limits from TxDOT,
divides each parent into pieces of at most 200 ft, assigns CV observations to
the pieces, calculates MOEs, builds connected corridors, and creates time-space
figures.

## Processing stages

| Script | Role |
| --- | --- |
| `paths.py` | Defines inputs, outputs, thresholds, and corridor settings. |
| `01_prepare_parent_links.py` | Finds all observed link keys, retrieves their OSM geometry, and matches TxDOT speed limits. |
| `02_create_moe_segments.py` | Creates ordered 200-ft pieces and final remainders with unique IDs and connected nodes. |
| `parallel_utils.py` | Creates reusable journey-safe Parquet partitions for parallel stages. |
| `03_calculate_segment_moes.py` | Assigns CV observations and calculates five-minute segment MOEs with multiple processes. |
| `04_build_corridors.py` | Counts observed transitions in parallel and traces connected corridors from node topology. |
| `05_visualize_corridors.py` | Reads only selected corridor rows and creates corridor figure sets in parallel. |
| `check_pipeline.py` | Checks paths, schemas, segmentation, nullable types, and journey-partition equivalence. |
| `run_pipeline.py` | Runs checks and all stages in order while saving a log. |

## Measures of effectiveness

The final `segment_moe_5min.parquet` contains average speed, speed reduction
percentage, slow-movement count, slow-movement percentage, degree of speed
harmonization, average segment travel time, and the observation or traversal
counts supporting each estimate. Missing estimates remain null. A null value
does not mean zero.

Speed reduction is calculated from the TxDOT posted speed limit:

`100 × (posted speed − average speed) / posted speed`

Segment travel times are calculated for complete parent-link traversals.
Observed positions and entry/exit times are interpolated at each 200-ft
boundary, and `complete_traversal_count` records the supporting sample size.

## Run

Activate the project environment and run the checks:

```powershell
& ".\.venv\Scripts\python.exe" ".\MOE for selected links\run_pipeline.py" --check-only
```

Run the full workflow unattended with the default 12-worker limit:

```powershell
& ".\.venv\Scripts\python.exe" ".\MOE for selected links\run_pipeline.py"
```

Set a different worker count when benchmarking shows it is faster:

```powershell
& ".\.venv\Scripts\python.exe" `
    ".\MOE for selected links\run_pipeline.py" `
    --workers 12
```

When Stages 1 and 2 already exist, resume directly from parallel Stage 3:

```powershell
& ".\.venv\Scripts\python.exe" `
    ".\MOE for selected links\run_pipeline.py" `
    --start-stage 3 `
    --skip-checks `
    --workers 12
```

Stages 3 and 4 reuse 32 Parquet partitions keyed by `journey_id`. This keeps
every journey intact while independent partitions run concurrently. The
partition manifest automatically rebuilds them when the CV source changes.
Stage 3 writes the complete segment-time result in bounded batches instead of
holding the full output grid in memory.

The existing TxDOT reference Parquet is reused automatically. Add
`--force-txdot` only when the TxDOT source file changes.

## Corridor settings

Corridors affect visualization only. Network-wide segment MOEs are calculated
before corridor selection. Edit the settings near the bottom of `paths.py` to
provide explicit start links, change the number of corridors, or change their
maximum length.
