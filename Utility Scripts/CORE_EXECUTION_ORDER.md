# Official pipeline — top to bottom

**Entrypoint:** `python run_offset_moe.py`

| Order | Source | Role | Important functions |
|---|---|---|---|
| 0 | `paths.py` | Configuration, input/output paths, validated network inventory | `check_inputs`, `source_summary` |
| 1 | `parallel_utils.py` (called from Stage 03) | Stream CV input and shard full journeys into 32 Parquet files | `build_journey_partitions`, `signature`, `resolve_workers` |
| 2 | `03_calculate_segment_moes.py` | Assign matched waypoint positions with link-local `x_ft` and lateral `y_ft` validity; form 500-ft sections and local 15-min bins | `assign_segments`, `prepare_sequence`, `calculate_speed_components` |
| 3 | `03_calculate_segment_moes.py` | Aggregate speed, slow movement, DSH, and intermediate travel metrics; join immutable 500-ft network inventory | `process_partition`, `reduce_partials`, `write_final_output` |
| 4 | `10_export_moe_csvs.py` | Official export-stage CLI called by runner | `main` |
| 5 | `export_four_production_moe_csvs.py` | Shared memory-bounded export engine, exact compact column schemas, count checks, atomic publish | `export` |

The official runner executes **calculation then export**, and does not run historical GPS comparisons. Historical GPS inputs are optional for the diagnostic scripts, never required to generate the offset MOEs except that the original saved network inventories may still be stored in the historical folder until copied into `Input/Network`.

## Position and time rules preserved unchanged

- `x_ft` must be finite and within approximately `[-1, parent_length_ft + 1]`, is clipped to parent bounds, then assigned via `floor(x_ft / 500) + 1`.
- `y_ft` must be finite with `abs(y_ft) <= 100.01` ft; no latitude/longitude is read from matched CV.
- `local_time` is floored to 15-minute bins; source capture time orders waypoints.
- One distinct `journey_id` per cell contributes one to `unique_veh_count`; re-entry as a new journey is another passage.
- The existing slow-movement formula and DSH are provisional pending a separate methodology decision. **No values are redefined here.**
- Intermediate MOE Parquet retains old interpolation fields for audits; final CSVs intentionally do not export them.

## Manual stages (if needed)

```powershell
& $py "$folder\core\03_calculate_segment_moes.py" --workers 12
& $py "$folder\core\10_export_moe_csvs.py" --method offset
```

Or run `python run_offset_moe.py --export-only` to only regenerate four CSVs from the saved completed Parquet. The export stage chooses a new numbered folder when the earlier output already exists.
