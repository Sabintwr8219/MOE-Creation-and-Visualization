# StreetLight CV — official offset-based MOE pipeline

**One production code directory + one optional `Utility Scripts` directory.** The GPS calculation pipeline is shelved. This folder is code only; keep large local data out of GitHub.

## Files (top-to-bottom production order)

```text
MOE using offsets/
├── run_offset_moe.py                   # MAIN ENTRYPOINT: calculate -> export
├── paths.py                            # 1. Input/output paths, constants, workers
├── progress.py                         # Progress and resource logging
├── parallel_utils.py                   # 2. Partition matched CV into journeys
├── 03_calculate_segment_moes.py        # 3. Offset assignment + MOEs to Parquet
├── 10_export_moe_csvs.py               # 4. Pipeline export-stage wrapper
├── export_four_production_moe_csvs.py  # 5. Four compact CSVs, shared engine
├── Utility Scripts/                    # Optional tools, historical diagnostics, tests, documentation
│   ├── check_input_paths.py
│   ├── export_saved_moe.py
│   ├── 11_compare_moe_methods.py
│   ├── 12_export_offset_corridor_review.py
│   ├── test_compact_export.py
│   ├── test_runner_sequence.py
│   ├── CORE_EXECUTION_ORDER.md
│   ├── CSV_SCHEMA.md
│   └── GITHUB_UPDATE.md
├── requirements.txt
└── .gitignore
```

## Production methodology

- Offsets: use matched `x_ft` to determine longitudinal position, with `y_ft` validity filtering; **no CV GPS latitude/longitude**.
- Grid: 500-ft parent-link sections by 15-minute `local_time` bins.
- `unique_veh_count`: number of distinct `journey_id` values **per cell**; repeat visits as new journeys count again.
- Results: four observed-cell-only CSVs: `speed_reduction.csv`, `slow_movement.csv`, `dsh.csv`, `travel_time.csv`.
- Speed reduction below zero is clipped to zero in the final CSV; no posted limit -> blank.
- Slow-movement and DSH formulas are deliberately unchanged and provisional pending professor discussion.

## Run locally in PowerShell

```powershell
$root = 'D:\Sabin\Streetlight-Data-Processing'
$folder = Join-Path $root 'MOE using offsets'
$py = Join-Path $root '.venv\Scripts\python.exe'
& $py "$folder\run_offset_moe.py" --check-inputs
& $py -m unittest discover -s "$folder\Utility Scripts" -p 'test_*.py' -v
```

**Normal production (recomputes all MOEs):**

```powershell
& $py "$folder\run_offset_moe.py"
```

**Export only using the finished offset Parquet (no CV recalculation):**

```powershell
& $py "$folder\run_offset_moe.py" --export-only
```

Files are written under `Results/Network_MOE/Compact_CSV_500ft_15min_offset`, or a numbered sibling if that folder already exists.

## Required local inputs

- Matched CV input defaults to `../Output/Final/9_20_2025/9_20_2025_hr=19.csv`; change with environment variable `OFFSET_MOE_CV_FILE`.
- `parent_links.parquet` and `moe_segments_500ft.parquet` prefer `Input/Network` under this folder; if absent, the configuration references the verified historical network **inventories** in `MOE for selected links/Results/Network_MOE`. This does **not** run GPS MOEs.
- `Results`, `Input`, all `.csv` / `.parquet` raw or output files are excluded by `.gitignore`.

The former `core/`, `tools/`, `diagnostics/`, `tests/`, and `docs/` structure is no longer used. The Python files in `Utility Scripts` are not part of the normal production run. See `Utility Scripts/CORE_EXECUTION_ORDER.md` for the annotated workflow.
