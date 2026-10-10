# Push updates to the EXISTING GitHub repository

There is no need for `git init`, a new remote, or a new repository.

## Repository layout

The active `MOE using offsets/` folder has its seven `.py` production files together and one `Utility Scripts/` folder. `README.md`, `.gitignore`, and `requirements.txt` remain at the root.

## Verify (no 54-million-row calculation)

```powershell
$root = 'D:\Sabin\Streetlight-Data-Processing'
$folder = Join-Path $root 'MOE using offsets'
$py = Join-Path $root '.venv\Scripts\python.exe'
& $py -m compileall -q $folder
& $py -m unittest discover -s "$folder\Utility Scripts" -p 'test_*.py' -v
& $py "$folder\run_offset_moe.py" --check-inputs
```

## Push from your EXISTING local Git clone

Copy the `MOE using offsets` code folder to the corresponding location in your existing clone. From the Git root:

```powershell
git remote -v
git status --short
git add -- 'MOE using offsets/run_offset_moe.py' 'MOE using offsets/paths.py' 'MOE using offsets/progress.py' 'MOE using offsets/parallel_utils.py' 'MOE using offsets/03_calculate_segment_moes.py' 'MOE using offsets/10_export_moe_csvs.py' 'MOE using offsets/export_four_production_moe_csvs.py' 'MOE using offsets/Utility Scripts' 'MOE using offsets/README.md' 'MOE using offsets/.gitignore' 'MOE using offsets/requirements.txt'
git diff --cached --stat
```

If the prior folder layout (`core/`, `tools/`, `diagnostics/`, `tests/`, `docs/`) is tracked, remove those **repository-only paths** from tracking after confirming their replacement files are staged:

```powershell
# Only if those old paths are tracked, and only from the existing repo
# git rm -r -- 'MOE using offsets/core' 'MOE using offsets/tools' 'MOE using offsets/diagnostics' 'MOE using offsets/tests' 'MOE using offsets/docs'
```

Then `git commit -m 'Flatten official offset MOE pipeline'` and `git push origin HEAD`. Verify staged changes first; never stage raw local input or output files.
