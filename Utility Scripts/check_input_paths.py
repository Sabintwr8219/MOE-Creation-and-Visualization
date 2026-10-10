"""Read-only check of configured official offset input paths."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

if __name__ == "__main__":
    for name, path in paths.source_summary().items():
        print(f"{name}: {path} | {'FOUND' if path.exists() else 'MISSING'}")
