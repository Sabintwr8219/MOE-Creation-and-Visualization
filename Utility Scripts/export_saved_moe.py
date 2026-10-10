"""Export four CSVs from saved offset Parquet without re-running CV calculations."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    subprocess.run([sys.executable, "-u", str(ROOT / "10_export_moe_csvs.py"),
                    "--method", "offset", *sys.argv[1:]], check=True, cwd=ROOT)
