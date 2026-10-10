"""Synthetic exporter checks; do not need user data or an installed PyArrow."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_four_production_moe_csvs import COMMON, OUTPUTS, export


class InMemoryParquet:
    """Minimal in-memory stand-in for a Parquet reader, not the actual writer."""
    def __init__(self, rows):
        self.frame = rows
        self.metadata = SimpleNamespace(num_rows=len(rows))
        self.schema_arrow = SimpleNamespace(names=list(rows.columns))

    def iter_batches(self, batch_size, columns):
        for i in range(0, len(self.frame), batch_size):
            yield SimpleNamespace(to_pandas=lambda f=self.frame.iloc[i:i+batch_size]: f[columns].copy())


def sample_records():
    return pd.DataFrame([
        dict(link_key="A_0_1", parent_start_ft=0, parent_end_ft=500,
             segment_length_ft=500, time_bin=pd.Timestamp("2025-09-20 19:00:00"),
             waypoint_count=10, journey_count=3, avg_speed_mph=70,
             posted_speed_limit_mph=65, slow_movement_count=0,
             slow_movement_pct=0, dsh_trip_count=1, DSH=.4,
             speed_estimated_travel_time_s=4.87),
        dict(link_key="A_0_1", parent_start_ft=500, parent_end_ft=610.5,
             segment_length_ft=110.5, time_bin=pd.Timestamp("2025-09-20 19:00:00"),
             waypoint_count=2, journey_count=1, avg_speed_mph=25,
             posted_speed_limit_mph=None, slow_movement_count=1,
             slow_movement_pct=0, dsh_trip_count=0, DSH=0,
             speed_estimated_travel_time_s=3.01),
        dict(link_key="A_0_1", parent_start_ft=0, parent_end_ft=500,
             segment_length_ft=500, time_bin=pd.Timestamp("2025-09-20 19:15:00"),
             waypoint_count=None, journey_count=None, avg_speed_mph=None,
             posted_speed_limit_mph=65, slow_movement_count=None,
             slow_movement_pct=None, dsh_trip_count=None, DSH=None,
             speed_estimated_travel_time_s=None),
    ])


class FourCsvTests(unittest.TestCase):
    def test_all_schemas_and_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "fake.parquet"
            source.touch()
            reader = InMemoryParquet(sample_records())
            out = export(source, folder / "published", batch_size=1000,
                         expected_total=3, expected_observed=2, expected_waypoints=12,
                         reader_factory=lambda _: reader)
            for filename, metrics in OUTPUTS.items():
                with (out / filename).open(encoding="utf-8-sig", newline="") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 2)
                self.assertEqual(list(rows[0]), COMMON + metrics)
                self.assertEqual([r["unique_veh_count"] for r in rows], ["3", "1"])
                self.assertEqual(rows[1]["segment_section_ft"], "500-610.5")
                self.assertEqual(rows[1]["start_time"], "2025-09-20 19:00:00")
                self.assertEqual(rows[1]["end_time"], "2025-09-20 19:15:00")
                if filename == "speed_reduction.csv":
                    self.assertEqual(float(rows[0]["speed_reduction_pct"]), 0)
                    self.assertEqual(rows[1]["speed_reduction_pct"], "")
            summary = json.loads((out / "export_manifest.json").read_text())
            self.assertEqual(summary["sum_unique_veh_count_over_cells"], 4)
            self.assertEqual(summary["sum_waypoint_count"], 12)
            # A second export must not overwrite the first directory.
            second = export(source, folder / "published", batch_size=1000,
                            reader_factory=lambda _: reader)
            self.assertNotEqual(out, second)
            self.assertTrue(out.is_dir() and second.is_dir())

    def test_observed_cell_without_journey_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "fake.parquet"
            source.touch()
            records = sample_records().iloc[[0]].copy()
            records["journey_count"] = float("nan")
            with self.assertRaises(ValueError):
                export(source, folder / "no_publish", batch_size=1000,
                       reader_factory=lambda _: InMemoryParquet(records))
            self.assertFalse((folder / "no_publish").exists())


if __name__ == "__main__":
    unittest.main()
