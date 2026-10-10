"""Validate runner stage ordering without touching PC datasets."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_offset_moe


class RunnerTests(unittest.TestCase):
    def test_regular_run_calls_offset_and_export_only(self):
        calls = []
        with patch.object(sys, 'argv', ['run_offset_moe.py']), \
             patch.object(run_offset_moe.paths, 'check_inputs'), \
             patch.object(run_offset_moe, 'run_script', side_effect=lambda *a: calls.append(a)):
            run_offset_moe.main()
        self.assertEqual([call[0].name for call in calls],
                         ['03_calculate_segment_moes.py', '10_export_moe_csvs.py'])

    def test_export_only_does_not_calculate(self):
        calls = []
        with patch.object(sys, 'argv', ['run_offset_moe.py', '--export-only']), \
             patch.object(run_offset_moe, 'run_script', side_effect=lambda *a: calls.append(a)):
            run_offset_moe.main()
        self.assertEqual([call[0].name for call in calls], ['10_export_moe_csvs.py'])

    def test_check_does_not_run_pipeline(self):
        calls = []
        with patch.object(sys, 'argv', ['run_offset_moe.py', '--check-inputs']), \
             patch.object(run_offset_moe.paths, 'check_inputs'), \
             patch.object(run_offset_moe, 'run_script', side_effect=lambda *a: calls.append(a)):
            run_offset_moe.main()
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
