"""python -m unittest test_session_a_fixes (from scripts): §188 order, §189 safety fixes."""
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

HERE = Path(__file__).resolve().parent
from fpl_model.atomic_write import to_csv_atomic  # noqa: E402
from fpl_model.match_order import chrono_order  # noqa: E402
import pipeline_stages  # noqa: E402


class ChronoOrder(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        pd.DataFrame({"Season": [2026, 2026], "Round": [1, 2], "Home": ["A", "C"], "Away": ["B", "A"]}).to_csv(
            Path(self.dir.name) / "fixtures.csv", index=False)

    def tearDown(self):
        self.dir.cleanup()

    def test_round_beats_matchid_and_old_seasons_keep_order(self):
        rows = pd.DataFrame({"Season": [2026, 2026, 2025, 2025], "MatchID": [10, 20, 2, 1], "Team": ["A"] * 4,
                             "Opponent": ["C", "B", "X", "X"], "Venue": ["Away", "Home", "Home", "Home"]})
        self.assertEqual(chrono_order(rows, self.dir.name).tolist(), [3, 2, 1, 0])

    def test_unknown_fixture_sorts_last_in_season(self):
        rows = pd.DataFrame({"Season": [2026, 2026], "MatchID": [1, 2], "Team": ["Z", "A"], "Opponent": ["Y", "B"],
                             "Venue": ["Home", "Home"]})
        self.assertEqual(chrono_order(rows, self.dir.name).tolist(), [1, 0])


class Atomic(unittest.TestCase):
    def test_failed_write_keeps_the_old_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.csv"
            path.write_text("old", encoding="utf-8")
            frame = pd.DataFrame({"a": [1]})
            with mock.patch.object(Path, "replace", side_effect=PermissionError):
                with self.assertRaises(PermissionError):
                    to_csv_atomic(frame, path, index=False)
            self.assertEqual(path.read_text(encoding="utf-8"), "old")
            self.assertEqual(os.listdir(d), ["x.csv"])
            to_csv_atomic(frame, path, index=False)
            self.assertEqual(path.read_text(encoding="utf-8").split(), ["a", "1"])


class Pipeline(unittest.TestCase):
    def test_missing_script_fails(self):
        stage = {"label": "x", "script": pipeline_stages.ROOT / "no_such_script.py"}
        self.assertFalse(pipeline_stages.run_stage(stage))


class TeamFresh(unittest.TestCase):
    def setUp(self):
        try:
            from fpl_model.optimizer import run_optimizer
        except ImportError as error:
            self.skipTest(f"solver not importable: {error}")
        self.check = run_optimizer.check_team_fresh
        self.events = [{"deadline_time": "2026-10-31T11:00:00Z"}, {"deadline_time": "2026-11-06T18:30:00Z"}]
        self.now = datetime(2026, 11, 3, tzinfo=timezone.utc)

    def test_saved_after_latest_deadline(self):
        self.check(datetime(2026, 11, 1, tzinfo=timezone.utc), self.events, self.now)

    def test_saved_before_latest_deadline(self):
        with self.assertRaises(ValueError):
            self.check(datetime(2026, 10, 30, tzinfo=timezone.utc), self.events, self.now)

    def test_future_deadline_does_not_count(self):
        self.check(datetime(2026, 10, 31, 12, tzinfo=timezone.utc), self.events, self.now - timedelta(hours=1))


if __name__ == "__main__":
    unittest.main()
