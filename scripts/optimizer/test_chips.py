"""Tests for the fixed-chip rules (chips.py) and the replay's chip helpers (§226).

    python -m unittest scripts/optimizer/test_chips.py
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chips  # noqa: E402
import replay_plans  # noqa: E402

# bootstrap-static's 2026/27 chip instances
FPL = {"chips": [
    {"name": "wildcard", "start_event": 2, "stop_event": 19}, {"name": "wildcard", "start_event": 20, "stop_event": 38},
    {"name": "freehit", "start_event": 2, "stop_event": 19}, {"name": "freehit", "start_event": 20, "stop_event": 38},
    {"name": "bboost", "start_event": 1, "stop_event": 19}, {"name": "bboost", "start_event": 20, "stop_event": 38},
    {"name": "3xc", "start_event": 1, "stop_event": 19}, {"name": "3xc", "start_event": 20, "stop_event": 38},
]}
# the user's team_data.json chips: first-half BB, TC, WC played; FH available
TEAM = {"chips": [
    {"name": "bboost", "status_for_entry": "played", "played_by_entry": [1], "start_event": 1, "stop_event": 19},
    {"name": "3xc", "status_for_entry": "played", "played_by_entry": [3], "start_event": 1, "stop_event": 19},
    {"name": "wildcard", "status_for_entry": "played", "played_by_entry": [4], "start_event": 2, "stop_event": 19},
    {"name": "freehit", "status_for_entry": "available", "played_by_entry": [], "start_event": 2, "stop_event": 19},
]}
GWS = list(range(11, 21))


class ValidateTests(unittest.TestCase):
    def test_available_chips_pass(self):
        self.assertEqual(chips.validate({"fh": 14, "wc": 20}, GWS, FPL, TEAM), {"wc": 20, "fh": 14, "bb": None, "tc": None})

    def test_played_first_half_chip_rejected(self):
        with self.assertRaisesRegex(ValueError, "has been played"):
            chips.validate({"wc": 12}, GWS, FPL, TEAM)

    def test_second_half_set_is_fresh(self):
        self.assertEqual(chips.validate({"bb": 20}, GWS, FPL, TEAM)["bb"], 20)

    def test_one_chip_per_week(self):
        with self.assertRaisesRegex(ValueError, "one chip"):
            chips.validate({"fh": 20, "bb": 20}, GWS, FPL, TEAM)

    def test_outside_horizon(self):
        with self.assertRaisesRegex(ValueError, "outside the horizon"):
            chips.validate({"fh": 21}, GWS, FPL, TEAM)

    def test_free_hit_not_in_gw1(self):
        with self.assertRaisesRegex(ValueError, "No Free Hit covers GW1"):
            chips.validate({"fh": 1}, [1, 2], FPL, {"chips": []})

    def test_unknown_chip(self):
        with self.assertRaises(ValueError):
            chips.normalise({"xx": 3})

    def test_solver_options_force_only_fixed_chips(self):
        options = chips.solver_options({"fh": 14})
        self.assertEqual(options["chip_limits"], {"wc": 0, "fh": 0, "bb": 0, "tc": 0})
        self.assertEqual((options["use_fh"], options["use_wc"], options["use_bb"], options["use_tc"]), ([14], [], [], []))


class ReplayHelperTests(unittest.TestCase):
    def test_bench_boost_week_scores_without_a_bench(self):
        """A BB week's bench is empty: no bench tie-breakers (the gw11_h10_chips benchmark crashed here)."""
        inputs = {"points": {(i, 1): 2.0 for i in range(15)}}
        points, extras = replay_plans.score_week(1, list(range(15)), [], 0, 1, inputs)
        self.assertEqual((points, extras), (32.0, replay_plans.VICE_WEIGHT * 2.0))

    def test_week_chip_reads_solver_text(self):
        self.assertEqual(replay_plans.week_chip(pd.DataFrame({"chip": ["FH", "FH"]})), "fh")
        self.assertEqual(replay_plans.week_chip(pd.DataFrame({"chip": [None, "TC", None]})), "tc")
        self.assertIsNone(replay_plans.week_chip(pd.DataFrame({"chip": [None, None]})))

    def test_ft_count_carries_over_a_chip(self):
        """1 FT then a Free Hit -> 1 FT the next week (user, 2026-11-10); a normal roll gains one."""
        self.assertEqual(replay_plans.next_fts(1, 0), 2)
        # chip weeks keep the count: replay_plan uses after = fts for wc/fh, checked end to end by the benchmark

    def test_free_hit_budget(self):
        """Bank + selling value of the real squad; kept players cost their selling price, others the current price."""
        types = {1: {"squad_select": 2}, 2: {"squad_select": 5}, 3: {"squad_select": 5}, 4: {"squad_select": 3}}
        positions = [1] * 2 + [2] * 5 + [3] * 5 + [4] * 3
        live = {i: {"element_type": positions[i % 15], "team": i, "now_cost": 50} for i in range(30)}
        inputs = {"live": live, "types": types}
        held = {i: 50 for i in range(15)}             # the real squad, 750 of selling value
        week = pd.DataFrame({"id": range(15, 30), "squad": 1})   # a new squad at 50 each = 750
        self.assertEqual(replay_plans.free_hit_squad(week, held, 0, inputs), list(range(15, 30)))
        live[15]["now_cost"] = 51
        with self.assertRaisesRegex(AssertionError, "Free Hit squad costs 751"):
            replay_plans.free_hit_squad(week, held, 0, inputs)
        self.assertEqual(len(replay_plans.free_hit_squad(week, held, 1, inputs)), 15)  # 0.1 in the bank covers it


if __name__ == "__main__":
    unittest.main()
