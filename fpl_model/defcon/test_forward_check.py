"""Outcome-join and leakage safeguards for the prospective comparison."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.defcon.forward_check import (
    assessment,
    attach_outcomes,
    compare,
    sha256,
    validate_unplayed,
    verify_snapshot,
)


def example():
    fixture = pd.DataFrame([{"Season": 2026, "MatchID": 3870, "Round": 8, "Home": "Everton", "Away": "Arsenal"}])
    forecasts = pd.DataFrame([{"Season": 2026, "MatchID": 3870, "Round": 8, "Home": "Everton", "Away": "Arsenal",
        "Player": p, "Team": "Everton", "Opponent": "Arsenal", "Venue": "Home",
        "P_legacy": .4, "P_updated": .5} for p in ("Regular", "Cameo", "Absent")])
    stats = []
    for team, opp, venue, names in (("Everton", "Arsenal", "Home", ["Regular", "Cameo"]+[f"E{i}" for i in range(9)]),
                                    ("Arsenal", "Everton", "Away", [f"A{i}" for i in range(11)])):
        for name in names:
            stats.append({"Season": 2026, "MatchID": 9999, "Team": team, "Opponent": opp,
                          "Venue": venue, "Player": name,
                          "Minutes": 30 if name == "Cameo" else 60 if name == "Regular" else 90,
                          "CBI": 5 if name == "Cameo" else 8, "Tkl": 0})
    results = fixture.drop(columns=["Round"]).assign(Home_Goals=0, Away_Goals=0)
    return forecasts, fixture, results, pd.DataFrame(stats)


class ForwardCheckTest(unittest.TestCase):
    def test_retrospective_freeze_refused_for_results_or_early_stats(self):
        _, fixtures, results, stats = example()
        with self.assertRaisesRegex(ValueError, "already in results"):
            validate_unplayed(fixtures, results, stats.iloc[:0])
        with self.assertRaisesRegex(ValueError, "already in stats"):
            validate_unplayed(fixtures, results.iloc[:0], stats)
        validate_unplayed(fixtures, results.iloc[:0], stats.iloc[:0])

    def test_exact_training_target_and_no_appearance_exclusion(self):
        rows, coverage = attach_outcomes(*example())
        self.assertEqual(rows.set_index("Player").Target.to_dict(), {"Regular": 1, "Cameo": 0})
        self.assertEqual(coverage["no_appearance_rows"], 1)
        self.assertEqual(coverage["scored_appearances"], 2)
        self.assertEqual(coverage["stats_ready_fixtures"], 1)
        # FPL player MatchIDs deliberately differ from fixture IDs; natural keys join.
        self.assertTrue(rows.MatchID.eq(3870).all())

    def test_unfinished_fixture_never_scores(self):
        f, fx, results, stats = example()
        rows, coverage = attach_outcomes(f, fx, results.iloc[:0], stats)
        self.assertTrue(rows.empty)
        self.assertEqual(coverage["pending_fixtures"], 1)

    def test_partial_stats_are_pending_not_false_non_appearances(self):
        f, fx, results, stats = example()
        rows, coverage = attach_outcomes(f, fx, results, stats.iloc[:-1])
        self.assertTrue(rows.empty)
        self.assertEqual(coverage["stats_pending_fixtures"], 1)
        self.assertEqual(coverage["no_appearance_rows"], 0)

    def test_duplicate_outcome_rejected(self):
        f, fx, results, stats = example()
        with self.assertRaisesRegex(ValueError, "Duplicate player"):
            attach_outcomes(f, fx, results, pd.concat([stats, stats.iloc[[0]]]))

    def test_missing_action_is_not_zero(self):
        f, fx, results, stats = example()
        stats.loc[stats.Player.eq("Regular"), "CBI"] = np.nan
        rows, coverage = attach_outcomes(f, fx, results, stats)
        self.assertEqual(rows.Player.tolist(), ["Cameo"])
        self.assertEqual(coverage["invalid_stat_rows"], 1)

    def test_equal_probabilities_have_zero_paired_differences(self):
        rows, _ = attach_outcomes(*example())
        rows["P_updated"] = rows.P_legacy
        result = compare(rows)
        self.assertEqual(result["delta"]["log_loss"], 0)
        self.assertEqual(result["bootstrap"]["Player"]["log_loss_delta_95ci"], [0., 0.])
        self.assertIsNone(result["bootstrap"]["MatchID"]["log_loss_delta_95ci"])

    def test_single_class_and_confident_mistakes_are_finite(self):
        rows, _ = attach_outcomes(*example())
        rows = rows[rows.Target.eq(1)].copy()
        rows["P_updated"] = 0.
        result = compare(rows)
        self.assertIsNone(result["updated"]["auc"])
        self.assertTrue(np.isfinite(result["updated"]["log_loss"]))
        self.assertGreater(result["updated"]["log_loss"], 30.)

    def test_interim_cannot_trigger_winner_or_rollback(self):
        rows, coverage = attach_outcomes(*example())
        self.assertEqual(assessment(compare(rows), coverage, 50), "interim_descriptive_only")
        self.assertEqual(assessment(compare(rows), coverage, 1), "final_insufficient_evidence")

    def test_forecast_tampering_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "forecasts.csv").write_text("original")
            (p / "manifest.json").write_text(json.dumps({"frozen_sha256": {"forecasts.csv": sha256(p/"forecasts.csv")}}))
            verify_snapshot(p)
            (p / "forecasts.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Frozen study file changed"):
                verify_snapshot(p)


if __name__ == "__main__":
    unittest.main()
