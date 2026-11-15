"""Weekly archive integrity, freshness and round-label tests."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from pipeline_stages import parse_gameweek, run_pipeline
from save_gameweek_outputs import PART1, PART2, save_snapshot, verify_snapshot, snapshot_plan


def inputs(data):
    data.mkdir()
    def write(name, rows):
        pd.DataFrame(rows).to_csv(data/name, index=False, encoding="utf-8-sig")
    fixtures = [{"MatchID": gw, "Season":2026, "Round":gw, "Home":"Everton", "Away":opp}
                for gw, opp in [(7,"Chelsea"),(8,"Arsenal"),(9,"Liverpool")]]
    write("fixtures.csv", fixtures)
    write("match_results.csv", [{"Season":2026, "Home":"Everton", "Away":"Chelsea", "Home_Goals":0, "Away_Goals":2}])
    write("players_current.csv", [{"Player":"James Tarkowski", "Web_Name":"Tarkowski", "Pos":"D", "Team":"Everton"}])
    write("player_match_stats.csv", [{"Season":2026,"Player":"James Tarkowski", "Team":"Everton", "Opponent":"Chelsea", "Venue":"Home", "Minutes":90}])
    for name in PART1:
        write(name, [{"Player":"Tarkowski", "Rating":.6}])
    write("predictions_defensive.csv", [{"MatchID":gw,"Player":"James Tarkowski","Probability":.5,"Probability_at_90min":.6} for gw in (8,9)])
    for name in PART2:
        if name in ("fixture_goal_predictions.csv","simulation_features.csv","bonus_predictions.csv"):
            write(name,[{"Round":gw,"Player":"James Tarkowski","GW":gw-7} for gw in (8,9)])
        elif name=="minutes_probabilities.csv":
            write(name,[{"Player":"Tarkowski","xMins_GW1":90,"xMins_GW2":80,"P60_GW1":.9,"P60_GW2":.8}])
        elif name=="penalty_predictions.csv":
            write(name,[{"Player":"James Tarkowski","PenPts_GW1":0.,"PenPts_GW2":.1}])
        elif name=="predicted_points.csv":
            write(name,[{"Player":"Tarkowski","Pos":"D","Team":"EVE","Price":6.,"8":4.,"9":3.,"Total":7.}])
        else:
            write(name,[{"Player":"Tarkowski","Value":0.}])
    # Final assembly is written last in the real pipeline as well.
    write("predicted_points.csv",[{"Player":"Tarkowski","Pos":"D","Team":"EVE","Price":6.,"8":4.,"9":3.,"Total":7.}])


class SaveOutputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data, self.output = self.root/"current", self.root/"snapshots"
        inputs(self.data)

    def tearDown(self):
        self.temp.cleanup()

    def test_full_snapshot_has_originals_and_absolute_round_views(self):
        saved = save_snapshot(self.data,self.output)
        manifest = verify_snapshot(saved)
        self.assertEqual(manifest["gameweek"],8)
        self.assertEqual(manifest["forecast_origin_gameweek"],8)
        self.assertEqual((saved/"outputs"/"predictions_defensive.csv").read_bytes(),(self.data/"predictions_defensive.csv").read_bytes())
        view=pd.read_csv(saved/"gameweek"/"predictions_defensive.csv",encoding="utf-8-sig")
        self.assertEqual(view.Round.tolist(),[8])
        mins=pd.read_csv(saved/"gameweek"/"minutes_probabilities.csv",encoding="utf-8-sig")
        self.assertEqual(mins.xMins.tolist(),[90])

    def test_benchmarks_copied_only_for_their_own_gameweek(self):
        from save_gameweek_outputs import benchmark_files
        data = self.root/"repo"/"data"/"current"
        raw = self.root/"repo"/"data"/"raw"/"solio"/"2026"/"GW08"
        data.mkdir(parents=True); raw.mkdir(parents=True)
        (raw/"solio_baselines.csv").write_text("x\n1\n"); (raw/"capture.json").write_text("{}")
        (data/"solio_baselines.csv").write_text("x\n1\n")
        (data/"solio_status.json").write_text(json.dumps({"gameweek": 8, "source_folder": "data/raw/solio/2026/GW08"}))
        (data/"fplreview_projections.csv").write_text("x\n1\n")
        (data/"fplreview_status.json").write_text(json.dumps({"usable_for_comparison": False}))  # leftover, no gameweek
        files = benchmark_files(data, 8)
        self.assertEqual(sorted(files), ["solio/raw/capture.json", "solio/raw/solio_baselines.csv",
                                         "solio/solio_baselines.csv", "solio/solio_status.json"])
        self.assertEqual(benchmark_files(data, 9), {})  # another week's capture is never filed under GW9

    def test_ratings_snapshot_omits_stale_part2(self):
        (self.data/"predicted_points.csv").unlink()
        saved=save_snapshot(self.data,self.output,part="1")
        self.assertFalse((saved/"outputs"/"predicted_points.csv").exists())
        self.assertTrue((saved/"outputs"/"defcon_average_probabilities.csv").exists())

    def test_change_during_validation_is_not_archived(self):
        def changing_plan(*args, **kwargs):
            result=snapshot_plan(*args, **kwargs)
            with (self.data/"player_ratings.csv").open("a") as stream:
                stream.write("\n")
            return result
        with patch("save_gameweek_outputs.snapshot_plan",side_effect=changing_plan):
            with self.assertRaisesRegex(ValueError,"changed while checking"):
                save_snapshot(self.data,self.output,part="1")
        self.assertFalse(self.output.exists())

    def test_rerun_keeps_both_revisions(self):
        first=save_snapshot(self.data,self.output,part="1")
        before=(first/"manifest.json").read_bytes()
        second=save_snapshot(self.data,self.output,part="1")
        self.assertNotEqual(first,second)
        self.assertEqual((first/"manifest.json").read_bytes(),before)
        verify_snapshot(first)

    def test_stale_origin_rejected_before_writing(self):
        p=pd.read_csv(self.data/"predicted_points.csv",encoding="utf-8-sig").rename(columns={"8":"7"})
        p.to_csv(self.data/"predicted_points.csv",index=False)
        with self.assertRaisesRegex(ValueError,"stale"):
            save_snapshot(self.data,self.output)
        self.assertFalse(self.output.exists())

    def test_newer_component_requires_reassembly(self):
        with (self.data/"bonus_predictions.csv").open("a") as stream:
            stream.write("\n")
        # Windows filesystems can give consecutive writes the same timestamp.
        newer=(self.data/"predicted_points.csv").stat().st_mtime_ns+2_000_000_000
        os.utime(self.data/"bonus_predictions.csv",ns=(newer,newer))
        with self.assertRaisesRegex(ValueError,"predate"):
            save_snapshot(self.data,self.output)

    def test_missing_required_file_is_not_silently_skipped(self):
        (self.data/"bonus_predictions.csv").unlink()
        with self.assertRaisesRegex(ValueError,"Missing required"):
            save_snapshot(self.data,self.output)

    def test_outcomes_refuse_prematch_label(self):
        stats=pd.read_csv(self.data/"player_match_stats.csv",encoding="utf-8-sig")
        stats["Opponent"]="Arsenal"
        stats.to_csv(self.data/"player_match_stats.csv",index=False)
        with self.assertRaisesRegex(ValueError,"outcomes already exist"):
            save_snapshot(self.data,self.output,part="1")

    def test_tampering_detected(self):
        saved=save_snapshot(self.data,self.output,part="1")
        (saved/"outputs"/"player_ratings.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError,"Saved file changed"):
            verify_snapshot(saved)

    def test_folder_and_manifest_carry_gameweek_and_timing(self):
        pd.DataFrame([{"Gameweek":8,"Deadline_UTC":"2026-10-23T17:30:00Z"}]).to_csv(self.data/"gameweek_deadlines.csv",index=False)
        for now,tag in ((datetime(2026,10,23,17,29,tzinfo=timezone.utc),"pre_deadline"),(datetime(2026,10,23,17,30,tzinfo=timezone.utc),"after_deadline")):
            saved=save_snapshot(self.data,self.output,part="1",now=now)
            self.assertEqual(saved.parent.name,"GW08")
            self.assertTrue(saved.name.endswith(f"_GW08_{tag}_part1"),saved.name)
            manifest=verify_snapshot(saved)
            self.assertEqual(manifest["timing"],tag)
            self.assertEqual(manifest["deadline_utc"],"2026-10-23T17:30:00+00:00")

    def test_unknown_deadline_is_tagged_not_guessed(self):
        saved=save_snapshot(self.data,self.output,part="1")
        self.assertIn("_GW08_deadline_unknown_",saved.name)
        self.assertIsNone(verify_snapshot(saved)["deadline_utc"])
        pd.DataFrame([{"Gameweek":1,"Deadline_UTC":"2026-08-01T17:30:00Z"}]).to_csv(self.data/"gameweek_deadlines.csv",index=False)
        self.assertIn("_GW08_deadline_unknown_",save_snapshot(self.data,self.output,part="1").name)

    def test_gameweek_argument_reaches_only_stages_that_accept_it(self):
        self.assertEqual(parse_gameweek(["run.py","--gameweek","10"]),10)
        self.assertIsNone(parse_gameweek(["run.py","--preseason"]))
        for argv in (["run.py","--gameweek"],["run.py","--gameweek","ten"],["run.py","--gameweek","0"]):
            with self.assertRaises(ValueError):
                parse_gameweek(argv)
        calls=[]
        def fake(command,**kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0,stdout="",stderr="")
        here=Path(__file__).resolve()
        with patch("pipeline_stages.subprocess.run",fake):
            self.assertEqual(run_pipeline("t",[{"label":"a","script":here},{"label":"b","script":here,"args":["--part","1"],"accepts_gameweek":True}],["x","--gameweek","10"]),0)
        self.assertNotIn("--gameweek",calls[0])
        self.assertEqual(calls[1][-4:],["--part","1","--gameweek","10"])
        self.assertEqual(run_pipeline("t",[],["x","--gameweek","ten"]),1)

    def test_explicit_future_week_preserves_origin_mapping(self):
        saved=save_snapshot(self.data,self.output,gameweek=9)
        manifest=verify_snapshot(saved)
        self.assertEqual(manifest["gameweek"],9)
        self.assertEqual(manifest["forecast_origin_gameweek"],8)
        mins=pd.read_csv(saved/"gameweek"/"minutes_probabilities.csv",encoding="utf-8-sig")
        self.assertEqual(mins.xMins.tolist(),[80])


if __name__ == "__main__":
    unittest.main()
