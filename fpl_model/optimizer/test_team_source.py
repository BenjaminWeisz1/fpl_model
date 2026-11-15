"""Tests for team_source.py and solver_jobs.py's pure parts, on fake FPL responses (no network):
python -m unittest fpl_model/optimizer/test_team_source.py"""
import copy
import unittest

from fpl_model.optimizer import solve_modes, solver_jobs, team_source
from fpl_model.optimizer.team_source import BASE_URL, TeamError  # noqa: E402

TYPES = [1] * 2 + [2] * 5 + [3] * 5 + [4] * 3  # players 1-15, then 16-20 as spare buys


def world(next_gw=5, transfers=(), chips=(), charged=None, first_bank=None):
    """A team that picked players 1-15 at 50 (start price) each in GW1 (bank 250), then made `transfers`
    ((gw, out, in, out_cost, in_cost), ...)."""
    elements = [{"id": i, "element_type": TYPES[i - 1] if i <= 15 else 3, "now_cost": 52, "cost_change_start": 2}
                for i in range(1, 21)]
    static = {"elements": elements, "events": [{"id": g, "is_next": g == next_gw} for g in range(1, 39)],
              "chips": [{"name": n, "number": 1, "start_event": 1 if n in ("bboost", "3xc") else 2, "stop_event": 19}
                        for n in ("wildcard", "freehit", "bboost", "3xc")]}
    rows = [{"event": gw, "element_out": o, "element_in": i, "element_out_cost": oc, "element_in_cost": ic,
             "time": f"2026-0{gw}-01T00:00:{k:02d}Z"} for k, (gw, o, i, oc, ic) in enumerate(transfers)]
    fh = {c[1] for c in chips if c[0] == "freehit"}
    squad, bank = set(range(1, 16)), 250
    by_gw = {}
    for r in rows:
        if r["event"] in fh or r["event"] >= next_gw:
            continue
        squad = (squad - {r["element_out"]}) | {r["element_in"]}
        bank += r["element_out_cost"] - r["element_in_cost"]
        by_gw[r["event"]] = (set(squad), bank)
    history = {"current": [{"event": g, "event_transfers_cost": (charged or {}).get(g, 0)} for g in range(1, next_gw)],
               "chips": [{"name": n, "event": e} for n, e in chips]}
    last = next_gw - 1
    latest_squad, latest_bank = set(range(1, 16)), 250
    for g in range(1, last + 1):
        if g in by_gw:
            latest_squad, latest_bank = by_gw[g]
    responses = {
        f"{BASE_URL}/bootstrap-static/": static,
        f"{BASE_URL}/entry/7/history/": history,
        f"{BASE_URL}/entry/7/transfers/": list(reversed(rows)),
        f"{BASE_URL}/entry/7/event/1/picks/": {"picks": [{"element": i} for i in range(1, 16)],
                                               "entry_history": {"bank": 250 if first_bank is None else first_bank}},
        f"{BASE_URL}/entry/7/event/{last}/picks/": {"picks": [{"element": i} for i in latest_squad],
                                                    "entry_history": {"bank": latest_bank}},
    }
    return lambda url: copy.deepcopy(responses[url])


class TestTeamFromId(unittest.TestCase):
    def test_no_transfers(self):
        team = team_source.team_from_id(7, world())
        self.assertEqual(team["transfers"], {"bank": 250, "limit": 4, "made": 0})  # 1 in GW2, +1 a week to GW5
        pick = team["picks"][0]
        self.assertEqual((pick["purchase_price"], pick["selling_price"]), (50, 51))  # half the 0.2 rise

    def test_transfers_bank_and_free_transfers(self):
        team = team_source.team_from_id(7, world(transfers=[(2, 8, 16, 52, 45), (3, 9, 17, 52, 60), (3, 10, 18, 52, 50)],
                                                 charged={3: 4}))
        self.assertEqual(team["transfers"]["bank"], 250 + 7 - 8 + 2)
        # GW2: 1 FT, 1 used -> 1 for GW3; GW3: 2 used (one hit) -> 1 for GW4; GW4: none used -> 2 for GW5
        self.assertEqual(team["transfers"]["limit"], 2)
        self.assertTrue({16, 17, 18} <= {p["element"] for p in team["picks"]})

    def test_hit_mismatch_is_an_error(self):
        fetch = world(transfers=[(2, 8, 16, 52, 45), (3, 9, 17, 52, 60), (3, 10, 18, 52, 50)], charged={3: 4})
        team_source.team_from_id(7, fetch)  # 4 points charged: the rule agrees
        with self.assertRaises(TeamError):
            team_source.team_from_id(7, world(transfers=[(2, 8, 16, 52, 45), (3, 9, 17, 52, 60), (3, 10, 18, 52, 50)],
                                              charged={3: 0}))

    def test_free_hit_moves_revert_and_keep_fts(self):
        team = team_source.team_from_id(7, world(next_gw=6, transfers=[(4, 8, 16, 52, 45)], chips=[("freehit", 4)]))
        self.assertNotIn(16, {p["element"] for p in team["picks"]})
        self.assertEqual(team["transfers"]["limit"], 4)  # GW2 1, GW3 2, GW4 3 carried over the Free Hit, GW5 4
        fh = next(c for c in team["chips"] if c["name"] == "freehit")
        self.assertEqual((fh["status_for_entry"], fh["played_by_entry"]), ("played", [4]))

    def test_pending_transfer_counts_as_made(self):
        team = team_source.team_from_id(7, world(transfers=[(5, 8, 16, 52, 45)]))
        self.assertEqual(team["transfers"]["made"], 1)
        self.assertIn(16, {p["element"] for p in team["picks"]})
        self.assertEqual(team["transfers"]["bank"], 257)

    def test_first_week_bank_mismatch_is_an_error(self):
        with self.assertRaises(TeamError):
            team_source.team_from_id(7, world(first_bank=240))

    def test_free_transfers_cap(self):
        self.assertEqual(team_source.free_transfers({}, 1, 12, set()), 5)


class TestShapeAndJobs(unittest.TestCase):
    def test_check_shape(self):
        good = {"picks": [{"element": i, "purchase_price": 50, "selling_price": 50} for i in range(15)],
                "transfers": {"bank": 0, "limit": 1, "made": 0}}
        self.assertEqual(team_source.check_shape(good)["chips"], [])
        with self.assertRaises(TeamError):
            team_source.check_shape({"picks": good["picks"][:14], "transfers": good["transfers"]})

    def test_error_message_is_the_python_error(self):
        job = solver_jobs.Job()
        job.state = {"lines": ["Traceback (most recent call last):", '  File "x.py"', "ValueError: Free Hit in GW25 is outside."]}
        self.assertEqual(job._tail(), "Free Hit in GW25 is outside.")
        job.state = {"lines": ["something else", "went wrong"]}
        self.assertEqual(job._tail(), "something else\nwent wrong")

    def test_time_limits_cover_the_solver_limits(self):
        for mode in solve_modes.MODES:
            options = solve_modes.solver_options(mode)
            self.assertGreater(solve_modes.time_limit(mode), options["secs"])

    def test_bad_requests_do_not_start(self):
        job = solver_jobs.Job()
        self.assertIn("mode", job.start({"mode": "nope", "team": {}}))
        self.assertIn("Horizon", job.start({"horizon": 0, "team": {"x": 1}}))
        self.assertIn("Load a team", job.start({}))
        self.assertEqual(job.status()["status"], "idle")


if __name__ == "__main__":
    unittest.main()
