"""Tests for solve_modes.py on a toy model named like the pinned solver's (python -m unittest scripts/optimizer/test_solve_modes.py)."""
import unittest

import highspy

from fpl_model.optimizer import solve_modes  # noqa: E402

BIN = highspy.HighsVarType.kInteger


def toy(cls, settings):
    """Two players, weeks 11-12. Pick one lineup player a week (maximise points); squad = lineup; one
    transfer_in per week allowed."""
    cls.settings = settings
    m = cls()
    m.setOptionValue("output_flag", False)
    pts = {(1, 11): 3.0, (2, 11): 2.0, (1, 12): 1.0, (2, 12): 4.0}
    v = {}
    for name in ("transfer_in", "squad", "lineup", "captain"):
        for p in (1, 2):
            for w in (11, 12):
                v[name, p, w] = m.addVariable(lb=0, ub=1, type=BIN, name=f"{name}({p},{w})")
    for w in (11, 12):
        m.addConstr(sum(v["lineup", p, w] for p in (1, 2)) == 1)
        m.addConstr(sum(v["captain", p, w] for p in (1, 2)) == 1)
        for p in (1, 2):
            m.addConstr(v["lineup", p, w] <= v["squad", p, w])
            m.addConstr(v["captain", p, w] <= v["lineup", p, w])
            m.addConstr(v["squad", p, w] <= v["transfer_in", p, w])
    m.setObjective(sum(pts[p, w] * (v["lineup", p, w] + v["captain", p, w]) for p in (1, 2) for w in (11, 12)),
                   sense=highspy.ObjSense.kMaximize)
    return m, v


class TestModeHighs(unittest.TestCase):
    def test_week_parse(self):
        self.assertEqual(solve_modes._week("squad(411,12)"), 12)
        self.assertEqual(solve_modes._week("use_wc(12)"), 12)
        self.assertEqual(solve_modes._week("bench(4,13,0)"), 13)

    def test_plain_run_is_optimal(self):
        m, _ = toy(solve_modes.ModeHighs, {})
        m.run()
        self.assertAlmostEqual(m.getInfo().objective_function_value, 14.0)

    def test_relax_fix_restores_integrality_and_stays_optimal(self):
        m, v = toy(solve_modes.ModeHighs, {"relax_fix": {"weeks": 1, "secs": 10}})
        m.run()
        self.assertAlmostEqual(m.getInfo().objective_function_value, 14.0)
        lp = m.getLp()
        self.assertTrue(all(int(t) == 1 for t in lp.integrality_))

    def test_repair_and_clear_on_second_run(self):
        m, v = toy(solve_modes.ModeHighs, {"repair": True, "clear": True})
        m.run()
        m.addConstr(v["lineup", 1, 11] == 0)  # a no-good on week 11
        m.run()
        self.assertEqual(m.plan_runs, 2)
        self.assertAlmostEqual(m.getInfo().objective_function_value, 12.0)

    def test_restrict_start_restores_bounds(self):
        # the restricted model is infeasible here (week 12 needs a transfer); the full solve must still be optimal
        m, _ = toy(solve_modes.ModeHighs, {"restrict_start": {"weeks": 1, "secs": 10}})
        m.run()
        self.assertAlmostEqual(m.getInfo().objective_function_value, 14.0)
        self.assertTrue(all(u == 1 for u in m.getLp().col_upper_))

    def test_modes_have_options(self):
        for mode in solve_modes.MODES:
            self.assertIn("gap", solve_modes.solver_options(mode))


if __name__ == "__main__":
    unittest.main()
