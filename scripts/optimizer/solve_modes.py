"""Solve modes for the transfer optimizer: Fast, Standard, Deep (SOLVER_SPEED_PLAN.md, §227).

Every solve-speed setting lives here and nowhere else; run_optimizer.py, optimizer_app.py and benchmark_solver.py
import it. The pinned solver (open_fpl_solver/dev/solver.py) stays unchanged: it builds its model with
`highspy.Highs()`, and `install(mode)` points that name at `ModeHighs`, a `highspy.Highs` subclass whose `run()`
adds the mode's levers around each plan's solve:

- `clear`: `clearSolver()` before plans 2 and 3 (drops HiGHS's search data from the earlier plans; the model stays).
- `repair`: plans 2 and 3 start from the previous plan's squads, lineups, captains and bench after GW-now, given as a
  partial solution that HiGHS completes with new GW-now moves (the no-good constraint forbids the old ones).
- `relax_fix`: before plan 1, a heuristic start: lineup, captain, vice-captain, bench and TC variables in weeks after
  GW-now + `weeks` - 1 are made continuous and that model is solved (time `secs`, gap `gap`); its transfers, squads
  and chips are then given to the real model as a partial solution. Exact modes stay exact: it is only a start.
- `continuous_assignment`: bench-order and vice-captain variables are continuous. Exact: once squads, lineups and
  captains are integral, the bench constraints are a transportation problem (players with capacity 0/1 against
  slots needing 1) and the vice-captain constraints a single-slot assignment; both polytopes have integral vertices,
  so the optimum is unchanged. If a final solution still has a fractional value there (a tie), every other integer
  column is fixed at its value and the model is re-solved with those columns integral again.
- `process_per_plan`: each plan is solved in its own process (`solve_plans`), so HiGHS's memory from one plan is
  returned to Windows before the next. Plan k's model gets the same no-good rows the pinned solver adds between
  plans (`this_gw_transfer_in_out`: differ from each earlier plan in at least one GW-now transfer in or out), built
  by `ModeHighs` from the setting `exclude_moves`. Falls back to one process when GW-now is a Free Hit week (the
  solver then compares lineups instead).
- `restrict_start`: before plan 1, the same model with every transfer banned from GW-now + `weeks` on is solved
  (time `secs`, gap `gap`). Any such plan is feasible for the full model (keep the squad), so its solution is a
  complete start. Exact modes stay exact: it is only a start.
- `transfer_weeks`: an approximation (Fast, Standard). Transfers are allowed only in the first `transfer_weeks`
  weeks (GW-now counts as 1) and in any week whose Wildcard is fixed on; the later weeks keep the squad. Applied as
  column bounds, which presolve removes (upstream's row form, `no_transfer_last_gws`, is far slower: §227). Lineups,
  captains and the end-of-horizon FT value still count every week, and the replay scores the real plan.
- `later_plans`: plans 2 and 3 use these settings instead (`transfer_weeks`, `gap`, `secs`): Standard solves plan 1
  exactly and the alternatives approximately, each with its measured loss.
- `options`: solver options (gap, secs, num_iterations) layered over run_optimizer.OVERRIDES.
- `highs`: extra HiGHS options.
"""
import time

import highspy
import numpy as np

BaseHighs = highspy.Highs  # the real class, kept before install() repoints highspy.Highs at a subclass

# Variables whose weekly values make up a plan's lineup decisions (relaxed by relax_fix, carried by repair).
LINEUP_PREFIXES = ("lineup(", "captain(", "vicecap(", "bench(", "use_tc(")
SQUAD_PREFIXES = ("squad(",)
ASSIGNMENT_PREFIXES = ("bench(", "vicecap(")
DECISION_PREFIXES = ("transfer_in(", "tr_out_first(", "tr_out_reg(", "squad(", "squad_fh(", "use_wc(", "use_bb(", "use_fh(")

# Measured on the six benchmark instances (SOLVER_SPEED_PLAN.md, CHANGELOG §227). Deep is the proven optimum; the
# others are approximations whose loss against Deep is measured, never assumed, and shown with their plans.
MODES = {
    # today's settings: gap 0, three plans, each must prove its optimum
    "deep": {"options": {"gap": 0, "secs": 1800, "num_iterations": 3}},
    # plan 1 exactly as Deep; plans 2 and 3 (the alternatives) with transfers only in the first 5 weeks, gap 0.05%,
    # at most 60 s each
    "standard": {"options": {"gap": 0, "secs": 1800, "num_iterations": 3},
                 "later_plans": {"transfer_weeks": 5, "gap": 0.0005, "secs": 60}},
    # one plan; transfers only in the first 4 weeks, gap 0.05%, at most 60 s
    "fast": {"options": {"gap": 0.0005, "secs": 60, "num_iterations": 1}, "transfer_weeks": 4},
}
PROGRESS = "[progress]"
PREP_SECONDS = 180  # margin over the solver's own limits for fetching, building the model and the replay
DEFAULT_MODE = "standard"  # the user's choice of default (website_plan.md "Three solve modes")
# The measured cost of each approximate mode against Deep (benchmark, §227), printed with its plans.
MEASURED = {
    "deep": "three plans, each proven optimal (gap 0); 25-28 min and 2.4-2.5 GB at 10 weeks",
    "standard": ("plan 1 is Deep's own (proven optimum on all six benchmark instances); plans 2-3 are approximations, "
                 "0 to 1.1 points below Deep's plans 2-3; about 10-11 min at 10 weeks against Deep's 25-28 min"),
    "fast": ("one plan in 34-63 s and under 350 MB; 0 to 1.15 points below Deep's best plan (0.63 at GW11 over 10 "
             "weeks); its GW-now moves differed from Deep's on 1 of 6 benchmark instances"),
}


def _week(name):
    """'squad(411,12)' -> 12; 'use_wc(12)' -> 12; 'bench(4,12,0)' -> 12."""
    inside = name[name.index("(") + 1:-1].split(",")
    return int(inside[1] if len(inside) > 1 else inside[0])


class ModeHighs(BaseHighs):
    """highspy.Highs with the mode's levers around each run() (one run per plan)."""

    settings = {}

    def __init__(self):
        super().__init__()
        self.plan_runs = 0
        self.phase = "main"  # "heuristic" while a lever's own sub-solve runs (recorders ignore those solutions)
        self._last = None

    # names are read once per model; the model only gains rows between plans
    def _names(self):
        if getattr(self, "_name_cache", None) is None or len(self._name_cache) != self.numVariables:
            self._name_cache = [self.variableName(i) for i in range(self.numVariables)]
        return self._name_cache

    def _next_gw(self):
        return min(_week(n) for n in self._names() if n.startswith("transfer_in("))

    def _indices(self, prefixes, weeks=None):
        return np.array([i for i, n in enumerate(self._names()) if n.startswith(prefixes) and (weeks is None or _week(n) in weeks)],
                        dtype=np.int32)

    def _option(self, key):
        value = self.getOptionValue(key)
        return value[1] if isinstance(value, tuple) else value  # highspy 1.15 returns (status, value)

    def _sub_solve(self, secs, gap):
        limit, rel_gap = self._option("time_limit"), self._option("mip_rel_gap")
        self.setOptionValue("time_limit", float(secs))
        self.setOptionValue("mip_rel_gap", float(gap))
        self.phase = "heuristic"
        try:
            super().run()
        finally:
            self.phase = "main"
            self.setOptionValue("time_limit", limit)
            self.setOptionValue("mip_rel_gap", rel_gap)
        ok = self.getModelStatus() in (highspy.HighsModelStatus.kOptimal, highspy.HighsModelStatus.kTimeLimit,
                                       highspy.HighsModelStatus.kInterrupt)
        solution = self.getSolution()
        return np.array(solution.col_value) if ok and solution.value_valid else None

    def relax_fix_start(self, weeks, secs, gap):
        """Plan 1's heuristic start (see module docstring). Returns the partial solution's size, or 0 if none."""
        next_gw = self._next_gw()
        all_weeks = {_week(n) for n in self._names() if n.startswith("transfer_in(")}
        later = self._indices(LINEUP_PREFIXES, {w for w in all_weeks if w >= next_gw + weeks})
        if len(later) == 0:
            return 0
        before = np.array([int(t) for t in self.getLp().integrality_])[later]
        self._set_integrality(later, False)
        try:
            values = self._sub_solve(secs, gap)
        finally:  # back to each column's own integrality (continuous_assignment columns stay continuous)
            self._set_integrality(later[before == 1], True)
        if values is None:
            return 0
        idx = self._indices(DECISION_PREFIXES)
        self.setSolution(len(idx), idx, np.round(values[idx]))
        return len(idx)

    def _set_integrality(self, idx, integer):
        kind = highspy.HighsVarType.kInteger if integer else highspy.HighsVarType.kContinuous
        self.changeColsIntegrality(len(idx), idx, np.full(len(idx), kind))

    def make_assignment_continuous(self):
        idx = self._indices(ASSIGNMENT_PREFIXES)
        self._set_integrality(idx, False)
        return len(idx)

    def round_assignment(self):
        """If a relaxed assignment column ended fractional, re-solve it integral with everything else fixed.
        Returns the number of fractional columns found."""
        idx = self._indices(ASSIGNMENT_PREFIXES)
        solution = self.getSolution()
        if not solution.value_valid:
            return 0
        values = np.array(solution.col_value)
        frac = int(np.sum(np.abs(values[idx] - np.round(values[idx])) > 1e-6))
        if frac == 0:
            return 0
        lp = self.getLp()
        integer = np.array([int(t) for t in lp.integrality_]) == 1
        others = np.setdiff1d(np.where(integer)[0], idx).astype(np.int32)
        lower, upper = np.array(lp.col_lower_)[others], np.array(lp.col_upper_)[others]
        fixed = np.round(values[others])
        self.changeColsBounds(len(others), others, fixed, fixed)
        self._set_integrality(idx, True)
        self.phase = "heuristic"
        try:
            super().run()
        finally:
            self.phase = "main"
            self.changeColsBounds(len(others), others, lower, upper)
            self._set_integrality(idx, False)
        return frac

    def restrict_start(self, weeks, secs, gap):
        """Plan 1's start from the model with no transfers from GW-now + weeks on. Returns its objective or None."""
        next_gw = self._next_gw()
        late = self._indices(("transfer_in(", "tr_out_reg(", "tr_out_first("),
                             {_week(n) for n in self._names() if n.startswith("transfer_in(") and _week(n) >= next_gw + weeks})
        if len(late) == 0:
            return None
        lp = self.getLp()
        lower, upper = np.array(lp.col_lower_)[late], np.array(lp.col_upper_)[late]
        self.changeColsBounds(len(late), late, np.zeros(len(late)), np.zeros(len(late)))
        try:
            values = self._sub_solve(secs, gap)
            objective = self.getInfo().objective_function_value if values is not None else None
        finally:
            self.changeColsBounds(len(late), late, lower, upper)
        if values is None:
            return None
        idx = np.arange(len(values), dtype=np.int32)
        self.setSolution(len(idx), idx, values)
        return objective

    def late_transfer_columns(self, weeks):
        """Transfer columns from GW-now + weeks on, except in weeks whose Wildcard is fixed on."""
        next_gw = self._next_gw()
        lp = self.getLp()
        lower = np.array(lp.col_lower_)
        wildcard = {_week(n) for i, n in enumerate(self._names()) if n.startswith("use_wc(") and lower[i] > 0.5}
        late = {_week(n) for n in self._names() if n.startswith("transfer_in(") and _week(n) >= next_gw + weeks} - wildcard
        return self._indices(("transfer_in(", "tr_out_reg(", "tr_out_first("), late)

    def restrict_transfers(self, weeks):
        late = self.late_transfer_columns(weeks)
        self.changeColsBounds(len(late), late, np.zeros(len(late)), np.zeros(len(late)))
        return len(late)

    def repair_start(self):
        """The previous plan from GW-now + 1 on, as a partial solution. Returns its size."""
        next_gw = self._next_gw()
        names = self._names()
        idx = np.array([i for i in self._indices(SQUAD_PREFIXES + LINEUP_PREFIXES) if _week(names[i]) > next_gw], dtype=np.int32)
        self.setSolution(len(idx), idx, np.round(self._last[idx]))
        return len(idx)

    def _gw_now_columns(self):
        """{player: transfer_in column}, {player: [transfer-out columns]} for GW-now."""
        next_gw = self._next_gw()
        tin, tout = {}, {}
        for i, n in enumerate(self._names()):
            if n.startswith(("transfer_in(", "tr_out_reg(", "tr_out_first(")) and _week(n) == next_gw:
                player = int(n[n.index("(") + 1:-1].split(",")[0])
                if n.startswith("transfer_in("):
                    tin[player] = i
                else:
                    tout.setdefault(player, []).append(i)
        return tin, tout

    def add_exclusions(self, exclusions):
        """The pinned solver's this_gw_transfer_in_out no-good row, once per earlier plan."""
        tin, tout = self._gw_now_columns()
        for moves in exclusions:
            ins, outs = set(moves["in"]), set(moves["out"])
            coef, rhs = {}, 1.0
            for player, col in tin.items():
                coef[col] = -1.0 if player in ins else 1.0
                rhs -= 1.0 if player in ins else 0.0
            for player, cols in tout.items():
                for col in cols:
                    coef[col] = -1.0 if player in outs else 1.0
                rhs -= 1.0 if player in outs else 0.0
            idx = np.array(list(coef), dtype=np.int32)
            self.addRow(rhs, highspy.kHighsInf, len(idx), idx, np.array([coef[i] for i in idx]))
        return len(exclusions)

    def gw_now_moves(self):
        """The current solution's GW-now transfers in and out (player ids), as the pinned solver reads them."""
        values = np.array(self.getSolution().col_value)
        tin, tout = self._gw_now_columns()
        return {"in": sorted(p for p, c in tin.items() if values[c] > 0.5),
                "out": sorted(p for p, cols in tout.items() if sum(values[c] for c in cols) > 0.5)}

    def before_plan(self):
        """Apply the mode's levers before this plan's solve; returns a record of what was done."""
        s = self.settings
        done = {}
        if self.plan_runs == 0 and s.get("exclude_moves"):
            done["exclusions"] = self.add_exclusions(s["exclude_moves"])
        for key, value in s.get("highs", {}).items():
            self.setOptionValue(key, value)
        if self.plan_runs > 0 and s.get("clear"):
            self.clearSolver()
            done["clear"] = True
        if self.plan_runs > 0 and s.get("repair") and self._last is not None:
            done["repair_entries"] = self.repair_start()
        if self.plan_runs == 0 and s.get("continuous_assignment"):
            done["continuous_columns"] = self.make_assignment_continuous()
        if self.plan_runs == 0 and s.get("transfer_weeks"):
            done["banned_transfer_columns"] = self.restrict_transfers(s["transfer_weeks"])
        later = s.get("later_plans")
        if self.plan_runs == 1 and later:  # from plan 2 on (the bounds and options then stay for plan 3)
            if later.get("transfer_weeks"):
                done["banned_transfer_columns"] = self.restrict_transfers(later["transfer_weeks"])
            for option, key in (("mip_rel_gap", "gap"), ("time_limit", "secs")):
                if key in later:
                    self.setOptionValue(option, float(later[key]))
            done["later_plans"] = True
        if self.plan_runs == 0 and s.get("restrict_start"):
            rs = s["restrict_start"]
            done["restrict_start_objective"] = self.restrict_start(rs["weeks"], rs["secs"], rs.get("gap", 0))
        if self.plan_runs == 0 and s.get("relax_fix"):
            rf = s["relax_fix"]
            done["relax_fix_entries"] = self.relax_fix_start(rf["weeks"], rf["secs"], rf.get("gap", 0))
        return done

    def after_plan(self):
        """Returns a record of what was done after the solve."""
        done = {}
        if self.settings.get("continuous_assignment"):
            done["fractional_rounded"] = self.round_assignment()
        self.plan_runs += 1
        solution = self.getSolution()
        if solution.value_valid:
            self._last = np.array(solution.col_value)
        return done

    def run(self):
        # PROGRESS lines are read by the website's solve job (solver_jobs.py) to show which plan is being solved
        print(f"{PROGRESS} plan {self.plan_runs + 1} started", flush=True)
        self.before_plan()
        t0 = time.perf_counter()
        status = super().run()
        info = self.getInfo()
        print(f"{PROGRESS} plan {self.plan_runs + 1} done in {time.perf_counter() - t0:.0f} s: "
              f"{self.modelStatusToString(self.getModelStatus())}, gap {info.mip_gap:.2e}", flush=True)
        self.after_plan()
        return status


def _plan_worker(job):
    """One plan in a fresh process: (results, recorder runs or None, GW-now moves)."""
    import importlib
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    for path in (str(here / "open_fpl_solver"), str(here)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from dev import solver
    module, name = job["cls"].split(":")
    cls = getattr(importlib.import_module(module), name)
    if hasattr(cls, "runs"):
        cls.runs = []
    holder = {}

    class Capturing(cls):
        def after_plan(self):
            done = super().after_plan()
            holder["moves"] = self.gw_now_moves() if self.getSolution().value_valid else None
            return done

    install(solver, settings=job["settings"], cls=Capturing)
    results = solver.solve_multi_period_fpl(job["data"], {**job["options"], "num_iterations": 1})
    return results, getattr(cls, "runs", None), holder.get("moves")


def solve_plans(data, options, settings, cls_path="solve_modes:ModeHighs"):
    """The pinned solver's plan loop with each plan in its own process. Returns (results, recorder runs)."""
    import multiprocessing
    n = options.get("num_iterations", 1)
    results, runs, exclusions = [], [], []
    ctx = multiprocessing.get_context("spawn")
    for k in range(n):
        job = {"cls": cls_path, "settings": {**settings, "exclude_moves": list(exclusions)}, "data": data, "options": options}
        with ctx.Pool(1) as pool:
            plan, plan_runs, moves = pool.apply(_plan_worker, (job,))
        if not plan:
            break
        plan[0]["iter"] = k
        results.extend(plan)
        runs.extend(plan_runs or [])
        if moves is None:
            break
        exclusions.append(moves)
    return results, runs


def uses_processes(settings, options, next_gw):
    """process_per_plan applies unless GW-now is a Free Hit week (the pinned solver then compares lineups)."""
    return bool(settings.get("process_per_plan")) and next_gw not in options.get("use_fh", []) and options.get("num_iterations", 1) > 1


def solver_options(mode):
    return dict(MODES[mode].get("options", {}))


def time_limit(mode):
    """Wall-clock limit for a whole run in this mode: twice the solver's own limits (plans 2-3 at later_plans' limit)
    plus PREP_SECONDS."""
    options = solver_options(mode)
    later = MODES[mode].get("later_plans", {})
    n = options.get("num_iterations", 1)
    return 2 * (options["secs"] + (n - 1) * later.get("secs", options["secs"])) + PREP_SECONDS


def install(solver_module, mode=None, settings=None, cls=ModeHighs):
    """Make the pinned solver build its model with `cls`, configured for `mode` (or explicit `settings`)."""
    cls.settings = settings if settings is not None else MODES[mode]
    solver_module.highspy.Highs = cls
    return cls
