"""Solver speed benchmark (website_plan.md, "Speed: research plan"; §226). Research only: writes nothing that
production reads.

    python -u -X utf8 -m research.optimizer.benchmark_solver build
    python -u -X utf8 -m research.optimizer.benchmark_solver run --instance gw11_h10 --variant base
    python -u -X utf8 -m research.optimizer.benchmark_solver report

build   freezes the test instances under data/optimizer/benchmark/instances/<name>/: bootstrap-static, fixtures,
        projection.csv and team_data.json, so every run sees identical inputs. GW9 and GW10 reuse the projection and
        bootstrap the optimizer saved in data/optimizer/GW09, GW10 (5-week forecasts); GW11 takes live bootstrap-static
        and the current predicted_points.csv (10 weeks). The squad is team_data.json's, with selling prices recomputed
        against each instance's prices (FPL's rule), so it is a fixed test state, not anyone's live team.
run     solves one instance with one settings variant, in this process, through the pinned solver unchanged. A
        highspy.Highs subclass records each improving solution and the dual bound about once a second, and the time
        of each of the plan iterations. Every plan is replayed by replay_plans.py's checks (chips included) and its
        replayed objective and first-week moves are stored. Results go to data/optimizer/benchmark/results/.
report  prints the table of all results against each instance's base run.
export  writes one instance's plan-1 model, exactly as HiGHS would solve it, to data/optimizer/benchmark/models/<name>.mps
        (for other solvers, e.g. the CP-SAT test, §227), and stops before solving.

Variants change one lever at a time from run_optimizer.OVERRIDES plus the Deep mode (BASE_MODE).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import highspy
import numpy as np
import pandas as pd
import psutil

from fpl_model import paths

_HERE = Path(__file__).resolve().parent
_ROOT = paths.ROOT
sys.path.insert(0, str(_HERE / "open_fpl_solver"))

from dev import solver  # noqa: E402

from fpl_model.optimizer import chips as chip_rules  # noqa: E402
from fpl_model.optimizer import (
    replay_plans,  # noqa: E402
    run_optimizer,  # noqa: E402
    solve_modes,  # noqa: E402
)

BENCH = paths.OPTIMIZER / "benchmark"
INSTANCES = paths.OPTIMIZER / "benchmark" / "instances"
RESULTS = paths.OPTIMIZER / "benchmark" / "results"
SAMPLE_SECONDS = 1.0
BASE_MODE = "deep"  # variants change one lever from this mode's settings (it was run_optimizer.OVERRIDES' until §227)
WARM_START_SECONDS = 60

# name -> (source, horizon, chips). Source "GW09"/"GW10" = that optimizer folder; "live" = current forecast + live prices.
# The two chip scenarios: a Free Hit the user could play (first-half Free Hit available), and every other chip type
# in the GW11 window, which this squad has already played (a solver and replay test, not a plan).
INSTANCE_SPECS = {
    "gw09_h5": ("GW09", 5, {}),
    "gw10_h5": ("GW10", 5, {}),
    "gw11_h5": ("live", 5, {}),
    "gw11_h10": ("live", 10, {}),
    "gw11_h10_fh": ("live", 10, {"fh": 14}),
    "gw11_h10_chips": ("live", 10, {"wc": 12, "tc": 15, "bb": 17}),
}

VARIANTS = {
    "base": {},
    "gap0001": {"gap": 0.001},
    "gap0005": {"gap": 0.005},
    "pool_tight": {"ev_per_price_cutoff": 50, "keep_top_ev_percent": 2},
    "warm": {"_warm_start": True},
    "heur03": {"_settings": {"highs": {"mip_heuristic_effort": 0.3}}},
    "threads1": {"_settings": {"highs": {"threads": 1}}},
    "nosample": {"_no_sample": True},  # the recorder's own overhead: improving solutions only, no bound sampling
    # §227 levers; each is a solve_modes.ModeHighs setting (see that module's docstring)
    "log": {"_log": True},  # base, with HiGHS's own log written to logs/<instance>_<variant>_highs.log
    "clear": {"_settings": {"clear": True}},
    "repair": {"_settings": {"repair": True}},
    "clear_repair": {"_settings": {"clear": True, "repair": True}},
    # reliability branching: strong-branch each candidate until it has this many pseudocost observations (HiGHS default 8)
    "psc0": {"_settings": {"highs": {"mip_pscost_minreliable": 0}}},
    "psc2": {"_settings": {"highs": {"mip_pscost_minreliable": 2}}},
    "psc4": {"_settings": {"highs": {"mip_pscost_minreliable": 4}}},
    "proc": {"_settings": {"process_per_plan": True}},  # each plan in its own process (memory, §227)
    "ftm": {"ft_tight_big_m": True},  # ft_tight_m.patch: smallest valid big-M in each FT row (§227)
    "cassign": {"_settings": {"continuous_assignment": True}},
    "cutpool2k": {"_settings": {"highs": {"mip_pool_soft_limit": 2000}}},  # HiGHS default 10000 (memory, §227)
    # Fast / Standard candidates: transfers only in the first weeks (upstream's no_transfer_last_gws); the replay
    # scores the plan in the full 10-week model, so the loss against Deep is measured, not assumed (§227)
    "nt5_1p": {"no_transfer_last_gws": 5, "num_iterations": 1, "gap": 0.0005, "secs": 120},
    "nt5_1p_g0": {"no_transfer_last_gws": 5, "num_iterations": 1, "secs": 300},
    "nt3": {"no_transfer_last_gws": 3},
    "nt2": {"no_transfer_last_gws": 2},
    # the same restriction as column bounds (solve_modes transfer_weeks)
    "tw5_1p": {"num_iterations": 1, "gap": 0.0005, "secs": 60, "_settings": {"transfer_weeks": 5}},
    "tw4_1p": {"num_iterations": 1, "gap": 0.0005, "secs": 60, "_settings": {"transfer_weeks": 4}},
    "tw3_1p": {"num_iterations": 1, "gap": 0.0005, "secs": 60, "_settings": {"transfer_weeks": 3}},
    "tw7": {"_settings": {"transfer_weeks": 7}},
    "tw8": {"_settings": {"transfer_weeks": 8}},
    "std_a": {"_settings": {"later_plans": {"transfer_weeks": 5, "gap": 0.0005, "secs": 60}}},
    "std_b": {"_settings": {"later_plans": {"transfer_weeks": 4, "gap": 0.0005, "secs": 60}}},
    "rs5": {"_settings": {"restrict_start": {"weeks": 5, "secs": 120, "gap": 0.0005}}},
    "rs3": {"_settings": {"restrict_start": {"weeks": 3, "secs": 60, "gap": 0.0005}}},
    "rf4": {"_settings": {"relax_fix": {"weeks": 4, "secs": 120, "gap": 0.001}}},
    "rf4_repair": {"_settings": {"relax_fix": {"weeks": 4, "secs": 120, "gap": 0.001}, "repair": True}},
}
SEED_SEP = "@"  # "<variant>@<seed>": the variant with options["random_seed"] = seed, saved as results/<instance>/<variant>@<seed>


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def selling_price(purchase, cost):
    return purchase + (cost - purchase) // 2 if cost > purchase else cost


def frozen_team(fpl):
    """team_data.json with selling prices recomputed against this instance's prices."""
    team = json.loads((_ROOT / "team_data.json").read_text(encoding="utf-8-sig"))
    costs = {e["id"]: e["now_cost"] for e in fpl["elements"]}
    for pick in team["picks"]:
        pick["selling_price"] = selling_price(pick["purchase_price"], costs[pick["element"]])
    return team


def build():
    live_fpl = run_optimizer.fetch(f"{run_optimizer.BASE_URL}/bootstrap-static/")
    fixtures = run_optimizer.fetch(f"{run_optimizer.BASE_URL}/fixtures/")
    next_gw = next(e["id"] for e in live_fpl["events"] if e["is_next"])
    for name, (source, horizon, chip_plan) in INSTANCE_SPECS.items():
        folder = INSTANCES / name
        folder.mkdir(parents=True, exist_ok=True)
        if source == "live":
            fpl = live_fpl
            gws = list(range(next_gw, next_gw + horizon))
            projection = run_optimizer.build_projection(fpl, gws)
        else:
            src = _ROOT / "data" / "optimizer" / source
            fpl = json.loads((src / "bootstrap_static.json").read_text(encoding="utf-8"))
            projection = pd.read_csv(src / "projection.csv", encoding="utf-8")
            gws = sorted(int(c.split("_")[0]) for c in projection.columns if c.endswith("_Pts"))
            assert len(gws) == horizon, (name, gws)
        team = frozen_team(fpl)
        assert {p["element"] for p in team["picks"]} <= set(projection["ID"]), name
        (folder / "bootstrap_static.json").write_text(json.dumps(fpl), encoding="utf-8")
        (folder / "fixtures.json").write_text(json.dumps(fixtures), encoding="utf-8")
        (folder / "team_data.json").write_text(json.dumps(team, indent=1), encoding="utf-8")
        projection.to_csv(folder / "projection.csv", index=False, encoding="utf-8")
        write_json(folder / "instance.json", {"source": source, "gws": gws, "chips": chip_rules.normalise(chip_plan),
                                              "built_utc": pd.Timestamp.utcnow().isoformat()})
        print(f"{name}: GW{gws[0]}-{gws[-1]}, {len(projection)} players, {chip_rules.describe(chip_plan)}")


class _Exported(Exception):
    pass


class Recorder(solve_modes.ModeHighs):
    """ModeHighs that logs, per plan: wall time, improving solutions, the dual bound and peak memory. Solutions found
    inside a lever's own sub-solve (phase "heuristic") are logged separately, never as the plan's incumbents."""

    warm_start = False
    sample = True
    export_path = None
    log_file = None
    runs = []

    def __init__(self):
        super().__init__()
        self._sampled = 0.0
        self.cbMipImprovingSolution.subscribe(self._on_improve)
        if self.sample:
            self.cbMipInterrupt.subscribe(self._on_interrupt)

    def _point(self, e):
        d = e.data_out
        return [round(time.perf_counter() - self._t0, 3), d.mip_primal_bound, d.mip_dual_bound]

    def _on_improve(self, e):
        key = "improving" if self.phase == "main" else "heuristic_improving"
        self.runs[-1].setdefault(key, []).append(self._point(e))

    def _on_interrupt(self, e):
        now = time.perf_counter()
        if now - self._sampled >= SAMPLE_SECONDS:
            self._sampled = now
            if self.phase == "main":
                self.runs[-1]["bound"].append(self._point(e))
            self._rss()

    def _rss(self):
        """Peak resident memory of this process during the run, MB (the 16 GB PC ran short once)."""
        mb = round(psutil.Process().memory_info().rss / 2**20)
        self.runs[-1]["peak_rss_mb"] = max(self.runs[-1]["peak_rss_mb"], mb)

    def _warm_start(self):
        """A feasible start: the same model with every transfer in fixed off (the current squad kept; a fixed Free Hit
        week still picks its own squad), solved for at most WARM_START_SECONDS; its solution goes to setSolution."""
        fixed = self._indices(("transfer_in(",))
        lp = self.getLp()
        lower, upper = np.array(lp.col_lower_)[fixed], np.array(lp.col_upper_)[fixed]
        self.changeColsBounds(len(fixed), fixed, np.zeros(len(fixed)), np.zeros(len(fixed)))
        values = self._sub_solve(WARM_START_SECONDS, self._option("mip_rel_gap"))
        self.changeColsBounds(len(fixed), fixed, lower, upper)
        if values is not None:
            idx = np.arange(len(values), dtype=np.int32)
            self.setSolution(len(idx), idx, values)
        return values is not None

    def run(self):
        if self.log_file is not None and not self.runs:
            self.setOptionValue("log_file", str(self.log_file))
            self.setOptionValue("output_flag", True)
        if self.export_path is not None:
            self.writeModel(str(self.export_path))
            raise _Exported
        self._t0 = time.perf_counter()
        self.runs.append({"improving": [], "bound": [], "warm": None, "peak_rss_mb": 0})
        if self.warm_start and len(self.runs) == 1:
            self.runs[-1]["warm"] = self._warm_start()
        done = self.before_plan()
        if done:
            self.runs[-1]["levers"] = {**done, "seconds": round(time.perf_counter() - self._t0, 3)}
        status = solve_modes.BaseHighs.run(self)
        after = self.after_plan()
        if after:
            self.runs[-1]["after"] = after
        self._rss()
        info = self.getInfo()
        self.runs[-1].update(seconds=round(time.perf_counter() - self._t0, 3), status=self.modelStatusToString(self.getModelStatus()),
                             objective=info.objective_function_value, dual_bound=info.mip_dual_bound, gap=info.mip_gap)
        return status


def split_variant(name):
    """'clear@2' -> ('clear', 2); 'clear' -> ('clear', None)."""
    if SEED_SEP in name:
        variant, seed = name.split(SEED_SEP)
        return variant, int(seed)
    return name, None


def instance_options(spec, horizon, variant):
    with open(_HERE / "open_fpl_solver" / "comprehensive_settings.json") as f:
        options = {**json.load(f), **run_optimizer.OVERRIDES, **solve_modes.solver_options(BASE_MODE)}
    options["horizon"] = horizon
    options["xmin_lb"] = round(run_optimizer.XMIN_LB_PER_GW * horizon)
    options.update(chip_rules.solver_options(spec["chips"]))
    options.update({k: v for k, v in VARIANTS[variant].items() if not k.startswith("_")})
    return options


def run(instance, variant_name, iterations=None, secs=None):
    variant, seed = split_variant(variant_name)
    folder = INSTANCES / instance
    spec = json.loads((folder / "instance.json").read_text(encoding="utf-8"))
    fpl = json.loads((folder / "bootstrap_static.json").read_text(encoding="utf-8"))
    fixtures = json.loads((folder / "fixtures.json").read_text(encoding="utf-8"))
    team = json.loads((folder / "team_data.json").read_text(encoding="utf-8"))
    projection = pd.read_csv(folder / "projection.csv", encoding="utf-8")
    options = instance_options(spec, len(spec["gws"]), variant)
    if iterations:
        options["num_iterations"] = iterations
    if secs:
        options["secs"] = secs
    if seed is not None:
        options["random_seed"] = seed

    responses = {f"{run_optimizer.BASE_URL}/bootstrap-static/": fpl, f"{run_optimizer.BASE_URL}/fixtures/": fixtures}
    solver.cached_request = responses.__getitem__
    solver.read_data = lambda _options: projection.copy()
    Recorder.runs = []
    Recorder.warm_start = bool(VARIANTS[variant].get("_warm_start"))
    Recorder.sample = not VARIANTS[variant].get("_no_sample")
    Recorder.log_file = BENCH / "logs" / f"{instance}_{variant_name}_highs.log" if VARIANTS[variant].get("_log") else None
    solve_modes.install(solver, settings=VARIANTS[variant].get("_settings", {}), cls=Recorder)  # the pinned solver builds its model with highspy.Highs()

    t0 = time.perf_counter()
    data = solver.prep_data(team, options)
    settings = VARIANTS[variant].get("_settings", {})
    if solve_modes.uses_processes(settings, options, spec["gws"][0]):
        results, Recorder.runs = solve_modes.solve_plans(data, options, settings, "fpl_model.optimizer.benchmark_solver:Recorder")
    else:
        results = solver.solve_multi_period_fpl(data, options)
    total = time.perf_counter() - t0

    out = RESULTS / instance / variant_name
    out.mkdir(parents=True, exist_ok=True)
    inputs = replay_plans.load_inputs(folder, team_file=folder / "team_data.json", check_source=False)
    for n, result in enumerate(results, start=1):  # every plan saved before any replay, so a replay bug loses no solve
        result["picks"].to_csv(out / f"plan_{n}.csv", index=False, encoding="utf-8")
    write_json(out / "solve_log.json", {"total_seconds": round(total, 2), "runs": Recorder.runs,
                                         "scores": [r["score"] for r in results]})
    plans = []
    for n, (result, rec) in enumerate(zip(results, Recorder.runs, strict=False), start=1):
        plan = replay_plans.replay_plan(out, n, inputs, spec["chips"])
        summary = {"plan": n, **{k: result[k] for k in ("score", "solve_status", "mip_gap", "ft_end")}}
        replay_plans.check_against_solver(plan, summary, inputs, spec["chips"], require_optimal=False)
        first = plan["weeks"][0]
        plans.append({**rec, "score": result["score"], "replay_objective": plan["objective"],
                      "gw_now": {"in": sorted(first["in"]), "out": sorted(first["out"]), "captain": first["captain"]},
                      "first_incumbent_s": rec["improving"][0][0] if rec["improving"] else None})
    record = {"instance": instance, "variant": variant_name, "changes": VARIANTS[variant], "seed": options.get("random_seed", 0),
              "gws": spec["gws"],
              "chips": spec["chips"], "pool_size": len(data["merged_data"]), "pool_ids": sorted(int(i) for i in data["merged_data"].index),
              "num_iterations": options["num_iterations"], "secs": options["secs"], "total_seconds": round(total, 2),
              "highspy": getattr(highspy, "__version__", None) or highspy.Highs().version(), "plans": plans}
    write_json(out / "result.json", record)
    for i, p in enumerate(plans, start=1):
        print(f"{instance} {variant_name} plan {i}: {p['seconds']:.1f}s {p['status']}, gap {p['gap']:.2e}, replay {p['replay_objective']:.4f}, "
              f"first incumbent {p['first_incumbent_s']}s, GW-now in {p['gw_now']['in']} out {p['gw_now']['out']}")
    print(f"total {total:.1f}s, pool {record['pool_size']}")
    return record


def export(instance):
    """The plan-1 model as MPS: the run below, stopped at its first run() call."""
    path = BENCH / "models" / f"{instance}.mps"
    path.parent.mkdir(parents=True, exist_ok=True)
    Recorder.export_path = path
    try:
        run(instance, "base")
    except _Exported:
        print(f"wrote {path}")
    finally:
        Recorder.export_path = None


def report():
    rows = []
    for path in sorted(RESULTS.glob("*/*/result.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        base_path = RESULTS / r["instance"] / "base" / "result.json"
        base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else None
        p1 = r["plans"][0]
        row = {"instance": r["instance"], "variant": r["variant"], "pool": r["pool_size"], "total_s": r["total_seconds"],
               "plan1_s": p1["seconds"], "first_s": p1["first_incumbent_s"], "status": p1["status"],
               "obj1": round(p1["replay_objective"], 4), "peak_mb": max(p.get("peak_rss_mb", 0) for p in r["plans"])}
        if base:
            b1 = base["plans"][0]
            row["d_obj1"] = round(p1["replay_objective"] - b1["replay_objective"], 4)
            row["same_gw_now"] = p1["gw_now"]["in"] == b1["gw_now"]["in"] and p1["gw_now"]["out"] == b1["gw_now"]["out"]
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def queue(pairs):
    """Run instance:variant pairs one after another, each in its own process (so no two solves share the CPU),
    skipping any that already has a result."""
    import subprocess
    logs = BENCH / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        instance, variant = pair.split(":")
        if (RESULTS / instance / variant / "result.json").exists():
            print(f"{pair}: done already")
            continue
        print(f"{pair}: running", flush=True)
        with open(logs / f"{instance}_{variant}.log", "w", encoding="utf-8") as log:
            code = subprocess.call([sys.executable, "-u", "-X", "utf8", "-W", "ignore", "-m", "research.optimizer.benchmark_solver", "run", "--instance", instance,
                                    "--variant", variant], stdout=log, stderr=subprocess.STDOUT)
        print(f"{pair}: exit {code}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--instance", required=True, choices=list(INSTANCE_SPECS))
    r.add_argument("--variant", default="base", help="a VARIANTS name, optionally @<random_seed>")
    r.add_argument("--iterations", type=int)
    r.add_argument("--secs", type=int)
    sub.add_parser("report")
    e = sub.add_parser("export")
    e.add_argument("--instance", required=True, choices=list(INSTANCE_SPECS))
    q = sub.add_parser("queue")
    q.add_argument("pairs", nargs="+", help="instance:variant")
    args = parser.parse_args()
    if args.cmd == "run" and split_variant(args.variant)[0] not in VARIANTS:
        parser.error(f"unknown variant {args.variant}")
    if args.cmd == "build":
        build()
    elif args.cmd == "run":
        run(args.instance, args.variant, args.iterations, args.secs)
    elif args.cmd == "export":
        export(args.instance)
    elif args.cmd == "queue":
        queue(args.pairs)
    else:
        report()


if __name__ == "__main__":
    main()
