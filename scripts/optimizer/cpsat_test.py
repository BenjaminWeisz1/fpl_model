"""CP-SAT feasibility test (SOLVER_SPEED_PLAN.md task 1, §227). Research only: writes nothing production reads.

    python -u -X utf8 scripts/optimizer/cpsat_test.py --instance gw11_h10 --workers 8 --seed 0 --secs 1800

Reads the plan-1 model exported by `benchmark_solver.py export` (exactly what HiGHS solves) and converts it to an
integer CP-SAT model by hand, so nothing depends on OR-Tools' floating-point MIP import:
- the only continuous columns are `itb*` (money in the bank, £0.1m steps); they become integers in £0.1m units;
- every row is multiplied by the smallest power of ten that makes its coefficients and bounds integral (checked);
- the objective is rounded at OBJ_SCALE. Each solution is re-scored with the exact float objective, and that exact
  score is what is compared with HiGHS's plan-1 objective (read from results/<instance>/base/result.json).
A watchdog stops the search if this process passes MEMORY_CAP_MB or the PC's free memory drops under MIN_FREE_MB.
Writes data/optimizer/benchmark/cpsat/<instance>/w<workers>_s<seed>.json.
"""
import argparse
import json
import math
import threading
import time
from pathlib import Path

import highspy
import numpy as np
import psutil
from ortools.sat.python import cp_model

_HERE = Path(__file__).resolve().parent
BENCH = _HERE.parents[1] / "data" / "optimizer" / "benchmark"
OBJ_SCALE = 10**4
ITB_STEP = 10  # itb in £0.1m units
ITB_UB = 2000  # £200m: far above any bank
SQUAD_SIZE = 15  # pt (paid transfers in a week) has no upper bound in the model; a week has at most 15 transfers in
MAX_ROW_EXP = 4
MEMORY_CAP_MB = 6000
MIN_FREE_MB = 1500


def load(path):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(str(path))
    lp = h.getLp()
    names = [h.variableName(i) for i in range(lp.num_col_)]
    return h, lp, names


def integral(values, tol=1e-7):
    return all(abs(v - round(v)) < tol for v in values)


def build(lp, names):
    integ = list(lp.integrality_)
    cont = [i for i in range(lp.num_col_) if int(integ[i]) == 0]
    assert all(names[i].startswith("itb") for i in cont), [names[i] for i in cont]
    col_scale = np.ones(lp.num_col_)
    col_scale[cont] = 1.0 / ITB_STEP  # original value = integer value / 10
    model = cp_model.CpModel()
    lo, up = np.array(lp.col_lower_), np.array(lp.col_upper_)
    x = []
    for i in range(lp.num_col_):
        if i in cont:
            lb = 0 if lo[i] <= 0 else math.ceil(lo[i] * ITB_STEP - 1e-9)
            ub = ITB_UB if not np.isfinite(up[i]) else math.floor(up[i] * ITB_STEP + 1e-9)
        else:
            if not np.isfinite(up[i]):
                assert names[i].startswith("pt"), names[i]
                up[i] = SQUAD_SIZE
            lb, ub = math.ceil(lo[i] - 1e-9), math.floor(up[i] + 1e-9)
        x.append(model.NewIntVar(lb, ub, names[i]))
    # rows, from the column-wise matrix
    a = lp.a_matrix_
    start, index, value = list(a.start_), list(a.index_), list(a.value_)
    rows = [[] for _ in range(lp.num_row_)]
    for j in range(lp.num_col_):
        for k in range(start[j], start[j + 1]):
            rows[index[k]].append((j, value[k] * col_scale[j]))
    rlo, rup = list(lp.row_lower_), list(lp.row_upper_)
    exps = []
    for r, terms in enumerate(rows):
        bounds = [b for b in (rlo[r], rup[r]) if np.isfinite(b)]
        for e in range(MAX_ROW_EXP + 1):
            f = 10**e
            if integral([c * f for _, c in terms]) and integral([b * f for b in bounds]):
                break
        else:
            raise ValueError(f"row {r} not integral at 10^{MAX_ROW_EXP}: {terms[:5]}")
        exps.append(e)
        expr = sum(round(c * f) * x[j] for j, c in terms)
        lb = round(rlo[r] * f) if np.isfinite(rlo[r]) else None
        ub = round(rup[r] * f) if np.isfinite(rup[r]) else None
        if lb is not None and ub is not None and lb == ub:
            model.Add(expr == lb)
        else:
            if lb is not None:
                model.Add(expr >= lb)
            if ub is not None:
                model.Add(expr <= ub)
    cost = np.array(lp.col_cost_) * col_scale
    sense = 1 if lp.sense_ == highspy.ObjSense.kMinimize else -1
    obj_terms = [(j, round(cost[j] * OBJ_SCALE)) for j in range(lp.num_col_) if cost[j] != 0]
    expr = sum(c * x[j] for j, c in obj_terms)
    if sense == 1:
        model.Minimize(expr)
    else:
        model.Maximize(expr)
    rounding = max(abs(cost[j] * OBJ_SCALE - c) for j, c in obj_terms) / OBJ_SCALE
    return model, x, cost, {"row_scale_exponents": {str(e): exps.count(e) for e in sorted(set(exps))},
                           "objective_terms": len(obj_terms), "max_coef_rounding": rounding, "offset": lp.offset_}


class Watcher(cp_model.CpSolverSolutionCallback):
    def __init__(self, x, cost, offset, t0):
        super().__init__()
        self.x, self.cost, self.offset, self.t0 = x, cost, offset, t0
        self.solutions = []

    def on_solution_callback(self):
        vals = np.array([self.Value(v) for v in self.x], dtype=float)
        exact = float(vals @ self.cost) + self.offset
        self.solutions.append([round(time.perf_counter() - self.t0, 3), exact, self.ObjectiveValue() / OBJ_SCALE,
                               self.BestObjectiveBound() / OBJ_SCALE])
        self.values = vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="gw11_h10")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--secs", type=float, default=1800)
    args = p.parse_args()
    out = BENCH / "cpsat" / args.instance / f"w{args.workers}_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    base = json.loads((BENCH / "results" / args.instance / "base" / "result.json").read_text(encoding="utf-8"))
    target = base["plans"][0]["objective"]
    target_s = base["plans"][0]["seconds"]

    t_build = time.perf_counter()
    _, lp, names = load(BENCH / "models" / f"{args.instance}.mps")
    model, x, cost, info = build(lp, names)
    build_s = time.perf_counter() - t_build
    print(f"built in {build_s:.1f}s: {info}", flush=True)

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = args.workers
    solver.parameters.random_seed = args.seed
    solver.parameters.max_time_in_seconds = args.secs
    solver.parameters.log_search_progress = True
    solver.parameters.log_to_stdout = False  # the callback prints each line once
    log_lines = []
    solver.log_callback = lambda line: (log_lines.append(line), print(line, flush=True))
    proc = psutil.Process()
    peak = {"mb": 0, "stopped_for_memory": False}
    done = threading.Event()

    def watchdog():
        while not done.wait(1.0):
            mb = proc.memory_info().rss / 2**20
            peak["mb"] = max(peak["mb"], mb)
            if mb > MEMORY_CAP_MB or psutil.virtual_memory().available / 2**20 < MIN_FREE_MB:
                peak["stopped_for_memory"] = True
                solver.StopSearch()
                return

    t0 = time.perf_counter()
    cb = Watcher(x, cost, info["offset"], t0)
    threading.Thread(target=watchdog, daemon=True).start()
    status = solver.Solve(model, cb)
    seconds = time.perf_counter() - t0
    done.set()
    peak_mb = max(peak["mb"], getattr(proc.memory_info(), "peak_wset", 0) / 2**20)

    sols = cb.solutions
    first = sols[0][0] if sols else None
    within = next((s[0] for s in sols if s[1] >= target - 0.1), None)
    hit = next((s[0] for s in sols if s[1] >= target - 1e-6), None)
    best = max((s[1] for s in sols), default=None)
    bound = solver.BestObjectiveBound() / OBJ_SCALE
    record = {"instance": args.instance, "workers": args.workers, "seed": args.seed, "secs": args.secs,
              "status": solver.StatusName(status), "seconds": round(seconds, 2), "build_seconds": round(build_s, 2),
              "highs_target": target, "highs_plan1_seconds": target_s, "best_exact": best,
              "best_minus_target": None if best is None else best - target, "final_bound": bound,
              "final_gap_points": None if best is None else bound - best, "first_solution_s": first,
              "within_0.1_s": within, "target_s": hit, "peak_rss_mb": round(peak_mb),
              "stopped_for_memory": peak["stopped_for_memory"], "model": info, "solutions": sols,
              "log_tail": log_lines[-60:]}
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k not in ("solutions", "log_tail")}, indent=1))


if __name__ == "__main__":
    main()
