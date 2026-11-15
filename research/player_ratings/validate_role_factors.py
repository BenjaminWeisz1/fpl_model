"""Step 3 of the position-switch plan: fit the role factor k.

Protocol, frozen before any role-factor run: data/validation/role_factors_20261101/protocol.json.
Phases, in order:  rates  ->  grid  ->  select  ->  final  ->  confirm
`select` reads 2018-2022 only. `confirm` reads 2023-2024. Nothing here reads 2025-26 errors.
Logs and ratings go to the evidence folder; production files are never written.
"""
from fpl_model import paths
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.player_ratings.role_shares import LINE_RATES_FILE, LINES, load_starts, role_table

ROOT = paths.ROOT
OUT = paths.VALIDATION / "role_factors_20261101"
BUILD = "fpl_model.player_ratings.build_player_ratings"
BASE_LOG = paths.VALIDATION / "role_move_error_20261101" / "prematch_log.csv"
GRID = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1]
KEYS = ["Season", "Team", "Opponent", "Venue", "Player"]
ENV = {"npxG": ("GOAL_MODEL_ROLE_K_NPG_UP", "GOAL_MODEL_ROLE_K_NPG_DOWN"),
       "xAG": ("GOAL_MODEL_ROLE_K_AST_UP", "GOAL_MODEL_ROLE_K_AST_DOWN")}
COLS = {"npxG": ("npxg90", "pred_goals_90_h", "M1_npg"), "xAG": ("xag90", "pred_ast_90_h", "M1_ast")}
N_BOOT, SEED = 5000, 172


def rates():
    log = pd.read_csv(BASE_LOG, encoding="utf-8-sig", low_memory=False)
    t = role_table()
    k = list(zip(log.Season, log.Team, log.Opponent, log.Venue, log.Player))
    log["Line"] = [t[x][1] if x in t else -1 for x in k]
    d = log[log.Season.between(2018, 2022) & (log.Line >= 0)]
    out = {}
    for stat, col in [("npxG", "npxg90"), ("xAG", "xag90")]:
        v = (d[col] * d.weight).groupby(d.Line).sum() / d.weight.groupby(d.Line).sum()
        out[stat] = [round(float(v[i]), 5) for i in range(len(LINES))]
    out["lines"] = LINES
    LINE_RATES_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out)


def run(tag, ks):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", GOAL_MODEL_PREMATCH_LOG=str(OUT / f"log_{tag}.csv"), **ks)
    return subprocess.Popen([sys.executable, "-u", "-X", "utf8", "-m", BUILD], env=env,
                            stdout=open(OUT / f"run_{tag}.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT)


def grid():
    all_env = [v for pair in ENV.values() for v in pair]
    for i in range(0, len(GRID), 4):
        procs = [(k, run(f"k{k}", {v: str(k) for v in all_env})) for k in GRID[i:i + 4]]
        for k, p in procs:
            if p.wait() != 0:
                raise SystemExit(f"run k={k} failed; see run_k{k}.log")
            print("done k", k, flush=True)


def load(tag):
    d = pd.read_csv(OUT / f"log_{tag}.csv", encoding="utf-8-sig", low_memory=False)
    t = role_table_cached()
    k = list(zip(d.Season, d.Team, d.Opponent, d.Venue, d.Player))
    disp = []
    for x in k:
        rec = t.get(x)
        disp.append(rec is not None and rec[4] >= 10 and int(rec[0].argmax()) != rec[3])
    d["displaced"] = disp
    for stat, (a, p, _) in COLS.items():
        d[f"se_{stat}"] = ((d[a] - d[p]) * d.weight) ** 2
    return d


_T = {}


def role_table_cached():
    if "t" not in _T:
        _T["t"] = role_table()
    return _T["t"]


def select():
    res, rows = {}, []
    logs = {k: load(f"k{k}") for k in GRID}
    for stat, (_, _, m1) in COLS.items():
        for name, sign in [("up", 1), ("down", -1)]:
            mses = {}
            for k, d in logs.items():
                f = d[d.Season.between(2018, 2022) & d.displaced & (np.sign(d[m1] - 1) == sign)]
                mses[k] = float(f[f"se_{stat}"].mean())
                n = len(f)
            best = min(mses, key=mses.get)
            res[f"{stat}_{name}"] = best
            print(stat, name, "rows", n, "best k", best, {k: round(v / mses[0] - 1, 5) for k, v in mses.items()})
            rows.append(dict(stat=stat, direction=name, rows=n, best=best, mse_vs_0={str(k): round(v / mses[0] - 1, 5) for k, v in mses.items()}))
    (OUT / "chosen.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (OUT / "selection.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def final():
    c = json.loads((OUT / "chosen.json").read_text(encoding="utf-8"))
    ks = {ENV["npxG"][0]: c["npxG_up"], ENV["npxG"][1]: c["npxG_down"], ENV["xAG"][0]: c["xAG_up"], ENV["xAG"][1]: c["xAG_down"]}
    p = run("final", {k: str(v) for k, v in ks.items()})
    if p.wait() != 0:
        raise SystemExit("final run failed")


def cluster_ci(a, b, pl, seed):
    """Relative MSE change of b v a, player-cluster interval."""
    u, inv = np.unique(pl, return_inverse=True)
    ta, tb = np.bincount(inv, a), np.bincount(inv, b)
    ix = np.random.default_rng(seed).integers(0, len(u), size=(N_BOOT, len(u)))
    v = tb[ix].sum(1) / ta[ix].sum(1) - 1
    return float(b.sum() / a.sum() - 1), [round(float(x), 5) for x in np.percentile(v, [2.5, 97.5])]


def confirm():
    c = json.loads((OUT / "chosen.json").read_text(encoding="utf-8"))
    base, fin = load("k0"), load("final")
    m = base.merge(fin, on=KEYS, suffixes=("_b", "_f"), validate="one_to_one")
    m = m[m.Season.between(2023, 2024)]
    res = dict(chosen=c, rows=len(m), displaced=int(m.displaced_b.sum()), players=int(m[m.displaced_b].Player.nunique()))
    for stat in COLS:
        m1 = COLS[stat][2] + "_b"
        d = m[m.displaced_b]
        ch, ci = cluster_ci(d[f"se_{stat}_b"].values, d[f"se_{stat}_f"].values, d.Player.values, SEED)
        out = dict(displaced=round(ch, 5), ci=ci)
        for name, sign in [("up", 1), ("down", -1)]:
            f = d[np.sign(d[m1] - 1) == sign]
            x, xci = cluster_ci(f[f"se_{stat}_b"].values, f[f"se_{stat}_f"].values, f.Player.values, SEED)
            out[name] = dict(rows=len(f), change=round(x, 5), ci=xci)
        out["all_rows"] = round(float(m[f"se_{stat}_f"].sum() / m[f"se_{stat}_b"].sum() - 1), 6)
        off = c[f"{stat}_up"] == 0 and c[f"{stat}_down"] == 0
        out["pass"] = bool(not off and ci[1] < 0 and out["up"]["change"] <= .005 and out["down"]["change"] <= .005
                           and out["all_rows"] <= .0005)
        res[stat] = out
        print(stat, json.dumps(out))
    res["verdict"] = "PASS" if res["npxG"]["pass"] or res["xAG"]["pass"] else "STOP"
    print("GATE", res["verdict"])
    (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    {"rates": rates, "grid": grid, "select": select, "final": final, "confirm": confirm}[sys.argv[1]]()
