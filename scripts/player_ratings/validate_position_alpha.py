"""
Goal/assist loop: learning-rate floor by position (CHANGELOG §196).
Protocol, frozen before scoring: data/validation/position_alpha_20261103/protocol.json.

    python validate_position_alpha.py run     # one pre-match log per floor (GOAL_MODEL_ALPHA); production untouched
    python validate_position_alpha.py score   # per-(group, target) floor on 2018-2023, test 2024-2026; results.json
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "current"
OUT = ROOT / "data" / "validation" / "position_alpha_20261103"
OOT_PRIORS = ROOT / "data" / "validation" / "new_player_priors_20261028" / "oot_priors.csv"
GRID = ["0", "0.0083", "0.0125", "0.025", "0.05"]
BASE = "0.025"
TARGETS = {"npxG": ("npxg90", "pred_goals_90"), "xAG": ("xag90", "pred_ast_90")}
GROUP = {"CB": "CB", "FB": "FB", "WB": "FB", "DM": "CM", "CM": "CM", "AM": "AW", "W": "AW", "WM": "AW", "FW": "FW"}


def log_path(a):
    return OUT / f"log_a{a}.csv"


def run_one(a):
    env = dict(os.environ, GOAL_MODEL_ALPHA=a, GOAL_MODEL_PREMATCH_LOG=str(log_path(a)),
               GOAL_MODEL_NEW_PRIORS_FILE=str(OOT_PRIORS), PYTHONHASHSEED="0")
    with open(OUT / f"log_a{a}.run.log", "w", encoding="utf-8") as fh:
        r = subprocess.run([sys.executable, "-u", "-X", "utf8", str(ROOT / "scripts" / "player_ratings" / "build_player_ratings.py")],
                           env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"floor {a} failed, see log_a{a}.run.log")
    return a


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as ex:
        for a in ex.map(run_one, GRID):
            print(f"floor {a} done", flush=True)


def groups(log):
    """Group per row from the latest transfers.csv row with Season <= the row's season (first listed position)."""
    t = pd.read_csv(DATA / "transfers.csv", encoding="utf-8-sig", usecols=["Player", "Season", "Pos"]).dropna()
    t["g"] = t.Pos.str.split(",").str[0].str.strip().map(GROUP)
    t = t.sort_values("Season")
    by = {p: (g.Season.to_numpy(), g.g.to_numpy()) for p, g in t.groupby("Player")}
    out = []
    for p, s in zip(log.Player.to_numpy(), log.Season.to_numpy()):
        v = by.get(p)
        if v is None:
            out.append(None)
            continue
        i = np.searchsorted(v[0], s, side="right") - 1
        out.append(v[1][i] if i >= 0 and isinstance(v[1][i], str) else None)   # GK / unmapped -> not scored
    return np.array(out, dtype=object)


def sq(log, t):
    a, p = TARGETS[t]
    return ((log[p].to_numpy() - log[a].to_numpy()) * log["weight"].to_numpy()) ** 2


def boot(base, cand, players, n=2000, seed=42):
    df = pd.DataFrame({"p": players, "b": base, "c": cand}).groupby("p")[["b", "c"]].sum()
    b, c = df.b.to_numpy(), df.c.to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(b), size=(n, len(b)))
    d = c[idx].sum(1) / b[idx].sum(1) - 1
    return round(float(c.sum() / b.sum() - 1) * 100, 4), [round(float(x) * 100, 4) for x in np.percentile(d, [2.5, 97.5])]


def rel(c, b, m):
    return round(float(c[m].sum() / b[m].sum() - 1) * 100, 4)


def score():
    logs = {a: pd.read_csv(log_path(a), encoding="utf-8-sig") for a in GRID}
    base = logs[BASE]
    for a, lg in logs.items():
        assert len(lg) == len(base) and (lg.Player.to_numpy() == base.Player.to_numpy()).all(), a
    first = ~base.duplicated("Player").to_numpy()
    for a, lg in logs.items():   # a player's first logged row can't depend on the floor
        assert np.allclose(lg.pred_goals_90.to_numpy()[first], base.pred_goals_90.to_numpy()[first]), a
    g = groups(base)
    season = base.Season.to_numpy()
    train, test, late = season <= 2023, season >= 2024, season >= 2025
    scored = np.array([x is not None for x in g])
    players = base.Player.to_numpy()
    err = {a: {t: sq(lg, t) for t in TARGETS} for a, lg in logs.items()}
    res = {"rows": {"train": int((train & scored).sum()), "test": int((test & scored).sum())}, "grid": {}, "choice": {}}
    for grp in sorted({x for x in g if x is not None}):
        m = g == grp
        for t in TARGETS:
            cell = {a: {"train": rel(err[a][t], err[BASE][t], m & train), "test": rel(err[a][t], err[BASE][t], m & test)} for a in GRID}
            res["grid"][f"{grp}_{t}"] = cell
            best = min(GRID, key=lambda a: err[a][t][m & train].sum())
            res["choice"][f"{grp}_{t}"] = best
    # Combined arm: each row's prediction from its group's chosen floor.
    combo = {t: err[BASE][t].copy() for t in TARGETS}
    for key, a in res["choice"].items():
        grp, t = key.split("_")
        m = g == grp
        combo[t][m] = err[a][t][m]
    res["test"] = {}
    for t in TARGETS:
        m = test & scored
        ch, ci = boot(err[BASE][t][m], combo[t][m], players[m])
        res["test"][t] = {"change_pct": ch, "ci": ci, "s2025_26": rel(combo[t], err[BASE][t], late & scored),
                          "train": rel(combo[t], err[BASE][t], train & scored)}
        res["test"][t]["by_group"] = {grp: rel(combo[t], err[BASE][t], test & (g == grp)) for grp in sorted({x for x in g if x is not None})}
    changed = [k for k, a in res["choice"].items() if a != BASE]
    worst = max([res["grid"][k][res["choice"][k]]["test"] for k in changed], default=0.0)
    ok = (all(res["test"][t]["change_pct"] <= 0 for t in TARGETS)
          and any(res["test"][t]["ci"][1] < 0 for t in TARGETS)
          and worst <= 0.5 and all(res["test"][t]["s2025_26"] <= 0.1 for t in TARGETS))
    res["worst_changed_cell_test"] = worst
    res["verdict"] = "PASS" if ok else "FAIL"
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
