"""
Goal/assist loop: a faster learning-rate floor for npxG only? (CHANGELOG §233)

    python validate_npxg_floor.py run     # one pre-match log per floor (GOAL_MODEL_ALPHA), production untouched
    python validate_npxg_floor.py score   # npxG MSE per season and pooled against the 0.025 floor -> results.json

The step max(1/(n + 16), floor) x minutes/90 is shared by both targets, but the npxG rating's update uses only npxG
data, so the npxG columns of a run at floor f are exactly the npxG loop at f (xAG columns are ignored). Base = the
§231 log (current code, floor 0.025, §104 out-of-time priors). Rows: outfield players (§196 position groups), seasons
2018-2026 (2017 has no earlier rows to learn from).
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_position_alpha as vpa  # noqa: E402

ROOT = vpa.ROOT
OUT = ROOT / "data" / "validation" / "npxg_floor_20261112"
BASE_LOG = ROOT / "data" / "validation" / "league_anchor_20261112" / "log_current.csv"
GRID = ["0.03", "0.035", "0.04", "0.045", "0.05"]
SEASONS = range(2018, 2027)


def log_path(a):
    return BASE_LOG if a == "0.025" else OUT / f"log_a{a}.csv"


def run_one(a):
    env = dict(os.environ, GOAL_MODEL_ALPHA=a, GOAL_MODEL_PREMATCH_LOG=str(log_path(a)),
               GOAL_MODEL_NEW_PRIORS_FILE=str(vpa.OOT_PRIORS), PYTHONHASHSEED="0")
    with open(OUT / f"log_a{a}.run.log", "w", encoding="utf-8") as fh:
        r = subprocess.run([sys.executable, "-u", "-X", "utf8", str(ROOT / "scripts" / "player_ratings" /
                            "build_player_ratings.py")], env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"floor {a} failed, see log_a{a}.run.log")
    return a


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as ex:
        for a in ex.map(run_one, GRID):
            print(f"floor {a} done", flush=True)


def score():
    logs = {a: pd.read_csv(log_path(a), encoding="utf-8-sig", low_memory=False) for a in ["0.025"] + GRID}
    base = logs["0.025"]
    for a, lg in logs.items():
        assert len(lg) == len(base) and (lg.Player.to_numpy() == base.Player.to_numpy()).all(), a
        assert np.allclose(lg.xag90.to_numpy(), base.xag90.to_numpy()), a
    g = vpa.groups(base)
    season = base.Season.to_numpy()
    scored = np.array([x is not None for x in g]) & np.isin(season, list(SEASONS))
    players = base.Player.to_numpy()
    eb = vpa.sq(base, "npxG")
    res = {"rows": int(scored.sum()), "pooled": {}, "by_season": {}, "by_group": {}}
    for a in GRID:
        e = vpa.sq(logs[a], "npxG")
        ch, ci = vpa.boot(eb[scored], e[scored], players[scored])
        per = {int(s): vpa.rel(e, eb, scored & (season == s)) for s in SEASONS}
        res["pooled"][a] = {"change_pct": ch, "ci": ci, "seasons_better": sum(v < 0 for v in per.values())}
        res["by_season"][a] = per
        res["by_group"][a] = {grp: vpa.rel(e, eb, scored & (g == grp)) for grp in sorted({x for x in g if x})}
    best = min(GRID, key=lambda a: res["pooled"][a]["change_pct"])
    r = res["pooled"][best]
    res["best"] = best
    res["recommend"] = bool(r["change_pct"] < 0 and r["ci"][1] < 0 and r["seasons_better"] >= 6)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
