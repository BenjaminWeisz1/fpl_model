"""
Dampening exponent on the goal/assist loop's opponent factor (CHANGELOG §123).

    python validate_opp_beta.py run     # one pre-match log per beta (same beta for both targets)
    python validate_opp_beta.py score   # beta* per target on 2018-2022, test 2023-2026 and 2025+ -> results.json
"""
from fpl_model import paths
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.player_ratings.validate_prior_k import boot_diff  # noqa: E402

ROOT = paths.ROOT
OUT = paths.VALIDATION / "opp_beta_20261030"
GRID = ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
TARGETS = {"npxG": ("npxg90", "pred_goals_90"), "xAG": ("xag90", "pred_ast_90")}


def log_path(b):
    return OUT / f"log_b{b}.csv"


def run_one(b):
    env = dict(os.environ, GOAL_MODEL_OPP_BETA_NPG=b, GOAL_MODEL_OPP_BETA_AST=b,
               GOAL_MODEL_PREMATCH_LOG=str(log_path(b)), PYTHONHASHSEED="0")
    with open(OUT / f"log_b{b}.run.log", "w", encoding="utf-8") as fh:
        subprocess.run([sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve().parent / "build_player_ratings.py")],
                       env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)
    return b


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        for b in ex.map(run_one, GRID):
            print(f"beta={b} done", flush=True)


def score():
    logs = {b: pd.read_csv(log_path(b), encoding="utf-8-sig", low_memory=False) for b in GRID}
    base = logs["1.0"]
    for b, lg in logs.items():
        assert len(lg) == len(base) and (lg["Player"].to_numpy() == base["Player"].to_numpy()).all(), b
    gk = set(pd.read_csv(ROOT / "data" / "current" / "transfers.csv", encoding="utf-8-sig").query("Pos == 'GK'")["Player"])
    keep = (base["Season"] >= 2018).to_numpy() & ~base["Player"].isin(gk).to_numpy()
    masks = {"train": keep & (base["Season"] <= 2022).to_numpy(), "test": keep & (base["Season"] >= 2023).to_numpy(),
             "test_2025plus": keep & (base["Season"] >= 2025).to_numpy()}
    w = base["weight"].to_numpy()
    opp = base["Opp_Factor"].to_numpy()
    res = {"rows": {k: int(m.sum()) for k, m in masks.items()}, "grid": {}}
    err = {b: {t: ((lg[p].to_numpy() - lg[a].to_numpy()) * w) ** 2 for t, (a, p) in TARGETS.items()} for b, lg in logs.items()}
    for b in GRID:
        res["grid"][b] = {f"{k}_{t}": float(err[b][t][m].sum() / err["1.0"][t][m].sum() - 1)
                          for k, m in masks.items() for t in TARGETS}
    players = base["Player"].to_numpy()
    for t, (a, p) in TARGETS.items():
        bstar = min(GRID, key=lambda b: res["grid"][b][f"train_{t}"])
        r = {"beta": bstar}
        for k in ("test", "test_2025plus"):
            m = masks[k]
            ch, ci = boot_diff(err["1.0"][t][m], err[bstar][t][m], players[m])
            r[k] = {"change": ch, "ci95": ci}
            # calibration by opponent tercile (within the window)
            q = pd.qcut(pd.Series(opp[m]).rank(method="first"), 3, labels=["strong", "middle", "weak"]).to_numpy()
            for name, lg in (("base", base), ("cand", logs[bstar])):
                y, mu = lg[a].to_numpy()[m] * w[m], lg[p].to_numpy()[m] * w[m]
                r[k][f"tercile_{name}"] = {c: round(float(y[q == c].sum() / mu[q == c].sum()), 3) for c in ("strong", "middle", "weak")}
        r["adopt"] = bool(r["test"]["change"] < 0 and r["test_2025plus"]["change"] <= 0 and bstar != "1.0")
        res[t] = r
        print(t, json.dumps(r), flush=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for b in GRID:
        print(b, {k: round(v * 100, 3) for k, v in res["grid"][b].items()})


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
