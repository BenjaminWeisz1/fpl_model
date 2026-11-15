"""
Separate prior weight for youth players in the goal/assist loop (CHANGELOG §126).

    python validate_youth_k.py run     # one pre-match log per K_youth (GOAL_MODEL_PRIOR_K_YOUTH)
    python validate_youth_k.py score   # youth entrants: train/test MSE; all-rows guard -> results.json
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.player_ratings.validate_prior_k import entrants, sq_errors, boot_diff, TARGETS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "validation" / "youth_k_20261030"
GRID = ["2", "4", "6", "8", "10", "13", "16"]


def log_path(k):
    return OUT / f"log_ky{k}.csv"


def run_one(k):
    env = dict(os.environ, GOAL_MODEL_PRIOR_K_YOUTH=k, GOAL_MODEL_PREMATCH_LOG=str(log_path(k)), PYTHONHASHSEED="0")
    with open(OUT / f"log_ky{k}.run.log", "w", encoding="utf-8") as fh:
        subprocess.run([sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve().parent / "build_player_ratings.py")],
                       env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)
    return k


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for k in ex.map(run_one, GRID):
            print(f"K_youth={k} done", flush=True)


def score():
    logs = {k: pd.read_csv(log_path(k), encoding="utf-8-sig", low_memory=False) for k in GRID}
    base = logs["16"]
    for k, lg in logs.items():
        assert len(lg) == len(base) and (lg["Player"].to_numpy() == base["Player"].to_numpy()).all(), k
    ent = entrants(base)
    youth = ent[ent["Type"] == "youth"]
    m = base[["Player", "Season"]].merge(youth, on="Player", how="left")
    scored = (m["Entry"].notna() & (m["Season"] <= m["Entry"] + 1)).to_numpy()
    masks = {"train": scored & (m["Entry"] <= 2022).to_numpy(), "test": scored & (m["Entry"] >= 2023).to_numpy(),
             "all_2019": (base["Season"] >= 2019).to_numpy()}
    err = {k: sq_errors(lg) for k, lg in logs.items()}
    res = {"youth_entrants": int(len(youth)), "rows": {k: int(v.sum()) for k, v in masks.items()}, "grid": {}}
    for k in GRID:
        r = {f"{n}_{t}": float(err[k][t][mk].sum() / err["16"][t][mk].sum() - 1) for n, mk in masks.items() for t in TARGETS}
        r["train_mean"] = (r["train_npxG"] + r["train_xAG"]) / 2
        res["grid"][k] = r
    kstar = min(GRID, key=lambda k: res["grid"][k]["train_mean"])
    res["k_star"] = kstar
    players = base["Player"].to_numpy()
    res["test"] = {}
    for t in TARGETS:
        mk = masks["test"]
        ch, ci = boot_diff(err["16"][t][mk], err[kstar][t][mk], players[mk])
        res["test"][t] = {"change": ch, "ci95": ci, "all_2019": res["grid"][kstar][f"all_2019_{t}"]}
    res["adopt"] = bool(kstar != "16" and all(res["test"][t]["change"] < 0 and res["test"][t]["all_2019"] <= 0.0005
                                             for t in TARGETS))
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k in GRID:
        print(k, {a: round(b * 100, 3) for a, b in res["grid"][k].items()})
    print("K_y*", kstar, json.dumps(res["test"]), "adopt", res["adopt"])


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
