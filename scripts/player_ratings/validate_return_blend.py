"""
Returning players' blend of old PL rating and new-league prior (CHANGELOG §114).

    python validate_return_blend.py run     # one pre-match log per H (GOAL_MODEL_RETURN_H)
    python validate_return_blend.py score   # train/test MSE on returners' rows; writes results.json
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
from validate_prior_k import boot_diff, sq_errors, OOT_PRIORS, TARGETS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "current"
OUT = ROOT / "data" / "validation" / "return_blend_20261029"
GRID = ["0", "1000", "2500", "5000", "10000", "25000", "inf"]


def log_path(h):
    return OUT / f"log_h{h}.csv"


def run_one(h):
    env = dict(os.environ, GOAL_MODEL_RETURN_H=h, GOAL_MODEL_PREMATCH_LOG=str(log_path(h)),
               GOAL_MODEL_NEW_PRIORS_FILE=str(OOT_PRIORS), PYTHONHASHSEED="0")
    with open(OUT / f"log_h{h}.run.log", "w", encoding="utf-8") as fh:
        r = subprocess.run([sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve().parent /
                            "build_player_ratings.py")], env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"H={h} failed, see log_h{h}.run.log")
    return h


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for h in ex.map(run_one, GRID):
            print(f"H={h} done", flush=True)


def score():
    logs = {h: pd.read_csv(log_path(h), encoding="utf-8-sig") for h in GRID}
    base = logs["5000"]
    for h, lg in logs.items():
        assert len(lg) == len(base) and (lg["Player"].to_numpy() == base["Player"].to_numpy()).all(), h
    t = pd.read_csv(DATA / "transfers.csv", encoding="utf-8-sig")
    ret = t[(t["Notes"] == "second season") & (t["Pos"] != "GK") & (t["Season"] >= 2019)][["Player", "Season"]]
    keys = set(map(tuple, ret.to_numpy()))
    # row -> the return season it belongs to (the return season itself or the next)
    rs = np.full(len(base), -1)
    for i, (p, s) in enumerate(zip(base["Player"].to_numpy(), base["Season"].to_numpy())):
        if (p, s) in keys:
            rs[i] = s
        elif (p, s - 1) in keys:
            rs[i] = s - 1
    train, test = (rs >= 2019) & (rs <= 2022), rs >= 2023
    all19 = (base["Season"] >= 2019).to_numpy()
    err = {h: sq_errors(lg) for h, lg in logs.items()}
    res = {"returns": int(len(ret)), "returns_logged": int(len(set(zip(base["Player"][rs > 0], rs[rs > 0])))),
           "rows": {"train": int(train.sum()), "test": int(test.sum())}, "grid": {}}
    for h in GRID:
        r = {}
        for tg in TARGETS:
            for name, mask in (("train", train), ("test", test), ("all_2019", all19)):
                r[f"{name}_{tg}"] = float(err[h][tg][mask].sum() / err["5000"][tg][mask].sum() - 1)
        r["train_mean"] = (r["train_npxG"] + r["train_xAG"]) / 2
        res["grid"][h] = r
    h_star = min(GRID, key=lambda h: res["grid"][h]["train_mean"])
    res["h_star"] = h_star
    res["test"] = {}
    players = base["Player"].to_numpy()
    for tg in TARGETS:
        d, ci = boot_diff(err["5000"][tg][test], err[h_star][tg][test], players[test])
        res["test"][tg] = {"change": d, "ci95": ci, "all_2019": res["grid"][h_star][f"all_2019_{tg}"]}
    res["adopt"] = bool(h_star != "5000" and all(res["test"][tg]["change"] < 0 and
                                                res["test"][tg]["all_2019"] <= 0.0005 for tg in TARGETS))
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
