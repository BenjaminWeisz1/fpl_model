"""
How many matches should the starting rating count as in the goal/assist loop? (CHANGELOG §112)

    python validate_prior_k.py run     # one pre-match log per K (GOAL_MODEL_PRIOR_K), out-of-time priors
    python validate_prior_k.py score   # train/test MSE of entrants' match npxG and xAG; writes results.json

Entrants: outfield players whose first logged PL match is in 2019 or later, scored on their first two logged
seasons. Train = entrants 2019-2022, test = entrants 2023-2026. Production is never written.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from fpl_model import paths

ROOT = paths.ROOT
DATA = paths.CURRENT
OUT = paths.VALIDATION / "prior_k_20261029"
OOT_PRIORS = paths.VALIDATION / "new_player_priors_20261028" / "oot_priors.csv"
GRID = [3, 5, 8, 12, 16, 22, 30, 40]
TARGETS = {"npxG": ("npxg90", "pred_goals_90"), "xAG": ("xag90", "pred_ast_90")}


def log_path(k):
    return OUT / f"log_k{k}.csv"


def run_one(k):
    env = dict(os.environ, GOAL_MODEL_PRIOR_K=str(k), GOAL_MODEL_PREMATCH_LOG=str(log_path(k)),
               GOAL_MODEL_NEW_PRIORS_FILE=str(OOT_PRIORS),
               PYTHONHASHSEED="0")   # the script's row order follows the hash seed (CLAUDE.md §5.2)
    with open(OUT / f"log_k{k}.run.log", "w", encoding="utf-8") as fh:
        r = subprocess.run([sys.executable, "-u", "-X", "utf8", "-m", "fpl_model.player_ratings.build_player_ratings"], env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"K={k} failed, see log_k{k}.run.log")
    return k


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for k in ex.map(run_one, GRID):
            print(f"K={k} done", flush=True)


def entrants(log, min_season=2019):
    """Player -> (entry season, entry type) for outfield players first logged in min_season or later."""
    t = pd.read_csv(DATA / "transfers.csv", encoding="utf-8-sig")
    oot = pd.read_csv(OOT_PRIORS, encoding="utf-8-sig")
    adopted = set(map(tuple, oot.loc[oot["Adopted"], ["Player", "Season"]].to_numpy()))
    first = log.groupby("Player")["Season"].min()
    rows = []
    for player, s0 in first.items():
        if s0 < min_season:
            continue
        cand = t[(t["Player"] == player) & (t["Season"] <= s0)].sort_values("Season")
        if cand.empty:
            continue
        tr = cand.iloc[-1]
        if tr["Pos"] == "GK":
            continue
        notes = tr["Notes"] if isinstance(tr["Notes"], str) else ""
        if notes == "youth":
            kind = "youth"
        elif notes == "" and tr["League"] != "Premier League":
            kind = "arrival_fitted" if (player, tr["Season"]) in adopted else "arrival_old"
        else:
            kind = "other"
        rows.append({"Player": player, "Entry": int(s0), "Type": kind})
    return pd.DataFrame(rows)


def sq_errors(log):
    w = log["weight"].to_numpy()
    return {t: ((log[p].to_numpy() - log[a].to_numpy()) * w) ** 2 for t, (a, p) in TARGETS.items()}


def boot_diff(base, cand, players, n=2000, seed=42):
    """Relative MSE change cand vs base with a player-cluster bootstrap."""
    df = pd.DataFrame({"p": players, "b": base, "c": cand}).groupby("p")[["b", "c"]].sum()
    b, c = df["b"].to_numpy(), df["c"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(n, len(b)))
    draws = c[idx].sum(1) / b[idx].sum(1) - 1
    return float(c.sum() / b.sum() - 1), [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def score():
    logs = {k: pd.read_csv(log_path(k), encoding="utf-8-sig") for k in GRID}
    base = logs[5]
    for k, lg in logs.items():   # same rows in the same order in every run
        assert len(lg) == len(base) and (lg["Player"].to_numpy() == base["Player"].to_numpy()).all(), k
    ent = entrants(base)
    m = base[["Player", "Season"]].merge(ent, on="Player", how="left")
    scored = (m["Entry"].notna() & (m["Season"] <= m["Entry"] + 1)).to_numpy()
    train = scored & (m["Entry"] <= 2022).to_numpy()
    test = scored & (m["Entry"] >= 2023).to_numpy()
    all19 = (base["Season"] >= 2019).to_numpy()
    err = {k: sq_errors(lg) for k, lg in logs.items()}

    res = {"entrants": ent["Type"].value_counts().to_dict(),
           "rows": {"train": int(train.sum()), "test": int(test.sum()), "all_2019": int(all19.sum())},
           "grid": {}}
    for k in GRID:
        r = {}
        for t in TARGETS:
            for name, mask in (("train", train), ("test", test), ("all_2019", all19)):
                r[f"{name}_{t}"] = float(err[k][t][mask].sum() / err[5][t][mask].sum() - 1)
        r["train_mean"] = (r["train_npxG"] + r["train_xAG"]) / 2
        res["grid"][k] = r
    k_star = min(GRID, key=lambda k: res["grid"][k]["train_mean"])
    res["k_star"] = k_star

    # per-type K* on train (secondary)
    res["k_star_by_type"] = {}
    for kind in sorted(ent["Type"].unique()):
        mk = train & (m["Type"] == kind).to_numpy()
        means = {k: np.mean([err[k][t][mk].sum() / err[5][t][mk].sum() - 1 for t in TARGETS]) for k in GRID}
        res["k_star_by_type"][kind] = {"k_star": min(means, key=means.get), "rows": int(mk.sum()),
                                       "mean_change": {k: round(v, 5) for k, v in means.items()}}

    # primary test with CI, plus test by type
    res["test"] = {}
    players = base["Player"].to_numpy()
    for t in TARGETS:
        d, ci = boot_diff(err[5][t][test], err[k_star][t][test], players[test])
        res["test"][t] = {"change": d, "ci95": ci, "all_2019": res["grid"][k_star][f"all_2019_{t}"]}
    res["test_by_type"] = {}
    for kind in sorted(ent["Type"].unique()):
        mk = test & (m["Type"] == kind).to_numpy()
        if mk.sum():
            res["test_by_type"][kind] = {t: float(err[k_star][t][mk].sum() / err[5][t][mk].sum() - 1)
                                         for t in TARGETS} | {"rows": int(mk.sum())}
    ok = all(res["test"][t]["change"] < 0 and res["test"][t]["all_2019"] <= 0.0005 for t in TARGETS)
    res["adopt"] = bool(ok and k_star != 5)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
