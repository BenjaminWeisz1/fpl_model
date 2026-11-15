"""
Validation of the opponent-defence multiplier in build_player_ratings.py (CHANGELOG §79).

Compares two GOAL_MODEL_PREMATCH_LOG runs under data/validation/opponent_defence_20261028/:
baseline (the §78 baseline: team predicted goals keyed on MatchID) and candidate (the
opponent's pre-match Defense_Rating / season mean, keyed on Season/Team/Opponent/Venue).
  primary    minutes-weighted one-step MAE of npxG/90 and xAG/90, each arm's own prediction,
             for FPL-era and fbref-era rows, player-cluster bootstrap (5,000 replicates, seed 42);
  secondary  MAE by tercile of the candidate multiplier; final-rating changes.

Run locally (after the logged run):
    python -m research.player_ratings.validate_opponent_defence [--dir <run directory>]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model import paths
from research.player_ratings.validate_opponent_lookup import HOME_ADVANTAGE, N_BOOT, SEED, finals, mae, paired

ROOT = paths.ROOT
DIR = paths.VALIDATION / "opponent_defence_20261028"
TARGETS = (("npxg90", "pred_goals_90"), ("xag90", "pred_ast_90"))


def era_metrics(f):
    venue = np.where(f["Venue"] == "Home", HOME_ADVANTAGE, 2 - HOME_ADVANTAGE)
    f = f.assign(mult=f["pred_goals_90_c"] * f["weight_c"] / (f["NPG_Rating_Pre_c"] * venue))
    out = {"rows": int(len(f)), "players": int(f["Player"].nunique())}
    for k, pred in TARGETS:
        f[f"e_{k}_b"] = (f[f"{k}_b"] - f[f"{pred}_b"]).abs() * f["weight_b"]
        f[f"e_{k}_c"] = (f[f"{k}_c"] - f[f"{pred}_c"]).abs() * f["weight_c"]
        mb = mae(f[f"{k}_b"] - f[f"{pred}_b"], f["weight_b"])
        mc = mae(f[f"{k}_c"] - f[f"{pred}_c"], f["weight_c"])
        g = f.groupby("Player")[[f"e_{k}_b", f"e_{k}_c", "weight_b"]].sum().to_numpy()
        idx = np.random.default_rng(SEED).integers(0, len(g), size=(N_BOOT, len(g)))
        s = g[idx].sum(axis=1)
        rel = 100 * (s[:, 1] / s[:, 0] - 1)
        out[k] = {"mae_baseline": mb, "mae_candidate": mc, "change_pct": 100 * (mc / mb - 1),
                  "ci95_pct": [float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))],
                  "p_worse": float((rel > 0).mean())}
    # Defense_Rating measures goals conceded: a low multiplier is a strong defence
    f["tercile"] = pd.qcut(f["mult"], 3, labels=["strong_defence_low", "mid", "weak_defence_high"])
    out["by_multiplier_tercile"] = {}
    for t, g in f.groupby("tercile", observed=True):
        row = {"rows": int(len(g)), "mult_range": [round(float(g["mult"].min()), 3), round(float(g["mult"].max()), 3)]}
        for k, _ in TARGETS:
            b, c = g[f"e_{k}_b"].sum() / g["weight_b"].sum(), g[f"e_{k}_c"].sum() / g["weight_c"].sum()
            row[k] = {"baseline": float(b), "candidate": float(c), "change_pct": float(100 * (c / b - 1))}
        out["by_multiplier_tercile"][str(t)] = row
    out["multiplier_mean"] = float(f["mult"].mean())
    out["prematch_rating_level_ratio_NPG"] = float(f["NPG_Rating_Pre_c"].mean() / f["NPG_Rating_Pre_b"].mean())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=DIR)
    run_dir = parser.parse_args().dir
    read = lambda name: pd.read_csv(run_dir / name, encoding="utf-8-sig")
    j = paired(read("baseline_log.csv"), read("candidate_log.csv"))
    out = {"logged_rows": int(len(j)),
           "fpl_era": era_metrics(j[j["Data_Source"] != "fbref"].copy()),
           "fbref_era": era_metrics(j[j["Data_Source"] == "fbref"].copy()),
           "final_ratings": finals(read("baseline_log.csv.ratings.csv"), read("candidate_log.csv.ratings.csv"))}
    fpl, fb = out["fpl_era"], out["fbref_era"]
    out["decision"] = {
        "fpl_npxg_improves": fpl["npxg90"]["change_pct"] < 0,
        "fpl_xag_within_0.5pct": fpl["xag90"]["change_pct"] <= 0.5,
        "fbref_npxg_within_0.5pct": fb["npxg90"]["change_pct"] <= 0.5,
        "fbref_xag_within_0.5pct": fb["xag90"]["change_pct"] <= 0.5,
        "no_nan_ratings": out["final_ratings"]["nan_ratings"] == 0}
    out["decision"]["adopt"] = all(out["decision"].values())
    (run_dir / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
