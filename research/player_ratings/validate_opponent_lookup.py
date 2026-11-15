"""
Validation of the opponent-lookup fix in build_player_ratings.py (CHANGELOG §78).

Compares GOAL_MODEL_PREMATCH_LOG runs under data/validation/opponent_lookup_20261028/:
baseline (old (MatchID, opponent) key), candidate ((Season, Team, Opponent, Venue) key)
and diagnostic (candidate with GOAL_MODEL_OPP_DIAG_RESCALE=1; reported, never adopted).
The candidate was not adopted; its script (with the diagnostic switch) is archived in
candidate_code/ of the run directory, and production build_player_ratings.py is the original.
  identity  every fbref-era logged row identical, baseline vs candidate;
  primary   minutes-weighted one-step MAE of npxG/90 and xAG/90 on FPL-era rows, with a
            player-cluster bootstrap interval of the relative change (5,000 replicates, seed 42);
  secondary MAE by tercile of the true opponent multiplier, rating level, final ratings.

Run locally (after the logged runs):
    python -m research.player_ratings.validate_opponent_lookup [--dir <run directory>]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model import paths

ROOT = paths.ROOT
DIR = paths.VALIDATION / "opponent_lookup_20261028"
KEY = ["Player", "Team", "Opponent", "Venue", "MatchID", "Data_Source"]
VALUES = ["NPG_Rating_Pre", "Assist_Rating_Pre", "npxg90", "pred_goals_90", "xag90", "pred_ast_90", "weight"]
AVG_CONCEDED = 1.32023
HOME_ADVANTAGE = 1.084340351
N_BOOT, SEED = 5000, 42


def occurrence(frame):
    # A few log keys repeat (duplicated 2025 Junior Kroupi rows); pair them by order.
    return frame.assign(_occ=frame.groupby(KEY).cumcount())


def mae(err, w):
    return float((np.abs(err) * w).sum() / w.sum()) if err.notna().all() else float("nan")


def paired(base, other):
    """Join two logs row by row; every row must pair."""
    j = occurrence(base).merge(occurrence(other), on=KEY + ["_occ"], how="outer",
                               suffixes=("_b", "_c"), indicator=True)
    assert (j["_merge"] == "both").all(), "logged rows differ between runs"
    return j


def compare(j, label):
    """Primary and secondary metrics on FPL-era paired rows."""
    f = j[j["Data_Source"] != "fbref"].copy()
    venue = np.where(f["Venue"] == "Home", HOME_ADVANTAGE, 2 - HOME_ADVANTAGE)
    # the true (candidate-key) multiplier, recovered from the candidate's own prediction
    f["mult"] = f["pred_goals_90_c"] * f["weight_c"] / (f["NPG_Rating_Pre_c"] * venue)
    out = {"rows": int(len(f)), "players": int(f["Player"].nunique())}
    for k, pred in (("npxg90", "pred_goals_90"), ("xag90", "pred_ast_90")):
        f[f"e_{k}_b"] = (f[f"{k}_b"] - f[f"{pred}_b"]).abs() * f["weight_b"]
        f[f"e_{k}_c"] = (f[f"{k}_c"] - f[f"{pred}_c"]).abs() * f["weight_c"]
        mb = mae(f[f"{k}_b"] - f[f"{pred}_b"], f["weight_b"])
        mc = mae(f[f"{k}_c"] - f[f"{pred}_c"], f["weight_c"])
        g = f.groupby("Player")[[f"e_{k}_b", f"e_{k}_c", "weight_b"]].sum().to_numpy()
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
        s = g[idx].sum(axis=1)
        rel = 100 * (s[:, 1] / s[:, 0] - 1)
        out[k] = {"mae_baseline": mb, "mae_" + label: mc, "change_pct": 100 * (mc / mb - 1),
                  "ci95_pct": [float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))],
                  "p_worse": float((rel > 0).mean())}
    f["tercile"] = pd.qcut(f["mult"], 3, labels=["weak_opp_low", "mid", "strong_attack_high"])
    out["by_multiplier_tercile"] = {}
    for t, g in f.groupby("tercile", observed=True):
        row = {"rows": int(len(g)), "mult_range": [round(float(g["mult"].min()), 3), round(float(g["mult"].max()), 3)]}
        for k in ("npxg90", "xag90"):
            b, c = g[f"e_{k}_b"].sum() / g["weight_b"].sum(), g[f"e_{k}_c"].sum() / g["weight_c"].sum()
            row[k] = {"baseline": float(b), label: float(c), "change_pct": float(100 * (c / b - 1))}
        out["by_multiplier_tercile"][str(t)] = row
    out["true_multiplier_mean_by_season"] = f.groupby("Season_c")["mult"].mean().round(4).to_dict() \
        if "Season_c" in f else f.groupby("Season")["mult"].mean().round(4).to_dict()
    out["prematch_rating_level_ratio"] = {
        "NPG": float(f["NPG_Rating_Pre_c"].mean() / f["NPG_Rating_Pre_b"].mean()),
        "Assist": float(f["Assist_Rating_Pre_c"].mean() / f["Assist_Rating_Pre_b"].mean())}
    return out


def finals(rb, rc):
    m = rb.merge(rc, on=["Player", "Team"], how="outer", suffixes=("_b", "_c"), indicator=True)
    both = m[m["_merge"] == "both"].copy()
    both["dNPG"] = both["NPG/90_c"] - both["NPG/90_b"]
    both["dAst"] = both["Assists/90_c"] - both["Assists/90_b"]
    big = both.reindex(both["dNPG"].abs().sort_values(ascending=False).index).head(10)
    return {"players_b": int(len(rb)), "players_c": int(len(rc)), "unpaired": int((m["_merge"] != "both").sum()),
            "nan_ratings": int(rc[["NPG/90", "Assists/90"]].isna().any(axis=1).sum()),
            "changed": int((both["dNPG"].abs() > 1e-12).sum()),
            "mean_ratio_NPG": float(both["NPG/90_c"].mean() / both["NPG/90_b"].mean()),
            "mean_ratio_Ast": float(both["Assists/90_c"].mean() / both["Assists/90_b"].mean()),
            "mean_abs_dNPG90": float(both["dNPG"].abs().mean()), "mean_abs_dAst90": float(both["dAst"].abs().mean()),
            "largest": big[["Player", "Team", "NPG/90_b", "NPG/90_c", "Assists/90_b", "Assists/90_c"]]
            .round(4).to_dict("records")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=DIR)
    run_dir = parser.parse_args().dir
    read = lambda name: pd.read_csv(run_dir / name, encoding="utf-8-sig")
    base, cand = read("baseline_log.csv"), read("candidate_log.csv")
    rb, rc = read("baseline_log.csv.ratings.csv"), read("candidate_log.csv.ratings.csv")

    j = paired(base, cand)
    fb = j[j["Data_Source"] == "fbref"]
    identity = bool(all((fb[f"{c}_b"] == fb[f"{c}_c"]).all() for c in VALUES))
    out = {"logged_rows": int(len(j)), "fbref_rows": int(len(fb)), "identity_fbref_rows": identity,
           "candidate": compare(j, "candidate"), "final_ratings_candidate": finals(rb, rc)}

    if (run_dir / "diagnostic_log.csv").exists():
        diag = read("diagnostic_log.csv")
        out["diagnostic_rescaled_never_adopted"] = compare(paired(base, diag), "diagnostic")
        out["final_ratings_diagnostic"] = finals(rb, read("diagnostic_log.csv.ratings.csv"))

    p = out["candidate"]["npxg90"]
    out["decision"] = {"identity": identity, "no_nan_ratings": out["final_ratings_candidate"]["nan_ratings"] == 0,
                       "npxg_not_worse": p["mae_candidate"] <= p["mae_baseline"]}
    out["decision"]["adopt"] = all(out["decision"].values())
    (run_dir / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
