"""
Validation of the opponent-defence multiplier on the current loop, scored by MSE (CHANGELOG §110).

Baseline: the §108 production pre-match log; candidate: the logged run of a temporary copy of
build_player_ratings.py whose multiplier is the opponent's pre-match Defense_Rating / season mean.
  primary    MSE of match npxG and xAG (per-90 x minutes/90) per era, player-cluster bootstrap;
  secondary  the §79 metric (minutes-weighted MAE of per-90 values); MSE and calibration by tercile
             of the candidate multiplier; forward ranks.

Run locally (after the candidate's logged run):
    python -m research.player_ratings.validate_opponent_defence_mse
"""
import json

import numpy as np
import pandas as pd

from fpl_model import paths
from research.player_ratings.validate_opponent_lookup import N_BOOT, SEED, finals, mae, paired

ROOT = paths.ROOT
DIR = paths.VALIDATION / "opponent_defence_mse_20261029"
BASELINE = paths.VALIDATION / "inactivity_shrink_20261029" / "baseline_log.csv"
TARGETS = {"npxG": ("npxg90", "pred_goals_90"), "xAG": ("xag90", "pred_ast_90")}
NAMED = ["Haaland", "Awoniyi", "Watkins", "Isak", "C. Wilson", "Ekitiké", "Mateta", "Wood", "Gyökeres", "João Pedro"]


def multiplier():
    m = pd.read_csv(ROOT / "data" / "current" / "team_ratings_history.csv")
    mean = pd.concat([m[["Season", "Home_Defense_Rating"]].set_axis(["Season", "Def"], axis=1),
                      m[["Season", "Away_Defense_Rating"]].set_axis(["Season", "Def"], axis=1)]).groupby("Season")["Def"].mean()
    home = pd.DataFrame({"Season": m["Season"], "Team": m["Home"], "Opponent": m["Away"], "Venue": "Home",
                         "mult": m["Away_Defense_Rating"] / m["Season"].map(mean)})
    away = pd.DataFrame({"Season": m["Season"], "Team": m["Away"], "Opponent": m["Home"], "Venue": "Away",
                         "mult": m["Home_Defense_Rating"] / m["Season"].map(mean)})
    return pd.concat([home, away])


def sq_errors(f, k, arm):
    act, pred = TARGETS[k]
    return ((f[f"{pred}_{arm}"] - f[f"{act}_{arm}"]) * f[f"weight_{arm}"]) ** 2


def era_metrics(f):
    out = {"rows": int(len(f)), "players": int(f["Player"].nunique()), "multiplier_mean": float(f["mult"].mean())}
    for k, (act, pred) in TARGETS.items():
        eb, ec = sq_errors(f, k, "b"), sq_errors(f, k, "c")
        g = pd.DataFrame({"p": f["Player"], "eb": eb, "ec": ec}).groupby("p")[["eb", "ec"]].sum().to_numpy()
        idx = np.random.default_rng(SEED).integers(0, len(g), size=(N_BOOT, len(g)))
        s = g[idx].sum(axis=1)
        rel = 100 * (s[:, 1] / s[:, 0] - 1)
        mb, mc = mae(f[f"{act}_b"] - f[f"{pred}_b"], f["weight_b"]), mae(f[f"{act}_c"] - f[f"{pred}_c"], f["weight_c"])
        out[k] = {"mse_baseline": float(eb.mean()), "mse_candidate": float(ec.mean()),
                  "mse_change_pct": float(100 * (ec.mean() / eb.mean() - 1)),
                  "mse_ci95_pct": [float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))],
                  "p_worse": float((rel > 0).mean()),
                  "mae_per90_change_pct": float(100 * (mc / mb - 1))}
    f["tercile"] = pd.qcut(f["mult"], 3, labels=["strong_defence", "mid", "weak_defence"])
    out["by_tercile"] = {}
    for t, g in f.groupby("tercile", observed=True):
        row = {"rows": int(len(g)), "mult_range": [round(float(g["mult"].min()), 3), round(float(g["mult"].max()), 3)]}
        for k, (act, pred) in TARGETS.items():
            cal = {a: float((g[f"{pred}_{a}"] * g[f"weight_{a}"]).sum() / (g[f"{act}_{a}"] * g[f"weight_{a}"]).sum())
                   for a in ("b", "c")}
            mb, mc = sq_errors(g, k, "b").mean(), sq_errors(g, k, "c").mean()
            row[k] = {"mse_change_pct": float(100 * (mc / mb - 1)), "calibration_baseline": round(cal["b"], 3),
                      "calibration_candidate": round(cal["c"], 3)}
        out["by_tercile"][str(t)] = row
    return out


def ranks(path):
    r = pd.read_csv(path, encoding="utf-8-sig")
    r = r[r["Pos"] == "F"].copy()
    r["rank"] = r["NPG/90"].rank(ascending=False).astype(int)
    return r.set_index("Player")[["NPG/90", "rank"]]


def main():
    j = paired(pd.read_csv(BASELINE, encoding="utf-8-sig"), pd.read_csv(DIR / "candidate_log.csv", encoding="utf-8-sig"))
    j = j.merge(multiplier().rename(columns={"Season": "Season_b"}), on=["Season_b", "Team", "Opponent", "Venue"], how="left")
    out = {"logged_rows": int(len(j)), "multiplier_coverage": float(j["mult"].notna().mean())}
    for era, g in (("fbref", j[j["Data_Source"] == "fbref"]), ("FPL", j[j["Data_Source"] != "fbref"])):
        out[era] = era_metrics(g.copy())
    out["final_ratings"] = finals(pd.read_csv(str(BASELINE) + ".ratings.csv", encoding="utf-8-sig"),
                                  pd.read_csv(DIR / "candidate_log.csv.ratings.csv", encoding="utf-8-sig"))
    rb, rc = ranks(str(BASELINE) + ".ratings.csv"), ranks(DIR / "candidate_log.csv.ratings.csv")
    out["forward_ranks"] = {n: {"baseline": [round(float(rb.loc[n, "NPG/90"]), 3), int(rb.loc[n, "rank"])],
                                "candidate": [round(float(rc.loc[n, "NPG/90"]), 3), int(rc.loc[n, "rank"])]}
                            for n in NAMED if n in rb.index and n in rc.index}
    out["candidate_top15_forwards"] = rc.sort_values("rank").head(15).round(3).reset_index().to_dict("records")
    out["decision"] = {f"{era}_{k}_mse_not_worse": out[era][k]["mse_change_pct"] <= 0 for era in ("fbref", "FPL") for k in TARGETS}
    out["decision"]["no_nan_ratings"] = out["final_ratings"]["nan_ratings"] == 0
    out["decision"]["adopt"] = all(out["decision"].values())
    (DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
