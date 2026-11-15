"""
Validation of the per-90 goal/assist loop (CHANGELOG §86).

Compares two GOAL_MODEL_PREMATCH_LOG runs under data/validation/per90_loop_20261028/:
baseline (the loop targets npxG per appearance) and candidate (per 90). Each arm is scored as
the forecast uses ratings: predicted match npxG = pre-match rating x the row's venue x opponent
factor x realized minutes / 90, against actual match npxG (xAG likewise).
  primary    MSE of match npxG and xAG per era, player-cluster bootstrap (5,000, seed 42);
  secondary  calibration (sum predicted / sum actual) by the player's average minutes per
             appearance in the era, and crossed with this match's minutes; MAE for reference.

Run locally (after the logged run):
    python validate_per90_loop.py
"""
from fpl_model import paths
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.player_ratings.validate_opponent_lookup import paired

ROOT = paths.ROOT
DIR = paths.VALIDATION / "per90_loop_20261028"
N_BOOT, SEED = 5000, 42
BANDS = [0, 45, 70, 85, 200]
BAND_LABELS = ["<=45", "46-70", "71-85", "86+"]
TARGETS = {"npxG": ("npxg90", "pred_goals_90", "NPG_Rating_Pre"),
           "xAG": ("xag90", "pred_ast_90", "Assist_Rating_Pre")}


def frame(j):
    """Forecast-consistent predictions for both arms. The baseline logs rating x factor x 90/min,
    the candidate rating x factor; the factor is recovered from the baseline and is the same in
    both arms (same lookups)."""
    w = j["weight_b"]
    out = pd.DataFrame({"Player": j["Player"], "era": np.where(j["Data_Source"] == "fbref", "fbref", "FPL"),
                        "minutes": w * 90})
    for t, (act, pred, rating) in TARGETS.items():
        factor = j[f"{pred}_b"] * w / j[f"{rating}_b"]
        out[f"{t}_actual"] = j[f"{act}_b"] * w
        out[f"{t}_base"] = j[f"{rating}_b"] * factor * w
        out[f"{t}_cand"] = j[f"{rating}_c"] * factor * w
    return out


def mse_compare(f, t):
    eb = (f[f"{t}_base"] - f[f"{t}_actual"]) ** 2
    ec = (f[f"{t}_cand"] - f[f"{t}_actual"]) ** 2
    g = pd.DataFrame({"Player": f["Player"], "eb": eb, "ec": ec}).groupby("Player")[["eb", "ec"]].sum().to_numpy()
    idx = np.random.default_rng(SEED).integers(0, len(g), size=(N_BOOT, len(g)))
    s = g[idx].sum(axis=1)
    rel = 100 * (s[:, 1] / s[:, 0] - 1)
    mb, mc = float(eb.mean()), float(ec.mean())
    mae_b = float((f[f"{t}_base"] - f[f"{t}_actual"]).abs().mean())
    mae_c = float((f[f"{t}_cand"] - f[f"{t}_actual"]).abs().mean())
    return {"mse_baseline": mb, "mse_candidate": mc, "change_pct": 100 * (mc / mb - 1),
            "ci95_pct": [float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))],
            "p_worse": float((rel > 0).mean()),
            "mae_change_pct_reference": 100 * (mae_c / mae_b - 1)}


def calibration(f, by):
    rows = {}
    for key, g in f.groupby(by, observed=True):
        rows[" / ".join(map(str, key)) if isinstance(key, tuple) else str(key)] = {
            "rows": int(len(g)),
            **{f"{t}_{arm}": round(float(g[f"{t}_{arm}"].sum() / g[f"{t}_actual"].sum()), 3)
               for t in TARGETS for arm in ("base", "cand")}}
    return rows


def main():
    read = lambda name: pd.read_csv(DIR / name, encoding="utf-8-sig")
    f = frame(paired(read("baseline_log.csv"), read("candidate_log.csv")))
    avg = f.groupby(["era", "Player"])["minutes"].transform("mean")
    f["usual"] = pd.cut(avg, BANDS, labels=BAND_LABELS)
    f["this_match"] = np.where(f["minutes"] >= 60, ">=60", np.where(f["minutes"] <= 45, "<=45", "46-59"))
    out = {"rows": int(len(f)), "eras": {}}
    for era, g in f.groupby("era"):
        out["eras"][era] = {"rows": int(len(g)), **{t: mse_compare(g, t) for t in TARGETS},
                            "calibration_by_usual_minutes": calibration(g, "usual")}
    out["calibration_by_usual_minutes_pooled"] = calibration(f, "usual")
    out["calibration_usual_x_this_match_pooled"] = calibration(f, ["usual", "this_match"])

    rb, rc = read("baseline_log.csv.ratings.csv"), read("candidate_log.csv.ratings.csv")
    m = rb.merge(rc, on=["Player", "Team"], suffixes=("_b", "_c"))
    m["ratio"] = m["NPG/90_c"] / m["NPG/90_b"]
    out["final_ratings"] = {"nan": int(rc[["NPG/90", "Assists/90"]].isna().any(axis=1).sum()),
                            "mean_ratio_NPG": float(m["NPG/90_c"].mean() / m["NPG/90_b"].mean()),
                            "mean_ratio_Ast": float(m["Assists/90_c"].mean() / m["Assists/90_b"].mean()),
                            "largest_rises": m.nlargest(10, "ratio")[["Player", "Team", "NPG/90_b", "NPG/90_c"]]
                            .round(4).to_dict("records")}
    cal = out["calibration_by_usual_minutes_pooled"]["<=45"]
    dec = {}
    for era in ("fbref", "FPL"):
        for t in TARGETS:
            dec[f"{era}_{t}_mse_improves"] = out["eras"][era][t]["mse_candidate"] < out["eras"][era][t]["mse_baseline"]
    for t in TARGETS:
        dec[f"le45_{t}_calibration_better"] = abs(cal[f"{t}_cand"] - 1) < abs(cal[f"{t}_base"] - 1)
    dec["no_nan_ratings"] = out["final_ratings"]["nan"] == 0
    dec["adopt"] = all(dec.values())
    out["decision"] = dec
    (DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
