"""
Validation of the starter-equivalent loop for substitute appearances (CHANGELOG §101).

Baseline: the current production pre-match log; candidate: the logged run with substitute uplifts.
Both are scored as the forecast uses ratings, on realized minutes and appearance type: predicted
match npxG = pre-match rating x venue/opponent factor x minutes/90, times the frozen uplift for
substitute appearances in the candidate only. Primary: MSE per era (FPL era out of sample for the
uplifts); secondary: starts only, and calibration for usual subs when starting and in cameos.

Run locally (after the candidate's logged run):
    python validate_sub_uplift.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.player_ratings.validate_opponent_lookup import paired

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "validation" / "sub_uplift_20261028"
# Same-data control: the production code with GOAL_MODEL_SUB_UPLIFT=0, run on the same inputs as the
# candidate. (The first scoring used the 20:50 Experiment D log, which predates the 22:04 team-ratings
# rebuild; 7 rows differed by <= 5e-8.)
BASELINE = DIR / "switch_off_log.csv"
SUB_UPLIFT = {"npxG": [(15, 1.663), (30, 1.277), (45, 1.170), (999, 0.868)],
              "xAG": [(15, 1.685), (30, 1.276), (45, 1.113), (999, 1.045)]}
TARGETS = {"npxG": ("npxg90", "pred_goals_90", "NPG_Rating_Pre"), "xAG": ("xag90", "pred_ast_90", "Assist_Rating_Pre")}
N_BOOT, SEED = 5000, 42


def uplift(kind, is_sub, minutes):
    out = np.ones(len(minutes))
    for upper, u in reversed(SUB_UPLIFT[kind]):
        out = np.where(is_sub & (minutes <= upper), u, out)
    return out


def build():
    j = paired(pd.read_csv(BASELINE, encoding="utf-8-sig"), pd.read_csv(DIR / "candidate_log.csv", encoding="utf-8-sig"))
    starts = pd.read_csv(ROOT / "data" / "current" / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False,
                         usecols=["Season", "Team", "Opponent", "Venue", "Player", "Is_Starter"])
    starts = starts.drop_duplicates(["Season", "Team", "Opponent", "Venue", "Player"])
    j = j.merge(starts, left_on=["Season_b", "Team", "Opponent", "Venue", "Player"],
                right_on=["Season", "Team", "Opponent", "Venue", "Player"], how="left")
    f = pd.DataFrame({"Player": j["Player"], "era": np.where(j["Data_Source"] == "fbref", "fbref", "FPL"),
                      "minutes": j["weight_b"] * 90,
                      "is_sub": j["Is_Starter"].astype(str).str.lower().eq("false").to_numpy()})
    for t, (act, pred, rating) in TARGETS.items():
        factor = j[f"{pred}_b"] / j[f"{rating}_b"]  # baseline per-90 prediction = rating x factor
        w = j["weight_b"]
        f[f"{t}_actual"] = j[f"{act}_b"] * w
        f[f"{t}_base"] = j[f"{rating}_b"] * factor * w
        f[f"{t}_cand"] = j[f"{rating}_c"] * factor * w * uplift(t, f["is_sub"].to_numpy(), f["minutes"].to_numpy())
    f["usual"] = pd.cut(f.groupby(["era", "Player"])["minutes"].transform("mean"), [0, 45, 70, 85, 200],
                        labels=["<=45", "46-70", "71-85", "86+"])
    f["role"] = np.where(f["is_sub"], "sub", np.where(f["minutes"] >= 60, "start_60+", "start_<60"))
    return f


def mse(f, t):
    eb, ec = (f[f"{t}_base"] - f[f"{t}_actual"]) ** 2, (f[f"{t}_cand"] - f[f"{t}_actual"]) ** 2
    g = pd.DataFrame({"p": f["Player"], "eb": eb, "ec": ec}).groupby("p")[["eb", "ec"]].sum().to_numpy()
    idx = np.random.default_rng(SEED).integers(0, len(g), size=(N_BOOT, len(g)))
    s = g[idx].sum(axis=1)
    rel = 100 * (s[:, 1] / s[:, 0] - 1)
    return {"mse_base": float(eb.mean()), "mse_cand": float(ec.mean()), "change_pct": float(100 * (ec.mean() / eb.mean() - 1)),
            "ci95_pct": [float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))]}


def calibration(f, by):
    return {" / ".join(map(str, k)) if isinstance(k, tuple) else str(k):
            {"rows": int(len(g)), **{f"{t}_{a}": round(float(g[f"{t}_{a}"].sum() / g[f"{t}_actual"].sum()), 3)
                                     for t in TARGETS for a in ("base", "cand")}}
            for k, g in f.groupby(by, observed=True)}


def main():
    f = build()
    out = {"rows": int(len(f)), "start_flag_coverage": float(f["is_sub"].notna().mean()), "eras": {}}
    for era, g in f.groupby("era"):
        out["eras"][era] = {"all": {t: mse(g, t) for t in TARGETS},
                            "starts_60plus": {t: mse(g[g["role"] == "start_60+"], t) for t in TARGETS},
                            "calibration_usual_x_role": calibration(g, ["usual", "role"])}
    rb = pd.read_csv(str(BASELINE) + ".ratings.csv", encoding="utf-8-sig")
    rc = pd.read_csv(DIR / "candidate_log.csv.ratings.csv", encoding="utf-8-sig")
    m = rb.merge(rc, on=["Player", "Team"], suffixes=("_b", "_c"))
    m["ratio"] = m["NPG/90_c"] / m["NPG/90_b"]
    out["final_ratings"] = {"nan": int(rc[["NPG/90", "Assists/90"]].isna().any(axis=1).sum()),
                            "mean_ratio_NPG": float(m["NPG/90_c"].mean() / m["NPG/90_b"].mean()),
                            "largest_falls": m.nsmallest(10, "ratio")[["Player", "Team", "NPG/90_b", "NPG/90_c"]].round(4).to_dict("records"),
                            "named": m[m["Player"].isin(["Wilson", "Awoniyi", "Haaland", "Enes Ünal"])][["Player", "Team", "NPG/90_b", "NPG/90_c"]].round(4).to_dict("records")}
    fpl = out["eras"]["FPL"]
    cal = fpl["calibration_usual_x_role"]["<=45 / start_60+"]
    out["decision"] = {"fpl_npxG_mse_not_worse": fpl["all"]["npxG"]["change_pct"] <= 0,
                       "fpl_xAG_mse_not_worse": fpl["all"]["xAG"]["change_pct"] <= 0,
                       "usual_subs_starting_npxG_closer_to_1": abs(cal["npxG_cand"] - 1) < abs(cal["npxG_base"] - 1),
                       "no_nan": out["final_ratings"]["nan"] == 0}
    out["decision"]["adopt"] = all(out["decision"].values())
    (DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
