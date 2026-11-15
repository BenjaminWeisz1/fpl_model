"""
Red-card factors estimated from the cached Understat shots (CHANGELOG §134).

Every Understat EPL match matched to a team_ratings_history.csv row is cut into segments at each goal (penalties and
own goals included), each red card and the clock boundaries 15/30/45/60/75'. Per team-segment the response is the
team's non-penalty Understat xG (own goals excluded); a Poisson quasi-likelihood GLM with offset
log(pre-match predicted goals x duration / 90) estimates the man-advantage effects, controlling for score
difference, clock bin and home. Factor = exp(-coef): the weight that turns a man-advantage segment's xG into its
11 v 11 equivalent. Only relative sizes of Understat xG enter, never its level. SEs clustered by match.

    python estimate_red_card_factors.py      # -> data/validation/red_card_20261030/factors_estimate.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from red_card_adjust import MATCHES, DATA, dismissals, counts_at, match_end, team_map  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "validation" / "red_card_20261030"
CLOCK = (15, 30, 45, 60, 75)
CUTOFFS = (2023, 2024, 2025, 2026)


def segments():
    """One row per team-segment."""
    names, season_of = team_map()
    hist = pd.read_csv(DATA / "team_ratings_history.csv", encoding="utf-8-sig")
    pred = {(r.Season, r.Home, r.Away): (r.Home_Prediction, r.Away_Prediction) for r in hist.itertuples()}
    rows = []
    for f in sorted(MATCHES.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        shots = d["shots"]["h"] + d["shots"]["a"]
        if not shots:
            continue
        s0 = shots[0]
        key = (season_of.get(f.stem, int(s0["season"])), names.get(s0["h_team"], s0["h_team"]),
               names.get(s0["a_team"], s0["a_team"]))
        if key not in pred:
            continue
        reds = dismissals(d)
        end = match_end(d, reds)
        goals = []   # (minute, side credited)
        for side in "ha":
            other = "a" if side == "h" else "h"
            for s in d["shots"][side]:
                if s["result"] == "Goal":
                    goals.append((int(s["minute"]), side))
                elif s["result"] == "OwnGoal":   # listed under the conceding side's player
                    goals.append((int(s["minute"]), other))
        cuts = sorted({0, end, *[c for c in CLOCK if c < end], *[g for g, _ in goals if 0 < g < end],
                       *[m for m, _, _ in reds if 0 < m < end]})
        first_red = reds[0][0] if reds else None
        for a, b in zip(cuts[:-1], cuts[1:]):
            n = counts_at(reds, b)
            score = {"h": sum(1 for g, sd in goals if g <= a and sd == "h"),
                     "a": sum(1 for g, sd in goals if g <= a and sd == "a")}
            for side in "ha":
                other = "a" if side == "h" else "h"
                xg = sum(float(s["xG"]) for s in d["shots"][side]
                         if s["situation"] != "Penalty" and s["result"] != "OwnGoal"
                         and (a < int(s["minute"]) <= b or (a == 0 and int(s["minute"]) == 0)))
                rows.append({"match": int(f.stem), "Season": key[0], "home": int(side == "h"),
                             "xg": xg, "dur": b - a, "pred": pred[key][0 if side == "h" else 1],
                             "state": int(np.clip(n[side] - n[other], -2, 2)),
                             "score": int(np.clip(score[side] - score[other], -2, 2)),
                             "clock": int(sum(a >= c for c in CLOCK)),
                             "late": int(first_red is not None and first_red >= 75)})
    return pd.DataFrame(rows)


def design(df, extra=()):
    X = pd.DataFrame(index=df.index)
    for s in (1, -1, 2, -2):
        X[f"state{s:+d}"] = (df["state"] == s).astype(float)
    for s in (-2, -1, 1, 2):
        X[f"score{s:+d}"] = (df["score"] == s).astype(float)
    for c in range(1, len(CLOCK) + 1):
        X[f"clock{c}"] = (df["clock"] == c).astype(float)
    X["home"] = df["home"].astype(float)
    if "late" in extra:
        for s in (1, -1):
            X[f"state{s:+d}_late"] = X[f"state{s:+d}"] * df["late"]
    if "home" in extra:
        for s in (1, -1):
            X[f"state{s:+d}_home"] = X[f"state{s:+d}"] * df["home"]
    return sm.add_constant(X)


def fit(df, extra=(), offset=True):
    X = design(df, extra)
    off = np.log(df["dur"] / 90.0) + (np.log(df["pred"]) if offset else 0.0)
    r = sm.GLM(df["xg"], X, family=sm.families.Poisson(), offset=off).fit(
        cov_type="cluster", cov_kwds={"groups": df["match"]})
    ci = r.conf_int()
    out = {}
    for name in X.columns:
        if name.startswith("state"):
            # factor = exp(-coef); its CI from the coefficient's CI (bounds swap)
            out[name] = {"coef": round(float(r.params[name]), 4), "se": round(float(r.bse[name]), 4),
                         "factor": round(float(np.exp(-r.params[name])), 3),
                         "factor_ci95": [round(float(np.exp(-ci.loc[name, 1])), 3),
                                         round(float(np.exp(-ci.loc[name, 0])), 3)]}
    out["scale_pearson"] = round(float(r.pearson_chi2 / r.df_resid), 3)
    return out, r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = segments()
    df = df[df["dur"] > 0]
    df.to_csv(OUT / "segments.csv", index=False)
    res = {"rows": len(df), "matches": int(df["match"].nunique()),
           "matches_with_red": int(df.loc[df["state"] != 0, "match"].nunique()),
           "unequal_team_minutes": {str(s): int(df.loc[df["state"] == s, "dur"].sum()) for s in (1, -1, 2, -2)}}
    res["main"], main_fit = fit(df)
    res["no_strength_offset"], _ = fit(df, offset=False)
    res["late_interaction"], _ = fit(df, extra=("late",))
    res["home_interaction"], _ = fit(df, extra=("home",))
    res["per_cutoff"] = {}
    for c in CUTOFFS:
        r, _ = fit(df[df["Season"] < c])
        res["per_cutoff"][str(c)] = {k: r[k]["factor"] for k in ("state+1", "state-1", "state+2", "state-2")}
    m = res["main"]
    lu, ld = abs(np.log(m["state+1"]["factor"])), abs(np.log(m["state-1"]["factor"]))
    res["hypotheses"] = {
        "i_10man_further_from_1": {"abs_log_f_up": round(lu, 3), "abs_log_f_down": round(ld, 3), "holds": bool(ld > lu)},
        "ranges": {"f_up_in_0.65_0.80": 0.65 <= m["state+1"]["factor"] <= 0.80,
                   "f_down_in_1.4_2.0": 1.4 <= m["state-1"]["factor"] <= 2.0}}
    (OUT / "factors_estimate.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    print(main_fit.summary())


if __name__ == "__main__":
    main()
