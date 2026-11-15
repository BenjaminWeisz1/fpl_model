"""
Money priors from pre-season squad values, five leagues pooled (pre-registered in CHANGELOG §206).

Stage 1 scores season-level priors (GF and GA per match) on Premier League team-seasons 2017-2025:
M0 production priors; M1 EPL OLS on value share; M2 EPL log-relative; M3 M2 + promoted term;
M4 M3 pooled over five leagues (primary); M4n M4 without the promoted term. Every test season is
fitted on earlier seasons only. Stage 2 (run only if stage 1 passes) reruns update_team_ratings.py on
a scratch copy with the chosen arm's priors, level-matched per season to production, and scores
walk-forward Poisson log loss 2017-2025.

Run locally:
    python validate_money_priors_pooled.py            # stage 1, then stage 2 if the rule passes
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "transfermarkt"
CUR = ROOT / "data" / "current"
OUT = ROOT / "data" / "validation" / "money_priors_pooled_20261106"
LEAGUES = ["GB1", "ES1", "L1", "IT1", "FR1"]
FIRST_MATCHES = 3
MAX_AGE_DAYS = 400
TEST = range(2017, 2026)
SEED, REPS = 42, 2000
TM_TO_PROJECT = {
    "Arsenal FC": "Arsenal", "Aston Villa": "Aston Villa", "AFC Bournemouth": "Bournemouth", "Brentford FC": "Brentford",
    "Brighton & Hove Albion": "Brighton", "Burnley FC": "Burnley", "Cardiff City": "Cardiff City",
    "Chelsea FC": "Chelsea", "Crystal Palace": "Crystal Palace", "Everton FC": "Everton", "Fulham FC": "Fulham",
    "Huddersfield Town": "Huddersfield", "Hull City": "Hull City", "Ipswich Town": "Ipswich Town",
    "Leeds United": "Leeds United", "Leicester City": "Leicester City", "Liverpool FC": "Liverpool",
    "Luton Town": "Luton Town", "Manchester City": "Manchester City", "Manchester United": "Manchester Utd",
    "Middlesbrough FC": "Middlesbrough", "Newcastle United": "Newcastle Utd", "Norwich City": "Norwich City",
    "Nottingham Forest": "Nott'ham Forest", "Queens Park Rangers": "QPR", "Sheffield United": "Sheffield Utd",
    "Southampton FC": "Southampton", "Stoke City": "Stoke City", "Sunderland AFC": "Sunderland",
    "Swansea City": "Swansea City", "Tottenham Hotspur": "Tottenham", "Watford FC": "Watford",
    "West Bromwich Albion": "West Brom", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"}


# ---------------------------------------------------------------- data
def build_table():
    g = pd.read_csv(RAW / "games.csv.gz", usecols=["game_id", "competition_id", "season", "date", "home_club_id",
                                                   "away_club_id", "home_club_goals", "away_club_goals",
                                                   "home_club_name", "away_club_name"])
    g = g[g["competition_id"].isin(LEAGUES) & g["season"].between(2012, 2025)].copy()
    g["date"] = pd.to_datetime(g["date"])
    side = pd.concat([
        g.rename(columns={"home_club_id": "club", "home_club_goals": "gf", "away_club_goals": "ga",
                          "home_club_name": "name"})[["game_id", "competition_id", "season", "date", "club", "gf", "ga", "name"]],
        g.rename(columns={"away_club_id": "club", "away_club_goals": "gf", "home_club_goals": "ga",
                          "away_club_name": "name"})[["game_id", "competition_id", "season", "date", "club", "gf", "ga", "name"]]])
    key = ["competition_id", "season", "club"]
    t = side.groupby(key).agg(GF=("gf", "mean"), GA=("ga", "mean"), n=("gf", "size"), name=("name", "first")).reset_index()
    prev = set(zip(t["competition_id"], t["season"] + 1, t["club"]))
    t["promoted"] = [(c, s, k) not in prev for c, s, k in zip(t["competition_id"], t["season"], t["club"])]
    start = g.groupby(["competition_id", "season"])["date"].min().rename("start")
    t = t.merge(start, on=["competition_id", "season"])

    first = side.sort_values("date").groupby(key).head(FIRST_MATCHES)
    lu = pd.read_csv(RAW / "game_lineups.csv.gz", usecols=["game_id", "player_id", "club_id"])
    lu = lu[lu["game_id"].isin(set(first["game_id"]))]
    sq = first[["game_id"] + key].merge(lu, left_on=["game_id", "club"], right_on=["game_id", "club_id"])
    sq = sq.drop_duplicates(key + ["player_id"]).merge(start, on=["competition_id", "season"])
    v = pd.read_csv(RAW / "player_valuations.csv.gz", usecols=["player_id", "date", "market_value_in_eur"])
    v["date"] = pd.to_datetime(v["date"])
    sq = pd.merge_asof(sq.sort_values("start"), v.sort_values("date"), left_on="start", right_on="date", by="player_id",
                       allow_exact_matches=False, direction="backward")
    ok = sq["date"].notna() & ((sq["start"] - sq["date"]).dt.days <= MAX_AGE_DAYS)
    sq["value"] = np.where(ok, sq["market_value_in_eur"], np.nan)
    agg = sq.groupby(key).agg(squad=("player_id", "size"), valued=("value", "count"), V=("value", "sum")).reset_index()
    t = t.merge(agg, on=key, how="left")
    t = t[t["season"] >= 2013].reset_index(drop=True)
    t["z"] = np.log(t["V"]) - t.groupby(["competition_id", "season"])["V"].transform(lambda x: np.log(x).mean())
    t["share"] = t["V"] / t.groupby(["competition_id", "season"])["V"].transform("sum")
    return t


# ---------------------------------------------------------------- arms
def level(t, comp, s):
    """League goals per team-match over its three previous seasons."""
    d = t[(t["competition_id"] == comp) & t["season"].between(s - 3, s - 1)]
    return float((d["GF"] * d["n"]).sum() / d["n"].sum())


def fit_loglinear(train, target, promoted):
    lm = train.groupby(["competition_id", "season"])[target].transform("mean")
    y = np.log(train[target] / lm)
    X = np.column_stack([np.ones(len(train)), train["z"]] + ([train["promoted"].astype(float)] if promoted else []))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    smear = float(np.exp(y - X @ coef).mean())
    return coef, smear


def predict_loglinear(t, test, train, promoted):
    out = {}
    for target in ("GF", "GA"):
        coef, smear = fit_loglinear(train, target, promoted)
        X = np.column_stack([np.ones(len(test)), test["z"]] + ([test["promoted"].astype(float)] if promoted else []))
        lv = np.array([level(t, c, s) for c, s in zip(test["competition_id"], test["season"])])
        out[target] = lv * np.exp(X @ coef) * smear
        out[target + "_coef"] = coef.round(4).tolist()
    return out


def stage1_predictions(t, m0):
    rows, coefs = [], {}
    for s in TEST:
        for comp in LEAGUES:
            has = t["V"] > 0  # La Liga 2018 has no lineups for its first matches: out of fits and scoring (§206)
            test = t[has & (t["competition_id"] == comp) & (t["season"] == s)].copy()
            if test.empty:
                continue
            own = t[has & (t["competition_id"] == comp) & (t["season"] < s)]
            pooled = t[has & (t["season"] < s)]
            for arm, train, prom in [("M2", own, False), ("M3", own, True), ("M4", pooled, True), ("M4n", pooled, False)]:
                p = predict_loglinear(t, test, train, prom)
                test[arm + "_GF"], test[arm + "_GA"] = p["GF"], p["GA"]
                coefs[f"{arm}_{comp}_{s}"] = {"GF": p["GF_coef"], "GA": p["GA_coef"]}
            if comp == "GB1":
                for target in ("GF", "GA"):
                    X = np.column_stack([np.ones(len(own)), own["share"]])
                    c, *_ = np.linalg.lstsq(X, own[target], rcond=None)
                    test["M1_" + target] = c[0] + c[1] * test["share"]
            rows.append(test)
    p = pd.concat(rows, ignore_index=True)
    p["Team"] = p["name"].map(TM_TO_PROJECT)
    epl = p["competition_id"] == "GB1"
    assert p.loc[epl, "Team"].notna().all(), "unmapped Premier League club"
    p = p.merge(m0.rename(columns={"Season": "season", "Money_GF": "M0_GF", "Money_GA": "M0_GA"}),
                on=["season", "Team"], how="left")
    assert p.loc[epl, "M0_GF"].notna().all(), "production prior missing for a Premier League team-season"
    return p, coefs


def sq_err(d, arm):
    return (d[arm + "_GF"] - d["GF"]) ** 2 + (d[arm + "_GA"] - d["GA"]) ** 2


def boot(d, a, b):
    diff = (sq_err(d, a) - sq_err(d, b)).to_numpy()
    rng = np.random.default_rng(SEED)
    return np.percentile([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(REPS)], [2.5, 97.5]).round(5).tolist()


def score(d, arms):
    out = {"n": int(len(d))}
    for a in arms:
        out[a] = {"mse": round(float(sq_err(d, a).mean()), 5),
                  "mse_GF": round(float(((d[a + "_GF"] - d["GF"]) ** 2).mean()), 5),
                  "mse_GA": round(float(((d[a + "_GA"] - d["GA"]) ** 2).mean()), 5),
                  "bias_GF": round(float((d[a + "_GF"] - d["GF"]).mean()), 4),
                  "bias_GA": round(float((d[a + "_GA"] - d["GA"]).mean()), 4)}
    return out


# ---------------------------------------------------------------- stage 2
def stage2(p, arm):
    from fpl_model.team_ratings import update_team_ratings as utr
    prod = pd.read_csv(CUR / "money_priors_by_season.csv")
    cand = p[p["competition_id"] == "GB1"][["season", "Team", arm + "_GF", arm + "_GA"]].rename(
        columns={"season": "Season", arm + "_GF": "GF", arm + "_GA": "GA"})
    new = prod.merge(cand, on=["Season", "Team"], how="left")
    sel = new["GF"].notna()
    assert sel.sum() == 20 * len(TEST), "candidate priors don't cover every 2017-2025 team-season"
    for col, src in (("Money_GF", "GF"), ("Money_GA", "GA")):
        scale = new[sel].groupby("Season")[col].transform("mean") / new[sel].groupby("Season")[src].transform("mean")
        new.loc[sel, col] = new.loc[sel, src] * scale
    new = new[prod.columns]

    def run(name, money):
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        money.to_csv(d / "money_priors_by_season.csv", index=False)
        utr.MONEY_PRIORS_FILE, utr.OUT_FILE, utr.CURRENT_RATINGS_FILE = (
            d / "money_priors_by_season.csv", d / "team_ratings_history.csv", d / "team_ratings_current.csv")
        utr.main()
        h = pd.read_csv(d / "team_ratings_history.csv", low_memory=False)
        return h[h["Home_NPG"].notna() & h["Home_Prediction"].notna()].reset_index(drop=True)

    P = run("control_production_priors", prod)
    c = pd.read_csv(OUT / "control_production_priors" / "team_ratings_current.csv").merge(
        pd.read_csv(CUR / "team_ratings_current.csv"), on="Team", suffixes=("", "_p"))
    assert len(c) == 20 and np.allclose(c[["Attack_Rating", "Defense_Rating"]], c[["Attack_Rating_p", "Defense_Rating_p"]],
                                        atol=1e-12), "control doesn't reproduce production"
    A = run("candidate_" + arm, new)
    assert (P["MatchID"].values == A["MatchID"].values).all()
    ll = lambda h: ((h.Home_Prediction - h.Home_NPG * np.log(h.Home_Prediction))
                    + (h.Away_Prediction - h.Away_NPG * np.log(h.Away_Prediction))) / 2
    d = ll(A) - ll(P)
    prev = {s: set(P[P.Season == s - 1].Home) for s in P.Season.unique()}
    prom = pd.Series([(h not in prev.get(s, set())) or (a not in prev.get(s, set()))
                      for s, h, a in zip(P.Season, P.Home, P.Away)])
    n = P.groupby("Season").cumcount()
    win = P.Season.between(2017, 2025)
    rng = np.random.default_rng(SEED)
    res = {}
    for name, m in {"all": win, "promoted_matches": win & prom, "first_50": win & (n < 50),
                    "rest": win & ~prom & (n >= 50)}.items():
        x = d[m].to_numpy()
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(REPS)]
        res[name] = {"n": int(m.sum()), "delta": round(float(x.mean()), 5),
                     "ci95": np.percentile(b, [2.5, 97.5]).round(5).tolist(), "base": round(float(ll(P)[m].mean()), 5)}
    res["rule_pass"] = bool(res["all"]["delta"] <= -0.001 and res["all"]["ci95"][1] < 0
                            and res["promoted_matches"]["delta"] <= 0)
    return res


def main():
    t = build_table()
    OUT.mkdir(parents=True, exist_ok=True)
    cov = t.groupby("competition_id").agg(team_seasons=("club", "size"), promoted=("promoted", "sum"),
                                          squad=("squad", "mean"), valued_share=("valued", "sum"))
    cov["valued_share"] = (t.groupby("competition_id")["valued"].sum() / t.groupby("competition_id")["squad"].sum()).round(3)
    res = {"coverage": cov.round(2).reset_index().to_dict("records"),
           "min_valued_share_team_season": round(float((t["valued"] / t["squad"]).min()), 3),
           "missing_squads": int(t["V"].isna().sum())}
    print(json.dumps(res, indent=1, default=str), flush=True)
    t.to_csv(OUT / "team_seasons.csv", index=False, encoding="utf-8-sig")
    if "--coverage" in sys.argv:
        return

    m0 = pd.read_csv(CUR / "money_priors_by_season.csv", encoding="utf-8-sig")
    p, coefs = stage1_predictions(t, m0)
    p.to_csv(OUT / "stage1_predictions.csv", index=False, encoding="utf-8-sig")
    arms = ["M0", "M1", "M2", "M3", "M4", "M4n"]
    epl = p[p["competition_id"] == "GB1"]
    s1 = {"epl": score(epl, arms), "epl_promoted": score(epl[epl["promoted"]], arms)}
    s1["ci95_vs_M0"] = {a: boot(epl, a, "M0") for a in arms[1:]}
    s1["seasons_won_vs_M0"] = {a: int(sum(sq_err(g, a).mean() < sq_err(g, "M0").mean() for _, g in epl.groupby("season")))
                               for a in arms[1:]}
    oth = p[p["competition_id"] != "GB1"]
    s1["other_leagues"] = {c: score(g, ["M3", "M4", "M4n"]) for c, g in oth.groupby("competition_id")}
    s1["other_leagues_pooled"] = score(oth, ["M3", "M4", "M4n"])
    s1["other_leagues_promoted"] = score(oth[oth["promoted"]], ["M3", "M4", "M4n"])
    s1["coef_M4_2025"] = coefs["M4_GB1_2025"]
    s1["coef_M3_GB1_2025"] = coefs["M3_GB1_2025"]

    def passes(a):
        return (s1["epl"][a]["mse"] < s1["epl"]["M0"]["mse"] and s1["ci95_vs_M0"][a][1] < 0
                and s1["seasons_won_vs_M0"][a] >= 5 and s1["epl_promoted"][a]["mse"] <= s1["epl_promoted"]["M0"]["mse"])
    s1["passes"] = {a: passes(a) for a in arms[1:]}
    choice = "M4" if s1["passes"]["M4"] else min((a for a in arms[1:] if s1["passes"][a]),
                                                 key=lambda a: s1["epl"][a]["mse"], default=None)
    s1["stage2_arm"] = choice
    res["stage1"] = s1
    print(json.dumps(s1, indent=1, default=str), flush=True)
    if choice:
        res["stage2"] = stage2(p, choice)
        print(json.dumps(res["stage2"], indent=1), flush=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
