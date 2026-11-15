"""CHANGELOG §163: opponent shots, shot quality and keeper shot-stopping, against §162's fixture model.

Walk-forward at realized minutes. Test only; no production writes.
Run: python -u -X utf8 scripts/assemble/validate_keeper_shots.py
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from research.assemble import validate_keeper_saves as K

OUT = K.ROOT / "data" / "validation" / "keeper_shots_20261101"
OUT.mkdir(parents=True, exist_ok=True)
WINDOW, PRIOR = 19, 8
GK_WINDOW, GK_PRIOR = 38, 19
TERMS = {"ctrl": ["lO"], "s": ["lO", "lOppSoT", "lOwnSoTA"], "q": ["lO", "lOppSoT", "lOwnSoTA", "lOppXGShot"], "k": ["lO", "KeeperGP"]}


def shots_table():
    s = pd.read_csv(K.DATA / "sofascore_player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=K.KEY + ["Sofascore_onTargetScoringAttempt", "Sofascore_shotOffTarget", "Sofascore_blockedScoringAttempt"])
    s = s.fillna(0).groupby(K.KEY, as_index=False).sum(numeric_only=True)
    s["SoT"] = s.Sofascore_onTargetScoringAttempt
    s["Shots"] = s.SoT + s.Sofascore_shotOffTarget + s.Sofascore_blockedScoringAttempt
    H = pd.read_csv(K.DATA / "team_ratings_history.csv", encoding="utf-8-sig", low_memory=False)
    a = H.assign(Team=H.Home, Opponent=H.Away, Venue="Home", xG=H.Home_NPxG_Adj)
    b = H.assign(Team=H.Away, Opponent=H.Home, Venue="Away", xG=H.Away_NPxG_Adj)
    t = pd.concat([a, b], ignore_index=True)[K.KEY + ["MatchID", "xG"]]
    t = t.merge(s[K.KEY + ["SoT", "Shots"]], on=K.KEY, how="left", validate="one_to_one")
    print("team-matches", len(t), "with shots", int(t.Shots.notna().sum()), "with xG", int(t.xG.notna().sum()))
    assert t.Shots.notna().mean() > .999 and t.xG.notna().mean() > .999
    return t.sort_values("MatchID").reset_index(drop=True)


def league_before(t, col):
    m = t.groupby("MatchID")[col].sum().sort_index()
    n = t.groupby("MatchID")[col].count().sort_index()
    return t.MatchID.map(m.cumsum().shift(1) / n.cumsum().shift(1))


def trailing(t, col, by, league, spell):
    g = t.groupby([by, spell])[col]
    n = g.transform(lambda s: s.notna().rolling(WINDOW, min_periods=1).sum().shift(1)).fillna(0)
    total = g.transform(lambda s: s.rolling(WINDOW, min_periods=1).sum().shift(1)).fillna(0)
    return (total + PRIOR * league) / (n + PRIOR), total, n


def team_features(t, spells):
    t = t.merge(spells, on=["MatchID", "Team"], how="left", validate="one_to_one")
    assert t.Spell.notna().all()
    league = {c: league_before(t, c) for c in ("SoT", "Shots", "xG")}
    t["SoTFor"], _, _ = trailing(t, "SoT", "Team", league["SoT"], "Spell")
    sf = t.groupby("Team")  # noqa: F841
    xg_total = trailing(t, "xG", "Team", league["xG"], "Spell")[1]
    shot_total = trailing(t, "Shots", "Team", league["Shots"], "Spell")[1]
    n = trailing(t, "Shots", "Team", league["Shots"], "Spell")[2]
    lxs = league["xG"] / league["Shots"]
    t["XGShotFor"] = (xg_total + PRIOR * league["xG"]) / (shot_total + PRIOR * league["Shots"])
    del n, lxs
    # What the side concedes: the opponent's row of the same match.
    against = t[["MatchID", "Team", "SoT"]].rename(columns={"Team": "Opponent", "SoT": "SoTA"})
    t = t.merge(against, on=["MatchID", "Opponent"], how="left", validate="one_to_one")
    t["SoTAgainst"], _, _ = trailing(t, "SoTA", "Team", league["SoT"], "Spell")
    opp = t[["MatchID", "Team", "SoTFor", "XGShotFor"]].rename(columns={"Team": "Opponent", "SoTFor": "OppSoT", "XGShotFor": "OppXGShot"})
    t = t.merge(opp, on=["MatchID", "Opponent"], how="left", validate="one_to_one")
    return t[K.KEY + ["OppSoT", "OppXGShot", "SoTAgainst"]].rename(columns={"SoTAgainst": "OwnSoTA"})


def keeper_table():
    e = pd.read_csv(K.DATA / "sofascore_player_extra.csv.gz", low_memory=False,
                    usecols=["Season", "Event_ID_Sofascore", "Player_ID_Sofascore", "Position", "minutesPlayed", "goalsPrevented", "expectedGoalsOnTarget"])
    e = e[(e.Position == "G") & (e.minutesPlayed.fillna(0) > 0)]
    s = pd.read_csv(K.DATA / "sofascore_player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=K.KEY + ["Event_ID_Sofascore", "Player_ID_Sofascore"])
    e = e.merge(s, on=["Season", "Event_ID_Sofascore", "Player_ID_Sofascore"], how="left", validate="one_to_one")
    assert e.Team.notna().all()
    return e


def keeper_features(e, x):
    order = x.groupby(K.KEY, as_index=False).MatchID.first()
    k = e.merge(order, on=K.KEY, how="inner", validate="many_to_one").sort_values("MatchID")
    k["Min90"] = k.minutesPlayed / 90
    k["GP"] = k.goalsPrevented
    k["SeenMin"] = k.Min90.where(k.GP.notna(), 0.0)
    g = k.groupby("Player_ID_Sofascore")
    gp = g.GP.transform(lambda s: s.fillna(0).rolling(GK_WINDOW, min_periods=1).sum().shift(1)).fillna(0)
    mn = g.SeenMin.transform(lambda s: s.rolling(GK_WINDOW, min_periods=1).sum().shift(1)).fillna(0)
    k["KeeperGP"] = gp / (mn + GK_PRIOR)
    main = k.sort_values("minutesPlayed").drop_duplicates(K.KEY, keep="last")
    return main[K.KEY + ["KeeperGP"]], k


def predict(x, s, terms, first=2018):
    tr = x[(x.Season < s) & (x.Season >= first)]
    def X(d):
        return sm.add_constant(d[terms], has_constant="add")
    m = sm.GLM(tr.Saves, X(tr), family=sm.families.Poisson(), offset=np.log(tr.Exposure)).fit()
    raw = lambda d: np.exp((X(d) * m.params).sum(axis=1) - m.params["const"])
    ref = float(np.average(raw(tr), weights=tr.Exposure))
    te = x[x.Season == s]
    return te.P * raw(te) / ref, m


def run(x, seasons, arms, first=2018):
    out, coef = [], []
    for s in seasons:
        te = x[x.Season == s].copy()
        for a in arms:
            r, m = predict(x, s, TERMS[a], first)
            te["r_" + a] = r
            coef.append({"Season": s, "Arm": a, **{k: float(v) for k, v in m.params.items()}, **{k + "_se": float(v) for k, v in m.bse.items() if k != "const"}})
        out.append(te)
    t = pd.concat(out, ignore_index=True)
    for a in arms:
        lam = (t["r_" + a] * t.Exposure).values
        t["pts_" + a], t["ll_" + a] = K.expected_points_per_three_saves(lam), K.poisson_ll(t.Saves.values, lam)
    t["Q"] = pd.qcut(np.log(t["T"] / t.O), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    t["Cluster"] = t.Team + "_" + t.Season.astype(str)
    return t, pd.DataFrame(coef)


def judge(t, later, arm):
    p = K.paired(t, "ctrl", arm)
    by = {int(s): float(((g.Pts - g["pts_" + arm]) ** 2).mean() - ((g.Pts - g.pts_ctrl) ** 2).mean()) for s, g in t.groupby("Season")}
    s0, s1 = K.spread(t, "ctrl"), K.spread(t, arm)
    l = K.paired(later, "ctrl", arm)
    checks = {"i": p["mse_hi"] < 0, "ii": sum(v < 0 for v in by.values()) >= 4, "iii": p["ll_lo"] > 0, "iv": abs(s1["gap"]) <= abs(s0["gap"]) + .02,
              "v": abs(s1["mean_bias"]) <= abs(s0["mean_bias"]) + .01, "vi": l["mse_pct"] <= 0}
    return {"pooled": p, "by_season": by, "spread": s1, "later": l, "checks": checks, "passes": all(checks.values())}


def keeper_diagnostics(x, e, k):
    res = {}
    g = e[e.Season == 2022].merge(x.groupby(K.KEY, as_index=False).agg(GA=("Pts", "size")), on=K.KEY)
    H = pd.read_csv(K.DATA / "team_ratings_history.csv", encoding="utf-8-sig", low_memory=False)
    a = H.assign(Team=H.Home, Opponent=H.Away, Venue="Home", Against=H.Away_Goals)
    b = H.assign(Team=H.Away, Opponent=H.Home, Venue="Away", Against=H.Home_Goals)
    ga = pd.concat([a, b])[K.KEY + ["Against"]]
    d = e.merge(ga, on=K.KEY, validate="many_to_one")
    d = d[d.goalsPrevented.notna() & d.expectedGoalsOnTarget.notna() & (d.minutesPlayed >= 90)]
    res["gp_definition"] = {"rows": len(d), "corr_xgot_minus_goals": float(d.goalsPrevented.corr(d.expectedGoalsOnTarget - d.Against)), "mean_gap": float((d.goalsPrevented - (d.expectedGoalsOnTarget - d.Against)).mean())}
    del g
    # Goals conceded route: one row a team-match, main keeper.
    m = x.sort_values("Minutes").drop_duplicates(K.KEY, keep="last").merge(ga, on=K.KEY, validate="one_to_one")
    m = m[(m.Season >= 2023) & m.KeeperGP.notna()].copy()
    fit = sm.GLM(m.Against, sm.add_constant(pd.DataFrame({"lO": m.lO, "KeeperGP": m.KeeperGP})), family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(m.Team + m.Season.astype(str))[0]})
    base = sm.GLM(m.Against, sm.add_constant(m[["lO"]]), family=sm.families.Poisson()).fit()
    res["goals_conceded_in_sample"] = {"rows": len(m), "coef": float(fit.params["KeeperGP"]), "se": float(fit.bse["KeeperGP"]), "lO": float(fit.params["lO"]), "ll_gain": float((fit.llf - base.llf) / len(m))}
    # Season-to-season keeper GP/90.
    v = e[e.goalsPrevented.notna()].groupby(["Player_ID_Sofascore", "Season"]).agg(n=("goalsPrevented", "size"), gp=("goalsPrevented", "sum"), m=("minutesPlayed", "sum")).reset_index()
    v = v[(v.n >= 15) & (v.Season.between(2022, 2025))].assign(r=lambda f: 90 * f.gp / f.m)
    p = v.merge(v.assign(Season=v.Season - 1), on=["Player_ID_Sofascore", "Season"], suffixes=("", "2"))
    res["gp_persistence"] = {"pairs": len(p), "r": float(p.r.corr(p.r2)), "sd_gp90": float(v.r.std())}
    return res


def main():
    pd.set_option("display.width", 200, "display.max_columns", 30)
    x = K.frame()
    spells = x[["MatchID", "Team", "Spell"]].drop_duplicates()
    t_raw = shots_table()
    # A team-match without a keeper row can't happen (§162 asserts it), so spells cover every row.
    f = team_features(t_raw, spells)
    x = x.merge(f, on=K.KEY, how="left", validate="many_to_one")
    e = keeper_table()
    kp, k = keeper_features(e, x)
    x = x.merge(kp, on=K.KEY, how="left", validate="many_to_one")
    x = x[x.Season >= 2018].reset_index(drop=True)
    x["lO"] = np.log(x.O)
    for c in ("OppSoT", "OwnSoTA", "OppXGShot"):
        assert x[c].notna().all(), c
        x["l" + c] = np.log(x[c])
    print(x[["lO", "lOppSoT", "lOwnSoTA", "lOppXGShot"]].corr().round(2).to_string())
    print("KeeperGP SD", round(float(x[x.Season >= 2023].KeeperGP.std()), 4), "null", int(x[x.Season >= 2023].KeeperGP.isna().sum()))

    t, coef = run(x, K.TEST + K.LATER, ["ctrl", "s", "q"])
    main_rows, later = t[t.Season.isin(K.TEST)].reset_index(drop=True), t[t.Season.isin(K.LATER)].reset_index(drop=True)
    res = {"rows": len(main_rows), "later_rows": len(later), "control_spread": K.spread(main_rows, "ctrl"), "control_mse": float(((main_rows.Pts - main_rows.pts_ctrl) ** 2).mean())}
    res["s"], res["q"] = judge(main_rows, later, "s"), judge(main_rows, later, "q")
    res["q_v_s"], res["q_v_s_later"] = K.paired(main_rows, "s", "q"), K.paired(later, "s", "q")
    print(coef.round(3).to_string(index=False))

    x2 = x.copy()
    x2["KeeperGP"] = x2.KeeperGP.fillna(0.0)
    tk, ck = run(x2[x2.Season >= 2022], [2023, 2024, 2025, 2026], ["ctrl", "k"], first=2022)
    pk = K.paired(tk, "ctrl", "k")
    byk = {int(s): float(((g.Pts - g.pts_k) ** 2).mean() - ((g.Pts - g.pts_ctrl) ** 2).mean()) for s, g in tk.groupby("Season")}
    full = sum(byk[s] < 0 for s in (2023, 2024, 2025))
    res["k"] = {"rows": len(tk), "pooled": pk, "by_season": byk, "passes": bool(pk["mse_hi"] < 0 and pk["ll_lo"] > 0 and full >= 2)}
    print(ck.round(3).to_string(index=False))
    res["keeper"] = keeper_diagnostics(x2, e, k)
    res["coef"], res["coef_k"] = coef.to_dict("records"), ck.to_dict("records")
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({a: v for a, v in res.items() if not a.startswith("coef")}, indent=1, default=float))


if __name__ == "__main__":
    main()
