"""
Log-form money prior for promoted teams only (pre-registered in CHANGELOG §207).

Part A (decisive): on the four other leagues' promoted team-seasons 2017-2025, the log-relative prior (§206 M2)
against a straight line on value share (§206 M1's form), each league fitted on its own earlier seasons.
Part B (only if A passes): Premier League production priors with only promoted team-seasons replaced by the log
prior, level-matched per season, through update_team_ratings.py on a scratch copy; walk-forward log loss.

Run locally:
    python -m research.team_ratings.validate_promoted_log_prior
"""
import json

import numpy as np
import pandas as pd

from fpl_model import paths
from research.team_ratings import validate_money_priors_pooled as vmp

OUT = vmp.ROOT / "data" / "validation" / "promoted_log_prior_20261106"
OTHER = ["ES1", "L1", "IT1", "FR1"]


def predictions(t, leagues):
    rows = []
    has = t["V"] > 0
    for s in vmp.TEST:
        for comp in leagues:
            test = t[has & (t["competition_id"] == comp) & (t["season"] == s)].copy()
            if test.empty:
                continue
            own = t[has & (t["competition_id"] == comp) & (t["season"] < s)]
            p = vmp.predict_loglinear(t, test, own, False)
            test["G_GF"], test["G_GA"] = p["GF"], p["GA"]
            for target in ("GF", "GA"):
                X = np.column_stack([np.ones(len(own)), own["share"]])
                c, *_ = np.linalg.lstsq(X, own[target], rcond=None)
                test["L_" + target] = c[0] + c[1] * test["share"]
            rows.append(test)
    return pd.concat(rows, ignore_index=True)


def part_b(t):
    from fpl_model.team_ratings import update_team_ratings as utr
    p = predictions(t, ["GB1"])
    p["Team"] = p["name"].map(vmp.TM_TO_PROJECT)
    assert p["Team"].notna().all()
    prod = pd.read_csv(paths.current_file("money_priors_by_season.csv", root=vmp.CUR))
    m = prod.merge(p[["season", "Team", "promoted", "G_GF", "G_GA"]].rename(columns={"season": "Season"}),
                   on=["Season", "Team"], how="left")
    sel = m["G_GF"].notna()
    assert sel.sum() == 20 * len(vmp.TEST)
    new = m.copy()
    for col, g in (("Money_GF", "G_GF"), ("Money_GA", "G_GA")):
        scale = m[sel].groupby("Season")[col].transform("mean") / m[sel].groupby("Season")[g].transform("mean")
        swap = sel & m["promoted"].eq(True)
        new.loc[swap, col] = (m.loc[sel, g] * scale)[swap[sel].to_numpy()].to_numpy()
    changed = new[sel & m["promoted"].eq(True)][["Season", "Team"]].assign(
        GF_old=m["Money_GF"], GF_new=new["Money_GF"], GA_old=m["Money_GA"], GA_new=new["Money_GA"])
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
        pd.read_csv(paths.current_file("team_ratings_current.csv", root=vmp.CUR)), on="Team", suffixes=("", "_p"))
    assert len(c) == 20 and np.allclose(c[["Attack_Rating", "Defense_Rating"]], c[["Attack_Rating_p", "Defense_Rating_p"]],
                                        atol=1e-12), "control doesn't reproduce production"
    A = run("candidate_promoted_log", new)
    assert (P["MatchID"].values == A["MatchID"].values).all()
    ll = lambda h: ((h.Home_Prediction - h.Home_NPG * np.log(h.Home_Prediction))
                    + (h.Away_Prediction - h.Away_NPG * np.log(h.Away_Prediction))) / 2
    d = ll(A) - ll(P)
    prev = {s: set(P[P.Season == s - 1].Home) for s in P.Season.unique()}
    prom = pd.Series([(h not in prev.get(s, set())) or (a not in prev.get(s, set()))
                      for s, h, a in zip(P.Season, P.Home, P.Away)])
    win = P.Season.between(2017, 2025)
    rng = np.random.default_rng(vmp.SEED)
    res = {"changed_priors": changed.round(3).to_dict("records")}
    for name, msk in {"all": win, "promoted_matches": win & prom, "other_matches": win & ~prom}.items():
        x = d[msk].to_numpy()
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(vmp.REPS)]
        res[name] = {"n": int(msk.sum()), "delta": round(float(x.mean()), 5),
                     "ci95": np.percentile(b, [2.5, 97.5]).round(5).tolist()}
    res["by_season_promoted"] = {int(s): round(float(d[win & prom & (P.Season == s)].mean()), 5)
                                 for s in range(2017, 2026)}
    res["rule_pass"] = bool(res["promoted_matches"]["ci95"][1] < 0 and res["all"]["delta"] <= 0)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = vmp.build_table()
    p = predictions(t, OTHER)
    pr = p[p["promoted"]].copy()
    pr.to_csv(OUT / "part_a_promoted_predictions.csv", index=False, encoding="utf-8-sig")
    res = {"part_a": {"pooled": vmp.score(pr, ["L", "G"]), "ci95_G_minus_L": vmp.boot(pr, "G", "L"),
                      "by_league": {c: vmp.score(g, ["L", "G"]) for c, g in pr.groupby("competition_id")},
                      "all_teams_pooled": vmp.score(p, ["L", "G"])}}
    a = res["part_a"]
    a["leagues_won"] = int(sum(v["G"]["mse"] < v["L"]["mse"] for v in a["by_league"].values()))
    a["rel_pct"] = round(100 * (a["pooled"]["G"]["mse"] / a["pooled"]["L"]["mse"] - 1), 2)
    a["pass"] = bool(a["pooled"]["G"]["mse"] < a["pooled"]["L"]["mse"] and a["ci95_G_minus_L"][1] < 0
                     and a["leagues_won"] >= 3)
    print(json.dumps(res, indent=1), flush=True)
    if a["pass"]:
        res["part_b"] = part_b(t)
        print(json.dumps(res["part_b"], indent=1), flush=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
