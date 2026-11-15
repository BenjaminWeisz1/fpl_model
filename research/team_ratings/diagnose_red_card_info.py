"""
Diagnostic (CHANGELOG §137): does a red card predict a team's next match beyond its rating?

Per team-match 2018-2026, from team_ratings_history.csv (chronological file order): the next match's attack residual
(own NPxG_Adj - own predicted goals) and defence residual (conceded NPxG_Adj - opponent's predicted goals), regressed
by OLS on two indicators for the team's previous match in the same season: it had a player sent off, and its opponent
did. SEs clustered by team-season. Report only.

    python -m research.team_ratings.diagnose_red_card_info     # -> data/validation/red_card_20261030/red_card_info.json
"""
import json

import pandas as pd
import statsmodels.formula.api as smf

from fpl_model import paths

ROOT = paths.ROOT
DATA = paths.CURRENT
OUT = paths.VALIDATION / "red_card_20261030"


def main():
    h = pd.read_csv(DATA / "team_ratings_history.csv", encoding="utf-8-sig").reset_index(names="order")
    rc = pd.read_csv(DATA / "red_card_factors.csv", encoding="utf-8-sig")
    reds = {(r.Season, r.Home, r.Away): ("(home)" in r.Reds, "(away)" in r.Reds) for r in rc.itertuples()}
    rows = []
    for r in h.itertuples():
        hr, ar = reds.get((r.Season, r.Home, r.Away), (False, False))
        rows.append({"order": r.order, "Season": r.Season, "Team": r.Home, "own_red": hr, "opp_red": ar,
                     "att": r.Home_NPxG_Adj - r.Home_Prediction, "dfn": r.Away_NPxG_Adj - r.Away_Prediction})
        rows.append({"order": r.order, "Season": r.Season, "Team": r.Away, "own_red": ar, "opp_red": hr,
                     "att": r.Away_NPxG_Adj - r.Away_Prediction, "dfn": r.Home_NPxG_Adj - r.Home_Prediction})
    t = pd.DataFrame(rows).sort_values(["Team", "Season", "order"])
    g = t.groupby(["Team", "Season"])
    t["own_red_lag"] = g["own_red"].shift(1)
    t["opp_red_lag"] = g["opp_red"].shift(1)
    t = t[t["own_red_lag"].notna() & (t["Season"] >= 2018)].copy()
    t[["own_red_lag", "opp_red_lag"]] = t[["own_red_lag", "opp_red_lag"]].astype(int)
    t["cluster"] = t["Team"] + "_" + t["Season"].astype(str)
    res = {"rows": len(t), "own_red_lag_n": int(t["own_red_lag"].sum()), "opp_red_lag_n": int(t["opp_red_lag"].sum())}
    for y in ("att", "dfn"):
        f = smf.ols(f"{y} ~ own_red_lag + opp_red_lag", data=t).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(t["cluster"])[0]})
        ci = f.conf_int()
        res[y] = {k: {"coef": round(float(f.params[k]), 4), "ci95": [round(float(ci.loc[k, 0]), 4),
                                                                     round(float(ci.loc[k, 1]), 4)],
                      "p": round(float(f.pvalues[k]), 3)} for k in ("own_red_lag", "opp_red_lag")}
        res[y]["mean_residual"] = round(float(t[y].mean()), 4)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "red_card_info.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
