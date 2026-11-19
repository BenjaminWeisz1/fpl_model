"""§155 supplements: rest counted from league matches only (the agenda's first step), and within-player output per 90 on
short rest (exploratory; expected -3% to 0%). No production writes.
Run: python -u -X utf8 -m research.absence.congestion_players
"""
import json

import pandas as pd
import statsmodels.formula.api as smf

from fpl_model import paths
from research.absence import build_absence_frame as B
from research.absence import validate_congestion as V


def main():
    tm = V.build()
    tm = tm.sort_values(["Team", "date"])
    tm["RestPL"] = tm.groupby("Team").date.diff().dt.days.fillna(30).clip(upper=30)
    tm["PL3H"] = ((tm.RestPL <= 3) & (tm.Home == 1)).astype(int)
    tm["PL3A"] = ((tm.RestPL <= 3) & (tm.Home == 0)).astype(int)
    tm["Mid"] = tm.RestPL.between(4, 5).astype(int)
    m = V.fit(tm, "ResDiff ~ PL3H + PL3A + Mid + C(Season) + C(DoW) + Home")
    res = {"league_only": {"n_le3": int((tm.RestPL <= 3).sum()), "home": V.row(m, "PL3H"), "away": V.row(m, "PL3A")}}

    s = B.load_squads()
    p = pd.read_csv(paths.current_file("player_match_stats.csv", root=B.DATA), encoding="utf-8-sig", low_memory=False,
                    usecols=["Season", "Player", "Team", "Opponent", "Venue", "Minutes", "npxG", "xAG", "CBI", "Tkl", "Recov", "Data_Source"])
    j = p.merge(s[["Season", "Local_Player", "Team", "Opponent", "Venue", "PID", "Is_Starter"]], left_on=["Season", "Player", "Team", "Opponent", "Venue"],
                right_on=["Season", "Local_Player", "Team", "Opponent", "Venue"], how="inner").drop_duplicates(["Season", "Team", "Opponent", "Venue", "PID"])
    j = j[j.Is_Starter & (j.Minutes >= 60)].merge(tm[["Season", "Team", "Opponent", "Venue", "Rest", "Home", "Pred", "PredA"]], on=["Season", "Team", "Opponent", "Venue"])
    j["Attack90"] = (j.npxG.fillna(0) + j.xAG.fillna(0)) * 90 / j.Minutes
    j["Def90"] = (j.CBI.fillna(0) + j.Tkl.fillna(0) + j.Recov.fillna(0)) * 90 / j.Minutes
    j["Short"] = (j.Rest <= 3).astype(int)
    j["PS"] = j.PID.astype(str) + "_" + j.Season.astype(str)
    out = {}
    for y in ("Attack90", "Def90"):
        cols = [y, "Short", "Home", "Pred", "PredA"]
        d = j[cols].astype(float) - j.groupby("PS")[cols].transform("mean").astype(float)
        d["PS"] = j.PS.values
        m = smf.ols(f"{y} ~ Short + Home + Pred + PredA - 1", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.PS)[0]})
        ci = m.conf_int().loc["Short"]
        base = float(j[y].mean())
        out[y] = {"b": float(m.params.Short), "lo": float(ci[0]), "hi": float(ci[1]), "mean": base, "pct": float(m.params.Short / base * 100),
                  "pct_lo": float(ci[0] / base * 100), "pct_hi": float(ci[1] / base * 100), "n": int(m.nobs)}
    res["players_within"] = out
    (B.OUT / "congestion_supplement.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
