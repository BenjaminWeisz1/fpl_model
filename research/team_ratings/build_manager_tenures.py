"""Manager tenures from Transfermarkt games (AGENDA 2026-10-31 item 9, CHANGELOG §158). Event flag only; no bounce term.

Writes data/current/manager_tenures.csv: one row per spell, Premier League 2017/18-2025/26. Spells under MIN_SPELL matches
that sit between two spells of the same manager are one-off stand-ins (Ian Woan, Everton 2023/24) and are folded into the
surrounding spell, flagged in Stand_Ins. 2026/27 changes must be added by hand; the source ends in July 2026.
Also runs the §158 check on team residuals. Run: python -u -X utf8 scripts/team_ratings/build_manager_tenures.py
"""
import json

import pandas as pd
import statsmodels.formula.api as smf

from fpl_model import paths

ROOT = paths.ROOT
from research.absence import build_absence_frame as B

MIN_SPELL = 3
OUT = paths.CURRENT / "manager_tenures.csv"


def team_games():
    g = pd.read_csv(ROOT / "data" / "raw" / "transfermarkt" / "games.csv.gz", low_memory=False)
    g = g[(g.competition_id == "GB1") & (g.season >= 2017)]
    a = g[["season", "date", "home_club_name", "home_club_manager_name"]].set_axis(["Season", "Date", "Team", "Manager"], axis=1)
    b = g[["season", "date", "away_club_name", "away_club_manager_name"]].set_axis(["Season", "Date", "Team", "Manager"], axis=1)
    t = pd.concat([a, b], ignore_index=True)
    t["Team"] = t.Team.map(lambda n: B.TM_TEAMS.get(n, n))
    t["Date"] = pd.to_datetime(t.Date)
    assert t.Manager.notna().all()
    return t.sort_values(["Team", "Date"]).reset_index(drop=True)


def spells(t):
    rows = []
    for team, g in t.groupby("Team"):
        run = (g.Manager != g.Manager.shift()).cumsum()
        s = g.groupby(run).agg(Manager=("Manager", "first"), Start=("Date", "min"), End=("Date", "max"), Matches=("Date", "size"), Season=("Season", "first")).reset_index(drop=True)
        merged, stand = [], []
        i = 0
        while i < len(s):
            r = s.iloc[i].to_dict()
            r["Stand_Ins"] = ""
            while i + 2 < len(s) and s.Matches[i + 1] < MIN_SPELL and s.Manager[i + 2] == r["Manager"]:
                r["Stand_Ins"] = (r["Stand_Ins"] + "; " if r["Stand_Ins"] else "") + f"{s.Manager[i + 1]} ({s.Matches[i + 1]})"
                r["End"], r["Matches"] = s.End[i + 2], r["Matches"] + s.Matches[i + 1] + s.Matches[i + 2]
                i += 2
            merged.append(r)
            i += 1
        d = pd.DataFrame(merged)
        d.insert(0, "Team", team)
        d["New_Manager_Event"] = [False] + [True] * (len(d) - 1)
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out["Short_Spell"] = out.Matches < MIN_SPELL
    gap = out.groupby("Team").End.shift()
    out["Returned_After_Relegation"] = (out.Start - gap).dt.days > 200
    out.loc[out.Returned_After_Relegation & (out.Manager == out.groupby("Team").Manager.shift()), "New_Manager_Event"] = False
    return out


def check(sp):
    x = pd.read_pickle(B.OUT / "team_matches.pkl")[["Season", "Team", "Kickoff", "ResDiff"]].sort_values(["Team", "Kickoff"]).reset_index(drop=True)
    x["date"] = x.Kickoff.dt.tz_localize(None).dt.normalize()
    ev = sp[sp.New_Manager_Event & ~sp.Short_Spell & (sp.Season >= 2018)]
    x["Post5"] = 0
    x["Pre5"] = 0
    n = 0
    for _, e in ev.iterrows():
        g = x[x.Team == e.Team]
        after = g[g.date >= e.Start].index[:5]
        before = g[g.date < e.Start].index[-5:]
        if len(after) == 5 and len(before) == 5 and (e.Start - x.date[before[-1]]).days < 60:
            x.loc[after, "Post5"] = 1
            x.loc[before, "Pre5"] = 1
            n += 1
    x["TS"] = x.Team + x.Season.astype(str)
    x["Trail5"] = x.groupby("Team").ResDiff.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    d = x.dropna(subset=["Trail5", "ResDiff"])
    a = smf.ols("ResDiff ~ Post5 + Pre5", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.TS)[0]})
    b = smf.ols("ResDiff ~ Post5 + Trail5", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.TS)[0]})
    out = {"events": n, "pre5": [float(a.params.Pre5), *map(float, a.conf_int().loc["Pre5"])], "post5": [float(a.params.Post5), *map(float, a.conf_int().loc["Post5"])],
           "post5_given_form": [float(b.params.Post5), *map(float, b.conf_int().loc["Post5"])]}
    (B.OUT / "manager_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(out)


def main():
    t = team_games()
    sp = spells(t)
    sp.assign(Start=sp.Start.dt.date, End=sp.End.dt.date).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(len(sp), "spells;", int(sp.New_Manager_Event.sum()), "changes;", int(sp.Short_Spell.sum()), "short spells kept;", int((sp.Stand_Ins != "").sum()), "with folded stand-ins")
    print(sp[sp.Stand_Ins != ""][["Team", "Manager", "Stand_Ins"]].head(12).to_string(index=False))
    print(sp[sp.Short_Spell][["Team", "Manager", "Start", "Matches"]].head(14).to_string(index=False))
    check(sp)


if __name__ == "__main__":
    main()
