"""Fixture congestion, pre-registered in CHANGELOG §155 (AGENDA 2026-10-31 item 6). Diagnostic: no production writes.
Calendar: all competitions from Transfermarkt plus the League Cup from openfootball, 2018/19-2025/26. Outcome: team npxG
difference minus the pre-match rating prediction. Run: python -u -X utf8 scripts/absence/validate_congestion.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
from research.absence import build_absence_frame as B

OUT, DATA = B.OUT, B.DATA
RAW = B.ROOT / "data" / "raw"
SEASONS = range(2018, 2026)
MONTHS = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
EUROPE = {"CL", "EL", "UCOL", "CLQ", "ELQ", "ECLQ", "USC"}
CUP = {"FAC", "CGB", "GBCS", "KLUB"}
pd.set_option("display.width", 200, "display.max_columns", 30)


def our_name(name, teams):
    n = B.TM_TEAMS.get(name, name)
    if n in teams:
        return n
    t = set(B.norm(name)) - {"fc", "afc"}
    for c in teams:
        if set(B.norm(c)) - {"fc", "afc"} == t:
            return c
    alias = {"manchester city": "Manchester City", "man city": "Manchester City", "manchester united": "Manchester Utd", "newcastle united": "Newcastle Utd",
             "tottenham hotspur": "Tottenham", "wolverhampton wanderers": "Wolves", "brighton hove albion": "Brighton", "west ham united": "West Ham",
             "sheffield united": "Sheffield Utd", "nottingham forest": "Nott'ham Forest", "west bromwich albion": "West Brom", "huddersfield town": "Huddersfield",
             "afc bournemouth": "Bournemouth", "leeds united": "Leeds United", "leicester city": "Leicester City", "norwich city": "Norwich City"}
    return alias.get(" ".join(B.norm(name)).replace(" fc", ""))


def league_cup():
    rows = []
    for path in sorted((RAW / "openfootball").glob("eflcup_*.txt")):
        start = int(path.stem.split("_")[1][:4])
        date = None
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) (\w{3}) (\d{1,2})(?: (\d{4}))?\s*$", line)
            if m:
                mon = MONTHS[m[1]]
                date = pd.Timestamp(int(m[3]) if m[3] else (start if mon >= 7 else start + 1), mon, int(m[2]))
                continue
            m = re.match(r"\s+(?:\d{1,2}:\d{2}\s+)?(.+?)\s+v\s+(.+?)\s{2,}(\S.*)$", line)
            if m and date is not None and "cancel" not in m[3].lower():
                rows.append({"date": date, "home": m[1].strip(), "away": m[2].strip(), "comp": "CGB"})
    return pd.DataFrame(rows)


def calendar():
    g = pd.read_csv(RAW / "transfermarkt" / "games.csv.gz", low_memory=False, usecols=["competition_id", "date", "home_club_name", "away_club_name"])
    g = g.rename(columns={"home_club_name": "home", "away_club_name": "away", "competition_id": "comp"})
    g = g[g.comp.isin(EUROPE | CUP)]
    g["date"] = pd.to_datetime(g.date)
    all_cup = pd.concat([g, league_cup()], ignore_index=True)
    hist = pd.read_csv(DATA / "team_ratings_history.csv", encoding="utf-8-sig", usecols=["Season", "Home", "Away"])
    teams = set(hist.Home)
    rows = []
    for _, r in all_cup.iterrows():
        for side, other, venue in ((r.home, r.away, "Home"), (r.away, r.home, "Away")):
            t = our_name(side, teams) if isinstance(side, str) else None
            if t:
                rows.append({"Team": t, "date": r.date, "comp": r.comp, "Venue": venue})
    c = pd.DataFrame(rows).drop_duplicates(["Team", "date", "comp"])
    return c


def build():
    tm = pd.read_pickle(OUT / "team_matches.pkl")
    tm = tm[tm.Season.isin(SEASONS)].copy()
    tm["date"] = tm.Kickoff.dt.tz_localize(None).dt.normalize()
    allm = B.team_matches().merge(
        B.load_squads().groupby(["Season", "Team", "Opponent", "Venue"], as_index=False).Kickoff.first(), on=["Season", "Team", "Opponent", "Venue"])
    league = allm.assign(date=allm.Kickoff.dt.tz_localize(None).dt.normalize(), comp="GB1")[["Team", "date", "comp", "Venue"]]
    cal = pd.concat([league, calendar()], ignore_index=True).sort_values(["Team", "date"]).drop_duplicates(["Team", "date"], keep="first")
    print(cal.comp.value_counts().to_dict(), flush=True)
    d, nd = {}, {}
    for t, g in cal.groupby("Team"):
        dates = g.date.values
        d[t] = (dates, g.comp.values, g.Venue.values)
    prev, ptype, nxt_cl, nxt_any = [], [], [], []
    for t, day in zip(tm.Team, tm.date):
        ds, cs, vs = d[t]
        i = np.searchsorted(ds, np.datetime64(day))
        gap = (np.datetime64(day) - ds[i - 1]) / np.timedelta64(1, "D") if i > 0 else 30
        prev.append(gap)
        c, v = (cs[i - 1], vs[i - 1]) if i > 0 else ("none", "none")
        ptype.append("EuAway" if c in EUROPE and v == "Away" else "EuHome" if c in EUROPE else "Cup" if c in CUP else "League")
        j = i + 1
        cl = 0
        while j < len(ds) and (ds[j] - np.datetime64(day)) / np.timedelta64(1, "D") <= 3:
            cl = max(cl, int(cs[j] == "CL"))
            j += 1
        nxt_cl.append(cl)
    tm["Rest"], tm["Prev"], tm["CLNext3"] = prev, ptype, nxt_cl
    tm["Rest"] = tm.Rest.clip(upper=30)
    tm["Bin"] = np.select([tm.Rest <= 3, tm.Rest == 4, tm.Rest == 5], ["R3", "R4", "R5"], "R6")
    o = tm[["Season", "Team", "Opponent", "Venue", "Bin", "CLNext3", "Prev"]].rename(columns={"Team": "Opponent", "Opponent": "Team", "Bin": "OppBin", "CLNext3": "OppCL", "Prev": "OppPrev"})
    o["Venue"] = o.Venue.map({"Home": "Away", "Away": "Home"})
    tm = tm.merge(o, on=["Season", "Team", "Opponent", "Venue"], how="left")
    tm["Home"] = (tm.Venue == "Home").astype(int)
    tm["DoW"] = tm.Kickoff.dt.dayofweek
    tm["TS"] = tm.Team + tm.Season.astype(str)
    tm["Cat"] = np.where(tm.Rest > 3, tm.Bin, np.where(tm.Prev == "EuAway", "R3EuAway", np.where(tm.Prev == "EuHome", "R3EuHome", np.where(tm.Prev == "Cup", "R3Cup", "R3League"))))
    tm["OppR3"] = (tm.OppBin == "R3").astype(int)
    return tm


def fit(d, formula):
    d = d.dropna(subset=["OppBin", "ResDiff"])
    return smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.TS)[0]})


def row(m, k):
    c90 = m.conf_int(alpha=.10).loc[k]
    c = m.conf_int().loc[k]
    return {"b": float(m.params[k]), "se": float(m.bse[k]), "lo95": float(c[0]), "hi95": float(c[1]), "lo90": float(c90[0]), "hi90": float(c90[1])}


def main():
    tm = build()
    print("rows", len(tm), tm.Bin.value_counts().to_dict(), tm.Cat.value_counts().to_dict(), "CLNext3", int(tm.CLNext3.sum()), flush=True)
    C = "C(Season) + C(DoW) + Home"
    res = {"n": len(tm), "residual_sd": float(tm.ResDiff.std()), "bins": tm.Bin.value_counts().to_dict(), "cats": tm.Cat.value_counts().to_dict(), "cl_next3": int(tm.CLNext3.sum())}
    for y in ("ResDiff", "ResFor", "ResAgainst"):
        tm["R3Home"], tm["R3Away"] = ((tm.Bin == "R3") & (tm.Home == 1)).astype(int), ((tm.Bin == "R3") & (tm.Home == 0)).astype(int)
        m1 = fit(tm, f"{y} ~ R3Home + R3Away + C(Bin, Treatment('R6'))[T.R4] + C(Bin, Treatment('R6'))[T.R5] + OppR3 + CLNext3 + OppCL + {C}".replace("C(Bin, Treatment('R6'))[T.R4] + C(Bin, Treatment('R6'))[T.R5]", "I(Bin=='R4') + I(Bin=='R5')"))
        m2 = fit(tm, f"{y} ~ C(Cat, Treatment('R6')) + OppR3 + CLNext3 + OppCL + {C}")
        res[y] = {"rest_le3_home": row(m1, "R3Home"), "rest_le3_away": row(m1, "R3Away"), "cl_next3": row(m1, "CLNext3"),
                  "euro_away_le3": row(m2, "C(Cat, Treatment('R6'))[T.R3EuAway]"), "euro_home_le3": row(m2, "C(Cat, Treatment('R6'))[T.R3EuHome]"),
                  "cup_le3": row(m2, "C(Cat, Treatment('R6'))[T.R3Cup]"), "league_le3": row(m2, "C(Cat, Treatment('R6'))[T.R3League]")}
        if y == "ResDiff":
            s = tm.dropna(subset=["gapC", "gapC_opp"])
            a = fit(s, f"{y} ~ R3Home + R3Away + I(Bin=='R4') + I(Bin=='R5') + OppR3 + CLNext3 + OppCL + {C}")
            b = fit(s, f"{y} ~ R3Home + R3Away + I(Bin=='R4') + I(Bin=='R5') + OppR3 + CLNext3 + OppCL + gapC + gapC_opp + {C}")
            a2 = fit(s, f"{y} ~ C(Cat, Treatment('R6')) + OppR3 + CLNext3 + OppCL + {C}")
            b2 = fit(s, f"{y} ~ C(Cat, Treatment('R6')) + OppR3 + CLNext3 + OppCL + gapC + gapC_opp + {C}")
            res["lineup_mediation"] = {"rest_le3_away": {"without": row(a, "R3Away"), "with": row(b, "R3Away")}, "rest_le3_home": {"without": row(a, "R3Home"), "with": row(b, "R3Home")},
                                       "cl_next3": {"without": row(a, "CLNext3"), "with": row(b, "CLNext3")},
                                       "euro_away_le3": {"without": row(a2, "C(Cat, Treatment('R6'))[T.R3EuAway]"), "with": row(b2, "C(Cat, Treatment('R6'))[T.R3EuAway]")},
                                       "share_of_lineup_gap_by_rest": tm.groupby("Bin").gapC.mean().round(3).to_dict()}
    (OUT / "congestion_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    def rnd(o):
        return {k: rnd(v) for k, v in o.items()} if isinstance(o, dict) else round(o, 3) if isinstance(o, float) else o
    for k in ("ResDiff", "ResFor", "ResAgainst", "lineup_mediation"):
        print(k, json.dumps(rnd(res[k])))
    print("n", res["n"], "sd", round(res["residual_sd"], 3))


if __name__ == "__main__":
    main()
