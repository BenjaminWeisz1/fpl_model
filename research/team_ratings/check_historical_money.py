"""
Data checks on historical_money.csv (CHANGELOG §209). Read-only diagnostic: duplicates, goals against Transfermarkt
games, squad values against Transfermarkt pre-season matchday squads (§206 team_seasons.csv), year-on-year share jumps,
wage-v-value mismatches. Run from the project root: python -m research.team_ratings.check_historical_money
"""
import numpy as np
import pandas as pd

from fpl_model import paths
from research.team_ratings import validate_money_priors_pooled as vmp

m = pd.read_csv(paths.current_file("historical_money.csv"), encoding="utf-8-sig")
print("dups", m.duplicated(["Season","Team"]).sum(), "rows/season", m.groupby("Season").size().unique())
k = m[m.GF.notna()]
g = k.groupby("Season")[["GF","GA"]].sum(); print("GF!=GA seasons:\n", g[g.GF!=g.GA])
print("non-integer goals", (k.GF%1!=0).sum(), (k.GA%1!=0).sum())
# vs Transfermarkt games
t = pd.read_csv(vmp.OUT/"team_seasons.csv", encoding="utf-8-sig"); t = t[t.competition_id=="GB1"].copy()
t["Team"] = t.name.map(lambda n: vmp.TM_TO_PROJECT.get(n, n))
x = k.merge(t[["season","Team","GF","GA","n","V","promoted"]].rename(columns={"season":"Season"}), on=["Season","Team"], how="outer", suffixes=("","_tm"), indicator=True)
print("unmatched:\n", x[x._merge!="both"][["Season","Team","_merge"]])
b = x[x._merge=="both"].copy()
b["dGF"] = b.GF - b.GF_tm*b.n; b["dGA"] = b.GA - b.GA_tm*b.n
print("goal mismatches:\n", b[(b.dGF.abs()>0.5)|(b.dGA.abs()>0.5)][["Season","Team","GF","GA","dGF","dGA","n"]])
# value vs TM pre-season squad value
b["ratio"] = b.Squad_Value / (b.V/1e6)
print("value ratio by season (median, min, max):\n", b.groupby("Season").ratio.agg(["median","min","max"]).round(2))
b["rr"] = b.ratio / b.groupby("Season").ratio.transform("median")
print("value outliers vs TM (ratio/season median <0.6 or >1.6):\n", b[(b.rr<0.6)|(b.rr>1.6)][["Season","Team","Squad_Value","V","rr"]].assign(V=lambda d:(d.V/1e6).round(1)).round(2))
# within-team year-on-year jumps
m = m.sort_values(["Team","Season"])
for c in ("Wages","Squad_Value"):
    m[c+"_share"] = m[c]/m.groupby("Season")[c].transform("sum")
    m[c+"_yoy"] = m.groupby("Team")[c+"_share"].pct_change()
    cons = m.Season.diff().eq(1) & m.Team.eq(m.Team.shift())
    j = m[cons & (m[c+"_yoy"].abs()>0.6)]
    print(f"{c} share jumps >60% y/y:\n", j[["Season","Team",c,c+"_yoy"]].round(2))
print("season totals:\n", m.groupby("Season")[["Wages","Squad_Value"]].agg(["sum","min","max"]))
lw = np.log(m.Wages_share); lv = np.log(m.Squad_Value_share); r = lw - lv
m["wv"] = (r - r.groupby(m.Season).transform("mean")); print("wage-vs-value mismatch |log|>0.7:\n", m[m.wv.abs()>0.7][["Season","Team","Wages","Squad_Value","wv"]].round(2))
print("\nvalue ratio extremes rr>1.35 or <0.75:\n", b[(b.rr>1.35)|(b.rr<0.75)][["Season","Team","Squad_Value","V","rr","promoted"]].assign(V=lambda d:(d.V/1e6).round(1)).round(2).sort_values("rr"))
w = m.pivot(index="Team", columns="Season", values="Wages")
print("\nwages 2024-2026:\n", w[[2024,2025,2026]].dropna(how="all", subset=[2026]).sort_values(2026))
print("\nLeeds wages:", w.loc["Leeds United"].dropna().to_dict())
pr = b[b.promoted][["Season","Team","Wages","Squad_Value"]]
pr["wshare_rank"] = [ (m[m.Season==s].Wages.rank()[m[(m.Season==s)&(m.Team==t)].index[0]]) for s,t in zip(pr.Season, pr.Team)]
print("\npromoted wages:\n", pr.sort_values("Season").to_string())
