"""§154 part 5, exploratory: how teammates' shares of team npxG, xAG and defcon move with a regular out.
Share baseline = the player's mean share over his earlier starts in the team's last 8 matches. No production writes."""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import build_absence_frame as B

xi = pd.read_pickle(B.OUT / "xi.pkl")
s = B.load_squads()
p = pd.read_csv(B.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                usecols=["Season", "Player", "Team", "Opponent", "Venue", "Minutes", "npxG", "xAG", "CBI", "Tkl", "Recov"])
j = p.merge(s[["Season", "Local_Player", "Team", "Opponent", "Venue", "PID", "Kickoff", "Is_Starter", "Pos"]], left_on=["Season", "Player", "Team", "Opponent", "Venue"],
            right_on=["Season", "Local_Player", "Team", "Opponent", "Venue"], how="inner").drop_duplicates(["Season", "Team", "Opponent", "Venue", "PID"])
keys = ["Season", "Team", "Opponent", "Venue"]
j["Def"] = j.CBI.fillna(0) + j.Tkl.fillna(0) + j.Recov.fillna(0)
for c in ("npxG", "xAG", "Def"):
    j[c] = j[c].fillna(0)
    j["S_" + c] = j[c] / j.groupby(keys)[c].transform("sum").replace(0, np.nan)
j = j[j.Is_Starter].sort_values("Kickoff")
for c in ("npxG", "xAG", "Def"):
    j["B_" + c] = j.groupby(["Team", "PID"])["S_" + c].transform(lambda x: x.shift(1).rolling(8, min_periods=4).mean())
    j["D_" + c] = j["S_" + c] - j["B_" + c]
e = xi[(xi.Set == "E") & (xi.Share >= .6)]
ab = e[~e.Why.isin(["started", "rested"])]
cnt = ab.groupby(keys + ["Pos"]).size().unstack(fill_value=0).reset_index().rename(columns={c: "abs" + c for c in "GDMF"})
d = j.merge(cnt, on=keys, how="left")
for c in "GDMF":
    d["abs" + c] = d["abs" + c].fillna(0)
reg = set(zip(e.Event, e.PID))
d = d.merge(s[keys + ["Event"]].drop_duplicates(), on=keys)
d["Regular"] = [(a, b) in reg for a, b in zip(d.Event, d.PID)]
d["TS"] = d.Team + d.Season.astype(str)
out = {}
for c in ("npxG", "xAG", "Def"):
    res = {}
    for pos in "DMF":
        g = d[d.Regular & (d.Pos == pos)].dropna(subset=["D_" + c])
        m = smf.ols(f"D_{c} ~ absD + absM + absF", g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.TS)[0]})
        res[pos] = {k: [round(float(m.params[k]), 4), round(float(m.bse[k]), 4)] for k in ("absD", "absM", "absF")}
        res[pos]["n"] = int(m.nobs)
    # Replacements: their share that day against the absent regular's own baseline.
    out[c] = res
act = xi[(xi.Set == "A")]
rep = act[~act.PID.isin([])].merge(xi[xi.Set == "E"][keys + ["PID"]].assign(inE=1), on=keys + ["PID"], how="left")
rep = rep[rep.inE.isna()].merge(j[keys + ["PID", "S_npxG", "S_xAG", "S_Def"]], on=keys + ["PID"], how="inner")
abs_base = ab.merge(j.groupby(["Team", "PID"], as_index=False)[["S_npxG", "S_xAG", "S_Def"]].mean(), on=["Team", "PID"], how="inner")
out["replacement_mean_share"] = rep[["S_npxG", "S_xAG", "S_Def"]].mean().round(4).to_dict()
out["absentee_usual_share"] = abs_base[["S_npxG", "S_xAG", "S_Def"]].mean().round(4).to_dict()
print(json.dumps(out, indent=0))
(B.OUT / "teammate_shares.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
