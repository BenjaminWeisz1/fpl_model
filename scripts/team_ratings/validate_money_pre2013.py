"""
Money prior with 2011-2012 added as value-only seasons (pre-registered in CHANGELOG §213; user request).

Builds data/current/historical_money_pre2013.csv (Season, Team, GF, GA, Squad_Value; no wages: fbref's wages start
in 2013-14) from football-data.co.uk results (data/raw/football_data/E0_2011.csv, E0_2012.csv) and Transfermarkt's
club market-value page at the 15 Aug cut-off (values below, read 2026-11-09 from
/<league>/marktwerteverein/wettbewerb/GB1..GB4/plus/?stichtag=2011-08-15 and 2012-08-15, rows whose league on that
date was the Premier League). Then scores, on 2017-2025 team-seasons, M0 (production priors), V13 (value-share OLS
on 2013..s-1) and V11 (the same on 2011..s-1).

Run locally:
    python validate_money_pre2013.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_money_priors_pooled as vmp  # noqa: E402
from atomic_write import to_csv_atomic  # noqa: E402

ROOT = vmp.ROOT
CUR = vmp.CUR
OUT = ROOT / "data" / "validation" / "money_pre2013_20261109"
PRE_FILE = CUR / "historical_money_pre2013.csv"
TEST = range(2017, 2026)

# EUR m, Transfermarkt cut-off 15 Aug, project team names
TM_VALUES = {
    2011: {"Manchester City": 441.50, "Arsenal": 279.35, "Chelsea": 397.30, "Liverpool": 293.55, "Manchester Utd": 405.30,
           "Tottenham": 290.00, "Newcastle Utd": 105.50, "Aston Villa": 127.00, "Sunderland": 119.20, "Everton": 167.80,
           "Fulham": 97.00, "Wolves": 79.05, "Swansea City": 28.63, "Norwich City": 22.93, "Stoke City": 83.00,
           "West Brom": 80.75, "QPR": 42.13, "Blackburn": 64.90, "Bolton": 87.60, "Wigan Athletic": 64.55},
    2012: {"Manchester City": 510.00, "Arsenal": 373.25, "Chelsea": 410.50, "Liverpool": 228.10, "Manchester Utd": 398.50,
           "Tottenham": 284.00, "Newcastle Utd": 148.95, "Aston Villa": 116.50, "Sunderland": 94.25, "Everton": 116.50,
           "Fulham": 94.50, "West Ham": 69.95, "Southampton": 43.30, "Swansea City": 67.25, "Norwich City": 59.70,
           "Stoke City": 84.10, "West Brom": 80.00, "QPR": 89.25, "Reading": 32.00, "Wigan Athletic": 77.50},
}
FD_NAMES = {"Man City": "Manchester City", "Man United": "Manchester Utd", "Newcastle": "Newcastle Utd",
            "Norwich": "Norwich City", "Stoke": "Stoke City", "Swansea": "Swansea City", "Wigan": "Wigan Athletic"}


def build_pre2013():
    rows = []
    for s in (2011, 2012):
        d = pd.read_csv(ROOT / "data" / "raw" / "football_data" / f"E0_{s}.csv", encoding="latin-1")
        d = d[d["HomeTeam"].notna()]
        assert len(d) == 380
        for c in ("HomeTeam", "AwayTeam"):
            d[c] = d[c].map(lambda n: FD_NAMES.get(n, n))
        gf = d.groupby("HomeTeam")["FTHG"].sum().add(d.groupby("AwayTeam")["FTAG"].sum())
        ga = d.groupby("HomeTeam")["FTAG"].sum().add(d.groupby("AwayTeam")["FTHG"].sum())
        assert set(gf.index) == set(TM_VALUES[s]), set(gf.index) ^ set(TM_VALUES[s])
        assert gf.sum() == ga.sum()
        for t in sorted(gf.index):
            rows.append((s, t, int(gf[t]), int(ga[t]), TM_VALUES[s][t]))
    pre = pd.DataFrame(rows, columns=["Season", "Team", "GF", "GA", "Squad_Value"])
    to_csv_atomic(pre, PRE_FILE, index=False)
    return pre


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pre = build_pre2013()
    hm = pd.read_csv(CUR / "historical_money.csv", encoding="utf-8-sig")
    allv = pd.concat([pre, hm[["Season", "Team", "GF", "GA", "Squad_Value", "Wages"]]], ignore_index=True)
    allv["V_pct"] = allv.groupby("Season")["Squad_Value"].transform(lambda x: x / x.sum())
    known = allv[allv["GF"].notna()]
    prod = pd.read_csv(CUR / "money_priors_by_season.csv", encoding="utf-8-sig")
    prev = allv.groupby("Season")["Team"].apply(set).to_dict()

    rows = []
    for s in TEST:
        test = known[known["Season"] == s].copy()
        for arm, start in (("V13", 2013), ("V11", 2011)):
            tr = known[(known["Season"] >= start) & (known["Season"] < s)]
            X = np.column_stack([np.ones(len(tr)), tr["V_pct"]])
            Xt = np.column_stack([np.ones(len(test)), test["V_pct"]])
            for t in ("GF", "GA"):
                c, *_ = np.linalg.lstsq(X, tr[t].to_numpy(), rcond=None)
                test[f"{arm}_{t}"] = Xt @ c / 38
        test = test.merge(prod.rename(columns={"Money_GF": "M0_GF", "Money_GA": "M0_GA"}), on=["Season", "Team"], how="left")
        assert test["M0_GF"].notna().all()
        test["promoted"] = ~test["Team"].isin(prev[s - 1])
        test["GF"], test["GA"] = test["GF"] / 38, test["GA"] / 38
        rows.append(test)
    p = pd.concat(rows, ignore_index=True)
    assert len(p) == 180 and p["promoted"].sum() == 27
    p.to_csv(OUT / "stage1_predictions.csv", index=False, encoding="utf-8-sig")
    arms = ["M0", "V13", "V11"]
    pr = p[p["promoted"]]
    won = lambda a, b: int(sum(vmp.sq_err(g, a).mean() < vmp.sq_err(g, b).mean() for _, g in p.groupby("Season")))
    res = {"pre2013_rows": len(pre), "all": vmp.score(p, arms), "promoted": vmp.score(pr, arms),
           "ci95": {"V11_minus_V13": {"all": vmp.boot(p, "V11", "V13"), "promoted": vmp.boot(pr, "V11", "V13")},
                    "V11_minus_M0": {"all": vmp.boot(p, "V11", "M0"), "promoted": vmp.boot(pr, "V11", "M0")}},
           "seasons_won": {"V11_v_V13": won("V11", "V13"), "V11_v_M0": won("V11", "M0")},
           "by_season": {int(s): {a: round(float(vmp.sq_err(g, a).mean()), 4) for a in arms} for s, g in p.groupby("Season")}}
    m = {a: res["all"][a]["mse"] for a in arms}
    res["rel_pct"] = {"V11_v_V13": round(100 * (m["V11"] / m["V13"] - 1), 2), "V11_v_M0": round(100 * (m["V11"] / m["M0"] - 1), 2)}
    res["extra_seasons_help"] = bool(m["V11"] < m["V13"])
    res["stage2"] = bool(m["V11"] < m["M0"] and res["ci95"]["V11_minus_M0"]["all"][1] < 0 and res["seasons_won"]["V11_v_M0"] >= 5
                         and res["promoted"]["V11"]["mse"] <= res["promoted"]["M0"]["mse"])
    print(json.dumps(res, indent=1), flush=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
