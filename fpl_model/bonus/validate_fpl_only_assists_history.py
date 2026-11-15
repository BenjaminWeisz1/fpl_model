"""
FPL-only assists on 2017-2024 via the vaastav archive: A2 vs A3 (pre-registered in CHANGELOG §94).

vaastav rows (FPL-definition assists) are matched to Sofascore played rows of the same team-match by
optimal assignment on name similarity, with minutes agreeing within 10. FPL-only = vaastav assists -
Sofascore goalAssist. Penalty data and the conventional part come through each Sofascore row's
Local_Player (player_match_stats.csv and the current code's pre-match log). Rolling one-season-ahead
folds inside 2017-2024; validity is checked before any arm is compared.

Run locally (after fetch_vaastav.py):
    python validate_fpl_only_assists_history.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linear_sum_assignment

from fpl_model.bonus import validate_bonus_calibration as calibration
from fpl_model.bonus import validate_fpl_only_assists as v

from fpl_model.fetch.build_position_crosswalk import name_similarity  # noqa: E402

RAW = calibration.ROOT / "data" / "raw" / "vaastav"
OUT_DIR = v.OUT_DIR
SEASONS = list(range(2017, 2025))
TEAM = {"Cardiff": "Cardiff City", "Leeds": "Leeds United", "Leicester": "Leicester City", "Luton": "Luton Town",
        "Man City": "Manchester City", "Man Utd": "Manchester Utd", "Newcastle": "Newcastle Utd",
        "Norwich": "Norwich City", "Nott'm Forest": "Nott'ham Forest", "Spurs": "Tottenham", "Stoke": "Stoke City",
        "Swansea": "Swansea City", "Ipswich": "Ipswich Town", "Hull": "Hull City"}
MIN_TOL = 10
NAME_MIN = 0.5


def read_csv_any(path):
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def load_vaastav():
    teams = pd.read_csv(RAW / "master_team_list.csv")
    out = []
    for season in SEASONS:
        tag = f"{season}-{str(season + 1)[2:]}"
        g = read_csv_any(RAW / tag / "gws" / "merged_gw.csv")
        g = g[g["minutes"] > 0].copy()
        g["name"] = g["name"].astype(str).str.replace(r"_\d+$", "", regex=True).str.replace("_", " ")
        if "team" in g.columns:
            g["team_name"] = g["team"]
            other = g.groupby("fixture")["team_name"].agg(lambda s: sorted(set(s)))
            g["opp_name"] = [next((t for t in other[f] if t != tm), None) for f, tm in zip(g["fixture"], g["team_name"])]
        else:
            ids = g.groupby("fixture")["opponent_team"].agg(lambda s: sorted(set(s)))
            g["team_id"] = [next((t for t in ids[f] if t != o), None) for f, o in zip(g["fixture"], g["opponent_team"])]
            names = teams[teams["season"] == tag].set_index("team")["team_name"]
            g["team_name"], g["opp_name"] = g["team_id"].map(names), g["opponent_team"].map(names)
        g["Team"] = g["team_name"].map(lambda t: TEAM.get(t, t))
        g["Opponent"] = g["opp_name"].map(lambda t: TEAM.get(t, t))
        g["Venue"] = np.where(g["was_home"].astype(str).str.lower().isin(["true", "1"]), "Home", "Away")
        g["Season"] = season
        out.append(g[["Season", "Team", "Opponent", "Venue", "name", "minutes", "assists", "fixture", "kickoff_time"]])
    return pd.concat(out, ignore_index=True)


def match(vs, so):
    """Optimal one-to-one assignment within each team-match; minutes must agree within MIN_TOL."""
    key = ["Season", "Team", "Opponent", "Venue"]
    so_groups = {k: g for k, g in so.groupby(key)}
    pairs = []
    for k, g in vs.groupby(key):
        s = so_groups.get(k)
        if s is None:
            continue
        sim = np.array([[name_similarity(a, b) if abs(ma - mb) <= MIN_TOL else 0.0
                         for b, mb in zip(s["Player_Sofascore"], s["mins"])] for a, ma in zip(g["name"], g["minutes"])])
        r, c = linear_sum_assignment(sim, maximize=True)
        for i, j in zip(r, c):
            if sim[i, j] >= NAME_MIN:
                pairs.append((g.index[i], s.index[j], sim[i, j]))
    p = pd.DataFrame(pairs, columns=["v_idx", "s_idx", "score"])
    m = so.loc[p["s_idx"]].reset_index(drop=True)
    m["Assists"] = vs.loc[p["v_idx"], "assists"].to_numpy()
    m["Minutes"] = vs.loc[p["v_idx"], "minutes"].to_numpy()
    m["match_score"] = p["score"].to_numpy()
    return m


def attach(m):
    pms = pd.read_csv(calibration.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                      usecols=["MatchID", "Season", "Player", "Team", "Opponent", "Venue", "PK", "PKatt", "Data_Source"])
    pms = pms[pms["Data_Source"].eq("fbref")]
    pms["Season"] = 2017 + pms["MatchID"] // 380
    key = ["Season", "Team", "Opponent", "Venue", "Player"]
    m = m.rename(columns={"Local_Player": "Player"}).merge(pms.drop_duplicates(key)[key + ["MatchID", "PK", "PKatt"]],
                                                           on=key, how="left")
    log = pd.read_csv(v.RATING_LOG, encoding="utf-8-sig")
    log = log[log["Data_Source"].eq("fbref")]
    lkey = ["Player", "Team", "Opponent", "Venue", "MatchID"]
    log = log.drop_duplicates(lkey, keep=False)
    m = m.merge(log[lkey + ["pred_ast_90"]], on=lkey, how="left")
    m["conv_part"] = m["pred_ast_90"] * m["Minutes"] / 90
    m["Pos"] = m["Position_Sofascore"]
    m = m[m["Pos"].isin(["G", "D", "M", "F"])].copy()
    m["conv_part"] = m["conv_part"].fillna(m.groupby(["Season", "Pos"])["conv_part"].transform("median"))
    m["PK"], m["PKatt"] = m["PK"].fillna(0), m["PKatt"].fillna(0)
    m["Match_Key"] = m["Season"].astype(str) + "_" + m["MatchID"].astype("Int64").astype(str)
    m["fpl_only"] = (m["Assists"] - m["conv_assists"]).clip(lower=0)
    return m


def fold(fit, test, prev):
    """A0/A2/A3 for one test season; shares from the previous season (prev)."""
    fit, test = v.add_rates(fit, fit), v.add_rates(test, fit)
    out_f = fit[fit["Pos"] != "G"]
    dummies = lambda f: pd.get_dummies(f["Pos"], dtype=float).reindex(columns=["M", "F"], fill_value=0.0)
    d2 = lambda f: sm.add_constant(pd.concat([dummies(f), f[v.VOLUMES]], axis=1), has_constant="add")
    d3 = lambda f: sm.add_constant(pd.concat([dummies(f), f[["rebound_rate"]]], axis=1), has_constant="add")
    offset = np.log(out_f["Minutes"].clip(lower=1) / 90)
    glm2 = sm.GLM(out_f["fpl_only"], d2(out_f), family=sm.families.Poisson(), offset=offset).fit()

    def identified(f):
        mate = f.groupby(["Match_Key", "Team"])["PK"].transform("sum") - f["PK"]
        return np.minimum(f["pens_won"], mate).clip(lower=0)

    def shares(src, f):
        att = src.groupby(["Team", "Player"])["PKatt"].sum()
        s = (att / att.groupby(level=0).transform("sum").where(lambda t: t > 0)).fillna(0)
        return pd.Series(s.reindex(pd.MultiIndex.from_arrays([f["Team"], f["Player"]])).to_numpy(), index=f.index).fillna(0)

    # in the fit window each season's own shares; in the test season the previous season's
    fit["share"] = pd.concat([shares(g, g) for _, g in fit.groupby("Season")])
    test["share"] = shares(prev, test)
    fit["pen_assist"], test["pen_assist"] = identified(fit), identified(test)
    c = float(fit["pen_assist"].sum() / (fit["pens_won"] * (1 - fit["share"])).sum())
    out_f = fit[fit["Pos"] != "G"]
    glm3 = sm.GLM((out_f["fpl_only"] - out_f["pen_assist"]).clip(lower=0), d3(out_f), family=sm.families.Poisson(),
                  offset=np.log(out_f["Minutes"].clip(lower=1) / 90)).fit()
    outfield = ~test["Pos"].eq("G")
    rate2 = np.exp(d2(test) @ glm2.params.to_numpy())
    rate3 = np.exp(d3(test) @ glm3.params.to_numpy()) + test["pens_rate"] * (1 - test["share"]) * c
    test["A0"] = test["conv_part"]
    test["A2"] = test["conv_part"] + np.where(outfield, rate2 * test["Minutes"] / 90, 0.0)
    test["A3"] = test["conv_part"] + np.where(outfield, rate3 * test["Minutes"] / 90, 0.0)
    return test, {"c": round(c, 4), "glm2": glm2.params.round(4).to_dict(), "glm3": glm3.params.round(4).to_dict()}


def main():
    vs = load_vaastav()
    so = v.sofascore_history()
    so = so[so["Season"].between(2017, 2024)].copy()
    m = match(vs, so)
    total_mins = so["mins"].sum()
    coverage = float(m["mins"].sum() / total_mins)
    by_season = m.groupby("Season").agg(fpl=("Assists", "sum"), conv=("conv_assists", "sum"))
    validity = {"minutes_coverage": round(coverage, 4), "coverage_ge_95": coverage >= 0.95,
                "fpl_ge_conventional_every_season": bool((by_season["fpl"] >= by_season["conv"]).all()),
                "season_totals": by_season.astype(int).to_dict("index")}
    out = {"vaastav_rows_with_minutes": int(len(vs)), "matched_rows": int(len(m)),
           "match_score_median": float(m["match_score"].median()), "validity": validity}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not (validity["coverage_ge_95"] and validity["fpl_ge_conventional_every_season"]):
        out["stopped"] = "validity check failed; arms not compared"
        (OUT_DIR / "history_results.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        print(json.dumps(out, indent=2, default=float))
        return
    m = attach(m)
    out["rows_after_attach"] = int(len(m))
    out["conventional_part_from_log"] = float(m["pred_ast_90"].notna().mean())
    out["fpl_over_conventional_by_season"] = (m.groupby("Season")["Assists"].sum() / m.groupby("Season")["conv_assists"].sum()).round(3).to_dict()
    tests, folds = [], {}
    for s in range(2018, 2025):
        t, info = fold(m[m["Season"] < s].copy(), m[m["Season"] == s].copy(), m[m["Season"] == s - 1])
        folds[s] = {**info, "A3_vs_A2": v.compare(t, "A2", "A3"), "A2_vs_A0": v.compare(t, "A0", "A2"),
                    "rows": int(len(t))}
        tests.append(t)
    pooled = pd.concat(tests, ignore_index=True)
    pooled["taker"] = np.where(pooled["share"] >= 0.5, "taker", "non_taker")
    cal = {k: {"rows": int(len(g)), "fpl_only_actual": int(g["fpl_only"].sum()),
               "A2": round(float((g["A2"] - g["A0"]).sum() / g["fpl_only"].sum()), 3),
               "A3": round(float((g["A3"] - g["A0"]).sum() / g["fpl_only"].sum()), 3)}
           for k, g in pooled.groupby("taker")}
    out.update({"folds": folds, "pooled": {"A3_vs_A2": v.compare(pooled, "A2", "A3"), "A2_vs_A0": v.compare(pooled, "A0", "A2"),
                                           "A3_vs_A0": v.compare(pooled, "A0", "A3"), "fpl_only_calibration_by_taker": cal}})
    p = out["pooled"]
    out["decision"] = {"A3_mse_within_0.2pct": p["A3_vs_A2"]["change_pct"] <= 0.2,
                       "takers_calibration_error_shrinks": abs(cal["taker"]["A3"] - 1) < abs(cal["taker"]["A2"] - 1)}
    out["decision"]["adopt_A3"] = all(out["decision"].values())
    m[["Season", "Team", "Opponent", "Venue", "Player", "Pos", "Minutes", "Assists", "conv_assists", "fpl_only",
       "pens_won", "PK", "PKatt", "conv_part", "match_score"]].to_csv(OUT_DIR / "history_matched_rows.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "history_results.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v_ for k, v_ in out.items() if k != "folds"}, indent=2, default=float))
    for s, f in folds.items():
        print(s, f["rows"], "c", f["c"], "A3vsA2", round(f["A3_vs_A2"]["change_pct"], 2), "A2vsA0", round(f["A2_vs_A0"]["change_pct"], 2))


if __name__ == "__main__":
    main()
