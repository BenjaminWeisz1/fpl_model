"""
Diagnostic screen of BPS components for explicit modelling (CHANGELOG §67).

Persistent player differences are already in the per-player residual mean, so a
component earns an explicit term only through (a) shrinkage that differs from the
residual's 10 matches, (b) an opponent effect, or (c) co-movement with the goals
and assists the simulator draws. For each component and position this reports,
per 60+ minute appearance under 2026/27 BPS weights:

  mean        average BPS
  tau         persistent between-player SD: sqrt(cov) of player-season means
              across consecutive seasons (>= 10 appearances each)
  r_yy        season-to-season correlation of those means
  k           matches for 50% reliability (within-player-season variance / tau^2)
  opp_sd      persistent opponent effect: sqrt(cov) of odd/even-match halves of
              opponent-season mean deviations from the player-season mean
  corr_ga     within-player match correlation with goals + assists

Seasons 2018-2024 only, before both bonus test windows. Sofascore components plus
FPL-era-free cards from the frozen fbref rows. Descriptive; selects candidates,
adopts nothing.

Run locally:
    python screen_sofascore_components.py
"""
from fpl_model import paths
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.bonus.validate_bonus_calibration import SLOT_POSITION

ROOT = paths.ROOT
DATA = paths.CURRENT
OUT_DIR = paths.VALIDATION / "sofascore_component_screen_20261028"
SEASONS = range(2018, 2025)
MIN_APPS = 10

WEIGHTS = {  # 2026/27 BPS per event
    "keyPass": 1, "bigChanceCreated": 3, "bigChanceMissed": -3, "accurateCross": 1,
    "wonContest": 1, "onTargetScoringAttempt": 2, "shotOffTarget": -1, "wasFouled": 1,
    "fouls": -1, "totalOffside": -1, "errorLeadToAShot": -1, "errorLeadToAGoal": -3,
    "clearanceOffLine": 9, "penaltyConceded": -3, "wonTackle": 2,
    "savedShotsFromInsideTheBox": 1,
}


def sofascore_components():
    cols = (["Season", "Canonical_Player_ID", "Opponent", "Appearance_Status", "Position_Sofascore",
             "Present_Fields", "Sofascore_minutesPlayed", "Sofascore_totalPass", "Sofascore_accuratePass",
             "Sofascore_totalClearance", "Sofascore_outfielderBlock", "Sofascore_interceptionWon",
             "Sofascore_ballRecovery", "Sofascore_goals", "Sofascore_goalAssist"]
            + ["Sofascore_" + k for k in WEIGHTS])
    so = pd.read_csv(DATA / "sofascore_player_match_stats.csv", encoding="utf-8-sig", usecols=cols, low_memory=False)
    so = so[so["Season"].isin(SEASONS) & so["Appearance_Status"].eq("played")
            & (so["Sofascore_minutesPlayed"] >= 60)].copy()
    so.columns = [c.replace("Sofascore_", "") for c in so.columns]
    presence = {k: float(so[k].notna().mean()) for k in WEIGHTS}
    num = list(WEIGHTS) + ["totalPass", "accuratePass", "totalClearance", "outfielderBlock",
                           "interceptionWon", "ballRecovery", "goals", "goalAssist"]
    so[num] = so[num].fillna(0)  # omitted counters are zero for played rows (SOFASCORE.md)
    comp = pd.DataFrame({k: so[k] * w for k, w in WEIGHTS.items()})
    share = so["accuratePass"] / so["totalPass"].where(so["totalPass"] > 0)
    comp["passBands"] = np.where(so["totalPass"] >= 30,
                                 np.select([share >= .9, share >= .8, share >= .7], [6, 4, 2], 0), 0)
    comp["cbi_div3"] = (so["totalClearance"] + so["outfielderBlock"] + so["interceptionWon"]) // 3
    comp["recov_div3"] = so["ballRecovery"] // 3
    frame = pd.concat([so[["Season", "Opponent"]], comp], axis=1)
    frame["player"] = so["Canonical_Player_ID"]
    frame["pos"] = so["Position_Sofascore"]
    frame["ga"] = so["goals"] + so["goalAssist"]
    return frame, presence


def card_components():
    s = pd.read_csv(DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=["Season", "Player", "Opponent", "Position", "Minutes", "CrdY", "CrdR",
                             "Goals", "Assists", "Data_Source"])
    s = s[s["Data_Source"].eq("fbref") & s["Season"].isin(SEASONS) & (s["Minutes"] >= 60)].copy()
    frame = pd.DataFrame({"Season": s["Season"], "Opponent": s["Opponent"],
                          "cards": -3 * s["CrdY"].fillna(0) - 9 * s["CrdR"].fillna(0)})
    frame["player"] = s["Player"]
    # fbref holds detailed slots (CB, LB, DM, ...); the first listed slot decides.
    frame["pos"] = s["Position"].astype(str).str.split(",").str[0].map(SLOT_POSITION)
    frame["ga"] = s["Goals"].fillna(0) + s["Assists"].fillna(0)
    return frame


def screen(frame, components):
    rows = []
    frame = frame.dropna(subset=["pos"]).copy()
    frame["order"] = frame.groupby(["player", "Season"]).cumcount()
    for pos, pf in frame.groupby("pos"):
        ps = pf.groupby(["player", "Season"])
        size = ps.size()
        for c in components:
            means = ps[c].mean()[size >= MIN_APPS]
            nxt = means.copy()
            nxt.index = pd.MultiIndex.from_arrays([means.index.get_level_values(0),
                                                   means.index.get_level_values(1) - 1])
            pair = pd.concat([means, nxt], axis=1, keys=["t", "t1"]).dropna()
            cov = pair.cov().iloc[0, 1] if len(pair) > 2 else np.nan
            tau2 = max(cov, 0)
            within = ps[c].var().mean()
            dev = pf[c] - ps[c].transform("mean")
            ga_dev = pf["ga"] - ps["ga"].transform("mean")
            halves = pd.DataFrame({"d": dev, "odd": pf["order"] % 2, "opp": pf["Opponent"], "s": pf["Season"]})
            opp = halves.groupby(["opp", "s", "odd"])["d"].mean().unstack()
            opp_cov = opp.cov().iloc[0, 1] if opp.shape == (opp.shape[0], 2) and len(opp) > 2 else np.nan
            rows.append({
                "pos": pos, "component": c, "n_apps": int(len(pf)), "player_season_pairs": int(len(pair)),
                "mean": pf[c].mean(), "tau": np.sqrt(tau2),
                "r_yy": pair["t"].corr(pair["t1"]) if len(pair) > 2 else np.nan,
                "k": within / tau2 if tau2 > 0 else np.inf,
                "opp_sd": np.sqrt(max(opp_cov, 0)) if not np.isnan(opp_cov) else np.nan,
                "corr_ga": dev.corr(ga_dev),
            })
    return pd.DataFrame(rows)


def main():
    so, presence = sofascore_components()
    components = list(WEIGHTS) + ["passBands", "cbi_div3", "recov_div3"]
    so["everything_else"] = so[components].sum(axis=1) - so["wonTackle"] - so["cbi_div3"] - so["recov_div3"]
    result = pd.concat([screen(so, components + ["everything_else"]), screen(card_components(), ["cards"])])
    result = result[result["pos"].isin(["G", "D", "M", "F"])]
    result = result.sort_values(["pos", "tau"], ascending=[True, False])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.round(4).to_csv(OUT_DIR / "component_screen.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "field_presence_2018_2024.json").write_text(json.dumps(presence, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(result.round(3).to_string(index=False))
    print("Field presence among played 60+ rows:", {k: round(v, 3) for k, v in presence.items()})


if __name__ == "__main__":
    main()
