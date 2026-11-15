"""
Screen: FPL-only assists (diagnostic, CHANGELOG §90). Nothing here changes a model.

FPL awards assists for more than conventional final passes: rebounds from saved, blocked or
woodwork shots, deflected passes (2025/26 rule), penalties and free kicks won and scored, and
forced own goals. FPL-only assists = FPL Assists - Sofascore goalAssist, on FPL-era rows (2025+)
joined to Sofascore. This reports, per position, the size of the gap, the share explained by the
identifiable mechanisms (penalty won and scored by a teammate; opponent own goal), whether players
differ beyond chance (Poisson dispersion, split-half persistence), and how a player's FPL-only rate
relates to stable volumes (shots that can rebound, crosses, key passes, xA, fouls won).

Run locally:
    python -m research.bonus.screen_fpl_only_assists
"""
import json

import numpy as np
import pandas as pd

from research.bonus import validate_bonus_accounting as acc
from research.bonus import validate_bonus_calibration as calibration

OUT_DIR = calibration.ROOT / "data" / "validation" / "fpl_only_assists_20261028"
POSITIONS = acc.POSITIONS
# Sofascore spellings not yet rebuilt after §88's player_match_stats renames
LOCAL_RENAMES = {"James Mcatee": "James McAtee"}
MIN_SPLIT = 600  # minutes per half for the split-half check
FEATURES = {
    "rebound_shots": "saved, blocked or woodwork shots",
    "shots": "all shots",
    "crosses": "crosses",
    "key_passes": "key passes",
    "big_chances_created": "big chances created",
    "xa": "Sofascore xA",
    "conv_assists": "conventional assists",
    "fouled": "fouls won",
    "pens_won": "penalties won",
}


def load():
    rows, _ = acc.load_rows()
    rows = rows[rows["Season"] >= 2025].copy()
    so = pd.read_csv(calibration.DATA / "sofascore_player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    so = so[so["Appearance_Status"].eq("played")].copy()
    so["Kickoff"] = pd.to_datetime(so["Kickoff"], utc=True)
    so["Local_Player"] = so["Local_Player"].replace(LOCAL_RENAMES)
    key = ["Season", "Team", "Opponent", "Venue", "Player"]
    s = so.rename(columns={"Local_Player": "Player"}).dropna(subset=["Player"]).drop_duplicates(key)
    num = lambda c: pd.to_numeric(s[f"Sofascore_{c}"], errors="coerce").fillna(0)
    feats = pd.DataFrame({k: s[k] for k in key + ["Kickoff"]})
    feats["so_goals"], feats["conv_assists"] = num("goals"), num("goalAssist")
    sot_saved = (num("onTargetScoringAttempt") - num("goals")).clip(lower=0)
    feats["rebound_shots"] = sot_saved + num("blockedScoringAttempt") + num("hitWoodwork")
    feats["shots"] = num("onTargetScoringAttempt") + num("shotOffTarget") + num("blockedScoringAttempt")
    feats["crosses"], feats["key_passes"] = num("totalCross"), num("keyPass")
    feats["big_chances_created"], feats["xa"] = num("bigChanceCreated"), num("expectedAssists")
    feats["fouled"], feats["pens_won"] = num("wasFouled"), num("penaltyWon")
    m = rows.merge(feats, on=key, how="left", indicator=True)
    joined = m["_merge"].eq("both")
    m = m[joined].drop(columns="_merge").copy()
    m["fpl_only"] = m["Assists"] - m["conv_assists"]
    return m, float(joined.mean()), int(len(rows))


def mechanisms(m):
    """Identifiable FPL-only assists: penalty won and scored by a teammate; opponent own goal."""
    team_pk = m.groupby(["Match_Key", "Team"])["PK"].transform("sum")
    teammate_pk = team_pk - m["PK"]
    m["pen_assist_possible"] = np.minimum(m["pens_won"], teammate_pk).clip(lower=0)
    og = m.groupby(["Match_Key", "Team"])["Own_Goals"].sum().rename("og_by_team").reset_index()
    og = og.rename(columns={"Team": "Opponent", "og_by_team": "opp_own_goals"})
    m = m.merge(og, on=["Match_Key", "Opponent"], how="left")
    m["opp_own_goals"] = m["opp_own_goals"].fillna(0)
    return m


def dispersion(g, col):
    """Method-of-moments between-player heterogeneity of a per-90 count rate, within one position."""
    e = g["mins"] / 90
    lam = g[col].sum() / e.sum()
    chi = float((((g[col] - lam * e) ** 2) / (lam * e)).sum() / (len(g) - 1))
    # true-rate variance tau^2 (per 90) from E[(c - lam e)^2] = lam e + tau^2 e^2
    tau2 = float(max((((g[col] - lam * e) ** 2 - lam * e).sum()) / (e ** 2).sum(), 0.0))
    return {"players": int(len(g)), "rate_per90": round(lam, 4), "dispersion_index": round(chi, 3),
            "true_rate_sd_per90": round(tau2 ** 0.5, 4), "cv_true_rate": round(tau2 ** 0.5 / lam, 3) if lam else None}


def weighted_corr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return float(cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)))


def main():
    m, join_rate, n_rows = load()
    m = mechanisms(m)
    out = {"fpl_era_rows": n_rows, "sofascore_join_rate": round(join_rate, 4), "rows_joined": int(len(m)),
           "seasons": sorted(int(s) for s in m["Season"].unique())}

    # 1. size by position, negative gaps, identifiable mechanisms
    size = {}
    for pos, g in m.groupby("Pos"):
        mins = g["Minutes"].sum()
        size[pos] = {"rows": int(len(g)), "fpl_assists": int(g["Assists"].sum()), "conventional": int(g["conv_assists"].sum()),
                     "fpl_only": int(g["fpl_only"].sum()), "fpl_over_conventional": round(g["Assists"].sum() / max(g["conv_assists"].sum(), 1), 3),
                     "fpl_only_per90": round(g["fpl_only"].sum() / mins * 90, 4),
                     "rows_negative_gap": int((g["fpl_only"] < 0).sum()),
                     "upper_bound_penalty_won": int(g.loc[g["fpl_only"] > 0, "pen_assist_possible"].clip(upper=g["fpl_only"]).sum())}
    out["size_by_position"] = size
    pos_fpl_only = m.loc[m["fpl_only"] > 0]
    out["fpl_only_rows_in_matches_with_opponent_own_goal"] = int((pos_fpl_only["opp_own_goals"] > 0).sum())
    out["fpl_only_total"] = int(m["fpl_only"].sum())

    # 2. heterogeneity: Poisson dispersion of FPL-only counts across players, per position
    p = m.groupby(["Player", "Pos"]).agg(mins=("Minutes", "sum"), fpl_only=("fpl_only", "sum"),
                                         conv=("conv_assists", "sum"),
                                         **{f: (f, "sum") for f in FEATURES if f != "conv_assists"})
    p = p[p["mins"] >= 900].reset_index()
    p["fpl_only"] = p["fpl_only"].clip(lower=0)
    out["dispersion"] = {pos: {"fpl_only": dispersion(g, "fpl_only"), "conventional": dispersion(g, "conv")}
                         for pos, g in p.groupby("Pos") if pos != "G" and len(g) > 10}

    # 3. split-half persistence (alternate appearances) of per-90 rates
    m = m.sort_values(["Player", "Season", "Kickoff"])
    m["half"] = m.groupby("Player").cumcount() % 2
    h = m.groupby(["Player", "Pos", "half"]).agg(mins=("Minutes", "sum"), fpl_only=("fpl_only", "sum"),
                                                 conv_assists=("conv_assists", "sum"),
                                                 **{f: (f, "sum") for f in FEATURES if f != "conv_assists"}).reset_index()
    wide = h.pivot_table(index=["Player", "Pos"], columns="half").dropna()
    wide = wide[(wide[("mins", 0)] >= MIN_SPLIT) & (wide[("mins", 1)] >= MIN_SPLIT)]
    rate = lambda c, k: wide[(c, k)] / wide[("mins", k)] * 90
    w = (wide[("mins", 0)] + wide[("mins", 1)]).to_numpy()
    persist = {"players": int(len(wide))}
    for c in ["fpl_only", "conv_assists", "rebound_shots", "crosses", "key_passes", "xa", "pens_won"]:
        persist[c] = round(weighted_corr(rate(c, 0).to_numpy(), rate(c, 1).to_numpy(), w), 3)
    # cross-half: FPL-only in one half against each volume in the other (both directions averaged)
    cross = {}
    for f in FEATURES:
        r = [weighted_corr(rate(f, a).to_numpy(), rate("fpl_only", b).to_numpy(), w) for a, b in ((0, 1), (1, 0))]
        cross[f] = round(float(np.mean(r)), 3)
    persist["fpl_only_in_one_half_vs_feature_in_other"] = cross
    out["split_half"] = persist

    # 4. player-level association within position (players >= 900 min): FPL-only per 90 vs volumes per 90
    assoc = {}
    for pos, g in p.groupby("Pos"):
        if pos == "G" or len(g) < 20:
            continue
        ww = g["mins"].to_numpy()
        y = (g["fpl_only"] / g["mins"] * 90).to_numpy()
        assoc[pos] = {"players": int(len(g)),
                      **{f: round(weighted_corr((g[f if f != "conv_assists" else "conv"] / g["mins"] * 90).to_numpy(), y, ww), 3)
                         for f in FEATURES}}
    out["player_level_correlation_with_fpl_only_per90"] = assoc

    # 5. pooled Poisson regression (rows), FPL-only on per-appearance volumes with a minutes offset
    import statsmodels.api as sm
    r = m[m["Pos"] != "G"].copy()
    r["fpl_only_pos"] = r["fpl_only"].clip(lower=0)
    X = pd.get_dummies(r["Pos"], drop_first=True, dtype=float)
    for f in ["rebound_shots", "crosses", "key_passes", "conv_assists", "pens_won"]:
        X[f] = r[f]
    X = sm.add_constant(X)
    fit = sm.GLM(r["fpl_only_pos"], X, family=sm.families.Poisson(),
                 offset=np.log(r["Minutes"].clip(lower=1) / 90)).fit()
    out["row_poisson_glm_same_match_events"] = {"note": "same-match association, not a forecast",
                                                "coef": fit.params.round(4).to_dict(), "se": fit.bse.round(4).to_dict()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "screen.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
