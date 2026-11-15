"""
Combines every facet built across this project into predicted_points.csv:
appearance + non-penalty goal + assist + clean sheet + penalty + defcon +
bonus points + goalkeeper saves, per player per gameweek.

Output schema: Player, Pos, Team (3-letter code, same as minutes.csv), Price,
one column per upcoming gameweek (headered by the ACTUAL round number, e.g.
11-20 when predicting rounds 11-20 -- matching minutes.csv's 11_xMins..20_xMins;
the window length is forecast_horizon.HORIZON),
Total -- ordered by Total points, descending.

Also writes predicted_components.csv: one row per player and fixture with the expected events
behind the points (goals including penalties, assists, defcon and clean-sheet probabilities) and
each component's points. It feeds the FPL Review comparison (scripts/fplreview/) and the snapshot;
no model stage reads it. It is written before predicted_points.csv, so the snapshot's freshness
check sees it as an input of the same run.

Run locally (after every individual facet script has been run):
    python -m fpl_model.assemble.build_predicted_points
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

from fpl_model import paths
from fpl_model.atomic_write import to_csv_atomic  # noqa: E402
from fpl_model.card_points import (  # noqa: E402
    ROLE_MIX,
    current_rates,
    expected_card_points,
    fixture_yellows_90,
    forecast_cameo_share,
)
from fpl_model.defcon_scaling import points_mu90, prob_at_minutes  # noqa: E402
from fpl_model.keeper_saves import (
    goalkeeper_event_rates,  # noqa: E402
    save_fixture_multiplier,
    save_fixture_reference,
)

_DATA = paths.CURRENT

MINUTES_PROBS_FILE = paths.CURRENT / "minutes_probabilities.csv"
SIM_FEATURES_FILE = paths.CURRENT / "simulation_features.csv"
PENALTY_FILE = paths.CURRENT / "penalty_predictions.csv"
BONUS_FILE = paths.CURRENT / "bonus_predictions.csv"
PLAYERS_CURRENT_FILE = paths.CURRENT / "players_current.csv"
PLAYER_MATCH_STATS_FILE = paths.CURRENT / "player_match_stats.csv"
PLAYER_MATCH_POSITIONS_FILE = paths.CURRENT / "player_match_positions.csv"
TEAM_RATINGS_HISTORY_FILE = paths.CURRENT / "team_ratings_history.csv"

OUT_FILE = paths.CURRENT / "predicted_points.csv"
COMPONENTS_FILE = paths.CURRENT / "predicted_components.csv"
from fpl_model.forecast_horizon import GAMEWEEKS  # noqa: E402  (the forecast window, set in one place)

PTS_PER_GOAL = {"G": 10, "D": 6, "M": 5, "F": 4}
CS_PTS = {"G": 4, "D": 4, "M": 1, "F": 0}
GK_POSITIONS = {"G", "GK"}



def expected_points_per_three_saves(expected_saves):
    """Expected FPL save points for a Poisson save count (1 point per 3 saves)."""
    means = np.asarray(expected_saves, dtype=float)
    points = np.zeros_like(means)
    max_mean = float(np.max(means)) if means.size else 0.0
    max_threshold = int(np.ceil(max_mean + 8 * np.sqrt(max_mean) + 8))
    for threshold in range(3, max_threshold + 1, 3):
        points += poisson.sf(threshold - 1, means)
    return points


def expected_goals_conceded_deduction(expected_goals):
    """Expected -1 per 2 goals conceded for a Poisson goals-conceded count."""
    lam = np.asarray(expected_goals, dtype=float)
    return lam / 2 - (1 - np.exp(-2 * lam)) / 4


def expected_clean_sheet_points(clean_sheet_probability, p60, points_per_clean_sheet):
    return (
        np.asarray(clean_sheet_probability, dtype=float)
        * np.asarray(p60, dtype=float)
        * np.asarray(points_per_clean_sheet, dtype=float)
    )


def expected_goalkeeper_save_points(saves_per_90, expected_minutes):
    expected_saves = saves_per_90 * np.asarray(expected_minutes, dtype=float) / 90
    return expected_points_per_three_saves(expected_saves)


def expected_goalkeeper_goals_conceded_points(expected_goals, expected_minutes):
    minutes_fraction = np.maximum(np.asarray(expected_minutes, dtype=float), 0) / 90
    return -expected_goals_conceded_deduction(
        np.asarray(expected_goals, dtype=float) * minutes_fraction
    )


def main():
    minutes_probs = pd.read_csv(MINUTES_PROBS_FILE, encoding="utf-8-sig")
    sim_features = pd.read_csv(SIM_FEATURES_FILE, encoding="utf-8-sig")
    penalties = pd.read_csv(PENALTY_FILE, encoding="utf-8-sig")
    bonus = pd.read_csv(BONUS_FILE, encoding="utf-8-sig")
    players_current = pd.read_csv(PLAYERS_CURRENT_FILE, encoding="utf-8-sig")
    gk_saves_per_90 = goalkeeper_event_rates(players_current)
    print(f"Goalkeeper history rate: {gk_saves_per_90:.3f} saves/90")

    # Defcon point-estimate probability, re-scaled to the CURRENT expected
    # minutes. This used to join predictions_defensive.csv's "Probability"
    # column, which predict.py had already scaled internally -- but predict.py
    # now runs in part 1 of the pipeline (run_post_gameweek.py), a day or more
    # before minutes_probabilities.csv is refreshed, so that column is scaled
    # by last week's minutes and its GW labels point at the wrong rounds.
    # simulation_features.csv carries the unscaled 90-minute rate
    # (Defcon_Mu90, inverted from Probability_at_90min) alongside this week's
    # xMins, so scaling here reproduces the same number from fresh inputs.
    sim_features["Defcon_Prob"] = prob_at_minutes(
        points_mu90(sim_features["Defcon_Mu90"].fillna(0.0), sim_features["Pos"]),
        sim_features["xMins"],
        sim_features["Threshold"],
    )

    for pen_col in ("PenPts", "PenGoals"):
        sim_features = sim_features.merge(
            penalties.melt(id_vars=["Player"], value_vars=[f"{pen_col}_GW{gw}" for gw in GAMEWEEKS],
                            var_name="_gw_col", value_name=pen_col).assign(
                GW=lambda d: d["_gw_col"].str.extract(r"GW(\d+)").astype(int))[["Player", "GW", pen_col]],
            on=["Player", "GW"], how="left")
        sim_features[pen_col] = sim_features[pen_col].fillna(0)

    sim_features = sim_features.merge(
        bonus[["Player", "GW", "Expected_Bonus"]], on=["Player", "GW"], how="left")
    sim_features["Expected_Bonus"] = sim_features["Expected_Bonus"].fillna(0)

    pts_per_goal = sim_features["Pos"].map(PTS_PER_GOAL)
    cs_pts_rate = sim_features["Pos"].map(CS_PTS)

    goal_pts = sim_features["NPG90_scaled"] * (sim_features["xMins"] / 90) * pts_per_goal
    assist_pts = sim_features["Ast90_scaled"] * (sim_features["xMins"] / 90) * 3
    is_gk = sim_features["Pos"].isin(GK_POSITIONS)
    defcon_pts = 2 * sim_features["Defcon_Prob"] * (~is_gk)
    cs_pts = expected_clean_sheet_points(
        sim_features["CS_Prob_Team"], sim_features["P60"], cs_pts_rate
    )

    # Goals-conceded points: -1 per 2 goals conceded, D only (GK excluded from
    # this model). This is non-linear in goals conceded, so it needs the true
    # Poisson expectation E[floor(X/2)] rather than a linear approximation --
    # closed form: E[floor(X/2)] = X/2 - (1 - e^(-2X))/4 for X ~ Poisson(lambda=X),
    # verified against brute-force summation to 5 decimal places. Preserve the
    # established outfield calculation; scale GK exposure by expected minutes
    # so cameos are not charged for a full-match goals-conceded distribution.
    lam = sim_features["Expected_Goals_Conceded"]
    e_floor_half = expected_goals_conceded_deduction(lam)
    is_def = (sim_features["Pos"] == "D").astype(float)
    gc_pts = -1 * e_floor_half * sim_features["P60"] * is_def
    gk_gc_pts = expected_goalkeeper_goals_conceded_points(
        lam, sim_features["xMins"]
    )
    gc_pts += gk_gc_pts * is_gk

    # AppPts pulled from minutes_probabilities.csv directly for consistency
    # with what was already built, rather than recomputed (P1+P60 would give
    # the same number, but this avoids any drift)
    app_pts_lookup = minutes_probs.set_index("Player")
    def get_apppts(row):
        col = f"AppPts_GW{row['GW']}"
        if row["Web_Name"] in app_pts_lookup.index and col in app_pts_lookup.columns:
            return app_pts_lookup.loc[row["Web_Name"], col]
        return 0.0
    app_pts = sim_features.apply(get_apppts, axis=1)

    # Fixture save rate (CHANGELOG §162, fpl_model/keeper_saves.py)
    history = pd.read_csv(TEAM_RATINGS_HISTORY_FILE, encoding="utf-8-sig", low_memory=False)
    save_rates = gk_saves_per_90 * save_fixture_multiplier(
        sim_features["Expected_Goals_Conceded"], save_fixture_reference(history)
    )
    gk_save_pts = expected_goalkeeper_save_points(
        save_rates, sim_features["xMins"]
    ) * is_gk

    # Yellow and red cards (CHANGELOG §175, fpl_model/card_points.py)
    rates = current_rates(players_current)
    club_played = rates.attrs["club_played"]
    names = players_current.drop_duplicates(["Web_Name", "Team"])[["Web_Name", "Team", "Player"]].rename(columns={"Player": "_full"})
    carded = sim_features[["Web_Name", "Team"]].merge(names, on=["Web_Name", "Team"], how="left").merge(
        rates.drop_duplicates("Player"), left_on="_full", right_on="Player", how="left")
    assert len(carded) == len(sim_features), "card join fanned out"
    unmatched = int(carded["Yellows_90"].isna().sum())
    if unmatched:
        print(f"WARNING: {unmatched} forecast rows have no card rate; they get none")
    missing_clubs = sorted(set(sim_features["Team"]) - set(club_played))
    if missing_clubs and club_played:
        print(f"WARNING: no league matches this season found for {missing_clubs}; club match counts from 1")
    # Club match number of each fixture: matches played + its place among the club's forecast fixtures.
    fixtures = sim_features[["Team", "GW", "Round"]].drop_duplicates().sort_values(["Team", "Round", "GW"])
    fixtures["_club_match"] = fixtures.groupby("Team").cumcount() + 1 + fixtures["Team"].map(club_played).fillna(0)
    club_match = sim_features[["Team", "GW", "Round"]].merge(fixtures, on=["Team", "GW", "Round"], how="left")["_club_match"]
    share = None
    if "cameo_minutes" in rates.attrs and ROLE_MIX == "forecast":
        raw = sim_features["xMins_Raw"] if "xMins_Raw" in sim_features else sim_features["xMins"]
        share = forecast_cameo_share(sim_features["P1"], sim_features["P60"], raw, rates.attrs["cameo_minutes"])
    yellows_90 = fixture_yellows_90(carded, club_match.values, share)
    card_pts = expected_card_points(np.nan_to_num(yellows_90), carded["Reds_90"].fillna(0).values,
                                    sim_features["xMins"].values)
    sim_features["CardPts"] = np.round(card_pts, 3)

    sim_features["AppPts"] = app_pts.round(3)
    sim_features["GoalPts"] = goal_pts.round(3)
    sim_features["AssistPts"] = assist_pts.round(3)
    sim_features["DefconPts"] = defcon_pts.round(3)
    sim_features["CSPts"] = cs_pts.round(3)
    sim_features["GoalsConcededPts"] = gc_pts.round(3)
    sim_features["SavesPts"] = gk_save_pts.round(3)
    sim_features["PenPts"] = sim_features["PenPts"].round(3)
    sim_features["BonusPts"] = sim_features["Expected_Bonus"].round(3)
    sim_features["TotalPts"] = (
        app_pts + goal_pts + assist_pts + defcon_pts + cs_pts + gc_pts + gk_save_pts +
        sim_features["PenPts"] + sim_features["Expected_Bonus"] + card_pts
    ).round(2)

    component_cols = [
        "AppPts", "GoalPts", "AssistPts", "DefconPts", "CSPts", "GoalsConcededPts",
        "SavesPts", "PenPts", "BonusPts", "CardPts", "TotalPts",
    ]
    out = sim_features.pivot_table(
        index=["Web_Name", "Pos", "Team", "Price"], columns="GW", values=component_cols, aggfunc="first"
    )
    out.columns = [f"{col}_GW{gw}" for col, gw in out.columns]
    out = out.reset_index().rename(columns={"Web_Name": "Player"})
    total_cols = [f"TotalPts_GW{gw}" for gw in GAMEWEEKS if f"TotalPts_GW{gw}" in out.columns]
    out["Total"] = out[total_cols].sum(axis=1).round(2)

    # Capture descending sort order while Total is unambiguously a single Series
    sort_idx = out["Total"].values.argsort()[::-1]

    # Column headers are the ACTUAL upcoming gameweek numbers, not GW1-GW<HORIZON>.
    # Every intermediate file labels its forecast columns GW1-GW<HORIZON>, but GW1
    # there means "first of the next HORIZON rounds" (e.g. round 11, matching
    # minutes.csv's 11_xMins..20_xMins). simulation_features.csv carries both
    # the GW label and the real Round per row -- use that mapping to rename.
    _gw_round = sim_features.drop_duplicates(subset=["GW"])[["GW", "Round"]].astype(int)
    gw_to_round = dict(zip(_gw_round["GW"], _gw_round["Round"]))
    out = out.rename(columns={f"TotalPts_GW{gw}": str(r) for gw, r in gw_to_round.items()})
    round_cols = [str(gw_to_round[gw]) for gw in GAMEWEEKS if gw in gw_to_round]

    # Team codes (ARS, CHE, ...) -- the same 3-letter codes minutes.csv uses --
    # instead of full team names
    team_short = dict(zip(players_current["Team"], players_current["Team_Short"]))
    out["Team"] = out["Team"].map(team_short)

    # Final schema: Player, Pos, Team, Price, <next HORIZON round numbers>, Total,
    # ordered by Total points descending
    out = out[["Player", "Pos", "Team", "Price"] + round_cols + ["Total"]]
    out = out.iloc[sort_idx].reset_index(drop=True)

    # Expected events per fixture, at the minutes production uses (xMins, rescaled to the recorded clock).
    team_short_map = dict(zip(players_current["Team"], players_current["Team_Short"]))
    components = sim_features.assign(
        Team=sim_features["Team"].map(team_short_map),
        NP_Goals=(sim_features["NPG90_scaled"] * sim_features["xMins"] / 90),
        Goals=(sim_features["NPG90_scaled"] * sim_features["xMins"] / 90 + sim_features["PenGoals"]),
        Assists=(sim_features["Ast90_scaled"] * sim_features["xMins"] / 90),
        Defcon_Prob=sim_features["Defcon_Prob"].where(~is_gk, 0.0),
        CS_Prob=sim_features["CS_Prob_Team"] * sim_features["P60"],
    )
    component_out = components[
        ["Web_Name", "Pos", "Team", "Price", "GW", "Round", "Opponent", "Venue", "xMins", "xMins_Raw", "P1", "P60",
         "NP_Goals", "PenGoals", "Goals", "Assists", "Defcon_Prob", "CS_Prob", "CS_Prob_Team", "Expected_Goals_Conceded",
         *component_cols]
    ].rename(columns={"Web_Name": "Player"}).sort_values(["Round", "TotalPts"], ascending=[True, False])
    to_csv_atomic(component_out.round(4), COMPONENTS_FILE, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(component_out)} player-fixture rows to {COMPONENTS_FILE.name}")

    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} players to {OUT_FILE}")
    print(f"Gameweek columns: {round_cols}")
    print(out.head(15))


if __name__ == "__main__":
    main()
