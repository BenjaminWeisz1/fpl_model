"""
Computes predicted penalty attempts, goals, and points per player per
gameweek: team-level predicted penalty attempts (from predicted non-penalty
goals x the empirical 792/7734 attempts-per-NPG ratio) allocated across
teammates via penalty_share.csv, then converted to points using a
player-specific smoothed historical conversion rate. The -6 BPS penalty for
missing is noted but not applied here --
that's a bonus-points-facet concern, handled when we build that model.

Run locally (after predict_fixture_goals.py):
    python -m fpl_model.minutes_penalties_cs.build_penalty_predictions
"""

import pandas as pd

from fpl_model import paths
from fpl_model.forecast_horizon import GAMEWEEKS  # noqa: E402  (the forecast window, set in one place)
from fpl_model.league_rates import check_minutes_round, penalty_rates, recorded_clock_scale  # noqa: E402

_DATA = paths.CURRENT

PEN_ATTEMPTS_PER_NPG, _ = penalty_rates()  # computed (§181)
PTS_PER_GOAL = {"G": 10, "D": 6, "M": 5, "F": 4}

FIXTURE_GOALS_FILE = paths.current_file("fixture_goal_predictions.csv", root=paths.CURRENT)
PENALTY_SHARE_FILE = paths.current_file("penalty_share.csv", root=paths.CURRENT)
PLAYERS_CURRENT_FILE = paths.current_file("players_current.csv", root=paths.CURRENT)
PENALTY_LOG_FILE = paths.current_file("penalty_log.csv", root=paths.CURRENT)
PLAYER_MATCH_STATS_FILE = paths.current_file("player_match_stats.csv", root=paths.CURRENT)
MINUTES_PROBS_FILE = paths.current_file("minutes_probabilities.csv", root=paths.CURRENT)
OUT_FILE = paths.current_file("penalty_predictions.csv", root=paths.CURRENT)


def resolve_names(pen_share, players_current, name_col):
    """Same resolution logic as build_minutes_probabilities.py -- tries
    (raw name, team code) first, then falls back to (Web_Name, team code) directly
    for files that are already in the disambiguated form. penalty_share.csv's Team
    holds codes such as CHE, so the lookups key on Team_Short (§183)."""
    if "Raw_Web_Name" not in players_current.columns:
        print("WARNING: players_current.csv has no Raw_Web_Name column yet (run "
              "fetch_bootstrap.py at least once) -- falling back to matching on "
              "Web_Name directly, which will misresolve any disambiguated names.")
        raw_name_lookup = {}
    else:
        raw_name_lookup = dict(zip(
            zip(players_current["Raw_Web_Name"], players_current["Team_Short"]),
            players_current["Web_Name"]))

    web_name_lookup = dict(zip(
        zip(players_current["Web_Name"], players_current["Team_Short"]),
        players_current["Web_Name"]))

    def resolve_one(name, team):
        if (name, team) in raw_name_lookup:
            return raw_name_lookup[(name, team)]
        if (name, team) in web_name_lookup:
            return web_name_lookup[(name, team)]
        return name

    return pen_share.apply(lambda r: resolve_one(r[name_col], r["Team"]), axis=1)


def on_pitch_share(minutes_probs, gw, scale):
    """Web_Name -> the share of this gameweek's match the player is expected to be on the pitch.

    FPL Review's shares apply while the player is on the pitch, so a team's shares sum above 1: a
    second-choice taker counts for when the first choice has gone off. Share x this fraction sums
    to about 1 per team (§183). xMins is on the recorded clock, as in build_simulation_features.py.
    """
    fraction = (minutes_probs[f"xMins_GW{gw}"].fillna(0) * scale / 90).clip(upper=1)
    return dict(zip(minutes_probs["Player"], fraction))


def build_conversion_rates(penalty_log, player_match_stats):
    """Return smoothed historical penalty conversion rates by player.

    The penalty log is authoritative for 2025 onward: each row is one
    attempt and Scored records whether it was converted. Earlier seasons use
    the consolidated match stats' PK and PKatt columns.
    """
    prior_stats = player_match_stats[pd.to_numeric(player_match_stats["Season"]) < 2025]
    prior = prior_stats.groupby("Player")[["PK", "PKatt"]].sum()

    recent = penalty_log.copy()
    recent["Penalties_Scored"] = recent["Scored"].astype(str).str.strip().str.upper().isin(
        ["TRUE", "1", "YES"])
    recent_totals = recent.groupby("Player").agg(
        Penalties_Scored=("Penalties_Scored", "sum"),
        Penalties_Attempted=("Player", "size"),
    )

    totals = prior.rename(columns={"PK": "Penalties_Scored", "PKatt": "Penalties_Attempted"})
    totals = totals.add(recent_totals, fill_value=0)
    return (7.9 + totals["Penalties_Scored"]) / (10 + totals["Penalties_Attempted"])


def main():
    fixture_goals = pd.read_csv(FIXTURE_GOALS_FILE)
    pen_share = pd.read_csv(PENALTY_SHARE_FILE)
    players_current = pd.read_csv(PLAYERS_CURRENT_FILE, encoding="utf-8-sig")
    penalty_log = pd.read_csv(PENALTY_LOG_FILE, encoding="utf-8-sig")
    player_match_stats = pd.read_csv(PLAYER_MATCH_STATS_FILE, encoding="utf-8-sig", low_memory=False)
    conversion_rates = build_conversion_rates(penalty_log, player_match_stats)
    minutes_probs = pd.read_csv(MINUTES_PROBS_FILE, encoding="utf-8-sig")
    clock = minutes_probs.melt(id_vars=["Team"], value_vars=[f"xMins_GW{gw}" for gw in GAMEWEEKS],
                               var_name="Round", value_name="xMins_Raw")
    scale = recorded_clock_scale(clock)

    # penalty_share.csv's own "Team" column is used for the (name, team)
    # resolution -- resolve names to canonical Web_Name ONCE, up front, rather
    # than inside the per-gameweek loop below
    pen_share = pen_share.copy()
    pen_share["Web_Name"] = resolve_names(pen_share, players_current, "Player")

    # team predicted NPG per round, one row per (team, round)
    home_side = fixture_goals[["Round", "Home", "Home_Pred_NPG"]].rename(
        columns={"Home": "Team", "Home_Pred_NPG": "Team_Pred_NPG"})
    away_side = fixture_goals[["Round", "Away", "Away_Pred_NPG"]].rename(
        columns={"Away": "Team", "Away_Pred_NPG": "Team_Pred_NPG"})
    team_npg = pd.concat([home_side, away_side], ignore_index=True)
    team_npg["Team_Pred_PenAttempts"] = team_npg["Team_Pred_NPG"] * PEN_ATTEMPTS_PER_NPG

    min_round = fixture_goals["Round"].min()
    check_minutes_round(min_round)
    round_to_gw = {min_round + i: gw for i, gw in enumerate(GAMEWEEKS)}

    out = players_current[["Player", "Web_Name", "Pos", "Team", "Price"]].copy()
    out["PenaltyConversionRate"] = out["Player"].map(conversion_rates).fillna(0.79)

    for round_num, gw in round_to_gw.items():
        team_pen_this_round = team_npg[team_npg["Round"] == round_num].set_index("Team")["Team_Pred_PenAttempts"]

        share_col = f"GW{gw}"
        pen_share_gw = pen_share[["Web_Name", share_col]].copy()
        pen_share_gw[share_col] = pen_share_gw[share_col].astype(float)

        merged = out[["Web_Name", "Team", "Pos", "PenaltyConversionRate"]].merge(
            pen_share_gw, on="Web_Name", how="left")
        merged["Team_PenAttempts"] = merged["Team"].map(team_pen_this_round).fillna(0)
        on_pitch = merged["Web_Name"].map(on_pitch_share(minutes_probs, gw, scale)).fillna(0)
        merged["Player_PenAttempts"] = merged["Team_PenAttempts"] * merged[share_col].fillna(0) * on_pitch
        merged["Player_PenGoals"] = merged["Player_PenAttempts"] * merged["PenaltyConversionRate"]

        pts_per_goal = merged["Pos"].map(PTS_PER_GOAL).fillna(0)
        merged["Player_PenPts"] = merged["Player_PenAttempts"] * (
            merged["PenaltyConversionRate"] * pts_per_goal
            - (1 - merged["PenaltyConversionRate"]) * 2
        )

        out[f"PenAttempts_GW{gw}"] = merged["Player_PenAttempts"].round(4)
        out[f"PenGoals_GW{gw}"] = merged["Player_PenGoals"].round(4)
        out[f"PenPts_GW{gw}"] = merged["Player_PenPts"].round(4)

    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} players to {OUT_FILE}")
    print(out[out["PenAttempts_GW1"] > 0.01].sort_values("PenAttempts_GW1", ascending=False).head(10))


if __name__ == "__main__":
    main()
