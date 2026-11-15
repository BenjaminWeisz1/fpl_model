"""Keeper save rate by fixture, shared by points and bonus (CHANGELOG §162, §165).

Saves rise with the goals a keeper's side is expected to concede. Each caller's pooled rate keeps
the level; this multiplier moves spread only. Refit SAVE_FIXTURE_BETA when a season ends.
"""

import numpy as np
import pandas as pd

from fpl_model import paths
from fpl_model.league_rates import penalty_rates  # noqa: E402

_DATA = paths.CURRENT

SAVE_FIXTURE_BETA = 0.66
SAVE_RATE_FIRST_SEASON = 2025  # the window of goalkeeper_event_rates' FPL API history
_ATTEMPTS, _CONVERSION = penalty_rates()
PENALTY_GOALS_FACTOR = 1 + _ATTEMPTS * _CONVERSION  # as build_simulation_features adds to Opp_Pred_NPG


def save_fixture_reference(history):
    """Mean of (non-penalty forecast)^beta over the rate's window, so the mean multiplier there is 1."""
    recent = history.loc[history["Season"] >= SAVE_RATE_FIRST_SEASON]
    forecasts = pd.concat([recent["Home_Prediction"], recent["Away_Prediction"]])
    return float((forecasts ** SAVE_FIXTURE_BETA).mean())


def save_fixture_multiplier(expected_goals_conceded, reference):
    open_play = np.asarray(expected_goals_conceded, dtype=float) / PENALTY_GOALS_FACTOR
    return open_play ** SAVE_FIXTURE_BETA / reference


def goalkeeper_history(players_current):
    """Played FPL API rows of goalkeepers."""
    stats = pd.read_csv(_DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    positions = pd.read_csv(_DATA / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False)
    keys = ["Season", "Team", "Opponent", "Venue", "Player"]
    history = stats.loc[stats["Data_Source"] == "FPL_API"].merge(
        positions[keys + ["Position_Resolved"]], on=keys, how="left", validate="many_to_one")
    current_gk_names = set(players_current.loc[players_current["Pos"] == "G", "Player"])
    is_gk = history["Position_Resolved"].eq("GK") | history["Player"].isin(current_gk_names)
    history = history.loc[is_gk & history["Minutes"].gt(0) & history["Saves"].notna()]
    if history.empty:
        raise ValueError("No played goalkeeper FPL API history available for save-rate estimates")
    return history


def goalkeeper_event_rates(players_current):
    """Pooled saves per 90 from FPL API goalkeeper history, shared by points and bonus."""
    history = goalkeeper_history(players_current)
    return 90 * history["Saves"].sum() / history["Minutes"].sum()


def goalkeeper_pen_save_rate(players_current):
    """Penalty saves per 60+ minute goalkeeper appearance."""
    history = goalkeeper_history(players_current)
    return float(history.loc[history["Minutes"] >= 60, "Penalties_Saved"].mean())
