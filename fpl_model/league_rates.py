"""Season and league rates computed from the data, so no session has to edit them by hand (CHANGELOG §181).

current_season() is the calendar year of GW1's deadline in gameweek_deadlines.csv, which fetch_bootstrap.py
rewrites first in part 1: 2026 for 2026/27, and it moves when FPL publishes the new game.
"""
from fpl_model import paths
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

_DATA = paths.CURRENT
RECORDED_TEAM_MINUTES = 11 * 90     # eleven players x 90 recorded minutes a match


@lru_cache(maxsize=1)
def current_season():
    deadlines = pd.read_csv(_DATA / "gameweek_deadlines.csv", encoding="utf-8-sig")
    first = deadlines.loc[deadlines["Gameweek"] == deadlines["Gameweek"].min(), "Deadline_UTC"].iloc[0]
    return int(str(first)[:4])


@lru_cache(maxsize=1)
def mean_npg():
    """Mean non-penalty goals per team-match, the ave_goals update_team_ratings.py trains on."""
    results = pd.read_csv(_DATA / "match_results.csv", encoding="utf-8-sig", usecols=["Home_NPG", "Away_NPG"])
    return float(pd.concat([results["Home_NPG"], results["Away_NPG"]]).mean())


@lru_cache(maxsize=1)
def penalty_rates():
    """(attempts per non-penalty goal, conversion) over every league match in player_match_stats.csv."""
    stats = pd.read_csv(_DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                        usecols=["PKatt", "PK", "NPG"])
    return float(stats["PKatt"].sum() / stats["NPG"].sum()), float(stats["PK"].sum() / stats["PKatt"].sum())


def check_minutes_round(first_round):
    """Stop if minutes.csv doesn't start at the forecast's first round (§184).

    Later stages pair its rounds with the forecast by position, so a mismatch would give every player
    the wrong week's minutes with no error. Column names such as 10_xMins carry the real round.
    """
    columns = pd.read_csv(_DATA / "minutes.csv", encoding="utf-8-sig", nrows=0).columns
    rounds = sorted(int(c.split("_")[0]) for c in columns if re.match(r"^\d+_xMins$", c))
    if not rounds or rounds[0] != first_round:
        raise ValueError(f"minutes.csv starts at round {rounds[0] if rounds else None} but the forecast starts at "
                         f"round {first_round}. Refresh minutes.csv, or run part 1 if the gameweek has finished.")


def recorded_clock_scale(rows):
    """RECORDED_TEAM_MINUTES / FPL Review's typical minutes for one team in one gameweek.

    Computed from this week's xMins, so it follows FPL Review if its clock changes. The median of team-gameweek
    totals keeps a double or blank gameweek from moving it. `rows` needs Team, Round and xMins_Raw.
    """
    return RECORDED_TEAM_MINUTES / rows.groupby(["Team", "Round"])["xMins_Raw"].sum().median()


@lru_cache(maxsize=1)
def minute_bands():
    """(mean minutes of a 60+ appearance, mean minutes of a 1-59 appearance), FPL-era recorded minutes."""
    stats = pd.read_csv(_DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                        usecols=["Data_Source", "Minutes"])
    minutes = stats.loc[stats["Data_Source"] == "FPL_API", "Minutes"]
    return float(minutes[minutes >= 60].mean()), float(minutes[(minutes >= 1) & (minutes < 60)].mean())
