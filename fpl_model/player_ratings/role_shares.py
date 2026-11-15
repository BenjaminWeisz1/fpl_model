"""Role percentages: each player's chance of each line, from their past starts (position plan, step 1).

Not wired into any pipeline stage. Lines match validate_role_move_error.py. Later steps should import this
so training and forecasts share one calculation. Frozen settings come from validate_role_shares.py.
"""
from fpl_model import paths
from pathlib import Path

import numpy as np
import pandas as pd

DATA = paths.CURRENT
LINES = ["CB", "WD", "CM", "ATT", "FW"]
LINE_OF = {"CB": 0, "LB": 1, "RB": 1, "WB": 1, "DM": 2, "CM": 2, "AM": 3, "LM": 3, "RM": 3, "LW": 3, "RW": 3, "FW": 4}


def load_starts():
    """Outfield starts in player-date order, with the integer line."""
    u = pd.read_csv(DATA / "understat_match_positions.csv", encoding="utf-8-sig", low_memory=False)
    s = u[(u.Is_Starter == True) & u.Position.isin(LINE_OF)].copy()  # noqa: E712
    s["Date"] = pd.to_datetime(s["Date"])
    s["L"] = s.Position.map(LINE_OF)
    return s.sort_values(["Player_ID_Understat", "Date"]).reset_index(drop=True)


def weighted_counts(lines, half_life):
    """Recency-weighted counts BEFORE each start (row t uses starts 0..t-1), plus one row after the last."""
    d = 0.5 ** (1 / half_life)
    out = np.zeros((len(lines) + 1, len(LINES)))
    w = np.zeros(len(LINES))
    for t, line in enumerate(lines):
        out[t] = w
        w = w * d
        w[line] += 1
    out[-1] = w
    return out


def shares(counts, prior):
    return (counts + prior / len(LINES)) / (counts.sum(1, keepdims=True) + prior)


# Lowest log loss on seasons to 2023 (CHANGELOG §170). Reproduce with `python validate_role_shares.py`;
# don't refit without a new pre-registration.
HALF_LIFE, PRIOR, RECENT_WEIGHT = 3, 0.1, 0.9


LINE_RATES_FILE = paths.VALIDATION / "role_factors_20261101" / "line_rates.json"


def role_table():
    """Per-start role state keyed on (Season, Team, Opponent, Venue, Player), for the loop and its scoring.

    Value: (chance of each line before the start, line played, chance after it, career modal line before
    the start, starts before it).
    """
    s = load_starts()
    keys = ["Season", "Team", "Opponent", "Venue", "Player_ID_Understat"]
    pos = pd.read_csv(DATA / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False,
                      usecols=keys + ["Player"]).drop_duplicates(keys)
    n = len(s)
    s = s.drop(columns=["Player"]).merge(pos, on=keys, how="left")
    assert len(s) == n, "role table join fanned out"
    table = {}
    for _, g in s.groupby("Player_ID_Understat", sort=False):
        lines = g.L.values
        p = role_probs(lines)
        counts = np.vstack([np.zeros(len(LINES)), np.cumsum(np.eye(len(LINES))[lines], axis=0)])
        for t, r in enumerate(g.itertuples()):
            table[(int(r.Season), r.Team, r.Opponent, r.Venue, r.Player)] = (
                p[t], int(lines[t]), p[t + 1], int(counts[t].argmax()), t)
    return table


def role_probs(lines):
    """Chance of each line BEFORE each start, plus one row for the next start. `lines` holds integer lines."""
    lines = np.asarray(lines)
    career = np.vstack([np.zeros(len(LINES)), np.cumsum(np.eye(len(LINES))[lines], axis=0)])
    recent = shares(weighted_counts(lines, HALF_LIFE), PRIOR)
    return RECENT_WEIGHT * recent + (1 - RECENT_WEIGHT) * shares(career, PRIOR)
