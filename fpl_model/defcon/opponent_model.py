"""Opponent factor (CHANGELOG §149, adopted), for training rows and forecasts.

OppFactor: for a position group, the actions conceded by an opponent / the pre-match expected actions of the players who
faced it, EW over its earlier matches, relative to the same ratio league-wide and shrunk toward 1. Sequenced by
match_results.csv row order (chronological), so it needs no kickoff times.

The §150 midfield count head that lived here was reversed (worse on FPL's real hits, its smoothed rate ignored transfer
priors) and removed in the 2026-11-16 cleanup; it is in git history before that.
"""
import numpy as np
import pandas as pd

from fpl_model import paths

OPP_HALF_LIFE, OPP_K = 19, 20        # §149, chosen on 2018-2021
WIDE_DEF = ["Pos_FB", "Pos_WB", "Pos_LB", "Pos_RB"]


def position_groups(frame, head):
    """head D: CB or wide defender. Head M: central (DM, CM) or other. Unknown slots fall in CB / other."""
    if head == "D":
        return np.where((frame["Pos_CB"].values == 0) & (frame[WIDE_DEF].sum(axis=1).values > 0), "WD", "CB")
    return np.where(frame[["Pos_DM", "Pos_CM"]].sum(axis=1).values > 0, "CM", "AM")


def match_order(data_dir):
    """Team-match keys in chronological order: one list per match, home side first."""
    m = pd.read_csv(paths.current_file("match_results.csv", root=data_dir), encoding="utf-8-sig", usecols=["Season", "Home", "Away"])
    return [((int(s), h, a, "Home"), (int(s), a, h, "Away")) for s, h, a in zip(m.Season, m.Home, m.Away)]


def opponent_factors(frame, head, matches, half_life=OPP_HALF_LIFE, k=OPP_K):
    """Per-row factor (uses only earlier matches) and {(opponent, group): factor} after the last match."""
    d, league_d = 2.0 ** (-1.0 / half_life), 2.0 ** (-1.0 / (half_life * 10))
    f = frame[["Season", "Team", "Opponent", "Venue", "Minutes", "MatchesPlayed", "Per90Stat", "AvgPer90", "Recent5AvgPer90"]].copy()
    f["_group"] = position_groups(frame, head)
    f["_exp"] = 0.5 * (f["AvgPer90"] + f["Recent5AvgPer90"])
    u = f[(f["Minutes"] >= 60) & (f["MatchesPlayed"] >= 5) & (f["_exp"] > 0)]
    agg = u.groupby(["Season", "Team", "Opponent", "Venue", "_group"]).agg(A=("Per90Stat", "sum"), E=("_exp", "sum"), N=("_exp", "size")).reset_index()
    totals = {(s, t, o, v, g): (a, e, n) for s, t, o, v, g, a, e, n in agg.values}
    groups = sorted(f["_group"].unique())
    state, league = {}, {g: [0.0, 0.0] for g in groups}

    def factor(opponent, g):
        A, E, N = state.get((opponent, g), (0.0, 0.0, 0.0))
        la, le = league[g]
        base = la / le if le > 0 else 1.0
        if N == 0:
            return 1.0
        ebar = E / N
        return (A + k * base * ebar) / (E + k * ebar) / base

    out = {}
    for pair in matches:
        for s, t, o, v in pair:
            for g in groups:
                out[(s, t, o, v, g)] = factor(o, g)
        for s, t, o, v in pair:
            for g in groups:
                a, e, n = totals.get((s, t, o, v, g), (0.0, 0.0, 0.0))
                A, E, N = state.get((o, g), (0.0, 0.0, 0.0))
                state[(o, g)] = (A * d + a, E * d + e, N * d + n)
                league[g][0] = league[g][0] * league_d + a
                league[g][1] = league[g][1] * league_d + e
    row = np.array([out.get(key, 1.0) for key in zip(f["Season"], f["Team"], f["Opponent"], f["Venue"], f["_group"])])
    latest = {(o, g): factor(o, g) for o, g in state}
    return row, latest
