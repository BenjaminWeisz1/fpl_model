"""Chronological order of player_match_stats.csv rows (§188).

FPL-era MatchIDs are per-row IDs in fetch order, and in 2026 GW1-2 that order isn't the order the matches
were played. fixtures.csv Round is the only date-like field covering those rows, so order on (Season, Round),
with MatchID keeping the existing order inside a round and in seasons fixtures.csv doesn't cover. MatchIDs
themselves are never renumbered.
"""

import numpy as np
import pandas as pd

from fpl_model import paths


def chrono_order(frame, data_dir):
    """A unique integer per row of frame (columns Season, MatchID, Team, Opponent, Venue); ascending = chronological."""
    fixtures = pd.read_csv(paths.current_file("fixtures.csv", root=data_dir))
    rounds = {(s, h.strip(), a.strip()): r for s, h, a, r in
              zip(fixtures["Season"], fixtures["Home"], fixtures["Away"], fixtures["Round"])}
    is_home = frame["Venue"].eq("Home")
    home = frame["Team"].where(is_home, frame["Opponent"])
    away = frame["Opponent"].where(is_home, frame["Team"])
    round_no = np.array([rounds.get(k, np.nan) for k in zip(frame["Season"], home, away)], dtype=float)

    covered = frame["Season"].isin(set(fixtures["Season"])).to_numpy()
    lost = covered & np.isnan(round_no)
    if lost.any():
        names = sorted(set(zip(frame["Season"][lost], home[lost], away[lost])))
        print(f"WARNING: {int(lost.sum())} rows have no fixtures.csv round and sort last in their season: {names[:5]}")
    round_no = np.where(covered, np.where(np.isnan(round_no), np.inf, round_no), 0.0)

    key = pd.DataFrame({"Season": frame["Season"].to_numpy(), "Round": round_no, "MatchID": frame["MatchID"].to_numpy()})
    ordered = key.sort_values(["Season", "Round", "MatchID"], kind="stable").index.to_numpy()
    order = np.empty(len(frame), dtype=np.int64)
    order[ordered] = np.arange(len(frame))
    return pd.Series(order, index=frame.index)
