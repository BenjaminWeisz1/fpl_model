"""
One-off maintenance helper (CHANGELOG §81): rebuild 2025 team xG in match_results.csv from
summed FPL player xG.

All 380 Season-2025 rows carried Understat team xG (Home_/Away_xG = Understat match xG;
_NPxG and _NPxG_Adj = that - 0.79 x PKatt), splicing a third xG model between fbref/Opta
(2017-2024) and summed FPL/Opta (2026). For 2025 rows only this sets
  _xG                = sum of the team's FPL player xG (penalty-inclusive)
  _NPxG, _NPxG_Adj   = sum of the team's FPL player npxG (penalty-free since §80)
editing only those cells as text; every other cell keeps its exact text. 2025 had no
red-card adjustments, so no manual value is lost. The Understat values stay in
data/raw/understat/ and in the §81 evidence folder.

Run locally (writes a copy; --apply edits data/current/match_results.csv):
    python rebuild_2025_team_xg.py --out <path> | --apply
"""
from fpl_model import paths
import argparse
from pathlib import Path

import pandas as pd

_DATA = paths.CURRENT
MATCH_RESULTS_FILE = paths.CURRENT / "match_results.csv"
PLAYER_STATS_FILE = paths.CURRENT / "player_match_stats.csv"
SEASON = 2025
COLS = [f"{s}_{c}" for s in ("Home", "Away") for c in ("xG", "NPxG", "NPxG_Adj")]


def rebuilt_text():
    """Returns (text frame with 2025 team xG rebuilt, number of rows changed)."""
    text = pd.read_csv(MATCH_RESULTS_FILE, dtype=str, keep_default_na=False, encoding="utf-8")
    stats = pd.read_csv(PLAYER_STATS_FILE, encoding="utf-8-sig", low_memory=False)
    stats = stats[stats["Data_Source"].eq("FPL_API") & (stats["Season"] == SEASON)]
    team = stats.groupby(["Team", "Opponent", "Venue"])[["xG", "npxG"]].sum()
    rows = text.index[text["Season"] == str(SEASON)]
    assert len(rows) == 380, f"expected 380 Season-{SEASON} matches, found {len(rows)}"
    for side, other in (("Home", "Away"), ("Away", "Home")):
        idx = pd.MultiIndex.from_arrays([text.loc[rows, side], text.loc[rows, other],
                                         pd.Series(side, index=rows)])
        t = team.reindex(idx)
        assert t.notna().all().all(), f"{side}: team-matches without FPL player rows"
        text.loc[rows, f"{side}_xG"] = [repr(round(float(v), 4)) for v in t["xG"]]
        text.loc[rows, f"{side}_NPxG"] = [repr(round(float(v), 4)) for v in t["npxG"]]
        text.loc[rows, f"{side}_NPxG_Adj"] = [repr(round(float(v), 4)) for v in t["npxG"]]
    return text, len(rows)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path)
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    text, n = rebuilt_text()
    target = MATCH_RESULTS_FILE if args.apply else args.out
    text.to_csv(target, index=False, encoding="utf-8")
    print(f"Rebuilt {COLS} on {n} Season-{SEASON} rows -> {target}")


if __name__ == "__main__":
    main()
