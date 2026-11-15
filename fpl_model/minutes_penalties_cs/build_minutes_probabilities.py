"""
Applies the AppPts/P60 lookup table to fplreview's xMins forecast (forecast_horizon.HORIZON gameweeks),
producing a shared intermediate table for every downstream facet (clean
sheets, defcon rate scaling, bonus-from-minutes) to consume.

xMins values above 90 (added time) are clipped to 90 before lookup -- the
overshoot is small (rarely more than a few minutes) and concentrated where
the P60 curve is already near-saturated, so clipping introduces negligible
error versus extrapolating.

Run locally:
    python build_minutes_probabilities.py
"""
import re
import sys

import numpy as np
import pandas as pd

from fpl_model import paths

# When stdout is a pipe (e.g. run_weekly_update.py's capture_output=True on
# Windows), Python encodes with cp1252, which cannot represent some player
# names ("Słonina" etc.).  Replace rather than crash -- the warnings these
# names appear in are diagnostic text only.
sys.stdout.reconfigure(errors="replace")
sys.stderr.reconfigure(errors="replace")

from fpl_model.forecast_horizon import GAMEWEEKS  # noqa: E402  (the forecast window, set in one place)
from fpl_model.player_availability import status_conflicts  # noqa: E402

_DATA = paths.CURRENT

MINUTES_FILE = paths.CURRENT / "minutes.csv"
NAME_COL = "Name"

LOOKUP_FILE = paths.CURRENT / "mins_lookup_table.csv"
PLAYERS_CURRENT_FILE = paths.CURRENT / "players_current.csv"

OUT_FILE = paths.CURRENT / "minutes_probabilities.csv"


def interpolate(xmins_series, lookup_x, lookup_y):
    """Linear interpolation against the lookup table, clipping inputs to the
    table's range first (handles both the >90 added-time case and any
    unexpected negative/blank values defensively)."""
    x_clipped = xmins_series.clip(lower=lookup_x.min(), upper=lookup_x.max())
    return np.interp(x_clipped, lookup_x, lookup_y)


def resolve_names(minutes, players_current):
    """fplreview's names mirror the raw FPL API (no disambiguation of its
    own), but players_current.csv's Web_Name may be disambiguated (e.g.
    "Sangare" -> "M. Sangare"/"I. Sangare" for two players sharing a raw
    web_name). A collision only ever happens ACROSS teams, so resolving via
    (raw name, Team_Short) rather than name alone fixes this with no manual
    pre-editing of minutes.csv needed each week.

    minutes.csv uses 3-letter team codes (CHE, ARS, ...) which match
    players_current.csv's Team_Short column -- NOT Team (full names like
    "Chelsea"). Keying on Team_Short is therefore essential for the join
    to actually work.

    Tries TWO lookups, in order: (1) treat the name as raw (Raw_Web_Name ->
    Web_Name) -- the expected case for a genuinely fresh fplreview paste;
    (2) treat the name as already-disambiguated (Web_Name -> Web_Name,
    i.e. a no-op) -- covers old files edited before this resolution logic
    existed, or any case where the name already happens to be correct."""
    team_col = "Team_Short" if "Team_Short" in players_current.columns else "Team"

    if "Raw_Web_Name" not in players_current.columns:
        print("WARNING: players_current.csv has no Raw_Web_Name column yet (run "
              "fetch_bootstrap.py at least once) -- falling back to matching on "
              "Web_Name directly, which will misresolve any disambiguated names.")
        raw_name_lookup = {}
    else:
        raw_name_lookup = dict(zip(
            zip(players_current["Raw_Web_Name"], players_current[team_col]),
            players_current["Web_Name"]))

    web_name_lookup = dict(zip(
        zip(players_current["Web_Name"], players_current[team_col]),
        players_current["Web_Name"]))

    def resolve_one(name, team):
        if (name, team) in raw_name_lookup:
            return raw_name_lookup[(name, team)]
        if (name, team) in web_name_lookup:
            return web_name_lookup[(name, team)]
        return name

    resolved = minutes.apply(lambda r: resolve_one(r[NAME_COL], r["Team"]), axis=1)

    known_web_names = set(players_current["Web_Name"])
    unresolved = (resolved == minutes[NAME_COL]) & (~minutes[NAME_COL].isin(known_web_names))
    if unresolved.any():
        print(f"WARNING: {unresolved.sum()} rows in minutes.csv have a (Name, Team) "
              f"combination not found in players_current.csv -- passing through "
              f"unresolved, check for a team-name or spelling mismatch:")
        print(minutes[unresolved][[NAME_COL, "Team"]].drop_duplicates().to_string(index=False))

    return resolved


def main():
    minutes = pd.read_csv(MINUTES_FILE)
    lookup = pd.read_csv(LOOKUP_FILE).sort_values('xMins')
    players_current = pd.read_csv(PLAYERS_CURRENT_FILE, encoding="utf-8-sig")
    lookup_x = lookup['xMins'].values
    lookup_apppts = lookup['AppPts'].values
    lookup_p60 = lookup['P60'].values

    # minutes.csv stays the rule (user decision); this only makes the disagreement visible (§168).
    conflicts = status_conflicts(players_current, minutes)
    if len(conflicts):
        print(f"WARNING: {len(conflicts)} players FPL marks unavailable (status u) still have xMins "
              f"in minutes.csv, so they keep a forecast -- check whether they have left:")
        print(conflicts[["Web_Name", "Team_Short", "Total_xMins"]].to_string(index=False))

    minutes = minutes.copy()
    minutes[NAME_COL] = resolve_names(minutes, players_current)

    out = minutes[['Pos', NAME_COL, 'Team']].copy()
    out = out.rename(columns={NAME_COL: 'Player'})

    # fplreview labels xMins columns by actual round number (e.g. 3_xMins,
    # 4_xMins, ... for rounds 3-12 when we're predicting the next 10 rounds).
    # Detect these dynamically and map them to GW1-GW<HORIZON> in sorted order, so
    # this script works regardless of which round the forecast starts at.
    xmins_cols = sorted(
        [c for c in minutes.columns if re.match(r'^\d+_xMins$', c)],
        key=lambda c: int(c.split('_')[0])
    )
    if len(xmins_cols) < len(GAMEWEEKS):
        raise ValueError(f"Expected {len(GAMEWEEKS)} xMins columns in minutes.csv, "
                         f"found {len(xmins_cols)}: {xmins_cols}")
    xmins_cols = xmins_cols[:len(GAMEWEEKS)]
    col_to_gw = dict(zip(xmins_cols, GAMEWEEKS))
    print(f"Detected xMins columns: {xmins_cols} -> GW{GAMEWEEKS[0]}-GW{GAMEWEEKS[-1]}")

    overshoot_summary = []
    for xmins_col, gw in col_to_gw.items():
        xmins = minutes[xmins_col]

        n_over90 = (xmins > 90).sum()
        max_val = xmins.max()
        overshoot_summary.append((gw, n_over90, max_val))

        out[f"xMins_GW{gw}"] = xmins
        out[f"AppPts_GW{gw}"] = interpolate(xmins, lookup_x, lookup_apppts).round(4)
        out[f"P60_GW{gw}"] = interpolate(xmins, lookup_x, lookup_p60).round(4)

    print("Overshoot summary (rows with xMins > 90, clipped before lookup):")
    for gw, n, maxval in overshoot_summary:
        print(f"  GW{gw}: {n} rows, max value {maxval}")

    out.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\nWrote {len(out)} players to {OUT_FILE}")
    print(out.head())


if __name__ == "__main__":
    main()
