"""Standalone prototype: size the sustained-role-change detector (AGENDA §4 (C)).

Replicates the exact row ordering / minutes filter build_player_ratings.py uses,
merges Position_Resolved on the NATIVE Season, canonicalizes slots, and counts
how many sustained role changes the detector would fire and how many player-match
rows fall inside a boost window. Structural sizing only -- no ratings, no MAE.
"""
import os
from collections import deque, defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

_DATA = Path(__file__).resolve().parent / "data" / "current"

# canonical map: resolved fbref-style slot -> position_combinations single-slot key
CANON = {
    "CB": "CB", "LB": "FB", "RB": "FB", "WB": "WB",
    "DM": "DM", "CM": "CM", "AM": "AM",
    "LM": "WM", "RM": "WM", "LW": "W", "RW": "W",
    "FW": "FW", "GK": "GK",
}
COARSE = {
    "CB": "DEF", "FB": "DEF", "WB": "DEF",
    "DM": "MID", "CM": "MID", "AM": "MID", "WM": "MID", "W": "MID",
    "FW": "FWD", "GK": "GK",
}

W = int(os.environ.get("W", 5))
MIN_MATCHES = int(os.environ.get("MIN_MATCHES", 8))
COOLDOWN = int(os.environ.get("COOLDOWN", 10))
FACTOR = float(os.environ.get("FACTOR", 1.5))

# position prior table (single-slot rows only)
pc = pd.read_csv(_DATA / "position_combinations.csv")
pc_single = pc[~pc["Pos"].str.contains(",")]
PRIOR = {r["Pos"]: (r["NPG/90"], r["Assists/90"]) for _, r in pc_single.iterrows()}


def attack(p):
    return PRIOR[p][0] + PRIOR[p][1]


def gate(ref, new):
    if ref not in PRIOR or new not in PRIOR:
        return False
    if COARSE.get(ref) != COARSE.get(new):
        return True
    a, b = attack(ref), attack(new)
    if a <= 0 or b <= 0:
        return False
    return max(a / b, b / a) >= FACTOR


pms = pd.read_csv(_DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
pms["_NativeSeason"] = pms["Season"]
pos = pd.read_csv(_DATA / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False)
pos = pos.dropna(subset=["Position_Resolved"])

m = pms.merge(
    pos[["Season", "Team", "Opponent", "Venue", "Player", "Position_Resolved"]],
    left_on=["_NativeSeason", "Team", "Opponent", "Venue", "Player"],
    right_on=["Season", "Team", "Opponent", "Venue", "Player"],
    how="left", suffixes=("", "_pos"),
)
print(f"pms rows {len(pms)}; merged Position_Resolved non-null "
      f"{int(m['Position_Resolved'].notna().sum())} "
      f"({m['Position_Resolved'].notna().mean():.1%})")

m = m.sort_values("MatchID").reset_index(drop=True)
m["PriorMatches"] = m.groupby("Player").cumcount()

m["_canon"] = m["Position_Resolved"].map(CANON)

recent = defaultdict(lambda: deque(maxlen=W))
ref_slot = {}
ref_count = defaultdict(Counter)
boost = defaultdict(int)
last_fire = {}
fires = []
boost_rows = 0

for _, row in m.iterrows():
    minutes = row.get("Minutes", 0)
    if not minutes or minutes <= 0:
        continue
    player = row["Player"]
    slot = row["_canon"]
    size = row["PriorMatches"]
    if pd.isna(slot):
        continue
    # establish reference over first MIN_MATCHES resolved matches
    if player not in ref_slot:
        ref_count[player][slot] += 1
        if sum(ref_count[player].values()) >= MIN_MATCHES:
            ref_slot[player] = ref_count[player].most_common(1)[0][0]
        continue
    recent[player].append(slot)
    fired = False
    if len(recent[player]) == W and len(set(recent[player])) == 1:
        S = slot
        if S != ref_slot[player] and gate(ref_slot[player], S):
            if size - last_fire.get(player, -10**9) >= COOLDOWN:
                fires.append((player, ref_slot[player], S, int(row["_NativeSeason"]),
                              int(row["MatchID"])))
                ref_slot[player] = S
                boost[player] = 8
                last_fire[player] = size
                recent[player].clear()
                fired = True
    if boost[player] > 0 and not fired:
        boost_rows += 1
        boost[player] -= 1
    elif fired:
        boost_rows += 1
        boost[player] -= 1

print(f"\nW={W} MIN_MATCHES={MIN_MATCHES} COOLDOWN={COOLDOWN} FACTOR={FACTOR}")
print(f"role-change fires: {len(fires)}")
print(f"distinct players with >=1 fire: {len(set(f[0] for f in fires))}")
print(f"boost-active player-match rows: {boost_rows}")
cc = Counter((f[1], f[2]) for f in fires)
print("\ntop transitions (ref->new):")
for (a, b), n in cc.most_common(20):
    print(f"  {a:>3} -> {b:<3} : {n}")
print("\nexample fires (first 15):")
for f in fires[:15]:
    print("  ", f)
