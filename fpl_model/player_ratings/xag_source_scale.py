"""
fbref xAG v Opta xA: per-slot factors that put FPL-era expected assists on the fbref scale (CHANGELOG §242).

player_match_stats.csv's xAG is fbref's (2017-2024) then FPL's expected assists (2025+), which are Opta's xA. The two
models credit different passes: on the 2022-2024 overlap, fbref xAG / Opta xA is about 1.46 for forwards and 0.92 for
centre-backs. fbref xAG tracked conventional assists in every slot; Opta xA under-states forwards' (§241).

The factor per canonical slot = sum of fbref xAG / sum of Sofascore expected assists (which equal FPL's xA, ratio 1.00
in every slot in 2025-26) over 2022-2024 starts with both, computed from the data on each run. Only expected-assist
values are used, never assist outcomes. `overall` is the same ratio over all slots, for rows without a position.
"""
from pathlib import Path

import pandas as pd

from fpl_model import paths

CANON = {"CB": "CB", "LB": "FB", "RB": "FB", "WB": "WB", "DM": "DM", "CM": "CM", "AM": "AM",
         "LM": "WM", "RM": "WM", "LW": "W", "RW": "W", "FW": "FW", "GK": "GK"}
KEY = ["Season", "Team", "Opponent", "Venue", "Player"]
OVERLAP = (2022, 2024)


def slot_factors(data_dir):
    """{canonical slot: fbref xAG / Opta xA on the 2022-2024 overlap}, plus 'overall'. GK is left at 1."""
    d = Path(data_dir)
    s = pd.read_csv(paths.current_file("player_match_stats.csv", root=d), encoding="utf-8-sig", low_memory=False,
                    usecols=KEY + ["Minutes", "xAG", "Data_Source"])
    s = s[s["Season"].between(*OVERLAP) & s["Data_Source"].eq("fbref") & s["Minutes"].gt(0)]
    p = pd.read_csv(paths.current_file("player_match_positions.csv", root=d), encoding="utf-8-sig", low_memory=False,
                    usecols=KEY + ["Position_Resolved", "Is_Starter"]).drop_duplicates(KEY)
    sf = pd.read_csv(paths.current_file("sofascore_player_match_stats.csv", root=d), encoding="utf-8-sig", low_memory=False,
                     usecols=["Season", "Team", "Opponent", "Venue", "Local_Player", "Sofascore_minutesPlayed",
                              "Sofascore_expectedAssists"]).rename(columns={"Local_Player": "Player"})
    sf = sf[(sf["Sofascore_minutesPlayed"].fillna(0) > 0) & sf["Sofascore_expectedAssists"].notna()].drop_duplicates(KEY)
    m = s.merge(p, on=KEY).merge(sf, on=KEY)
    m = m[m["Is_Starter"].astype(str).str.lower().eq("true") & m["Sofascore_expectedAssists"].notna()]
    m["Slot"] = m["Position_Resolved"].map(CANON)
    m = m[m["Slot"].notna() & m["Slot"].ne("GK")]
    g = m.groupby("Slot")[["xAG", "Sofascore_expectedAssists"]].sum()
    out = (g["xAG"] / g["Sofascore_expectedAssists"]).to_dict()
    out["overall"] = float(m["xAG"].sum() / m["Sofascore_expectedAssists"].sum())
    out["GK"] = 1.0
    return {k: float(v) for k, v in out.items()}


def _starts_with_sofascore(data_dir, seasons):
    d = Path(data_dir)
    s = pd.read_csv(paths.current_file("player_match_stats.csv", root=d), encoding="utf-8-sig", low_memory=False,
                    usecols=KEY + ["Minutes", "xAG", "Data_Source"])
    s = s[s["Season"].between(*seasons) & s["Minutes"].gt(0)]
    p = pd.read_csv(paths.current_file("player_match_positions.csv", root=d), encoding="utf-8-sig", low_memory=False,
                    usecols=KEY + ["Position_Resolved", "Is_Starter"]).drop_duplicates(KEY)
    sf = pd.read_csv(paths.current_file("sofascore_player_match_stats.csv", root=d), encoding="utf-8-sig", low_memory=False,
                     usecols=["Season", "Team", "Opponent", "Venue", "Local_Player", "Sofascore_minutesPlayed",
                              "Sofascore_goalAssist", "Sofascore_expectedAssists"]).rename(columns={"Local_Player": "Player"})
    sf = sf[sf["Sofascore_minutesPlayed"].fillna(0) > 0].drop_duplicates(KEY)
    m = s.merge(p, on=KEY).merge(sf, on=KEY)
    m = m[m["Is_Starter"].astype(str).str.lower().eq("true")]
    m["Slot"] = m["Position_Resolved"].map(CANON)
    m = m[m["Slot"].notna() & m["Slot"].ne("GK")]
    m["assists"] = m["Sofascore_goalAssist"].fillna(0)   # Sofascore omits zero stats on played rows
    return m


def assist_factors(data_dir, source):
    """{slot: conventional assists / expected assists} plus 'overall' (CHANGELOG §243).
    source 'opta': Sofascore expected assists (= Opta xA = FPL's) on 2022-2024 starts, before the FPL-era test seasons.
    source 'fbref': fbref xAG on 2017-2024 starts. Assists are Sofascore's conventional assists in both."""
    if source == "opta":
        m = _starts_with_sofascore(data_dir, OVERLAP)
        m = m[m["Sofascore_expectedAssists"].notna()]
        x = "Sofascore_expectedAssists"
    elif source == "fbref":
        m = _starts_with_sofascore(data_dir, (2017, 2024))
        m = m[m["Data_Source"].eq("fbref")]
        x = "xAG"
    else:
        raise ValueError(source)
    g = m.groupby("Slot")[["assists", x]].sum()
    out = (g["assists"] / g[x]).to_dict()
    out["overall"] = float(m["assists"].sum() / m[x].sum())
    out["GK"] = 1.0
    return {k: float(v) for k, v in out.items()}


if __name__ == "__main__":
    import json
    root = Path(__file__).resolve().parents[2]
    print(json.dumps({k: round(v, 4) for k, v in slot_factors(root / "data" / "current").items()}, indent=1))
