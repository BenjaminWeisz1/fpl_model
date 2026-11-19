"""
Position rates from per-match slots, season by season (CHANGELOG §246; experiment, default off).

The starting ratings of youth, second-season, missed-season and arriving players carry a position part: the npxG/90
and xAG/90 of their listed position(s), from `position_combinations.csv` (hand-made from fbref 2017-2024 averages,
multi-position players split equally). This module computes the same table from Premier League starts instead:

    slot_rates(season, window)  ->  DataFrame Pos, NPG/90, Assists/90 (the shape of position_combinations.csv)

  * rows: PL starts with minutes > 0 in seasons before `season` (window "all": every earlier season; "3": the last 3),
    slot = player_match_positions.csv Position_Resolved mapped by SLOT_CANON (the labels of transfers.csv's single
    positions), rates minutes-weighted (sum of the stat / sum of minutes x 90);
  * the stats are the loop's own targets: player_match_stats.csv npxG / xAG with the §136 red-card player factors and
    the §243 assist calibration applied exactly as build_player_ratings.py applies them (the same switches);
  * xAG source break (CLAUDE.md §5.2): for season <= 2024 fbref-era rows only; for season >= 2025 Opta xA only, i.e.
    FPL-era rows plus 2022-2024 Sofascore `expectedAssists` (= Opta xA), with the same red-card factor and the same §243
    FW / CB factors. npxG has no break (all rows in the window);
  * a slot with fewer than MIN_STARTS starts in the window (per target) keeps the file's value; keepers keep the file.

Switch GOAL_MODEL_POSITION_RATES = file (default, the file) | slots (all earlier seasons) | slots3 (last 3 seasons).
Combined positions keep the §77 rule (each listed position counts twice the next) on top of these single-slot rates.
"""
from functools import cache, lru_cache

import numpy as np
import pandas as pd

from fpl_model import config, paths

DATA = paths.CURRENT
MODE = config.get("GOAL_MODEL_POSITION_RATES")
WINDOW = {"file": None, "slots": "all", "slots3": "3"}[MODE]
MIN_STARTS = 300            # pre-registered in data/validation/position_rates_20261114/phase2/protocol.json
OPTA_FROM = 2022            # first season with Sofascore expected assists (= Opta xA) for PL starts
FPL_ERA = 2025
SLOT_CANON = {"CB": "CB", "LB": "FB", "RB": "FB", "WB": "WB", "DM": "DM", "CM": "CM", "AM": "AM",
              "LM": "WM", "RM": "WM", "LW": "W", "RW": "W", "FW": "FW", "GK": "GK"}
KEY = ["Season", "Team", "Opponent", "Venue", "Player"]


def file_rates():
    return pd.read_csv(paths.current_file("position_combinations.csv", root=DATA)).set_index("Pos")


@lru_cache(maxsize=1)
def starts_frame():
    """Every PL start with minutes > 0: Season, Slot, Minutes, npxG, xAG (the loop's target), xA_opta (Opta-scale
    expected assists where they exist: FPL-era xAG, or 2022-2024 Sofascore expectedAssists), Data_Source."""
    s = pd.read_csv(paths.current_file("player_match_stats.csv", root=DATA), encoding="utf-8-sig", low_memory=False,
                    usecols=KEY + ["Minutes", "npxG", "xAG", "Data_Source"])
    s = s[s["Minutes"] > 0].copy()
    p = pd.read_csv(paths.current_file("player_match_positions.csv", root=DATA), encoding="utf-8-sig", low_memory=False,
                    usecols=KEY + ["Position_Resolved", "Is_Starter"]).drop_duplicates(KEY)
    s = s.merge(p, on=KEY, how="inner")
    s = s[s["Is_Starter"].astype(str).str.lower().eq("true")].copy()
    s["Slot"] = s["Position_Resolved"].map(SLOT_CANON)
    s = s[s["Slot"].notna()].copy()
    # §136 red-card player factors, as build_player_ratings.py (GOAL_MODEL_RED_CARD_FILE)
    rc_file = config.get("GOAL_MODEL_RED_CARD_FILE") or paths.current_file("red_card_player_factors.csv", root=DATA)
    fx = np.ones(len(s))
    fn = np.ones(len(s))
    if rc_file.exists():
        rc = pd.read_csv(rc_file, encoding="utf-8-sig")
        f = s[KEY].merge(rc[KEY + ["Factor_npxG", "Factor_xAG"]].drop_duplicates(KEY), on=KEY, how="left")
        assert len(f) == len(s)
        fn = f["Factor_npxG"].fillna(1.0).to_numpy()
        fx = f["Factor_xAG"].fillna(1.0).to_numpy()
    s["npxG"] = s["npxG"].to_numpy() * fn
    s["xAG"] = s["xAG"].to_numpy() * fx
    # Sofascore expected assists (= Opta xA) for the 2022-2024 starts
    sf = pd.read_csv(paths.current_file("sofascore_player_match_stats.csv", root=DATA), encoding="utf-8-sig", low_memory=False,
                     usecols=["Season", "Team", "Opponent", "Venue", "Local_Player", "Sofascore_minutesPlayed",
                              "Sofascore_expectedAssists"]).rename(columns={"Local_Player": "Player"})
    sf = sf[(sf["Sofascore_minutesPlayed"].fillna(0) > 0)].drop_duplicates(KEY)
    s = s.merge(sf[KEY + ["Sofascore_expectedAssists"]], on=KEY, how="left")
    fpl = s["Data_Source"].eq("FPL_API").to_numpy()
    sofa = (s["Season"].between(OPTA_FROM, FPL_ERA - 1) & s["Data_Source"].eq("fbref")
            & s["Sofascore_expectedAssists"].notna()).to_numpy()
    xa = np.where(fpl, np.nan, np.where(sofa, s["Sofascore_expectedAssists"].to_numpy(dtype=float) * fx, np.nan))
    # §243 calibration, as build_player_ratings.py (GOAL_MODEL_XAG_CALIBRATION): FPL-era rows' xAG is calibrated there;
    # here the same factors go on FPL-era rows and on the Sofascore Opta rows.
    cal = config.get("GOAL_MODEL_XAG_CALIBRATION")
    if cal != "off":
        from fpl_model.player_ratings.xag_source_scale import assist_factors
        opta = assist_factors(DATA, "opta")
        keep = ["FW", "CB"] if cal == "fw_cb" else None
        fac = s["Slot"].map(opta).fillna(opta["overall"])
        if keep is not None:
            fac = fac.where(s["Slot"].isin(keep), 1.0)
        fac = fac.to_numpy()
        s["xAG"] = np.where(fpl, s["xAG"].to_numpy() * fac, s["xAG"].to_numpy())
        xa = xa * fac
        if cal == "both":
            fb = assist_factors(DATA, "fbref")
            s["xAG"] = np.where(fpl, s["xAG"], s["xAG"] * s["Slot"].map(fb).fillna(fb["overall"]).to_numpy())
    s["xA_opta"] = np.where(fpl, s["xAG"].to_numpy(), xa)
    return s[["Season", "Team", "Opponent", "Venue", "Player", "Slot", "Minutes", "npxG", "xAG", "xA_opta",
              "Data_Source"]].reset_index(drop=True)


def window_seasons(season, window):
    if window == "all":
        return list(range(2017, season))
    if window == "3":
        return list(range(max(2017, season - 3), season))
    raise ValueError(window)


@cache
def _slot_rates(season, window):
    f = file_rates()
    out = f[["NPG/90", "Assists/90"]].copy()
    out["source_npxG"] = "file"
    out["source_xAG"] = "file"
    out["starts_npxG"] = 0
    out["starts_xAG"] = 0
    seasons = window_seasons(season, window)
    if not seasons:
        return out
    s = starts_frame()
    s = s[s["Season"].isin(seasons) & s["Slot"].ne("GK")]
    # npxG: every row in the window
    g = s.groupby("Slot").agg(x=("npxG", "sum"), m=("Minutes", "sum"), n=("npxG", "size"))
    for slot, r in g.iterrows():
        out.loc[slot, "starts_npxG"] = int(r["n"])
        if r["n"] >= MIN_STARTS:
            out.loc[slot, "NPG/90"] = r["x"] / r["m"] * 90
            out.loc[slot, "source_npxG"] = "slots"
    # xAG: fbref-era rows for season <= 2024; Opta xA rows (2022-2024 Sofascore, FPL era) from 2025
    if season < FPL_ERA:
        a = s[s["Data_Source"].eq("fbref")]
        col = "xAG"
    else:
        a = s[s["xA_opta"].notna()]
        col = "xA_opta"
    g = a.groupby("Slot").agg(x=(col, "sum"), m=("Minutes", "sum"), n=(col, "size"))
    for slot, r in g.iterrows():
        out.loc[slot, "starts_xAG"] = int(r["n"])
        if r["n"] >= MIN_STARTS:
            out.loc[slot, "Assists/90"] = r["x"] / r["m"] * 90
            out.loc[slot, "source_xAG"] = "slots"
    return out


def slot_rates(season, window=None):
    """position_combinations.csv's single-position table for `season` (Pos index, NPG/90, Assists/90 and diagnostics).
    window None uses the switch; "file" mode returns the file."""
    window = WINDOW if window is None else window
    if window is None:
        return file_rates()
    return _slot_rates(int(season), window).copy()


def combined_rate(pos, season, window=None):
    """The §77 rule on top of the season's single-slot rates: {'NPG/90', 'Assists/90'} for a listing like 'W,AM'."""
    single = slot_rates(season, window)
    parts = [p.strip() for p in str(pos).split(",")]
    w = [2.0 ** (len(parts) - 1 - i) for i in range(len(parts))]
    return {c: float(np.average(single.loc[parts, c], weights=w)) for c in ["NPG/90", "Assists/90"]}


def apply_season_rates(df, window=None):
    """Replace a merged frame's NPG/90 and Assists/90 by the season's rates for its Pos (rows with Pos and Season).
    No-op in file mode. Used by build_player_ratings.py after every positions_df merge."""
    window = WINDOW if window is None else window
    if window is None or not len(df):
        return df
    df = df.copy()
    ok = [isinstance(p, str) and s == s for p, s in zip(df["Pos"], df["Season"])]
    vals = [combined_rate(p, int(s), window) if k else None for p, s, k in zip(df["Pos"], df["Season"], ok)]
    df["NPG/90"] = [v["NPG/90"] if v else x for v, x in zip(vals, df["NPG/90"])]
    df["Assists/90"] = [v["Assists/90"] if v else x for v, x in zip(vals, df["Assists/90"])]
    return df


if __name__ == "__main__":
    import sys
    w = sys.argv[1] if len(sys.argv) > 1 else "all"
    rows = []
    for S in range(2017, 2027):
        t = slot_rates(S, w)
        for pos, r in t.iterrows():
            rows.append({"Season": S, "Pos": pos, **r.to_dict()})
    print(pd.DataFrame(rows).pivot(index="Pos", columns="Season", values="NPG/90").round(3).to_string())
    print(pd.DataFrame(rows).pivot(index="Pos", columns="Season", values="Assists/90").round(3).to_string())
