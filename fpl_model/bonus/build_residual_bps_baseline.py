"""
Builds a per-player "residual BPS" baseline (mean + std) from the current
BPS rules, restating seasons played under older ones.

Seasons before bps_rules.CBI_ONE_PER_THREE_SEASON used the 1-per-2 CBI rate, so
they are adjusted; later seasons already use 1-per-3 and must not be. The model also
keeps the old "being tackled" deduction out of the residual: we do not have a
historical count of times-dispossessed, so the remaining bias is documented but
not subtracted.

The residual is actual BPS minus exactly the terms simulate_bonus.py adds
explicitly, so every component is counted once (CHANGELOG §64): goals conceded
for D/G at any minutes, keeper saves on 60+ minute rows at the simulator's save
scale, and the defcon count // 3 (D: CBI+Tkl, M/F: CBI+Tkl+Recov). Keep these
definitions in step with simulate_bonus.py. `add_known_bps(legacy=True)` gives the
pre-§64 subtraction (Tkl*2 + CBI//3 + Recov//3, no goals conceded or saves) for research.

When pass bands are active (pass_bands.active, CHANGELOG §68) the residual also
excludes actual Sofascore pass-band BPS and uses position-specific shrinkage
K_REM; residual_bps_mode.json records which mode was used, and simulate_bonus.py
refuses to run if its inputs disagree.

Since §146 the fit also uses the previous season, centred on the current season-position mean.

Run locally:
    python -m fpl_model.bonus.build_residual_bps_baseline
"""
import json

import numpy as np
import pandas as pd

from fpl_model import paths
from fpl_model.bonus import pass_bands
from fpl_model.bonus.bps_rules import (
    ASSIST_BPS,
    CBI_ONE_PER_THREE_SEASON,
    CLEAN_SHEET_BPS,
    GC_BPS,
    GOAL_BPS,
    MISS_PEN_BPS,
    PEN_GOAL_BPS,
    PEN_SAVE_BPS,
    SAVE_BPS,
)

_DATA = paths.CURRENT

STATS_FILE = paths.current_file("player_match_stats.csv", root=paths.CURRENT)  # consolidated file
PLAYERS_CURRENT_FILE = paths.current_file("players_current.csv", root=paths.CURRENT)
OUT_FILE = paths.current_file("residual_bps_baseline.csv", root=paths.CURRENT)

MIN_POS_ROWS_TO_CENTRE = 50

CS_BPS = CLEAN_SHEET_BPS

MIN_MATCHES_FOR_OWN_STD = 5
SHRINKAGE_MATCHES = 10


def add_known_bps(frame, legacy=False, save_scale=1.0):
    """Add the explicitly simulated BPS components and their sum, `known_bps`."""
    frame = frame.copy()
    full = frame["Minutes"] >= 60
    frame["minutes_bps"] = np.where(full, 6, 3)
    frame["goal_bps"] = frame["NPG"] * frame["Pos"].map(GOAL_BPS)
    frame["pen_goal_bps"] = frame["PK"] * PEN_GOAL_BPS
    frame["assist_bps"] = frame["Assists"] * ASSIST_BPS
    frame["cs_bps"] = np.where(
        frame["Pos"].isin(["D", "G"]) & (frame["Clean_Sheet"] > 0) & full, CS_BPS, 0)
    frame["missed_pens"] = (frame["PKatt"] - frame["PK"]).clip(lower=0)
    frame["miss_pen_bps"] = frame["missed_pens"] * MISS_PEN_BPS

    if legacy:
        defensive_bps = frame["CBI"] // 3 + frame["Tkl"] * 2 + frame["Recov"] // 3
        gc_bps = 0
        save_bps = 0
    else:
        # The simulator's defcon count is the defcon model's target, scored // 3.
        defcon_count = np.where(
            frame["Pos"].eq("D"), frame["CBI"] + frame["Tkl"],
            np.where(frame["Pos"].isin(["M", "F"]),
                     frame["CBI"] + frame["Tkl"] + frame["Recov"], 0))
        defensive_bps = defcon_count // 3
        gc_bps = np.where(frame["Pos"].isin(["D", "G"]), frame["Goals_Conceded"] * GC_BPS, 0)
        save_bps = np.where(
            frame["Pos"].eq("G") & full,
            save_scale * (frame["Saves"] * SAVE_BPS + frame["Penalties_Saved"] * PEN_SAVE_BPS),
            0)
    frame["defensive_bps"] = defensive_bps
    frame["gc_bps"] = gc_bps
    frame["save_bps"] = save_bps

    frame["known_bps"] = (
        frame["minutes_bps"] + frame["goal_bps"] + frame["pen_goal_bps"] + frame["assist_bps"]
        + frame["cs_bps"] + frame["miss_pen_bps"]
        + frame["defensive_bps"] + frame["gc_bps"] + frame["save_bps"]
    )
    return frame


def subtract_pass_bands(frame):
    """Remove actual Sofascore pass-band BPS from the residual (§68); unmatched rows count 0."""
    so = pass_bands.load_sofascore()
    key = ["Season", "Team", "Opponent", "Venue", "Player"]
    so = so.rename(columns={"Local_Player": "Player"}).dropna(subset=["Player"]).drop_duplicates(key)
    frame = frame.merge(so[key + ["band"]], on=key, how="left")
    frame["pass_band_bps"] = frame["band"].fillna(0)
    frame["residual"] = frame["residual"] - frame["pass_band_bps"]
    return frame.drop(columns="band"), float(frame["band"].notna().mean())


def aggregate_residuals(rule_df, shrinkage=SHRINKAGE_MATCHES):
    """Per-player residual mean/std with position-level shrinkage, plus position defaults.

    `shrinkage` is a number of matches, or a dict of them by position.
    """
    player_agg = rule_df.groupby(["Player", "Pos"]).agg(
        residual_mean=("residual", "mean"),
        residual_std=("residual", "std"),
        n_matches=("residual", "count"),
    ).reset_index()

    pos_agg = rule_df.groupby("Pos")["residual"].agg(["mean", "std"]).rename(
        columns={"mean": "pos_mean", "std": "pos_std"})

    player_agg = player_agg.merge(pos_agg, on="Pos", how="left")

    k = player_agg["Pos"].map(shrinkage) if isinstance(shrinkage, dict) else shrinkage
    player_agg["ResidualBPS_Mean"] = (
        (player_agg["n_matches"] * player_agg["residual_mean"] + k * player_agg["pos_mean"]) /
        (player_agg["n_matches"] + k)
    )
    player_agg["ResidualBPS_Std"] = np.where(
        player_agg["n_matches"] >= MIN_MATCHES_FOR_OWN_STD,
        player_agg["residual_std"].fillna(player_agg["pos_std"]),
        player_agg["pos_std"],
    )

    out = player_agg[["Player", "Pos", "ResidualBPS_Mean", "ResidualBPS_Std", "n_matches"]]
    pos_defaults = pos_agg.reset_index().rename(columns={"Pos": "Pos_"})
    return out, pos_defaults


def centre_previous_season(rule_df, season):
    """Move earlier rows to the current season-position mean (§146), so a rule or definition change
    cannot move a level; no shift while a position has under MIN_POS_ROWS_TO_CENTRE current rows."""
    rule_df = rule_df.copy()
    cur = rule_df[rule_df["Season"] == season].groupby("Pos")["residual"].agg(["mean", "size"])
    old = rule_df["Season"] < season
    old_mean = rule_df[old].groupby("Pos")["residual"].mean()
    shift = (rule_df.loc[old, "Pos"].map(cur["mean"]) - rule_df.loc[old, "Pos"].map(old_mean)).where(
        rule_df.loc[old, "Pos"].map(cur["size"]).fillna(0) >= MIN_POS_ROWS_TO_CENTRE, 0).fillna(0)
    rule_df.loc[old, "residual"] = rule_df.loc[old, "residual"] + shift
    return rule_df


def main():
    all_stats = pd.read_csv(STATS_FILE, encoding="utf-8-sig")
    # Only FPL_API rows have BPS data (fbref rows don't have this column)
    df = all_stats[all_stats["Data_Source"] == "FPL_API"].copy()
    players_current = pd.read_csv(PLAYERS_CURRENT_FILE, encoding="utf-8-sig")

    pos_lookup = dict(zip(players_current["Player"], players_current["Pos"]))
    df["Pos"] = df["Player"].map(pos_lookup)
    df = df[df["Pos"].isin(["D", "M", "F", "G"])].copy()
    df = df[df["Minutes"] > 0].copy()

    # Seasons before CBI_ONE_PER_THREE_SEASON paid 1 BPS per 2 CBI, so their BPS is restated at the current
    # rate. The baseline is the latest season plus, since §146, the one before it (§181: from the data).
    df = add_known_bps(df)
    old_rule = df["Season"] < CBI_ONE_PER_THREE_SEASON
    cbi_adjustment = np.where(old_rule, df["CBI"] // 2 - df["CBI"] // 3, 0)
    df["residual"] = df["BPS"] - cbi_adjustment - df["known_bps"]
    latest = int(df["Season"].max())
    current_df = df[df["Season"] == latest]
    previous_rows = df[df["Season"] == latest - 1]  # §146: also fit on the previous season
    rule_df = pd.concat([current_df, previous_rows], ignore_index=True)
    if rule_df.empty:
        raise ValueError("No FPL_API rows available to build residual BPS baseline")

    # Exclude the removed being-tackled penalty from the historical residual:
    # we do not track dispossession counts, so a residual from 2025/26 still carries a
    # small downward bias for players who are tackled often; this is the documented limit.
    rule_df["residual"] = rule_df["residual"].astype(float)

    # Pass bands (§68) leave the residual only when the simulator will draw them.
    use_bands, reason = pass_bands.active()
    shrinkage = SHRINKAGE_MATCHES
    if use_bands:
        rule_df, matched = subtract_pass_bands(rule_df)
        shrinkage = pass_bands.K_REM
        reason += f"; {matched:.1%} of rows matched to Sofascore"
    if not previous_rows.empty:
        rule_df = centre_previous_season(rule_df, latest)
    with open(paths.current_file("residual_bps_mode.json", root=_DATA), "w", encoding="utf-8") as h:
        json.dump({"pass_bands": use_bands, "reason": reason}, h, indent=2)

    # ---- per-player aggregation with position-level shrinkage ----
    out, pos_defaults = aggregate_residuals(rule_df, shrinkage)

    print(f"Accounting: consistent (§64); "
          f"pass bands {'on' if use_bands else 'off'} ({reason})")
    for _, prow in pos_defaults.iterrows():
        print(f"Position default ({prow['Pos_']}): mean={prow['pos_mean']:.2f}, std={prow['pos_std']:.2f}")

    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    pos_defaults.to_csv(paths.current_file("residual_bps_position_defaults.csv", root=_DATA), index=False, encoding="utf-8-sig")
    print(f"\nBuilt current-rule residual baseline on {len(rule_df)} rows from {rule_df['Season'].nunique()} season(s)")
    print(f"Wrote {len(out)} players to {OUT_FILE}")
    print(out.sort_values("ResidualBPS_Mean", ascending=False).head(10))


if __name__ == "__main__":
    main()
