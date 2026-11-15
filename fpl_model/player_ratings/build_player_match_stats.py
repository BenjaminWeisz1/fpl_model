"""
Merges all_player_stats.csv (fbref, 2017-2024) and player_stats_2025_with_npg.csv
(FPL-API, 2025/26+) into player_match_stats.csv -- one unified schema, with
genuine NULLs (not guessed values) wherever a source doesn't provide a given
stat, and a Data_Source column so downstream models (e.g. the defcon model)
can use source as an explicit feature rather than the two eras being silently
indistinguishable.

Column mapping:
  - Shared concepts get ONE unified name (e.g. fbref's "Min" and FPL's
    "minutes" both become "Minutes").
  - fbref-only concepts (shots, fouls, crosses, offside, individual
    clearances/blocks/interceptions, Team_Attack_Rating, Moved_Club) are
    NULL for FPL-API rows.
  - FPL-only concepts (bonus, BPS, clean sheets, saves, defensive_contribution,
    penalties missed/saved) are NULL for fbref rows.
  - Position is NULL for FPL-API rows -- that source has no detailed tactical
    position data at all (confirmed when building player_matchlogs.csv
    earlier in this project).
  - NPG is computed consistently as Goals-PK for BOTH sources (not taken
    directly from FPL's own non_penalty_goals column), so the derivation
    method doesn't silently differ by era.
  - CBI (Clearances+Blocks+Interceptions) is computed for both sources too --
    summed from the three separate fbref columns, taken directly from FPL's
    native combined column -- since this is what the defcon model actually
    trains on. CAVEAT (measured 2026-09): the two sources' CBI values are
    NOT on the same scale for the same players -- FPL's mid/fwd CBI runs
    ~26% below fbref's (defender CBI is continuous across the switch), and
    FPL recoveries run 14-24% below fbref's for everyone; tackles match.
    This file deliberately keeps the RAW native values (the BPS baseline
    models FPL's actual bonus scoring from FPL-native counts); the defcon
    model reconciles the scales at load time (see _FPL_STAT_SCALE in
    scripts/defcon/predict.py).

One-off, already run: refuses to run while player_match_stats.csv exists. Don't delete it to get past that.
"""
from fpl_model import paths
import pandas as pd
from pathlib import Path

_DATA = paths.CURRENT
_RAW = paths.RAW

OLD_FILE = paths.RAW / "all_player_stats.csv"      # fbref, 2017-2024
NEW_FILE = paths.RAW / "player_stats_2025_with_npg.csv"  # FPL-API, 2025/26+
OUT_FILE = paths.CURRENT / "player_match_stats.csv"


def main():
    # This rebuild knows only the frozen raw files: it would delete every row fetched since, the penalty
    # reconciliation, and the repairs made in place (§182, §189).
    if OUT_FILE.exists():
        raise SystemExit(f"{OUT_FILE} exists and this script would overwrite it, losing the current season. "
                         f"Nothing was written. To rebuild on purpose, move the file out first.")
    old = pd.read_csv(OLD_FILE)
    new = pd.read_csv(NEW_FILE, encoding="utf-8-sig")
    new = new.loc[:, ~new.columns.str.startswith("Unnamed")]
    # Never count a row twice: six exact duplicate Kroupi rows were found in player_match_stats.csv (CHANGELOG §129;
    # not from this raw file, which has none). Guard only.
    n_dup = int(new.duplicated().sum())
    if n_dup:
        print(f"Dropping {n_dup} exact duplicate rows from {NEW_FILE.name}")
        new = new.drop_duplicates()

    # ---- fbref portion ----
    old_out = pd.DataFrame({
        "MatchID": old["MatchID"],
        "Player": old["Player"],
        "Team": old["Team"],
        "Opponent": old["Opponent"],
        "Venue": old["Venue"],
        "Position": old["Pos"],
        "Minutes": old["Min"],
        "PK": old["PK"],
        "PKatt": old["PKatt"],
        "CrdY": old["CrdY"],
        "CrdR": old["CrdR"],
        "Sh": old["Sh"],
        "SoT": old["SoT"],
        "npxG": old["npxG"],
        "xAG": old["xAG"],
        "Goals": old["Goals"],
        "Assists": old["Assists"],
        "Touches_ITB": old["Touches itb"],
        "Fls": old["Fls"],
        "Fld": old["Fld"],
        "Tkl": old["Tkl"],
        "Int": old["Int"],
        "Blocks": old["Blocks"],
        "Clr": old["Clr"],
        "Off": old["Off"],
        "Crs": old["Crs"],
        "TklW": old["TklW"],
        "Recov": old["Recov"],
        "Moved_Club": old["Moved_Club"],
        "CBI": old["Clr"] + old["Blocks"] + old["Int"],
        "Data_Source": "fbref",
    })
    old_out["NPG"] = old_out["Goals"] - old_out["PK"]

    # ---- FPL-API portion ----
    MATCHID_OFFSET = old["MatchID"].max() + 1
    new_out = pd.DataFrame({
        "MatchID": MATCHID_OFFSET + new["round"],
        "Player": new["player"],
        "Team": new["Team"],
        "Opponent": new["opponent_team"],
        "Venue": new["Venue"],
        "Position": pd.NA,  # no detailed position data in this source at all
        "Minutes": new["minutes"],
        "PK": new["penalty_goals"],
        "PKatt": new["penalty_attempts"],
        "CrdY": new["yellow_cards"],
        "CrdR": new["red_cards"],
        # FPL xG includes penalties: npxG is derived from xG by penalty_reconcile.py --apply
        "npxG": new["expected_goals"],
        "xG": new["expected_goals"],
        "xAG": new["expected_assists"],
        "Goals": new["goals_scored"],
        "Assists": new["assists"],
        "Tkl": new["tackles"],
        "Recov": new["recoveries"],
        "CBI": new["clearances_blocks_interceptions"],
        "Defensive_Contribution": new["defensive_contribution"],
        "Clean_Sheet": new["clean_sheets"],
        "Goals_Conceded": new["goals_conceded"],
        "Own_Goals": new["own_goals"],
        "Saves": new["saves"],
        "Bonus": new["bonus"],
        "BPS": new["bps"],
        "Penalties_Missed": new["penalties_missed"],
        "Penalties_Saved": new["penalties_saved"],
        "Data_Source": "FPL_API",
    })
    new_out["NPG"] = new_out["Goals"] - new_out["PK"]

    combined = pd.concat([old_out, new_out], ignore_index=True, sort=False)
    combined = combined.sort_values(["MatchID", "Team"]).reset_index(drop=True)

    # ---- validation ----
    expected_rows = len(old) + len(new)
    assert len(combined) == expected_rows, f"Row count mismatch: {len(combined)} vs expected {expected_rows}"

    fbref_rows = combined[combined["Data_Source"] == "fbref"]
    fpl_rows = combined[combined["Data_Source"] == "FPL_API"]
    print(f"Total rows: {len(combined)} ({len(fbref_rows)} fbref + {len(fpl_rows)} FPL_API)")

    # spot-check: NPG should be consistent with FPL's own non_penalty_goals for the FPL portion
    npg_check = new_out["NPG"].values == new["non_penalty_goals"].values
    print(f"NPG (computed as Goals-PK) matches FPL's own non_penalty_goals column: {npg_check.all()} "
          f"({npg_check.sum()}/{len(npg_check)} rows)")

    print("\nNull counts, fbref-only columns, FPL_API rows (should all equal len(fpl_rows) if truly fbref-only):")
    fbref_only_cols = ["Sh", "SoT", "Touches_ITB", "Fls", "Fld", "Int", "Blocks", "Clr", "Off", "Crs", "TklW", "Moved_Club"]
    print(fpl_rows[fbref_only_cols].isna().sum())

    print("\nNull counts, FPL-only columns, fbref rows (should all equal len(fbref_rows) if truly FPL-only):")
    fpl_only_cols = ["Defensive_Contribution", "Clean_Sheet", "Goals_Conceded", "Own_Goals", "Saves", "Bonus", "BPS", "Penalties_Missed", "Penalties_Saved"]
    print(fbref_rows[fpl_only_cols].isna().sum())

    combined.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()