"""
Validation of the native-Season fix in build_player_ratings.py (CHANGELOG §74).

Compares two GOAL_MODEL_PREMATCH_LOG runs (baseline = before the fix, candidate =
after) under data/validation/season_fix_20261028/:
  identity  players with no transfers.csv row of Season >= 2026 must have identical
            logged pre-match ratings and identical final ratings;
  accuracy  2026 rows of players the baseline never rated (group a): minutes-weighted
            MAE of npxG/90 and xAG/90, rating held at its initial value vs updated,
            each player's first 2026 row excluded; group b (Season-2026 rows plus
            earlier history) as baseline log vs candidate log.

Run locally (after both logged runs):
    python validate_season_fix.py [--dir <run directory>]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "validation" / "season_fix_20261028"
KEY = ["Player", "Team", "Opponent", "Venue", "MatchID", "Data_Source"]


def mae(err, w):
    """Weighted MAE; NaN errors are not skipped silently but make the result NaN."""
    return float((np.abs(err) * w).sum(min_count=len(err)) / w.sum()) if err.notna().all() else float("nan")


def occurrence(frame):
    # A few log keys repeat (duplicated 2025 Junior Kroupi rows); pair them by order.
    return frame.assign(_occ=frame.groupby(KEY).cumcount())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=DIR)
    run_dir = parser.parse_args().dir
    base = pd.read_csv(run_dir / "baseline_log.csv", encoding="utf-8-sig")
    cand = pd.read_csv(run_dir / "candidate_log.csv", encoding="utf-8-sig")
    rb = pd.read_csv(run_dir / "baseline_log.csv.ratings.csv", encoding="utf-8-sig")
    rc = pd.read_csv(run_dir / "candidate_log.csv.ratings.csv", encoding="utf-8-sig")
    transfers = pd.read_csv(ROOT / "data" / "current" / "transfers.csv")
    players = pd.read_csv(ROOT / "data" / "current" / "players_current.csv", encoding="utf-8-sig")
    touched = set(transfers.loc[transfers["Season"] >= 2026, "Player"])
    web = dict(zip(players["Player"], players["Web_Name"]))
    touched_web = {web[p] for p in touched if p in web}

    # identity on untouched players
    bu, cu = base[~base["Player"].isin(touched)], cand[~cand["Player"].isin(touched)]
    m = occurrence(bu).merge(occurrence(cu), on=KEY + ["_occ"], how="outer", suffixes=("_b", "_c"), indicator=True)
    rows_equal = bool((m["_merge"] == "both").all() and (m["NPG_Rating_Pre_b"] == m["NPG_Rating_Pre_c"]).all()
                      and (m["Assist_Rating_Pre_b"] == m["Assist_Rating_Pre_c"]).all())
    fr = rb[~rb["Player"].isin(touched_web)].merge(rc[~rc["Player"].isin(touched_web)], on=["Player", "Team"],
                                                   how="outer", suffixes=("_b", "_c"), indicator=True)
    final_equal = bool((fr["_merge"] == "both").all() and (fr["NPG/90_b"] == fr["NPG/90_c"]).all()
                       and (fr["Assists/90_b"] == fr["Assists/90_c"]).all())

    # group a: 2026 rows of players never rated by the baseline
    rated_before = set(base["Player"])
    c26 = cand[(cand["Season"] >= 2026)].sort_values("MatchID")
    a = c26[~c26["Player"].isin(rated_before)].copy()
    first = a.groupby("Player")["MatchID"].transform("min")
    init = a.groupby("Player")[["NPG_Rating_Pre", "Assist_Rating_Pre"]].transform("first")
    a = a[a["MatchID"] != first]
    # predictions are proportional to the rating, so the held-rating prediction rescales
    held_g = a["pred_goals_90"] * init.loc[a.index, "NPG_Rating_Pre"] / a["NPG_Rating_Pre"]
    held_a = a["pred_ast_90"] * init.loc[a.index, "Assist_Rating_Pre"] / a["Assist_Rating_Pre"]
    group_a = {"players": int(a["Player"].nunique()), "rows": int(len(a)),
               "npxg90_mae_held": mae(a["npxg90"] - held_g, a["weight"]),
               "npxg90_mae_updated": mae(a["npxg90"] - a["pred_goals_90"], a["weight"]),
               "xag90_mae_held": mae(a["xag90"] - held_a, a["weight"]),
               "xag90_mae_updated": mae(a["xag90"] - a["pred_ast_90"], a["weight"])}
    for k in ("npxg90", "xag90"):
        group_a[f"{k}_mae_change_pct"] = 100 * (group_a[f"{k}_mae_updated"] / group_a[f"{k}_mae_held"] - 1)

    # group b: touched players the baseline did rate, on their 2026 rows
    bb = base[base["Player"].isin(touched) & (base["Season"] >= 2026)]
    cb = cand[cand["Player"].isin(touched) & (cand["Season"] >= 2026)]
    jb = occurrence(bb).merge(occurrence(cb), on=KEY + ["_occ"], suffixes=("_b", "_c"))
    group_b = {"players": int(jb["Player"].nunique()), "rows": int(len(jb)),
               "candidate_rows_with_nan_rating": int(jb["NPG_Rating_Pre_c"].isna().sum()),
               "candidate_players_with_nan_rating": int(jb.loc[jb["NPG_Rating_Pre_c"].isna(), "Player"].nunique()),
               "rows_with_changed_rating": int((jb["NPG_Rating_Pre_b"] != jb["NPG_Rating_Pre_c"]).sum())}
    if len(jb):
        group_b.update({"npxg90_mae_baseline": mae(jb["npxg90_b"] - jb["pred_goals_90_b"], jb["weight_b"]),
                        "npxg90_mae_candidate": mae(jb["npxg90_c"] - jb["pred_goals_90_c"], jb["weight_c"]),
                        "xag90_mae_baseline": mae(jb["xag90_b"] - jb["pred_ast_90_b"], jb["weight_b"]),
                        "xag90_mae_candidate": mae(jb["xag90_c"] - jb["pred_ast_90_c"], jb["weight_c"])})

    # effect on final ratings of touched players
    ft = rb[rb["Player"].isin(touched_web)].merge(rc[rc["Player"].isin(touched_web)], on=["Player", "Team"],
                                                  suffixes=("_b", "_c"))
    ft["dNPG"] = ft["NPG/90_c"] - ft["NPG/90_b"]
    ft["dAst"] = ft["Assists/90_c"] - ft["Assists/90_b"]
    out = {"identity_untouched_prematch_rows": rows_equal, "identity_untouched_final_ratings": final_equal,
           "untouched_rows_compared": int(len(m)), "untouched_players_compared": int(len(fr)),
           "group_a": group_a, "group_b": group_b,
           "final_ratings_touched": {"players": int(len(ft)), "changed": int((ft["dNPG"].abs() > 1e-12).sum()),
                                     "mean_abs_dNPG90": float(ft["dNPG"].abs().mean()),
                                     "mean_abs_dAst90": float(ft["dAst"].abs().mean()),
                                     "largest": ft.reindex(ft["dNPG"].abs().sort_values(ascending=False).index)
                                     .head(10)[["Player", "Team", "NPG/90_b", "NPG/90_c", "Assists/90_b", "Assists/90_c"]]
                                     .round(4).to_dict("records")}}
    out["candidate_final_ratings_nan"] = int(rc["NPG/90"].isna().sum())
    out["decision"] = {"identity": rows_equal and final_equal,
                       "no_nan_ratings": out["candidate_final_ratings_nan"] == 0,
                       "npxg_not_worse": group_a["npxg90_mae_updated"] <= group_a["npxg90_mae_held"]}
    out["decision"]["adopt"] = all(out["decision"].values())
    (run_dir / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
