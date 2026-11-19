"""
Check 4 of the FPL-era npxG penalty fix (CHANGELOG §80).

Compares the §78 baseline GOAL_MODEL_PREMATCH_LOG run (penalty-inclusive FPL-era npxG) with a
run on the fixed data, under data/validation/npxg_penalty_20261028/:
  identity  players with no FPL-era penalty attempt: logged pre-match rows and final ratings
            exactly identical;
  effect    penalty takers' final NPG/90 and Assists/90 before and after.

Run locally (after the logged run):
    python -m research.player_ratings.validate_npxg_penalty
"""
import json

import pandas as pd

from fpl_model import paths
from research.player_ratings.validate_opponent_lookup import paired

ROOT = paths.ROOT
DIR = paths.VALIDATION / "npxg_penalty_20261028"
BASE = paths.VALIDATION / "opponent_lookup_20261028"
VALUES = ["NPG_Rating_Pre", "Assist_Rating_Pre", "npxg90", "pred_goals_90", "xag90", "pred_ast_90", "weight"]


def main():
    read = lambda p: pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
    stats = read(paths.current_file("player_match_stats.csv"))
    players = read(paths.current_file("players_current.csv"))
    fpl = stats[stats["Data_Source"].eq("FPL_API") & (stats["Season"] >= 2025)]
    takers = set(fpl.loc[fpl["PKatt"] > 0, "Player"])
    web = dict(zip(players["Player"], players["Web_Name"]))
    takers_web = {web[p] for p in takers if p in web}

    j = paired(read(BASE / "baseline_log.csv"), read(DIR / "after_log.csv"))
    other = j[~j["Player"].isin(takers)]
    rows_equal = bool(all((other[f"{c}_b"] == other[f"{c}_c"]).all() for c in VALUES))

    rb, ra = read(BASE / "baseline_log.csv.ratings.csv"), read(DIR / "after_log.csv.ratings.csv")
    f = rb.merge(ra, on=["Player", "Team"], how="outer", suffixes=("_b", "_a"), indicator=True)
    assert (f["_merge"] == "both").all()
    rest = f[~f["Player"].isin(takers_web)]
    final_equal = bool(((rest["NPG/90_b"] == rest["NPG/90_a"]) & (rest["Assists/90_b"] == rest["Assists/90_a"])).all())
    t = f[f["Player"].isin(takers_web)].copy()
    t["NPG_change_pct"] = 100 * (t["NPG/90_a"] / t["NPG/90_b"] - 1)
    t["Ast_change_pct"] = 100 * (t["Assists/90_a"] / t["Assists/90_b"] - 1)
    out = {"takers_fpl_era": len(takers), "takers_rated_now": int(len(t)),
           "identity_non_takers_prematch_rows": rows_equal, "non_taker_rows_compared": int(len(other)),
           "identity_non_takers_final_ratings": final_equal, "non_takers_compared": int(len(rest)),
           "nan_ratings": int(ra[["NPG/90", "Assists/90"]].isna().any(axis=1).sum()),
           "takers_NPG_fell": int((t["NPG_change_pct"] < 0).sum()),
           "takers_NPG_change_pct_median": float(t["NPG_change_pct"].median()),
           "takers": t.sort_values("NPG_change_pct")[["Player", "Team", "NPG/90_b", "NPG/90_a", "NPG_change_pct",
                                                      "Assists/90_b", "Assists/90_a"]].round(4).to_dict("records")}
    (DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "takers"}, indent=2, ensure_ascii=False))
    print(t.sort_values("NPG_change_pct")[["Player", "Team", "NPG/90_b", "NPG/90_a", "NPG_change_pct"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
