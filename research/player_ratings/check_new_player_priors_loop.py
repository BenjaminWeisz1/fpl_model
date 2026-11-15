"""
Shadow checks of the new-player priors in the rating loop (CHANGELOG §103-§105).

    python check_new_player_priors_loop.py

1. Switch off (GOAL_MODEL_NEW_PRIORS=0) reproduces the baseline log and ratings exactly.
2. Switch on: players without an adopted prior row have identical pre-match rows and final ratings; no NaN.
3. Secondary (pre-registered in §103): the loop with each adopted arrival's out-of-time prior (oot_log.csv)
   against the baseline, MSE of match npxG/xAG on the loop's own predictions, for adopted arrivals'
   first-season rows and for all their rows.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.player_ratings import validate_new_player_priors as v  # noqa: E402

D = v.DIR
KEY = ["Player", "Team", "Opponent", "Venue", "MatchID", "Season"]


def load(name):
    return pd.read_csv(D / name, encoding="utf-8-sig")


def boot(err_b, err_c, players):
    rng = np.random.default_rng(42)
    codes, uniq = pd.factorize(players)
    sb = np.bincount(codes, err_b); sc = np.bincount(codes, err_c)
    idx = rng.integers(0, len(uniq), size=(5000, len(uniq)))
    ch = sc[idx].sum(1) / sb[idx].sum(1) - 1
    return {"change": float(sc.sum() / sb.sum() - 1), "ci95": [float(np.percentile(ch, 2.5)), float(np.percentile(ch, 97.5))],
            "rows": int(len(err_b)), "players": int(len(uniq))}


def main():
    base, off, on, oot = load("baseline_log.csv"), load("switch_off_log.csv"), load("switch_on_log.csv"), load("oot_log.csv")
    res = {}
    res["switch_off_identical_log"] = bool(base.equals(off))
    rb, ro = load("baseline_log.csv.ratings.csv"), load("switch_off_log.csv.ratings.csv")
    res["switch_off_identical_ratings"] = bool(rb.sort_values("Player").reset_index(drop=True).equals(
        ro.sort_values("Player").reset_index(drop=True)))
    pri = pd.read_csv(v.npp.DATA / "new_player_priors.csv", encoding="utf-8-sig")
    touched = set(pri.loc[pri["Adopted"], "Player"])
    for name, other in [("on", on), ("oot", oot)]:
        assert len(other) == len(base)
        same = base["Player"].isin(touched)
        if name == "oot":
            ot = pd.read_csv(D / "oot_priors.csv", encoding="utf-8-sig")
            same = base["Player"].isin(set(ot.loc[ot["Adopted"], "Player"]))
        cols = ["NPG_Rating_Pre", "Assist_Rating_Pre"]
        res[f"{name}_untouched_rows_identical"] = bool(np.allclose(base.loc[~same, cols], other.loc[~same, cols], rtol=0, atol=0))
        res[f"{name}_nan"] = int(other[cols].isna().sum().sum())
        res[f"{name}_rows_changed"] = int((~np.isclose(base[cols], other[cols], rtol=0, atol=0)).any(axis=1).sum())
    ron = load("switch_on_log.csv.ratings.csv")
    res["on_ratings_nan"] = int(ron[["NPG/90", "Assists/90"]].isna().sum().sum())
    # secondary: loop effect of the out-of-time priors
    ot = pd.read_csv(D / "oot_priors.csv", encoding="utf-8-sig")
    ad = ot[ot["Adopted"]]
    first = dict(zip(ad["Player"], ad["Season"]))
    sel = base["Player"].isin(first)
    fs = sel & (base["Season"] == base["Player"].map(first))
    sec = {}
    for t, (act, pred) in {"npxG": ("npxg90", "pred_goals_90"), "xAG": ("xag90", "pred_ast_90")}.items():
        y = base[act] * base["weight"]
        eb = (base[pred] * base["weight"] - y) ** 2
        ec = (oot[pred] * oot["weight"] - y) ** 2
        sec[t] = {"first_season": boot(eb[fs].to_numpy(), ec[fs].to_numpy(), base.loc[fs, "Player"].to_numpy()),
                  "all_rows_of_adopted_arrivals": boot(eb[sel].to_numpy(), ec[sel].to_numpy(), base.loc[sel, "Player"].to_numpy()),
                  "all_rows": boot(eb.to_numpy(), ec.to_numpy(), base["Player"].to_numpy())}
    res["loop_effect_oot"] = sec
    (D / "loop_checks.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
