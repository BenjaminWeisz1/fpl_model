"""
FPL-era scale correction for the new-player priors (CHANGELOG §104).

Reuses the §103 out-of-time candidate predictions (test_predictions.csv) and the same arrivals dataset.
    python -m research.player_ratings.validate_new_player_priors_era
"""
import json

import numpy as np
import pandas as pd

from fpl_model.player_ratings import new_player_priors as npp  # noqa: E402
from research.player_ratings import validate_new_player_priors as v  # noqa: E402

ADOPTED = ["championship", "ligue1"]


def era_scale(before_season, exclude=None):
    """s_t = FPL-era per-90 (seasons 2025 .. before_season-1, or before_season itself when exclude is given)
    / 2017-2024 per-90, minutes-weighted over all PL rows with minutes > 0."""
    p = pd.read_csv(npp.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    p = p[p["Minutes"] > 0]
    old = p[p["Season"] <= 2024]
    if exclude is None:
        new = p[(p["Season"] >= 2025) & (p["Season"] < before_season)]
    else:
        new = p[(p["Season"] == before_season) & ~p["Player"].isin(exclude)]
    return {t: float((new[c].sum() / new["Minutes"].sum()) / (old[c].sum() / old["Minutes"].sum()))
            for t, c in [("npxG", "npxG"), ("xAG", "xAG")]}


def main():
    arr, _ = v.dataset()
    pred = pd.read_csv(v.DIR / "test_predictions.csv", encoding="utf-8-sig")
    a = arr.merge(pred[["Player", "Season", "cand_npxG", "cand_xAG"]], on=["Player", "Season"], how="inner")
    assert len(a) == len(pred)
    s26 = era_scale(2026)
    s25 = era_scale(2025, exclude=set(a.loc[a["Season"] == 2025, "Player"]))
    res = {"s_2026_from_2025": s26, "s_2025_contemporaneous": s25, "primary_2026": {}, "adopted_2026": {},
           "secondary_2025": {}, "fpl_era_calibration": {}, "decision": {}}
    for t in npp.TARGETS:
        scale = np.where(a["Season"] == 2026, s26[t], np.where(a["Season"] == 2025, s25[t], 1.0))
        a[f"era_{t}"] = a[f"cand_{t}"] * scale
        m26 = (a["Season"] == 2026).to_numpy()
        x = a[m26].reset_index(drop=True)
        res["primary_2026"][t] = v.boot_change(x, x[f"cand_{t}"].to_numpy(), x[f"era_{t}"].to_numpy(), t)
        ad = x[x["Group"].isin(ADOPTED)].reset_index(drop=True)
        res["adopted_2026"][t] = v.boot_change(ad, ad[f"cand_{t}"].to_numpy(), ad[f"era_{t}"].to_numpy(), t)
        y = a[(a["Season"] == 2025).to_numpy()].reset_index(drop=True)
        res["secondary_2025"][t] = v.boot_change(y, y[f"cand_{t}"].to_numpy(), y[f"era_{t}"].to_numpy(), t)
        fe = a[a["Season"] >= 2025]
        cal = {}
        for g in v.GROUPS + ["adopted"]:
            s = fe[fe["Group"].isin(ADOPTED)] if g == "adopted" else fe[fe["Group"] == g]
            cal[g] = {arm: v.calib(s, s[col].to_numpy(), t) for arm, col in
                      [("incumbent", f"inc_{t}"), ("candidate", f"cand_{t}"), ("candidate_x_s", f"era_{t}")]}
        res["fpl_era_calibration"][t] = cal
        c = cal["adopted"]
        res["decision"][t] = bool(res["primary_2026"][t]["change"] < 0
                                  and abs(c["candidate_x_s"] - 1) < abs(c["candidate"] - 1))
    (v.DIR / "era_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ["s_2026_from_2025", "s_2025_contemporaneous", "decision"]}, indent=1))
    for t in npp.TARGETS:
        print(t, "2026", {k: res["primary_2026"][t][k] for k in ["change", "ci95", "players"]},
              "adopted26", round(res["adopted_2026"][t]["change"], 4), "2025", round(res["secondary_2025"][t]["change"], 4))
        print({g: {k: round(x, 3) for k, x in d.items()} for g, d in res["fpl_era_calibration"][t].items()})


if __name__ == "__main__":
    main()
