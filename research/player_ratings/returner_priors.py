"""
New-league priors for returning players (`second season` rows) from the production fitted models (CHANGELOG §128).

For a Championship returner: the FotMob-path model if FotMob matches him in his last season, else the §103
manual-path model on his transfers.csv values; x the §104 xAG era scale from 2025. Frozen parameters only.

    python -m research.player_ratings.returner_priors <out.csv>
"""
import json
import sys

import numpy as np
import pandas as pd

from fpl_model.player_ratings import fotmob_priors as fp  # noqa: E402
from fpl_model.player_ratings import new_player_priors as npp  # noqa: E402
from fpl_model.player_ratings.build_new_player_priors import PARAMS, PARAMS_FOTMOB, era_scales  # noqa: E402

LEAGUES = {"Championship"}


def returners():
    t = pd.read_csv(npp.DATA / "transfers.csv", encoding="utf-8-sig")
    for c in ["Min_Old", "npxG_Old", "xAG_Old", "NPG_Old", "Assists_Old", "Team_Attack_Rating"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    return t[(t["Notes"] == "second season") & t["League"].isin(LEAGUES) & (t["Pos"] != "GK")].copy()


def build():
    arr = returners()
    uplayers, _ = npp.load_understat()
    arr, slots = npp.attach_priors(arr, uplayers, {})
    arr["Age"] = 0.0
    p103 = json.loads(PARAMS.read_text(encoding="utf-8"))
    pf = json.loads(PARAMS_FOTMOB.read_text(encoding="utf-8"))
    farr, fslots = fp.attach_fotmob(arr, slots, fp.load_fotmob())
    use_f = (farr["Path"] == "fotmob").to_numpy()
    scales = era_scales(sorted(arr["Season"].unique()))
    out = arr[["Player", "Season", "League"]].copy()
    out["Path"] = np.where(use_f, "fotmob", "manual")
    for t, col in [("npxG", "Initial_NPG_Rating"), ("xAG", "Initial_Assist_Rating")]:
        v = np.where(use_f, fp.FPriorModel.predict_with(pf[t], farr, fslots, t),
                     npp.PriorModel.predict_with(p103[t], arr, slots, t))
        if t in p103["era_scale_targets"]:
            v = v * arr["Season"].map(lambda S: scales[S][t]).to_numpy()
        out[col] = v
    assert out[["Initial_NPG_Rating", "Initial_Assist_Rating"]].notna().all().all()
    return out


if __name__ == "__main__":
    o = build()
    o.to_csv(sys.argv[1], index=False, encoding="utf-8-sig")
    print(f"{len(o)} returner priors ({(o['Path'] == 'fotmob').sum()} FotMob path) -> {sys.argv[1]}")
