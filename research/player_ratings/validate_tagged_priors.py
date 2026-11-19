"""
Tagged arrivals (mid-season, two leagues) in the adopted groups with the production fitted prior (CHANGELOG §130).

    python -m research.player_ratings.validate_tagged_priors
"""
import json

import numpy as np
import pandas as pd

from fpl_model import paths
from fpl_model.player_ratings import fotmob_priors as fp  # noqa: E402
from fpl_model.player_ratings import new_player_priors as npp  # noqa: E402
from fpl_model.player_ratings.build_new_player_priors import (  # noqa: E402
    PARAMS,
    PARAMS_FOTMOB,
    era_scales,
    name_match_ids,
)
from research.player_ratings.validate_new_player_priors import boot_change, calib  # noqa: E402

LOG = npp.ROOT / "data" / "validation" / "prior_refit_beta_20261030" / "log_beta.csv"
OUT = npp.ROOT / "data" / "validation" / "tagged_priors_20261030"
LEAGUES = ["Championship", "Ligue 1"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(paths.current_file("transfers.csv", root=npp.DATA), encoding="utf-8-sig")
    for c in ["Min_Old", "npxG_Old", "xAG_Old", "NPG_Old", "Assists_Old", "Team_Attack_Rating"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    arr = t[t["Notes"].notna() & ~t["Notes"].isin(["youth", "second season", "missed season"]) & (t["Pos"] != "GK")
            & (t["Season"] > 2016) & t["League"].isin(LEAGUES)].copy()
    pms = pd.read_csv(paths.current_file("player_match_stats.csv", root=npp.DATA), encoding="utf-8-sig", low_memory=False)
    log = pd.read_csv(LOG, encoding="utf-8-sig", low_memory=False)
    arr = npp.attach_outcomes(arr, pms, log)
    uplayers, _ = npp.load_understat()
    ids = npp.understat_ids()
    arr, slots = npp.attach_priors(arr, uplayers, {**ids, **name_match_ids(arr, uplayers, ids)})
    arr["Age"] = 0.0
    arr = npp.incumbent_from_log(arr, log)
    keep = (arr["New"] & (arr["all_npxG_N"] > 0) & arr["inc_npxG"].notna()).to_numpy()
    arr, slots = npp.subset(arr, slots, keep)
    p103 = json.loads(PARAMS.read_text(encoding="utf-8"))
    pf = json.loads(PARAMS_FOTMOB.read_text(encoding="utf-8"))
    farr, fslots = fp.attach_fotmob(arr, slots, fp.load_fotmob())
    use_f = ((farr["Path"] == "fotmob") & (farr["League"] == "Championship")).to_numpy()
    scales = era_scales(sorted(arr["Season"].unique()))
    res = {"arrivals": int(len(arr)), "by_league": arr["League"].value_counts().to_dict(),
           "fotmob_path": int(use_f.sum()), "understat_path": int((arr["Path"] == "understat").sum())}
    cand = {}
    for t_ in npp.TARGETS:
        v = np.where(use_f, fp.FPriorModel.predict_with(pf[t_], farr, fslots, t_),
                     npp.PriorModel.predict_with(p103[t_], arr, slots, t_))
        if t_ in p103["era_scale_targets"]:
            v = v * arr["Season"].map(lambda S: scales[S][t_]).to_numpy()
        cand[t_] = v
    for name in ["all"] + LEAGUES:
        m = np.ones(len(arr), bool) if name == "all" else (arr["League"] == name).to_numpy()
        s = arr[m]
        res[name] = {t_: {**{k: v for k, v in boot_change(s, s[f"inc_{t_}"].to_numpy(), cand[t_][m], t_).items()
                             if k in ("change", "ci95", "players")},
                          "calibration_inc": calib(s, s[f"inc_{t_}"].to_numpy(), t_),
                          "calibration_cand": calib(s, cand[t_][m], t_)} for t_ in npp.TARGETS}
        if name != "all":
            r = res[name]
            res[name]["adopt"] = bool(all(r[t_]["change"] < 0 and abs(r[t_]["calibration_cand"] - 1)
                                          < abs(r[t_]["calibration_inc"] - 1) for t_ in npp.TARGETS))
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
