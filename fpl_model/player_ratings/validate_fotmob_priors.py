"""
Out-of-time test of the FotMob path for new-player priors (CHANGELOG §116).

    python validate_fotmob_priors.py coverage   # matching and season counts only (no fitting, no outcomes scored)
    python validate_fotmob_priors.py test       # rolling S = 2023..2026, arms, decision -> results.json

Incumbent = each arrival's first logged rating in the §112 K = 16 log (production rules, §104 out-of-time priors),
with the §104 FPL-era xAG scale for the adopted §103 groups. Static loop-scale scoring as §103.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.player_ratings import new_player_priors as npp  # noqa: E402
from fpl_model.player_ratings import fotmob_priors as fp  # noqa: E402
from fpl_model.player_ratings.build_new_player_priors import era_scales  # noqa: E402
from fpl_model.player_ratings.validate_new_player_priors import boot_change, calib  # noqa: E402

LOG = npp.ROOT / "data" / "validation" / "prior_k_20261029" / "log_k16.csv"
OUT = npp.ROOT / "data" / "validation" / "fotmob_priors_20261029"
TEST_SEASONS = range(2023, 2027)
GROUPS = ["championship", "other_europe", "rest_of_world"]
ADOPTED_103 = {"championship", "ligue1"}


def dataset():
    arr, slots = npp.build_dataset(LOG)
    keep = (arr["New"] & (arr["all_npxG_N"] > 0)).to_numpy()
    arr, slots = npp.subset(arr, slots, keep)
    assert arr["inc_npxG"].notna().all() and arr["inc_xAG"].notna().all()
    fm = fp.load_fotmob()
    farr, fslots = fp.attach_fotmob(arr, slots, fm)
    scales = era_scales(sorted(arr["Season"].unique()))
    s_xag = farr["Season"].map(lambda S: scales[S]["xAG"]).to_numpy()
    farr["inc_xAG"] = np.where(farr["Group"].isin(ADOPTED_103), farr["inc_xAG"] * s_xag, farr["inc_xAG"])
    return farr, fslots, slots, s_xag, fm


def coverage():
    OUT.mkdir(parents=True, exist_ok=True)
    arr, _, _, _, fm = dataset()
    a = arr[arr["Season"] >= 2021]
    in22 = a["League"].isin(fp.LEAGUES)
    res = {"fotmob_rows": int(len(fm)), "league_seasons": int(fm.groupby(["League", "Season"]).ngroups),
           "league_seasons_without_xG": sorted(f"{l} {s}" for (l, s), g in fm.groupby(["League", "Season"])
                                               if not g["Has_xG"].all()),
           "arrivals_2021plus_in_22_leagues": int(in22.sum()),
           "matched_by_season": a[in22].groupby("Season")["FM_ID"].apply(lambda s: f"{s.notna().sum()}/{len(s)}").to_dict(),
           "unmatched_2023plus": a[in22 & a["FM_ID"].isna() & (a["Season"] >= 2023)][["Player", "Season", "League"]]
           .to_dict("records"),
           "fm_seasons_found": a[a["Path"] == "fotmob"]["FM_Seasons"].value_counts().sort_index().to_dict()}
    (OUT / "coverage.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))


def oot(arr, slots, fitter, s_xag):
    preds = {t: np.full(len(arr), np.nan) for t in npp.TARGETS}
    fits = {}
    for S in TEST_SEASONS:
        tr, trs = npp.subset(arr, slots, (arr["Season"] < S).to_numpy())
        te_mask = (arr["Season"] == S).to_numpy()
        te, tes = npp.subset(arr, slots, te_mask)
        for t in npp.TARGETS:
            mdl = fitter(tr, trs, t)
            p = mdl.predict(te, tes)
            preds[t][te_mask] = p * (s_xag[te_mask] if t == "xAG" else 1.0)
            fits[f"{S}_{t}"] = {**mdl.summary(), "lambda": mdl.lam_cv, "converged": mdl.converged}
        print(f"  S={S}: train {len(tr)}, test {len(te)}", flush=True)
    return preds, fits


def test():
    OUT.mkdir(parents=True, exist_ok=True)
    arr, fslots, mslots, s_xag, _ = dataset()
    print("candidate (FotMob path)", flush=True)
    cand, fits = oot(arr, fslots, lambda a, s, t: fp.fit_with_cv(a, s, t), s_xag)
    print("no club strength (beta_F = 0)", flush=True)
    ns, _ = oot(arr, fslots, lambda a, s, t: fp.fit_with_cv(a, s, t, use_strength_f=False), s_xag)
    print("§103 model on manual values (no FotMob path)", flush=True)
    man, _ = oot(arr, mslots, lambda a, s, t: npp.fit_with_cv(a, s, t), s_xag)

    tm = arr["Season"].between(min(TEST_SEASONS), max(TEST_SEASONS)).to_numpy()
    fpath = tm & (arr["Path"] == "fotmob").to_numpy()
    coef = pd.read_csv(npp.DATA / "league_coefficients.csv", encoding="utf-8-sig").set_index("League")["Coefficient"]
    res = {"test_arrivals": int(tm.sum()), "fotmob_path_test": int(fpath.sum()),
           "fotmob_path_by_group": arr[fpath].groupby("Group").size().to_dict(),
           "fits": fits, "by_group": {}, "secondary": {}}
    for g in GROUPS + ["all_fotmob"]:
        m = fpath & ((arr["Group"] == g).to_numpy() if g != "all_fotmob" else True)
        if not m.sum():
            continue
        s = arr[m]
        r = {"arrivals": int(m.sum())}
        for t in npp.TARGETS:
            inc, c = s[f"inc_{t}"].to_numpy(), cand[t][m]
            r[t] = boot_change(s, inc, c, t)
            r[t]["calibration_inc"] = calib(s, inc, t)
            r[t]["calibration_cand"] = calib(s, c, t)
            r[t]["strength_effect"] = boot_change(s, ns[t][m], c, t)["change"]
            r[t]["calibration_no_strength"] = calib(s, ns[t][m], t)
            r[t]["data_effect_vs_manual_103"] = boot_change(s, man[t][m], c, t)["change"]
            r[t]["calibration_manual_103"] = calib(s, man[t][m], t)
        res["by_group"][g] = r
    # (c) weak leagues and the Eredivisie, FotMob-path arrivals
    weak = fpath & (arr["League"].map(coef) < 79.5).to_numpy()
    ered = fpath & (arr["League"] == "Eredivisie").to_numpy()
    for name, m in (("weak_leagues", weak), ("eredivisie", ered)):
        if m.sum():
            s = arr[m]
            res["secondary"][name] = {"arrivals": int(m.sum()), **{
                t: {"calibration_inc": calib(s, s[f"inc_{t}"].to_numpy(), t),
                    "calibration_cand": calib(s, cand[t][m], t),
                    "mse_change": boot_change(s, s[f"inc_{t}"].to_numpy(), cand[t][m], t)["change"]}
                for t in npp.TARGETS}}
    res["secondary"]["fm_seasons_found"] = arr[fpath]["FM_Seasons"].value_counts().sort_index().to_dict()
    dec = {}
    for g in GROUPS:
        r = res["by_group"].get(g)
        if r is None:
            continue
        ok = all(r[t]["change"] < 0 and abs(r[t]["calibration_cand"] - 1) < abs(r[t]["calibration_inc"] - 1)
                 for t in npp.TARGETS)
        dec[g] = "adopt" if ok else "keep incumbent"
    res["decision"] = dec
    out = arr.loc[tm, ["Player", "Season", "League", "Group", "Path", "FM_ID", "FM_Seasons", "inc_npxG", "inc_xAG"]].copy()
    for t in npp.TARGETS:
        out[f"cand_{t}"], out[f"ns_{t}"], out[f"man_{t}"] = cand[t][tm], ns[t][tm], man[t][tm]
    out.to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for g, r in res["by_group"].items():
        print(g, r["arrivals"], {t: (round(r[t]["change"], 4), r[t]["ci95"], round(r[t]["calibration_inc"], 3),
                                     round(r[t]["calibration_cand"], 3)) for t in npp.TARGETS})
    print("decision", dec)


if __name__ == "__main__":
    {"coverage": coverage, "test": test}[sys.argv[1]]()
