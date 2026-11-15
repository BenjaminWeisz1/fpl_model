"""
Out-of-time test of refitted new-player priors (CHANGELOG §119).

    python validate_prior_refit.py arm A    # §116 model refit on the current log -> preds_A.npz
    python validate_prior_refit.py arm B    # + coefficient / scoring-level anchoring -> preds_B.npz
    python validate_prior_refit.py arm C    # B with the scoring-level exponent fixed at 1 (§231) -> preds_C.npz
    python validate_prior_refit.py score    # per-group comparison with the production incumbent -> results.json
                                            # (arms scored: PRIOR_REFIT_ARMS, default "A,B")
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import new_player_priors as npp  # noqa: E402
import fotmob_priors as fp  # noqa: E402
import anchored_priors as ap  # noqa: E402
import validate_fotmob_priors as vfp  # noqa: E402
from validate_new_player_priors import boot_change, calib  # noqa: E402

OUT = Path(__import__("os").environ.get("PRIOR_REFIT_OUT", npp.ROOT / "data" / "validation" / "prior_refit_20261029"))
if __import__("os").environ.get("PRIOR_REFIT_LOG"):   # e.g. a log on a later loop scale (§127)
    vfp.LOG = Path(__import__("os").environ["PRIOR_REFIT_LOG"])
P116 = npp.ROOT / "data" / "validation" / "fotmob_priors_20261029" / "test_predictions.csv"
TEST_SEASONS = range(2019, 2027)
GROUPS = ["top3", "ligue1", "championship", "other_europe", "rest_of_world"]


def dataset():
    arr, slots, _, s_xag, fm = vfp.dataset()
    # incumbent: Championship arrivals on the §116 FotMob path from 2023 use the §116 out-of-time predictions
    p = pd.read_csv(P116, encoding="utf-8-sig")
    p = p[(p["Path"] == "fotmob") & (p["Group"] == "championship")].set_index(["Player", "Season"])
    for t in npp.TARGETS:
        key = list(zip(arr["Player"], arr["Season"]))
        new = pd.Series(key).map(lambda k: p[f"cand_{t}"].get(k, np.nan)).to_numpy()
        arr[f"inc_{t}"] = np.where(np.isnan(new), arr[f"inc_{t}"], new)
    return arr, slots, s_xag, fm


def run_arm(name):
    OUT.mkdir(parents=True, exist_ok=True)
    arr, slots, s_xag, fm = dataset()
    if name == "A":
        fitter = lambda a, s, t: fp.fit_with_cv(a, s, t)  # noqa: E731
    else:
        cz, lv = ap.slot_covariates(sorted({L for L in slots["league"].ravel() if L}), ap.league_levels(fm),
                                    ap.load_coeffs())
        model = ap.CPriorModel if name == "C" else ap.BPriorModel
        fitter = lambda a, s, t: ap.fit_with_cv(a, s, t, cz, lv, model)  # noqa: E731
    vfp.TEST_SEASONS = TEST_SEASONS
    preds, fits = vfp.oot(arr, slots, fitter, s_xag)
    keys = np.array([f"{a}|{b}" for a, b in zip(arr["Player"], arr["Season"])])
    np.savez(OUT / f"preds_{name}.npz", keys=keys, **preds)   # keys: the scorer checks row alignment
    (OUT / f"fits_{name}.json").write_text(json.dumps(fits, indent=1, default=float), encoding="utf-8")


def score():
    arr, _, _, _ = dataset()
    arms = __import__("os").environ.get("PRIOR_REFIT_ARMS", "A,B").split(",")
    P = {k: dict(np.load(OUT / f"preds_{k}.npz")) for k in arms}
    keys = np.array([f"{a}|{b}" for a, b in zip(arr["Player"], arr["Season"])])
    for k in P:   # the data must not have changed between the arm runs and scoring
        assert (P[k].pop("keys") == keys).all(), f"arm {k} was fitted on a different dataset"
    tm = arr["Season"].between(min(TEST_SEASONS), max(TEST_SEASONS)).to_numpy()
    coef = ap.load_coeffs()
    res = {"test_arrivals": int(tm.sum()), "by_group": {}, "secondary": {}, "decision": {}}
    for g in GROUPS + ["all"]:
        m = tm & ((arr["Group"] == g).to_numpy() if g != "all" else True)
        s = arr[m]
        r = {"arrivals": int(m.sum()), "by_path": s["Path"].value_counts().to_dict()}
        for t in npp.TARGETS:
            inc = s[f"inc_{t}"].to_numpy()
            r[t] = {"calibration_inc": calib(s, inc, t)}
            for k in P:
                c = P[k][t][m]
                r[t][k] = {**boot_change(s, inc, c, t), "calibration": calib(s, c, t)}
            if "A" in P and "B" in P:
                r[t]["B_vs_A"] = boot_change(s, P["A"][t][m], P["B"][t][m], t)["change"]
            if "B" in P and "C" in P:
                r[t]["C_vs_B"] = boot_change(s, P["B"][t][m], P["C"][t][m], t)["change"]
        res["by_group"][g] = r
        if g == "all":
            continue
        ok = {k: all(r[t][k]["change"] < 0 and abs(r[t][k]["calibration"] - 1) < abs(r[t]["calibration_inc"] - 1)
                     for t in npp.TARGETS) for k in P}
        mean = {k: np.mean([r[t][k]["change"] for t in npp.TARGETS]) for k in P}
        passing = [k for k in P if ok[k]]
        res["decision"][g] = min(passing, key=mean.get) if passing else "incumbent"
    for name, mask in (("eredivisie", (arr["League"] == "Eredivisie").to_numpy()),
                       ("weak_leagues", (arr["League"].map(coef) < 79.5).to_numpy())):
        m = tm & mask
        s = arr[m]
        res["secondary"][name] = {"arrivals": int(m.sum()), **{t: {
            "calibration_inc": calib(s, s[f"inc_{t}"].to_numpy(), t),
            **{f"calibration_{k}": calib(s, P[k][t][m], t) for k in P},
            **{f"mse_{k}": boot_change(s, s[f"inc_{t}"].to_numpy(), P[k][t][m], t)["change"] for k in P}}
            for t in npp.TARGETS}}
    for S in TEST_SEASONS:
        m = (arr["Season"] == S).to_numpy()
        res["secondary"].setdefault("by_season", {})[S] = {
            t: {k: float(npp.sse(P[k][t][m], arr[m], t).sum() / npp.sse(arr.loc[m, f"inc_{t}"].to_numpy(), arr[m], t).sum() - 1)
                for k in P} for t in npp.TARGETS}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    out = arr.loc[tm, ["Player", "Season", "League", "Group", "Path", "inc_npxG", "inc_xAG"]].copy()
    for k in P:
        for t in npp.TARGETS:
            out[f"{k}_{t}"] = P[k][t][tm]
    out.to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    for g, r in res["by_group"].items():
        print(g, r["arrivals"], {t: (round(r[t]["calibration_inc"], 3),
                                     {k: (round(r[t][k]["change"], 4), round(r[t][k]["calibration"], 3)) for k in P})
                                 for t in npp.TARGETS})
    print("decision", res["decision"])


if __name__ == "__main__":
    run_arm(sys.argv[2]) if sys.argv[1] == "arm" else score()
