"""
Out-of-time test of the new-player goal/assist priors (CHANGELOG §103).

    python -m research.player_ratings.validate_new_player_priors diagnostic   # step 2: the incumbent's calibration by league group
    python -m research.player_ratings.validate_new_player_priors test         # step 7: rolling out-of-time test, both targets

Population: `transfers.csv` arrivals with blank Notes, a non-PL League and a non-GK position whose first
PL match is in their arrival season (players returning after an earlier PL spell keep their old rating in
the loop, so their transfers.csv row is never used). Incumbent = each arrival's first pre-match rating in
the baseline log. Each arm predicts match npxG (xAG) = prior x minutes / 90, no updating.
"""
import json
import sys

import numpy as np
import pandas as pd

from fpl_model.player_ratings import new_player_priors as npp  # noqa: E402

DIR = npp.ROOT / "data" / "validation" / "new_player_priors_20261028"
LOG = DIR / "baseline_log.csv"
GROUPS = ["top3", "ligue1", "championship", "other_europe", "rest_of_world"]
N_BOOT, SEED = 5000, 42
TEST_SEASONS = range(2019, 2027)


def dataset():
    arr, slots = npp.build_dataset(LOG)
    keep = (arr["New"] & (arr["all_npxG_N"] > 0)).to_numpy()
    print(f"in scope: {len(arr)}; new with PL minutes in the arrival season: {keep.sum()}; "
          f"earlier PL spell: {int((arr['First_PL_Season'] < arr['Season']).sum())}; "
          f"never played: {int(arr['First_PL_Season'].isna().sum())}; "
          f"first PL season later than the row: {int((arr['First_PL_Season'] > arr['Season']).sum())}")
    arr, slots = npp.subset(arr, slots, keep)
    assert arr["inc_npxG"].notna().all() and arr["inc_xAG"].notna().all(), "arrival without a logged prior"
    return arr, slots


def calib(arr, pred, t, tag="all"):
    return float((pred * arr[f"{tag}_{t}_M"]).sum() / arr[f"{tag}_{t}_Y"].sum())


def diagnostic():
    arr, _ = dataset()
    a = arr[arr["Season"] >= 2018]
    rows = []
    for g in GROUPS + ["all"]:
        s = a if g == "all" else a[a["Group"] == g]
        rows.append({"group": g, "arrivals": len(s), "match_90s": round(float(s["all_npxG_M"].sum()), 1),
                     "npxG_calibration": round(calib(s, s["inc_npxG"], "npxG"), 3),
                     "xAG_calibration": round(calib(s, s["inc_xAG"], "xAG"), 3),
                     "npxG_calibration_plain": round(calib(s, s["inc_npxG"], "npxG", "plain"), 3),
                     "xAG_calibration_plain": round(calib(s, s["inc_xAG"], "xAG", "plain"), 3)})
    by_league = []
    for L, s in a.groupby("League"):
        if len(s) >= 15:
            by_league.append({"league": L, "arrivals": len(s),
                              "npxG_calibration": round(calib(s, s["inc_npxG"], "npxG"), 3),
                              "xAG_calibration": round(calib(s, s["inc_xAG"], "xAG"), 3)})
    out = {"by_group": rows, "by_league_15plus": by_league}
    (DIR / "diagnostic.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(by_league).to_string(index=False))


def boot_change(arr, base, cand, t, tag="all"):
    """MSE change (cand vs base) with a player-cluster bootstrap; also returns the point MSEs."""
    eb, ec, n = npp.sse(base, arr, t, tag), npp.sse(cand, arr, t, tag), arr[f"{tag}_{t}_N"].to_numpy()
    point = ec.sum() / eb.sum() - 1
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(arr), size=(N_BOOT, len(arr)))
    ch = ec[idx].sum(1) / eb[idx].sum(1) - 1
    return {"mse_base": float(eb.sum() / n.sum()), "mse_cand": float(ec.sum() / n.sum()),
            "change": float(point), "ci95": [float(np.percentile(ch, 2.5)), float(np.percentile(ch, 97.5))],
            "p_worse": float((ch > 0).mean()), "players": int(len(arr)), "rows": int(n.sum())}


def poisson_dev(arr, pred, t):
    """Per-player approximation is not possible from sufficient statistics; report the season-total
    deviance instead: 2 sum[y log(y/mu) - (y - mu)] on each arrival's first-season total."""
    y, mu = arr[f"all_{t}_Y"].to_numpy(), pred * arr[f"all_{t}_M"].to_numpy()
    mu = np.maximum(mu, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * (term - (y - mu)).sum())


def oot_predictions(arr, slots, use_strength=True, n_seasons=None, use_age=False):
    """Rolling out-of-time candidate predictions for S in TEST_SEASONS (fit on arrivals before S)."""
    preds = {t: np.full(len(arr), np.nan) for t in npp.TARGETS}
    fits = {}
    for S in TEST_SEASONS:
        tr_mask = (arr["Season"] < S).to_numpy()
        te_mask = (arr["Season"] == S).to_numpy()
        tr, trs = npp.subset(arr, slots, tr_mask)
        te, tes = npp.subset(arr, slots, te_mask)
        if n_seasons is not None:
            for d in (trs, tes):
                d["m"][:, n_seasons:] = 0
                for v in d["x"].values():
                    v[:, n_seasons:] = 0
        for t in npp.TARGETS:
            mdl = npp.fit_with_cv(tr, trs, t, use_strength, use_age)
            preds[t][te_mask] = mdl.predict(te, tes)
            fits[f"{S}_{t}"] = {**mdl.summary(), "lambda": mdl.lam_cv, "converged": mdl.converged}
        print(f"  S={S}: train {tr_mask.sum()}, test {te_mask.sum()}", flush=True)
    return preds, fits


def test():
    arr, slots = dataset()
    print(arr.groupby(["Group", "Path"]).size().to_string())
    preds, fits = oot_predictions(arr, slots)
    print("no team strength:", flush=True)
    preds_ns, _ = oot_predictions(arr, slots, use_strength=False)
    print("with age:", flush=True)
    preds_age, fits_age = oot_predictions(arr, slots, use_age=True)
    t_mask = arr["Season"].between(min(TEST_SEASONS), max(TEST_SEASONS)).to_numpy()
    a = arr[t_mask].reset_index(drop=True)
    res = {"population": {"test_arrivals": int(t_mask.sum()),
                          "by_group_path": {f"{g}/{p}": int(n) for (g, p), n in a.groupby(["Group", "Path"]).size().items()}},
           "fits": fits, "primary": {}, "first10": {}, "by_group": {}, "strength_effect": {},
           "poisson_deviance": {}, "by_season": {}, "age_effect": {}, "plain_scale": {}, "fits_age": fits_age}
    out_rows = []
    for t in npp.TARGETS:
        inc = a[f"inc_{t}"].to_numpy()
        cand = preds[t][t_mask]
        ns = preds_ns[t][t_mask]
        res["primary"][t] = boot_change(a, inc, cand, t)
        res["first10"][t] = boot_change(a, inc, cand, t, "f10")
        res["strength_effect"][t] = boot_change(a, ns, cand, t)
        age = preds_age[t][t_mask]
        res["age_effect"][t] = boot_change(a, cand, age, t)
        res["plain_scale"][t] = boot_change(a, inc, cand, t, "plain")
        res["poisson_deviance"][t] = {"incumbent": poisson_dev(a, inc, t), "candidate": poisson_dev(a, cand, t)}
        for g in GROUPS:
            m = (a["Group"] == g).to_numpy()
            s = a[m]
            if not len(s):
                continue
            r = boot_change(s, inc[m], cand[m], t)
            r["calibration_inc"] = calib(s, inc[m], t)
            r["calibration_cand"] = calib(s, cand[m], t)
            r["calibration_cand_no_strength"] = calib(s, ns[m], t)
            r["first10"] = boot_change(s, inc[m], cand[m], t, "f10")
            res["by_group"].setdefault(g, {})[t] = r
        for S in TEST_SEASONS:
            m = (a["Season"] == S).to_numpy()
            eb, ec = npp.sse(inc[m], a[m], t).sum(), npp.sse(cand[m], a[m], t).sum()
            res["by_season"].setdefault(str(S), {})[t] = float(ec / eb - 1)
        a[f"cand_{t}"] = cand
        a[f"cand_ns_{t}"] = ns
        a[f"cand_age_{t}"] = age
    # decision per group (pre-registered rule, CHANGELOG §103)
    dec = {}
    for g, r in res["by_group"].items():
        ok = all(r[t]["change"] < 0 and abs(r[t]["calibration_cand"] - 1) < abs(r[t]["calibration_inc"] - 1)
                 for t in npp.TARGETS)
        dec[g] = "adopt" if ok else "keep incumbent"
    res["decision"] = dec
    (DIR / "test_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    keep = ["Player", "Season", "League", "Group", "Path", "Pos", "inc_npxG", "cand_npxG", "cand_ns_npxG",
            "inc_xAG", "cand_xAG", "cand_ns_xAG", "cand_age_npxG", "cand_age_xAG", "Age", "all_npxG_Y", "all_xAG_Y", "all_npxG_M", "all_npxG_N"]
    a[keep].to_csv(DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    for t in npp.TARGETS:
        print(t, "primary", {k: res["primary"][t][k] for k in ["change", "ci95", "p_worse"]})
    for t in npp.TARGETS:
        print(t, "strength", res["strength_effect"][t]["change"], "age", res["age_effect"][t]["change"])
    for g, r in res["by_group"].items():
        print(g, {t: (round(r[t]["change"], 4), round(r[t]["calibration_inc"], 3), round(r[t]["calibration_cand"], 3))
                  for t in npp.TARGETS}, dec[g])


if __name__ == "__main__":
    {"diagnostic": diagnostic, "test": test}[sys.argv[1]]()
