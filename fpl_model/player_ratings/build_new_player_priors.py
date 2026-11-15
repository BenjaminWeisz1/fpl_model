"""
Writes data/current/new_player_priors.csv: goal/assist priors for players new to the PL (CHANGELOG §103).

    python build_new_player_priors.py          # apply the frozen parameters (production)
    python build_new_player_priors.py --fit    # refit and freeze the parameters (one-off; needs a new
                                               # pre-registration, as the frozen bonus constants do)
    python build_new_player_priors.py --fit-fotmob   # refit and freeze the FotMob-path model (§116; same caveat)

Rows: every `transfers.csv` row with blank Notes, a non-PL League, a non-GK position and Season >= 2017.
An arrival from La Liga, Bundesliga, Serie A, Ligue 1 or the RFPL is automated: his last three seasons come
from the frozen Understat league files by Understat ID (the crosswalk, else a unique name match in his
League's previous season). Every other arrival uses his manual Min_Old / npxG_Old / xAG_Old with the fitted
league factor. `Adopted` marks the league groups that passed the §103 test; build_player_ratings.py uses
only those rows.

FotMob path (CHANGELOG §116): an arrival in a group adopted there (the Championship) whom FotMob's frozen league
lists (`data/raw/fotmob/`, `scripts/fetch/fetch_fotmob_leagues.py`) match by name in his League's previous season
takes the FotMob-path model instead (`new_player_prior_params_fotmob.json`): up to three FotMob seasons in any of
22 leagues.

The ratings include the fitted new-team adjustment (Team_Attack_Rating / 1.32023)^gamma, so
build_player_ratings.py must not apply its own weight_team factor to them again.
"""
import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.player_ratings import new_player_priors as npp  # noqa: E402
from fpl_model.player_ratings import fotmob_priors as fp  # noqa: E402

PARAMS = Path(__file__).resolve().parent / "new_player_prior_params.json"
OUT = npp.DATA / "new_player_priors.csv"
FIT_LOG = npp.ROOT / "data" / "validation" / "new_player_priors_20261028" / "baseline_log.csv"
FIT_LAST_SEASON = 2025
PARAMS_FOTMOB = Path(__file__).resolve().parent / "new_player_prior_params_fotmob.json"
# §116 was tested on the post-§112 loop scale; its fit uses that log
FIT_LOG_FOTMOB = npp.ROOT / "data" / "validation" / "prior_k_20261029" / "log_k16.csv"
FOTMOB_RESULTS = npp.ROOT / "data" / "validation" / "fotmob_priors_20261029" / "results.json"


def _norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s).split()


def name_match_ids(arr, uplayers, ids):
    """Understat IDs for arrivals missing from the crosswalk: a unique name match among the players of
    his League's Understat file in the season before arrival (all name tokens contained)."""
    found = {}
    for r in arr.itertuples():
        if r.Player in ids or r.League not in npp.UNDERSTAT_LEAGUE:
            continue
        pool = uplayers[(uplayers["ULeague"] == npp.UNDERSTAT_LEAGUE[r.League]) & (uplayers["Season"] == r.Season - 1)]
        tok = set(_norm(r.Player))
        names = list(zip(pool["Understat_ID"], pool["Name"]))
        # 1) every name token contained either way (min. two tokens on the shorter side);
        # 2) else a close spelling (difflib ratio >= 0.88), e.g. Emmanuel / Emanuel. Unique hits only.
        hits = {int(u) for u, n in names
                if tok and (tok <= set(_norm(n)) or (len(set(_norm(n))) >= 2 and set(_norm(n)) <= tok))}
        if not hits:
            full = " ".join(_norm(r.Player))
            hits = {int(u) for u, n in names
                    if difflib.SequenceMatcher(None, full, " ".join(_norm(n))).ratio() >= 0.88}
        if len(hits) == 1:
            found[r.Player] = hits.pop()
    return found


def dataset(with_outcomes, log_path=None):
    uplayers, _ = npp.load_understat()
    arr = npp.in_scope_transfers()
    if with_outcomes:
        pms = pd.read_csv(npp.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
        log = pd.read_csv(log_path or FIT_LOG, encoding="utf-8-sig")
        arr = npp.attach_outcomes(arr, pms, log)
    ids = npp.understat_ids()
    matched = name_match_ids(arr, uplayers, ids)
    arr, slots = npp.attach_priors(arr, uplayers, {**ids, **matched})
    arr["ID_Source"] = np.where(arr["Player"].isin(matched), "name match",
                                np.where(arr["Understat_ID"].notna(), "crosswalk", ""))
    arr = npp.attach_age(arr, npp.birth_dates())
    return arr, slots


def fit(last_season=FIT_LAST_SEASON, write=True):
    """Fit the §103 model on arrivals 2017-last_season. write=False returns the parameters without freezing them
    (build_oot_priors.py fits season by season this way)."""
    arr, slots = dataset(with_outcomes=True)
    keep = (arr["New"] & (arr["all_npxG_N"] > 0) & (arr["Season"] <= last_season)).to_numpy()
    arr, slots = npp.subset(arr, slots, keep)
    decision = json.loads((FIT_LOG.parent / "test_results.json").read_text(encoding="utf-8"))
    # pre-registered: age only if the age arm improved the pooled MSE for both targets
    use_age = all(decision["age_effect"][t]["change"] < 0 for t in npp.TARGETS)
    era = json.loads((FIT_LOG.parent / "era_results.json").read_text(encoding="utf-8"))
    out = {"fitted_on": f"arrivals 2017-{last_season} ({len(arr)})", "use_age": use_age,
           "adopted_groups": sorted(g for g, d in decision["decision"].items() if d == "adopt"),
           "era_scale_targets": sorted(t for t, ok in era["decision"].items() if ok)}
    for t in npp.TARGETS:
        mdl = npp.fit_with_cv(arr, slots, t, use_strength=True, use_age=use_age)
        p = mdl.p
        out[t] = {"d": p["d"], "beta": p["beta"], "b": p["b"], "M0": p["M0"], "gamma": p["gamma"],
                  "alpha": p["alpha"], "mu_U": p["mu_U"], "mu_M": p["mu_M"], "u": p["u"], "lambda": mdl.lam_cv}
    if write:
        PARAMS.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"froze parameters -> {PARAMS}")
    return out


def fit_fotmob(last_season=FIT_LAST_SEASON, write=True):
    """Freeze the §116 FotMob-path model, fitted on arrivals 2017-last_season (every path)."""
    arr, slots = dataset(with_outcomes=True, log_path=FIT_LOG_FOTMOB)
    keep = (arr["New"] & (arr["all_npxG_N"] > 0) & (arr["Season"] <= last_season)).to_numpy()
    arr, slots = npp.subset(arr, slots, keep)
    arr, slots = fp.attach_fotmob(arr, slots, fp.load_fotmob())
    decision = json.loads(FOTMOB_RESULTS.read_text(encoding="utf-8"))["decision"]
    base = json.loads(PARAMS.read_text(encoding="utf-8"))
    out = {"fitted_on": f"arrivals 2017-{last_season} ({len(arr)}; {int((arr['Path'] == 'fotmob').sum())} FotMob path)",
           "fit_log": str(FIT_LOG_FOTMOB.relative_to(npp.ROOT)),
           "adopted_groups": sorted(g for g, d in decision.items() if d == "adopt"),
           "era_scale_targets": base["era_scale_targets"]}
    for t in npp.TARGETS:
        mdl = fp.fit_with_cv(arr, slots, t)
        p = mdl.p
        out[t] = {k: p[k] for k in ("d", "beta", "b", "M0", "gamma", "alpha", "mu_U", "mu_M", "mu_F", "beta_F")}
        out[t].update({"u": p["u"], "lambda": mdl.lam_cv})
    if write:
        PARAMS_FOTMOB.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"froze FotMob-path parameters -> {PARAMS_FOTMOB}")
    return out


PARAMS_REFIT = Path(__file__).resolve().parent / "new_player_prior_params_refit.json"
FIT_LOG_REFIT = npp.ROOT / "data" / "validation" / "prior_refit_beta_20261030" / "log_beta.csv"
REFIT_RESULTS = npp.ROOT / "data" / "validation" / "prior_refit_beta_20261030" / "results.json"
# Groups adopted by user decision against the §119 rule (§139: top3 failed only on a +0.1% xAG MSE, noise)
USER_ADOPTED_REFIT = {"top3"}


def fit_refit(last_season=FIT_LAST_SEASON, write=True):
    """Freeze the §127 arm-A model (§116 structure refit on the β-dampened loop), arrivals 2017-last_season."""
    arr, slots = dataset(with_outcomes=True, log_path=FIT_LOG_REFIT)
    keep = (arr["New"] & (arr["all_npxG_N"] > 0) & (arr["Season"] <= last_season)).to_numpy()
    arr, slots = npp.subset(arr, slots, keep)
    arr, slots = fp.attach_fotmob(arr, slots, fp.load_fotmob())
    decision = json.loads(REFIT_RESULTS.read_text(encoding="utf-8"))["decision"]
    base = json.loads(PARAMS.read_text(encoding="utf-8"))
    out = {"fitted_on": f"arrivals 2017-{last_season} ({len(arr)})", "fit_log": str(FIT_LOG_REFIT.relative_to(npp.ROOT)),
           "adopted_groups": sorted({g for g, d in decision.items() if d == "A"} | USER_ADOPTED_REFIT),
           "era_scale_targets": base["era_scale_targets"]}
    for t in npp.TARGETS:
        mdl = fp.fit_with_cv(arr, slots, t)
        p = mdl.p
        out[t] = {k: p[k] for k in ("d", "beta", "b", "M0", "gamma", "alpha", "mu_U", "mu_M", "mu_F", "beta_F")}
        out[t].update({"u": p["u"], "lambda": mdl.lam_cv})
    if write:
        PARAMS_REFIT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"froze §127 refit parameters -> {PARAMS_REFIT} (groups {out['adopted_groups']})")
    return out


def era_scales(seasons):
    """FPL-era scale per season (CHANGELOG §104): PL per-90 in the FPL-era seasons before S (S itself for
    2025, the first) / PL per-90 in 2017-2024, minutes-weighted over every row with minutes > 0."""
    p = pd.read_csv(npp.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    p = p[p["Minutes"] > 0]
    old = p[p["Season"] <= 2024]
    out = {}
    for S in seasons:
        if S < 2025:
            out[S] = {"npxG": 1.0, "xAG": 1.0}
            continue
        new = p[(p["Season"] >= 2025) & (p["Season"] < max(S, 2026))]
        out[S] = {t: float((new[t].sum() / new["Minutes"].sum()) / (old[t].sum() / old["Minutes"].sum()))
                  for t in ("npxG", "xAG")}
    return out


def tagged_championship(params, scales, pf=None):
    """§130: Championship arrivals with a tag in Notes (mid-season, two leagues, ...) get the same fitted prior as a
    blank-Notes arrival: the FotMob path if FotMob matches them, else the §103 manual path."""
    t = pd.read_csv(npp.DATA / "transfers.csv", encoding="utf-8-sig")
    for c in ["Min_Old", "npxG_Old", "xAG_Old", "NPG_Old", "Assists_Old", "Team_Attack_Rating"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    arr = t[t["Notes"].notna() & ~t["Notes"].isin(["youth", "second season", "missed season"])
            & (t["Pos"] != "GK") & (t["Season"] > 2016) & (t["League"] == "Championship")].copy()
    uplayers, _ = npp.load_understat()
    arr, slots = npp.attach_priors(arr, uplayers, {})
    arr["Age"] = 0.0
    sc = era_scales(sorted(set(arr["Season"]) | set(scales)))
    out = arr[["Player", "Season", "Pos", "League", "Group", "Path", "Team_Attack_Rating"]].copy()
    use_f = np.zeros(len(arr), bool)
    if pf is None and PARAMS_FOTMOB.exists():
        pf = json.loads(PARAMS_FOTMOB.read_text(encoding="utf-8"))
    if pf is not None:
        farr, fslots = fp.attach_fotmob(arr, slots, fp.load_fotmob())
        use_f = (farr["Path"] == "fotmob").to_numpy()
    for t_, col in [("npxG", "Initial_NPG_Rating"), ("xAG", "Initial_Assist_Rating")]:
        v = npp.PriorModel.predict_with(params[t_], arr, slots, t_)
        if use_f.any():
            v = np.where(use_f, fp.FPriorModel.predict_with(pf[t_], farr, fslots, t_), v)
        if t_ in params["era_scale_targets"]:
            v = v * arr["Season"].map(lambda S: sc[S][t_]).to_numpy()
        out[col] = v
    out["Path"] = np.where(use_f, "fotmob", "manual")
    out["Adopted"] = True
    out["Understat_ID"] = np.nan
    out["ID_Source"] = "tagged"
    out["Understat_Seasons"] = 0
    out["Prior_Minutes_Used"] = np.where(use_f, fslots["m"].sum(1) if use_f.any() else 0, slots["m"].sum(1))
    out["Age"] = np.nan
    return out


def build():
    params = json.loads(PARAMS.read_text(encoding="utf-8"))
    out = priors_frame()
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    cur = out[out["Season"] == out["Season"].max()]
    print(f"Wrote {len(out)} priors to {OUT.name}; adopted groups {params['adopted_groups']}")
    print(f"Season {int(cur['Season'].max())}: {len(cur)} arrivals, {int((cur['Path'] == 'understat').sum())} "
          f"automated from Understat ({(cur['ID_Source'] == 'name match').sum()} by name match)")


def priors_frame(params=None, pf=None, pr=None):
    """The priors table build() writes. params / pf / pr are the §103, §116 and §127 parameter sets; None reads the
    frozen JSONs. build_oot_priors.py passes its
    season-by-season fits instead."""
    params = json.loads(PARAMS.read_text(encoding="utf-8")) if params is None else params
    if pf is None and PARAMS_FOTMOB.exists():
        pf = json.loads(PARAMS_FOTMOB.read_text(encoding="utf-8"))
    if pr is None and PARAMS_REFIT.exists():
        pr = json.loads(PARAMS_REFIT.read_text(encoding="utf-8"))
    arr, slots = dataset(with_outcomes=False)
    scales = era_scales(sorted(arr["Season"].unique()))
    out = arr[["Player", "Season", "Pos", "League", "Group", "Path", "Understat_ID", "ID_Source", "Age",
               "Team_Attack_Rating"]].copy()
    for t, col in [("npxG", "Initial_NPG_Rating"), ("xAG", "Initial_Assist_Rating")]:
        out[col] = npp.PriorModel.predict_with(params[t], arr, slots, t)
        if t in params["era_scale_targets"]:
            out[col] *= arr["Season"].map(lambda S: scales[S][t]).to_numpy()
    out["Understat_Seasons"] = (slots["m"] > 0).sum(1) * (arr["Path"] == "understat")
    out["Prior_Minutes_Used"] = slots["m"].sum(1)
    out["Adopted"] = out["Group"].isin(params["adopted_groups"])
    if pf is not None:
        farr, fslots = fp.attach_fotmob(arr, slots, fp.load_fotmob())
        use = ((farr["Path"] == "fotmob") & farr["Group"].isin(pf["adopted_groups"])).to_numpy()
        for t, col in [("npxG", "Initial_NPG_Rating"), ("xAG", "Initial_Assist_Rating")]:
            v = fp.FPriorModel.predict_with(pf[t], farr, fslots, t)
            if t in pf["era_scale_targets"]:
                v = v * arr["Season"].map(lambda S: scales[S][t]).to_numpy()
            out.loc[use, col] = v[use]
        out.loc[use, "Path"] = "fotmob"
        out.loc[use, "Prior_Minutes_Used"] = fslots["m"].sum(1)[use]
        out.loc[use, "Adopted"] = True
        print(f"[§116] FotMob path for {int(use.sum())} arrivals in {pf['adopted_groups']}")
    if pr is not None:
        farr, fslots = fp.attach_fotmob(arr, slots, fp.load_fotmob())
        use = farr["Group"].isin(pr["adopted_groups"]).to_numpy()
        for t, col in [("npxG", "Initial_NPG_Rating"), ("xAG", "Initial_Assist_Rating")]:
            v = fp.FPriorModel.predict_with(pr[t], farr, fslots, t)
            if t in pr["era_scale_targets"]:
                v = v * arr["Season"].map(lambda S: scales[S][t]).to_numpy()
            out.loc[use, col] = v[use]
        out.loc[use, "Path"] = farr["Path"].to_numpy()[use]
        out.loc[use, "Prior_Minutes_Used"] = fslots["m"].sum(1)[use]
        out.loc[use, "Adopted"] = True
        print(f"[§127] refit model for {int(use.sum())} arrivals in {pr['adopted_groups']}")
    out["Age"] = arr["Age"].where(arr["Has_DOB"])
    out = out[["Player", "Season", "Initial_NPG_Rating", "Initial_Assist_Rating", "Adopted", "Group", "Path",
               "League", "Pos", "Understat_ID", "ID_Source", "Understat_Seasons", "Prior_Minutes_Used", "Age",
               "Team_Attack_Rating"]]
    tg = tagged_championship(params, scales, pf)
    out = pd.concat([out, tg[out.columns]], ignore_index=True)
    assert not out.duplicated(["Player", "Season"]).any()
    print(f"[§130] {len(tg)} tagged Championship arrivals (mid-season / two leagues) get the fitted prior")
    if out[["Initial_NPG_Rating", "Initial_Assist_Rating"]].isna().any().any():
        bad = out[out[["Initial_NPG_Rating", "Initial_Assist_Rating"]].isna().any(axis=1)]
        raise ValueError(f"NaN priors:\n{bad[['Player', 'Season', 'League']]}")
    return out


if __name__ == "__main__":
    if "--fit-refit" in sys.argv:
        fit_refit()
    elif "--fit-fotmob" in sys.argv:
        fit_fotmob()
    elif "--fit" in sys.argv:
        fit()
    else:
        build()
