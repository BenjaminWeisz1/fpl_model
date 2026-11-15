"""
Out-of-time arrival priors that match production's prior setup (CHANGELOG §245).

    python -m research.player_ratings.build_oot_priors --check   # cut-off 2025: must reproduce the three frozen parameter JSONs and
                                         # data/current/new_player_priors.csv
    python -m research.player_ratings.build_oot_priors           # season by season, 2019-2026 -> <out>/oot_priors.csv

For each arrival season S from 2019 to 2026, every parameter set production uses is fitted on arrivals from seasons
before S only (cut-off S-1), with the same code, paths, groups, adoption flags and fit logs as
build_new_player_priors.py (fit / fit_fotmob / fit_refit with write=False, then priors_frame): the §103 base model
(fit log new_player_priors_20261028/baseline_log.csv), the §116 FotMob-path model (prior_k_20261029/log_k16.csv) and the
§127 refit (prior_refit_beta_20261030/log_beta.csv), the §104 era scales (already past-only) and the §130 tagged
Championship rows. The season-S rows of the resulting table are kept. Seasons 2017-2018 have no earlier arrivals to fit
on (2017) or too few: they take production's frozen priors, which are in-sample (disclosed, as §104 did).

Nothing here writes production files: the frozen JSONs are never rewritten (write=False) and the output goes to
--out (default data/validation/position_rates_20261114/phase1/). GOAL_MODEL_POSITION_RATES (Phase 2, §246) selects the
position-rate table the priors use, through new_player_priors.position_priors.

Output columns: Player, Season, Initial_NPG_Rating, Initial_Assist_Rating, Adopted, Path, Param_Set, Group, Cutoff.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.atomic_write import to_csv_atomic  # noqa: E402
from fpl_model.player_ratings import build_new_player_priors as bnp  # noqa: E402
from fpl_model.player_ratings import fotmob_priors as fp  # noqa: E402
from fpl_model.player_ratings import new_player_priors as npp  # noqa: E402

DEFAULT_OUT = npp.ROOT / "data" / "validation" / "position_rates_20261114" / "phase1"
OOT_SEASONS = range(2019, 2027)
FROZEN_SEASONS = (2017, 2018)

# ── memoise the slow loaders (pure functions of frozen files) ─────────────────
_memo = {}


def _cached(name, fn):
    def wrap(*a, **k):
        key = (name, a, tuple(sorted(k.items())))
        if key not in _memo:
            _memo[key] = fn(*a, **k)
        return copy.deepcopy(_memo[key])
    return wrap


npp.load_understat = _cached("understat", npp.load_understat)
npp.birth_dates = _cached("dob", npp.birth_dates)
fp.load_fotmob = _cached("fotmob", fp.load_fotmob)
bnp.dataset = _cached("dataset", bnp.dataset)


def fit_all(cutoff):
    return {"base": bnp.fit(last_season=cutoff, write=False),
            "fotmob": bnp.fit_fotmob(last_season=cutoff, write=False),
            "refit": bnp.fit_refit(last_season=cutoff, write=False)}


def param_set(frame, ps):
    """Which parameter set produced each row (priors_frame's order: base, then FotMob path, then refit groups)."""
    out = np.where(frame["Group"].isin(ps["refit"]["adopted_groups"]), "refit_127",
                   np.where((frame["Path"] == "fotmob") & frame["Group"].isin(ps["fotmob"]["adopted_groups"]),
                            "fotmob_116", "base_103"))
    tagged = (frame["ID_Source"] == "tagged").to_numpy()
    out = np.where(tagged, np.where(frame["Path"] == "fotmob", "tagged_fotmob_116", "tagged_base_103"), out)
    return np.where(frame["Adopted"], out, "not_adopted")


def _rel(a, b):
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(b), 1e-12)


def compare_params(new, frozen, path=""):
    """Largest relative difference over every numeric leaf; lists of non-numeric leaves that differ."""
    worst, diffs = 0.0, []
    if isinstance(frozen, dict):
        for k in frozen:
            if k == "fitted_on" or k == "fit_log":
                continue
            if k not in new:
                diffs.append(f"{path}/{k} missing")
                continue
            w, d = compare_params(new[k], frozen[k], f"{path}/{k}")
            worst = max(worst, w); diffs += d
        for k in set(new) - set(frozen):
            diffs.append(f"{path}/{k} extra")
    elif isinstance(frozen, (int, float)) and not isinstance(frozen, bool):
        r = _rel(new, frozen)
        if r > 1e-6:
            diffs.append(f"{path}: {new!r} v frozen {frozen!r} (rel {r:.2e})")
        worst = r
    elif new != frozen:
        diffs.append(f"{path}: {new!r} v frozen {frozen!r}")
    return worst, diffs


def check(out_dir):
    t0 = time.time()
    ps = fit_all(bnp.FIT_LAST_SEASON)
    frozen = {"base": bnp.PARAMS, "fotmob": bnp.PARAMS_FOTMOB, "refit": bnp.PARAMS_REFIT}
    res = {"cutoff": bnp.FIT_LAST_SEASON, "params": {}}
    for k, path in frozen.items():
        fz = json.loads(path.read_text(encoding="utf-8"))
        worst, diffs = compare_params(ps[k], fz)
        res["params"][k] = {"file": path.name, "max_rel_diff": worst, "n_diffs_over_1e-6": len(diffs),
                            "diffs": diffs[:40], "fitted_on_new": ps[k]["fitted_on"], "fitted_on_frozen": fz["fitted_on"]}
        print(f"[check] {path.name}: max rel diff {worst:.3e}, {len(diffs)} leaves over 1e-6; "
              f"fitted on {ps[k]['fitted_on']} (frozen: {fz['fitted_on']})", flush=True)
        (out_dir / f"check_params_{k}.json").write_text(json.dumps(ps[k], indent=1, ensure_ascii=False), encoding="utf-8")
    new = bnp.priors_frame(ps["base"], ps["fotmob"], ps["refit"])
    cur = pd.read_csv(bnp.OUT, encoding="utf-8-sig")
    m = cur.merge(new, on=["Player", "Season"], how="outer", suffixes=("_cur", "_new"), indicator=True)
    pr = {"rows_current": int(len(cur)), "rows_new": int(len(new)),
          "unmatched": int((m["_merge"] != "both").sum())}
    for c in ["Initial_NPG_Rating", "Initial_Assist_Rating"]:
        d = (m[c + "_cur"] - m[c + "_new"]).abs()
        pr[c] = {"max_abs_diff": float(d.max()), "rows_over_1e-6": int((d > 1e-6).sum())}
    pr["adopted_identical"] = bool((m["Adopted_cur"] == m["Adopted_new"]).all())
    res["priors_file"] = pr
    res["seconds"] = round(time.time() - t0)
    print(f"[check] new_player_priors.csv: {json.dumps(pr)}", flush=True)
    (out_dir / "check_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def build(out_dir):
    t0 = time.time()
    frozen = bnp.priors_frame()
    parts = []
    fz = frozen[frozen["Season"].isin(FROZEN_SEASONS)].copy()
    fz_ps = {"base": json.loads(bnp.PARAMS.read_text(encoding="utf-8")),
             "fotmob": json.loads(bnp.PARAMS_FOTMOB.read_text(encoding="utf-8")),
             "refit": json.loads(bnp.PARAMS_REFIT.read_text(encoding="utf-8"))}
    fz["Param_Set"] = param_set(fz, fz_ps)
    fz["Cutoff"] = "frozen 2025 (in-sample)"
    parts.append(fz)
    pdir = out_dir / "oot_params"
    pdir.mkdir(parents=True, exist_ok=True)
    for S in OOT_SEASONS:
        ps = fit_all(S - 1)
        for k, v in ps.items():
            (pdir / f"S{S}_{k}.json").write_text(json.dumps(v, indent=1, ensure_ascii=False), encoding="utf-8")
        fr = bnp.priors_frame(ps["base"], ps["fotmob"], ps["refit"])
        fr = fr[fr["Season"] == S].copy()
        fr["Param_Set"] = param_set(fr, ps)
        fr["Cutoff"] = str(S - 1)
        parts.append(fr)
        print(f"[oot] S={S}: {len(fr)} rows, {int(fr['Adopted'].sum())} adopted "
              f"({fr.loc[fr['Adopted'], 'Param_Set'].value_counts().to_dict()}); {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(parts, ignore_index=True)
    assert not out.duplicated(["Player", "Season"]).any()
    assert len(out) == len(frozen), (len(out), len(frozen))
    cols = ["Player", "Season", "Initial_NPG_Rating", "Initial_Assist_Rating", "Adopted", "Path", "Param_Set", "Group",
            "League", "Pos", "Cutoff"]
    to_csv_atomic(out[cols], out_dir / "oot_priors.csv", index=False, encoding="utf-8-sig")
    print(f"wrote {len(out)} rows -> {out_dir / 'oot_priors.csv'} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    check(od) if a.check else build(od)
