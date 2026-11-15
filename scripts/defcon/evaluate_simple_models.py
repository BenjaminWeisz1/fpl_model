"""Item 2, AGENDA 2026-10-31 (CHANGELOG §150): simpler defcon models against the incumbent heads, out of time.

Arms: count (negative-binomial GLM on rate, team, opponent, venue, position), smoothing (opponent-adjusted exponential
smoothing, a floor), logistic (regularised logistic regression). Each ends in the same 5-fold isotonic calibration on
training seasons. Uses item 1's frames and predictions as the incumbent. No production writes.
Run: python -u -X utf8 scripts/defcon/evaluate_simple_models.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.stats import nbinom
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import overnight_frames as H
from defender_model import CenteredIsotonic

YEARS = [2022, 2023, 2024, 2025, 2026]
THRESH = {"D": 10, "M": 12, "F": 12}
HALF_LIVES = [4, 8, 16, 32]
PRIOR_K = [2, 5, 10]


def player_states(f, half_life, k):
    """Opponent- and venue-adjusted EW rate per player before each row, shrunk to the position group's mean of the
    earlier seasons (the head's expected-rate column gives the fallback). Rows must be in chronological player order."""
    d = 2.0 ** (-1.0 / half_life)
    sm_ = f[f.Minutes >= 60].groupby(["_group", "Season"]).Per90Stat.mean().reset_index()
    prev = {}
    for grp, gg in sm_.groupby("_group"):
        vals = dict(zip(gg.Season, gg.Per90Stat))
        for s in range(2017, 2028):
            past = [v for y, v in vals.items() if y < s]
            prev[(grp, s)] = float(np.mean(past[-3:])) if past else float(np.mean(list(vals.values())[:1]))
    base = np.array([prev[(a, b)] for a, b in zip(f._group.values, f.Season.values)])
    rate = np.zeros(len(f))
    adj = f.Per90Stat.values / np.clip(f.OppFactor.values, .5, 2)
    for _, pos in f.groupby("Player", sort=False).indices.items():
        num = den = 0.0
        for j in pos:
            rate[j] = (num + k * base[j]) / (den + k)
            if f.Minutes.values[j] >= 60:
                num = num * d + adj[j]
                den = den * d + 1.0
    return rate


def select_smoothing(f):
    dev = f[(f.Minutes >= 60) & f.Season.between(2018, 2021)]
    res = {}
    for h in HALF_LIVES:
        for k in PRIOR_K:
            r = player_states(f, h, k)[dev.index] * dev.OppFactor.values
            r = r * dev.Per90Stat.sum() / r.sum()
            y = dev.Per90Stat.values
            with np.errstate(divide="ignore", invalid="ignore"):
                res[(h, k)] = float((2 * (np.where(y > 0, y * np.log(y / r), 0) - (y - r))).mean())
    return min(res, key=res.get), {f"{a}/{b}": v for (a, b), v in res.items()}


def add_features(f, head):
    f = f.sort_values(["Player", "Kickoff"]).reset_index(drop=True)
    best, grid = select_smoothing(f)
    f["SmoothRate"] = player_states(f, *best)
    # Trailing share of appearances reaching 60 minutes: the incumbent sees this through AvgRaw / AvgPer90.
    start = np.full(len(f), .7)
    d = 2.0 ** (-1.0 / 8)
    for _, pos in f.groupby("Player", sort=False).indices.items():
        num, den = 2.8, 4.0
        for j in pos:
            start[j] = num / den
            full = float(f.Minutes.values[j] >= 60)
            num, den = num * d + full, den * d + 1
    f["Full60Share"] = start
    f["log_rate"] = np.log(np.clip(f.SmoothRate, .5, None))
    f["log_avg"] = np.log(np.clip(f.AvgPer90, .5, None))
    f["log_recent"] = np.log(np.clip(f.Recent5AvgPer90, .5, None))
    f["log_team"] = np.log(np.clip(f.TeamAvgStat, .5, None))
    f["log_opp"] = np.log(np.clip(f.OppFactor, .4, 2.5))
    f["log_matches"] = np.log1p(f.MatchesPlayed)
    f["logit_full"] = np.log(f.Full60Share / (1 - f.Full60Share))
    return f, best, grid


def pos_cols(f):
    return [c for c in f if c.startswith("Pos_")]


def nb_prob(mu, r, t):
    return nbinom.sf(np.ceil(t) - 1, r, r / (r + mu))


def count_design(f):
    x = f[["log_rate", "log_avg", "log_team", "log_opp", "IsHome", "Is_FPL_API"] + pos_cols(f)].astype(float)
    return sm.add_constant(x, has_constant="add")


def count_fit(tr):
    q = tr[tr.Minutes >= 60]
    y = np.round(q.Per90Stat.values)
    m = sm.GLM(y, count_design(q), family=sm.families.Poisson()).fit_regularized(alpha=1e-5, L1_wt=0, maxiter=200)
    mu = np.exp(np.asarray(count_design(q)) @ m.params)
    # Moment estimate of the NB dispersion from Pearson-type residuals.
    r = max(1.0, float((mu ** 2).sum() / max(((y - mu) ** 2 - mu).sum(), 1e-6)))
    return m.params, r


def count_score(params, r, f, head):
    mu = np.exp(np.asarray(count_design(f)) @ params)
    return nb_prob(mu, r, THRESH[head])


def smooth_score(tr_r, f, head):
    mu = np.clip(f.SmoothRate.values * f.OppFactor.values * tr_r["home"] ** (2 * f.IsHome.values - 1), .1, None)
    return nb_prob(mu, tr_r["r"], THRESH[head])


def smooth_fit(tr):
    q = tr[tr.Minutes >= 60]
    mu0 = np.clip(q.SmoothRate.values * q.OppFactor.values, .1, None)
    y = q.Per90Stat.values
    home = float(np.sqrt((y[q.IsHome == 1].sum() / mu0[q.IsHome == 1].sum()) / (y[q.IsHome == 0].sum() / mu0[q.IsHome == 0].sum())))
    mu = mu0 * home ** (2 * q.IsHome.values - 1)
    r = max(1.0, float((mu ** 2).sum() / max(((y - mu) ** 2 - mu).sum(), 1e-6)))
    return {"home": home, "r": r}


def logit_cols(f):
    return ["log_rate", "log_avg", "log_recent", "log_team", "log_opp", "IsHome", "Is_FPL_API", "log_matches", "logit_full",
            "PctAboveThreshold"] + pos_cols(f)


def logit_fit(tr):
    c = logit_cols(tr)
    sc = StandardScaler().fit(tr[c].values)
    return sc, LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(tr[c].values), tr.Target.values)


def calibrated(head, tr, va, fit, score):
    """Five out-of-fold training scores -> isotonic -> held-out scores. A logit layer adds the 60-minute share first, so
    count and smoothing arms carry the same who-plays-60 information as the incumbent."""
    oof = np.zeros(len(tr))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for a, b in cv.split(tr, tr.Target.values):
        oof[b] = score(fit(tr.iloc[a]), tr.iloc[b])
    oof = np.clip(oof, 1e-5, 1 - 1e-5)
    full = fit(tr)
    pv = np.clip(score(full, va), 1e-5, 1 - 1e-5)
    def layer(p, g):
        return np.column_stack([np.log(p / (1 - p)), g.logit_full.values])
    lg = LogisticRegression(C=100, max_iter=1000).fit(layer(oof, tr), tr.Target.values)
    a = lg.predict_proba(layer(oof, tr))[:, 1]
    b = lg.predict_proba(layer(pv, va))[:, 1]
    cal = (CenteredIsotonic() if head == "D" else IsotonicRegression(out_of_bounds="clip")).fit(a, tr.Target.values)
    return np.clip(cal.predict(b), 1e-6, 1 - 1e-6)


def one(head, f, year, arm):
    tr, va = f[f.Season < year], f[f.Season == year]
    if arm == "count":
        p = calibrated(head, tr, va, count_fit, lambda m, g: count_score(m[0], m[1], g, head))
    elif arm == "smoothing":
        p = calibrated(head, tr, va, smooth_fit, lambda m, g: smooth_score(m, g, head))
    else:
        def lscore(m, g):
            return m[1].predict_proba(m[0].transform(g[logit_cols(g)].values))[:, 1]
        oof = np.zeros(len(tr))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(tr, tr.Target.values):
            oof[b] = lscore(logit_fit(tr.iloc[a]), tr.iloc[b])
        cal = (CenteredIsotonic() if head == "D" else IsotonicRegression(out_of_bounds="clip")).fit(np.clip(oof, 1e-6, 1 - 1e-6), tr.Target.values)
        p = np.clip(cal.predict(lscore(logit_fit(tr), va)), 1e-6, 1 - 1e-6)
    print(head, year, arm, flush=True)
    return head, year, arm, va.index.values, p


def main():
    z = pd.read_pickle(H.OUT / "item1_frames.pkl")
    data, sel = {}, {}
    for h, f in z["data"].items():
        data[h], best, grid = add_features(f, h)
        sel[h] = {"selected": best, "grid": grid}
        print(h, "smoothing selected", best, flush=True)
    data_inc = {h: pd.read_csv(H.OUT / f"item1_predictions_{h}.csv", encoding="utf-8-sig") for h in data}
    jobs = [(h, y, a) for h in data for y in YEARS for a in ("count", "smoothing", "logistic")]
    res = Parallel(n_jobs=4, backend="loky")(delayed(one)(h, data[h], y, a) for h, y, a in jobs)
    out, rows = {"selection": {k: {"selected": v["selected"], "grid": v["grid"]} for k, v in sel.items()}}, []
    for h, f in data.items():
        for hh, y, a, idx, p in res:
            if hh == h:
                f.loc[idx, f"p_{a}"] = p
        key = ["Season", "Player", "Team", "Opponent", "Venue"]
        inc = data_inc[h].drop_duplicates(key).copy()
        if h in ("D", "M"):  # §149 passed for these heads: opp_factor is the incumbent to beat
            inc["p_control"] = inc["p_opp_factor"]
        v = f[f.Season.isin(YEARS)].merge(inc[key + ["p_control"]], on=key, how="inner", validate="one_to_one")
        assert len(v) == len(data_inc[h]), (len(v), len(data_inc[h]))
        out[h] = {}
        for a in ("control", "count", "smoothing", "logistic"):
            s = {"pooled": H.score(v.Target.values, v[f"p_{a}"].values),
                 "by_year": {str(y): H.score(g.Target.values, g[f"p_{a}"].values) for y, g in v.groupby("Season")}}
            q = v[v.Minutes >= 60]
            s["over_60_pooled"] = H.score(q.Target.values, q[f"p_{a}"].values)
            s["vs_control"] = H.paired_bootstrap(v.Player.values, v.Target.values, v.p_control.values, v[f"p_{a}"].values)
            out[h][a] = s
            rows.append((h, a, s["pooled"]["ll"], s["pooled"]["auc"], s["over_60_pooled"]["ll"], s["vs_control"]["lo"], s["vs_control"]["hi"],
                         *[s["by_year"][str(y)]["ll"] - out[h]["control"]["by_year"][str(y)]["ll"] for y in YEARS]))
        v[key + ["Minutes", "Target", "p_control", "p_count", "p_smoothing", "p_logistic"]].to_csv(H.OUT / f"item2_predictions_{h}.csv", index=False, encoding="utf-8-sig")
    (H.OUT / "item2_results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(pd.DataFrame(rows, columns=["head", "arm", "ll", "auc", "ll60", "lo", "hi"] + [f"d{y}" for y in YEARS]).round(5).to_string())


if __name__ == "__main__":
    main()
