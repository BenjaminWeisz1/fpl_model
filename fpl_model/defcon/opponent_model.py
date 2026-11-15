"""Opponent factor (CHANGELOG §149, adopted) and the midfielder count head (§150, not served), for training rows and forecasts.

OppFactor: for a position group, the actions conceded by an opponent / the pre-match expected actions of the players who
faced it, EW over its earlier matches, relative to the same ratio league-wide and shrunk toward 1. Sequenced by
match_results.csv row order (chronological), so it needs no kickoff times.

Midfield count head: Poisson GLM on >= 60-minute rows, negative-binomial tail at the threshold, then a logistic layer with
the trailing 60-minute share and isotonic calibration, both on 5-fold out-of-fold scores.
`DEFCON_OPP_FACTOR=0` reverts in predict.py. The count head is off by default (§150: worse on FPL's real hits, and its smoothed
rate ignores transfer priors); `DEFCON_MID_MODEL=count` serves it.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import nbinom
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

OPP_HALF_LIFE, OPP_K = 19, 20        # §149, chosen on 2018-2021
RATE_HALF_LIFE, RATE_K = 16, 2       # §150, chosen on 2018-2021
FULL_HALF_LIFE, FULL_PRIOR, FULL_PRIOR_N = 8, .7, 4.0
THRESHOLD = 12
WIDE_DEF = ["Pos_FB", "Pos_WB", "Pos_LB", "Pos_RB"]


def position_groups(frame, head):
    """head D: CB or wide defender. Head M: central (DM, CM) or other. Unknown slots fall in CB / other."""
    if head == "D":
        return np.where((frame["Pos_CB"].values == 0) & (frame[WIDE_DEF].sum(axis=1).values > 0), "WD", "CB")
    return np.where(frame[["Pos_DM", "Pos_CM"]].sum(axis=1).values > 0, "CM", "AM")


def match_order(data_dir):
    """Team-match keys in chronological order: one list per match, home side first."""
    m = pd.read_csv(data_dir / "match_results.csv", encoding="utf-8-sig", usecols=["Season", "Home", "Away"])
    return [((int(s), h, a, "Home"), (int(s), a, h, "Away")) for s, h, a in zip(m.Season, m.Home, m.Away)]


def opponent_factors(frame, head, matches, half_life=OPP_HALF_LIFE, k=OPP_K):
    """Per-row factor (uses only earlier matches) and {(opponent, group): factor} after the last match."""
    d, league_d = 2.0 ** (-1.0 / half_life), 2.0 ** (-1.0 / (half_life * 10))
    f = frame[["Season", "Team", "Opponent", "Venue", "Minutes", "MatchesPlayed", "Per90Stat", "AvgPer90", "Recent5AvgPer90"]].copy()
    f["_group"] = position_groups(frame, head)
    f["_exp"] = 0.5 * (f["AvgPer90"] + f["Recent5AvgPer90"])
    u = f[(f["Minutes"] >= 60) & (f["MatchesPlayed"] >= 5) & (f["_exp"] > 0)]
    agg = u.groupby(["Season", "Team", "Opponent", "Venue", "_group"]).agg(A=("Per90Stat", "sum"), E=("_exp", "sum"), N=("_exp", "size")).reset_index()
    totals = {(s, t, o, v, g): (a, e, n) for s, t, o, v, g, a, e, n in agg.values}
    groups = sorted(f["_group"].unique())
    state, league = {}, {g: [0.0, 0.0] for g in groups}

    def factor(opponent, g):
        A, E, N = state.get((opponent, g), (0.0, 0.0, 0.0))
        la, le = league[g]
        base = la / le if le > 0 else 1.0
        if N == 0:
            return 1.0
        ebar = E / N
        return (A + k * base * ebar) / (E + k * ebar) / base

    out = {}
    for pair in matches:
        for s, t, o, v in pair:
            for g in groups:
                out[(s, t, o, v, g)] = factor(o, g)
        for s, t, o, v in pair:
            for g in groups:
                a, e, n = totals.get((s, t, o, v, g), (0.0, 0.0, 0.0))
                A, E, N = state.get((o, g), (0.0, 0.0, 0.0))
                state[(o, g)] = (A * d + a, E * d + e, N * d + n)
                league[g][0] = league[g][0] * league_d + a
                league[g][1] = league[g][1] * league_d + e
    row = np.array([out.get(key, 1.0) for key in zip(f["Season"], f["Team"], f["Opponent"], f["Venue"], f["_group"])])
    latest = {(o, g): factor(o, g) for o, g in state}
    return row, latest


def rate_bases(frame, groups):
    """(group, season) -> mean >= 60-minute per-90 of up to three earlier seasons."""
    q = frame.assign(_g=groups)
    q = q[q["Minutes"] >= 60]
    means = q.groupby(["_g", "Season"])["Per90Stat"].mean()
    bases = {}
    for g, s in means.groupby(level=0):
        vals = {int(i[1]): v for i, v in s.items()}
        first = vals[min(vals)]
        for season in range(min(vals), max(vals) + 2):
            past = [vals[y] for y in sorted(vals) if y < season]
            bases[(g, season)] = float(np.mean(past[-3:])) if past else first
    return bases


def add_rate_columns(frame):
    """Adds SmoothRate and Full60Share (pre-match, frame in production order). Returns (frame, latest states, bases)."""
    f = frame.copy()
    groups = position_groups(f, "M")
    bases = rate_bases(f, groups)
    base = np.array([bases[(g, int(s))] for g, s in zip(groups, f["Season"])])
    d, dfull = 2.0 ** (-1.0 / RATE_HALF_LIFE), 2.0 ** (-1.0 / FULL_HALF_LIFE)
    adj = f["Per90Stat"].values / np.clip(f["OppFactor"].values, .5, 2)
    mins = f["Minutes"].values
    rate, full, latest = np.zeros(len(f)), np.zeros(len(f)), {}
    for player, pos in f.groupby("Player", sort=False).indices.items():
        num = den = 0.0
        fn, fd = FULL_PRIOR * FULL_PRIOR_N, FULL_PRIOR_N
        for j in pos:
            rate[j] = (num + RATE_K * base[j]) / (den + RATE_K)
            full[j] = fn / fd
            q = float(mins[j] >= 60)
            if q:
                num, den = num * d + adj[j], den * d + 1.0
            fn, fd = fn * dfull + q, fd * dfull + 1.0
        latest[player] = (num, den, fn / fd)
    f["SmoothRate"], f["Full60Share"] = rate, full
    return f, latest, bases


def serving_rate(player, latest, base):
    num, den, full = latest.get(player, (0.0, 0.0, FULL_PRIOR))
    return (num + RATE_K * base) / (den + RATE_K), full


def _design(f, pos_cols):
    x = pd.DataFrame({
        "log_rate": np.log(np.clip(f["SmoothRate"].values.astype(float), .5, None)),
        "log_avg": np.log(np.clip(f["AvgPer90"].values.astype(float), .5, None)),
        "log_team": np.log(np.clip(f["TeamAvgStat"].values.astype(float), .5, None)),
        "log_opp": np.log(np.clip(f["OppFactor"].values.astype(float), .4, 2.5)),
        "IsHome": f["IsHome"].values.astype(float), "Is_FPL_API": f["Is_FPL_API"].values.astype(float),
    })
    for c in pos_cols:
        x[c] = f[c].values.astype(float)
    return sm.add_constant(x, has_constant="add")


def _full_logit(f):
    s = f["Full60Share"].values.astype(float)
    return np.log(s / (1 - s))


def _layer(p, f):
    return np.column_stack([np.log(p / (1 - p)), _full_logit(f)])


class CountHead:
    """Scores a frame carrying SmoothRate, Full60Share, AvgPer90, TeamAvgStat, OppFactor, IsHome, Is_FPL_API, Pos_*."""

    def __init__(self, pos_cols):
        self.pos_cols = pos_cols

    def _glm(self, tr):
        q = tr[tr["Minutes"] >= 60]
        y = np.round(q["Per90Stat"].values)
        x = _design(q, self.pos_cols)
        params = sm.GLM(y, x, family=sm.families.Poisson()).fit_regularized(alpha=1e-5, L1_wt=0, maxiter=200).params
        mu = np.exp(np.asarray(x) @ np.asarray(params))
        r = max(1.0, float((mu ** 2).sum() / max(((y - mu) ** 2 - mu).sum(), 1e-6)))
        return np.asarray(params), r

    def _tail(self, fit, f):
        params, r = fit
        mu = np.exp(np.asarray(_design(f, self.pos_cols)) @ params)
        return np.clip(nbinom.sf(THRESHOLD - 1, r, r / (r + mu)), 1e-5, 1 - 1e-5)

    def fit(self, tr):
        tr = tr.reset_index(drop=True)
        y = tr["Target"].values
        oof = np.zeros(len(tr))
        for a, b in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(tr, y):
            oof[b] = self._tail(self._glm(tr.iloc[a]), tr.iloc[b])
        oof = np.clip(oof, 1e-5, 1 - 1e-5)
        self.fit_ = self._glm(tr)
        self.layer_ = LogisticRegression(C=100, max_iter=1000).fit(_layer(oof, tr), y)
        self.cal_ = IsotonicRegression(out_of_bounds="clip").fit(self.layer_.predict_proba(_layer(oof, tr))[:, 1], y)
        return self

    def predict(self, f):
        p = self.layer_.predict_proba(_layer(self._tail(self.fit_, f), f))[:, 1]
        return np.clip(self.cal_.predict(p), 1e-6, 1 - 1e-6)
