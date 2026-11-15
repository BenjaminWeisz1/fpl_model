"""Step 2 of the position-switch plan: does a regular's return predict the move back beyond the role percentage?

Protocol, frozen before any return effect was computed:
data/validation/cover_return_20261101/protocol.json. Uses who started where only. Changes no production file.
"""
from fpl_model import paths
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fpl_model.player_ratings.role_shares import LINES, load_starts, role_probs

OUT = paths.VALIDATION / "cover_return_20261101"
MIN_STARTS, TRAIN_END = 10, 2021
N_GAIN, N_COEF, SEED = 5000, 2000, 171
K = len(LINES)


def find_rows(s):
    team_dates = {t: sorted(g.Date.unique()) for t, g in s.groupby("Team")}
    starters = defaultdict(dict)   # (team, date) -> {pid: (slot, line)}
    for r in s.itertuples():
        starters[(r.Team, r.Date)][r.Player_ID_Understat] = (r.Position, r.L)

    rows, windows, eps = [], [], []
    for pid, g in s.groupby("Player_ID_Understat", sort=False):
        if len(g) <= MIN_STARTS:
            continue
        L, T, D, P, S = g.L.values, g.Team.values, g.Date.values, g.Position.values, g.Season.values
        probs = role_probs(L)
        counts = np.vstack([np.zeros(K), np.cumsum(np.eye(K)[L], axis=0)])
        ep = None
        for t in range(MIN_STARTS, len(L)):
            N, O = int(probs[t].argmax()), int(counts[t].argmax())
            displaced = N != O and T[t] == T[t - 1]
            if not displaced or (ep is not None and ep["team"] != T[t]):
                ep = None
            if not displaced:
                continue
            if ep is None:
                ep = open_episode(pid, g.Player.iloc[0], t, L, T, D, P, N, O, team_dates, starters)
                eps.append(ep)
                if ep["R"] is not None:
                    windows.extend(window_rows(ep, T[t], D[t], team_dates, starters))
            if ep["R"] is None or ep["N"] != N or ep["O"] != O:
                continue
            here = starters[(T[t], D[t])]
            rows.append(dict(pid=pid, ep=ep["id"], season=int(S[t]), share_O=probs[t][O], y=int(L[t] == O),
                             stayed=int(L[t] == N), ret=int(ep["R"] in here),
                             ret_line=int(ep["R"] in here and here[ep["R"]][1] == N)))
    return pd.DataFrame(rows), pd.DataFrame(windows), pd.DataFrame(eps)


def open_episode(pid, name, t, L, T, D, P, N, O, team_dates, starters):
    ep = dict(id=f"{pid}_{t}", pid=pid, name=name, team=T[t], N=N, O=O, R=None, date=D[t], run=0)
    j = t - 1
    while j >= 0 and L[j] == N and T[j] == T[t]:
        j -= 1
    run = list(range(j + 1, t))
    ep["run"] = len(run)
    if not run:
        return ep
    slot = Counter(P[run]).most_common(1)[0][0]
    dates = team_dates[T[t]]
    k = int(np.searchsorted(dates, D[run[0]]))
    c = Counter()
    for d in dates[max(0, k - 10):k]:
        for q, (sl, _) in starters[(T[t], d)].items():
            if sl == slot and q != pid:
                c[q] += 1
    if not c or c.most_common(1)[0][1] < 5:
        return ep
    r = c.most_common(1)[0][0]
    span = [d for d in dates[k:] if d < D[t]]
    if sum(r in starters[(T[t], d)] for d in span) <= 1:
        ep["R"] = r
    return ep


def window_rows(ep, team, date, team_dates, starters):
    dates = team_dates[team]
    k = int(np.searchsorted(dates, date))
    out = []
    for d in dates[k:k + 10]:
        here = starters[(team, d)]
        p = here.get(ep["pid"])
        out.append(dict(ep=ep["id"], pid=ep["pid"], ret=int(ep["R"] in here), started=int(p is not None),
                        in_O=int(p is not None and p[1] == ep["O"]), in_N=int(p is not None and p[1] == ep["N"])))
    return out


def features(d, col="ret"):
    lo = np.log(np.clip(d.share_O, 1e-4, 1 - 1e-4) / (1 - np.clip(d.share_O, 1e-4, 1 - 1e-4)))
    return lo.values.reshape(-1, 1), np.column_stack([lo.values, d[col].values])


def fit(x, y):
    return LogisticRegression(C=1e6, max_iter=1000).fit(x, y)


def row_ll(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def evaluate(r, col):
    tr, te = r[r.season <= TRAIN_END], r[r.season > TRAIN_END]
    b_tr, f_tr = features(tr, col)
    b_te, f_te = features(te, col)
    base, full = fit(b_tr, tr.y), fit(f_tr, tr.y)
    pb, pf = base.predict_proba(b_te)[:, 1], full.predict_proba(f_te)[:, 1]
    diff = row_ll(pb, te.y.values) - row_ll(pf, te.y.values)
    u, inv = np.unique(te.pid.values, return_inverse=True)
    tot, cnt = np.bincount(inv, diff), np.bincount(inv)
    rng = np.random.default_rng(SEED)
    ix = rng.integers(0, len(u), size=(N_GAIN, len(u)))
    v = tot[ix].sum(1) / cnt[ix].sum(1)
    on, off = f_te.copy(), f_te.copy()
    on[:, 1], off[:, 1] = 1, 0
    rise = full.predict_proba(on)[:, 1] - full.predict_proba(off)[:, 1]

    _, f_all = features(r, col)
    y_all, groups = r.y.values, r.pid.values
    by_p = {p: np.where(groups == p)[0] for p in np.unique(groups)}
    keys = list(by_p)
    cs = []
    for _ in range(N_COEF):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([by_p[keys[i]] for i in pick])
        if y_all[idx].min() == y_all[idx].max():
            continue
        cs.append(fit(f_all[idx], y_all[idx]).coef_[0][1])
    c_all = fit(f_all, y_all).coef_[0][1]
    return dict(rows_train=len(tr), rows_score=len(te), players_score=int(te.pid.nunique()),
                return_rows_train=int(tr[col].sum()), return_rows_score=int(te[col].sum()),
                c_train=round(float(full.coef_[0][1]), 3), b_train=round(float(full.coef_[0][0]), 3),
                ll_gain=round(float(diff.mean()), 5),
                ll_gain_ci=[round(float(x), 5) for x in np.percentile(v, [2.5, 97.5])],
                mean_rise_points=round(float(rise.mean()) * 100, 1),
                c_all=round(float(c_all), 3), c_all_ci=[round(float(x), 3) for x in np.percentile(cs, [2.5, 97.5])])


def main():
    s = load_starts()
    r, w, e = find_rows(s)
    cov = e[e.R.notna()]
    print(f"episodes {len(e)}; cover {len(cov)}; players {r.pid.nunique()}; rows {len(r)}")
    print("run length at opening, cover: median", cov.run.median())
    print("\nraw rates, p's starts in cover episodes")
    raw = r.groupby("ret").agg(rows=("y", "size"), in_old=("y", "mean"), in_new=("stayed", "mean"),
                               share_old=("share_O", "mean")).round(3)
    print(raw.to_string())
    res = dict(episodes=len(e), cover_episodes=len(cov), players=int(r.pid.nunique()), rows=len(r),
               raw=raw.reset_index().to_dict("records"))
    res["primary"] = evaluate(r, "ret")
    res["same_line_sensitivity"] = evaluate(r, "ret_line")
    print("\nprimary", json.dumps(res["primary"]))
    print("same-line sensitivity", json.dumps(res["same_line_sensitivity"]))
    win = w.groupby("ret").agg(club_matches=("started", "size"), p_starts=("started", "mean"),
                               p_in_old=("in_O", "mean"), p_in_new=("in_N", "mean")).round(3)
    win["old_given_start"] = (win.p_in_old / win.p_starts).round(3)
    win["new_given_start"] = (win.p_in_new / win.p_starts).round(3)
    print("\n10 club matches from each opening, by whether R started\n", win.to_string())
    res["windows"] = win.reset_index().to_dict("records")
    p = res["primary"]
    g1, g2 = p["ll_gain_ci"][0] > 0, p["mean_rise_points"] >= 10
    res["gate"] = dict(held_out_gain=bool(g1), rise_10_points=bool(g2), verdict="PASS" if g1 and g2 else "STOP")
    print("\nGATE", res["gate"])
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    cov.assign(N=cov.N.map(lambda i: LINES[i]), O=cov.O.map(lambda i: LINES[i])).to_csv(
        OUT / "cover_episodes.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
