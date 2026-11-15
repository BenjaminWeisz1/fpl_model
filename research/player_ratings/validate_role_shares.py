"""Step 1 of the position-switch plan: choose and calibrate role percentages.

Protocol, frozen before any calibration was computed:
data/validation/role_shares_20261101/protocol.json. Uses who started where only.
Choose on seasons to 2023, score 2024-2026 once. Changes no production file.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from fpl_model import paths
from fpl_model.player_ratings.role_shares import LINES, load_starts, shares, weighted_counts

OUT = paths.VALIDATION / "role_shares_20261101"
HALF_LIVES = [2, 3, 4, 5, 6, 8, 10, 13]
PRIORS = [0.01, 0.03, 0.1, 0.25, 0.5]
BLENDS = [0.6, 0.8, 0.9, 0.95]
MIN_STARTS, MIN_GAIN = 10, 0.0005
K = len(LINES)
BINS = [0, .05, .2, .4, .6, .8, .95, 1.0000001]
N_BOOT, SEED = 5000, 170


def groups(s):
    return [(pid, g.L.values, g.Season.values, g.Player.iloc[0]) for pid, g in s.groupby("Player_ID_Understat", sort=False)
            if len(g) > MIN_STARTS]


def stack(gs):
    """Row metadata: player index, season, actual line, career counts before the start."""
    pl, se, y, car, ref = [], [], [], [], []
    for i, (_, L, S, _) in enumerate(gs):
        c = np.vstack([np.zeros(K), np.cumsum(np.eye(K)[L], axis=0)])[:-1]
        r = last5_reference(L)
        pl.append(np.full(len(L) - MIN_STARTS, i))
        se.append(S[MIN_STARTS:])
        y.append(L[MIN_STARTS:])
        car.append(c[MIN_STARTS:])
        ref.append(r[MIN_STARTS:])
    return tuple(np.concatenate(x) for x in (pl, se, y, car, ref))


def last5_reference(L):
    """Last-5 rule's line before each start; the first MIN_STARTS rows set the reference."""
    r = np.zeros(len(L), dtype=int)
    cur = np.bincount(L[:MIN_STARTS], minlength=K).argmax()
    for t in range(len(L)):
        r[t] = cur
        if t >= MIN_STARTS - 1:
            last = L[t - 4:t + 1]
            if len(set(last)) == 1 and last[0] != cur:
                cur = last[0]
    return r


def ew_matrix(gs, hl):
    return np.concatenate([weighted_counts(L, hl)[MIN_STARTS:-1] for _, L, _, _ in gs])


def row_ll(p, y):
    return -np.log(p[np.arange(len(y)), y])


def temper(p, t):
    q = p ** t
    return q / q.sum(1, keepdims=True)


def reliability(p, y, mask, min_n, limit):
    flat = p[mask].ravel()
    hit = np.eye(K)[y[mask]].ravel()
    rows, ok = [], True
    idx = np.digitize(flat, BINS) - 1
    for b in range(len(BINS) - 1):
        m = idx == b
        n = int(m.sum())
        if not n:
            continue
        f, a = float(flat[m].mean()), float(hit[m].mean())
        judged = n >= min_n
        ok &= (not judged) or abs(f - a) <= limit
        rows.append(dict(bin=f"{BINS[b]:.2f}-{min(BINS[b + 1], 1):.2f}", n=n, forecast=round(f, 4), actual=round(a, 4),
                         gap=round(f - a, 4), judged=judged))
    return rows, bool(ok)


def boot(diff, pl, seed):
    """Player-cluster interval for the mean of a row-level difference."""
    u, inv = np.unique(pl, return_inverse=True)
    tot, cnt = np.bincount(inv, diff), np.bincount(inv)
    ix = np.random.default_rng(seed).integers(0, len(u), size=(N_BOOT, len(u)))
    v = tot[ix].sum(1) / cnt[ix].sum(1)
    return [round(float(x), 5) for x in np.percentile(v, [2.5, 97.5])]


def main():
    s = load_starts()
    gs = groups(s)
    pl, se, y, car, ref = stack(gs)
    dev, test = se <= 2023, se >= 2024
    print(f"rows {len(y)}; dev {dev.sum()}; test {test.sum()}; players {len(gs)}")

    ew = {h: ew_matrix(gs, h) for h in HALF_LIVES}
    grid = [(h, a, row_ll(shares(ew[h], a)[dev], y[dev]).mean()) for h in HALF_LIVES for a in PRIORS]
    g = pd.DataFrame(grid, columns=["half_life", "prior", "dev_ll"]).sort_values("dev_ll")
    print(g.head(6).round(5).to_string(index=False))
    h, a, ll_a = g.iloc[0]
    h = int(h)
    pa = shares(ew[h], a)

    t = minimize_scalar(lambda x: row_ll(temper(pa[dev], x), y[dev]).mean(), bounds=(.5, 3), method="bounded").x
    ll_b = row_ll(temper(pa[dev], t), y[dev]).mean()
    pc = shares(car, a)
    blend = [(w, row_ll(w * pa[dev] + (1 - w) * pc[dev], y[dev]).mean()) for w in BLENDS]
    w, ll_c = min(blend, key=lambda x: x[1])
    print(f"A hl={h} prior={a} {ll_a:.5f} | B T={t:.3f} {ll_b:.5f} | C w={w} {ll_c:.5f}")

    chosen, p = "A", pa
    best = ll_a
    if ll_b <= ll_a - MIN_GAIN and ll_b <= ll_c:
        chosen, p = "B", temper(pa, t)
    elif ll_c <= ll_a - MIN_GAIN:
        chosen, p = "C", w * pa + (1 - w) * pc
    print("chosen", chosen)

    # last-5 rule as probabilities, fitted on dev
    h_rate = float((ref[dev] == y[dev]).mean())
    other = pc.copy()
    other[np.arange(len(y)), ref] = 0
    other = other / other.sum(1, keepdims=True)
    pb = (1 - h_rate) * other
    pb[np.arange(len(y)), ref] = h_rate

    ll_diff = row_ll(pb, y) - row_ll(p, y)
    top = p.argmax(1)
    hit_diff = (top == y).astype(float) - (ref == y).astype(float)
    career_mode = car.argmax(1)
    disp = top != career_mode

    res = dict(chosen=chosen, half_life=h, prior=float(a), temperature=round(float(t), 4), blend=w,
               dev_ll=dict(A=round(float(ll_a), 5), B=round(float(ll_b), 5), C=round(float(ll_c), 5)),
               last5_dev_hit=round(h_rate, 4), rows_test=int(test.sum()), displaced_test=int((test & disp).sum()))
    res["test"] = dict(
        ll=round(float(row_ll(p, y)[test].mean()), 5), ll_last5=round(float(row_ll(pb, y)[test].mean()), 5),
        ll_gain=round(float(ll_diff[test].mean()), 5), ll_gain_ci=boot(ll_diff[test], pl[test], SEED),
        hit=round(float((top == y)[test].mean()), 4), hit_last5=round(float((ref == y)[test].mean()), 4),
        hit_gain=round(float(hit_diff[test].mean()), 5), hit_gain_ci=boot(hit_diff[test], pl[test], SEED + 1),
        displaced_hit=round(float((top == y)[test & disp].mean()), 4),
        displaced_hit_last5=round(float((ref == y)[test & disp].mean()), 4))
    res["calibration_dev"], _ = reliability(p, y, dev, 200, .03)
    res["calibration_dev_displaced"], _ = reliability(p, y, dev & disp, 100, .05)
    res["calibration_test"], c1 = reliability(p, y, test, 200, .03)
    res["calibration_test_displaced"], c2 = reliability(p, y, test & disp, 100, .05)
    g1 = res["test"]["ll_gain_ci"][0] > 0
    g2 = res["test"]["hit_gain_ci"][0] > 0
    res["gate"] = dict(ll_gain=g1, hit_gain=g2, calibration_all=c1, calibration_displaced=c2,
                       verdict="PASS" if g1 and g2 and c1 and c2 else "FAIL")

    # current shares for players with a 2026 start, for reading only
    rows = []
    for pid, L, S, nm in gs:
        if S[-1] == 2026:
            pp = shares(weighted_counts(L, h), a)[-1:]
            pp = temper(pp, t) if chosen == "B" else pp
            if chosen == "C":
                c = np.bincount(L, minlength=K)[None, :].astype(float)
                pp = w * pp + (1 - w) * shares(c, a)
            rows.append(dict(Player=nm, Player_ID_Understat=pid, Starts=len(L), **dict(zip(LINES, pp[0].round(4)))))
    cur = pd.DataFrame(rows)
    cur.to_csv(OUT / "current_role_shares.csv", index=False, encoding="utf-8-sig")
    (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))
    print(cur[cur.Player.str.contains("Cuyper|Gvardiol|Saka|Reece James|Hume")].to_string(index=False))


if __name__ == "__main__":
    main()
