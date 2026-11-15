"""Suspension threat and yellow cards (CHANGELOG §191). Protocol: data/validation/card_suspension_20261102/protocol.json.
No production writes. Run: python -u -X utf8 scripts/validate_card_suspension.py
"""
import json

import numpy as np
import pandas as pd

from card_points import DATA, K, build_frame, decayed_multiplier, trailing_level

OUT = DATA.parent / "validation" / "card_suspension_20261102"
N_BOOT, SEED = 5000, 175
STATES = ["brink_1", "brink_2", "brink_3", "two_away", "placebo"]


def prepare(count_two_yellow=False):
    d = build_frame()
    d["lvl_y"] = trailing_level(d, "yellow")
    d["lvl_r"] = trailing_level(d, "red")
    d["e_y"] = d.lvl_y * d.n90
    d["m"] = decayed_multiplier(d, "yellow", "e_y", K).clip(1e-3)
    u = pd.read_csv(DATA / "understat_match_positions.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=["Season", "Team", "Date"]).drop_duplicates()
    u["Date"] = pd.to_datetime(u.Date)
    u = u.sort_values("Date")
    u["club_match"] = u.groupby(["Season", "Team"]).cumcount() + 1
    n = len(d)
    d = d.merge(u, on=["Season", "Team", "Date"], how="left", validate="many_to_one")
    assert len(d) == n and d.club_match.notna().all()
    two = (d.CrdY.fillna(0) >= 2).astype(float) if "CrdY" in d else 0.0
    d["counted"] = d.yellow + (2 * two if count_two_yellow else 0)
    d = d.sort_values(["Date", "Team", "Player"]).reset_index(drop=True)
    # A player's count runs across clubs inside a season.
    d["season_y"] = d.groupby(["Season", "Player"]).counted.cumsum() - d.counted
    y, c = d.season_y.values, d.club_match.values
    d["brink_1"] = (y == 4) & (c <= 19)
    d["brink_2"] = (y == 9) & (c <= 32)
    d["brink_3"] = y == 14
    d["two_away"] = ((y == 3) & (c <= 19)) | ((y == 8) & (c <= 32))
    d["placebo"] = ((y == 4) & (c > 19)) | ((y == 9) & (c > 32))
    d["brink"] = d.brink_1 | d.brink_2 | d.brink_3
    d["f_y"] = d.lvl_y * d.m * d.n90
    d["pred0"] = -(d.f_y + 3 * d.lvl_r * d.n90)
    return d


def ratio_ci(g):
    """Actual / forecast yellows with a player-cluster interval."""
    u, inv = np.unique(g.Player.values, return_inverse=True)
    a, f = np.bincount(inv, g.yellow.values), np.bincount(inv, g.f_y.values)
    ix = np.random.default_rng(SEED).integers(0, len(u), size=(N_BOOT, len(u)))
    b = a[ix].sum(1) / f[ix].sum(1)
    return dict(rows=len(g), players=len(u), yellows=int(a.sum()), forecast=round(float(f.sum()), 1), ratio=round(float(a.sum() / f.sum()), 3),
                lo=round(float(np.percentile(b, 2.5)), 3), hi=round(float(np.percentile(b, 97.5)), 3),
                per90=round(float(a.sum() / g.n90.sum()), 3))


def table(d):
    return {s: ratio_ci(d[d[s]]) for s in STATES + ["brink"]} | {"other": ratio_ci(d[~d.brink & ~d.placebo & ~d.two_away])}


def cluster(prev, new, a, pl):
    se0, se1 = (a - prev) ** 2, (a - new) ** 2
    u, inv = np.unique(pl, return_inverse=True)
    t0, t1 = np.bincount(inv, se0), np.bincount(inv, se1)
    ix = np.random.default_rng(SEED).integers(0, len(u), size=(N_BOOT, len(u)))
    v = t1[ix].sum(1) / t0[ix].sum(1) - 1
    return round(float(se1.sum() / se0.sum() - 1) * 100, 4), [round(float(x) * 100, 4) for x in np.percentile(v, [2.5, 97.5])]


def state_check(d):
    """Do Sofascore's suspension lists line up with the count?"""
    m = pd.read_csv(DATA / "sofascore_missing_players.csv", encoding="utf-8-sig", low_memory=False)
    s = pd.read_csv(DATA / "sofascore_player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=["Season", "Player_ID_Sofascore", "Local_Player"]).dropna().drop_duplicates(["Season", "Player_ID_Sofascore"])
    m = m[m.Reason.isin([3, 11, 12, 13]) & (m.Type == "missing")].merge(s, on=["Season", "Player_ID_Sofascore"], how="inner")
    end = d.groupby(["Season", "Player"]).agg(total=("counted", "sum"), reds=("red", "sum")).reset_index()
    x = m.merge(end, left_on=["Season", "Local_Player"], right_on=["Season", "Player"], how="left")
    return x.groupby("Reason").agg(rows=("total", "size"), mean_season_yellows=("total", "mean"), share_5plus=("total", lambda v: float((v >= 5).mean())),
                                   mean_reds=("reds", "mean")).round(3).reset_index().to_dict("records")


def fouls_check(d):
    """Fouls per 90 in each state relative to the player's own earlier fouls per 90 in that season's previous rows."""
    q = d[d.fouls.notna()].copy()
    g = q.groupby("Player")
    q["c_f"] = g.fouls.cumsum() - q.fouls
    q["c_n"] = g.n90.cumsum() - q.n90
    q = q[q.c_n >= 10]
    q["e_f"] = q.c_f / q.c_n * q.n90
    return {s: round(float(q[q[s]].fouls.sum() / q[q[s]].e_f.sum()), 3) for s in STATES + ["brink"]} | {"other": round(float(q[~q.brink].fouls.sum() / q[~q.brink].e_f.sum()), 3)}


def main():
    d = prepare()
    print("rows", len(d), "state rows", {s: int(d[s].sum()) for s in STATES}, flush=True)
    res = {"state_check": state_check(d)}
    print(res["state_check"], flush=True)
    d = d[d.Season >= 2018].reset_index(drop=True)
    tune, test = d.Season <= 2023, d.Season >= 2024
    res["tune"], res["test"], res["all"] = table(d[tune]), table(d[test]), table(d)
    res["fouls_relative"] = fouls_check(d)
    b = d[tune & d.brink]
    theta = float(b.yellow.sum() / b.f_y.sum())
    pb = d[tune & d.placebo]
    place = float(pb.yellow.sum() / pb.f_y.sum())
    res["theta"], res["placebo_tune"] = theta, place

    def pred(t):
        f = np.where(d.brink, t, 1.0) * d.f_y
        return (-(f + 3 * d.lvl_r * d.n90)).values

    a, pl = d.pts.values, d.Player.values
    w24, w25 = (d.Season == 2024).values, (d.Season >= 2025).values
    te = test.values
    for name, t in (("theta", theta), ("theta_over_placebo", theta / place)):
        p0, p1 = pred(1.0), pred(t)
        ch, ci = cluster(p0[te], p1[te], a[te], pl[te])
        res[name + "_test"] = dict(t=round(t, 3), change_pct=ch, ci=ci,
                                   s2024=round(float(((a - p1)[w24] ** 2).sum() / ((a - p0)[w24] ** 2).sum() - 1) * 100, 4),
                                   s2025_26=round(float(((a - p1)[w25] ** 2).sum() / ((a - p0)[w25] ** 2).sum() - 1) * 100, 4))
    sens = prepare(count_two_yellow=True)
    sens = sens[sens.Season >= 2018]
    res["sensitivity_two_yellows_count"] = {k: ratio_ci(sens[sens[k]]) for k in ("brink", "placebo")}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=0, default=float))


if __name__ == "__main__":
    main()
