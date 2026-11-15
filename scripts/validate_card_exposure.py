"""Card exposure, starts v cameos (CHANGELOG §193). Protocol: data/validation/card_exposure_20261102/protocol.json.
No production writes. Run: python -u -X utf8 scripts/validate_card_exposure.py
"""
import json

import numpy as np
import pandas as pd

from card_points import DATA, K, KEYS, build_frame, decayed_multiplier, trailing_level

OUT = DATA.parent / "validation" / "card_exposure_20261102"
N_BOOT, SEED = 5000, 175


def prepare():
    d = build_frame()
    s = pd.read_csv(DATA / "sofascore_player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=["Season", "Local_Player", "Team", "Opponent", "Venue", "Is_Starter"]).rename(columns={"Local_Player": "Player"})
    s = s.dropna(subset=["Player"]).drop_duplicates(KEYS)
    n = len(d)
    d = d.merge(s, on=KEYS, how="left")
    assert len(d) == n
    print("role coverage", round(float(d.Is_Starter.notna().mean()), 4), flush=True)
    d["start"] = d.Is_Starter.astype("boolean").fillna(d.Minutes >= 46).astype(bool)
    d["role"] = np.where(d.start, "S", "C")
    d["lvl_y"] = trailing_level(d, "yellow")
    d["lvl_r"] = trailing_level(d, "red")
    d["lvl_ys"] = trailing_level(d, "yellow", by=("group", "role"))
    # Both role levels at every row's date, so a forecast can use either.
    for role, col in (("S", "lvl_S"), ("C", "lvl_C")):
        day = d[d.role == role].groupby(["group", "Season", "Date"]).lvl_ys.first().reset_index().rename(columns={"lvl_ys": col})
        d = d.sort_values("Date").reset_index(drop=True)
        d = pd.merge_asof(d, day.sort_values("Date"), on="Date", by=["group", "Season"], direction="backward")
    d = d.sort_values(["Date", "Team", "Player"]).reset_index(drop=True)
    d["lvl_S"] = d.groupby(["group", "Season"]).lvl_S.transform(lambda v: v.bfill())
    d["lvl_C"] = d.groupby(["group", "Season"]).lvl_C.transform(lambda v: v.bfill())
    d["e_y"] = d.lvl_y * d.n90
    d["e_r"] = d.lvl_ys * d.n90
    d["m0"] = decayed_multiplier(d, "yellow", "e_y", K).clip(1e-3)
    d["m1"] = decayed_multiplier(d, "yellow", "e_r", K).clip(1e-3)
    return d[d.Season >= 2018].reset_index(drop=True)


def panel(d):
    u = pd.read_csv(DATA / "understat_match_positions.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=["Season", "Team", "Date"]).drop_duplicates()
    u["Date"] = pd.to_datetime(u.Date)
    span = d.groupby(["Season", "Team", "Player"]).Date.agg(first="min", last="max").reset_index()
    p = span.merge(u, on=["Season", "Team"], how="left")
    p = p[(p.Date >= p["first"]) & (p.Date <= p["last"])]
    cols = ["Season", "Team", "Player", "Date", "n90", "pts", "yellow", "red", "start", "group", "lvl_y", "lvl_r", "lvl_S", "lvl_C", "m0", "m1", "Minutes"]
    p = p.merge(d[cols], on=["Season", "Team", "Player", "Date"], how="left").sort_values(["Season", "Team", "Player", "Date"]).reset_index(drop=True)
    print("panel rows", len(p), "played share", round(float(p.Minutes.notna().mean()), 3), flush=True)
    p["played"] = p.Minutes.notna()
    g = ["Season", "Team", "Player"]
    for c in ("group", "lvl_y", "lvl_r", "lvl_S", "lvl_C", "m0", "m1"):
        p[c] = p.groupby(g)[c].ffill()
    for c in ("pts", "yellow", "n90"):
        p[c] = p[c].fillna(0.0)
    p["st"] = p.start.fillna(False).astype(bool)
    p["cm"] = p.played & ~p.st
    p["mS"] = p.Minutes.where(p.st, 0.0).fillna(0.0)
    p["mC"] = p.Minutes.where(p.cm, 0.0).fillna(0.0)
    r = p.groupby(g, sort=False)
    for c in ("mS", "mC"):
        p["t" + c] = r[c].transform(lambda v: v.shift(1).rolling(5, min_periods=3).mean())
    return p[p.tmS.notna()].reset_index(drop=True)


def cluster(prev, new, a, pl):
    se0, se1 = (a - prev) ** 2, (a - new) ** 2
    u, inv = np.unique(pl, return_inverse=True)
    t0, t1 = np.bincount(inv, se0), np.bincount(inv, se1)
    ix = np.random.default_rng(SEED).integers(0, len(u), size=(N_BOOT, len(u)))
    v = t1[ix].sum(1) / t0[ix].sum(1) - 1
    return round(float(se1.sum() / se0.sum() - 1) * 100, 4), [round(float(x) * 100, 4) for x in np.percentile(v, [2.5, 97.5])]


def windows(name, seasons, a, p0, p1, pl):
    te, tu = seasons >= 2024, seasons <= 2023
    w = lambda m: round(float(((a - p1)[m] ** 2).sum() / ((a - p0)[m] ** 2).sum() - 1) * 100, 4)
    ch, ci = cluster(p0[te], p1[te], a[te], pl[te])
    r = dict(change_pct=ch, ci=ci, s2024=w(seasons == 2024), s2025_26=w(seasons >= 2025), s2018_23=w(tu), mean_actual=float(a[te].mean()), mean0=float(p0[te].mean()), mean1=float(p1[te].mean()))
    r["verdict"] = "PASS" if ci[1] < 0 and r["s2024"] < 0 and r["s2025_26"] < 0 and r["s2018_23"] < 0 else "FAIL"
    print(name, r, flush=True)
    return r


def band_table(d, f0, f1):
    q = d.assign(f0=f0, f1=f1)
    q["band"] = pd.cut(q.Minutes, [0, 15, 30, 60, 89, 200], labels=["1-15", "16-30", "31-60", "61-89", "90"])
    keys = {"band": q.groupby("band", observed=True), "role": q.groupby("role")}
    return {k: g.apply(lambda x: pd.Series(dict(rows=len(x), yellows=x.yellow.sum(), per90=x.yellow.sum() / x.n90.sum(), ratio_control=x.yellow.sum() / x.f0.sum(), ratio_split=x.yellow.sum() / x.f1.sum())),
                       include_groups=False).round(3).reset_index().to_dict("records") for k, g in keys.items()}


def snapshot_gw8(d):
    """One-gameweek check of the clock: last pre-deadline GW8 snapshot (as §177), one rate. Reported, not a gate."""
    snap = "GW08/20261022T170038_665091Z_partall"
    x = pd.read_csv(DATA.parent / "snapshots" / "2026" / snap / "gameweek" / "simulation_features.csv", encoding="utf-8-sig")
    x = x[x.xMins.fillna(0) > 0][["Player", "xMins"]].groupby("Player", as_index=False).xMins.sum()
    r = d[d.Season == 2026]
    last = r.Date.max()
    gw8 = r[r.Date >= last - pd.Timedelta(days=5)]
    prev = r[r.Date < gw8.Date.min()].sort_values("Date").groupby("Player").tail(1).assign(rate=lambda q: q.lvl_y * q.m0)[["Player", "rate"]]
    j = x.merge(prev, on="Player", how="left")
    raw = float((j.rate * j.xMins / 90).sum())
    return dict(snapshot=snap, players=len(j), with_rate=int(j.rate.notna().sum()), yellows_actual=int(gw8.yellow.sum()), raw=raw, recorded=raw * 990 / 1043,
                realized_minutes=float((gw8.lvl_y * gw8.m0 * gw8.n90).sum()))


def mix_table(p, r0, r1):
    """Actual / forecast by how much of the trailing minutes came from cameos, 2024 on."""
    q = p.assign(r0=r0, r1=r1)
    q = q[q.Season >= 2024]
    tot = (q.tmS + q.tmC) / 5
    q["mix"] = pd.cut(q.tmC / (q.tmS + q.tmC).where(q.tmS + q.tmC > 0), [-.01, 0, .1, .5, 1.01], labels=["starts only", "<10% cameo", "10-50%", ">50%"])
    y = -q.pts
    return q.assign(y=y).groupby("mix", observed=True).apply(lambda g: pd.Series(dict(rows=len(g), ratio_control=g.y.sum() / g.r0.sum(), ratio_split=g.y.sum() / g.r1.sum())),
                                                             include_groups=False).round(3).reset_index().to_dict("records")


def main():
    d = prepare()
    res = {}
    f0 = (d.lvl_y * d.m0 * d.n90).values
    f1 = (d.lvl_ys * d.m1 * d.n90).values
    reds = (3 * d.lvl_r * d.n90).values
    res["levels_2024_on"] = d[d.Season >= 2024].groupby(["group", "role"]).apply(
        lambda x: pd.Series(dict(rows=len(x), per90=x.yellow.sum() / x.n90.sum())), include_groups=False).round(3).reset_index().to_dict("records")
    res["bands"] = band_table(d, f0, f1)
    res["bands_test"] = band_table(d[d.Season >= 2024], f0[d.Season >= 2024], f1[d.Season >= 2024])
    res["R"] = windows("R", d.Season.values, d.pts.values, -(f0 + reds), -(f1 + reds), d.Player.values)
    p = panel(d)
    x0 = (p.tmS + p.tmC) / 90
    r0 = (p.lvl_y * p.m0 * x0 + 3 * p.lvl_r * x0).values
    r1 = (p.m1 * (p.lvl_S * p.tmS + p.lvl_C * p.tmC) / 90 + 3 * p.lvl_r * x0).values
    keep = np.isfinite(r0) & np.isfinite(r1)
    print("panel usable", int(keep.sum()), "of", len(p), flush=True)
    p, r0, r1 = p[keep], r0[keep], r1[keep]
    res["F"] = windows("F", p.Season.values, p.pts.values, -r0, -r1, p.Player.values)
    reg = p[p.Season >= 2024]
    regular = (reg.tmS + reg.tmC) >= 75
    res["regular_charge_change_per_match"] = float((r1[p.Season.values >= 2024][regular.values] - r0[p.Season.values >= 2024][regular.values]).mean())
    res["regular_charge_share"] = float(r1[p.Season.values >= 2024][regular.values].sum() / r0[p.Season.values >= 2024][regular.values].sum() - 1)
    res["F_mix"] = mix_table(p, r0, r1)
    res["gw8_clock"] = snapshot_gw8(d)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=0, default=float))


if __name__ == "__main__":
    main()
