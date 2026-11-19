"""Session B item 6 (CHANGELOG §190): defcon team/venue/opponent averages, training v serving definition.

Protocol, frozen before scoring: data/validation/defcon_team_aggregates_20261102/protocol.json. No production writes.
Run: python -u -X utf8 -m research.defcon.evaluate_team_aggregates [extract|run|score]
"""
import os

from fpl_model import paths

os.environ.setdefault("OMP_NUM_THREADS", "2")
import json
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from fpl_model.defcon_scaling import DEFENDER_POINTS_FACTOR, nb_sf, prob_at_minutes
from research.defcon import overnight_frames as H

OUT = H.ROOT / "data" / "validation" / "defcon_team_aggregates_20261102"
H.OUT, H.FRAMES = OUT, OUT / "frames.pkl"
DATA = paths.CURRENT
YEARS = [2022, 2023, 2024, 2025, 2026]
DECAY, RATIO = 0.8, 1.067      # RECENCY_DECAY, PROMOTED_DEF_RATIO in predict.py
FEATS = {"TeamAvgStat": "Team", "VenueAvg": "Venue", "OpponentAvgStat": "Opponent"}
KEY = ["Season", "Team", "Opponent", "Venue"]
THR = {"D": 10, "M": 12, "F": 12}
ARMS = ["A", "B", "C", "D"]


def dates():
    u = pd.read_csv(paths.current_file("understat_match_positions.csv", root=DATA), encoding="utf-8-sig", low_memory=False, usecols=KEY + ["Date"]).drop_duplicates(KEY)
    u["Date"] = pd.to_datetime(u.Date)
    return u


def aggregates(f, key):
    """Per row: B (training definition, recomputed), A (serving), C (row-weighted), and the two global means."""
    q = f[f.Minutes >= 60]
    season = q.groupby(["Season", key]).Per90Stat.agg(["mean", "size"]).reset_index()
    day = q.groupby(["Season", key, "Date"]).Per90Stat.agg(["sum", "size"]).reset_index().sort_values("Date")
    day["c_s"] = day.groupby(["Season", key])["sum"].cumsum()
    day["c_n"] = day.groupby(["Season", key])["size"].cumsum()
    days = {k: g for k, g in day.groupby(["Season", key])}
    seasons = sorted(season.Season.unique())
    by = {s: g for s, g in season.groupby("Season")}
    rows = f[["Season", key, "Date"]].drop_duplicates()
    res = []
    for (s, k), g in rows.groupby(["Season", key], sort=False):
        d = days.get((s, k))
        if d is None:
            res.append(pd.DataFrame({"Season": s, key: k, "Date": g.Date.values, "cur_s": 0.0, "cur_n": 0.0}))
            continue
        # Cumulative totals strictly before each row's date.
        pos = np.searchsorted(d.Date.values, g.Date.values, side="left") - 1
        ends, ns = np.r_[0.0, d.c_s.values], np.r_[0.0, d.c_n.values]
        res.append(pd.DataFrame({"Season": s, key: k, "Date": g.Date.values, "cur_s": ends[pos + 1], "cur_n": ns[pos + 1]}))
    cur = pd.concat(res, ignore_index=True)
    pri = []
    for s in seasons:
        rec = {}
        for ps in seasons:
            if ps >= s:
                continue
            wb, wa = DECAY ** (s - 1 - ps), DECAY ** (s - ps)
            for k, m, n in by[ps][[key, "mean", "size"]].values:
                r = rec.setdefault(k, dict(bn=0., bd=0., an=0., ad=0., cn=0., cd=0.))
                r["bn"] += wb * m
                r["bd"] += wb
                r["an"] += wa * m
                r["ad"] += wa
                r["cn"] += wa * n * m
                r["cd"] += wa * n
        pri += [{"Season": s, key: k, **r} for k, r in rec.items()]
    pri = pd.DataFrame(pri)
    glob = pri.assign(ga=pri.an / pri.ad, gc=pri.cn / pri.cd).groupby("Season")[["ga", "gc"]].mean().reset_index()
    out = f[["Season", key, "Date"]].merge(cur, on=["Season", key, "Date"], how="left").merge(pri, on=["Season", key], how="left").merge(glob, on="Season", how="left")
    assert len(out) == len(f)
    z = out.fillna({c: 0.0 for c in ("cur_s", "cur_n", "bn", "bd", "an", "ad", "cn", "cd")})
    has = (z.cur_n > 0).astype(float)
    m = (z.cur_s / z.cur_n.where(z.cur_n > 0)).fillna(0.0)
    b = (z.bn / z.bd.where(z.bd > 0)).values
    den_a = z.ad + has
    a = ((z.an + m * has) / den_a.where(den_a > 0)).values
    den_c = z.cd + z.cur_n
    c = ((z.cn + z.cur_s) / den_c.where(den_c > 0)).values
    return b, a, c, out.ga.values, out.gc.values


def build(head, frame, feats, d):
    f = frame.reset_index(drop=True).merge(d, on=KEY, how="left", validate="many_to_one")
    assert len(f) == len(frame)
    lost = f.Date.isna()
    print(head, "rows", len(f), "no date", int(lost.sum()), f[lost].groupby("Season").size().to_dict(), flush=True)
    assert lost.mean() < .002
    f["Date"] = f.Date.fillna(f.groupby("Season").Date.transform("max") + pd.Timedelta(days=1))
    for col in feats:
        key = FEATS[col]
        b, a, c, ga, gc = aggregates(f, key)
        have = ~np.isnan(b)
        gap = np.abs(b[have] - f[col].values[have])
        print(head, col, "training definition recomputed: max gap", float(gap.max()), "unseen rows", int((~have).sum()), flush=True)
        assert gap.max() < 1e-6, (head, col, gap.max())
        mult = RATIO if head == "D" and col == "TeamAvgStat" else 1.0
        f[col + "_A"] = np.where(np.isnan(a), np.where(np.isnan(ga), f[col].values, ga * mult), a)
        f[col + "_C"] = np.where(np.isnan(c), np.where(np.isnan(gc), f[col].values, gc * mult), c)
    t = f[["Season", "Team", "Date"]].drop_duplicates().sort_values("Date")
    t["TeamMatchNo"] = t.groupby(["Season", "Team"]).cumcount() + 1
    f = f.merge(t, on=["Season", "Team", "Date"], how="left")
    f["Arrival"] = f.Season == f.groupby("Player").Season.transform("min")
    return f


def extract():
    fr = H.extract()
    print("frames", {k: fr[k].shape for k in ("df_def", "df_mid", "df_fwd")}, "predict md5", fr["predict_md5"], flush=True)


def load():
    fr = H.frames()
    d = dates()
    heads = {"D": (fr["df_def"], fr["DEF_FEATURE_COLS"]), "M": (fr["df_mid"], fr["MID_FEATURE_COLS"]), "F": (fr["df_fwd"], fr["FWD_FEATURE_COLS"])}
    return {h: (build(h, frame, [c for c in FEATS if c in cols], d), cols) for h, (frame, cols) in heads.items()}


def one(head, f, cols, year):
    t0 = time.monotonic()
    tr, va = f[f.Season < year], f[f.Season == year]

    def swap(fr, tag):
        return fr[[c + tag if c in FEATS else c for c in cols]].values

    base = H.fit_head(head, tr[cols].values, tr.Target.values, cols)
    out = {"p_A": H.predict(base, swap(va, "_A")), "p_B": H.predict(base, va[cols].values), "p_C": H.predict(base, swap(va, "_C"))}
    out["p_D"] = H.predict(H.fit_head(head, swap(tr, "_A"), tr.Target.values, cols), swap(va, "_A"))
    out["p_A43"] = H.predict(H.fit_head(head, tr[cols].values, tr.Target.values, cols, seed=43), swap(va, "_A"))
    print(head, year, f"{time.monotonic() - t0:.0f}s", flush=True)
    return head, year, va.index.values, out


def run():
    data = load()
    res = Parallel(n_jobs=5, backend="loky")(delayed(one)(h, f, cols, y) for h, (f, cols) in data.items() for y in YEARS)
    for h, (f, cols) in data.items():
        for name in ("p_A", "p_B", "p_C", "p_D", "p_A43"):
            f[name] = np.nan
        for hh, y, idx, o in res:
            if hh == h:
                for name, p in o.items():
                    f.loc[idx, name] = p
        v = f[f.Season.isin(YEARS)]
        keep = ["Season", "Date", "Player", "Team", "Opponent", "Venue", "Minutes", "Target", "IsForward", "Arrival", "TeamMatchNo", "MatchesPlayed", "AvgPer90", "RecentEWPer90",
                "TeamAvgStat", "TeamAvgStat_A", "TeamAvgStat_C", "VenueAvg", "VenueAvg_A", "p_A", "p_B", "p_C", "p_D", "p_A43"]
        v[[c for c in keep if c in v]].to_csv(OUT / f"predictions_{h}.csv.gz", index=False, encoding="utf-8-sig")
    print("predictions written", flush=True)


def real_hits(v, thr, head):
    p = pd.read_csv(paths.current_file("player_match_stats.csv", root=DATA), encoding="utf-8-sig", low_memory=False)
    p = p[p.Data_Source == "FPL_API"][KEY + ["Player", "Defensive_Contribution"]].drop_duplicates(KEY + ["Player"])
    d = v[v.Season >= 2025].drop_duplicates(KEY + ["Player"]).merge(p, on=KEY + ["Player"], how="inner", validate="one_to_one")
    d["hit"] = (d.Defensive_Contribution >= thr).astype(float)
    grid = np.linspace(.05, 40, 800)
    fac = DEFENDER_POINTS_FACTOR if head == "D" else 1.0
    for arm in ARMS + ["A43"]:
        mu = np.interp(d[f"p_{arm}"].values, nb_sf(grid, thr - 1), grid) * fac
        d[f"r_{arm}"] = np.clip(np.asarray(prob_at_minutes(mu, d.Minutes.values, thr), dtype=float), 1e-6, 1 - 1e-6)
    return d


def verdict(t):
    d, lo, hi = t["target"]["mean"], t["target"]["lo"], t["target"]["hi"]
    ok = (d <= -.0003 and hi < 0 and t["real"]["mean"] <= 0 and max(t["by_year"].values()) <= .0005
          and t["arrivals_target"]["mean"] <= .001 and t["arrivals_real"]["mean"] <= .001)
    if ok:
        return "PASS"
    return "TIE" if abs(d) <= .0003 and lo < 0 < hi else "FAIL"


def score():
    out, rows = {}, []
    pd.set_option("display.width", 220, "display.max_columns", 40)
    for h, thr in THR.items():
        v = pd.read_csv(OUT / f"predictions_{h}.csv.gz", encoding="utf-8-sig", low_memory=False)
        if h == "M":
            v = v[v.IsForward == 0]
        r = real_hits(v, thr, h)
        res = {"n": len(v), "real_n": len(r), "real_hits": int(r.hit.sum()),
               "levels": {a: {"p": float(v[f"p_{a}"].mean()), "actual": float(v.Target.mean()), "real_p": float(r[f"r_{a}"].mean()), "real_actual": float(r.hit.mean())} for a in ARMS},
               "feature_gap_A_vs_B": float((v.TeamAvgStat_A - v.TeamAvgStat).abs().mean()),
               "feature_gap_early": float((v.TeamAvgStat_A - v.TeamAvgStat)[v.TeamMatchNo <= 8].abs().mean()),
               "seed_noise_target": H.paired_bootstrap(v.Player.values, v.Target.values, v.p_A.values, v.p_A43.values),
               "seed_noise_real": H.paired_bootstrap(r.Player.values, r.hit.values, r.r_A.values, r.r_A43.values)}
        for arm in ARMS[1:]:
            t = {"target": H.paired_bootstrap(v.Player.values, v.Target.values, v.p_A.values, v[f"p_{arm}"].values)}
            t["by_year"] = {str(y): H.paired_bootstrap(g.Player.values, g.Target.values, g.p_A.values, g[f"p_{arm}"].values, n=2)["mean"] for y, g in v.groupby("Season")}
            e, a, ra = v[v.TeamMatchNo <= 8], v[v.Arrival], r[r.Arrival]
            t["early"] = H.paired_bootstrap(e.Player.values, e.Target.values, e.p_A.values, e[f"p_{arm}"].values)
            t["arrivals_target"] = H.paired_bootstrap(a.Player.values, a.Target.values, a.p_A.values, a[f"p_{arm}"].values)
            t["arrivals_means"] = {"n": len(a), "A": float(a.p_A.mean()), arm: float(a[f"p_{arm}"].mean()), "actual": float(a.Target.mean())}
            t["real"] = H.paired_bootstrap(r.Player.values, r.hit.values, r.r_A.values, r[f"r_{arm}"].values)
            t["arrivals_real"] = H.paired_bootstrap(ra.Player.values, ra.hit.values, ra.r_A.values, ra[f"r_{arm}"].values)
            t["real_points_mse"] = {"A": float(((2 * r.r_A - 2 * r.hit) ** 2).mean()), arm: float(((2 * r[f"r_{arm}"] - 2 * r.hit) ** 2).mean())}
            t["verdict"] = verdict(t)
            res[arm] = t
            rows.append(dict(head=h, arm=arm, d_ll=t["target"]["mean"], lo=t["target"]["lo"], hi=t["target"]["hi"], early=t["early"]["mean"], real=t["real"]["mean"],
                             r_lo=t["real"]["lo"], r_hi=t["real"]["hi"], arr=t["arrivals_target"]["mean"], arr_r=t["arrivals_real"]["mean"], worst_yr=max(t["by_year"].values()), verdict=t["verdict"]))
        out[h] = res
        print(h, {k: res[k] for k in ("n", "real_n", "real_hits", "feature_gap_A_vs_B", "feature_gap_early", "seed_noise_target", "seed_noise_real")}, flush=True)
    print(pd.DataFrame(rows).round(5).to_string())
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    {"extract": extract, "run": run, "score": score}[sys.argv[1]]()
