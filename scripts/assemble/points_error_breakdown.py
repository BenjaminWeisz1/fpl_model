"""Item 4, AGENDA 2026-10-31 (CHANGELOG §152): per-component points error, against what chance alone would give.

For each player-match: pre-match ratings and team forecasts, the realized minutes and the season's realized penalty share.
Each component's error is set against the variance its own forecast implies, so noise is not read as model error.
Rows: the pre-match bonus hindcast, 2018-2024 (vaastav outcomes) and its 2025-26 windows. Defcon uses the §149-§150
out-of-time probabilities, 2022-2026. Diagnostic only; no production writes.
Run: python -u -X utf8 scripts/assemble/points_error_breakdown.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from defcon_scaling import nb_sf, prob_at_minutes  # noqa: E402

DATA = ROOT / "data" / "current"
VAL = ROOT / "data" / "validation"
OUT = VAL / "points_error_20261031"
OUT.mkdir(exist_ok=True)
DEFCON = ROOT / "data" / "validation" / "defcon_overnight_20261031"
PTS_GOAL = {"G": 6, "D": 6, "M": 5, "F": 4}     # the rule in force in these seasons, so components sum to total_points
CS = {"G": 4, "D": 4, "M": 1, "F": 0}
PEN_RATE, PEN_CONV = 792 / 7734, 0.79
KEY = ["Season", "Player", "Team", "Opponent", "Venue"]


def history_rows():
    h = pd.read_csv(VAL / "prematch_bonus_history_20261101" / "hindcast_predictions.csv", encoding="utf-8-sig", low_memory=False)  # identities repaired, §167
    v = pd.read_csv(DATA / "vaastav_player_match_history.csv", encoding="utf-8-sig", low_memory=False)
    v["Match_Key"] = v.Season.astype(str) + "_" + v.Fixture_FPL.astype(str)
    v["Player"] = v.Local_Player.fillna(v.Name_FPL)
    v = v.rename(columns={"minutes": "Min2"})
    pms = pd.read_csv(DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False, usecols=KEY + ["PK", "PKatt"]).drop_duplicates(KEY)
    v = v.merge(pms, on=KEY, how="left")
    v[["PK", "PKatt"]] = v[["PK", "PKatt"]].fillna(0)
    cols = ["Match_Key", "Player", "Team", "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves", "penalties_saved",
            "penalties_missed", "own_goals", "yellow_cards", "red_cards", "total_points", "PK", "PKatt"]
    k = ["Match_Key", "Player", "Team"]
    dup = int(h.duplicated(k, keep=False).sum())  # same-name teammates cannot be told apart here, so they are dropped
    print("duplicate name rows dropped:", dup)
    h = h[~h.duplicated(k, keep=False)]
    r = h.merge(v[cols].drop_duplicates(k), on=k, how="left", validate="one_to_one")
    assert r.total_points.notna().mean() > .99, r.total_points.notna().mean()
    r = r[r.total_points.notna()].copy()
    r["NPG"] = (r.goals_scored - r.PK).clip(lower=0)
    return r


def recent_rows():
    h = pd.read_csv(VAL / "prematch_bonus_rerun_20261028" / "hindcast_predictions.csv", encoding="utf-8-sig", low_memory=False)
    p = pd.read_csv(DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    p = p[p.Data_Source == "FPL_API"].drop_duplicates(KEY)
    r = h.merge(p[KEY + ["Clean_Sheet", "Goals_Conceded", "Saves", "PK", "PKatt", "Own_Goals", "CrdY", "CrdR", "Penalties_Saved", "Penalties_Missed"]],
                on=KEY, how="left", validate="one_to_one")
    r = r.rename(columns={"Clean_Sheet": "clean_sheets", "Goals_Conceded": "goals_conceded", "Saves": "saves", "Assists": "assists",
                          "Own_Goals": "own_goals", "CrdY": "yellow_cards", "CrdR": "red_cards", "Penalties_Saved": "penalties_saved",
                          "Penalties_Missed": "penalties_missed"})
    r["Match_Key"] = r.Season.astype(str) + "_" + r[["Team", "Opponent"]].min(axis=1) + "_" + r[["Team", "Opponent"]].max(axis=1) + "_" + r.Round.astype(str)
    r["total_points"] = np.nan
    return r


def floor_moments(lam, k):
    """E and Var of floor(X / k), X ~ Poisson(lam), on a grid."""
    grid = np.linspace(0, 12, 601)
    xs = np.arange(0, 45)
    pm = poisson.pmf(xs[None, :], grid[:, None])
    f = xs // k
    e = (pm * f).sum(1)
    v = (pm * f ** 2).sum(1) - e ** 2
    return np.interp(lam, grid, e), np.interp(lam, grid, v)


def actual_penalty_share(r):
    pk = pd.read_csv(DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False, usecols=["Season", "Player", "Team", "PKatt"])
    t = pk.groupby(["Season", "Team", "Player"]).PKatt.sum()
    return (t / t.groupby(level=[0, 1]).transform("sum")).fillna(0)


def components(r):
    r = r.copy()
    m = r.Minutes.values / 90.0
    full = (r.Minutes.values >= 60).astype(float)
    goal_pts, cs_pts = r.Pos.map(PTS_GOAL).values.astype(float), r.Pos.map(CS).values.astype(float)
    c = {}
    lam = np.clip(r.NPG90_scaled.values, 0, None) * m
    c["goals"] = (lam * goal_pts, r.NPG.values * goal_pts, lam * goal_pts ** 2)
    lam = np.clip(r.Ast90_scaled.values, 0, None) * m
    c["assists"] = (lam * 3, r.assists.values * 3, lam * 9)

    share = actual_penalty_share(r)
    s = share.reindex(pd.MultiIndex.from_arrays([r.Season, r.Team, r.Player])).fillna(0).values
    att = r.Team_Pred_NPG.values * PEN_RATE * s
    per = PEN_CONV * goal_pts - (1 - PEN_CONV) * 2
    pk, miss = r.PK.fillna(0).values, (r.PKatt.fillna(0).values - r.PK.fillna(0).values).clip(0)
    second = PEN_CONV * goal_pts ** 2 + (1 - PEN_CONV) * 4
    c["penalties"] = (att * per, pk * goal_pts - miss * 2, att * second)

    egc = r.Opp_Pred_NPG.values * (1 + PEN_RATE * PEN_CONV)
    pcs = np.exp(-egc)
    is_cs = (r.clean_sheets.values >= 1).astype(float)
    c["clean_sheet"] = (pcs * full * cs_pts, is_cs * cs_pts * full, pcs * (1 - pcs) * cs_pts ** 2 * full)
    defn = r.Pos.isin(["D", "G"]).values.astype(float) * full
    e, v = floor_moments(egc * m, 2)
    gc = r.goals_conceded.fillna(0).values
    c["goals_conceded"] = (-e * defn, -np.floor(gc / 2) * defn, v * defn)
    keeper = r.Pos.eq("G").values.astype(float)
    rate = float(r.saves[r.Pos.eq("G")].sum() / (r.Minutes[r.Pos.eq("G")].sum() / 90))
    e, v = floor_moments(rate * m, 3)
    c["saves"] = (e * keeper, np.floor(r.saves.fillna(0).values / 3) * keeper, v * keeper)

    pb = r.pred_coherent.values.astype(float)
    any_ = np.clip(r.pany_coherent.values.astype(float), 1e-6, 1)
    mean_any = np.clip(pb / any_, 1, 3)
    # Spread of the bonus awarded given any bonus: read from outcomes by forecast band; a calibrated forecast shares it.
    band = pd.qcut(pd.Series(mean_any), 8, duplicates="drop")
    b = pd.Series(r.Bonus.values.astype(float))
    spread = b[b > 0].groupby(band[b.values > 0], observed=True).var().to_dict()
    sv = band.map(spread).astype(float).fillna(.6).values
    c["bonus"] = (pb, b.values, any_ * (mean_any ** 2 + sv) - pb ** 2)

    other = (r.penalties_saved.fillna(0).values * 5 - r.yellow_cards.fillna(0).values - 3 * r.red_cards.fillna(0).values - 2 * r.own_goals.fillna(0).values)
    c["cards_other_unmodelled"] = (np.zeros(len(r)), other, np.zeros(len(r)))
    app = (r.Minutes.values > 0).astype(float) + full
    return c, app


def add_defcon(r):
    """2 points per threshold hit; the out-of-time probability re-scaled to the realized minutes, as production does."""
    fr = []
    for h, col, f in (("D", "p_control", "item2_predictions_D.csv"), ("M", "p_count", "item2_predictions_M.csv"), ("F", "p_control", "item2_predictions_F.csv")):
        d = pd.read_csv(DEFCON / f, encoding="utf-8-sig")
        fr.append(d[["Season", "Player", "Team", "Opponent", "Venue", "Target", col]].rename(columns={col: "p90"}).assign(Head=h))
    d = pd.concat(fr, ignore_index=True)
    z = pd.read_pickle(DEFCON / "item1_frames.pkl")["data"]
    raw = pd.concat([v[["Season", "Player", "Team", "Opponent", "Venue", "RawStat"]].assign(Head=k) for k, v in z.items()], ignore_index=True)
    d = d.merge(raw, on=KEY + ["Head"], how="left", validate="one_to_one")
    d = d.drop_duplicates(KEY + ["Head"])
    return r.assign(Head=r.Pos).merge(d, on=KEY + ["Head"], how="left", validate="many_to_one")


def invert_vec(p, thr):
    grid = np.linspace(.05, 40, 800)
    sf = nb_sf(grid, thr - 1)
    return np.interp(p, sf, grid)


def defcon_component(r):
    v = r[r.p90.notna() & r.Pos.isin(["D", "M", "F"])].copy()
    thr = np.where(v.Pos == "D", 10, 12)
    mu = np.array([invert_vec(np.array([p]), t)[0] for p, t in zip(v.p90.values, thr)])
    p = prob_at_minutes(mu, v.Minutes.values, thr)
    p = np.asarray(p, dtype=float)
    return v, (2 * p, 2 * (v.RawStat.values >= thr).astype(float), 4 * p * (1 - p))


def table(c, clusters, index=None, reps=500):
    rows = []
    rng = np.random.default_rng(1)
    codes, _ = pd.factorize(clusters)
    k = codes.max() + 1
    for name, (pred, act, var) in c.items():
        err2 = (act - pred) ** 2
        ex = err2 - var
        s = np.bincount(codes, ex, k)
        n = np.bincount(codes, minlength=k).astype(float)
        w = rng.multinomial(k, np.ones(k) / k, size=reps)
        boot = (w * s).sum(1) / (w * n).sum(1)
        rows.append({"component": name, "n": len(act), "pred": pred.mean(), "actual": act.mean(), "bias": (pred - act).mean(),
                     "mse": err2.mean(), "chance": var.mean(), "excess": ex.mean(), "lo": np.quantile(boot, .025), "hi": np.quantile(boot, .975),
                     "excess_share": ex.mean() / err2.mean() if err2.mean() > 0 else np.nan})
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 200, "display.max_columns", 30)
    hist, rec = history_rows(), recent_rows()
    hist["Block"], rec["Block"] = "2018-2024", "2025-26 windows"
    out = {}
    for name, r in (("2018-2024", hist), ("2025-26 windows", rec)):
        c, app = components(r.reset_index(drop=True))
        r = r.reset_index(drop=True)
        tab = table(c, r.Match_Key.values)
        total_p = sum(v[0] for v in c.values())
        total_a = sum(v[1] for v in c.values())
        print(f"\n=== {name}: {len(r)} player-matches ===")
        print(tab.round(4).to_string())
        if r.total_points.notna().any():
            gap = (total_a + app - r.total_points.values)
            print("Components + appearance vs FPL total_points: exact on", round(float((np.abs(gap) < 1e-9).mean()), 4), "mean gap", round(float(gap.mean()), 4))
        print("Total error (excl. appearance): MSE", round(float(((total_a - total_p) ** 2).mean()), 3), "sum of component chance", round(float(sum(v[2] for v in c.values()).mean()), 3),
              "bias", round(float((total_p - total_a).mean()), 4))
        out[name] = {"all": tab.to_dict("records")}
        by = {}
        for pos, g in r.groupby("Pos"):
            sub = {k: (v[0][g.index], v[1][g.index], v[2][g.index]) for k, v in c.items()}
            by[pos] = table(sub, g.Match_Key.values).to_dict("records")
        out[name]["by_position"] = by
        d = pd.concat([pd.DataFrame(v) for v in by.values()], keys=by.keys()).reset_index(level=0).rename(columns={"level_0": "Pos"})
        print(d.pivot(index="component", columns="Pos", values="excess").round(4).to_string())
    both = pd.concat([hist[hist.Season >= 2022], rec], ignore_index=True)
    both = add_defcon(both)
    v, comp = defcon_component(both)
    for label, mask in (("2022-2024", v.Season <= 2024), ("2025-2026 windows", v.Season >= 2025)):
        m = mask.values
        tab = table({"defcon": tuple(x[m] for x in comp)}, v.Match_Key.values[m])
        print(f"\nDefcon {label}, scaled to realized minutes"); print(tab.round(4).to_string())
        out[f"defcon {label}"] = tab.to_dict("records")
        for pos, g in v[mask].groupby("Pos"):
            i = (v.index.get_indexer(g.index))
            print(pos, table({"defcon": tuple(x[i] for x in comp)}, g.Match_Key.values).round(4).drop(columns="component").to_string(header=False))
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
