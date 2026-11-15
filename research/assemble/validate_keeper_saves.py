"""CHANGELOG §162: keeper saves by fixture and by team, against production's single pooled rate.

Walk-forward test, rate error at realized minutes. Test only; no production writes.
Run: python -u -X utf8 scripts/assemble/validate_keeper_saves.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import gammaln

from fpl_model import paths

HERE = Path(__file__).resolve().parent
ROOT = paths.ROOT
from fpl_model.assemble.build_predicted_points import expected_points_per_three_saves  # noqa: E402

DATA = paths.CURRENT
OUT = paths.VALIDATION / "keeper_saves_20261101"
OUT.mkdir(parents=True, exist_ok=True)
KEY = ["Season", "Team", "Opponent", "Venue"]
TEST = list(range(2019, 2025))
LATER = [2025, 2026]
K_GRID = [10, 20, 40, 80, 160]
WINDOW = 38
REPS = 2000


def team_matches():
    H = pd.read_csv(DATA / "team_ratings_history.csv", encoding="utf-8-sig", low_memory=False)
    k = ["Season", "MatchID", "Team", "Opponent", "Venue", "T", "O"]
    a = H.assign(Team=H.Home, Opponent=H.Away, Venue="Home", T=H.Home_Prediction, O=H.Away_Prediction)[k]
    b = H.assign(Team=H.Away, Opponent=H.Home, Venue="Away", T=H.Away_Prediction, O=H.Home_Prediction)[k]
    return pd.concat([a, b], ignore_index=True)


def keeper_rows():
    v = pd.read_csv(DATA / "vaastav_player_match_history.csv", encoding="utf-8-sig", low_memory=False)
    v = v.assign(Player=v.Local_Player.fillna(v.Name_FPL), Minutes=v.minutes, Saves=v.saves)
    p = pd.read_csv(DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    p = p[p.Data_Source == "FPL_API"]
    r = pd.concat([v[KEY + ["Player", "Minutes", "Saves"]], p[KEY + ["Player", "Minutes", "Saves"]]], ignore_index=True)
    r = r.drop_duplicates(KEY + ["Player"])
    pos = pd.read_csv(DATA / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False)
    pos = pos[KEY + ["Player", "Position_Resolved"]].drop_duplicates(KEY + ["Player"])
    r = r.merge(pos, on=KEY + ["Player"], how="left", validate="one_to_one")
    cur = pd.read_csv(DATA / "players_current.csv", encoding="utf-8-sig")
    cur = set(cur.loc[cur.Pos == "G", "Player"])
    gk = r.Position_Resolved.eq("GK") | r.Saves.fillna(0).gt(0) | (r.Season.ge(2025) & r.Player.isin(cur))
    r = r[gk & r.Minutes.gt(0) & r.Saves.notna()].copy()
    return r


def frame():
    r, t = keeper_rows(), team_matches()
    x = r.merge(t, on=KEY, how="left", validate="many_to_one")
    assert x.MatchID.notna().all(), x[x.MatchID.isna()].head()
    have = x.groupby(["MatchID", "Team"]).size()
    print("team-matches with a keeper row", len(have), "of", len(t), "rows", len(x))
    assert len(have) / len(t) > .995, "keeper rows missing"
    minutes = x.groupby(["MatchID", "Team"]).Minutes.sum()
    print("team-match keeper minutes: under 80 in", int((minutes < 80).sum()))
    x["Pts"] = np.floor(x.Saves / 3)
    x["Exposure"] = x.Minutes / 90
    # Pooled rate of all earlier matches. The same match never informs itself.
    m = x.groupby("MatchID").agg(s=("Saves", "sum"), e=("Exposure", "sum")).sort_index()
    x["P"] = x.MatchID.map(m.s.cumsum().shift(1) / m.e.cumsum().shift(1))
    assert x[x.Season >= 2018].P.notna().all()
    seasons = t.groupby("Season").Team.agg(set).to_dict()
    tm = t[["Season", "MatchID", "Team"]].sort_values("MatchID").copy()
    tm["New"] = [s not in seasons or (s - 1 in seasons and team not in seasons[s - 1]) for s, team in zip(tm.Season, tm.Team)]
    first = tm.groupby(["Team", "Season"]).MatchID.transform("min") == tm.MatchID
    tm["Spell"] = (first & tm.New).astype(int).groupby(tm.Team).cumsum()
    return x.merge(tm[["MatchID", "Team", "Spell"]], on=["MatchID", "Team"], how="left", validate="many_to_one")


def fit_fixture(x, s):
    tr = x[x.Season < s]
    m = sm.GLM(tr.Saves, sm.add_constant(np.log(tr.O).rename("l")), family=sm.families.Poisson(), offset=np.log(tr.Exposure)).fit()
    beta, se = float(m.params["l"]), float(m.bse["l"])
    ref = float(np.average(tr.O ** beta, weights=tr.Exposure))
    return beta, se, ref


def team_record(x, rate):
    """The team's saves S and expected saves E over its previous matches, this match excluded."""
    d = x.assign(_e=rate * x.Exposure).groupby(["Team", "Spell", "MatchID"], as_index=False).agg(S=("Saves", "sum"), E=("_e", "sum")).sort_values(["Team", "Spell", "MatchID"])
    g = d.groupby(["Team", "Spell"])
    for c in "SE":
        d[c] = g[c].transform(lambda s: s.rolling(WINDOW, min_periods=1).sum().shift(1)).fillna(0.0)
    return x[["MatchID", "Team"]].merge(d, on=["MatchID", "Team"], how="left", validate="many_to_one")[["S", "E"]].to_numpy().T


def poisson_ll(y, lam):
    return y * np.log(lam) - lam - gammaln(y + 1)


def choose_k(x, base, s):
    S, E = team_record(x, base)
    use = ((x.Season < s) & (x.Season >= 2018)).values
    scores = {k: float(poisson_ll(x.Saves.values[use], (base * (S + k) / (E + k) * x.Exposure).values[use]).mean()) for k in K_GRID}
    return max(scores, key=scores.get), S, E


def walk_forward(x):
    rows, fits = [], []
    for s in TEST + LATER:
        beta, se, ref = fit_fixture(x, s)
        y = x[x.Season <= s].reset_index(drop=True)
        ctrl = y.P.copy()
        a = y.P * y.O ** beta / ref
        kb, Sb, Eb = choose_k(y, ctrl, s)
        kc, Sc, Ec = choose_k(y, a, s)
        b = ctrl * (Sb + kb) / (Eb + kb)
        c = a * (Sc + kc) / (Ec + kc)
        t = y[y.Season == s].copy()
        idx = t.index
        t["r_ctrl"], t["r_a"], t["r_b"], t["r_c"] = ctrl[idx], a[idx], b[idx], c[idx]
        rows.append(t)
        fits.append({"Season": s, "beta": beta, "beta_se": se, "k_b": kb, "k_c": kc})
        print(fits[-1])
    return pd.concat(rows, ignore_index=True), pd.DataFrame(fits)


def add_points(t):
    for arm in ("ctrl", "a", "b", "c"):
        lam = (t["r_" + arm] * t.Exposure).values
        t["pts_" + arm] = expected_points_per_three_saves(lam)
        t["ll_" + arm] = poisson_ll(t.Saves.values, lam)
    t["Strength"] = np.log(t["T"] / t.O)
    t["Q"] = pd.qcut(t.Strength, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    t["Cluster"] = t.Team + "_" + t.Season.astype(str)
    return t


def paired(t, base, arm, rng_seed=1):
    """Share change in MSE and LL change, with a paired team-season bootstrap."""
    codes, _ = pd.factorize(t.Cluster)
    k = codes.max() + 1
    w = np.random.default_rng(rng_seed).multinomial(k, np.ones(k) / k, size=REPS)
    e0, e1 = (t.Pts - t["pts_" + base]) ** 2, (t.Pts - t["pts_" + arm]) ** 2
    s0, s1 = np.bincount(codes, e0.values, k), np.bincount(codes, e1.values, k)
    n = np.bincount(codes, minlength=k).astype(float)
    pct = 100 * ((w * s1).sum(1) / (w * s0).sum(1) - 1)
    l = np.bincount(codes, (t["ll_" + arm] - t["ll_" + base]).values, k)
    dl = (w * l).sum(1) / (w * n).sum(1)
    return {"mse_base": float(e0.mean()), "mse_arm": float(e1.mean()), "mse_pct": float(100 * (e1.mean() / e0.mean() - 1)), "mse_lo": float(np.quantile(pct, .025)), "mse_hi": float(np.quantile(pct, .975)),
            "ll_gain": float((t["ll_" + arm] - t["ll_" + base]).mean()), "ll_lo": float(np.quantile(dl, .025)), "ll_hi": float(np.quantile(dl, .975))}


def spread(t, arm):
    b = (t.Pts - t["pts_" + arm]).groupby(t.Q, observed=True).mean()
    return {"bias": b.round(4).to_dict(), "gap": float(b["Q4"] - b["Q1"]), "mean_bias": float((t.Pts - t["pts_" + arm]).mean())}


def rule(t, later, arm):
    p = paired(t, "ctrl", arm)
    by = {int(s): float(((g.Pts - g["pts_" + arm]) ** 2).mean() - ((g.Pts - g.pts_ctrl) ** 2).mean()) for s, g in t.groupby("Season")}
    s0, s1 = spread(t, "ctrl"), spread(t, arm)
    l = paired(later, "ctrl", arm)
    checks = {"i_mse": p["mse_hi"] < 0, "ii_seasons": sum(v < 0 for v in by.values()) >= 4, "iii_ll": p["ll_lo"] > 0, "iv_gap": abs(s1["gap"]) <= abs(s0["gap"]) / 2,
              "v_level": abs(s1["mean_bias"]) <= abs(s0["mean_bias"]) + .01, "vi_later": l["mse_pct"] <= 0}
    return {"pooled": p, "by_season": by, "spread": s1, "later": l, "checks": checks, "passes": all(checks.values())}


def diagnostics(t, x):
    out = {}
    for arm in ("ctrl", "a", "c"):
        lam = t["r_" + arm] * t.Exposure
        out["dispersion_" + arm] = float((((t.Saves - lam) ** 2) / lam).mean())
    full = t[t.Minutes >= 60]
    b = pd.DataFrame({"pred_ctrl": full.pts_ctrl, "pred_a": full.pts_a, "pred_c": full.pts_c, "act": full.Pts}).groupby(pd.qcut(full.pts_a, 5, labels=False)).mean()
    out["calibration_by_a_quintile"] = b.round(4).to_dict("list")
    ts = t.assign(e=t.r_a * t.Exposure).groupby(["Team", "Season"]).agg(S=("Saves", "sum"), E=("e", "sum")).reset_index()
    ts["R"] = np.log(ts.S / ts.E)
    nxt = ts.merge(ts.assign(Season=ts.Season - 1), on=["Team", "Season"], suffixes=("", "2"))
    nxt = nxt[nxt.Season < 2025]
    out["season_residual_r"] = float(nxt.R.corr(nxt.R2))
    out["season_residual_pairs"] = len(nxt)
    early = t[t.groupby(["Team", "Season"]).MatchID.rank(method="dense") <= 5]
    out["first_five_matches"] = {arm: paired(early, "ctrl", arm) for arm in ("a", "c")}
    return out


def main():
    pd.set_option("display.width", 200, "display.max_columns", 30)
    x = frame()
    t, fits = walk_forward(x)
    t = add_points(t)
    main_rows, later = t[t.Season.isin(TEST)].reset_index(drop=True), t[t.Season.isin(LATER)].reset_index(drop=True)
    print("rows", len(main_rows), "later", len(later))
    res = {"rows": len(main_rows), "later_rows": len(later), "fits": fits.to_dict("records"), "control_spread": spread(main_rows, "ctrl")}
    for arm in "abc":
        res[arm] = rule(main_rows, later, arm)
    res["c_v_a"] = paired(main_rows, "a", "c")
    res["c_v_a_later"] = paired(later, "a", "c")
    res["diagnostics"] = diagnostics(main_rows, x)
    # Bias by team-season, the §160 case, in points per match.
    teams = main_rows.groupby("Team").apply(lambda g: pd.Series({"Matches": len(g), "ctrl": (g.Pts - g.pts_ctrl).mean(), "a": (g.Pts - g.pts_a).mean(), "c": (g.Pts - g.pts_c).mean()}), include_groups=False)
    teams.sort_values("ctrl").round(3).to_csv(OUT / "team_bias.csv", encoding="utf-8-sig")
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float))
    print(teams.sort_values("ctrl").round(3).to_string())


if __name__ == "__main__":
    main()
