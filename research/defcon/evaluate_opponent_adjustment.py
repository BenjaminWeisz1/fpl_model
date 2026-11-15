"""Item 1, AGENDA 2026-10-31 (CHANGELOG §149): opponent-adjusted defcon features, out of time.

Replaces OpponentAvgStat with actions conceded / the expected actions of the players who faced that opponent, using only
earlier matches, shrunk, recency-weighted and by position group; also tests the opponent's pre-match Attack_Rating and
expected possession (Sofascore pass share) as pressure features. Hyperparameters are chosen on 2018-2021 rows only.
No production writes. Run: python -u -X utf8 scripts/defcon/evaluate_opponent_adjustment.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from research.defcon import overnight_frames as H

DATA = H.ROOT / "data" / "current"
YEARS = [2022, 2023, 2024, 2025, 2026]
HALF_LIVES = [10, 19, 38]   # opponent matches
KS = [20, 60, 120]          # pseudo player-rows at the league ratio
KEY = ["Season", "Team", "Opponent", "Venue"]


def calendar():
    """(Season, Team, Opponent, Venue) -> kickoff, plus each team's pass share in that match."""
    s = pd.read_csv(DATA / "sofascore_player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=KEY + ["Kickoff", "Sofascore_totalPass"])
    v = s.groupby(KEY, as_index=False).agg(Kickoff=("Kickoff", "first"), Passes=("Sofascore_totalPass", "sum"))
    v["Kickoff"] = pd.to_datetime(v.Kickoff, utc=True)
    mirror = v[KEY + ["Passes"]].rename(columns={"Team": "Opponent", "Opponent": "Team", "Passes": "OppPasses"})
    mirror["Venue"] = mirror.Venue.map({"Home": "Away", "Away": "Home"})
    v = v.merge(mirror, on=KEY, how="left", validate="one_to_one")
    v["PassShare"] = v.Passes / (v.Passes + v.OppPasses)
    assert v.PassShare.notna().all() and len(v) == 7000, (len(v), v.PassShare.isna().sum())
    return v.sort_values("Kickoff").reset_index(drop=True)


def trailing_possession(cal, half_life=10, k=3.0):
    """Each team's EW pass share before each match, shrunk to 0.5; ExpPoss = 0.5 + (team - opponent) / 2."""
    d = 2.0 ** (-1.0 / half_life)
    # All teams' states are read before either side of a kickoff updates.
    num, den, before = {}, {}, np.zeros(len(cal))
    for _, idx in cal.groupby("Kickoff", sort=True).indices.items():
        for i in idx:
            t = cal.Team.values[i]
            before[i] = (num.get(t, 0.0) + 0.5 * k) / (den.get(t, 0.0) + k)
        for i in idx:
            t = cal.Team.values[i]
            num[t] = num.get(t, 0.0) * d + cal.PassShare.values[i]
            den[t] = den.get(t, 0.0) * d + 1.0
    c = cal.assign(TrailShare=before)
    m = c[["Season", "Team", "Opponent", "Venue", "TrailShare"]].rename(
        columns={"Team": "Opponent", "Opponent": "Team", "TrailShare": "OppTrailShare"})
    m["Venue"] = m.Venue.map({"Home": "Away", "Away": "Home"})
    c = c.merge(m, on=KEY, how="left", validate="one_to_one")
    c["ExpPoss"] = 0.5 + (c.TrailShare - c.OppTrailShare) / 2
    return c


def opponent_attack():
    h = pd.read_csv(DATA / "team_ratings_history.csv", encoding="utf-8-sig")
    a = h[["Season", "Home", "Away", "Away_Attack_Rating", "Home_Attack_Rating"]]
    x = a.rename(columns={"Home": "Team", "Away": "Opponent", "Away_Attack_Rating": "OppAttack", "Home_Attack_Rating": "TeamAttack"}).assign(Venue="Home")
    y = a.rename(columns={"Away": "Team", "Home": "Opponent", "Home_Attack_Rating": "OppAttack", "Away_Attack_Rating": "TeamAttack"}).assign(Venue="Away")
    return pd.concat([x, y], ignore_index=True)


def position_groups(frame, head):
    if head == "D":
        return np.where(frame.Pos_CB.values == 1, "CB", np.where(frame[["Pos_FB", "Pos_WB", "Pos_LB", "Pos_RB"]].sum(axis=1).values > 0, "WD", "CB"))
    if head == "F":
        return np.full(len(frame), "FW")
    central = frame[["Pos_DM", "Pos_CM"]].sum(axis=1).values > 0
    return np.where(central, "CM", "AM")


def add_expected(frame):
    f = frame.copy()
    # Pre-match expected per-90: mean of the career and recent rates; only rows with real evidence enter the ratio.
    f["_exp"] = 0.5 * (f.AvgPer90 + f.Recent5AvgPer90)
    f["_use"] = (f.Minutes >= 60) & (f.MatchesPlayed >= 5) & (f._exp > 0)
    return f


def opponent_factor(frame, cal, half_life, k):
    """Pre-match opponent factor per row: EW actions / EW expected of those who faced the opponent, by group, relative
    to the same ratio league-wide, shrunk toward 1 by k pseudo rows. Opponent state ages one step per opponent match."""
    d = 2.0 ** (-1.0 / half_life)
    league_d = 2.0 ** (-1.0 / (half_life * 10))
    u = frame[frame._use]
    agg = u.groupby(KEY + ["_group"]).agg(A=("Per90Stat", "sum"), E=("_exp", "sum"), N=("_exp", "size")).reset_index()
    lookup = {(s, t, o, v, g): (a, e, n) for s, t, o, v, g, a, e, n in agg.values}
    groups = sorted(frame._group.unique())
    state = {}    # (opponent, group) -> [A, E, N]
    league = {g: [0.0, 0.0] for g in groups}
    out = {}
    for _, idx in cal.groupby("Kickoff", sort=True).indices.items():
        rows = cal.iloc[idx]
        for s, t, o, v in zip(rows.Season, rows.Team, rows.Opponent, rows.Venue):
            for g in groups:
                A, E, N = state.get((o, g), (0.0, 0.0, 0.0))
                la, le = league[g]
                base = la / le if le > 0 else 1.0
                ratio = (A + k * base * (E / N if N > 0 else 1.0)) / (E + k * (E / N if N > 0 else 1.0)) if N > 0 else base
                out[(s, t, o, v, g)] = ratio / base
        for s, t, o, v in zip(rows.Season, rows.Team, rows.Opponent, rows.Venue):
            for g in groups:
                a, e, n = lookup.get((s, t, o, v, g), (0.0, 0.0, 0.0))
                A, E, N = state.get((o, g), (0.0, 0.0, 0.0))
                state[(o, g)] = (A * d + a, E * d + e, N * d + n)
                league[g][0] = league[g][0] * league_d + a
                league[g][1] = league[g][1] * league_d + e
    keys = zip(frame.Season, frame.Team, frame.Opponent, frame.Venue, frame._group)
    return np.array([out.get(key, 1.0) for key in keys])


def select(frame, cal):
    """Poisson deviance of per-row actions on 2018-2021, expected rate x factor. Scale is fixed by a per-grid level."""
    dev = frame[frame._use & frame.Season.between(2018, 2021)]
    res = {}
    for h in HALF_LIVES:
        for k in KS:
            f = opponent_factor(frame, cal, h, k)[dev.index]
            mu = dev._exp.values * f
            mu = mu * dev.Per90Stat.sum() / mu.sum()
            y = dev.Per90Stat.values
            with np.errstate(divide="ignore", invalid="ignore"):
                d = 2 * (np.where(y > 0, y * np.log(y / mu), 0) - (y - mu))
            res[(h, k)] = float(d.mean())
    best = min(res, key=res.get)
    base = dev._exp.values * dev.Per90Stat.sum() / dev._exp.sum()
    y = dev.Per90Stat.values
    with np.errstate(divide="ignore", invalid="ignore"):
        d0 = 2 * (np.where(y > 0, y * np.log(y / base), 0) - (y - base))
    return best, {f"{a}/{b}": v for (a, b), v in res.items()}, float(d0.mean())


def prepare(head, frame, cal, attack):
    f = add_expected(frame.reset_index(drop=True))
    f["_group"] = position_groups(f, head)
    best, grid, null = select(f, cal)
    f["OppFactor"] = opponent_factor(f, cal, *best)
    n = len(f)
    f = f.merge(cal[KEY + ["ExpPoss", "Kickoff"]], on=KEY, how="left", validate="many_to_one")
    f = f.merge(attack, on=KEY, how="left", validate="many_to_one")
    assert len(f) == n
    print(head, "selected", best, "null dev", round(null, 4), "best", round(grid[f"{best[0]}/{best[1]}"], 4),
          "missing", int(f.OppAttack.isna().sum()), int(f.ExpPoss.isna().sum()), flush=True)
    return f, {"selected": best, "grid": grid, "null_deviance": null}


def arms(cols):
    no_opp = [c for c in cols if c != "OpponentAvgStat"]
    return {"control": cols,
            "opp_factor": no_opp + ["OppFactor"],
            "opp_factor_attack": no_opp + ["OppFactor", "OppAttack"],
            "opp_factor_poss": no_opp + ["OppFactor", "ExpPoss"],
            "opp_factor_attack_poss": no_opp + ["OppFactor", "OppAttack", "ExpPoss"]}


def one(head, frame, year, name, cols):
    t0 = time.monotonic()
    tr, va = frame[frame.Season < year], frame[frame.Season == year]
    fit = H.fit_head(head, tr[cols].values, tr.Target.values, cols)
    p = H.predict(fit, va[cols].values)
    print(head, year, name, f"{time.monotonic() - t0:.0f}s", flush=True)
    return head, year, name, va.index.values, p


def main():
    t0 = time.monotonic()
    fr = H.frames()
    cal = trailing_possession(calendar())
    attack = opponent_attack()
    data, sel = {}, {}
    heads = {"D": (fr["df_def"], fr["DEF_FEATURE_COLS"]), "M": (fr["df_mid"], fr["MID_FEATURE_COLS"]), "F": (fr["df_fwd"], fr["FWD_FEATURE_COLS"])}
    for head, (frame, cols) in heads.items():
        data[head], sel[head] = prepare(head, frame, cal, attack)
    pd.to_pickle({"data": data, "sel": sel}, H.OUT / "item1_frames.pkl")
    jobs = [(h, y, n, c) for h, (fr_, cols) in heads.items() for y in YEARS for n, c in arms(cols).items()]
    results = Parallel(n_jobs=4, backend="loky")(
        delayed(one)(h, data[h], y, n, c) for h, y, n, c in jobs)
    preds = {}
    for h, y, n, idx, p in results:
        preds.setdefault(h, {}).setdefault(n, {})[y] = (idx, p)
    out = {"selection": {k: {"selected": v["selected"], "null_deviance": v["null_deviance"], "grid": v["grid"]} for k, v in sel.items()}}
    rows = []
    for h in heads:
        f = data[h]
        # The M head scores midfielders only, as production serves it; forwards are served by F.
        mask = f.IsForward.values == 0 if h == "M" else np.ones(len(f), bool)
        for n, by_year in preds[h].items():
            for y, (idx, p) in by_year.items():
                f.loc[idx, f"p_{n}"] = p
        v = f[f.Season.isin(YEARS) & mask]
        res = {}
        for n in preds[h]:
            res[n] = {"pooled": H.score(v.Target.values, v[f"p_{n}"].values),
                      "by_year": {str(y): H.score(g.Target.values, g[f"p_{n}"].values) for y, g in v.groupby("Season")},
                      "vs_control": H.paired_bootstrap(v.Player.values, v.Target.values, v.p_control.values, v[f"p_{n}"].values)}
        out[h] = res
        keep = ["Season", "Player", "Team", "Opponent", "Venue", "Minutes", "Target", "Per90Stat", "IsForward", "OppFactor", "OppAttack", "ExpPoss"]
        v[keep + [c for c in v if c.startswith("p_")]].to_csv(H.OUT / f"item1_predictions_{h}.csv", index=False, encoding="utf-8-sig")
        for n, r in res.items():
            rows.append((h, n, r["pooled"]["ll"], r["pooled"]["brier"], r["pooled"]["auc"], r["vs_control"]["mean"], r["vs_control"]["lo"], r["vs_control"]["hi"],
                         *[r["by_year"][str(y)]["ll"] - out[h]["control"]["by_year"][str(y)]["ll"] if "control" in out[h] else np.nan for y in YEARS]))
    (H.OUT / "item1_results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 200, "display.max_columns", 30)
    tab = pd.DataFrame(rows, columns=["head", "arm", "ll", "brier", "auc", "d_ll", "lo", "hi"] + [f"d{y}" for y in YEARS])
    print(tab.round(5).to_string())
    print(f"done {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
