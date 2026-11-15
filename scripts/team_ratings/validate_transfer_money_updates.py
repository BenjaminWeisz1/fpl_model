"""
Money prior updated for in-season transfers (pre-registered in CHANGELOG §212; user idea).

The money prior is the 15 Aug squad value (§210-§211). After that cut-off, each counted arrival adds the player's
market value at the move and each counted departure subtracts it; wages scale in proportion; shares are recomputed
over the season's clubs; the season's past-only OLS (§202) gives the updated prior. Before each match, each side's
ratings move by w x (change in prior since 15 Aug - change already applied), w = 0.3 continuing, 1.0 promoted.

Arms: C (no update; must reproduce production team_ratings_history.csv), T1 (arrival counts once named in a Premier
League matchday squad for the new club, effective after that match; departure counts if named for the old club in the
365 days before leaving), T0 (every move from its date), T1h (T1 at half weight). Walk-forward Poisson log loss
2017-2025. The rating loop is update_team_ratings.py's, copied, with only the adjustment hook added.

Run locally:
    python validate_transfer_money_updates.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import update_team_ratings as utr  # noqa: E402
import validate_money_priors_pooled as vmp  # noqa: E402

ROOT = vmp.ROOT
RAW = vmp.RAW
CUR = vmp.CUR
OUT = ROOT / "data" / "validation" / "transfer_money_updates_20261109"
TEST = range(2017, 2026)
FLOOR = 0.05
SEED, REPS = 42, 2000


def cutoff(s):
    return pd.Timestamp({2014: "2014-10-23", 2020: "2020-09-15"}.get(s, f"{s}-08-15"))


# ------------------------------------------------------------------ money model
def money_model():
    hm = pd.read_csv(CUR / "historical_money.csv", encoding="utf-8-sig")
    for c in ("Wages", "Squad_Value"):
        hm[c + "_pct"] = hm.groupby("Season")[c].transform(lambda x: x / x.sum())
    known = hm[hm["GF"].notna()]
    coef = {}
    for s in TEST:
        tr = known[known["Season"] < s]
        X = np.column_stack([np.ones(len(tr)), tr["Wages_pct"], tr["Squad_Value_pct"]])
        coef[s] = {t: np.linalg.lstsq(X, tr[t].to_numpy(), rcond=None)[0] for t in ("GF", "GA")}
    return hm, coef


def priors(base, dv, c):
    """base: DataFrame (Team, Wages, Squad_Value) for one season; dv: dict team -> EUR m change. Returns GF, GA / 38."""
    V = base["Squad_Value"].to_numpy() + np.array([dv.get(t, 0.0) for t in base["Team"]])
    W = base["Wages"].to_numpy() * V / base["Squad_Value"].to_numpy()
    X = np.column_stack([np.ones(len(V)), W / W.sum(), V / V.sum()])
    return {t: (gf, ga) for t, gf, ga in zip(base["Team"], X @ c["GF"] / 38, X @ c["GA"] / 38)}


# ------------------------------------------------------------------ events
def tm_tables():
    g = pd.read_csv(RAW / "games.csv.gz", usecols=["game_id", "competition_id", "season", "date", "home_club_id",
                                                   "away_club_id", "home_club_name", "away_club_name"])
    # 2016 included so the departure rule can look back 365 days from the start of 2017
    g = g[(g["competition_id"] == "GB1") & g["season"].between(min(TEST) - 1, max(TEST))].copy()
    g["date"] = pd.to_datetime(g["date"])
    g["Home"] = g["home_club_name"].map(lambda n: vmp.TM_TO_PROJECT.get(n, n))
    g["Away"] = g["away_club_name"].map(lambda n: vmp.TM_TO_PROJECT.get(n, n))
    names = pd.concat([g[["season", "home_club_id", "Home"]].set_axis(["season", "club_id", "Team"], axis=1),
                       g[["season", "away_club_id", "Away"]].set_axis(["season", "club_id", "Team"], axis=1)]).drop_duplicates()
    lu = pd.read_csv(RAW / "game_lineups.csv.gz", usecols=["game_id", "player_id", "club_id"])
    lu = lu[lu["game_id"].isin(set(g["game_id"]))].merge(g[["game_id", "season", "date"]], on="game_id")
    return g, names, lu


def events(names, lu, mode):
    """Rows: season, Team, eff (matches strictly after this date see it), dv (EUR m), kind."""
    t = pd.read_csv(RAW / "transfers.csv.gz")
    t["d"] = pd.to_datetime(t["transfer_date"])
    t = t[t["from_club_id"] != t["to_club_id"]]
    v = pd.read_csv(RAW / "player_valuations.csv.gz", usecols=["player_id", "date", "market_value_in_eur"])
    v["date"] = pd.to_datetime(v["date"])
    end = lu.groupby("season")["date"].max()
    named = lu.groupby(["player_id", "club_id"])["date"].apply(lambda x: np.sort(x.to_numpy())).to_dict()
    rows, skipped = [], 0
    for s in TEST:
        clubs = dict(zip(names.loc[names["season"] == s, "club_id"], names.loc[names["season"] == s, "Team"]))
        w = t[(t["d"] > cutoff(s)) & (t["d"] <= end[s])]
        w = w[w["to_club_id"].isin(clubs) | w["from_club_id"].isin(clubs)].copy()
        miss = w["market_value_in_eur"].isna()
        if miss.any():
            fb = pd.merge_asof(w[miss].sort_values("d"), v.sort_values("date"), left_on="d", right_on="date",
                               by="player_id", direction="backward", suffixes=("", "_v"))
            fb = fb[(fb["d"] - fb["date"]).dt.days <= 400]
            w.loc[miss, "market_value_in_eur"] = w.loc[miss].merge(
                fb[["player_id", "d", "market_value_in_eur_v"]], on=["player_id", "d"], how="left")["market_value_in_eur_v"].to_numpy()
        skipped += int(w["market_value_in_eur"].isna().sum())
        w = w[w["market_value_in_eur"].notna()]
        for _, r in w.iterrows():
            val = r["market_value_in_eur"] / 1e6
            if r["to_club_id"] in clubs:
                if mode == "T0":
                    rows.append((s, clubs[r["to_club_id"]], r["d"], val, "in"))
                else:
                    ds = named.get((r["player_id"], r["to_club_id"]), np.array([], dtype="datetime64[ns]"))
                    ds = ds[(ds >= r["d"].to_datetime64()) & (ds <= end[s].to_datetime64())]
                    if len(ds):
                        rows.append((s, clubs[r["to_club_id"]], pd.Timestamp(ds[0]), val, "in"))
            if r["from_club_id"] in clubs:
                if mode == "T0":
                    rows.append((s, clubs[r["from_club_id"]], r["d"], -val, "out"))
                else:
                    ds = named.get((r["player_id"], r["from_club_id"]), np.array([], dtype="datetime64[ns]"))
                    lo = (r["d"] - pd.Timedelta(days=365)).to_datetime64()
                    if ((ds >= lo) & (ds <= r["d"].to_datetime64())).any():
                        rows.append((s, clubs[r["from_club_id"]], r["d"], -val, "out"))
    return pd.DataFrame(rows, columns=["season", "Team", "eff", "dv", "kind"]), skipped


# ------------------------------------------------------------------ rating loop (update_team_ratings.main, copied)
def run_loop(df, prior_map, dates, promoted, hm, coef, ev, weight_scale):
    alpha, beta, home_advantage, money_weight = utr.alpha, utr.beta, utr.home_advantage, utr.money_weight
    ave_goals = pd.concat([df["Home_NPG"], df["Away_NPG"]]).mean()
    ratings = pd.DataFrame(columns=["Team", "Attack_Rating", "Defense_Rating"])
    current_season = None
    out = {k: [] for k in ("hp", "ap", "hatk", "hdef", "aatk", "adef", "shift")}
    by_season = {s: g.sort_values("eff") for s, g in ev.groupby("season")} if ev is not None else {}
    base = {s: g[["Team", "Wages", "Squad_Value"]].reset_index(drop=True) for s, g in hm.groupby("Season")}
    for idx, row in df.iterrows():
        season = row["Season"]
        if season != current_season:
            current_season = season
            season_teams = set(df[df["Season"] == season]["Home"]) | set(df[df["Season"] == season]["Away"])
            ratings = ratings[ratings["Team"].isin(season_teams)].reset_index(drop=True)
            for i in ratings.index:
                team = ratings.at[i, "Team"]
                if (season, team) in prior_map:
                    money_atk, money_def = prior_map[(season, team)]
                    ratings.at[i, "Attack_Rating"] = (1 - money_weight) * ratings.at[i, "Attack_Rating"] + money_weight * money_atk
                    ratings.at[i, "Defense_Rating"] = (1 - money_weight) * ratings.at[i, "Defense_Rating"] + money_weight * money_def
            applied = {}
            sev = by_season.get(season)
            p0 = priors(base[season], {}, coef[season]) if (sev is not None and season in coef) else None
        for team in (row["Home"], row["Away"]):
            if team not in ratings["Team"].values:
                money_atk, money_def = prior_map[(season, team)]
                ratings.loc[len(ratings)] = [team, money_atk, money_def]

        # ---- adjustment hook (§212)
        shift = 0.0
        if p0 is not None:
            d = dates[idx]
            dv = sev[sev["eff"] < d].groupby("Team")["dv"].sum().to_dict()
            if dv:
                pt = priors(base[season], dv, coef[season])
                for team in (row["Home"], row["Away"]):
                    w = (1.0 if (season, team) in promoted else money_weight) * weight_scale
                    tgt = (w * (pt[team][0] - p0[team][0]), w * (pt[team][1] - p0[team][1]))
                    done = applied.get(team, (0.0, 0.0))
                    m = ratings["Team"] == team
                    ratings.loc[m, "Attack_Rating"] = max(FLOOR, ratings.loc[m, "Attack_Rating"].iloc[0] + tgt[0] - done[0])
                    ratings.loc[m, "Defense_Rating"] = max(FLOOR, ratings.loc[m, "Defense_Rating"].iloc[0] + tgt[1] - done[1])
                    applied[team] = tgt
                    shift = max(shift, abs(tgt[0]), abs(tgt[1]))
        # ---- end hook

        home, away = row["Home"], row["Away"]
        h_atk = ratings.loc[ratings["Team"] == home, "Attack_Rating"].iloc[0]
        a_atk = ratings.loc[ratings["Team"] == away, "Attack_Rating"].iloc[0]
        h_def = ratings.loc[ratings["Team"] == home, "Defense_Rating"].iloc[0]
        a_def = ratings.loc[ratings["Team"] == away, "Defense_Rating"].iloc[0]
        pred_home_goals = h_atk * (a_def / ave_goals) * home_advantage
        pred_away_goals = a_atk * (h_def / ave_goals) * (2 - home_advantage)
        for k, val in zip(("hp", "ap", "hatk", "hdef", "aatk", "adef", "shift"),
                          (pred_home_goals, pred_away_goals, h_atk, h_def, a_atk, a_def, shift)):
            out[k].append(val)
        h_npxg, a_npxg = row["Home_NPxG_Adj"], row["Away_NPxG_Adj"]
        h_npg, a_npg = row["Home_NPG"], row["Away_NPG"]
        eps = 1e-6
        h_atk *= (1 - alpha + alpha * ((h_npxg * beta + h_npg * (1 - beta)) / max(pred_home_goals, eps)))
        a_atk *= (1 - alpha + alpha * ((a_npxg * beta + a_npg * (1 - beta)) / max(pred_away_goals, eps)))
        h_def *= (1 - alpha + alpha * ((a_npxg * beta + a_npg * (1 - beta)) / max(pred_away_goals, eps)))
        a_def *= (1 - alpha + alpha * ((h_npxg * beta + h_npg * (1 - beta)) / max(pred_home_goals, eps)))
        ratings.loc[ratings["Team"] == home, "Attack_Rating"] = h_atk
        ratings.loc[ratings["Team"] == home, "Defense_Rating"] = h_def
        ratings.loc[ratings["Team"] == away, "Attack_Rating"] = a_atk
        ratings.loc[ratings["Team"] == away, "Defense_Rating"] = a_def
    h = df.copy()
    h["Home_Prediction"], h["Away_Prediction"] = out["hp"], out["ap"]
    h["Home_Attack_Rating"], h["Home_Defense_Rating"] = out["hatk"], out["hdef"]
    h["Away_Attack_Rating"], h["Away_Defense_Rating"] = out["aatk"], out["adef"]
    h["shift"] = out["shift"]
    return h


def ll(h):
    return ((h.Home_Prediction - h.Home_NPG * np.log(h.Home_Prediction))
            + (h.Away_Prediction - h.Away_NPG * np.log(h.Away_Prediction))) / 2


def boot(x, rng):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(REPS)]
    return np.percentile(b, [2.5, 97.5]).round(5).tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(utr.MATCH_RESULTS_FILE)
    money = pd.read_csv(utr.MONEY_PRIORS_FILE)
    prior_map = {(r.Season, r.Team): (r.Money_GF, r.Money_GA) for r in money.itertuples()}
    hm, coef = money_model()
    for s in TEST:  # the season's OLS at the 15 Aug value must be production's prior
        p = priors(hm[hm["Season"] == s][["Team", "Wages", "Squad_Value"]].reset_index(drop=True), {}, coef[s])
        err = max(abs(p[t][0] - prior_map[(s, t)][0]) + abs(p[t][1] - prior_map[(s, t)][1]) for t in p)
        assert err < 1e-8, (s, err)
    teams_by = hm.groupby("Season")["Team"].apply(set).to_dict()
    promoted = {(s, t) for s in TEST for t in teams_by[s] if t not in teams_by[s - 1]}

    g, names, lu = tm_tables()
    key = g.set_index(["season", "Home", "Away"])["date"]
    dates = pd.Series([key.get((s, h, a), pd.NaT) for s, h, a in zip(df.Season, df.Home, df.Away)], index=df.index)
    win = df["Season"].isin(TEST)
    assert dates[win].notna().all(), df[win & dates.isna()][["Season", "Home", "Away"]].head()
    for s in TEST:
        assert set(names.loc[names["season"] == s, "Team"]) == teams_by[s], s

    res = {"events": {}}
    hC = run_loop(df, prior_map, dates, promoted, hm, coef, None, 0)
    prod = pd.read_csv(utr.OUT_FILE, low_memory=False)
    cols = ["Home_Prediction", "Away_Prediction", "Home_Attack_Rating", "Home_Defense_Rating"]
    assert np.allclose(hC[cols].to_numpy(float), prod[cols].to_numpy(float), atol=1e-12, equal_nan=True), "control != production"
    print("control reproduces production", flush=True)
    arms = {}
    for name, mode, scale in (("T1", "T1", 1.0), ("T0", "T0", 1.0), ("T1h", "T1", 0.5)):
        if mode not in res["events"]:
            ev, skipped = events(names, lu, mode)
            ev.to_csv(OUT / f"events_{mode}.csv", index=False, encoding="utf-8-sig")
            res["events"][mode] = {"n": len(ev), "arrivals": int((ev.kind == "in").sum()), "departures": int((ev.kind == "out").sum()),
                                   "skipped_no_value": skipped, "net_eur_m_by_season": ev.groupby("season")["dv"].sum().round(1).to_dict()}
            print(mode, res["events"][mode], flush=True)
        arms[name] = run_loop(df, prior_map, dates, promoted, hm, coef, pd.read_csv(OUT / f"events_{mode}.csv", parse_dates=["eff"]), scale)

    base = hC[win].reset_index(drop=True)
    lb = ll(base)
    prev = {s: teams_by[s - 1] for s in TEST}
    prom = pd.Series([(h not in prev[s]) or (a not in prev[s]) for s, h, a in zip(base.Season, base.Home, base.Away)])
    d_ = dates[win].reset_index(drop=True)
    jan = d_ >= pd.to_datetime((base.Season + 1).astype(str) + "-01-01")
    rng = np.random.default_rng(SEED)
    for name, h in arms.items():
        hA = h[win].reset_index(drop=True)
        dl = (ll(hA) - lb).to_numpy()
        tiers = {"all": np.ones(len(dl), bool), "promoted_matches": prom.to_numpy(), "after_1_jan": jan.to_numpy(),
                 "shift_ge_0.02": (hA["shift"] >= 0.02).to_numpy(), "2017_2020": base.Season.between(2017, 2020).to_numpy(),
                 "2021_2025": base.Season.between(2021, 2025).to_numpy()}
        r = {k: {"n": int(m.sum()), "delta": round(float(dl[m].mean()), 6) if m.any() else None,
                 "ci95": boot(dl[m], rng) if m.sum() > 1 else None} for k, m in tiers.items()}
        r["by_season"] = {int(s): round(float(dl[(base.Season == s).to_numpy()].mean()), 6) for s in TEST}
        r["base_ll"] = round(float(lb.mean()), 5)
        r["rule_pass"] = bool(r["all"]["delta"] < 0 and r["all"]["ci95"][1] < 0 and r["promoted_matches"]["delta"] <= 0
                              and r["2017_2020"]["delta"] < 0 and r["2021_2025"]["delta"] < 0)
        res[name] = r
        print(name, json.dumps({k: r[k] for k in ("all", "promoted_matches", "after_1_jan", "shift_ge_0.02", "2017_2020", "2021_2025", "rule_pass")}), flush=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
