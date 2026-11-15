"""CHANGELOG §160: which players and player types are predicted worst. Extends the §152 event-level breakdown.

Forecasts are pre-match rates at realized minutes, so this measures rate errors given minutes, not minutes errors.
Diagnostic only; no production writes.
Run: python -u -X utf8 scripts/assemble/player_error_breakdown.py
"""
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from fpl_model.assemble import points_error_breakdown as P  # noqa: E402
from fpl_model.defcon_scaling import DEFENDER_POINTS_FACTOR, nb_sf, prob_at_minutes  # noqa: E402

OUT = P.VAL / "player_error_20261101"
OUT.mkdir(exist_ok=True)
TM = P.ROOT / "data" / "raw" / "transfermarkt"
COMPS = ["goals", "assists", "penalties", "clean_sheet", "goals_conceded", "saves", "bonus", "cards_other_unmodelled"]
MIN_MATCHES, MIN_SEASON_MATCHES = 38, 20


def block_frame(r, block):
    r = r.reset_index(drop=True)
    c, _ = P.components(r)
    f = r[["Season", "Round", "Match_Key", "Player", "Pos", "Team", "Opponent", "Venue", "Minutes", "Team_Pred_NPG", "Opp_Pred_NPG"]].copy()
    f["Block"] = block
    for k in COMPS:
        f["p_" + k], f["a_" + k], f["v_" + k] = c[k]
    return f


def defcon_points(f):
    """2025+ only: real FPL hits against the production head at realized minutes, with the adopted defender factor."""
    keys = P.KEY
    fr = [pd.read_csv(P.DEFCON / f"item2_predictions_{h}.csv", encoding="utf-8-sig")[keys + ["p_control"]].assign(Pos=h) for h in "DMF"]
    d = pd.concat(fr, ignore_index=True).drop_duplicates(keys + ["Pos"])
    pm = pd.read_csv(P.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    pm = pm[pm.Data_Source == "FPL_API"].drop_duplicates(keys)[keys + ["Defensive_Contribution"]]
    f = f.merge(d, on=keys + ["Pos"], how="left", validate="many_to_one").merge(pm, on=keys, how="left", validate="many_to_one")
    f["p_defcon"], f["a_defcon"], f["v_defcon"] = 0.0, 0.0, 0.0
    use = (f.Season >= 2025) & f.p_control.notna() & f.Defensive_Contribution.notna()
    print("defcon rows", int(use.sum()), "of 2025+ outfield", int(((f.Season >= 2025) & (f.Pos != "G")).sum()))
    grid = np.linspace(.05, 40, 800)
    x = f[use]
    thr = np.where(x.Pos == "D", 10, 12)
    mu = np.where(thr == 10, np.interp(x.p_control, nb_sf(grid, 9), grid) * DEFENDER_POINTS_FACTOR, np.interp(x.p_control, nb_sf(grid, 11), grid))
    p = np.asarray(prob_at_minutes(mu, x.Minutes.values, thr), dtype=float)
    f.loc[use, "p_defcon"], f.loc[use, "a_defcon"], f.loc[use, "v_defcon"] = 2 * p, 2.0 * (x.Defensive_Contribution.values >= thr), 4 * p * (1 - p)
    return f


def level_fixes(f):
    k = ["Block", "Pos"]
    ratio = f.groupby(k).a_assists.transform("sum") / f.groupby(k).p_assists.transform("sum")
    print("assist level factors", f.assign(r=ratio).groupby(k).r.first().round(3).to_dict())
    f["p_assists"] *= ratio
    f["v_assists"] *= ratio
    f["p_cards_other_unmodelled"] = f.groupby("Pos").a_cards_other_unmodelled.transform("mean")
    f["v_cards_other_unmodelled"] = f.groupby("Pos").a_cards_other_unmodelled.transform("var")
    return f


def totals(f):
    all_c = COMPS + ["defcon"]
    f["pred"] = f[["p_" + c for c in all_c]].sum(axis=1)
    f["act"] = f[["a_" + c for c in all_c]].sum(axis=1)
    f["chance"] = f[["v_" + c for c in all_c]].sum(axis=1)
    f["err"] = f.act - f.pred
    f["excess"] = f.err ** 2 - f.chance
    return f


def norm(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(ch)).lower().strip()


def attributes(f):
    pm = pd.read_csv(P.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False, usecols=["Season", "Player", "Team"])
    apps = pm.groupby(["Player", "Season"]).size().rename("n").reset_index()
    prior = {}
    for player, g in apps.groupby("Player"):
        c = g.sort_values("Season")
        prior.update({(player, s): int(v) for s, v in zip(c.Season, c.n.cumsum() - c.n)})
    f["Prior_Apps"] = [prior.get(x, 0) for x in zip(f.Player, f.Season)]
    f["Experience"] = pd.cut(f.Prior_Apps, [-1, 0, 38, 114, 228, 10000], labels=["1 PL debut season", "2 1-38 prior", "3 39-114", "4 115-228", "5 229+"])
    f.loc[f.Season < 2022, "Experience"] = np.nan

    main = pm.groupby(["Player", "Season"]).Team.agg(lambda t: t.value_counts().index[0]).to_dict()
    f["Moved_PL"] = [(main.get((p, s - 1)) is not None and main.get((p, s - 1)) != t) for p, s, t in zip(f.Player, f.Season, f.Team)]
    teams = pm.groupby("Season").Team.agg(set).to_dict()
    f["Promoted"] = [t not in teams.get(s - 1, {t}) for t, s in zip(f.Team, f.Season)]

    tr = pd.read_csv(P.DATA / "transfers.csv", encoding="utf-8-sig", low_memory=False)[["Player", "Season", "League"]].drop_duplicates(["Player", "Season"])
    f = f.merge(tr, on=["Player", "Season"], how="left", validate="many_to_one")
    group = {"Championship": "Championship", "League One": "Lower English", "League Two": "Lower English", "Ligue 1": "Top five abroad", "La Liga": "Top five abroad",
             "Bundesliga": "Top five abroad", "Serie A": "Top five abroad"}
    f["Route"] = np.where(f.Prior_Apps > 0, "Established PL player", None)
    arrive = f.League.notna() & (f.League != "Premier League") & (f.Prior_Apps == 0)
    f.loc[arrive, "Route"] = f.League[arrive].map(group).fillna("Other league")
    f.loc[f.Route.isna(), "Route"] = "Debut, no prior league (youth or unlisted)"
    f.loc[f.Season < 2022, "Route"] = np.nan

    pos = pd.read_csv(P.DATA / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False)
    k = ["Season", "Team", "Opponent", "Venue", "Player"]
    f = f.merge(pos[k + ["Position_Resolved"]].drop_duplicates(k), on=k, how="left", validate="many_to_one")
    f["Detail"] = f.Position_Resolved.replace({"LB": "FB", "RB": "FB", "LW": "Winger", "RW": "Winger", "LM": "Wide mid", "RM": "Wide mid"})

    f["Role"] = pd.cut(f.Minutes, [0, 30, 59, 85, 90], labels=["1-30 min", "31-59 min", "60-85 min", "86-90 min"])
    f["Rate"] = f.pred / (f.Minutes / 90)
    f["Tier"] = f.groupby("Pos").pred.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]))
    f["Team_Attack"] = pd.qcut(f.Team_Pred_NPG, 4, labels=["Q1 weakest", "Q2", "Q3", "Q4 strongest"])
    f["Opp_Attack"] = pd.qcut(f.Opp_Pred_NPG, 4, labels=["Q1 weakest", "Q2", "Q3", "Q4 strongest"])
    pts = f.groupby(["Season", "Team", "Player"]).a_penalties.sum().rename("pen")
    share = P.actual_penalty_share(f)
    f["Penalty_Taker"] = share.reindex(pd.MultiIndex.from_arrays([f.Season, f.Team, f.Player])).fillna(0).values >= .5
    del pts

    try:
        tm = pd.read_csv(TM / "players.csv.gz", low_memory=False)
        val = pd.read_csv(TM / "player_valuations.csv.gz", low_memory=False)
        tm["key"] = tm.name.map(norm)
        tm = tm[~tm.duplicated("key", keep=False)]
        ids = dict(zip(tm.key, tm.player_id))
        dob = dict(zip(tm.player_id, pd.to_datetime(tm.date_of_birth, errors="coerce")))
        f["TM_ID"] = f.Player.map(lambda x: ids.get(norm(x)))
        cover = f.TM_ID.notna().mean()
        print("Transfermarkt coverage", round(float(cover), 3))
        if cover >= .8:
            start = pd.to_datetime(f.Season.astype(str) + "-08-01")
            f["Age"] = (start - f.TM_ID.map(dob)).dt.days / 365.25
            f["Age_Band"] = pd.cut(f.Age, [0, 21, 24, 27, 30, 50], labels=["Under 21", "21-23", "24-26", "27-29", "30+"], right=False)
            val["date"] = pd.to_datetime(val.date)
            val = val.sort_values("date")
            frames = []
            for season in sorted(f.Season.unique()):
                cut = pd.Timestamp(f"{season}-08-01")
                v = val[val.date <= cut].groupby("player_id").market_value_in_eur.last().rename("MV").reset_index().assign(Season=season)
                frames.append(v)
            f = f.merge(pd.concat(frames).rename(columns={"player_id": "TM_ID"}), on=["TM_ID", "Season"], how="left", validate="many_to_one")
            print("market value coverage", round(float(f.MV.notna().mean()), 3))
            f["Value_Band"] = f.groupby(["Season", "Pos"]).MV.transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1 cheapest", "Q2", "Q3", "Q4 dearest"]) if s.notna().sum() >= 8 else np.nan)
    except (OSError, KeyError, ValueError, AttributeError) as e:
        print("Transfermarkt attributes skipped:", repr(e))
    return f


def category(f, col, reps=500):
    x = f[f[col].notna()].copy()
    codes, _ = pd.factorize(x.Player)
    k = codes.max() + 1
    w = np.random.default_rng(1).multinomial(k, np.ones(k) / k, size=reps)
    rows = []
    for level, g in x.groupby(col, observed=True):
        i = codes[x.index.get_indexer(g.index)]
        n = np.bincount(i, minlength=k).astype(float)
        e, s = np.bincount(i, g.err.values, k), np.bincount(i, g.excess.values, k)
        den = (w * n).sum(1)
        b, ex = (w * e).sum(1) / den, (w * s).sum(1) / den
        comp = {c: float((g["a_" + c] - g["p_" + c]).mean()) for c in COMPS + ["defcon"]}
        rows.append({"Category": col, "Level": str(level), "Matches": len(g), "Players": g.Player.nunique(), "Pred": g.pred.mean(), "Actual": g.act.mean(),
                     "Bias": g.err.mean(), "Bias_lo": np.quantile(b, .025), "Bias_hi": np.quantile(b, .975), "MSE": (g.err ** 2).mean(), "Excess": g.excess.mean(),
                     "Excess_lo": np.quantile(ex, .025), "Excess_hi": np.quantile(ex, .975), "Excess_Share": g.excess.mean() / (g.err ** 2).mean(),
                     "Top_Component": max(comp, key=lambda c: abs(comp[c])), "Top_Component_Bias": comp[max(comp, key=lambda c: abs(comp[c]))]})
    return rows


def player_table(f, by, minimum):
    g = f.groupby(by)
    t = g.agg(Pos=("Pos", "last"), Team=("Team", "last"), Matches=("err", "size"), Minutes=("Minutes", "sum"), Pred=("pred", "sum"), Actual=("act", "sum"),
              Err=("err", "sum"), Chance=("chance", "sum"), MSE=("err", lambda s: (s ** 2).mean()), Excess=("excess", "mean")).reset_index()
    t = t[t.Matches >= minimum].copy()
    t["Bias_Per_Match"] = t.Err / t.Matches
    t["Z"] = t.Err / np.sqrt(t.Chance)
    comp = f.assign(**{"d_" + c: f["a_" + c] - f["p_" + c] for c in COMPS + ["defcon"]}).groupby(by)[["d_" + c for c in COMPS + ["defcon"]]].sum().reset_index()
    t = t.merge(comp, on=by, how="left")
    d = t[["d_" + c for c in COMPS + ["defcon"]]]
    t["Main_Component"] = d.abs().idxmax(axis=1).str[2:]
    t["Main_Component_Pts"] = [row[f"d_{m}"] for (_, row), m in zip(d.iterrows(), t.Main_Component)]
    return t


def bh(z):
    from scipy.stats import norm as nd
    p = 2 * nd.sf(np.abs(z))
    order = np.argsort(p)
    q = np.empty(len(p))
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return q


def top30(f, cols, cur):
    """The options people consider: ranked on forecasts alone, each season and position."""
    full = f[f.Minutes >= 60].groupby(["Season", "Pos", "Player"]).pred.agg(["mean", "size"]).reset_index()
    full = full[full["size"] >= np.where(full.Season <= 2024, 10, 3)]
    full["Rank"] = full.groupby(["Season", "Pos"])["mean"].rank(ascending=False, method="first")
    full["Band"] = pd.cut(full.Rank, [0, 10, 30, 1000], labels=["Top 1-10", "Top 11-30", "Rest"])
    x = f.merge(full[["Season", "Pos", "Player", "Rank", "Band"]], on=["Season", "Pos", "Player"], how="inner", validate="many_to_one")
    x = x.reset_index(drop=True)
    print("\n=== TOP 30 ===\nranked player-seasons", len(full), "matches", len(x))
    bands = pd.DataFrame(category(x, "Band") + [r for p, g in x.groupby("Pos") for r in [{**d, "Level": f"{p} {d['Level']}"} for d in category(g.reset_index(drop=True), "Band")]])
    print(bands.drop(columns=["Excess_lo", "Excess_hi", "Category"]).round(3).to_string(index=False))
    t = x[x.Band != "Rest"].reset_index(drop=True)
    comp = {c: float((t["a_" + c] - t["p_" + c]).mean()) for c in COMPS + ["defcon"]}
    print("top 30 bias by component", {k: round(v, 3) for k, v in comp.items()})
    print(pd.DataFrame({p: {c: (g["a_" + c] - g["p_" + c]).mean() for c in COMPS + ["defcon"]} for p, g in t.groupby("Pos")}).round(3).to_string())
    cats = pd.DataFrame([r for c in cols if c in t and c != "Tier" for r in category(t, c)])
    print(cats.drop(columns=["Excess_lo", "Excess_hi"]).round(3).to_string(index=False))
    pd.concat([bands, cats]).to_csv(OUT / "top30_categories.csv", index=False, encoding="utf-8-sig")

    pl = player_table(t, ["Player"], 25)
    pl["Z_Adj"] = pl.Z / pl.Z.std()
    pl["Q"] = bh(pl.Z_Adj.values)
    pl["Current"] = pl.Player.isin(cur)
    pl.sort_values("Z").to_csv(OUT / "top30_players.csv", index=False, encoding="utf-8-sig")
    show = ["Player", "Pos", "Team", "Matches", "Pred", "Actual", "Bias_Per_Match", "Z", "Z_Adj", "Q", "Excess", "Main_Component", "Main_Component_Pts", "Current"]
    print("top 30 players (25+ matches in the group)", len(pl), "SD z", round(float(pl.Z.std()), 3), "pass FDR", int((pl.Q < .05).sum()))
    for p, g in pl.groupby("Pos"):
        print(p, "OVER\n", g.nsmallest(8, "Z")[show].round(3).to_string(index=False))
        print(p, "UNDER\n", g.nlargest(8, "Z")[show].round(3).to_string(index=False))
    now = pl[pl.Current]
    print("current players, OVER\n", now.nsmallest(15, "Z")[show].round(3).to_string(index=False))
    print("current players, UNDER\n", now.nlargest(15, "Z")[show].round(3).to_string(index=False))

    ps = player_table(t, ["Player", "Season"], 10)
    a = ps[["Player", "Season", "Z", "Pos"]]
    s2 = a.merge(a.assign(Season=a.Season - 1).rename(columns={"Z": "Z2"}).drop(columns="Pos"), on=["Player", "Season"])
    y = t[t.Player.isin(pl.Player)].copy()
    y["Half"] = y.groupby("Player").cumcount() % 2
    h = y.groupby(["Player", "Half"]).agg(e=("err", "sum"), v=("chance", "sum")).reset_index()
    h["z"] = h.e / np.sqrt(h.v)
    h = h.pivot(index="Player", columns="Half", values="z").dropna()
    res = {"sd_z": float(pl.Z.std()), "players": len(pl), "fdr": int((pl.Q < .05).sum()), "split_half_r": float(h[0].corr(h[1])), "season_to_season_r": float(s2.Z.corr(s2.Z2)), "season_pairs": len(s2),
           "season_r_by_pos": {p: float(g.Z.corr(g.Z2)) for p, g in s2.groupby("Pos")}, "split_half_by_pos": {p: float(g[0].corr(g[1])) for p, g in h.join(pl.set_index("Player").Pos).groupby("Pos")}}
    print(json.dumps(res, indent=1))
    (OUT / "top30_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


def main():
    pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 200, "display.max_colwidth", 42)
    hist, rec = P.history_rows(), P.recent_rows()
    f = pd.concat([block_frame(hist, "2018-2024"), block_frame(rec, "2025-26")], ignore_index=True)
    # Rows under FPL full names the stats file doesn't use got a default rating in the hindcast, not the player's own (§160).
    known = set(pd.read_csv(P.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False, usecols=["Player"]).Player)
    bad = ~f.Player.isin(known)
    print("dropped unrated-name rows", int(bad.sum()), "players", f[bad].Player.nunique())
    f = f[~bad].reset_index(drop=True)
    f = defcon_points(f)
    f = totals(level_fixes(f))
    f = attributes(f).reset_index(drop=True)
    print("rows", len(f), "bias", round(float(f.err.mean()), 4), "MSE", round(float((f.err ** 2).mean()), 3), "chance", round(float(f.chance.mean()), 3), "excess share", round(float(f.excess.mean() / (f.err ** 2).mean()), 3))

    cols = ["Pos", "Detail", "Tier", "Role", "Experience", "Route", "Moved_PL", "Promoted", "Team_Attack", "Opp_Attack", "Venue", "Penalty_Taker", "Season", "Age_Band", "Value_Band"]
    cats = pd.DataFrame([r for c in cols if c in f for r in category(f, c)])
    cats.to_csv(OUT / "categories.csv", index=False, encoding="utf-8-sig")
    print(cats.drop(columns=["Excess_lo", "Excess_hi"]).round(3).to_string(index=False))

    pl = player_table(f, ["Player"], MIN_MATCHES)
    phi = float(pl.Z.std())
    pl["Z_Adj"] = pl.Z / phi
    pl["Q"] = bh(pl.Z_Adj.values)
    cur = set(pd.read_csv(P.DATA / "players_current.csv", encoding="utf-8-sig").Player)
    pl["Current"] = pl.Player.isin(cur)
    pl.sort_values("Z").to_csv(OUT / "players.csv", index=False, encoding="utf-8-sig")
    show = ["Player", "Pos", "Team", "Matches", "Pred", "Actual", "Bias_Per_Match", "Z", "Z_Adj", "Q", "Excess", "Main_Component", "Main_Component_Pts", "Current"]
    print("\nplayers", len(pl), "SD z", round(phi, 3), "pass FDR", int((pl.Q < .05).sum()))
    print("OVER-predicted\n", pl.nsmallest(20, "Z")[show].round(3).to_string(index=False))
    print("UNDER-predicted\n", pl.nlargest(20, "Z")[show].round(3).to_string(index=False))
    print("Largest excess\n", pl.nlargest(15, "Excess")[show].round(3).to_string(index=False))
    print(pl.groupby("Pos")[["Z", "Excess"]].agg(["std", "mean"]).round(3).to_string())

    ps = player_table(f, ["Player", "Season"], MIN_SEASON_MATCHES)
    ps["Z_Adj"] = ps.Z / ps.Z.std()
    ps["Q"] = bh(ps.Z_Adj.values)
    ps.sort_values("Z").to_csv(OUT / "player_seasons.csv", index=False, encoding="utf-8-sig")
    sshow = ["Player", "Season", "Pos", "Team", "Matches", "Pred", "Actual", "Bias_Per_Match", "Z", "Q", "Main_Component", "Main_Component_Pts"]
    print("\nplayer-seasons", len(ps), "SD z", round(float(ps.Z.std()), 3), "pass FDR", int((ps.Q < .05).sum()))
    print(ps.nsmallest(15, "Z")[sshow].round(3).to_string(index=False))
    print(ps.nlargest(15, "Z")[sshow].round(3).to_string(index=False))

    top30(f, cols, cur)

    # persistence
    x = f[f.Player.isin(pl.Player)].copy()
    x["Half"] = x.groupby("Player").cumcount() % 2
    h = x.groupby(["Player", "Half"]).agg(e=("err", "sum"), v=("chance", "sum")).reset_index()
    h["z"] = h.e / np.sqrt(h.v)
    h = h.pivot(index="Player", columns="Half", values="z").dropna()
    a = ps[["Player", "Season", "Z", "Bias_Per_Match", "Pos"]]
    nxt = a.assign(Season=a.Season - 1).rename(columns={"Z": "Z2", "Bias_Per_Match": "B2"})
    s2 = a.merge(nxt.drop(columns="Pos"), on=["Player", "Season"])
    res = {"sd_z_players": phi, "players": len(pl), "fdr_players": int((pl.Q < .05).sum()), "sd_z_seasons": float(ps.Z.std()), "fdr_seasons": int((ps.Q < .05).sum()),
           "split_half_r": float(h[0].corr(h[1])), "season_to_season_r": float(s2.Z.corr(s2.Z2)), "season_pairs": len(s2),
           "split_half_by_pos": {p: float(g[0].corr(g[1])) for p, g in h.join(pl.set_index("Player").Pos).groupby("Pos")},
           "season_r_by_pos": {p: float(g.Z.corr(g.Z2)) for p, g in s2.groupby("Pos")}}
    print(json.dumps(res, indent=1))
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
