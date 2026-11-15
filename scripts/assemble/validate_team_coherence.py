"""CHANGELOG §161: tie player goal/assist forecasts to the team forecast (Arm A) and stretch the team forecast (Arm B).

Rates are pre-match, scored at realized minutes; c and the slope are fitted on earlier seasons only.
Test only; no production writes.
Run: python -u -X utf8 scripts/assemble/validate_team_coherence.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "bonus")]
import player_error_breakdown as E  # noqa: E402
import points_error_breakdown as P  # noqa: E402
import validate_prematch_coherence as vpc  # noqa: E402

OUT = P.VAL / "team_coherence_20261101"
OUT.mkdir(exist_ok=True)
LOG = P.VAL / "prematch_bonus_history_20261030" / "player_ratings_prematch.csv"
PTS = {"G": 6, "D": 6, "M": 5, "F": 4}
TEST = range(2019, 2025)
UPLIFT = {"npg": 1.2003, "ast": 1.1964}
NAMES = {"Jorge Luiz Frello Filho": "Jorginho", "Rodrigo Hernandez": "Rodrigo 'Rodri' Hernandez Cascante", "Fabio Henrique Tavares": "Fabinho",
         "Frederico Rodrigues de Paula Santos": "Fred", "Carlos Henrique Casimiro": "Casemiro", "Diogo Teixeira da Silva": "Diogo Jota",
         "Raphael Dias Belloli": "Raphinha", "Mathias Jorgensen": "Mathias Jørgensen", "Mahmoud Ahmed Ibrahim Hassan": "Trézéguet",
         "Emerson Leite de Souza Junior": "Emerson", "Emerson Aparecido Leite de Souza Junior": "Emerson", "James Mcatee": "James McAtee",
         "Victor da Silva": "Vitinho", "Vitor Ferreira": "Vitinha", "Jose Luis Mato Sanmartín": "Joselu", "Mateus Cardoso Lemos Martins": "Tetê",
         "Francisco Jorge Tomás Oliveira": "Chiquinho", "Joshua King": "Josh King", "José Ángel Esmorís Tasende": "Angeliño", "Hugo Bueno López": "Hugo Bueno"}


def repaired_history():
    """§160's unrated rows: real pre-match ratings, penalties and NPG. Scaling comes from teammates in the same match."""
    h = P.history_rows().reset_index(drop=True)
    log = pd.read_csv(LOG, encoding="utf-8-sig", low_memory=False)
    k = P.KEY
    log = log.drop_duplicates(k, keep="first")[k + ["NPG_Rating_Pre", "Assist_Rating_Pre"]]
    h["Mapped"] = h.Player.isin(NAMES)
    h["Player"] = h.Player.replace(NAMES)
    h = h.merge(log, on=k, how="left", validate="many_to_one")
    pms = pd.read_csv(P.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False, usecols=k + ["PK", "PKatt"]).drop_duplicates(k)
    pms = pms.rename(columns={"PK": "PK2", "PKatt": "PKatt2"})
    h = h.merge(pms, on=k, how="left", validate="many_to_one")
    conv = pd.concat([h.loc[h.Season == s, "Player"].map(vpc.conversion_lookup(int(s))) for s in h.Season.unique()]).reindex(h.index).fillna(1.0)
    short = h.Minutes < 60
    un, ua = np.where(short, UPLIFT["npg"], 1.0), np.where(short, UPLIFT["ast"], 1.0)
    base_n, base_a = h.NPG_Rating_Pre * conv * un, h.Assist_Rating_Pre * ua
    ok = ~h.Mapped & (base_n > 1e-4) & (base_a > 1e-4)
    team = ["Season", "Team", "Opponent", "Venue"]
    h["Gn"] = (h.NPG90_scaled / base_n).where(ok).groupby([h[c] for c in team]).transform("median")
    h["Ga"] = (h.Ast90_scaled / base_a).where(ok).groupby([h[c] for c in team]).transform("median")
    fix = h.Mapped & h.NPG_Rating_Pre.notna()
    h.loc[fix, "NPG90_scaled"] = (base_n * h.Gn)[fix]
    h.loc[fix, "Ast90_scaled"] = (base_a * h.Ga)[fix]
    h.loc[fix, "PK"], h.loc[fix, "PKatt"] = h.PK2[fix].fillna(0), h.PKatt2[fix].fillna(0)
    h.loc[fix, "NPG"] = (h.goals_scored[fix] - h.PK[fix]).clip(lower=0)
    print("repaired rows", int(fix.sum()), "of mapped", int(h.Mapped.sum()), "Gn null", int(h.Gn[fix].isna().sum()))
    h["Conv"] = conv
    return h, fix


def player_outcomes():
    p = pd.read_csv(P.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False, usecols=P.KEY + ["npxG", "xAG"])
    return p.drop_duplicates(P.KEY)


def frame():
    h, fix = repaired_history()
    known = set(pd.read_csv(P.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False, usecols=["Player"]).Player)
    extra = h[["Gn", "Ga", "NPG_Rating_Pre", "Conv"]].reset_index(drop=True)
    h = h.reset_index(drop=True)
    a = E.block_frame(h, "2018-2024").join(extra).join(h[["NPG", "assists"]].rename(columns={"assists": "Ast"}))
    r = P.recent_rows().reset_index(drop=True)
    b = E.block_frame(r, "2025-26").join(r[["NPG", "assists"]].rename(columns={"assists": "Ast"}))
    f = pd.concat([a, b], ignore_index=True)
    bad = ~f.Player.isin(known)
    print("still unrated rows dropped", int(bad.sum()), "players", f[bad].Player.nunique())
    f = f[~bad].reset_index(drop=True)
    f = f.merge(player_outcomes(), on=P.KEY, how="left", validate="many_to_one")
    f["gp"] = f.Pos.map(PTS).astype(float)
    f["lam"] = f.p_goals / f.gp          # expected non-penalty goals at realized minutes
    f["alam"] = f.p_assists / 3          # expected assists at realized minutes
    f["m"] = f.Minutes / 90
    return f


def team_table(f):
    k = ["Block", "Season", "Team", "Match_Key"]
    t = f.groupby(k).agg(T=("Team_Pred_NPG", "first"), S=("lam", "sum"), Sa=("alam", "sum"), G=("NPG", "sum"), A=("Ast", "sum")).reset_index()
    return t


def fit_coherence(t, y, s):
    x = np.log(t["T"] / t[s])
    m = sm.GLM(t[y], sm.add_constant(x.rename("g")), family=sm.families.Poisson(), offset=np.log(t[s])).fit()
    return float(m.params["const"]), float(m.params["g"]), float(m.bse["g"])


def assist_level(f, col):
    k = ["Block", "Pos"]
    return f[col] * f.groupby(k).Ast.transform("sum") / f.groupby(k)[col].transform("sum")


def apply_arm(f, t, fits):
    """fits: {Season: (c, g, ca, ga)}. Adds lam_c, alam_c for rows of those seasons."""
    x = f.merge(t[["Match_Key", "Team", "T", "S", "Sa"]], on=["Match_Key", "Team"], how="left", validate="many_to_one")
    c = x.Season.map(lambda s: fits[s][0]).values
    g = x.Season.map(lambda s: fits[s][1]).values
    ca = x.Season.map(lambda s: fits[s][2]).values
    ga = x.Season.map(lambda s: fits[s][3]).values
    x["mult"], x["amult"] = np.exp(c) * (x["T"] / x.S) ** g, np.exp(ca) * (x["T"] / x.Sa) ** ga
    x["lam_c"], x["alam_c"] = x.lam * x.mult, x.alam * x.amult
    return x


def score(x, lam="lam_c", alam="alam_c", clusters=None, reps=1000):
    """Points MSE and paired difference for goals + assists; assist level matched in both arms."""
    y = x.NPG * x.gp + x.Ast * 3
    p0 = x.lam * x.gp + assist_level(x, "alam") * 3
    p1 = x[lam] * x.gp + assist_level(x, alam) * 3
    d = (y - p1) ** 2 - (y - p0) ** 2
    out = {"rows": len(x), "mse_inc": float(((y - p0) ** 2).mean()), "mse_arm": float(((y - p1) ** 2).mean())}
    out["diff"], out["pct"] = float(d.mean()), float(100 * d.mean() / ((y - p0) ** 2).mean())
    if clusters is not None:
        c, _ = pd.factorize(clusters)
        k = c.max() + 1
        s, n = np.bincount(c, d.values, k), np.bincount(c, minlength=k).astype(float)
        w = np.random.default_rng(1).multinomial(k, np.ones(k) / k, size=reps)
        b = (w * s).sum(1) / (w * n).sum(1)
        out["lo"], out["hi"] = float(np.quantile(b, .025)), float(np.quantile(b, .975))
    return out, y - p0, y - p1


def pre_match_sum(f):
    """A-pre: the sum without hindsight. Last rating × mean minutes over the team's previous five matches, this match's opponent and venue."""
    h = f[f.Block == "2018-2024"].dropna(subset=["Round"]).copy()
    h["base"] = h.NPG_Rating_Pre * h.Conv
    a = h.alam / h.m / h.Ga                 # the short-minutes uplift is left in rows under 60 minutes, as a small upward tilt
    h["abase"] = a.where(h.Minutes >= 60, a / UPLIFT["ast"])
    rows = []
    for (season, team), g in h.groupby(["Season", "Team"]):
        order = g.groupby("Round").agg(Gn=("Gn", "first"), Ga=("Ga", "first"), key=("Match_Key", "first")).sort_index()
        rounds = list(order.index)
        for i, rnd in enumerate(rounds):
            if rnd < 6 or i < 5:
                continue
            past = g[g.Round.isin(rounds[i - 5:i])]
            last = past.sort_values("Round").groupby("Player").tail(1).set_index("Player")
            mins = past.groupby("Player").Minutes.sum() / 5
            s = float((last.base * mins / 90).sum() * order.Gn[rnd])
            sa = float((last.abase * mins / 90).sum() * order.Ga[rnd])
            rows.append({"Match_Key": order.key[rnd], "Team": team, "Spre": s, "Sapre": sa})
    return pd.DataFrame(rows)


def team_spread(t, seasons):
    """Arm B on the full team-match table of match_results: exp(c)·T^delta fitted on earlier seasons."""
    k = ["Season", "Team", "NPG", "GA", "T", "O"]
    H = pd.read_csv(P.DATA / "team_ratings_history.csv", encoding="utf-8-sig", low_memory=False)
    H = H[H.Season >= 2018]
    a = H.assign(Team=H.Home, NPG=H.Home_NPG, GA=H.Away_Goals, T=H.Home_Prediction, O=H.Away_Prediction)[k]
    b = H.assign(Team=H.Away, NPG=H.Away_NPG, GA=H.Home_Goals, T=H.Away_Prediction, O=H.Home_Prediction)[k]
    d = pd.concat([a, b], ignore_index=True)
    pen = 1 + P.PEN_RATE * P.PEN_CONV
    rows, tot = [], []
    for s in seasons:
        tr, te = d[d.Season < s], d[d.Season == s].copy()
        m = sm.GLM(tr.NPG, sm.add_constant(np.log(tr["T"]).rename("l")), family=sm.families.Poisson()).fit()
        c, dl = float(m.params["const"]), float(m.params["l"])
        tp, op = np.exp(c) * te["T"] ** dl, np.exp(c) * te.O ** dl
        ll = lambda p: float((p - te.NPG * np.log(p)).mean())
        cs = (te.GA == 0).astype(float)
        bl = lambda o: float(-(cs * np.log(np.exp(-o * pen)) + (1 - cs) * np.log(1 - np.exp(-o * pen))).mean())
        rows.append({"Season": s, "c": c, "delta": dl, "ll_inc": ll(te["T"]), "ll_arm": ll(tp), "cs_inc": bl(te.O), "cs_arm": bl(op), "n": len(te)})
    r = pd.DataFrame(rows)
    w = r.n / r.n.sum()
    return r, {"ll_diff": float(((r.ll_arm - r.ll_inc) * w).sum()), "cs_diff": float(((r.cs_arm - r.cs_inc) * w).sum()), "seasons_better": int((r.ll_arm < r.ll_inc).sum())}


def gap(x, lam, alam):
    """Goal + assist bias by team-forecast quartile, cut on all rows of the frame."""
    y = x.NPG * x.gp + x.Ast * 3
    b = y - (x[lam] * x.gp + assist_level(x, alam) * 3)
    return b.groupby(x.Q, observed=True).mean()


def main():
    pd.set_option("display.width", 220, "display.max_columns", 40)
    f = frame()
    t = team_table(f)
    h = t[t.Block == "2018-2024"]
    fits, rows = {}, []
    for s in TEST:
        tr = h[h.Season < s]
        c, g, se = fit_coherence(tr, "G", "S")
        ca, ga, sea = fit_coherence(tr, "A", "Sa")
        fits[s] = (c, g, ca, ga)
        rows.append({"Season": s, "c": c, "gamma": g, "se": se, "c_ast": ca, "gamma_ast": ga, "se_ast": sea})
    c, g, se = fit_coherence(h, "G", "S")
    ca, ga, sea = fit_coherence(h, "A", "Sa")
    for s in (2025, 2026):
        fits[s] = (c, g, ca, ga)
    rows.append({"Season": "2025-26", "c": c, "gamma": g, "se": se, "c_ast": ca, "gamma_ast": ga, "se_ast": sea})
    fit_t = pd.DataFrame(rows)
    print(fit_t.round(3).to_string(index=False))

    x = apply_arm(f[f.Season >= 2019].reset_index(drop=True), t, fits)
    x["Q"] = pd.qcut(x.Team_Pred_NPG, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    x["cluster"] = x.Season.astype(str) + x.Team
    o = x[x.Season <= 2024].reset_index(drop=True)
    pooled, e0, e1 = score(o, clusters=o.cluster)
    by = {int(s): score(g2.reset_index(drop=True))[0] for s, g2 in o.groupby("Season")}
    print("pooled", pooled)
    print(pd.DataFrame(by).T[["rows", "mse_inc", "mse_arm", "pct"]].round(4))

    xg = {}
    for col, lam, base in (("npxG", "lam_c", "lam"), ("xAG", "alam_c", "alam")):
        v = o[o[col].notna()]
        xg[col] = {"mse_inc": float(((v[col] - v[base]) ** 2).mean()), "mse_arm": float(((v[col] - v[lam]) ** 2).mean()), "rows": len(v)}
    print(xg)

    g0, g1 = gap(o, "lam", "alam"), gap(o, "lam_c", "alam_c")
    print("gap by quartile\n", pd.DataFrame({"incumbent": g0, "arm": g1}).round(4))
    pos = {}
    for p, g2 in o.groupby("Pos"):
        g2 = g2.reset_index(drop=True)
        y = g2.NPG * g2.gp + g2.Ast * 3
        pos[p] = {"inc": float((y - (g2.lam * g2.gp + assist_level(g2, "alam") * 3)).mean()), "arm": float((y - (g2.lam_c * g2.gp + assist_level(g2, "alam_c") * 3)).mean())}
    # Position-level assist matching hides nothing here: block_pos level is matched inside assist_level, so goals drive this.
    print("position bias", pos)
    qp = o.groupby(["Pos", "Q"], observed=True).apply(lambda g2: pd.Series({"inc": (g2.NPG - g2.lam).mean() * g2.gp.iloc[0], "arm": (g2.NPG - g2.lam_c).mean() * g2.gp.iloc[0]}), include_groups=False).unstack("Q")
    print("goal points bias by position and quartile\n", qp.round(3))

    rec = x[x.Season >= 2025].reset_index(drop=True)
    recent = score(rec, clusters=rec.cluster)[0]
    print("2025-26", recent)
    rg = pd.DataFrame({"incumbent": gap(rec, "lam", "alam"), "arm": gap(rec, "lam_c", "alam_c")})
    print(rg.round(4))

    sp = pre_match_sum(f[f.Season <= 2024])
    y = o.merge(sp, on=["Match_Key", "Team"], how="inner", validate="many_to_one").reset_index(drop=True)
    tp = y.drop_duplicates(["Match_Key", "Team"])
    print("A-pre rows", len(y), "corr log S, Spre", round(float(np.corrcoef(np.log(tp.S), np.log(tp.Spre))[0, 1]), 3), "mean S", round(float(tp.S.mean()), 3), "Spre", round(float(tp.Spre.mean()), 3))
    pre_fits = {}
    tall = team_table(f[f.Season <= 2024]).merge(sp, on=["Match_Key", "Team"])
    for s in TEST:
        tr = tall[tall.Season < s]
        if tr.Season.nunique() == 0:
            tr = None
        if tr is None:
            continue
        c1, g1_, _ = fit_coherence(tr, "G", "Spre")
        c2, g2_, _ = fit_coherence(tr, "A", "Sapre")
        pre_fits[s] = (c1, g1_, c2, g2_)
        print("A-pre fit", s, round(c1, 3), round(g1_, 3), round(c2, 3), round(g2_, 3))
    y["lam_p"] = y.lam * y.Season.map(lambda s: np.exp(pre_fits[s][0])) * (y["T"] / y.Spre) ** y.Season.map(lambda s: pre_fits[s][1])
    y["alam_p"] = y.alam * y.Season.map(lambda s: np.exp(pre_fits[s][2])) * (y["T"] / y.Sapre) ** y.Season.map(lambda s: pre_fits[s][3])
    pre = score(y, "lam_p", "alam_p", clusters=y.cluster)[0]
    same = score(y, clusters=y.cluster)[0]
    print("A-pre", pre, "\nA on same rows", same)
    pg = pd.DataFrame({"incumbent": gap(y, "lam", "alam"), "A": gap(y, "lam_c", "alam_c"), "A-pre": gap(y, "lam_p", "alam_p")})
    print(pg.round(4))

    b_t, b = team_spread(t, range(2019, 2026))
    print(b_t.round(4).to_string(index=False), b)

    # total-points view: every other component as in §160, goals and assists replaced
    z = o.copy()
    other = sum(z["a_" + k] - z["p_" + k] for k in E.COMPS if k not in ("goals", "assists"))
    tot0 = other + (z.NPG * z.gp + z.Ast * 3) - (z.lam * z.gp + assist_level(z, "alam") * 3)
    tot1 = other + (z.NPG * z.gp + z.Ast * 3) - (z.lam_c * z.gp + assist_level(z, "alam_c") * 3)
    total = pd.DataFrame({"incumbent": tot0.groupby(z.Q, observed=True).mean(), "arm": tot1.groupby(z.Q, observed=True).mean()})
    print("all components except defcon, by quartile\n", total.round(4))

    rule = {"i": pooled["hi"] < 0, "ii": sum(v["diff"] < 0 for v in by.values()) >= 4,
            "iii": all(v["mse_arm"] <= v["mse_inc"] for v in xg.values()),
            "iv": (g1.Q4 - g1.Q1) <= .5 * (g0.Q4 - g0.Q1),
            "v": all(abs(v["arm"]) - abs(v["inc"]) <= .02 for v in pos.values()),
            "vi": pre["diff"] <= .5 * same["diff"], "vii": recent["diff"] <= 0}
    rule = {k: bool(v) for k, v in rule.items()}
    rule["adopt_worthy"] = all(rule.values())
    res = {"fits": fit_t.to_dict("records"), "pooled": pooled, "by_season": by, "xg": xg, "gap_inc": g0.to_dict(), "gap_arm": g1.to_dict(), "position": pos, "recent": recent,
           "recent_gap": rg.to_dict(), "a_pre": pre, "a_same_rows": same, "pre_gap": pg.to_dict(), "arm_b": b, "total_gap": total.to_dict(), "rule": rule}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    b_t.to_csv(OUT / "arm_b.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(rule, indent=1))


if __name__ == "__main__":
    main()
