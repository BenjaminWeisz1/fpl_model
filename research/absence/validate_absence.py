"""Key-player absence tests, pre-registered in CHANGELOG §154. Diagnostic: no production writes.
Run after build_absence_frame.py: python -u -X utf8 -m research.absence.validate_absence
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
from research.absence import build_absence_frame as B

OUT, DATA = B.OUT, B.DATA
RES = {}
pd.set_option("display.width", 200, "display.max_columns", 30)


def base_frame(xi):
    t = B.team_matches()
    keys = ["Season", "Team", "Opponent", "Venue"]
    f = xi.drop_duplicates("Event")  # placeholder to confirm one event per match per team below
    tm = xi[keys + ["Event", "Kickoff", "Round_Team"]].drop_duplicates(keys).merge(t, on=keys, how="left", validate="one_to_one")
    tm["ResDiff"] = (tm.NPxG - tm.NPxGA) - (tm.Pred - tm.PredA)
    tm["ResFor"], tm["ResAgainst"] = tm.NPxG - tm.Pred, tm.NPxGA - tm.PredA
    tm["ResDiff"] -= tm.groupby(["Season", "Venue"]).ResDiff.transform("mean")
    tm["ResFor"] -= tm.groupby(["Season", "Venue"]).ResFor.transform("mean")
    tm["ResAgainst"] -= tm.groupby(["Season", "Venue"]).ResAgainst.transform("mean")
    print("team-matches", len(tm), "residual SD", round(tm.ResDiff.std(), 3), flush=True)
    return tm


def gaps(xi, col):
    e = xi[xi.Set == "E"].groupby(["Season", "Team", "Opponent", "Venue"])[col].sum()
    a = xi[xi.Set == "A"].groupby(["Season", "Team", "Opponent", "Venue"])[col].sum()
    return (e - a).rename("Gap")


def attach_gap(tm, xi, col, name):
    g = gaps(xi, col)
    tm = tm.merge(g.rename(name).reset_index(), on=["Season", "Team", "Opponent", "Venue"], how="left")
    o = g.rename(name + "_opp").reset_index().rename(columns={"Team": "Opponent", "Opponent": "Team"})
    o["Venue"] = o.Venue.map({"Home": "Away", "Away": "Home"})
    tm = tm.merge(o, on=["Season", "Team", "Opponent", "Venue"], how="left")
    tm[name + "_net"] = tm[name] - tm[name + "_opp"]
    return tm


def reason_gaps(xi):
    """Pairs each absentee with a replacement, then sums the Attack gap by the absentee's reason."""
    rows = []
    for key, g in xi.groupby(["Season", "Team", "Opponent", "Venue"], sort=False):
        e, a = g[g.Set == "E"], g[g.Set == "A"]
        rec = dict(zip(["Season", "Team", "Opponent", "Venue"], key))
        for ra, rs in B.pair_gap(e, a, "Attack"):
            w = ra.Why
            rec[f"G_{w}"] = rec.get(f"G_{w}", 0.0) + ra.Attack - rs.Attack
            rec[f"N_{w}"] = rec.get(f"N_{w}", 0) + 1
        rows.append(rec)
    r = pd.DataFrame(rows).fillna(0)
    return r


def ols(frame, formula, cluster):
    d = frame.dropna(subset=[c for c in frame.columns if c in formula])
    m = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    return m


def jedelhauser(tm, xi):
    s = B.load_squads()
    m = pd.read_csv(DATA / "sofascore_missing_players.csv", encoding="utf-8-sig")
    m = m[m.Type == "missing"]
    side = s.drop_duplicates(["Event", "Venue"])[["Event", "Venue", "Team", "Season"]].assign(Side=lambda d: d.Venue.str.lower())
    m = m.merge(side.drop(columns="Season"), left_on=["Event_ID_Sofascore", "Side"], right_on=["Event", "Side"], how="inner")
    extra = m.rename(columns={"Player_ID_Sofascore": "PID", "Player": "Player_Sofascore"})[["Season", "Team", "PID", "Player_Sofascore"]]
    mv = B.tm_values(pd.concat([s[["Season", "Team", "PID", "Player_Sofascore"]], extra], ignore_index=True))
    mv = mv[mv.MV.notna()].drop_duplicates(["Season", "PID"])
    mv["rank"] = mv.groupby("Season").MV.rank(pct=True, ascending=False)
    mv["Group"] = np.where(mv["rank"] <= .05, "super", np.where(mv["rank"] <= .20, "second", "rest"))
    a = m.rename(columns={"Player_ID_Sofascore": "PID"}).merge(mv[["Season", "PID", "Group"]], on=["Season", "PID"], how="left")
    a = a[~a.Description.fillna("").str.lower().str.contains("doping")]
    c = a.groupby(["Event", "Team"]).agg(Second=("Group", lambda x: int((x == "second").any())), Super=("Group", lambda x: int((x == "super").any())),
                                         nG=("Position", lambda x: int((x == "G").sum())), nD=("Position", lambda x: int((x == "D").sum())),
                                         nM=("Position", lambda x: int((x == "M").sum())), nF=("Position", lambda x: int((x == "F").sum()))).reset_index()
    d = tm[tm.Season <= 2025].merge(c, on=["Event", "Team"], how="left")
    for k in ("Second", "Super", "nG", "nD", "nM", "nF"):
        d[k] = d[k].fillna(0)
    d["Diff"] = d.NPxG - d.NPxGA
    d["Home"] = (d.Venue == "Home").astype(int)
    d["TeamMV"] = d.set_index(["Season", "Team"]).index.map(mv.groupby(["Season", "Team"]).MV.mean() / 1e7)
    pts = pd.read_csv(DATA / "match_results.csv", encoding="utf-8-sig", usecols=["Season", "Home", "Away", "Home_Goals", "Away_Goals"])
    h = pts.assign(Team=pts.Home, P=np.select([pts.Home_Goals > pts.Away_Goals, pts.Home_Goals == pts.Away_Goals], [3, 1], 0))
    w = pts.assign(Team=pts.Away, P=np.select([pts.Away_Goals > pts.Home_Goals, pts.Home_Goals == pts.Away_Goals], [3, 1], 0))
    tab = pd.concat([h, w]).groupby(["Season", "Team"]).P.sum().reset_index()
    tab["Rank"] = tab.groupby("Season").P.rank(ascending=False, method="first")
    prev = tab.assign(Season=tab.Season + 1).rename(columns={"Team": "Opponent", "Rank": "OppRank"})[["Season", "Opponent", "OppRank"]]
    d = d.merge(prev, on=["Season", "Opponent"], how="left")
    d["OppRank"] = d.OppRank.fillna(18)
    d["RoundN"] = d.groupby(["Season", "Team"]).Kickoff.rank(method="first").astype(int)
    d["TeamMV"] = d.TeamMV.fillna(d.TeamMV.mean())
    ctrl = "Second + Super + nG + nD + nM + nF + Home + OppRank + TeamMV + C(Team) + C(Season) + C(RoundN)"
    out = {}
    for label, y, formula in (("npxG difference, their fixed effects", "Diff", ctrl), ("goal difference, their fixed effects", "GD", ctrl),
                              ("rating-adjusted residual", "ResDiff", "Second + Super + nG + nD + nM + nF")):
        m = ols(d.assign(Diff=d.Diff), f"{y} ~ {formula}", "Event")
        ci = m.conf_int()
        out[label] = {k: {"b": float(m.params[k]), "se": float(m.bse[k]), "lo": float(ci.loc[k, 0]), "hi": float(ci.loc[k, 1])} for k in ("Second", "Super")}
        out[label]["n"] = int(m.nobs)
    out["prevalence"] = {"second": float(d.Second.mean()), "super": float(d.Super.mean())}
    print("\n1. Jedelhauser replication"); print(json.dumps(out, indent=0)[:1500], flush=True)
    RES["jedelhauser"] = out


def slope(tm, xi):
    tm = attach_gap(tm, xi, "Attack", "Gap")
    tm = attach_gap(tm, xi, "NPG_Rating_Pre_filled", "GapNPG")
    r = reason_gaps(xi)
    tm = tm.merge(r, on=["Season", "Team", "Opponent", "Venue"], how="left").fillna({c: 0 for c in r.columns})
    out = {}
    tm["TeamSeason"] = tm.Team + tm.Season.astype(str)
    for name, y, x in (("difference on net attack gap", "ResDiff", "Gap_net"), ("npxG for on own gap", "ResFor", "Gap"),
                       ("npxG against on opponent gap", "ResAgainst", "Gap_opp"), ("difference on net npxG-only gap", "ResDiff", "GapNPG_net")):
        m = ols(tm, f"{y} ~ {x}", "TeamSeason")
        ci = m.conf_int().loc[x]
        out[name] = {"beta": float(m.params[x]), "se": float(m.bse[x]), "lo": float(ci[0]), "hi": float(ci[1]), "n": int(m.nobs)}
    reasons = ["suspended", "injured", "rested", "unlisted", "other"]
    m = ols(tm, "ResFor ~ " + " + ".join(f"G_{w}" for w in reasons), "TeamSeason")
    out["own npxG for by reason"] = {w: {"beta": float(m.params[f"G_{w}"]), "se": float(m.bse[f"G_{w}"]), "n_absentees": int(tm[f"N_{w}"].sum())} for w in reasons}
    m = ols(tm, "ResDiff ~ " + " + ".join(f"G_{w}" for w in reasons), "TeamSeason")
    out["difference by reason"] = {w: {"beta": float(m.params[f"G_{w}"]), "se": float(m.bse[f"G_{w}"])} for w in reasons}
    print("\n2. Slope"); print(json.dumps(out, indent=0), flush=True)
    RES["slope"] = out
    return tm


def split_half(tm, xi):
    """Reliability of the gap: npxG+xAG per 90 over a player's last 20 qualifying matches, odd against even matches."""
    s = B.load_squads()
    p = pd.read_csv(DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False, usecols=["Season", "Player", "Team", "Opponent", "Venue", "Minutes", "npxG", "xAG"])
    j = p.merge(s[["Season", "Local_Player", "Team", "Opponent", "Venue", "PID", "Kickoff"]], left_on=["Season", "Player", "Team", "Opponent", "Venue"],
                right_on=["Season", "Local_Player", "Team", "Opponent", "Venue"], how="inner")
    j = j[j.Minutes >= 30].sort_values("Kickoff")
    j["r"] = (j.npxG.fillna(0) + j.xAG.fillna(0)) * 90 / j.Minutes
    hist = {k: (g.Kickoff.values, g.r.values) for k, g in j.groupby("PID")}
    odd, even = np.full(len(xi), np.nan), np.full(len(xi), np.nan)
    for i, (pid, ko) in enumerate(zip(xi.PID.values, xi.Kickoff.values)):
        h = hist.get(pid)
        if h is None:
            continue
        n = np.searchsorted(h[0], ko)
        v = h[1][max(0, n - 20):n]
        if len(v) >= 6:
            odd[i], even[i] = v[::2].mean(), v[1::2].mean()
    xi = xi.assign(_odd=odd, _even=even)
    med = xi.groupby(["Season", "Pos"])[["_odd", "_even"]].transform("median")
    xi["_odd"], xi["_even"] = xi._odd.fillna(med._odd).fillna(0), xi._even.fillna(med._even).fillna(0)
    a, b = gaps(xi, "_odd"), gaps(xi, "_even")
    mask = (a != 0) | (b != 0)
    r = float(np.corrcoef(a[mask], b[mask])[0, 1])
    rel = 2 * r / (1 + r)
    print("\nsplit-half r", round(r, 3), "Spearman-Brown reliability", round(rel, 3), flush=True)
    RES["reliability"] = {"split_half_r": r, "spearman_brown": rel}
    return rel


def position_groups(tm, xi):
    pos = pd.read_csv(DATA / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False, usecols=["Player", "Position_Resolved"])
    mode = pos.dropna().groupby("Player").Position_Resolved.agg(lambda x: x.mode().iat[0])
    s = B.load_squads().dropna(subset=["Local_Player"]).drop_duplicates("PID").set_index("PID").Local_Player
    x = xi[(xi.Set == "E") & (xi.Share >= .5)].copy()
    x["Slot"] = x.PID.map(s).map(mode)
    x["Grp"] = np.select([x.Pos == "G", x.Slot == "CB", x.Pos == "D", x.Pos == "M", x.Pos == "F"], ["GK", "CB", "WD", "MF", "FW"], "MF")
    out = x[~x.Why.isin(["started", "rested"])].groupby(["Season", "Team", "Opponent", "Venue", "Grp"]).size().unstack(fill_value=0).reset_index()
    d = tm.merge(out, on=["Season", "Team", "Opponent", "Venue"], how="left")
    grp = ["GK", "CB", "WD", "MF", "FW"]
    d[grp] = d[grp].fillna(0)
    o = out.rename(columns={"Team": "Opponent", "Opponent": "Team", **{g: "o" + g for g in grp}})
    o["Venue"] = o.Venue.map({"Home": "Away", "Away": "Home"})
    d = d.merge(o, on=["Season", "Team", "Opponent", "Venue"], how="left")
    ogrp = ["o" + g for g in grp]
    d[ogrp] = d[ogrp].fillna(0)
    res = {}
    for y in ("ResFor", "ResAgainst", "ResDiff"):
        m = ols(d, f"{y} ~ " + " + ".join(grp + ogrp), "TeamSeason")
        ci = m.conf_int()
        res[y] = {g: {"b": float(m.params[g]), "lo": float(ci.loc[g, 0]), "hi": float(ci.loc[g, 1])} for g in grp}
    res["n_team_matches_with"] = {g: int((d[g] > 0).sum()) for g in grp}
    print("\n3. Position groups (regular absentee, not rested)"); print(json.dumps(res, indent=0), flush=True)
    RES["position_groups"] = res


def rapm(xi):
    """Position-penalised ridge plus-minus on match npxG difference, refit every ten rounds, using earlier matches only."""
    s = B.load_squads()
    tm = B.team_matches()
    home = tm[tm.Venue == "Home"].assign(Y=lambda d: d.NPxG - d.NPxGA)
    ev = s.groupby(["Season", "Team", "Opponent", "Venue"], as_index=False).agg(Event=("Event", "first"), Kickoff=("Kickoff", "first"))
    home = home.merge(ev, on=["Season", "Team", "Opponent", "Venue"]).sort_values("Kickoff").reset_index(drop=True)
    pids = sorted(s.PID.unique())
    col = {p: i for i, p in enumerate(pids)}
    pos = s[s.Pos.notna()].groupby("PID").Pos.agg(lambda x: x.mode().iat[0]).to_dict()
    penalty = np.array([{"G": 3.0}.get(pos.get(p), 1.0) for p in pids])
    played = s[s.Min > 0]
    idx = {e: g for e, g in played.groupby("Event")}
    rows, cols, vals = [], [], []
    for i, (e, t) in enumerate(zip(home.Event, home.Team)):
        g = idx[e]
        sign = np.where(g.Team.values == t, 1.0, -1.0)
        rows += [i] * len(g)
        cols += [col[p] for p in g.PID]
        vals += list(sign * g.Min.values / 90.0)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(home), len(pids)))
    # Box-score prior: half the player's attack rating relative to his position, on the rating table at refit time.
    prior_tab = xi[["PID", "Kickoff", "Attack", "Pos"]].sort_values("Kickoff")
    y = home.Y.values
    home["Block"] = home.groupby("Season").cumcount() // 100
    blocks = home.groupby(["Season", "Block"], sort=True).Kickoff.min().reset_index().sort_values("Kickoff")
    out = []
    lam = 15.0
    for _, b in blocks.iterrows():
        cut = b.Kickoff
        tr = (home.Kickoff < cut).values
        if tr.sum() < 500:
            continue
        lastq = prior_tab[prior_tab.Kickoff < cut].groupby("PID").agg(a=("Attack", "last"), p=("Pos", "last"))
        lastq["dev"] = lastq.a - lastq.groupby("p").a.transform("mean")
        theta0 = np.array([0.5 * lastq.dev.get(p, 0.0) for p in pids])
        age = (cut - home.Kickoff[tr]).dt.days.values / 365.25
        w = 0.5 ** (age / 2.0)
        Xs = X[tr] @ sp.diags(1 / np.sqrt(penalty))
        target = y[tr] - X[tr] @ theta0
        m = Ridge(alpha=lam, fit_intercept=True).fit(Xs, target, sample_weight=w)
        theta = theta0 + m.coef_ / np.sqrt(penalty)
        end = blocks.Kickoff[blocks.Kickoff > cut].min()
        out.append(pd.DataFrame({"PID": pids, "Kickoff": cut, "RAPM": theta}))
    return pd.concat(out, ignore_index=True)


def candidates(tm, xi):
    rap = rapm(xi)
    # Block values apply from their cut onward; asof with an exact match allowed is safe because the fit uses < cut.
    x = pd.merge_asof(xi.sort_values("Kickoff"), rap.sort_values("Kickoff"), on="Kickoff", by="PID", allow_exact_matches=True)
    x["RAPM"] = x.RAPM.fillna(0.0)
    for name, col in (("gapA", "Attack"), ("gapB", "LogMV_filled"), ("gapC", "SofaQ_filled"), ("gapD", "RAPM")):
        tm = attach_gap(tm, x, col, name)
    absent = xi[(xi.Set == "E") & ~xi.Why.isin(["started", "rested"])].groupby(["Season", "Team", "Opponent", "Venue"]).size().rename("nAbs").reset_index()
    tm = tm.merge(absent, on=["Season", "Team", "Opponent", "Venue"], how="left")
    tm["nAbs"] = tm.nAbs.fillna(0)
    d = tm[tm.nAbs > 0].copy()
    names = {"gapA_net": "attack ratings", "gapB_net": "log market value", "gapC_net": "Sofascore rating", "gapD_net": "plus-minus (RAPM)"}
    d = d.dropna(subset=list(names) + ["ResDiff"])
    out = {}
    test = [s for s in range(2021, 2026)]  # ends 2025 for the market-value common sample
    all_rows = {k: [] for k in names}
    base = []
    for s in test:
        tr, te = d[d.Season < s], d[d.Season == s]
        base.append((te.ResDiff.values - tr.ResDiff.mean()) ** 2)
        for k in names:
            b = np.polyfit(tr[k].values, tr.ResDiff.values, 1)
            all_rows[k].append((te.ResDiff.values - np.polyval(b, te[k].values)) ** 2)
    b0 = np.concatenate(base)
    ev = np.concatenate([d[d.Season == s].Event.values for s in test])
    codes, _ = pd.factorize(ev)
    K = codes.max() + 1
    rng = np.random.default_rng(3)
    w = rng.multinomial(K, np.ones(K) / K, size=2000)
    n = np.bincount(codes, minlength=K).astype(float)
    for k, nm in names.items():
        e = np.concatenate(all_rows[k])
        diff = e - b0
        s_ = np.bincount(codes, diff, K)
        boot = (w * s_).sum(1) / (w * n).sum(1)
        m = ols(d, f"ResDiff ~ {k}", "Event")
        out[nm] = {"mse": float(e.mean()), "mse_null": float(b0.mean()), "d_mse": float(diff.mean()), "lo": float(np.quantile(boot, .025)), "hi": float(np.quantile(boot, .975)),
                   "fullsample_slope": float(m.params[k]), "t": float(m.tvalues[k]), "sd_gap": float(d[k].std())}
    joint = ols(d, "ResDiff ~ gapA_net + gapB_net + gapC_net + gapD_net", "Event")
    out["joint_t"] = {k: float(joint.tvalues[k]) for k in names}
    out["rows"] = len(b0)
    print("\n4. Value measures, walk-forward on absence matches"); print(json.dumps(out, indent=0), flush=True)
    RES["candidates"] = out
    return tm


def main():
    xi = pd.read_pickle(OUT / "xi.pkl")
    tm = base_frame(xi)
    RES["residual_sd"] = float(tm.ResDiff.std())
    jedelhauser(tm, xi)
    tm = slope(tm, xi)
    rel = split_half(tm, xi)
    b = RES["slope"]["difference on net attack gap"]
    RES["slope_corrected"] = {"beta": b["beta"] / rel, "lo": b["lo"] / rel, "hi": b["hi"] / rel}
    print("noise-corrected beta", RES["slope_corrected"], flush=True)
    position_groups(tm, xi)
    tm = candidates(tm, xi)
    tm.to_pickle(OUT / "team_matches.pkl")
    (OUT / "results.json").write_text(json.dumps(RES, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
