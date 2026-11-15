"""
Scoring for the position-rates overnight session (CHANGELOG §245-§247; data/validation/position_rates_20261114/).

    python validate_position_rates.py phase1     # §245: new control log v the earlier controls, descriptive
    python validate_position_rates.py phase2     # §246: slot rates v the file, pre-registered (phase2/protocol.json)

Entrants. A player's logged rows are split into segments at each point the loop takes a transfers.csv rating: his
first logged row (the latest transfers.csv row with Season <= that row's season, as the loop picks it) and the first
row of a season with a 'second season' row once he already has a rating (the re-blend). The segment's entry type comes
from that transfers.csv row: youth, second season, missed season, arrival (non-PL League, Season >= 2017, Notes not
youth / second season / missed season) split into fitted (an Adopted row of the priors file the log used) and unfitted
(the coefficient formula); anything else (keepers, 2016 baseline rows, PL-listed rows) is "other" and not an entrant.
Entrant rows = a segment's rows in its first two logged seasons (seasons S0 and S0 + 1, S0 = the season of the
segment's first row).

Scores: MSE of match values, (prediction per 90 - actual per 90) x minutes / 90, summed; change = candidate / control
- 1; player-cluster bootstrap, 2,000 draws, seed 42. Calibration = sum actual / sum predicted.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "current"
OUT = ROOT / "data" / "validation" / "position_rates_20261114"
TARGETS = {"npxG": ("npxg90", "pred_goals_90", "NPG_Rating_Pre"), "xAG": ("xag90", "pred_ast_90", "Assist_Rating_Pre")}
GROUP = {"CB": "CB", "FB": "FB", "WB": "FB", "DM": "CM", "CM": "CM", "AM": "AW", "W": "AW", "WM": "AW", "FW": "FW"}
KEY = ["Player", "Team", "Opponent", "Venue", "MatchID", "Season"]
ENTRY_TYPES = ["youth", "second season", "missed season", "arrival unfitted", "arrival fitted"]
N_BOOT, SEED = 2000, 42


def load_log(path):
    log = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    return log


def aligned(a, b):
    return len(a) == len(b) and all((a[c].to_numpy() == b[c].to_numpy()).all() for c in KEY)


def _branch(r):
    """The transfers.csv branch build_player_ratings.py puts a row in (initial_ratings_df)."""
    notes = r.Notes if isinstance(r.Notes, str) else ""
    if r.Pos == "GK" and notes != "second season":
        return "other"
    if r.Pos == "GK":
        return None   # GK 'second season' rows are not in initial_ratings_df... except via second_season_df (Pos != GK)
    if notes == "second season":
        return "second season"
    if r.Season == 2016:
        return "other"
    if notes == "youth":
        return "youth"
    if notes == "missed season":
        return "missed season"
    if r.Season > 2016:
        return "arrival" if r.League != "Premier League" else "other"
    return None


def segments(log, priors_file):
    """Per log row: entry type ('' = not an entrant row), segment start season, segment id (-1 = none),
    transfers.csv position of the segment and its season, and whether the row is in the segment's first two seasons."""
    t = pd.read_csv(DATA / "transfers.csv", encoding="utf-8-sig")
    t["branch"] = [_branch(r) for r in t.itertuples()]
    t = t[t["branch"].notna()]
    pri = pd.read_csv(priors_file, encoding="utf-8-sig")
    adopted = set(zip(pri.loc[pri["Adopted"].astype(str).str.lower() == "true", "Player"],
                      pri.loc[pri["Adopted"].astype(str).str.lower() == "true", "Season"]))
    init = t[t["branch"] != "second season"]
    init = pd.concat([init, t[t["branch"] == "second season"]])    # initial_ratings_df holds every branch
    by_player = {p: g.sort_values("Season", kind="stable") for p, g in init.groupby("Player")}
    second = {(r.Player, int(r.Season)): r for r in t[t["branch"] == "second season"].itertuples()}
    players = log["Player"].to_numpy()
    seasons = log["Season"].to_numpy()
    n = len(log)
    etype = np.full(n, "", dtype=object)
    seg_id = np.full(n, -1)
    seg_pos = np.full(n, None, dtype=object)
    seg_s0 = np.zeros(n, dtype=int)
    seg_tseason = np.zeros(n, dtype=int)
    order = np.argsort(players, kind="stable")
    sid = -1
    cur = {}
    seen_season = {}
    for i in order:
        p, s = players[i], int(seasons[i])
        start = None
        if p not in cur:
            g = by_player.get(p)
            if g is not None:
                g = g[g["Season"] <= s]
                if len(g):
                    r = g.iloc[-1]
                    start = r
        elif (p, s) in second and seen_season.get(p) != s:
            start = second[(p, s)]
        seen_season[p] = s
        if start is not None:
            sid += 1
            b = start["branch"] if isinstance(start, pd.Series) else start.branch
            ps = int(start["Season"] if isinstance(start, pd.Series) else start.Season)
            pos = start["Pos"] if isinstance(start, pd.Series) else start.Pos
            if b == "arrival":
                b = "arrival fitted" if (p, ps) in adopted else "arrival unfitted"
            cur[p] = (sid, b, s, pos, ps)
        if p in cur:
            sid_, b, s0, pos, ps = cur[p]
            seg_id[i], seg_s0[i], seg_pos[i], seg_tseason[i] = sid_, s0, pos, ps
            if b in ENTRY_TYPES and s <= s0 + 1:
                etype[i] = b
    return {"type": etype, "seg": seg_id, "s0": seg_s0, "pos": seg_pos, "tseason": seg_tseason}


def pos_group(pos):
    if not isinstance(pos, str):
        return None
    return GROUP.get(pos.split(",")[0].strip())


def err(log, t):
    a, p, _ = TARGETS[t]
    w = log["weight"].to_numpy()
    return ((log[p].to_numpy() - log[a].to_numpy()) * w) ** 2


def boot(base, cand, players, n=N_BOOT, seed=SEED):
    if len(base) == 0:
        return None, [None, None]
    df = pd.DataFrame({"p": players, "b": base, "c": cand}).groupby("p")[["b", "c"]].sum()
    b, c = df.b.to_numpy(), df.c.to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(b), size=(n, len(b)))
    d = c[idx].sum(1) / b[idx].sum(1) - 1
    return round(float(c.sum() / b.sum() - 1) * 100, 4), [round(float(x) * 100, 4) for x in np.percentile(d, [2.5, 97.5])]


def rel(c, b, m):
    return round(float(c[m].sum() / b[m].sum() - 1) * 100, 4) if m.any() else None


def calib(log, t, m):
    a, p, _ = TARGETS[t]
    w = log["weight"].to_numpy()
    pred = (log[p].to_numpy() * w)[m].sum()
    return round(float((log[a].to_numpy() * w)[m].sum() / pred), 4) if pred > 0 else None


# ── Phase 1 (§245): descriptive ───────────────────────────────────────────────

def phase1():
    d = OUT / "phase1"
    new = load_log(d / "log_control.csv")
    refs = {"check_default (§235, 12 Nov, pre-C3, old OOT priors)":
            ROOT / "data" / "validation" / "defender_npxg_20261112" / "check_default.csv",
            "old OOT priors, today's code and data (C3 on)": d / "log_oldoot.csv",
            "frozen production priors (in-sample 2019-2025; the plan's fallback)": d / "log_control_frozen.csv"}
    oot_new = pd.read_csv(d / "oot_priors.csv", encoding="utf-8-sig")
    oot_old = pd.read_csv(ROOT / "data" / "validation" / "new_player_priors_20261028" / "oot_priors.csv",
                          encoding="utf-8-sig")
    res = {"priors": {}, "logs": {}}
    # starting ratings by league group
    m = oot_new.merge(oot_old, on=["Player", "Season"], how="left", suffixes=("", "_old"))
    m["old_adopted"] = m["Adopted_old"].fillna(False).astype(bool)
    for (grp, ps), g in m[m["Adopted"]].groupby(["Group", "Param_Set"]):
        res["priors"][f"{grp} / {ps}"] = {
            "rows": int(len(g)), "seasons": f"{int(g.Season.min())}-{int(g.Season.max())}",
            "had_fitted_prior_before": int(g["old_adopted"].sum()),
            "mean_npxG_new": round(float(g.Initial_NPG_Rating.mean()), 4),
            "mean_npxG_old_where_fitted": (round(float(g.loc[g.old_adopted, "Initial_NPG_Rating_old"].mean()), 4)
                                           if g.old_adopted.any() else None),
            "mean_xAG_new": round(float(g.Initial_Assist_Rating.mean()), 4),
            "mean_xAG_old_where_fitted": (round(float(g.loc[g.old_adopted, "Initial_Assist_Rating_old"].mean()), 4)
                                          if g.old_adopted.any() else None)}
    # the loop's own starting rating per arrival (first logged row), new v reference logs
    seg = segments(new, d / "oot_priors.csv")
    grp_of = dict(zip(zip(oot_new.Player, oot_new.Season), oot_new.Group))
    row_group = np.array([grp_of.get((p, s)) if t.startswith("arrival") else None
                          for p, s, t in zip(new.Player, seg["tseason"], seg["type"])], dtype=object)
    players = new["Player"].to_numpy()
    season = new["Season"].to_numpy()
    test = season >= 2023
    for name, path in refs.items():
        ref = load_log(path)
        if not aligned(ref, new):
            res["logs"][name] = "not row-aligned"
            continue
        out = {}
        for t in TARGETS:
            eb, ec = err(ref, t), err(new, t)
            cell = {"all_rows_2023_26": boot(eb[test], ec[test], players[test]),
                    "entrant_rows_2023_26": boot(eb[test & (seg["type"] != "")], ec[test & (seg["type"] != "")],
                                                 players[test & (seg["type"] != "")])}
            for g in ["top3", "ligue1", "championship", "other_europe", "rest_of_world"]:
                mm = test & (row_group == g)
                cell[f"arrivals_{g}"] = {"change_pct": rel(ec, eb, mm), "rows": int(mm.sum()),
                                         "calibration_ref": calib(ref, t, mm), "calibration_new": calib(new, t, mm)}
            for et in ENTRY_TYPES:
                mm = test & (seg["type"] == et)
                cell[f"type_{et}"] = {"change_pct": rel(ec, eb, mm), "rows": int(mm.sum())}
            out[t] = cell
        res["logs"][name] = out
    (d / "phase1_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


# ── Phase 2 (§246): pre-registered, phase2/protocol.json ─────────────────────

ARMS = ["slots", "slots3"]
GROUPS5 = ["CB", "FB", "CM", "AW", "FW"]


def calib_balance(log, t, m, grp):
    """D = sum_g A_g |cal_g - 1| / sum A_g over position groups (protocol 'on balance'), plus the per-group values."""
    a, p, _ = TARGETS[t]
    w = log["weight"].to_numpy()
    act, pred = log[a].to_numpy() * w, log[p].to_numpy() * w
    per, num, den = {}, 0.0, 0.0
    for g in GROUPS5:
        mm = m & (grp == g)
        if not mm.any() or pred[mm].sum() <= 0:
            continue
        cal = act[mm].sum() / pred[mm].sum()
        per[g] = round(float(cal), 4)
        num += act[mm].sum() * abs(cal - 1)
        den += act[mm].sum()
    return (round(float(num / den), 5) if den else None), per


def target_consistency(log):
    """The rates' inputs (position_rates.starts_frame) v the log's own targets on the same starts, by season and slot."""
    import position_rates as pr
    s = pr.starts_frame()
    lg = log.copy()
    lg["npxG_m"], lg["xAG_m"] = lg["npxg90"] * lg["weight"], lg["xag90"] * lg["weight"]
    m = s.merge(lg[["Player", "Team", "Opponent", "Venue", "Season", "npxG_m", "xAG_m"]].drop_duplicates(
        ["Player", "Team", "Opponent", "Venue", "Season"]), on=["Player", "Team", "Opponent", "Venue", "Season"])
    g = m.groupby(["Season", "Slot"])[["npxG", "npxG_m", "xAG", "xAG_m"]].sum()
    return {"starts_matched": int(len(m)), "starts_in_rates": int(len(s)),
            "max_rel_diff_npxG": float(((g.npxG - g.npxG_m).abs() / g.npxG_m.abs().clip(lower=1e-9)).max()),
            "max_rel_diff_xAG": float(((g.xAG - g.xAG_m).abs() / g.xAG_m.abs().clip(lower=1e-9)).max())}


def fitted_settings(pdir):
    out = {}
    for f in sorted(Path(pdir).glob("S*_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[f.stem] = {t: {k: round(float(d[t][k]), 4) for k in ("b", "M0", "gamma")} for t in TARGETS}
    return out


def phase2():
    d = OUT / "phase2"
    ctrl = load_log(OUT / "phase1" / "log_control.csv")
    seg = segments(ctrl, OUT / "phase1" / "oot_priors.csv")
    grp = np.array([pos_group(p) for p in seg["pos"]], dtype=object)
    players = ctrl["Player"].to_numpy()
    season = ctrl["Season"].to_numpy()
    ent = seg["type"] != ""
    train = ent & (seg["s0"] >= 2018) & (seg["s0"] <= 2022)
    test = ent & (seg["s0"] >= 2023)
    late = ent & (seg["s0"] >= 2025)
    allt = season >= 2023
    res = {"population": {"train_rows": int(train.sum()), "test_rows": int(test.sum()), "all_rows_test": int(allt.sum()),
                          "segments_by_type_test": {et: int(len(set(seg["seg"][test & (seg["type"] == et)])))
                                                    for et in ENTRY_TYPES}},
           "target_consistency": target_consistency(ctrl), "arms": {},
           "settings": {"control": fitted_settings(OUT / "phase1" / "oot_params")}}
    seg_first = {}
    for i in np.argsort(players, kind="stable"):
        if seg["seg"][i] >= 0 and seg["seg"][i] not in seg_first:
            seg_first[seg["seg"][i]] = i
    for arm in ARMS:
        path = d / f"log_{arm}.csv"
        if not path.exists():
            res["arms"][arm] = "missing"
            continue
        lg = load_log(path)
        assert aligned(lg, ctrl), arm
        seg_a = segments(lg, d / arm / "oot_priors.csv")
        assert (seg_a["type"] == seg["type"]).all(), "entry types differ between arms"
        out = {}
        for t in TARGETS:
            eb, ec = err(ctrl, t), err(lg, t)
            c = {"train_pct": rel(ec, eb, train)}
            c["test_pct"], c["test_ci"] = boot(eb[test], ec[test], players[test])
            c["all_rows_test_pct"], c["all_rows_test_ci"] = boot(eb[allt], ec[allt], players[allt])
            c["entrants_2025_26_pct"] = rel(ec, eb, late)
            c["by_type_test"] = {et: rel(ec, eb, test & (seg["type"] == et)) for et in ENTRY_TYPES}
            c["by_type_train"] = {et: rel(ec, eb, train & (seg["type"] == et)) for et in ENTRY_TYPES}
            c["by_group_test"] = {g: rel(ec, eb, test & (grp == g)) for g in GROUPS5}
            dc, pc = calib_balance(ctrl, t, test, grp)
            da, pa = calib_balance(lg, t, test, grp)
            c["calibration_test"] = {"D_control": dc, "D_arm": da, "by_group_control": pc, "by_group_arm": pa,
                                     "all_entrants_control": calib(ctrl, t, test), "all_entrants_arm": calib(lg, t, test)}
            # starting rating change for CM- and WB-first-listed entrants (first row of each segment)
            rcol = TARGETS[t][2]
            sc = {}
            for pos in ["CM", "WB", "FB", "W", "WM", "DM", "AM", "FW", "CB"]:
                idx = [i for s_, i in seg_first.items() if seg["type"][i] != "" and isinstance(seg["pos"][i], str)
                       and seg["pos"][i].split(",")[0].strip() == pos]
                if idx:
                    r0, r1 = ctrl[rcol].to_numpy()[idx], lg[rcol].to_numpy()[idx]
                    ok = r0 > 0
                    sc[pos] = {"segments": len(idx), "mean_ratio": round(float(np.mean(r1[ok] / r0[ok])), 4)}
            c["start_ratio_by_first_pos"] = sc
            out[t] = c
        out["train_sum_pct"] = round(out["npxG"]["train_pct"] + out["xAG"]["train_pct"], 4)
        res["arms"][arm] = out
        res["settings"][arm] = fitted_settings(d / arm / "oot_params")
    done = [a for a in ARMS if isinstance(res["arms"][a], dict)]
    if done:
        chosen = min(done, key=lambda a: res["arms"][a]["train_sum_pct"])
        res["chosen_window"] = chosen
        a = res["arms"][chosen]
        per = {}
        for t in TARGETS:
            c = a[t]
            per[t] = {"entrant_improves": c["test_pct"] < 0, "entrant_ci_below_0": c["test_ci"][1] < 0,
                      "all_rows_not_worse": not (c["all_rows_test_ci"][0] > 0),
                      "calibration_towards_1": c["calibration_test"]["D_arm"] < c["calibration_test"]["D_control"]}
        joint = (all(per[t]["entrant_improves"] for t in TARGETS) and any(per[t]["entrant_ci_below_0"] for t in TARGETS)
                 and all(per[t]["all_rows_not_worse"] for t in TARGETS)
                 and all(per[t]["calibration_towards_1"] for t in TARGETS))
        res["rule"] = {"conditions": per, "joint_recommend": joint,
                       "single_target_recommend": {t: (not joint) and all(per[t].values()) for t in TARGETS}}
    (d / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"phase1": phase1, "phase2": phase2}[sys.argv[1]]()
