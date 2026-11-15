"""User's form of the position plan: same-player move factors, one-off and two-way.

Protocol, frozen before any 2025-26 error was computed:
data/validation/move_factors_20261101/protocol.json.
Phases, in order:  fit  ->  run  ->  score
`fit` reads rows to 2024 only. `score` reads 2025-26, once. Production files are never written.
"""
from fpl_model import paths
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.player_ratings import validate_role_move_error as vm
from fpl_model.player_ratings.role_shares import role_table

ROOT = paths.ROOT
OUT = paths.VALIDATION / "move_factors_20261101"
BUILD = "fpl_model.player_ratings.build_player_ratings"
KEYS = ["Season", "Team", "Opponent", "Venue", "Player"]
SHRINK, N_BOOT, SEED = 0.5, 5000, 173
STATS = {"npxG": ("act_npxG", "exp_npxG", "NPG_Rating_Pre", "npxg90", "pred_goals_90"),
         "xAG": ("act_xAG", "exp_xAG", "Assist_Rating_Pre", "xag90", "pred_ast_90")}


def fit():
    s = vm.starts_table()
    moves, all_moves = vm.find_moves(s)
    rows, excl = vm.tag_rows(s, all_moves, moves)
    d, _ = vm.add_expected(s, excl)      # seasons 2018-2024 only
    pos_of = {p: g.index.values for p, g in s.groupby("Player_ID_Understat", sort=False)}
    by_p = {p: g.pos.sort_values().values for p, g in all_moves.groupby("pid")}
    out, recs = {}, []
    for _, m in moves.iterrows():
        idx = pos_of[m.pid]
        later = by_p[m.pid][by_p[m.pid] > m.pos]
        stop = later[0] if len(later) else len(idx)
        first = m.pos + vm.W
        if first >= min(stop, len(idx)) or idx[first] not in d.index:
            continue
        r0 = d.loc[idx[first]]
        for j in range(first, min(first + 15, stop)):
            row = idx[j]
            if row in d.index and s.at[row, "Line"] == m.new:
                x = d.loc[row]
                recs.append(dict(pid=m.pid, fwd=bool(m.fwd),
                                 a_npxG=x.act_npxG, e_npxG=x.exp_npxG * r0.NPG_Rating_Pre / x.NPG_Rating_Pre,
                                 a_xAG=x.act_xAG, e_xAG=x.exp_xAG * r0.Assist_Rating_Pre / x.Assist_Rating_Pre))
    t = pd.DataFrame(recs)
    rng = np.random.default_rng(SEED)
    res = {"rows": len(t), "players": int(t.pid.nunique())}
    factors = {"npxG": {}, "xAG": {}}
    for stat in factors:
        for name, flag in [("forward", True), ("back", False)]:
            f = t[t.fwd == flag]
            p = f.groupby("pid")[[f"a_{stat}", f"e_{stat}"]].sum()
            a, e = p.iloc[:, 0].values, p.iloc[:, 1].values
            ix = rng.integers(0, len(p), size=(N_BOOT, len(p)))
            raw = float(a.sum() / e.sum())
            ci = np.percentile(a[ix].sum(1) / e[ix].sum(1), [2.5, 97.5])
            factors[stat][name] = round(raw ** SHRINK, 4)
            res[f"{stat}_{name}"] = dict(rows=len(f), raw=round(raw, 3), raw_ci=[round(float(x), 3) for x in ci],
                                         shrunk=factors[stat][name])
    print(json.dumps(res, indent=1))
    g = s.groupby("Player_ID_Understat", sort=False)
    trig = []
    for _, m in all_moves.iterrows():
        r = g.get_group(m.pid).iloc[m.pos + vm.W - 1]
        if isinstance(r.Player, str):
            trig.append([[int(r.Season), r.Team, r.Opponent, r.Venue, r.Player],
                         dict(ref=m.ref, new=m.new, fwd=bool(m.fwd))])
    (OUT / "move_table.json").write_text(json.dumps(dict(factors=factors, triggers=trig)), encoding="utf-8")
    ones = {k: {d_: 1.0 for d_ in v} for k, v in factors.items()}
    (OUT / "move_table_ones.json").write_text(json.dumps(dict(factors=ones, triggers=trig)), encoding="utf-8")
    (OUT / "fit.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("triggers", len(trig))


def run():
    procs = []
    for tag, table in [("final", "move_table.json"), ("ones", "move_table_ones.json")]:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", GOAL_MODEL_PREMATCH_LOG=str(OUT / f"log_{tag}.csv"),
                   GOAL_MODEL_MOVE_TABLE=str(OUT / table))
        procs.append((tag, subprocess.Popen([sys.executable, "-u", "-X", "utf8", "-m", BUILD], env=env,
                                            stdout=open(OUT / f"run_{tag}.log", "w", encoding="utf-8"),
                                            stderr=subprocess.STDOUT)))
    for tag, p in procs:
        if p.wait() != 0:
            raise SystemExit(f"run {tag} failed")
    a = pd.read_csv(vm.LOG, encoding="utf-8-sig", low_memory=False)
    b = pd.read_csv(OUT / "log_ones.csv", encoding="utf-8-sig", low_memory=False)
    m = a.drop_duplicates(KEYS, keep=False).merge(b.drop_duplicates(KEYS, keep=False), on=KEYS, suffixes=("_a", "_b"))
    gap = max(float(np.abs(m.pred_goals_90_a - m.pred_goals_90_b).max()), float(np.abs(m.pred_ast_90_a - m.pred_ast_90_b).max()))
    print("all-1 factors v baseline: rows", len(m), "max prediction gap", gap)
    if gap != 0:
        raise SystemExit("all-1 check failed")


def cluster(a, b, pl):
    u, inv = np.unique(pl, return_inverse=True)
    ta, tb = np.bincount(inv, a), np.bincount(inv, b)
    ix = np.random.default_rng(SEED).integers(0, len(u), size=(N_BOOT, len(u)))
    return round(float(b.sum() / a.sum() - 1), 5), [round(float(x), 5) for x in np.percentile(tb[ix].sum(1) / ta[ix].sum(1) - 1, [2.5, 97.5])]


def score():
    base = pd.read_csv(vm.LOG, encoding="utf-8-sig", low_memory=False)
    fin = pd.read_csv(OUT / "log_final.csv", encoding="utf-8-sig", low_memory=False)
    m = base.merge(fin, on=KEYS, suffixes=("_b", "_f"), validate="one_to_one")
    m = m[m.Season >= 2025].copy()
    t = role_table()
    disp, up = [], []
    for k in zip(m.Season, m.Team, m.Opponent, m.Venue, m.Player):
        r = t.get(k)
        d = r is not None and r[4] >= 10 and int(r[0].argmax()) != r[3]
        disp.append(d)
        up.append(d and int(r[0].argmax()) > r[3])
    m["disp"], m["up"] = disp, up

    s = vm.starts_table()
    _, all_moves = vm.find_moves(s)
    pos_of = {p: g for p, g in s.groupby("Player_ID_Understat", sort=False)}
    after = set()
    for pid, g in all_moves.groupby("pid"):
        ps = sorted(g.pos)
        ss = pos_of[pid]
        for i, p in enumerate(ps):
            stop = ps[i + 1] if i + 1 < len(ps) else len(ss)
            for j in range(p + vm.W, min(p + vm.W + 20, stop)):
                r = ss.iloc[j]
                after.add((int(r.Season), r.Team, r.Opponent, r.Venue, r.Player))
    m["after"] = [k in after for k in zip(m.Season, m.Team, m.Opponent, m.Venue, m.Player)]

    res = dict(rows=len(m), displaced=int(m.disp.sum()), displaced_players=int(m[m.disp].Player.nunique()),
               after_trigger=int(m.after.sum()), rows_changed=int((m.pred_goals_90_b != m.pred_goals_90_f).sum()))
    for stat, (_, _, _, act, pred) in STATS.items():
        for c in "bf":
            m[f"se_{c}"] = ((m[f"{act}_{c}"] - m[f"{pred}_{c}"]) * m[f"weight_{c}"]) ** 2
        d = m[m.disp]
        ch, ci = cluster(d.se_b.values, d.se_f.values, d.Player.values)
        out = dict(displaced=ch, ci=ci)
        for name, f in [("up", d[d.up]), ("down", d[~d.up])]:
            x, xci = cluster(f.se_b.values, f.se_f.values, f.Player.values)
            out[name] = dict(rows=len(f), change=x, ci=xci)
        a = m[m.after]
        x, xci = cluster(a.se_b.values, a.se_f.values, a.Player.values)
        out["after_trigger"] = dict(change=x, ci=xci)
        out["all_rows"] = round(float(m.se_f.sum() / m.se_b.sum() - 1), 6)
        out["pass"] = bool(ci[1] < 0 and out["up"]["change"] <= .005 and out["down"]["change"] <= .005 and out["all_rows"] <= .0005)
        res[stat] = out
        print(stat, json.dumps(out))
    res["verdict"] = "PASS" if res["npxG"]["pass"] or res["xAG"]["pass"] else "STOP"
    print({k: res[k] for k in ["rows", "displaced", "displaced_players", "after_trigger", "rows_changed"]})
    print("GATE", res["verdict"])
    (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    r = pd.read_csv(OUT / "log_final.csv.ratings.csv", encoding="utf-8-sig")
    b = pd.read_csv(str(vm.LOG) + ".ratings.csv", encoding="utf-8-sig")
    j = r.merge(b, on=["Player", "Team"], suffixes=("_f", "_b"))
    j["ch"] = j["NPG/90_f"] / j["NPG/90_b"] - 1
    print(j[j.ch.abs() > 1e-9][["Player", "Team", "NPG/90_b", "NPG/90_f", "Assists/90_b", "Assists/90_f"]].round(3).to_string(index=False))


if __name__ == "__main__":
    {"fit": fit, "run": run, "score": score}[sys.argv[1]]()
