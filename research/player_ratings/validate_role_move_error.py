"""Step 0 of the position-switch plan: does the goal/assist rating miss after a position move?

Protocol, frozen before any error was computed:
data/validation/role_move_error_20261101/protocol.json. Reads the pre-match log written by
GOAL_MODEL_PREMATCH_LOG=<log> python -m fpl_model.player_ratings.build_player_ratings
Scores seasons 2018-2024 only; 2025-26 stays untouched. Changes no production file.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd

from fpl_model import paths

ROOT = paths.ROOT
DATA = paths.CURRENT
OUT = paths.VALIDATION / "role_move_error_20261101"
LOG = paths.VALIDATION / "role_move_error_20261101" / "prematch_log.csv"

LINE = {"CB": "CB", "LB": "WD", "RB": "WD", "WB": "WD", "DM": "CM", "CM": "CM", "AM": "ATT",
        "LM": "ATT", "RM": "ATT", "LW": "ATT", "RW": "ATT", "FW": "FW"}
RANK = {"CB": 0, "WD": 1, "CM": 2, "ATT": 3, "FW": 4}
BIG = {("ATT", "WD"), ("CB", "CM"), ("CM", "WD"), ("FW", "WD"), ("CM", "FW")}
MIN_REF, W, REF_SPAN = 15, 5, 38
SEASONS = (2018, 2024)
N_BOOT, SEED, CI = 5000, 169, 97.5
KEYS = ["Season", "Team", "Opponent", "Venue"]


def starts_table():
    """Starts with date, line and pre-match log values, in player-date order."""
    u = pd.read_csv(DATA / "understat_match_positions.csv", encoding="utf-8-sig", low_memory=False)
    s = u[(u.Is_Starter == True) & u.Position.isin(LINE)].copy()  # noqa: E712
    s["Date"] = pd.to_datetime(s["Date"])
    s["Line"] = s.Position.map(LINE)
    s = s[KEYS + ["Player_ID_Understat", "Date", "Line"]]
    pos = pd.read_csv(DATA / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False,
                      usecols=KEYS + ["Player", "Player_ID_Understat"])
    pos = pos.drop_duplicates(KEYS + ["Player_ID_Understat"])
    n = len(s)
    s = s.merge(pos, on=KEYS + ["Player_ID_Understat"], how="left")
    assert len(s) == n, "position join fanned out"
    log = pd.read_csv(LOG, encoding="utf-8-sig", low_memory=False)
    log = log.drop_duplicates(KEYS + ["Player"], keep=False)
    s = s.merge(log.drop(columns=["Is_Starter"]), on=KEYS + ["Player"], how="left")
    assert len(s) == n, "log join fanned out"
    print(f"starts {n}; matched to the log {s.pred_goals_90.notna().mean():.1%} "
          f"(2018-2024: {s[s.Season.between(*SEASONS)].pred_goals_90.notna().mean():.1%})")
    return s.sort_values(["Player_ID_Understat", "Date"]).reset_index(drop=True)


def find_moves(s):
    moves = []
    for pid, g in s.groupby("Player_ID_Understat", sort=False):
        L, T = g.Line.values, g.Team.values
        i, ref_from = MIN_REF, 0
        while i <= len(L) - W:
            ref = Counter(L[max(ref_from, i - REF_SPAN):i]).most_common(1)[0][0]
            win = L[i:i + W]
            if len(set(win)) == 1 and win[0] != ref and T[i - 1] == T[i]:
                moves.append(dict(pid=pid, first=int(g.index[i]), pos=i, ref=ref, new=win[0],
                                  season=int(g.Season.iloc[i]), fwd=RANK[win[0]] > RANK[ref],
                                  big=tuple(sorted((ref, win[0]))) in BIG))
                ref_from = i
                i += W
            else:
                i += 1
    m = pd.DataFrame(moves)
    return m[m.season.between(*SEASONS)].reset_index(drop=True), m


def tag_rows(s, all_moves, moves):
    """Row indices per window, for each kept move; each stops at the player's next move."""
    by_p = {p: g.sort_values("pos") for p, g in all_moves.groupby("pid")}
    out = {"primary": [], "secondary": [], "falsification": []}
    excl = set()
    for pid, g in s.groupby("Player_ID_Understat", sort=False):
        if pid not in by_p:
            continue
        idx, L = g.index.values, g.Line.values
        ps = by_p[pid].pos.values
        for _, r in by_p[pid].iterrows():
            excl.update(idx[max(0, r.pos - 15):r.pos + 20])
        for _, r in moves[moves.pid == pid].iterrows():
            later = ps[ps > r.pos]
            stop = later[0] if len(later) else len(L)
            for j in range(r.pos, min(r.pos + 15, stop)):
                if L[j] == r.new:
                    out["primary"].append((r.name, idx[j]))
            for j in range(r.pos + W, min(r.pos + W + 10, stop)):
                out["secondary"].append((r.name, idx[j]))
            for j in range(max(0, r.pos - 15), r.pos):
                out["falsification"].append((r.name, idx[j]))
    return out, excl


def add_expected(s, excl):
    d = s[s.Season.between(*SEASONS) & s.pred_goals_90.notna()].copy()
    ctrl = ~d.index.isin(excl)
    table = {}
    for stat, rating, act, pred in [("npxG", "NPG_Rating_Pre", "npxg90", "pred_goals_90"),
                                    ("xAG", "Assist_Rating_Pre", "xag90", "pred_ast_90")]:
        d[f"act_{stat}"] = d[act] * d.weight
        raw = d[pred] * d.weight
        edges = np.unique(d.loc[ctrl, rating].quantile(np.linspace(0, 1, 11)).values)
        b = np.clip(np.searchsorted(edges, d[rating].values, side="right") - 1, 0, len(edges) - 2)
        cal = d.loc[ctrl, f"act_{stat}"].groupby(b[ctrl]).sum() / raw[ctrl].groupby(b[ctrl]).sum()
        d[f"exp_{stat}"] = raw * pd.Series(b, index=d.index).map(cal)
        d[f"raw_{stat}"] = raw
        table[stat] = cal.round(3).to_dict()
    print(f"control rows {int(ctrl.sum())} of {len(d)}; decile calibration {table}")
    return d, table


def score(d, moves, rows, keep):
    """Ratios by direction with the bootstrap of D, on the moves in `keep`."""
    t = pd.DataFrame(rows, columns=["move", "row"])
    t = t[t.move.isin(moves.index[keep]) & t.row.isin(d.index)]
    t = t.join(moves[["fwd", "pid"]], on="move").join(
        d[["act_npxG", "exp_npxG", "raw_npxG", "act_xAG", "exp_xAG", "raw_xAG"]], on="row")
    res = dict(moves=int(t.move.nunique()), players=int(t.pid.nunique()), rows=len(t),
               moves_fwd=int(t[t.fwd].move.nunique()), moves_back=int(t[~t.fwd].move.nunique()))
    rng = np.random.default_rng(SEED)
    for stat in ["npxG", "xAG"]:
        p = t.groupby(["pid", "fwd"])[[f"act_{stat}", f"exp_{stat}"]].sum().unstack("fwd", fill_value=0.0)
        a_b, a_f = p[(f"act_{stat}", False)].values, p[(f"act_{stat}", True)].values
        e_b, e_f = p[(f"exp_{stat}", False)].values, p[(f"exp_{stat}", True)].values
        fw, bk = a_f.sum() / e_f.sum(), a_b.sum() / e_b.sum()
        ix = rng.integers(0, len(p), size=(N_BOOT, len(p)))
        bf = a_f[ix].sum(1) / e_f[ix].sum(1)
        bb = a_b[ix].sum(1) / e_b[ix].sum(1)
        q = [(100 - CI) / 2, 100 - (100 - CI) / 2]
        lo, hi = np.percentile(np.log(bf) - np.log(bb), q)
        res[stat] = dict(forward=round(fw, 3), back=round(bk, 3),
                         forward_ci=[round(x, 3) for x in np.percentile(bf, q)],
                         back_ci=[round(x, 3) for x in np.percentile(bb, q)],
                         raw_forward=round(t[t.fwd][f"act_{stat}"].sum() / t[t.fwd][f"raw_{stat}"].sum(), 3),
                         raw_back=round(t[~t.fwd][f"act_{stat}"].sum() / t[~t.fwd][f"raw_{stat}"].sum(), 3),
                         exp_D=round(float(np.exp(np.log(fw) - np.log(bk))), 3),
                         exp_D_ci=[round(float(np.exp(lo)), 3), round(float(np.exp(hi)), 3)])
    return res


def main():
    s = starts_table()
    moves, all_moves = find_moves(s)
    print(f"moves {len(moves)}; big-gap {int(moves.big.sum())}; "
          f"big-gap forward {int((moves.big & moves.fwd).sum())}, back {int((moves.big & ~moves.fwd).sum())}")
    rows, excl = tag_rows(s, all_moves, moves)
    d, cal = add_expected(s, excl)
    results = {"decile_calibration": cal}
    for name, keep in [("big_gap", moves.big), ("all_pairs", pd.Series(True, index=moves.index))]:
        for win in rows:
            results[f"{name}_{win}"] = score(d, moves, rows[win], keep)
            print(f"\n{name} / {win}: {json.dumps(results[f'{name}_{win}'])}")
    p = results["big_gap_primary"]
    gate = {k: bool(p[k]["exp_D_ci"][0] > 1 and p[k]["exp_D"] >= 1.15) for k in ["npxG", "xAG"]}
    results["gate"] = dict(by_stat=gate, verdict="PASS" if any(gate.values()) else "STOP")
    print(f"\nGATE: {results['gate']}")
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    moves.to_csv(OUT / "moves.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
