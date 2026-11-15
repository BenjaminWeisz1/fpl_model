"""
Set pieces in goal and assist ratings: size, stability, and what the current rating misses (CHANGELOG §196).
Protocol: data/validation/set_piece_diagnostic_20261103/protocol.json. Diagnostic only; no production writes.

    python diagnose_set_pieces.py      # needs validate_position_alpha.py run (its floor-0.025 log) for part C
"""
import json
import sys

import numpy as np
import pandas as pd

from fpl_model import paths

ROOT = paths.ROOT
DATA = paths.CURRENT
MATCHES = paths.RAW / "understat" / "matches"
OUT = paths.VALIDATION / "set_piece_diagnostic_20261103"
LOG = paths.VALIDATION / "position_alpha_20261103" / "log_a0.025.csv"
SET_PIECE = {"FromCorner", "SetPiece", "DirectFreekick"}
GROUP = {"CB": "CB", "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB", "WB": "FB", "DM": "CM", "CM": "CM",
         "AM": "AW", "LM": "AW", "RM": "AW", "LW": "AW", "RW": "AW", "FW": "FW"}
MIN_MINUTES = 900


def match_events():
    """Per (Understat match, player id): open-play / set-piece xG taken and xA created, penalties dropped."""
    rows = []
    for f in sorted(MATCHES.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        mid = int(f.stem)
        ids = {}
        for side in ("h", "a"):
            for r in d["rosters"][side].values():
                ids[(side, r["player"])] = int(r["player_id"])
        for side in ("h", "a"):
            for s in d["shots"][side]:
                if s["situation"] == "Penalty":
                    continue
                sp = s["situation"] in SET_PIECE
                xg = float(s["xG"])
                rows.append((mid, int(s["player_id"]), "sp_xg" if sp else "op_xg", xg))
                a = s.get("player_assisted")
                if a and (side, a) in ids:
                    rows.append((mid, ids[(side, a)], "sp_xa" if sp else "op_xa", xg))
    e = pd.DataFrame(rows, columns=["MatchID_Understat", "Player_ID_Understat", "kind", "v"])
    return e.pivot_table(index=["MatchID_Understat", "Player_ID_Understat"], columns="kind", values="v", aggfunc="sum", fill_value=0.0).reset_index()


def player_seasons():
    p = pd.read_csv(DATA / "understat_match_positions.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=["Season", "MatchID_Understat", "Player", "Player_ID_Understat", "Position", "Is_Starter", "Minutes"])
    p = p[p.Minutes > 0].drop_duplicates(["MatchID_Understat", "Player_ID_Understat"])
    e = match_events()
    n = len(p)
    p = p.merge(e, on=["MatchID_Understat", "Player_ID_Understat"], how="left")
    assert len(p) == n
    for c in ("op_xg", "sp_xg", "op_xa", "sp_xa"):
        p[c] = p[c].fillna(0.0)
    starts = p[p.Is_Starter.astype(bool)]
    mode = starts.groupby(["Season", "Player"]).Position.agg(lambda s: s.mode().iloc[0]).rename("pos")
    s = p.groupby(["Season", "Player"])[["Minutes", "op_xg", "sp_xg", "op_xa", "sp_xa"]].sum().join(mode).reset_index()
    s["group"] = s.pos.map(GROUP)
    for c in ("op_xg", "sp_xg", "op_xa", "sp_xa"):
        s[c + "90"] = s[c] / s.Minutes * 90
    return s[s.group.notna()]


def size(s):
    q = s[s.Season <= 2025]
    t = q.groupby("group")[["op_xg", "sp_xg", "op_xa", "sp_xa"]].sum()
    out = {g: {"sp_share_npxg": round(float(r.sp_xg / (r.sp_xg + r.op_xg)), 3), "sp_share_xa": round(float(r.sp_xa / (r.sp_xa + r.op_xa)), 3)} for g, r in t.iterrows()}
    tot = t.sum()
    out["all"] = {"sp_share_npxg": round(float(tot.sp_xg / (tot.sp_xg + tot.op_xg)), 3), "sp_share_xa": round(float(tot.sp_xa / (tot.sp_xa + tot.op_xa)), 3)}
    # Main takers: top 40 player-seasons by set-piece xA per 90 among 900+ minutes.
    big = q[q.Minutes >= MIN_MINUTES].nlargest(40, "sp_xa90")
    out["top40_sp_xa_players_sp_share_xa"] = round(float(big.sp_xa.sum() / (big.sp_xa + big.op_xa).sum()), 3)
    return out


def pairs(s):
    a = s[s.Minutes >= MIN_MINUTES]
    b = a.assign(Season=a.Season + 1)
    return a.merge(b, on=["Player", "Season"], suffixes=("", "_prev"))


def stability(pr):
    out = {}
    for g, x in list(pr.groupby("group")) + [("all", pr)]:
        out[g] = {c: round(float(np.corrcoef(x[c], x[c + "_prev"])[0, 1]), 3) for c in ("op_xg90", "sp_xg90", "op_xa90", "sp_xa90")}
        out[g]["pairs"] = len(x)
    return out


def r2(y, x):
    if len(y) < 10 or np.std(x) == 0:
        return None
    return round(float(np.corrcoef(y, x)[0, 1] ** 2), 4)


def miss(pr):
    lg = pd.read_csv(LOG, encoding="utf-8-sig")
    lg["act_g"], lg["prd_g"] = lg.npxg90 * lg.weight, lg.pred_goals_90 * lg.weight
    lg["act_a"], lg["prd_a"] = lg.xag90 * lg.weight, lg.pred_ast_90 * lg.weight
    ps = lg.groupby(["Player", "Season"])[["act_g", "prd_g", "act_a", "prd_a", "weight"]].sum().reset_index()
    ps["res_g"] = (ps.act_g - ps.prd_g) / ps.weight
    ps["res_a"] = (ps.act_a - ps.prd_a) / ps.weight
    m = pr[pr.Season >= 2019].merge(ps, on=["Player", "Season"], how="inner")
    for c in ("op_xg90", "sp_xg90", "op_xa90", "sp_xa90"):
        m["d_" + c] = m[c] - m[c + "_prev"]
    out = {"rows": len(m)}
    for g, x in list(m.groupby("group")) + [("all", m)]:
        change = (x.d_sp_xa90.abs() >= 0.05)
        out[g] = {"rows": len(x),
                  "xAG_r2_sp_change": r2(x.res_a, x.d_sp_xa90), "xAG_r2_op_change": r2(x.res_a, x.d_op_xa90),
                  "npxG_r2_sp_change": r2(x.res_g, x.d_sp_xg90), "npxG_r2_op_change": r2(x.res_g, x.d_op_xg90),
                  "taker_changes": int(change.sum()),
                  "xAG_mean_abs_res_taker_change": round(float(x.res_a[change].abs().mean()), 4) if change.any() else None,
                  "xAG_mean_abs_res_other": round(float(x.res_a[~change].abs().mean()), 4),
                  "xAG_res_sd": round(float(x.res_a.std()), 4), "npxG_res_sd": round(float(x.res_g.std()), 4)}
    return out


def main():
    s = player_seasons()
    pr = pairs(s)
    res = {"player_seasons": len(s), "A_size": size(s), "B_stability": stability(pr)}
    if LOG.exists():
        res["C_miss"] = miss(pr)
    else:
        print("part C skipped: run validate_position_alpha.py first", file=sys.stderr)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
