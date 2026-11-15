"""
Arrivals' starting rating: let the position part follow the positions played at the new club (CHANGELOG §238).
Protocol, frozen before scoring: data/validation/arrival_position_20261113/protocol.json.

    python validate_arrival_position.py    # -> data/validation/arrival_position_20261113/results.json

Exact replay of production's rule from the production pre-match log (§235's check_default.csv). The loop is linear in
its rating, so a change d to an arrival's starting rating moves his rating before row t by s_t x d, s_t = the share
of the start still in the rating. The candidate moves the position part of the start from the listed position's rate
towards the minutes-weighted rate of the slots he has played since arriving:
    rating'_t = rating_t + c x s_t x (p_t - p_listed),
    p_t = (Kpos x p_listed + sum w_j slot_rate_j) / (Kpos + sum w_j)   over earlier rows of the arrival segment.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.player_ratings import validate_position_alpha as vpa  # noqa: E402  (boot, rel, groups)

ROOT = vpa.ROOT
DATA = ROOT / "data" / "current"
OUT = ROOT / "data" / "validation" / "arrival_position_20261113"
LOG = ROOT / "data" / "validation" / "defender_npxg_20261112" / "check_default.csv"
K = 16.0
FLOOR = {"npxG": 0.035, "xAG": 0.025}
FLOOR_CB_NPXG = 0.018
PAIRS = {"npxG": ("npxg90", "pred_goals_90", "NPG_Rating_Pre", "NPG/90"),
         "xAG": ("xag90", "pred_ast_90", "Assist_Rating_Pre", "Assists/90")}
SLOT_CANON = {"CB": "CB", "LB": "FB", "RB": "FB", "WB": "WB", "DM": "DM", "CM": "CM", "AM": "AM",
              "LM": "WM", "RM": "WM", "LW": "W", "RW": "W", "FW": "FW", "GK": "GK"}
POS_GROUP = {"CB": "CB", "LB": "FB", "RB": "FB", "WB": "FB", "DM": "CM", "CM": "CM",
             "AM": "AW", "LM": "AW", "RM": "AW", "LW": "AW", "RW": "AW", "FW": "FW"}
C_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
KPOS_GRID = [1, 2, 4, 8, 16, 32]
TRAIN, TEST = (2017, 2022), (2023, 2026)
BIG_GAP = 0.05


def load():
    log = pd.read_csv(LOG, encoding="utf-8-sig", low_memory=False)
    # The log is in match order with players interleaved; group each player's rows together, keeping their
    # chronological order (a stable sort), so a segment is a contiguous run of rows.
    log = log.sort_values("Player", kind="stable").reset_index(drop=True)
    pos = pd.read_csv(DATA / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False,
                      usecols=["Season", "Team", "Opponent", "Venue", "Player", "Position_Resolved"])
    pos = pos.drop_duplicates(["Season", "Team", "Opponent", "Venue", "Player"])
    m = log.merge(pos, on=["Season", "Team", "Opponent", "Venue", "Player"], how="left")
    assert len(m) == len(log)
    log["Slot"] = m["Position_Resolved"].map(SLOT_CANON).to_numpy(dtype=object)
    # centre-back rows for the npxG floor, as production (§235): match position, else previous match, else transfers group
    tg = vpa.groups(log)
    g = m["Position_Resolved"].map(POS_GROUP).to_numpy(dtype=object)
    grp, last = [], {}
    for p, gi, ti in zip(log["Player"].to_numpy(), g, tg):
        if not isinstance(gi, str):
            gi = last.get(p, ti)
        if isinstance(gi, str):
            last[p] = gi
        grp.append(gi)
    log["is_cb"] = np.array([x == "CB" for x in grp])
    log["w"] = log["weight"]
    log["n"] = log.groupby("Player").cumcount()
    for t, (a, p, r, _) in PAIRS.items():
        log[f"f_{t}"] = np.where(log[r] > 0, log[p] / log[r], 0.0)
        log[f"z_{t}"] = np.where(log[f"f_{t}"] > 0, log[a] / log[f"f_{t}"].replace(0, np.nan), 0.0)
    return log


def floors(log, t):
    if t == "npxG":
        return np.where(log["is_cb"].to_numpy(), FLOOR_CB_NPXG, FLOOR["npxG"])
    return np.full(len(log), FLOOR["xAG"])


def step_weights(log, t):
    """Production's step weight per row (0 where the row doesn't update), and resets (rows the rule can't replay)."""
    rp = log[PAIRS[t][2]].to_numpy()
    f, z, w, n = log[f"f_{t}"].to_numpy(), log[f"z_{t}"].to_numpy(), log["w"].to_numpy(), log["n"].to_numpy()
    fl = floors(log, t)
    a = np.where((rp > 0) & (f > 0), np.maximum(1.0 / (n + K), fl) * w, 0.0)
    nxt = rp + a * (z - rp)
    players = log["Player"].to_numpy()
    same = np.r_[False, players[1:] == players[:-1]]
    expect = np.r_[np.nan, nxt[:-1]]
    reset = ~same | ~np.isclose(expect, rp, rtol=1e-9, atol=1e-12)
    return a, reset


def listed_rates():
    single = pd.read_csv(DATA / "position_combinations.csv").set_index("Pos")

    def rate(pos, col):
        parts = [x.strip() for x in str(pos).split(",")]
        w = [2.0 ** (len(parts) - 1 - i) for i in range(len(parts))]
        return float(np.average(single.loc[parts, col], weights=w))
    return single, rate


def arrival_segments(log, resets):
    """(player, arrival season, listed Pos, row indices of the segment) for each in-scope arrival."""
    t = pd.read_csv(DATA / "transfers.csv", encoding="utf-8-sig", usecols=["Player", "Season", "Pos", "League", "Notes"])
    t = t[t["Notes"].isna() & t["League"].ne("Premier League") & t["Pos"].notna() & t["Pos"].ne("GK") & t["Season"].ge(2017)]
    t = t[~t["Pos"].str.contains("GK")]
    idx = log.groupby(["Player", "Season"]).indices
    by_player = log.groupby("Player").indices
    segs, missing, not_reset = [], 0, 0
    for r in t.itertuples():
        rows = idx.get((r.Player, r.Season))
        if rows is None:
            missing += 1
            continue
        start = int(rows.min())
        if not resets[start]:
            not_reset += 1
            continue
        allrows = np.sort(by_player[r.Player])
        later = allrows[allrows > start]
        nxt = later[resets[later]]
        end = int(nxt.min()) if len(nxt) else int(allrows.max()) + 1
        segs.append((r.Player, int(r.Season), r.Pos, np.arange(start, end)))
    return segs, {"arrivals_in_scope": int(len(t)), "not_in_log": missing, "start_not_a_reset": not_reset,
                  "segments": len(segs)}


def segment_features(log, segs, t, a):
    """Per segment row: s_t (remaining prior share), cumulative weighted slot rate and weight before the row, p_listed."""
    single, rate = listed_rates()
    col = PAIRS[t][3]
    slot_rate = log["Slot"].map(lambda s: single.loc[s, col] if isinstance(s, str) and s != "GK" else np.nan).to_numpy()
    w = log["w"].to_numpy()
    rows, S, SW, SWR, PL, SEASON, PLAYER = [], [], [], [], [], [], []
    for player, season, pos, seg in segs:
        pl = rate(pos, col)
        s, sw, swr = 1.0, 0.0, 0.0
        for i in seg:
            rows.append(i); S.append(s); SW.append(sw); SWR.append(swr); PL.append(pl); SEASON.append(season); PLAYER.append(player)
            s *= 1.0 - a[i]
            if not np.isnan(slot_rate[i]) and w[i] > 0:
                sw += w[i]; swr += w[i] * slot_rate[i]
    return {"row": np.array(rows), "s": np.array(S), "sw": np.array(SW), "swr": np.array(SWR), "pl": np.array(PL),
            "season": np.array(SEASON), "player": np.array(PLAYER, dtype=object)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = load()
    res = {"log": str(LOG.relative_to(ROOT)).replace("\\", "/"), "targets": {}}
    a_by, reset_by = {}, {}
    for t in PAIRS:
        a_by[t], reset_by[t] = step_weights(log, t)
    resets = reset_by["npxG"] | reset_by["xAG"]
    # replay check: production's rule reproduces the log on every non-reset row (done inside step_weights via expect)
    res["replay"] = {t: {"non_reset_rows": int((~resets).sum()), "reset_rows": int(resets.sum())} for t in PAIRS}
    segs, counts = arrival_segments(log, resets)
    res["arrivals"] = counts
    print(counts, flush=True)
    for t, (act_col, pred_col, rat_col, _) in PAIRS.items():
        F = segment_features(log, segs, t, a_by[t])
        r = F["row"]
        w = log["w"].to_numpy()[r]
        f = log[f"f_{t}"].to_numpy()[r]
        rating = log[rat_col].to_numpy()[r]
        act = log[act_col].to_numpy()[r] * w
        base_pred = log[pred_col].to_numpy()[r] * w
        assert np.allclose(base_pred, rating * f * w), "prediction = rating x f x w must hold"
        base = (base_pred - act) ** 2
        tr = (F["season"] >= TRAIN[0]) & (F["season"] <= TRAIN[1])
        te = (F["season"] >= TEST[0]) & (F["season"] <= TEST[1])
        late = F["season"] >= 2025
        live = rating > 0

        def pred(c, kpos):
            p_t = (kpos * F["pl"] + F["swr"]) / (kpos + F["sw"])
            adj = np.where(live, rating + c * F["s"] * (p_t - F["pl"]), rating)
            return np.maximum(adj, 0.0) * f * w

        grid = {}
        for kpos in KPOS_GRID:
            for c in C_GRID:
                e = (pred(c, kpos) - act) ** 2
                grid[(float(c), kpos)] = float(e[tr].sum())
        best = min(grid, key=grid.get)
        cand_pred = pred(*best)
        cand = (cand_pred - act) ** 2
        # big gap: the segment's full observed slot rate differs from the listing by more than BIG_GAP
        seg_gap = {}
        for player, season, pos, seg in segs:
            m = (F["player"] == player) & (F["season"] == season)
            if m.any():
                last = np.where(m)[0][-1]
                sw, swr = F["sw"][last], F["swr"][last]
                seg_gap[(player, season)] = abs(swr / sw - F["pl"][last]) if sw > 0 else 0.0
        big = np.array([seg_gap.get((p, s), 0.0) > BIG_GAP for p, s in zip(F["player"], F["season"])])
        out = {"chosen": {"c": best[0], "Kpos": best[1]}, "rows": {"train": int(tr.sum()), "test": int(te.sum()),
               "test_big_gap": int((te & big).sum())},
               "train_change_pct": vpa.rel(cand, base, tr)}
        out["test_change_pct"], out["test_ci"] = vpa.boot(base[te], cand[te], F["player"][te])
        out["test_2025_26_pct"] = vpa.rel(cand, base, te & late)
        out["test_big_gap_pct"], out["test_big_gap_ci"] = vpa.boot(base[te & big], cand[te & big], F["player"][te & big])
        out["train_big_gap_pct"] = vpa.rel(cand, base, tr & big)
        out["calibration_test"] = {"production": float(base_pred[te].sum() / act[te].sum()),
                                   "candidate": float(cand_pred[te].sum() / act[te].sum())}
        all_base = ((log[pred_col].to_numpy() - log[act_col].to_numpy()) * log["w"].to_numpy()) ** 2
        all_cand = all_base.copy()
        all_cand[r] = cand
        seasons = log["Season"].to_numpy()
        out["all_rows_test_pct"] = vpa.rel(all_cand, all_base, seasons >= TEST[0])
        out["grid_train_pct"] = {f"c{c}_K{k}": round((v / float(base[tr].sum()) - 1) * 100, 4) for (c, k), v in grid.items()}
        cal_ok = abs(out["calibration_test"]["candidate"] - 1) <= abs(out["calibration_test"]["production"] - 1)
        out["recommend"] = bool(out["test_change_pct"] < 0 and out["test_ci"][1] < 0 and out["test_2025_26_pct"] <= 0.1 and cal_ok)
        # the case that prompted the test: the rating his matches would carry now
        mz = [i for i, (p, s, _, _) in enumerate(segs) if p == "Johan Manzambi" and s == 2026]
        if mz:
            m = (F["player"] == "Johan Manzambi") & (F["season"] == 2026)
            last = np.where(m)[0][-1]
            i = r[last]
            s_next = F["s"][last] * (1 - a_by[t][i])
            sw = F["sw"][last] + (w[last] if isinstance(log["Slot"].iat[i], str) else 0)
            single, _ = listed_rates()
            sr = single.loc[log["Slot"].iat[i], PAIRS[t][3]] if isinstance(log["Slot"].iat[i], str) else 0
            swr = F["swr"][last] + (w[last] * sr if isinstance(log["Slot"].iat[i], str) else 0)
            c, kpos = best
            p_now = (kpos * F["pl"][last] + swr) / (kpos + sw)
            out["manzambi"] = {"listed_rate": float(F["pl"][last]), "observed_rate": float(swr / sw) if sw else None,
                               "p_now": float(p_now), "remaining_prior_share": float(s_next),
                               "rating_shift_now": float(c * s_next * (p_now - F["pl"][last]))}
        res["targets"][t] = out
        print(t, json.dumps({k: v for k, v in out.items() if k != "grid_train_pct"}), flush=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
