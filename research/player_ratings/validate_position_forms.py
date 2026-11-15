"""
Position-update forms for entrants' starting ratings (CHANGELOG §247; protocol data/validation/position_rates_20261114/
phase3/protocol.json, frozen before scoring).

    python validate_position_forms.py <log> <priors_file> <window> <out_json>
        window: file | all | 3   (the position-rate table the log's starting ratings used)

§238's exact linear replay, generalised. The loop is linear in its rating, so a change d to a segment's starting
rating moves its rating before row t by s_t x d (s_t = share of the start still in the rating). A form moves the
position part of the start from the listed position's rate p_listed towards an evidence rate:
    rating'_t = rating_t + c x s_t x (p_t - p_listed),
    p_t = (Kpos x p_listed + sum_j e_j x r_j) / (Kpos + sum_j e_j)     over the segment's earlier rows j,
r_j = the slot rate of row j's slot (the segment's season's table), e_j = the row's evidence weight:
    F0  all appearances, e = minutes/90 (§238)
    F1  starts only, e = minutes/90 on starts
    F2  F1, but a covering start counts x q (q in {0, 0.5}): slot X is not the player's modal slot (his starts so far
        in the segment, else his first listed position) and a usual occupant of X didn't start (usual occupants: the
        players with the most starts in that raw slot over the team's previous 10 matches, as many as the slot's median
        number of starters there)
    F3  F1, but a start in a slot outside the listing counts only once that slot is >= m of the player's last M starts
        (m/M in 2/3, 3/5, 4/6)
    F4  F1 with evidence decaying by half every H starts (H in 3, 6, 12)
    ORACLE  p = the segment's full-segment start slot rate from its first row (hindsight; descriptive only)
Population: every entrant (validate_position_rates.segments: youth, second season, missed season, arrivals unfitted
and fitted), rows in the segment's first two seasons; train S0 2017-2022, test S0 2023-2026.
"""
import json
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

from research.player_ratings import validate_arrival_position as vap  # noqa: E402  (load, step_weights, PAIRS, SLOT_CANON)
from research.player_ratings import validate_position_rates as vpr  # noqa: E402  (segments, boot, rel)
from fpl_model.player_ratings import position_rates as pr  # noqa: E402

C_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
KPOS_GRID = [1, 2, 4, 8, 16, 32]
EXTRA = {"F0": [None], "F1": [None], "F2": [0.0, 0.5], "F3": [(2, 3), (3, 5), (4, 6)], "F4": [3, 6, 12]}
TRAIN, TEST = (2017, 2022), (2023, 2026)
DATA = vpr.DATA


def rate_table(window):
    cache = {}

    def get(season):
        if season not in cache:
            cache[season] = pr.file_rates() if window == "file" else pr.slot_rates(season, window)
        return cache[season]
    return get


def combined(tbl, pos, col):
    parts = [x.strip() for x in str(pos).split(",")]
    w = [2.0 ** (len(parts) - 1 - i) for i in range(len(parts))]
    return float(np.average(tbl.loc[parts, col], weights=w))


def covering_flags(log):
    """(Season, Team, Opponent, Venue, Player) -> True when a usual occupant of the player's raw slot didn't start.
    Team matches are ordered by their first row in the (chronological) control log."""
    p = pd.read_csv(DATA / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=["Season", "Team", "Opponent", "Venue", "Player", "Position_Resolved", "Is_Starter"])
    p = p.drop_duplicates(["Season", "Team", "Opponent", "Venue", "Player"])
    p = p[p["Is_Starter"].astype(str).str.lower().eq("true") & p["Position_Resolved"].notna()]
    raw = vpr.load_log(LOG_PATH)   # chronological order (the loop's match order)
    mk = list(zip(raw["Season"], raw["Team"], raw["Opponent"], raw["Venue"]))
    order = {}
    for i, k in enumerate(mk):
        order.setdefault(k, i)
    p["t"] = [order.get(k, np.inf) for k in zip(p["Season"], p["Team"], p["Opponent"], p["Venue"])]
    p = p[np.isfinite(p["t"])]
    flags = {}
    for team, g in p.groupby("Team"):
        hist = deque(maxlen=10)
        for key, mg in g.sort_values("t").groupby("t", sort=True):
            starters = set(mg["Player"])
            for r in mg.itertuples():
                X = r.Position_Resolved
                cnt = Counter(pl for m in hist for pl in m.get(X, []))
                flag = False
                if cnt:
                    n_x = max(1, int(round(np.median([len(m.get(X, [])) for m in hist]))))
                    top = sorted(cnt.values(), reverse=True)
                    cut = top[min(n_x, len(top)) - 1]
                    usual = {pl for pl, c in cnt.items() if c >= cut}
                    flag = (r.Player not in usual) and any(u not in starters for u in usual)
                flags[(r.Season, r.Team, r.Opponent, r.Venue, r.Player)] = flag
            m = {}
            for r in mg.itertuples():
                m.setdefault(r.Position_Resolved, []).append(r.Player)
            hist.append(m)
    return flags


def evidence(log, seg, rows_by_seg, slot_rate, listed_slots, first_listed, is_start, cover, form, extra, w):
    """Per row: (sum of e_j x r_j, sum of e_j) over the segment's earlier rows, for one form and setting."""
    n = len(log)
    SW, SWR = np.zeros(n), np.zeros(n)
    slots = log["Slot"].to_numpy()
    for sid, rows in rows_by_seg.items():
        sw = swr = 0.0
        hist = []            # (slot, rate, weight) of earlier evidence rows, in order
        recent = deque(maxlen=extra[1] if form == "F3" else 1)
        modal = Counter()
        for i in rows:
            if form == "F4":
                if hist:
                    k = np.arange(len(hist))[::-1]          # starts after j, before t
                    dec = 0.5 ** (k / extra)
                    ws = np.array([h[2] for h in hist]); rs = np.array([h[1] for h in hist])
                    SW[i], SWR[i] = float((dec * ws).sum()), float((dec * ws * rs).sum())
            else:
                SW[i], SWR[i] = sw, swr
            s, r = slots[i], slot_rate[i]
            if not isinstance(s, str) or s == "GK" or np.isnan(r) or w[i] <= 0:
                continue
            if form == "F0":
                sw += w[i]; swr += w[i] * r
                continue
            if not is_start[i]:
                continue
            e = w[i]
            if form == "F2":
                mod = modal.most_common(1)[0][0] if modal else first_listed[sid]
                if s != mod and cover[i]:
                    e *= extra
            if form == "F3":
                recent.append(s)
                if s not in listed_slots[sid] and sum(x == s for x in recent) < extra[0]:
                    e = 0.0
            modal[s] += 1
            if form == "F4":
                hist.append((s, r, e))
            sw += e; swr += e * r
    return SW, SWR


def main(log_path, priors_file, window, out_json):
    global LOG_PATH
    LOG_PATH = Path(log_path)
    vap.LOG = LOG_PATH
    log = vap.load()     # sorted stably by player, with Slot, is_cb, w, n, f_, z_
    a_by, resets = {}, None
    for t in vap.PAIRS:
        a_by[t], r = vap.step_weights(log, t)
        resets = r if resets is None else (resets | r)
    seg = vpr.segments(log, priors_file)
    sid = seg["seg"]
    starts = np.r_[True, sid[1:] != sid[:-1]]
    res = {"log": str(log_path), "priors": str(priors_file), "window": window,
           "replay": {"non_reset_rows": int((~resets).sum()), "reset_rows": int(resets.sum()),
                      "segment_starts": int(starts.sum()), "starts_equal_resets": bool((starts == resets).all())}}
    assert (starts == resets).all(), "segment starts must be exactly the replay's reset rows"
    ent = seg["type"] != ""
    ent_seg = sorted(set(sid[ent]))
    # entrant rows are a prefix of their segment (seasons S0, S0 + 1); later rows are never scored or used
    rows_by_seg = {s_: np.where((sid == s_) & ent)[0] for s_ in ent_seg}
    tbl = rate_table(window)
    is_start = log["Is_Starter"].astype(str).str.lower().eq("true").to_numpy()
    first_listed, listed_slots = {}, {}
    for s_, rows in rows_by_seg.items():
        pos = seg["pos"][rows[0]]
        parts = [x.strip() for x in str(pos).split(",")]
        first_listed[s_], listed_slots[s_] = parts[0], set(parts)
    print("covering flags...", flush=True)
    fl = covering_flags(log)
    cover = np.array([fl.get(k, False) for k in zip(log["Season"], log["Team"], log["Opponent"], log["Venue"],
                                                     log["Player"])])
    w = log["w"].to_numpy()
    s0 = seg["s0"]
    rows_all = np.concatenate([rows_by_seg[s_] for s_ in ent_seg])
    mask_rows = ent[rows_all]
    R = rows_all[mask_rows]                                  # scored rows: entrant rows
    tr = (s0[R] >= TRAIN[0]) & (s0[R] <= TRAIN[1])
    te = (s0[R] >= TEST[0]) & (s0[R] <= TEST[1])
    late = s0[R] >= 2025
    players = log["Player"].to_numpy()[R]
    arrivals = np.array([x.startswith("arrival") for x in seg["type"][R]])
    res["population"] = {"segments": len(ent_seg), "rows_train": int(tr.sum()), "rows_test": int(te.sum()),
                         "covering_starts": int((cover & is_start)[R].sum()), "starts": int(is_start[R].sum())}
    res["targets"] = {}
    for t, (act_col, pred_col, rat_col, col) in vap.PAIRS.items():
        seg_tbl_season = seg["tseason"]
        slot_rate = np.array([tbl(int(seg_tbl_season[i])).loc[s, col] if isinstance(s, str) and s != "GK" else np.nan
                              for i, s in enumerate(log["Slot"])])
        pl_seg = {s_: combined(tbl(int(seg_tbl_season[rows[0]])), seg["pos"][rows[0]], col)
                  for s_, rows in rows_by_seg.items()}
        a = a_by[t]
        S = np.zeros(len(log))
        for s_, rows in rows_by_seg.items():
            s = 1.0
            for i in rows:
                S[i] = s
                s *= 1.0 - a[i]
        PL = np.array([pl_seg[sid[i]] for i in R])
        f = log[f"f_{t}"].to_numpy()[R]
        rating = log[rat_col].to_numpy()[R]
        act = log[act_col].to_numpy()[R] * w[R]
        base_pred = log[pred_col].to_numpy()[R] * w[R]
        assert np.allclose(base_pred, rating * f * w[R])
        base = (base_pred - act) ** 2
        live = rating > 0
        sR = S[R]

        def pred(c, kpos, SW, SWR):
            p_t = (kpos * PL + SWR) / (kpos + SW)
            return np.maximum(np.where(live, rating + c * sR * (p_t - PL), rating), 0.0) * f * w[R]

        forms = {}
        for form, extras in EXTRA.items():
            best = None
            for ex in extras:
                SW, SWR = evidence(log, seg, rows_by_seg, slot_rate, listed_slots, first_listed, is_start, cover,
                                   form, ex, w)
                SW, SWR = SW[R], SWR[R]
                for kpos in KPOS_GRID:
                    for c in C_GRID:
                        l = float(((pred(c, kpos, SW, SWR) - act) ** 2)[tr].sum())
                        if best is None or l < best[0]:
                            best = (l, float(c), kpos, ex, SW, SWR)
            l, c, kpos, ex, SW, SWR = best
            cp = pred(c, kpos, SW, SWR)
            cand = (cp - act) ** 2
            o = {"c": c, "Kpos": kpos, "extra": ex, "train_pct": vpr.rel(cand, base, tr)}
            o["test_pct"], o["test_ci"] = vpr.boot(base[te], cand[te], players[te])
            o["test_2025_26_pct"] = vpr.rel(cand, base, te & late)
            o["test_arrivals_pct"], o["test_arrivals_ci"] = vpr.boot(base[te & arrivals], cand[te & arrivals],
                                                                     players[te & arrivals])
            o["test_by_type"] = {et: vpr.rel(cand, base, te & (seg["type"][R] == et)) for et in vpr.ENTRY_TYPES}
            o["calibration_test"] = {"control": float(act[te].sum() / base_pred[te].sum()),
                                     "form": float(act[te].sum() / cp[te].sum())}
            forms[form] = o
            print(t, form, json.dumps({k: v for k, v in o.items() if k != "test_by_type"}), flush=True)
        # oracle: the segment's full-segment start slot rate (F1 evidence at the segment's end), from row one
        SW1, SWR1 = evidence(log, seg, rows_by_seg, slot_rate, listed_slots, first_listed, is_start, cover, "F1",
                             None, w)
        fin = {}
        for s_, rows in rows_by_seg.items():
            i = rows[-1]
            sw_end = SW1[i] + (w[i] if is_start[i] and isinstance(log["Slot"].iat[i], str) and not np.isnan(slot_rate[i]) else 0)
            swr_end = SWR1[i] + (w[i] * slot_rate[i] if is_start[i] and isinstance(log["Slot"].iat[i], str) and not np.isnan(slot_rate[i]) else 0)
            fin[s_] = swr_end / sw_end if sw_end > 0 else pl_seg[s_]
        p_or = np.array([fin[sid[i]] for i in R])
        best = None
        for c in np.arange(0, 1.51, 0.05):
            e = (np.maximum(np.where(live, rating + c * sR * (p_or - PL), rating), 0) * f * w[R] - act) ** 2
            if best is None or e[tr].sum() < best[0]:
                best = (e[tr].sum(), c)
        oc = (np.maximum(np.where(live, rating + best[1] * sR * (p_or - PL), rating), 0) * f * w[R] - act) ** 2
        orc = {"c": round(float(best[1]), 2), "train_pct": vpr.rel(oc, base, tr)}
        orc["test_pct"], orc["test_ci"] = vpr.boot(base[te], oc[te], players[te])
        orc["test_arrivals_pct"] = vpr.rel(oc, base, te & arrivals)
        # choose among forms on train: summed later across targets
        res["targets"][t] = {"forms": forms, "oracle": orc,
                             "manzambi": manzambi(log, seg, rows_by_seg, forms, t, slot_rate, pl_seg, S, is_start, cover,
                                                  listed_slots, first_listed, w)}
    tot = {f_: res["targets"]["npxG"]["forms"][f_]["train_pct"] + res["targets"]["xAG"]["forms"][f_]["train_pct"]
           for f_ in EXTRA}
    res["train_sum_pct"] = {k: round(v, 4) for k, v in tot.items()}
    primary = min(tot, key=tot.get)
    res["primary_form"] = primary
    verdict = {}
    for t in vap.PAIRS:
        o = res["targets"][t]["forms"][primary]
        cal = o["calibration_test"]
        verdict[t] = {"test_ci_below_0": o["test_ci"][1] < 0 and o["test_pct"] < 0,
                      "late_not_above_0.1": o["test_2025_26_pct"] is None or o["test_2025_26_pct"] <= 0.1,
                      "calibration_not_further": abs(cal["form"] - 1) <= abs(cal["control"] - 1)}
        verdict[t]["recommend"] = all(verdict[t].values())
    res["verdict"] = verdict
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"train_sum_pct": res["train_sum_pct"], "primary": primary, "verdict": verdict}, indent=1))


def manzambi(log, seg, rows_by_seg, forms, t, slot_rate, pl_seg, S, is_start, cover, listed_slots, first_listed, w):
    """The rating shift each form (at its chosen settings) gives Manzambi now (after his last logged row)."""
    pl = log["Player"].to_numpy()
    out = {}
    for s_, rows in rows_by_seg.items():
        if pl[rows[0]] != "Johan Manzambi" or seg["s0"][rows[0]] != 2026:
            continue
        i = rows[-1]
        for form, o in forms.items():
            # evidence including the last row: rerun the accumulation on the segment with one virtual extra row
            SW, SWR = evidence(log, seg, {s_: np.r_[rows, rows[-1]]}, slot_rate, listed_slots, first_listed,
                               is_start, cover, form, o["extra"], w)
            sw, swr = SW[rows[-1]], SWR[rows[-1]]   # the virtual row sees every real row
            p_now = (o["Kpos"] * pl_seg[s_] + swr) / (o["Kpos"] + sw)
            out[form] = {"listed": round(pl_seg[s_], 4), "p_now": round(float(p_now), 4),
                         "shift_approx": round(float(o["c"] * S[i] * (p_now - pl_seg[s_])), 4),
                         "rating_pre_last": round(float(log[vap.PAIRS[t][2]].to_numpy()[i]), 4)}
    return out


if __name__ == "__main__":
    main(*sys.argv[1:5])
