"""
Slower npxG smoothing for defenders only (CHANGELOG §235).

    python validate_defender_npxg.py    # -> data/validation/defender_npxg_20261112/results.json

Exact replay of production's npxG rule from the production log, with defender rows (match position CB, LB, RB, WB)
taking a variant: a lower floor, the career average (floor 0) or a moving average over the last N appearances.
All other rows keep the 0.035 floor; xAG is not touched. Data preparation (groups, readings, resets) is shared with
validate_kalman_smoothing.py.
"""
import json
from collections import deque

import numpy as np

from research.player_ratings import validate_kalman_smoothing as vk  # noqa: E402
from research.player_ratings import validate_position_alpha as vpa  # noqa: E402

OUT = vk.ROOT / "data" / "validation" / "defender_npxg_20261112"
PROD_FLOOR, K = 0.035, 16.0
DEF = {"CB", "FB"}
VARIANTS = {"floor_0.035": ("floor", 0.035), "floor_0.025": ("floor", 0.025), "floor_0.018": ("floor", 0.018),
            "floor_0.0125": ("floor", 0.0125), "floor_0.0083": ("floor", 0.0083), "career_average": ("floor", 0.0),
            "moving_38": ("window", 38), "moving_76": ("window", 76)}


def replay(log, kind, val):
    """Pre-match npxG rating per row under the variant (defender rows only; others at the production floor)."""
    players = log["Player"].to_numpy()
    z, f, w = log["z_npxG"].to_numpy(), log["f_npxG"].to_numpy(), log["w"].to_numpy()
    rp = log["NPG_Rating_Pre"].to_numpy()
    reset = log["reset"].to_numpy()
    is_def = log["grp"].isin(DEF).to_numpy()
    n = log.groupby("Player").cumcount().to_numpy()
    out = np.empty(len(log))
    th, buf, prior, filled = {}, {}, {}, {}
    for i in range(len(log)):
        pl = players[i]
        if reset[i] or pl not in th:
            th[pl], prior[pl] = rp[i], rp[i]
            buf[pl] = deque(maxlen=int(val)) if kind == "window" else None
            filled[pl] = False
        theta = th[pl]
        out[i] = theta
        if not (f[i] > 0 and theta > 0):
            continue
        if kind == "window":
            b = buf[pl]
            b.append((w[i], z[i]))
            if len(b) == b.maxlen:
                filled[pl] = True
        if is_def[i] and kind == "window":
            b = buf[pl]
            sw = sum(x for x, _ in b)
            sz = sum(x * y for x, y in b)
            pk = 0.0 if filled[pl] else K
            th[pl] = (pk * prior[pl] + sz) / (pk + sw) if pk + sw > 0 else theta
        else:
            floor = val if (is_def[i] and kind == "floor") else PROD_FLOOR
            th[pl] = theta + max(1.0 / (n[i] + K), floor) * w[i] * (z[i] - theta)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = vk.load()
    s = log["Season"].to_numpy()
    out_ = log["outfield"].to_numpy()
    grp = log["grp"].to_numpy()
    players = log["Player"].to_numpy()
    w, f = log["w"].to_numpy(), log["f_npxG"].to_numpy()
    act = log["npxg90"].to_numpy() * w
    rp = log["NPG_Rating_Pre"].to_numpy()
    base_pred = log["pred_goals_90"].to_numpy() * w
    tr, te = out_ & (s >= 2018) & (s <= 2023), out_ & (s >= 2024) & (s <= 2026)
    d = np.isin(grp, list(DEF))
    masks = {"def_train": tr & d, "def_test": te & d, "cb_test": te & (grp == "CB"), "fb_test": te & (grp == "FB"),
             "all_train": tr, "all_test": te}
    # check: the production variant reproduces the log on every non-reset row
    prod = replay(log, "floor", PROD_FLOOR)
    ok = np.allclose(prod, rp, rtol=1e-9, atol=1e-12)
    assert ok, "production replay differs from the log"
    base = (base_pred - act) ** 2
    res = {"replay_exact": bool(ok), "rows": {k: int(v.sum()) for k, v in masks.items()}, "variants": {}}
    for name, (kind, val) in VARIANTS.items():
        th = replay(log, kind, val)
        cand = (th * f * w - act) ** 2
        r = {k: vpa.rel(cand, base, m) for k, m in masks.items()}
        r["def_test_ci"] = vpa.boot(base[masks["def_test"]], cand[masks["def_test"]], players[masks["def_test"]])[1]
        r["def_test_calibration"] = float((th * f * w)[masks["def_test"]].sum() / act[masks["def_test"]].sum())
        res["variants"][name] = r
        print(name, r, flush=True)
    best = min((k for k in VARIANTS if k != "floor_0.035"), key=lambda k: res["variants"][k]["def_train"])
    b = res["variants"][best]
    res["best_on_train"] = best
    res["recommend"] = bool(b["def_test"] < 0 and b["def_test_ci"][1] < 0 and b["all_test"] <= 0)
    res["base_def_test_calibration"] = float(base_pred[masks["def_test"]].sum() / act[masks["def_test"]].sum())
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("best", best, "recommend", res["recommend"])


if __name__ == "__main__":
    main()
