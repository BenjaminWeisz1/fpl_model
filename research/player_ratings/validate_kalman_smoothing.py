"""
Goal/assist smoothing by position: a local-level (Kalman) model against production (CHANGELOG §234).

    python -m research.player_ratings.validate_kalman_smoothing fit     # fit K1 and K5 on 2018-2023 -> fits.json
    python -m research.player_ratings.validate_kalman_smoothing score   # test 2024-2026 against production -> results.json

Replayed from the production pre-match log (npxG floor 0.035, xAG 0.025, §233). Per row: reading z = actual per 90 /
f, f = pred_90 / rating_pre (venue x opponent x substitute uplift), w = minutes / 90. Filter per player and target:
P += q_g; G = P / (P + r_g / w); theta += G (z - theta); P *= 1 - G; start theta = logged prior, P0 = r_g / 16.
g = position group of the match. At a re-blend row (production's rule can't be replayed there) theta := logged
rating and P := P0.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fpl_model import paths
from research.player_ratings import validate_position_alpha as vpa  # noqa: E402

ROOT = vpa.ROOT
OUT = ROOT / "data" / "validation" / "kalman_smoothing_20261112"
LOG = ROOT / "data" / "validation" / "npxg_floor_20261112" / "check_default.csv"
POS = paths.current_file("player_match_positions.csv")
FLOORS = {"npxG": 0.035, "xAG": 0.025}
K_PRIOR = 16.0
PAIRS = {"npxG": ("npxg90", "pred_goals_90", "NPG_Rating_Pre"), "xAG": ("xag90", "pred_ast_90", "Assist_Rating_Pre")}
GROUPS = ["CB", "FB", "CM", "AW", "FW"]
POS_GROUP = {"CB": "CB", "LB": "FB", "RB": "FB", "WB": "FB", "DM": "CM", "CM": "CM",
             "AM": "AW", "LM": "AW", "RM": "AW", "LW": "AW", "RW": "AW", "FW": "FW"}
TRAIN, TEST = (2018, 2023), (2024, 2026)


def load():
    log = pd.read_csv(LOG, encoding="utf-8-sig", low_memory=False)
    pos = pd.read_csv(POS, encoding="utf-8-sig", low_memory=False,
                      usecols=["Season", "Team", "Opponent", "Venue", "Player", "Position_Resolved"])
    pos = pos.drop_duplicates(["Season", "Team", "Opponent", "Venue", "Player"])
    m = log.merge(pos, on=["Season", "Team", "Opponent", "Venue", "Player"], how="left")
    assert len(m) == len(log)
    tg = vpa.groups(log)                      # transfers.csv group (None for GK / unmapped)
    g = m["Position_Resolved"].map(POS_GROUP).to_numpy(dtype=object)
    log["outfield"] = np.array([x is not None for x in tg])
    # fill a match without a position from the player's previous match, else his transfers.csv group
    grp = []
    last = {}
    for p, gi, ti in zip(log["Player"].to_numpy(), g, tg):
        if not isinstance(gi, str):
            gi = last.get(p, ti)
        if isinstance(gi, str):
            last[p] = gi
        grp.append(gi if isinstance(gi, str) else "CM")
    log["grp"] = grp
    log["gi"] = log["grp"].map({k: i for i, k in enumerate(GROUPS)}).to_numpy()
    log["w"] = log["weight"]
    for t, (a, p, r) in PAIRS.items():
        log[f"f_{t}"] = np.where(log[r] > 0, log[p] / log[r], 0.0)
        log[f"z_{t}"] = np.where(log[f"f_{t}"] > 0, log[a] / log[f"f_{t}"].replace(0, np.nan), 0.0)
    # rows where production's rule doesn't replay (re-blends) and each player's first row
    n = log.groupby("Player").cumcount().to_numpy()
    reset = {}
    for t, (a, p, r) in PAIRS.items():
        rp = log[r].to_numpy()
        f, z, w = log[f"f_{t}"].to_numpy(), log[f"z_{t}"].to_numpy(), log["w"].to_numpy()
        players = log["Player"].to_numpy()
        expect = np.full(len(log), np.nan)
        state = {}
        for i in range(len(log)):
            pl = players[i]
            if pl in state:
                expect[i] = state[pl]
            cur = rp[i]   # production's rating before this match
            nxt = cur
            if cur > 0 and f[i] > 0:
                nxt = cur + max(1.0 / (n[i] + K_PRIOR), FLOORS[t]) * w[i] * (z[i] - cur)
            state[pl] = nxt
        reset[t] = (n == 0) | ~np.isclose(expect, rp, rtol=1e-9, atol=1e-12)
        reset[t][n == 0] = True
    log["reset"] = reset["npxG"] | reset["xAG"]
    return log


def run_filter(log, t, q, r, return_pred=False):
    """q, r: arrays per group. Returns NLL over train rows (and the pre-match theta per row)."""
    players = log["Player"].to_numpy()
    gi = log["gi"].to_numpy()
    z, f, w = log[f"z_{t}"].to_numpy(), log[f"f_{t}"].to_numpy(), log["w"].to_numpy()
    rp = log[PAIRS[t][2]].to_numpy()
    reset = log["reset"].to_numpy()
    fit_mask = log["fit_mask"].to_numpy()
    th_pre = np.empty(len(log))
    th, P = {}, {}
    nll = 0.0
    for i in range(len(log)):
        pl, g = players[i], gi[i]
        if reset[i] or pl not in th:
            th[pl], P[pl] = rp[i], r[g] / K_PRIOR
        else:
            P[pl] += q[g]
        theta, p = th[pl], P[pl]
        th_pre[i] = theta
        if f[i] > 0 and theta > 0:
            R = r[g] / w[i]
            S = p + R
            v = z[i] - theta
            if fit_mask[i]:
                nll += np.log(S) + v * v / S
            G = p / S
            th[pl] = theta + G * v
            P[pl] = (1 - G) * p
    return (nll, th_pre) if return_pred else nll


def fit(log, t, pooled):
    k = 1 if pooled else len(GROUPS)

    def unpack(x):
        q, r = np.exp(x[:k]), np.exp(x[k:])
        return (np.repeat(q, len(GROUPS)), np.repeat(r, len(GROUPS))) if pooled else (q, r)

    # start near production: steady-state gain G* gives q/r = G*^2 / (1 - G*); r from reading variance
    g0 = FLOORS[t]
    zv = log.loc[log["fit_mask"], f"z_{t}"]
    r0 = float(np.var(zv) * np.mean(log.loc[log["fit_mask"], "w"]))
    x0 = np.r_[np.full(k, np.log(r0 * g0 ** 2 / (1 - g0))), np.full(k, np.log(r0))]
    res = minimize(lambda x: run_filter(log, t, *unpack(x)), x0, method="L-BFGS-B",
                   options={"maxiter": 200, "eps": 1e-4})
    q, r = unpack(res.x)
    lam = q / r
    gain = (-lam + np.sqrt(lam ** 2 + 4 * lam)) / 2
    return {"q": q.tolist(), "r": r.tolist(), "steady_gain": dict(zip(GROUPS, gain.tolist())),
            "nll": float(res.fun), "converged": bool(res.success), "iterations": int(res.nit)}


def prepare():
    log = load()
    s = log["Season"].to_numpy()
    log["fit_mask"] = log["outfield"] & (s >= TRAIN[0]) & (s <= TRAIN[1])
    return log


def do_fit():
    OUT.mkdir(parents=True, exist_ok=True)
    log = prepare()
    fits = {}
    for t in PAIRS:
        for arm, pooled in (("K1", True), ("K5", False)):
            fits[f"{arm}_{t}"] = fit(log, t, pooled)
            print(arm, t, json.dumps(fits[f"{arm}_{t}"]["steady_gain"]), fits[f"{arm}_{t}"]["converged"], flush=True)
    fits["rows"] = {"all": len(log), "resets": int(log["reset"].sum()),
                    "groups": log.loc[log["outfield"], "grp"].value_counts().to_dict()}
    (OUT / "fits.json").write_text(json.dumps(fits, indent=1), encoding="utf-8")


def score():
    log = prepare()
    fits = json.loads((OUT / "fits.json").read_text(encoding="utf-8"))
    s = log["Season"].to_numpy()
    out = log["outfield"].to_numpy()
    masks = {"train": out & (s >= TRAIN[0]) & (s <= TRAIN[1]), "test": out & (s >= TEST[0]) & (s <= TEST[1])}
    players = log["Player"].to_numpy()
    grp = log["grp"].to_numpy()
    w = log["w"].to_numpy()
    res = {"fits": {k: v["steady_gain"] for k, v in fits.items() if k != "rows"}, "rows": fits["rows"], "arms": {}}
    for t, (a, p, r) in PAIRS.items():
        act = log[a].to_numpy() * w
        base = (log[p].to_numpy() * w - act) ** 2
        f = log[f"f_{t}"].to_numpy()
        for arm in ("K1", "K5"):
            ft = fits[f"{arm}_{t}"]
            _, th = run_filter(log, t, np.array(ft["q"]), np.array(ft["r"]), return_pred=True)
            cand = (th * f * w - act) ** 2
            r_ = {}
            for name, mk in masks.items():
                ch, ci = vpa.boot(base[mk], cand[mk], players[mk])
                r_[name] = {"change_pct": ch, "ci": ci}
            r_["test_by_group"] = {g: vpa.rel(cand, base, masks["test"] & (grp == g)) for g in GROUPS}
            r_["test_calibration"] = float((th * f * w)[masks["test"]].sum() / act[masks["test"]].sum())
            res["arms"][f"{arm}_{t}"] = r_
            log[f"cand_{arm}_{t}"] = cand
        res["base_calibration_test_" + t] = float((log[p].to_numpy() * w)[masks["test"]].sum() / act[masks["test"]].sum())
    for t in PAIRS:   # K5 v K1 on test (the position effect)
        mk = masks["test"]
        res["arms"][f"K5_vs_K1_{t}"] = vpa.boot(log[f"cand_K1_{t}"].to_numpy()[mk], log[f"cand_K5_{t}"].to_numpy()[mk],
                                                players[mk])
    k5 = [res["arms"][f"K5_{t}"] for t in PAIRS]
    res["recommend_K5"] = bool(all(x["test"]["change_pct"] < 0 for x in k5)
                               and all(v <= 0.3 for x in k5 for v in x["test_by_group"].values()))
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"fit": do_fit, "score": score}[sys.argv[1]]()
