"""
First-season adaptation check for new arrivals (CHANGELOG §231 part 2; diagnostic only).

    python diagnose_arrival_adaptation.py <pre-match log> <out.json>

Arrivals: blank-Notes non-PL outfield rows of transfers.csv in their first PL season (as the prior dataset,
`new_player_priors.in_scope_transfers` / `attach_outcomes`). Controls: players with an earlier PL season.
Per log row: actual = npxg90 x weight, forecast = pred_goals_90 x weight (xAG: xag90, pred_ast_90).
Match number = order within (Player, Season) in the log (the loop's order).
dip = (arrivals' ratio 1-10 / arrivals' 20+) / (controls' ratio 1-10 / controls' 20+), ratio = sum actual / sum
forecast; player-cluster bootstrap (2,000, seed 42). Complete seasons only (2019-2025): the current season has no
20+ matches yet, so including it would put it in the early bin alone.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import new_player_priors as npp  # noqa: E402

SEASONS = range(2019, 2026)
BINS = [("1-10", 0, 10), ("11-19", 10, 19), ("20+", 19, 10_000)]
PAIRS = {"npxG": ("npxg90", "pred_goals_90"), "xAG": ("xag90", "pred_ast_90")}
N_BOOT, SEED = int(__import__("os").environ.get("ADAPT_N_BOOT", "2000")), 42


def load(log_path):
    log = pd.read_csv(log_path, encoding="utf-8-sig", low_memory=False)
    log = log[log["Season"].isin(SEASONS)].copy()
    log["k"] = log.groupby(["Player", "Season"]).cumcount()
    pms = pd.read_csv(npp.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    pms = pms[pms["Minutes"] > 0].copy()
    pms["Season"] = np.where(pms["Data_Source"] == "fbref", 2017 + pms["MatchID"] // 380, pms["Season"])
    first_pl = pms.groupby("Player")["Season"].min()
    arr = npp.in_scope_transfers()
    arr = arr[arr["Player"].map(first_pl) == arr["Season"]]
    akeys = set(zip(arr["Player"], arr["Season"]))
    log["first_pl"] = log["Player"].map(first_pl)
    is_arr = np.array([k in akeys for k in zip(log["Player"], log["Season"])])
    log["role"] = np.where(is_arr, "arrival", np.where(log["first_pl"] < log["Season"], "control", "other"))
    log["bin"] = None
    for name, lo, hi in BINS:
        log.loc[(log["k"] >= lo) & (log["k"] < hi), "bin"] = name
    for t, (a, p) in PAIRS.items():
        log[f"act_{t}"] = log[a] * log["weight"]
        log[f"fc_{t}"] = log[p] * log["weight"]
    log["cluster"] = log["Player"] + "|" + log["Season"].astype(str)
    return log[log["role"] != "other"]


def sums(log):
    """(cluster x [role, bin, target, act|fc]) table of sums, for a fast bootstrap."""
    cols = [f"{x}_{t}" for t in PAIRS for x in ("act", "fc")]
    g = log.groupby(["cluster", "role", "bin"])[cols].sum().unstack(["role", "bin"]).fillna(0.0)
    return g


def stats(tab, w=None):
    w = np.ones(len(tab)) if w is None else w
    tot = (tab.to_numpy() * w[:, None]).sum(0)
    s = pd.Series(tot, index=tab.columns)
    out = {}
    for t in PAIRS:
        r = {}
        for role in ("arrival", "control"):
            for b, _, _ in BINS:
                fc = s.get((f"fc_{t}", role, b), np.nan)
                r[f"{role}_{b}"] = s.get((f"act_{t}", role, b), np.nan) / fc if fc else np.nan
        r["dip"] = (r["arrival_1-10"] / r["arrival_20+"]) / (r["control_1-10"] / r["control_20+"])
        r["dip_11_19"] = (r["arrival_11-19"] / r["arrival_20+"]) / (r["control_11-19"] / r["control_20+"])
        out[t] = r
    return out


def main(log_path, out_path):
    log = load(log_path)
    tab = sums(log)
    point = stats(tab)
    rng = np.random.default_rng(SEED)
    # player-cluster bootstrap: resample player-seasons with replacement, separately within arrivals and controls
    roles = log.groupby("cluster")["role"].first().reindex(tab.index)
    idx = {r: np.flatnonzero((roles == r).to_numpy()) for r in ("arrival", "control")}
    draws = {t: {"dip": [], "dip_11_19": []} for t in PAIRS}
    for _ in range(N_BOOT):
        w = np.zeros(len(tab))
        for r, ix in idx.items():
            np.add.at(w, rng.choice(ix, len(ix)), 1.0)
        st = stats(tab, w)
        for t in PAIRS:
            for k in draws[t]:
                draws[t][k].append(st[t][k])
    res = {"log": str(log_path), "seasons": [min(SEASONS), max(SEASONS)],
           "arrival_player_seasons": len(idx["arrival"]), "control_player_seasons": len(idx["control"]),
           "rows": log.groupby(["role", "bin"]).size().unstack().to_dict()}
    for t in PAIRS:
        res[t] = {**{k: float(v) for k, v in point[t].items()},
                  **{f"{k}_ci": [float(np.nanpercentile(draws[t][k], 2.5)), float(np.nanpercentile(draws[t][k], 97.5))]
                     for k in draws[t]}}
    Path(out_path).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
