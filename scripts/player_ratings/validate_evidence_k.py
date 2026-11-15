"""
Evidence-weighted starting weight K for new arrivals (CHANGELOG §232).

    python validate_evidence_k.py identity <log>   # replay at K = 16 must reproduce the logged ratings
    python validate_evidence_k.py score <log>      # grid on train, test on later entrants -> results.json

The loop's update for one player depends only on his own rows (opponent, venue and substitute factors are fixed
per row), so a different K schedule can be replayed exactly from a pre-match log: per row, factor f = pred_90 /
rating_pre, value y = actual per 90, step a = max(1/(n + K), 0.025) x weight, rating <- rating x (1 + a (y/(r f) - 1)).
n = the row's position in the player's logged rows (checked by `identity`).

Candidate: for arrivals (blank-Notes non-PL rows of transfers.csv, outfield), K_i = K0 x (E_i / 2000)^rho, clipped
to [4, 64], E_i = Min_Old (minutes behind the prior's last season). Everyone else keeps K = 16. K0 = 16, rho = 0
is production.
Entrants as §112 (first logged PL match 2019+, outfield, entry type from transfers.csv). Train = entry 2019-2022,
test = entry 2023-2026. Primary rows: arrivals' first two logged seasons. Guard: all arrivals' rows from entry.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_prior_k as vpk  # noqa: E402

OUT = vpk.ROOT / "data" / "validation" / "evidence_k_20261112"
FLOOR, K_PROD, E_REF, K_MIN, K_MAX = 0.025, 16.0, 2000.0, 4.0, 64.0
K0_GRID = [12, 16, 22, 30]
RHO_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
PAIRS = {"npxG": ("npxg90", "pred_goals_90", "NPG_Rating_Pre"), "xAG": ("xag90", "pred_ast_90", "Assist_Rating_Pre")}


def load(log_path):
    log = pd.read_csv(log_path, encoding="utf-8-sig", low_memory=False)
    log["n"] = log.groupby("Player").cumcount()
    ent = vpk.entrants(log)
    t = pd.read_csv(vpk.DATA / "transfers.csv", encoding="utf-8-sig")
    t["Min_Old"] = pd.to_numeric(t["Min_Old"], errors="coerce")
    # evidence = Min_Old of the transfers.csv row that started the player (latest row with Season <= entry)
    ev = {}
    for p, s0 in zip(ent["Player"], ent["Entry"]):
        c = t[(t["Player"] == p) & (t["Season"] <= s0)].sort_values("Season")
        ev[p] = float(c["Min_Old"].iloc[-1]) if len(c) else np.nan
    ent["E"] = ent["Player"].map(ev)
    ent["arrival"] = ent["Type"].isin(["arrival_fitted", "arrival_old"])
    m = log[["Player", "Season"]].merge(ent, on="Player", how="left")
    for c in ("Entry", "Type", "E", "arrival"):
        log[c] = m[c].to_numpy()
    log = log[log["Entry"].notna()].copy()
    for t_, (a, p, r) in PAIRS.items():
        log[f"f_{t_}"] = np.where(log[r] > 0, log[p] / log[r], 0.0)
    return log, ent


def replay(g, k, t):
    """Pre-match ratings for one player's rows under starting weight k."""
    a, p, r = PAIRS[t]
    y, f, w = g[a].to_numpy(), g[f"f_{t}"].to_numpy(), g["weight"].to_numpy()
    out = np.empty(len(g))
    rating = g[r].iloc[0]
    for i in range(len(g)):
        out[i] = rating
        if rating > 0 and f[i] > 0:
            step = max(1.0 / (i + k), FLOOR) * w[i]
            rating *= 1 + step * (y[i] / (rating * f[i]) - 1)
    return out


def k_of(E, k0, rho):
    if not np.isfinite(E) or E <= 0:
        return k0
    return float(np.clip(k0 * (E / E_REF) ** rho, K_MIN, K_MAX))


def identity(log_path):
    log, ent = load(log_path)
    bad = []
    for pl, g in log.groupby("Player", sort=False):
        assert (g["n"].to_numpy() == np.arange(len(g))).all()
        for t, (_, _, r) in PAIRS.items():
            sim = replay(g, K_PROD, t)
            if not np.allclose(sim, g[r].to_numpy(), rtol=1e-9, atol=1e-12):
                bad.append((pl, t, int(np.argmax(~np.isclose(sim, g[r].to_numpy(), rtol=1e-9, atol=1e-12)))))
    players = log["Player"].nunique()
    cut = {}
    for pl, _, i in bad:   # a 'second season' re-blend: the replay is exact only up to that row
        cut[pl] = min(i, cut.get(pl, i))
    n_after = int(sum((log["Player"] == pl).sum() - i for pl, i in cut.items()))
    print(f"entrant players {players}, rows {len(log)}, mismatched player-targets {len(bad)}, "
          f"players cut at their first re-blend {len(cut)} ({n_after} rows dropped)")
    return cut


def identity_after(log):
    """Players whose replay at K = 16 still differs from the log (should be none)."""
    return [pl for pl, g in log.groupby("Player", sort=False) for t, (_, _, r) in PAIRS.items()
            if not np.allclose(replay(g, K_PROD, t), g[r].to_numpy(), rtol=1e-9, atol=1e-12)]


def predictions(log, k0, rho):
    """Per-row predicted match values under the candidate (arrivals get K_i, others 16)."""
    pred = {t: np.empty(len(log)) for t in PAIRS}
    pos = 0
    for pl, g in log.groupby("Player", sort=False):
        k = k_of(g["E"].iloc[0], k0, rho) if g["arrival"].iloc[0] else K_PROD
        for t, (_, p, r) in PAIRS.items():
            pred[t][pos:pos + len(g)] = replay(g, k, t) * g[f"f_{t}"].to_numpy() * g["weight"].to_numpy()
        pos += len(g)
    return pred


def score(log_path):
    OUT.mkdir(parents=True, exist_ok=True)
    cut = identity(log_path)
    log, ent = load(log_path)
    log = log[log["n"] < log["Player"].map(cut).fillna(np.inf)]
    assert not identity_after(log), "replay still differs after the cuts"
    log = log.iloc[np.argsort(pd.Categorical(log["Player"], log["Player"].unique()).codes, kind="stable")]
    act = {t: (log[a] * log["weight"]).to_numpy() for t, (a, _, _) in PAIRS.items()}
    arr = log["arrival"].astype(bool).to_numpy()
    first2 = (log["Season"] <= log["Entry"] + 1).to_numpy()
    train_e, test_e = (log["Entry"] <= 2022).to_numpy(), (log["Entry"] >= 2023).to_numpy()
    masks = {"train": arr & first2 & train_e, "test": arr & first2 & test_e,
             "guard_train": arr & train_e, "guard_test": arr & test_e}
    se = {}
    for k0 in K0_GRID:
        for rho in RHO_GRID:
            pr = predictions(log, k0, rho)
            se[(k0, rho)] = {t: (pr[t] - act[t]) ** 2 for t in PAIRS}
    base = se[(16, 0.0)]
    grid = {}
    for key, s in se.items():
        grid[f"{key[0]}|{key[1]}"] = {f"{m}_{t}": float(s[t][mk].sum() / base[t][mk].sum() - 1)
                                     for m, mk in masks.items() for t in PAIRS}
        g = grid[f"{key[0]}|{key[1]}"]
        g["train_mean"] = (g["train_npxG"] + g["train_xAG"]) / 2
    best = min(se, key=lambda k: grid[f"{k[0]}|{k[1]}"]["train_mean"])
    players = log["Player"].to_numpy()
    res = {"players_cut_at_reblend": len(cut), "arrivals": int(ent["arrival"].sum()),
           "rows": {m: int(mk.sum()) for m, mk in masks.items()},
           "E_quartiles_test": [float(x) for x in np.nanpercentile(log.loc[masks["test"]].groupby("Player")["E"].first(), [25, 50, 75])],
           "grid": grid, "best": {"K0": best[0], "rho": best[1]}, "test": {}}
    for t in PAIRS:
        d, ci = vpk.boot_diff(base[t][masks["test"]], se[best][t][masks["test"]], players[masks["test"]])
        res["test"][t] = {"change": d, "ci95": ci, "guard_test": grid[f"{best[0]}|{best[1]}"][f"guard_test_{t}"]}
    # secondary: test by evidence tercile and by arrival type, at the chosen setting
    tl = log.loc[masks["test"]]
    cuts = np.nanpercentile(tl.groupby("Player")["E"].first(), [100 / 3, 200 / 3])
    res["test_by_evidence"] = {}
    for name, lo, hi in (("low", -np.inf, cuts[0]), ("mid", cuts[0], cuts[1]), ("high", cuts[1], np.inf)):
        mk = masks["test"] & (log["E"] > lo).to_numpy() & (log["E"] <= hi).to_numpy()
        res["test_by_evidence"][name] = {t: float(se[best][t][mk].sum() / base[t][mk].sum() - 1) for t in PAIRS} | {
            "rows": int(mk.sum())}
    res["test_by_type"] = {}
    for kind in ("arrival_fitted", "arrival_old"):
        mk = masks["test"] & (log["Type"] == kind).to_numpy()
        res["test_by_type"][kind] = {t: float(se[best][t][mk].sum() / base[t][mk].sum() - 1) for t in PAIRS} | {
            "rows": int(mk.sum())}
    res["adopt"] = bool(best != (16, 0.0) and all(
        res["test"][t]["change"] < 0 and res["test"][t]["guard_test"] <= 0.0005 for t in PAIRS))
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "grid"}, indent=1))


if __name__ == "__main__":
    {"identity": identity, "score": score}[sys.argv[1]](sys.argv[2])
