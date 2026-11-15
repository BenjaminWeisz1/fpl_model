"""
Goal/assist ratings shrunk toward their position group at forecast time (CHANGELOG §122). Offline on a pre-match log.

    python validate_rating_shrink.py [log.csv]
"""
from fpl_model import paths
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = paths.ROOT
OUT = paths.VALIDATION / "rating_shrink_20261030"
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "validation" / "calibration_diag_20261030" / "log.csv"
TARGETS = {"npxG": ("npxg90", "pred_goals_90", "NPG_Rating_Pre"), "xAG": ("xag90", "pred_ast_90", "Assist_Rating_Pre")}
GROUP = {"CB": "DEF", "FB": "DEF", "WB": "DEF", "DM": "MID", "CM": "MID", "WM": "MID",
         "AM": "ATT", "W": "ATT", "FW": "ATT", "GK": "GK"}
GRID = np.round(np.arange(0.50, 1.0001, 0.02), 2)


def position_groups(log):
    """Player's group in each season: first position of his latest transfers.csv row with Season <= season."""
    t = pd.read_csv(ROOT / "data" / "current" / "transfers.csv", encoding="utf-8-sig").dropna(subset=["Pos"])
    t = t.sort_values("Season")
    rows = {p: list(zip(g["Season"], g["Pos"])) for p, g in t.groupby("Player")}
    out = {}
    for p, s in log[["Player", "Season"]].drop_duplicates().itertuples(index=False):
        pos = [q for ss, q in rows.get(p, []) if ss <= s]
        out[(p, s)] = GROUP.get(str(pos[-1]).split(",")[0].strip(), None) if pos else None
    return out


def frame():
    log = pd.read_csv(LOG, encoding="utf-8-sig")
    g = position_groups(log)
    log["group"] = [g[(p, s)] for p, s in zip(log["Player"], log["Season"])]
    d = log[(log["Season"] >= 2018) & log["group"].isin(["DEF", "MID", "ATT"])].copy()
    for t_, (a, p, r) in TARGETS.items():
        d = d[d[r] > 0]
        lw = (np.log(d[r]) * d["weight"]).groupby([d["group"], d["Season"]]).transform("sum")
        w = d["weight"].groupby([d["group"], d["Season"]]).transform("sum")
        d[f"mlog_{t_}"] = np.exp(lw / w)
        d[f"mlin_{t_}"] = (d[r] * d["weight"]).groupby([d["group"], d["Season"]]).transform("sum") / w
    return d


def shrunk_pred(d, t_, lam, kind):
    a, p, r = TARGETS[t_]
    if kind == "log":
        m = d[f"mlog_{t_}"]
        rr = m * (d[r] / m) ** lam
    else:
        m = d[f"mlin_{t_}"]
        rr = m + lam * (d[r] - m)
    return d[p] * rr / d[r]


def sse(d, t_, pred):
    a = TARGETS[t_][0]
    return ((pred - d[a]) * d["weight"]) ** 2


def boot(d, base, cand, n=2000, seed=42):
    g = pd.DataFrame({"p": d["Player"].to_numpy(), "b": base.to_numpy(), "c": cand.to_numpy()}).groupby("p").sum()
    b, c = g["b"].to_numpy(), g["c"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(b), (n, len(b)))
    dr = c[idx].sum(1) / b[idx].sum(1) - 1
    return float(c.sum() / b.sum() - 1), [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))]


def deciles(d, t_, pred):
    a, p, r = TARGETS[t_]
    q = pd.qcut(d[r].rank(method="first"), 10, labels=False) + 1
    y, m = d[a] * d["weight"], pred * d["weight"]
    return {int(k): round(float(y[q == k].sum() / m[q == k].sum()), 3) for k in (1, 2, 10)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = frame()
    tr, te, te25 = d[d["Season"] <= 2022], d[d["Season"] >= 2023], d[d["Season"] >= 2025]
    res = {"rows": {"train": len(tr), "test": len(te), "test_2025plus": len(te25)}}
    for kind in ("log", "linear"):
        for t_ in TARGETS:
            p = TARGETS[t_][1]
            curve = {float(l): float(sse(tr, t_, shrunk_pred(tr, t_, l, kind)).sum()) for l in GRID}
            lam = min(curve, key=curve.get)
            r = {"lambda": lam, "train_change": curve[lam] / curve[1.0] - 1}
            for name, s in (("test", te), ("test_2025plus", te25)):
                base, cand = sse(s, t_, s[p]), sse(s, t_, shrunk_pred(s, t_, lam, kind))
                ch, ci = boot(s, base, cand)
                r[name] = {"change": ch, "ci95": ci, "deciles_base": deciles(s, t_, s[p]),
                           "deciles_cand": deciles(s, t_, shrunk_pred(s, t_, lam, kind))}
            res[f"{kind} {t_}"] = r
            print(kind, t_, "lambda", lam, "train", round(r["train_change"], 5),
                  {k: (round(r[k]["change"], 5), [round(x, 5) for x in r[k]["ci95"]], r[k]["deciles_base"], r[k]["deciles_cand"])
                   for k in ("test", "test_2025plus")}, flush=True)
    ok = all(res[f"log {t}"]["test"]["change"] < 0 and all(
        abs(res[f"log {t}"]["test"]["deciles_cand"][k] - 1) < abs(res[f"log {t}"]["test"]["deciles_base"][k] - 1)
        for k in (1, 10)) for t in TARGETS)
    res["adopt_log"] = bool(ok)
    res["linear_beats_log"] = bool(all(res[f"linear {t}"]["test"]["change"] < res[f"log {t}"]["test"]["change"] for t in TARGETS))
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("adopt log arm:", res["adopt_log"], "| linear beats log:", res["linear_beats_log"])


if __name__ == "__main__":
    main()
