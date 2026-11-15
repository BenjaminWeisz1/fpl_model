"""
Diagnostic: is over-prediction of high goal/assist ratings larger when they rest on little recent evidence?
(CHANGELOG §109). Nothing here changes a model. Reuses the §108 log and target starts.

Run locally (after diagnose_inactivity.py's logged run):
    python diagnose_evidence.py
"""
import json

import numpy as np
import pandas as pd

from fpl_model.player_ratings import diagnose_inactivity as d


def ratio(x, k):
    return round(float(x[f"{k}_pred"].sum() / x[f"{k}_act"].sum()), 3)


def main():
    f, t = d.build()
    # minutes logged in the current and previous season before this match
    ev = np.zeros(len(f))
    hist = {}
    for i, (p, s, m) in enumerate(zip(f["Player"], f["Season"], f["minutes"])):
        h = hist.setdefault(p, [])
        ev[i] = sum(mm for ss, mm in h if ss >= s - 1)
        h.append((s, m))
    t = t.join(pd.Series(ev, index=f.index, name="evidence"))
    t["tercile"] = t.groupby("era")["evidence"].transform(lambda x: pd.qcut(x, 3, labels=["low", "mid", "high"]))
    out = {}
    for name, sub in (("high_rated", t[t["high_rated"]]), ("all", t)):
        o = {"tercile_minutes": sub.groupby("tercile", observed=True)["evidence"].median().round(0).to_dict()}
        for b, g in sub.groupby("tercile", observed=True):
            o[str(b)] = {"rows": int(len(g)), "players": int(g["Player"].nunique()), "npxG": ratio(g, "npxG"), "xAG": ratio(g, "xAG")}
        for e, ge in sub.groupby("era"):
            o[f"era_{e}"] = {str(b): {"rows": int(len(g)), "npxG": ratio(g, "npxG"), "xAG": ratio(g, "xAG")}
                             for b, g in ge.groupby("tercile", observed=True)}
        s = sub[sub["tercile"].isin(["low", "high"])]
        cols = [s[c].where(s["tercile"] == lab, 0) for k in ("npxG", "xAG") for lab in ("low", "high") for c in (f"{k}_pred", f"{k}_act")]
        agg = pd.concat(cols, axis=1).groupby(s["Player"]).sum().to_numpy()
        idx = np.random.default_rng(d.SEED).integers(0, len(agg), size=(d.N_BOOT, len(agg)))
        bs = agg[idx].sum(axis=1)
        for j, k in enumerate(("npxG", "xAG")):
            q = 4 * j
            diff = bs[:, q] / bs[:, q + 1] - bs[:, q + 2] / bs[:, q + 3]
            o[f"{k}_low_minus_high_ci95"] = np.percentile(diff, [2.5, 97.5]).round(3).tolist()
        out[name] = o
    h = out["high_rated"]
    out["decision"] = {"npxG_low_vs_high_pct": round(100 * (h["low"]["npxG"] / h["high"]["npxG"] - 1), 1),
                       "ci_excludes_zero": h["npxG_low_minus_high_ci95"][0] > 0}
    out["decision"]["goes_forward"] = out["decision"]["npxG_low_vs_high_pct"] >= 5 and out["decision"]["ci_excludes_zero"]
    (d.DIR / "evidence_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
