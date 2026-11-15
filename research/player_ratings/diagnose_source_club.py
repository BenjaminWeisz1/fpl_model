"""
Diagnostic (CHANGELOG §147): are arrivals from clubs that dominate their league over-rated?

Calibration of each arrival's prior (his first logged rating) against his first PL season, by source-club
strength z (log team npxG per match / league mean, minutes-weighted over his prior seasons). Slope c: the
multiplier exp(c z) on the prior minimising squared match error; c < 0 means strong-club arrivals are over-rated.
"Clear" (fixed before the first run): a group of 50+ arrivals with a 95% player-bootstrap interval for c below zero.

Run locally:
    python -m research.player_ratings.diagnose_source_club <pre-match log>
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from fpl_model.player_ratings import build_new_player_priors as bnp  # noqa: E402
from fpl_model.player_ratings import fotmob_priors as fp  # noqa: E402
from fpl_model.player_ratings import new_player_priors as npp  # noqa: E402

OUT = npp.ROOT / "data" / "validation" / "source_club_20261030" / "results.json"
N_BOOT = 2000
MIN_ARRIVALS = 50


def slope(z, A, B):
    """c minimising sum[ exp(2cz) B - 2 exp(cz) A ]; A = sum(y m p), B = sum(m^2 p^2)."""
    f = lambda c: (np.exp(2 * c * z) * B - 2 * np.exp(c * z) * A).sum()
    return minimize_scalar(f, bounds=(-3, 3), method="bounded").x


def main():
    log_path = sys.argv[1]
    arr, slots = bnp.dataset(with_outcomes=True, log_path=log_path)
    arr, slots = fp.attach_fotmob(arr, slots, fp.load_fotmob())
    log = pd.read_csv(log_path, encoding="utf-8-sig")
    arr = npp.incumbent_from_log(arr, log)
    w = slots["m"]
    arr["z"] = (w * slots["z"]).sum(1) / np.where(w.sum(1) > 0, w.sum(1), 1)
    a = arr[arr["New"] & (arr["all_npxG_N"] > 0) & arr["Path"].isin(["understat", "fotmob"])
            & arr["inc_npxG"].notna()].reset_index(drop=True)
    print("arrivals with a source-club z", len(a), a.groupby(["Group", "Path"]).size().to_dict())
    groups = {"all": a.index, **{g: x.index for g, x in a.groupby("Group")}, "2026_all": a.index[a["Season"] == 2026],
              **{f"season_{s}": a.index[a["Season"] == s] for s in range(2021, 2026)}}
    rng = np.random.default_rng(42)
    res = {}
    for name, idx in groups.items():
        g = a.loc[idx]
        if len(g) < 15:
            continue
        res[name] = {"n": len(g)}
        for t, inc in (("npxG", "inc_npxG"), ("xAG", "inc_xAG")):
            p = g[inc].to_numpy(float)
            y, m, b = g[f"all_{t}_Y"].to_numpy(float), g[f"all_{t}_M"].to_numpy(float), g[f"all_{t}_B"].to_numpy(float)
            A = g[f"all_{t}_A"].to_numpy(float) * p
            B = b * p ** 2
            z = g["z"].to_numpy(float)
            c = slope(z, A, B)
            boots = [slope(z[i], A[i], B[i]) for i in rng.integers(0, len(g), (N_BOOT, len(g)))]
            q = pd.qcut(z, 3, labels=False, duplicates="drop")
            terc = {}
            for k in sorted(set(q)):
                s = q == k
                terc[f"T{k + 1}"] = {"z": round(float(z[s].mean()), 3), "n": int(s.sum()),
                                     "pred_over_actual": round(float((p[s] * m[s]).sum() / max(y[s].sum(), 1e-9)), 3)}
            res[name][t] = {"c": round(float(c), 3), "ci95": np.percentile(boots, [2.5, 97.5]).round(3).tolist(),
                            "pred_over_actual": round(float((p * m).sum() / y.sum()), 3), "terciles": terc}
    res["clear"] = {k: [t for t in ("npxG", "xAG") if v[t]["ci95"][1] < 0]
                    for k, v in res.items() if isinstance(v, dict) and v.get("n", 0) >= MIN_ARRIVALS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, v in res.items():
        if k == "clear":
            print("clear:", v)
            continue
        print(k, v["n"])
        for t in ("npxG", "xAG"):
            print("  ", t, "c", v[t]["c"], v[t]["ci95"], "cal", v[t]["pred_over_actual"],
                  {x: (y["z"], y["pred_over_actual"]) for x, y in v[t]["terciles"].items()})
    top = a.assign(p=a["inc_npxG"] * a["all_npxG_M"], y=a["all_npxG_Y"]).sort_values("z", ascending=False).head(20)
    print(top[["Player", "Season", "League", "z", "p", "y"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
