"""
Out-of-time test of red-card variants for the team ratings (CHANGELOG §135).

Builds one match_results file per variant (only red-card matches' NPxG_Adj differ), runs
evaluate_team_ratings_oot.py on each with --dump-predictions, and scores the held-out predictions at full precision
(Poisson LL kernel pred - NPG x ln(pred) per team-match, the harness's primary metric).

    python validate_red_card_variants.py run
    python validate_red_card_variants.py score     # -> data/validation/red_card_20261030/variants/results.json
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import red_card_adjust as rca  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "data" / "validation" / "red_card_20261030"
OUT = VAL / "variants"
HARNESS = Path(__file__).resolve().parent / "evaluate_team_ratings_oot.py"
CUTOFFS = (2023, 2024, 2025, 2026)
FIXED = {"a_none": (1.0, 1.0, None, None), "b_auto": (rca.UP, rca.DOWN, rca.UP2, rca.DOWN2),
         "c1_research": (0.72, 1.65, 0.52, 2.5)}


def write(m, vals, path):
    m = m.copy()
    for side, v in vals.items():
        m.loc[v.index, f"{side}_NPxG_Adj"] = v.to_numpy()
    m.to_csv(path, index=False, encoding="utf-8-sig")


def build_files():
    """{(variant, cutoff or None): path}"""
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(rca.DATA / "match_results.csv", encoding="utf-8-sig")
    stats = pd.read_csv(rca.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    prod = rca.build(out_path=None)
    b = rca.bases(m, stats)
    ratio = rca.level_ratios(m, rca.red_rows(m, prod), b)   # no-red rows are identical in every variant
    files = {}
    for name, (up, down, up2, down2) in FIXED.items():
        fac = rca.build(up, down, up2, down2, out_path=None)
        vals, _ = rca.adjusted_values(m, fac, stats, ratio)
        files[(name, None)] = OUT / f"match_results_{name}.csv"
        write(m, vals, files[(name, None)])
    est = json.loads((VAL / "factors_estimate.json").read_text(encoding="utf-8"))["per_cutoff"]
    for c in CUTOFFS:
        f = est[str(c)]
        fac = rca.build(f["state+1"], f["state-1"], f["state+2"], f["state-2"], out_path=None)
        vals, _ = rca.adjusted_values(m, fac, stats, ratio)
        files[("c2_fitted", c)] = OUT / f"match_results_c2_fitted_{c}.csv"
        write(m, vals, files[("c2_fitted", c)])
    # (d) post-red minutes carry no information: pre-red xG share of the base + predicted goals for unequal minutes
    hist = pd.read_csv(rca.DATA / "team_ratings_history.csv", encoding="utf-8-sig")
    pred = hist.set_index(["Season", "Home", "Away"])[["Home_Prediction", "Away_Prediction"]]
    red = rca.red_rows(m, prod)
    f = prod.set_index(["Season", "Home", "Away"])
    idx = m.index[red]
    k = pd.MultiIndex.from_arrays([m.loc[idx, "Season"], m.loc[idx, "Home"], m.loc[idx, "Away"]])
    level = m["Season"].map(ratio).fillna(1.0)[idx]
    frac = (f["Unequal_Minutes"] / f["Match_Minutes"]).reindex(k).to_numpy()
    vals = {s: b[s][idx] * level * (1 - f[f"{s}_Post_Share"].reindex(k).to_numpy())
            + pred[f"{s}_Prediction"].reindex(k).to_numpy() * frac for s in ("Home", "Away")}
    files[("d_drop", None)] = OUT / "match_results_d_drop.csv"
    write(m, vals, files[("d_drop", None)])
    files[("manual", None)] = VAL / "match_results_manual.csv"
    return files


def run_one(item):
    (name, c), path = item
    tag = name if c is None else f"{name}_{c}"
    cmd = [sys.executable, "-u", "-X", "utf8", str(HARNESS), "--label", f"rc_{tag}", "--match-results", str(path),
           "--no-equivalence-check", "--dump-predictions", str(OUT / f"pred_{tag}.csv")]
    if c is not None:
        cmd += ["--cutoffs", str(c)]
    with open(OUT / f"oot_{tag}.log", "w", encoding="utf-8") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
    return tag


def run():
    files = build_files()
    with ThreadPoolExecutor(max_workers=4) as ex:
        for tag in ex.map(run_one, files.items()):
            print(f"{tag} done", flush=True)


def ll(p, y):
    return p - y * np.log(p)


def score():
    preds = {}
    for name in ("a_none", "b_auto", "c1_research", "d_drop", "manual"):
        preds[name] = pd.read_csv(OUT / f"pred_{name}.csv")
    preds["c2_fitted"] = pd.concat([pd.read_csv(OUT / f"pred_c2_fitted_{c}.csv") for c in CUTOFFS], ignore_index=True)
    res = {}
    for name, p in preds.items():
        r = {}
        for mode, (h, a) in (("frozen", ("Frozen_Home", "Frozen_Away")), ("walk_forward", ("WF_Home", "WF_Away"))):
            per = ll(p[h], p["Home_NPG"]) + ll(p[a], p["Away_NPG"])
            r[mode] = float(per.sum() / (2 * len(p)))
            r[f"{mode}_by_cutoff"] = {str(c): float(per[p["Cutoff"] == c].sum() / (2 * (p["Cutoff"] == c).sum()))
                                      for c in CUTOFFS}
        res[name] = r
    b = res["b_auto"]
    for name, r in res.items():
        better = sum(r["frozen_by_cutoff"][str(c)] < b["frozen_by_cutoff"][str(c)] for c in CUTOFFS)
        r["vs_b"] = {"frozen_pct": 100 * (r["frozen"] / b["frozen"] - 1),
                     "walk_forward_pct": 100 * (r["walk_forward"] / b["walk_forward"] - 1),
                     "cutoffs_better": better}
        r["passes"] = bool(name not in ("b_auto", "manual") and r["frozen"] < b["frozen"] and better >= 3
                           and r["walk_forward"] <= b["walk_forward"])
    passing = [n for n, r in res.items() if r["passes"]]
    res["adopt"] = min(passing, key=lambda n: res[n]["frozen"]) if passing else "b_auto"
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for name, r in res.items():
        if name != "adopt":
            print(f"{name:12s} frozen {r['frozen']:.6f} ({r['vs_b']['frozen_pct']:+.4f}%) "
                  f"cutoffs better {r['vs_b']['cutoffs_better']}/4 | wf {r['walk_forward']:.6f} "
                  f"({r['vs_b']['walk_forward_pct']:+.4f}%) | by cutoff "
                  f"{ {c: round(v, 5) for c, v in r['frozen_by_cutoff'].items()} } passes={r['passes']}")
    print("adopt:", res["adopt"])


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
