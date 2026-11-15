"""§39 sensitivity: characterize the falsification (not a ship-fishing sweep).

Runs matched APPLY=0 / APPLY=1 arms for each config and reports the boost-active
subset MAE delta + a player-clustered bootstrap CI, to show whether the negative
result is (a) monotonic in boost strength (faster learning on a noisy per-match
target always hurts), and (b) robust to a tighter role-change gate.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
_SCRIPT = _ROOT / "scripts" / "player_ratings" / "build_player_ratings.py"

CONFIGS = [
    # (label, boost_alpha, factor, window)  -- primary already measured in _role_change_eval
    ("gentler  a=0.08 f=1.5 w=8", "0.08", "1.5", "8"),
    ("stronger a=0.25 f=1.5 w=8", "0.25", "1.5", "8"),
    ("tight    a=0.15 f=2.5 w=8", "0.15", "2.5", "8"),
]


def run(apply_flag, boost_alpha, factor, window, diag_out):
    env = dict(os.environ)
    env.update({
        "PYTHONHASHSEED": "0",
        "GOAL_MODEL_UNDERSTAT_POS": "1",
        "GOAL_MODEL_DIAG": "1",
        "GOAL_ROLE_APPLY": apply_flag,
        "GOAL_ROLE_BOOST_ALPHA": boost_alpha,
        "GOAL_ROLE_FACTOR": factor,
        "GOAL_ROLE_WINDOW": window,
        "GOAL_DIAG_OUT": diag_out,
    })
    subprocess.run([sys.executable, str(_SCRIPT)], env=env, cwd=str(_ROOT),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def subset_delta(off_csv, on_csv):
    off = pd.read_csv(_ROOT / off_csv, encoding="utf-8-sig")
    on = pd.read_csv(_ROOT / on_csv, encoding="utf-8-sig")
    mask = off["BoostActive"] == 1
    o, n = off[mask], on[mask]
    out = {}
    for lab, np_, pp in [("npxG", "npxg90", "pred_goals_90"),
                         ("xAG", "xag90", "pred_ast_90")]:
        eo = (np.abs(o[np_] - o[pp]) * o["weight"])
        en = (np.abs(n[np_] - n[pp]) * n["weight"])
        out[lab] = (eo.sum() / o["weight"].sum(), en.sum() / n["weight"].sum(),
                    (eo - en).values, o["weight"].values, o["Player"].values)
    out["n_rows"] = int(mask.sum())
    out["n_players"] = int(o["Player"].nunique())
    return out


def boot(d, w, players, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"p": players, "d": d, "w": w})
    agg = df.groupby("p")[["d", "w"]].sum()
    D, W = agg["d"].values, agg["w"].values
    n = len(D)
    stats = np.empty(2000)
    for b in range(2000):
        idx = rng.integers(0, n, n)
        stats[b] = D[idx].sum() / W[idx].sum()
    return np.percentile(stats, 2.5), np.percentile(stats, 97.5)


for label, a, f, w in CONFIGS:
    run("0", a, f, w, "_sens_off.csv")
    run("1", a, f, w, "_sens_on.csv")
    r = subset_delta("_sens_off.csv", "_sens_on.csv")
    print(f"\n### {label}   [subset {r['n_rows']} rows / {r['n_players']} players]")
    for lab in ("npxG", "xAG"):
        o_mae, n_mae, d, wt, pl = r[lab]
        lo, hi = boot(d, wt, pl)
        pt = d.sum() / wt.sum()
        v = ("ON better*" if lo > 0 else "ON WORSE*" if hi < 0 else "ns")
        print(f"  {lab}: OFF {o_mae:.5f} -> ON {n_mae:.5f}  delta(ON-OFF) {n_mae-o_mae:+.5f}  "
              f"| paired reduction {pt:+.5f} CI [{lo:+.5f},{hi:+.5f}] {v}")
print("\n(* CI excludes 0 at 95%, player-clustered)")
