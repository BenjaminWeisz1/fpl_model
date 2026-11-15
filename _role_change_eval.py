"""§39 evaluation: paired OFF(detect-only) vs ON(boost) on identical flagged rows.

Content-equivalence of the APPLY=0 arm vs the baseline, aggregate MAE, the
boost-active subset MAE, and a player-clustered paired bootstrap CI on the
per-row error reduction (the §34 falsification lesson: gate on a bootstrap CI at
per-entity granularity, not a point MAE delta).
"""
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent

base = pd.read_csv(_ROOT / "_pos_goal_baseline.csv", encoding="utf-8-sig")
off_ratings = pd.read_csv(_ROOT / "_pos_goal_applyoff.csv", encoding="utf-8-sig")
on_ratings = pd.read_csv(_ROOT / "_pos_goal_applyon.csv", encoding="utf-8-sig")


def content_equal(a, b):
    ka = a.sort_values(list(a.columns)).reset_index(drop=True)
    kb = b.sort_values(list(b.columns)).reset_index(drop=True)
    return ka.equals(kb)


print("=== criterion 1: equivalence (content, not md5 -- md5 is unstable) ===")
print(f"  APPLY=0 ratings content == baseline : {content_equal(base, off_ratings)}")
mm = base.merge(on_ratings, on="Player", suffixes=("_b", "_on"))
print(f"  APPLY=1 players whose NPG/90 moved  : "
      f"{int((np.abs(mm['NPG/90_b']-mm['NPG/90_on'])>1e-12).sum())}/{len(mm)}")
print(f"  APPLY=1 players whose Assists/90 moved: "
      f"{int((np.abs(mm['Assists/90_b']-mm['Assists/90_on'])>1e-12).sum())}/{len(mm)}")

off = pd.read_csv(_ROOT / "_role_diag_off.csv", encoding="utf-8-sig")
on = pd.read_csv(_ROOT / "_role_diag_on.csv", encoding="utf-8-sig")

# positional alignment is exact: same loop, same processed rows, same order
assert len(off) == len(on), (len(off), len(on))
for c in ["Player", "Season", "MatchID", "Slot", "BoostActive"]:
    a, b = off[c].values, on[c].values
    same = (a == b) | (pd.isna(a) & pd.isna(b))
    assert same.all(), f"flag mismatch on {c}"
print(f"\n  diag rows aligned: {len(off):,}; boost-active flagged identically: "
      f"{int(off['BoostActive'].sum()):,}")

# per-row weighted absolute error
for df in (off, on):
    df["abs_npg"] = np.abs(df["npxg90"] - df["pred_goals_90"]) * df["weight"]
    df["abs_ast"] = np.abs(df["xag90"] - df["pred_ast_90"]) * df["weight"]


def wmae(num, den):
    return num.sum() / den.sum() if den.sum() else np.nan


print("\n=== criterion 3: aggregate (total-effect) MAE, all processed rows ===")
for lab, col in [("npxG", "abs_npg"), ("xAG", "abs_ast")]:
    o = wmae(off[col], off["weight"])
    n = wmae(on[col], on["weight"])
    print(f"  {lab}: OFF {o:.5f}  ON {n:.5f}  delta(ON-OFF) {n-o:+.5f}")

mask = off["BoostActive"] == 1
print(f"\n=== criterion 2: boost-active subset ({int(mask.sum()):,} rows, "
      f"{off.loc[mask,'Player'].nunique()} players) ===")
sub_o = off[mask]
sub_n = on[mask]
for lab, col in [("npxG", "abs_npg"), ("xAG", "abs_ast")]:
    o = wmae(sub_o[col], sub_o["weight"])
    n = wmae(sub_n[col], sub_n["weight"])
    print(f"  {lab}: OFF {o:.5f}  ON {n:.5f}  delta(ON-OFF) {n-o:+.5f}  "
          f"(improvement {o-n:+.5f})")

# paired per-row reduction (OFF - ON; positive = ON better), clustered by player
print("\n=== paired player-clustered bootstrap (2000 resamples) on boost rows ===")
rng = np.random.default_rng(42)
red = pd.DataFrame({
    "Player": sub_o["Player"].values,
    "w": sub_o["weight"].values,
    "d_npg": (sub_o["abs_npg"] - sub_n["abs_npg"]).values,
    "d_ast": (sub_o["abs_ast"] - sub_n["abs_ast"]).values,
})
players = red["Player"].unique()
# precompute per-player weighted totals so the bootstrap is a fast array gather
agg = red.groupby("Player")[["w", "d_npg", "d_ast"]].sum()
agg = agg.reindex(players)
W = agg["w"].values
D = {"d_npg": agg["d_npg"].values, "d_ast": agg["d_ast"].values}
n_players = len(players)


def boot_ci(dcol):
    d = D[dcol]
    stats = np.empty(2000)
    for b in range(2000):
        idx = rng.integers(0, n_players, n_players)
        den = W[idx].sum()
        stats[b] = d[idx].sum() / den if den else np.nan
    return np.nanmean(stats), np.nanpercentile(stats, 2.5), np.nanpercentile(stats, 97.5)



for lab, dcol in [("npxG", "d_npg"), ("xAG", "d_ast")]:
    pt = red[dcol].sum() / red["w"].sum()
    m, lo, hi = boot_ci(dcol)
    verdict = "CI excludes 0 (ON better)" if lo > 0 else (
        "CI excludes 0 (ON WORSE)" if hi < 0 else "CI straddles 0 (not significant)")
    print(f"  {lab} weighted mean per-row error reduction (OFF-ON): point {pt:+.5f}; "
          f"bootstrap mean {m:+.5f}, 95% CI [{lo:+.5f}, {hi:+.5f}] -> {verdict}")
