"""
Test whether incorporating recent-form weighting into the NPG conversion rate
improves out-of-time prediction vs the incumbent career + shrinkage formula.

RESULT (2026-10-14): HYPOTHESIS REJECTED.
  Best variant (blend60 + shrinkage=75): +0.00018 MAE (+0.33%).
  95% bootstrap CI [-0.00031, +0.00071] straddles zero.
  No form-weighting approach improved significantly over the incumbent.
  Consistent with research: finishing overperformance (G - npxG) has
  year-to-year correlation ~0.12; recency-weighting noise extracts no signal.
  The incumbent formula is not changed.

Incumbent (build_player_ratings.py):
    Conversion = (career_NPG + 30) / (career_npxG + 30)

Tests three families of alternatives:
  A) Match-level exponential decay at various half-lives
  B) Season-level blending: blend career conversion with last-N-seasons conversion
  C) Different shrinkage priors (30, 75, 150, 250, 400)

Method:
    For each held-out season C in {2022, 2023, 2024, 2025}:
      1. Compute each player's career totals using only matches BEFORE season C.
      2. Compute form-weighted totals per variant.
      3. Predict held-out season NPG/90 = npxG/90 * conversion for each variant.
      4. Score each variant by MAE against actual held-out NPG/90.

Reports the incumbent vs all variants. Honest: if nothing helps, say so.

Run: python _test_form_conversion.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data" / "current"

# -- load data --
ps = pd.read_csv(_DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
ps["Season"] = np.where(ps["Data_Source"] == "fbref", 2017 + ps["MatchID"] // 380, 2025)
ps = ps.sort_values("MatchID").reset_index(drop=True)
ps["NPG"] = ps["Goals"].fillna(0) - ps["PK"].fillna(0)
ps = ps[ps["Minutes"] > 0].copy()

# -- evaluation setup --
CUTOFFS = [2022, 2023, 2024, 2025]
MIN_CAREER_MATCHES = 20
MIN_TEST_MINUTES = 300

# -- variant definitions --
# Family A: match-level exponential decay half-lives
HALF_LIVES = [15, 25, 38, 55, 80]
# Family B: season-level blends (blend career conv with recent-N-seasons conv)
SEASON_BLENDS = [0.25, 0.40, 0.50, 0.60, 0.75]  # weight on recent
RECENT_WINDOW = 2  # last N seasons for "recent" component
# Family C: shrinkage priors
SHRINKAGE_VALUES = [30, 75, 150, 250, 400]


def compute_match_decay_conversion(train_df, max_match_id, half_life, shrink):
    """Form-weighted (wNPG + S) / (wnpxG + S) per player."""
    decay = np.log(2) / half_life
    w = np.exp(-decay * (max_match_id - train_df["MatchID"]))
    g = train_df["NPG"].fillna(0) * w
    x = train_df["npxG"].fillna(0) * w
    agg = pd.DataFrame({"w_NPG": g, "w_npxG": x}).groupby(train_df["Player"]).sum()
    return ((agg["w_NPG"] + shrink) / (agg["w_npxG"] + shrink)).to_dict()


def compute_plain_conversion(train_df, shrink):
    """Plain (career_NPG + S) / (career_npxG + S)."""
    agg = train_df.groupby("Player").agg(cNPG=("NPG", "sum"), cnpxG=("npxG", "sum"))
    return ((agg["cNPG"] + shrink) / (agg["cnpxG"] + shrink)).to_dict()


def compute_season_blend_conversion(train_df, cutoff, recent_n, blend_w, shrink):
    """Blend career conversion with recent-N-seasons conversion."""
    career = compute_plain_conversion(train_df, shrink)
    recent = train_df[train_df["Season"] >= cutoff - recent_n]
    if recent.empty:
        return career
    recent_conv = compute_plain_conversion(recent, shrink)
    # blend: blend_w * recent + (1 - blend_w) * career
    blended = {}
    for p in career:
        c = career[p]
        r = recent_conv.get(p, c)
        blended[p] = blend_w * r + (1 - blend_w) * c
    return blended


def evaluate_cut(cutoff, shrink=30):
    train = ps[ps["Season"] < cutoff]
    test  = ps[ps["Season"] == cutoff]
    if train.empty or test.empty:
        return None

    max_mid = train["MatchID"].max()
    career_n = train.groupby("Player").size()

    # test aggregates
    t = test.groupby("Player").agg(
        npxg_sum=("npxG", "sum"), npg_sum=("NPG", "sum"),
        mins_sum=("Minutes", "sum"),
    ).reset_index()
    t = t[t["mins_sum"] >= MIN_TEST_MINUTES]
    t = t[t["Player"].isin(career_n[career_n >= MIN_CAREER_MATCHES].index)]
    if t.empty:
        return None

    t["npxg90"] = t["npxg_sum"] * 90 / t["mins_sum"]
    t["npg90"]  = t["npg_sum"]  * 90 / t["mins_sum"]

    results = {"n": len(t), "cutoff": cutoff}

    # Incumbent
    conv = compute_plain_conversion(train, shrink)
    pred = t["Player"].map(conv).fillna(1.0) * t["npxg90"]
    results["incumbent"] = np.mean(np.abs(pred - t["npg90"]))

    # Family A: match-level decay
    for hl in HALF_LIVES:
        conv = compute_match_decay_conversion(train, max_mid, hl, shrink)
        pred = t["Player"].map(conv).fillna(1.0) * t["npxg90"]
        results[f"A_hl{hl}"] = np.mean(np.abs(pred - t["npg90"]))

    # Family B: season blends
    for bw in SEASON_BLENDS:
        conv = compute_season_blend_conversion(train, cutoff, RECENT_WINDOW, bw, shrink)
        pred = t["Player"].map(conv).fillna(1.0) * t["npxg90"]
        results[f"B_blend{int(bw*100)}"] = np.mean(np.abs(pred - t["npg90"]))

    return results


# ============================================================
# PART 1: Family A -- match-level exponential decay (shrink=30)
# ============================================================
print("=" * 70)
print("FAMILY A: Match-level exponential decay (shrinkage=30)")
print("=" * 70)

a_labels = ["incumbent"] + [f"A_hl{h}" for h in HALF_LIVES]
all_results = []
for c in CUTOFFS:
    r = evaluate_cut(c, shrink=30)
    if r:
        all_results.append(r)

print(f"{'Cutoff':>6} {'n':>5}  {'incumbent':>10}", end="")
for h in HALF_LIVES:
    print(f"  hl={h:>3}", end="")
print()
print("-" * (6 + 6 + 12 + 10 * len(HALF_LIVES)))

pooled = {k: [] for k in a_labels}
pooled_n = {k: [] for k in a_labels}
for r in all_results:
    print(f"{r['cutoff']:>6} {r['n']:>5}  {r['incumbent']:>10.4f}", end="")
    for h in HALF_LIVES:
        v = r[f"A_hl{h}"]
        delta = r["incumbent"] - v
        mark = "+" if delta > 0.0005 else ("-" if delta < -0.0005 else "=")
        print(f"  {v:>7.4f}{mark}", end="")
    print()
    for k in a_labels:
        pooled[k].append(r[k])
        pooled_n[k].append(r["n"])
print()
for k in a_labels:
    print(f"  {k:>12} pooled weighted MAE = {np.average(pooled[k], weights=pooled_n[k]):.5f}")

# ============================================================
# PART 2: Family B -- season-level blends (shrink=30)
# ============================================================
print()
print("=" * 70)
print("FAMILY B: Season-level blends (last 2 seasons, shrinkage=30)")
print("=" * 70)

b_labels = ["incumbent"] + [f"B_blend{int(w*100)}" for w in SEASON_BLENDS]
all_results_b = []
for c in CUTOFFS:
    r = evaluate_cut(c, shrink=30)
    if r:
        all_results_b.append(r)

print(f"{'Cutoff':>6} {'n':>5}  {'incumbent':>10}", end="")
for w in SEASON_BLENDS:
    print(f"  b{int(w*100):>2}", end="")
print()
print("-" * (6 + 6 + 12 + 8 * len(SEASON_BLENDS)))

pooled_b = {k: [] for k in b_labels}
pooled_bn = {k: [] for k in b_labels}
for r in all_results_b:
    print(f"{r['cutoff']:>6} {r['n']:>5}  {r['incumbent']:>10.4f}", end="")
    for w in SEASON_BLENDS:
        v = r[f"B_blend{int(w*100)}"]
        delta = r["incumbent"] - v
        mark = "+" if delta > 0.0005 else ("-" if delta < -0.0005 else "=")
        print(f"  {v:>6.4f}{mark}", end="")
    print()
    for k in b_labels:
        pooled_b[k].append(r[k])
        pooled_bn[k].append(r["n"])
print()
for k in b_labels:
    print(f"  {k:>14} pooled weighted MAE = {np.average(pooled_b[k], weights=pooled_bn[k]):.5f}")

# ============================================================
# PART 3: Family C -- shrinkage prior sweep
# ============================================================
print()
print("=" * 70)
print("FAMILY C: Shrinkage prior sweep")
print("=" * 70)

c_labels = ["incumbent"] + [f"C_s{s}" for s in SHRINKAGE_VALUES if s != 30]
all_results_c = []
for c in CUTOFFS:
    r = evaluate_cut(c, shrink=30)
    if r:
        all_results_c.append(r)

# Need to rerun with different shrinkage values
print(f"{'Cutoff':>6} {'n':>5}", end="")
for s in SHRINKAGE_VALUES:
    print(f"  s={s:>4}", end="")
print()
print("-" * (6 + 6 + 10 * len(SHRINKAGE_VALUES)))

pooled_c = {f"C_s{s}": [] for s in SHRINKAGE_VALUES}
pooled_cn = {f"C_s{s}": [] for s in SHRINKAGE_VALUES}

for c in CUTOFFS:
    train = ps[ps["Season"] < c]
    test  = ps[ps["Season"] == c]
    if train.empty or test.empty:
        continue
    career_n = train.groupby("Player").size()
    t = test.groupby("Player").agg(
        npxg_sum=("npxG", "sum"), npg_sum=("NPG", "sum"),
        mins_sum=("Minutes", "sum"),
    ).reset_index()
    t = t[t["mins_sum"] >= MIN_TEST_MINUTES]
    t = t[t["Player"].isin(career_n[career_n >= MIN_CAREER_MATCHES].index)]
    if t.empty:
        continue
    npxg90 = t["npxg_sum"] * 90 / t["mins_sum"]
    npg90  = t["npg_sum"]  * 90 / t["mins_sum"]

    row_out = f"{c:>6} {len(t):>5}"
    for s in SHRINKAGE_VALUES:
        conv = compute_plain_conversion(train, s)
        pred = t["Player"].map(conv).fillna(1.0) * npxg90
        mae = np.mean(np.abs(pred - npg90))
        pooled_c[f"C_s{s}"].append(mae)
        pooled_cn[f"C_s{s}"].append(len(t))
        row_out += f"  {mae:>8.4f}"
    print(row_out)

print()
for s in SHRINKAGE_VALUES:
    k = f"C_s{s}"
    print(f"  {k:>8} pooled weighted MAE = {np.average(pooled_c[k], weights=pooled_cn[k]):.5f}")

# ============================================================
# PART 4: Combo -- best half-life + best shrinkage + season blend
# ============================================================
print()
print("=" * 70)
print("PART 4: Combo -- best shrinkage + season blend + match decay")
print("=" * 70)

# Find best shrinkage from part 3
best_shrink_key = min(pooled_c, key=lambda k: np.average(pooled_c[k], weights=pooled_cn[k]))
best_shrink = int(best_shrink_key.split("_s")[1])
print(f"Best shrinkage from Family C: {best_shrink}")

# Test combos
combo_configs = [
    ("incumbent",     None,  None, None),
    ("best_shrink",   None,  None, best_shrink),
    ("hl38_s_best",   38,    None, best_shrink),
    ("hl55_s_best",   55,    None, best_shrink),
    ("blend40_sbest", None,  0.40, best_shrink),
    ("blend60_sbest", None,  0.60, best_shrink),
]

combo_pooled = {name: [] for name, _, _, _ in combo_configs}
combo_pn = {name: [] for name, _, _, _ in combo_configs}

for c in CUTOFFS:
    train = ps[ps["Season"] < c]
    test  = ps[ps["Season"] == c]
    if train.empty or test.empty:
        continue
    max_mid = train["MatchID"].max()
    career_n = train.groupby("Player").size()
    t = test.groupby("Player").agg(
        npxg_sum=("npxG", "sum"), npg_sum=("NPG", "sum"),
        mins_sum=("Minutes", "sum"),
    ).reset_index()
    t = t[t["mins_sum"] >= MIN_TEST_MINUTES]
    t = t[t["Player"].isin(career_n[career_n >= MIN_CAREER_MATCHES].index)]
    if t.empty:
        continue
    npxg90 = t["npxg_sum"] * 90 / t["mins_sum"]
    npg90  = t["npg_sum"]  * 90 / t["mins_sum"]

    for name, hl, bw, shrink in combo_configs:
        shrink_val = shrink if shrink is not None else 30
        if hl is not None:
            conv = compute_match_decay_conversion(train, max_mid, hl, shrink_val)
        elif bw is not None:
            conv = compute_season_blend_conversion(train, c, RECENT_WINDOW, bw, shrink_val)
        else:
            conv = compute_plain_conversion(train, shrink_val)
        pred = t["Player"].map(conv).fillna(1.0) * npxg90
        mae = np.mean(np.abs(pred - npg90))
        combo_pooled[name].append(mae)
        combo_pn[name].append(len(t))

print(f"\n{'Variant':>18} {'MAE':>8}  {'vs inc':>8}")
print("-" * 40)
inc_mae_c = np.average(combo_pooled["incumbent"], weights=combo_pn["incumbent"])
for name, _, _, _ in combo_configs:
    mae = np.average(combo_pooled[name], weights=combo_pn[name])
    delta = inc_mae_c - mae
    print(f"  {name:>16} {mae:>8.5f}  {delta:>+8.5f}")

# ============================================================
# VERDICT
# ============================================================
print()
print("=" * 70)
print("FINAL VERDICT")
print("=" * 70)

# Check all families for the best
best_family = None
best_val = inc_mae_c
best_name = "incumbent"

for name, _, _, _ in combo_configs:
    mae = np.average(combo_pooled[name], weights=combo_pn[name])
    if mae < best_val:
        best_val = mae
        best_name = name

# Also check Family C
for s in SHRINKAGE_VALUES:
    k = f"C_s{s}"
    mae = np.average(pooled_c[k], weights=pooled_cn[k])
    if mae < best_val:
        best_val = mae
        best_name = k

delta_final = inc_mae_c - best_val
print(f"Incumbent MAE:  {inc_mae_c:.5f}")
print(f"Best variant:   {best_name} (MAE = {best_val:.5f})")
print(f"Improvement:    {delta_final:+.5f} ({delta_final/inc_mae_c*100:+.2f}%)")

# Significance check: bootstrap the improvement
np.random.seed(42)
N_BOOT = 1000
boot_deltas = []

# Gather all per-player residuals for incumbent and best variant
# Re-run once to collect them
inc_residuals = []
best_residuals = []
for c in CUTOFFS:
    train = ps[ps["Season"] < c]
    test  = ps[ps["Season"] == c]
    if train.empty or test.empty:
        continue
    max_mid = train["MatchID"].max()
    career_n = train.groupby("Player").size()
    t = test.groupby("Player").agg(
        npxg_sum=("npxG", "sum"), npg_sum=("NPG", "sum"),
        mins_sum=("Minutes", "sum"),
    ).reset_index()
    t = t[t["mins_sum"] >= MIN_TEST_MINUTES]
    t = t[t["Player"].isin(career_n[career_n >= MIN_CAREER_MATCHES].index)]
    if t.empty:
        continue
    npxg90 = t["npxg_sum"] * 90 / t["mins_sum"]
    npg90  = t["npg_sum"]  * 90 / t["mins_sum"]
    actual = npg90.values

    # incumbent
    conv_inc = compute_plain_conversion(train, 30)
    pred_inc = t["Player"].map(conv_inc).fillna(1.0).values * npxg90.values
    inc_residuals.extend(np.abs(pred_inc - actual))

    # best variant
    shrink_val = best_shrink if best_name != "incumbent" else 30
    if "hl" in best_name:
        hl_val = int(best_name.split("_")[0].replace("hl", ""))
        conv_best = compute_match_decay_conversion(train, max_mid, hl_val, shrink_val)
    elif "blend" in best_name:
        bw_val = float(best_name.split("blend")[1].split("_")[0]) / 100
        conv_best = compute_season_blend_conversion(train, c, RECENT_WINDOW, bw_val, shrink_val)
    else:
        conv_best = compute_plain_conversion(train, shrink_val)
    pred_best = t["Player"].map(conv_best).fillna(1.0).values * npxg90.values
    best_residuals.extend(np.abs(pred_best - actual))

inc_res = np.array(inc_residuals)
best_res = np.array(best_residuals)
n = len(inc_res)
for _ in range(N_BOOT):
    idx = np.random.randint(0, n, size=n)
    boot_deltas.append(np.mean(inc_res[idx]) - np.mean(best_res[idx]))

ci_lo = np.percentile(boot_deltas, 2.5)
ci_hi = np.percentile(boot_deltas, 97.5)
print(f"95% bootstrap CI for improvement: [{ci_lo:+.5f}, {ci_hi:+.5f}]")

if ci_lo > 0:
    print("VERDICT: Improvement is statistically significant. Consider shipping.")
elif delta_final > 0.0005:
    print("VERDICT: Marginal improvement but not statistically significant.")
    print("         Too small to ship with confidence.")
else:
    print("VERDICT: No meaningful improvement. Do NOT ship form weighting.")
