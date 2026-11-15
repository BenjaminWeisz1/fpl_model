"""
Hyperparameter tuning for the team ratings model (alpha, beta,
money_weight) -- implements TEAM_RATINGS_TUNING.md against the OOT
harness from the previous step (evaluate_team_ratings_oot.py, CHANGELOG
§20). Does NOT modify update_team_ratings.py: this script only measures
and reports; shipping is a separate human decision (guardrail 4).

Design (fixed before the run):

- Full grid, centered on production: alpha 0.02..0.08 step 0.005 (13)
  x beta 0.50..1.00 step 0.05 (11) x money_weight 0.15..0.45 step 0.05
  (7) = 1,001 combinations. No reduced grid -- the fast engine
  (ratings_fast.py) makes the full grid a minutes-long job.
- Rolling-origin cutoffs 2023, 2024, 2025, 2026 (the harness's four
  splits): each combination trains the sequential process on Season < C
  only and is scored on Season == C only -- multiple windows so a
  winner must generalize across them, per the tuning doc's
  anti-overfitting requirement.
- Scoring (frozen mode, the harness's PRIMARY): pooled Poisson LL per
  team-match across ALL held-out matches of the four cutoffs, i.e.
  sample-size-weighted across cutoffs (2023-2025 contribute ~33% each;
  2026's 19 matches ~1.7% -- the doc's small-window concern argues
  against unweighted cutoff averaging). The unweighted mean of the four
  per-cutoff LLs is reported alongside, and a winner that flips between
  the two is flagged. Walk-forward pooled LL is recorded per combination
  as a production-realism cross-check.
- The production combination (alpha=0.045, beta=0.8, money_weight=0.3
  -- alpha is NOT a grid point) is evaluated through the identical code
  path as the guardrail-1 baseline every candidate is compared against.
- Verification guard (runs unconditionally at startup, aborts on
  failure): ratings_fast.verify_against_reference() compares the fast
  engine to the verbatim reference loop on production params across all
  four cutoff frames (walk-forward predictions + boundary snapshot
  values + frozen predictions end-to-end) and on the 8 grid corners
  over the full history.
- Guardrails 2-3: boundary-winner flag; distance-from-production report
  with an update-dynamics comparison (mean |rating step| per update).

Pre-registered prediction (2026-09-04, stated before the grid ran and
approved by the user): the optimum sits near production alpha
(0.04-0.05); money_weight is the parameter most likely to move
materially; any boundary winner is a range-extension flag, not a ship.
The report scores this prediction honestly after the run.

Outputs (data/current/team_ratings_oot/): tuning_grid.csv (all 1,002
rows incl. baseline), tuning_report.txt, tuning_summary.json.

Run locally:
    python -m research.team_ratings.tune_team_ratings
"""
import itertools
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from fpl_model.team_ratings.update_team_ratings import alpha, beta, home_advantage, money_weight
from research.team_ratings import ratings_fast
from research.team_ratings.evaluate_team_ratings_oot import OUT_DIR, cs_bin_table, load_inputs, summarize_predictions
from research.team_ratings.ratings_fast import fast_run, frozen_predict_fast

ALPHAS = [round(0.02 + 0.005 * i, 10) for i in range(13)]
BETAS = [round(0.50 + 0.05 * i, 10) for i in range(11)]
MWEIGHTS = [round(0.15 + 0.05 * i, 10) for i in range(7)]
CUTOFFS = (2023, 2024, 2025, 2026)
CORNER_PARAMS = list(itertools.product(
    (ALPHAS[0], ALPHAS[-1]), (BETAS[0], BETAS[-1]), (MWEIGHTS[0], MWEIGHTS[-1])))
NEAR_TIE_MARGIN = 0.001  # pooled LL/tm window for the flat-optimum check

_REPORT_LINES = []


def say(line=""):
    print(line)
    _REPORT_LINES.append(line)


def build_context(df):
    """Per-cutoff frames, train-only ave_goals, held-out masks/targets --
    computed once, reused by every grid combination."""
    ctx = {"frames": {}, "ave": {}, "masks": {}, "yh": {}, "ya": {}, "val": {}}
    for c in CUTOFFS:
        frame = df[df["Season"] <= c].reset_index(drop=True)
        train = df[df["Season"] < c]
        ctx["frames"][c] = frame
        ctx["ave"][c] = float(pd.concat([train["Home_NPG"], train["Away_NPG"]]).mean())
        ctx["masks"][c] = frame["Season"].values == c
        ctx["yh"][c] = frame["Home_NPG"].values[ctx["masks"][c]].astype(float)
        ctx["ya"][c] = frame["Away_NPG"].values[ctx["masks"][c]].astype(float)
        ctx["val"][c] = frame[ctx["masks"][c]]
    return ctx


def evaluate_combo(params, ctx, prior_map):
    """Score one (alpha, beta, money_weight) combination across all
    cutoffs. Returns (per-cutoff frozen summaries, pooled frozen summary,
    pooled frozen CS probs/events, pooled walk-forward summary)."""
    per = {}
    f_ph_all, f_pa_all, yh_all, ya_all = [], [], [], []
    w_ph_all, w_pa_all = [], []
    for c in CUTOFFS:
        frame = ctx["frames"][c]
        ph, pa, snaps = fast_run(frame, ctx["ave"][c], params, prior_map,
                                 snapshot_seasons={c})
        m = ctx["masks"][c]
        fph, fpa = frozen_predict_fast(ctx["val"][c], snaps[c], prior_map,
                                       ctx["ave"][c], params)
        s, _, _ = summarize_predictions(fph, fpa, ctx["yh"][c], ctx["ya"][c])
        per[c] = s
        f_ph_all.append(fph)
        f_pa_all.append(fpa)
        yh_all.append(ctx["yh"][c])
        ya_all.append(ctx["ya"][c])
        w_ph_all.append(ph[m])
        w_pa_all.append(pa[m])
    pooled_f, probs, events = summarize_predictions(
        np.concatenate(f_ph_all), np.concatenate(f_pa_all),
        np.concatenate(yh_all), np.concatenate(ya_all))
    pooled_w, _, _ = summarize_predictions(
        np.concatenate(w_ph_all), np.concatenate(w_pa_all),
        np.concatenate(yh_all), np.concatenate(ya_all))
    return per, pooled_f, probs, events, pooled_w


def combo_row(params, per, pooled_f, pooled_w, is_baseline=False):
    return {
        "alpha": params["alpha"],
        "beta": params["beta"],
        "money_weight": params["money_weight"],
        "is_baseline": is_baseline,
        **{f"ll_frozen_{c}": per[c]["poisson_ll_per_tm"] for c in CUTOFFS},
        "ll_frozen_pooled": pooled_f["poisson_ll_per_tm"],
        "ll_frozen_meancut": float(np.mean([per[c]["poisson_ll_per_tm"] for c in CUTOFFS])),
        "mae_frozen_pooled": pooled_f["mae"],
        "cs_ll_frozen_pooled": pooled_f["cs_ll"],
        "ll_wf_pooled": pooled_w["poisson_ll_per_tm"],
    }


def fmt_params(p):
    return f"alpha={p['alpha']}, beta={p['beta']}, money_weight={p['money_weight']}"


def rating_dynamics(params, df, ave_full, prior_map):
    _, _, _, dyn = fast_run(df, ave_full, params, prior_map,
                            collect_dynamics=True)
    return dyn


def main():
    t_start = time.perf_counter()
    df, prior_map, data_meta = load_inputs()
    prod_params = {"alpha": alpha, "beta": beta,
                   "home_advantage": home_advantage, "money_weight": money_weight}
    ave_full = float(pd.concat([df["Home_NPG"], df["Away_NPG"]]).mean())
    ctx = build_context(df)

    say("=" * 60)
    say("TEAM RATINGS HYPERPARAMETER TUNING (alpha, beta, money_weight)")
    say("=" * 60)
    say(f"  data: match_results.csv {data_meta['match_results']['rows']} rows "
        f"(md5 {data_meta['match_results']['md5'][:10]}..., "
        f"{data_meta['match_results']['mtime']}), "
        f"money_priors {data_meta['money_priors']['rows']} rows")
    say(f"  grid: alpha {ALPHAS[0]}-{ALPHAS[-1]} step 0.005 ({len(ALPHAS)}) x "
        f"beta {BETAS[0]}-{BETAS[-1]} step 0.05 ({len(BETAS)}) x "
        f"money_weight {MWEIGHTS[0]}-{MWEIGHTS[-1]} step 0.05 ({len(MWEIGHTS)}) "
        f"= {len(ALPHAS) * len(BETAS) * len(MWEIGHTS)} combinations")
    say(f"  cutoffs: {list(CUTOFFS)} (rolling origin: train Season < C, score Season == C)")
    say(f"  production (baseline): {fmt_params(prod_params)}, "
        f"home_advantage={home_advantage:.9f} (fixed, not tuned)")
    say("  primary metric: pooled FROZEN Poisson LL per team-match across all "
        "cutoffs' held-out matches (sample-size-weighted);")
    say("  unweighted cutoff mean + walk-forward pooled reported alongside")
    say()
    say("PRE-REGISTERED PREDICTION (2026-09-04, before the grid ran):")
    say("  1. optimum near production alpha (0.04-0.05)")
    say("  2. money_weight the parameter most likely to move materially")
    say("  3. any boundary winner is a range-extension flag, not a ship")

    # ── Verification guard (unconditional; abort on failure) ──
    say()
    say("-" * 60)
    say("VERIFICATION GUARD: fast engine vs verbatim reference loop")
    say("-" * 60)
    checks = []
    for c in CUTOFFS:
        checks.append((f"production params, cutoff {c} frame "
                       f"({len(ctx['frames'][c])} rows; walk-forward + snapshot + frozen)",
                       ctx["frames"][c], ctx["ave"][c], prod_params, c))
    for a_, b_, m_ in CORNER_PARAMS:
        p = {"alpha": a_, "beta": b_, "home_advantage": home_advantage,
             "money_weight": m_}
        checks.append((f"grid corner alpha={a_}, beta={b_}, mw={m_} "
                       f"(full history, {len(df)} rows; walk-forward)",
                       df, ave_full, p, None))
    t_guard = time.perf_counter()
    guard_ok = ratings_fast.verify_against_reference(checks, prior_map, log=say)
    guard_time = time.perf_counter() - t_guard
    say(f"  guard wall time: {guard_time:.0f}s")
    if not guard_ok:
        say()
        say("  GUARD FAILED -- aborting. Grid numbers would be untrustworthy;")
        say("  fix the fast engine / reference divergence before tuning.")
        return 1

    # ── Guardrail 1: production baseline first ──
    say()
    say("-" * 60)
    say("BASELINE: PRODUCTION VALUES ON THIS HARNESS (guardrail 1)")
    say("-" * 60)
    base_per, base_pooled_f, base_probs, base_events, base_pooled_w = \
        evaluate_combo(prod_params, ctx, prior_map)
    base_row = combo_row(prod_params, base_per, base_pooled_f, base_pooled_w,
                         is_baseline=True)
    say(f"  {fmt_params(prod_params)}")
    say(f"  {'cutoff':<8s}{'LL/tm':>9s}{'MAE':>9s}{'CS LL':>9s}   (frozen -- primary)")
    for c in CUTOFFS:
        s = base_per[c]
        say(f"  {c:<8d}{s['poisson_ll_per_tm']:>9.4f}{s['mae']:>9.4f}{s['cs_ll']:>9.4f}")
    say(f"  {'pooled':<8s}{base_pooled_f['poisson_ll_per_tm']:>9.4f}"
        f"{base_pooled_f['mae']:>9.4f}{base_pooled_f['cs_ll']:>9.4f}   "
        f"({base_pooled_f['n_matches']} held-out matches)")
    say(f"  walk-forward pooled: LL/tm {base_pooled_w['poisson_ll_per_tm']:.4f} | "
        f"MAE {base_pooled_w['mae']:.4f} | CS LL {base_pooled_w['cs_ll']:.4f}")

    # Cross-check against the stored harness baseline (same code path,
    # verbatim loop there vs guard-verified fast engine here).
    baseline_json = OUT_DIR / "baseline.json"
    if baseline_json.exists():
        with open(baseline_json, encoding="utf-8") as f:
            stored = json.load(f)
        same_md5 = (stored["data"]["match_results"]["md5"]
                    == data_meta["match_results"]["md5"])
        if same_md5:
            ref = stored["pooled"]["frozen"]["poisson_ll_per_tm"]
            d = abs(ref - base_pooled_f["poisson_ll_per_tm"])
            say(f"  cross-check vs stored harness baseline.json: "
                f"{ref:.10f} vs {base_pooled_f['poisson_ll_per_tm']:.10f} "
                f"(|diff| = {d:.2e}) -> {'MATCH' if d < 1e-9 else 'MISMATCH'}")
        else:
            say("  cross-check vs stored baseline.json: SKIPPED "
                "(match_results.csv changed since that run -- re-run the "
                "harness baseline on this snapshot)")

    # ── Grid search ──
    say()
    say("-" * 60)
    say(f"GRID SEARCH: {len(ALPHAS) * len(BETAS) * len(MWEIGHTS)} combinations x "
        f"{len(CUTOFFS)} cutoffs")
    say("-" * 60)
    rows = [base_row]
    t_grid = time.perf_counter()
    n_done = 0
    for a_ in ALPHAS:
        for b_ in BETAS:
            for m_ in MWEIGHTS:
                p = {"alpha": a_, "beta": b_, "home_advantage": home_advantage,
                     "money_weight": m_}
                per, pooled_f, _, _, pooled_w = evaluate_combo(p, ctx, prior_map)
                rows.append(combo_row(p, per, pooled_f, pooled_w))
                n_done += 1
                if n_done % 200 == 0:
                    print(f"    ... {n_done}/1001 combinations "
                          f"({time.perf_counter() - t_grid:.0f}s)")
    grid_time = time.perf_counter() - t_grid
    say(f"  evaluated {n_done} combinations in {grid_time:.0f}s "
        f"(fast engine; the verbatim loop would need ~{n_done * 59 / 3600:.0f} hours)")

    grid = pd.DataFrame(rows)
    grid = grid.sort_values("ll_frozen_pooled").reset_index(drop=True)
    grid.insert(0, "rank", grid.index + 1)
    base_rank = int(grid.loc[grid["is_baseline"], "rank"].iloc[0])
    winner = grid.loc[~grid["is_baseline"]].iloc[0]
    win_params = {"alpha": float(winner["alpha"]), "beta": float(winner["beta"]),
                  "home_advantage": home_advantage,
                  "money_weight": float(winner["money_weight"])}

    # ── Results: top table ──
    say()
    say("-" * 60)
    say("TOP 10 BY POOLED FROZEN POISSON LL (primary)")
    say("-" * 60)
    say(f"  {'rank':>4s}  {'alpha':>6s}{'beta':>6s}{'mw':>6s}   "
        f"{'LL/tm':>8s}{'mean-cut':>9s}{'MAE':>8s}{'CS LL':>8s}{'LL wf':>8s}")
    for _, r in grid.head(10).iterrows():
        tag = "  [production]" if r["is_baseline"] else ""
        say(f"  {int(r['rank']):>4d}  {r['alpha']:>6.3f}{r['beta']:>6.2f}"
            f"{r['money_weight']:>6.2f}   {r['ll_frozen_pooled']:>8.4f}"
            f"{r['ll_frozen_meancut']:>9.4f}{r['mae_frozen_pooled']:>8.4f}"
            f"{r['cs_ll_frozen_pooled']:>8.4f}{r['ll_wf_pooled']:>8.4f}{tag}")
    say(f"  (production baseline ranks #{base_rank} of {len(grid)} on the "
        f"primary metric)")

    # ── Winner analysis ──
    say()
    say("-" * 60)
    say("WINNER ANALYSIS")
    say("-" * 60)
    say(f"  best grid combination: {fmt_params(win_params)}")
    improvement = base_row["ll_frozen_pooled"] - float(winner["ll_frozen_pooled"])
    say(f"  pooled frozen LL/tm:  {base_row['ll_frozen_pooled']:.4f} (production) "
        f"-> {winner['ll_frozen_pooled']:.4f} (winner) = "
        f"{improvement:+.4f} ({'improvement' if improvement > 0 else 'NO improvement'})")
    say(f"  pooled frozen MAE:    {base_row['mae_frozen_pooled']:.4f} -> "
        f"{winner['mae_frozen_pooled']:.4f} "
        f"({winner['mae_frozen_pooled'] - base_row['mae_frozen_pooled']:+.4f})")
    say(f"  pooled frozen CS LL:  {base_row['cs_ll_frozen_pooled']:.4f} -> "
        f"{winner['cs_ll_frozen_pooled']:.4f} "
        f"({winner['cs_ll_frozen_pooled'] - base_row['cs_ll_frozen_pooled']:+.4f})")
    say(f"  pooled walk-fwd LL:   {base_row['ll_wf_pooled']:.4f} -> "
        f"{winner['ll_wf_pooled']:.4f} "
        f"({winner['ll_wf_pooled'] - base_row['ll_wf_pooled']:+.4f})")

    say()
    say("  per-cutoff frozen LL/tm vs baseline (a winner that only wins one")
    say("  season is a red flag, per the tuning doc):")
    improved = 0
    for c in CUTOFFS:
        d = float(winner[f"ll_frozen_{c}"]) - base_row[f"ll_frozen_{c}"]
        better = d < 0
        improved += int(better)
        say(f"    {c}: {base_row[f'll_frozen_{c}']:.4f} -> "
            f"{winner[f'll_frozen_{c}']:.4f} ({d:+.4f}) "
            f"{'better' if better else 'WORSE'}")
    say(f"  cutoffs improved: {improved}/{len(CUTOFFS)}")

    # Unweighted-mean agreement
    mean_cut_order = grid.sort_values("ll_frozen_meancut").reset_index(drop=True)
    mc_rank = int(mean_cut_order[
        (mean_cut_order["alpha"] == win_params["alpha"])
        & (mean_cut_order["beta"] == win_params["beta"])
        & (mean_cut_order["money_weight"] == win_params["money_weight"])
    ].index[0]) + 1
    mc_best = mean_cut_order.loc[~mean_cut_order["is_baseline"]].iloc[0]
    agree = (mc_best["alpha"] == win_params["alpha"]
             and mc_best["beta"] == win_params["beta"]
             and mc_best["money_weight"] == win_params["money_weight"])
    say()
    say(f"  unweighted cutoff-mean metric: winner ranks #{mc_rank} under it; "
        f"its own best is alpha={mc_best['alpha']:.3f}, beta={mc_best['beta']:.2f}, "
        f"mw={mc_best['money_weight']:.2f} "
        f"(LL {mc_best['ll_frozen_meancut']:.4f}) -> "
        f"{'same winner (robust to averaging choice)' if agree else 'DIFFERENT WINNER -- flag: averaging choice matters here'}")

    # Near-tie / flat-optimum check
    near = grid[(grid["ll_frozen_pooled"] <= winner["ll_frozen_pooled"] + NEAR_TIE_MARGIN)
                & (~grid["is_baseline"])]
    say()
    say(f"  near-ties within {NEAR_TIE_MARGIN} LL/tm of the winner: {len(near)} "
        f"combinations")
    if len(near) > 1:
        say(f"    alpha range {near['alpha'].min():.3f}-{near['alpha'].max():.3f}, "
            f"beta {near['beta'].min():.2f}-{near['beta'].max():.2f}, "
            f"mw {near['money_weight'].min():.2f}-{near['money_weight'].max():.2f} "
            f"(flat optimum -- the exact pick within this set is not decisive)")

    # Guardrail 2: boundary flags
    boundary = []
    if win_params["alpha"] in (ALPHAS[0], ALPHAS[-1]):
        boundary.append(f"alpha={win_params['alpha']} (range {ALPHAS[0]}-{ALPHAS[-1]})")
    if win_params["beta"] in (BETAS[0], BETAS[-1]):
        boundary.append(f"beta={win_params['beta']} (range {BETAS[0]}-{BETAS[-1]})")
    if win_params["money_weight"] in (MWEIGHTS[0], MWEIGHTS[-1]):
        boundary.append(f"money_weight={win_params['money_weight']} "
                        f"(range {MWEIGHTS[0]}-{MWEIGHTS[-1]})")
    say()
    say("  GUARDRAIL 2 (boundary check): "
        + ("NO boundary values in the winner" if not boundary
           else "WINNER AT GRID BOUNDARY: " + "; ".join(boundary)
                + " -- true optimum may lie outside the tested range; extend "
                  "and re-run before concluding anything"))

    # Guardrail 3: distance from production + update dynamics
    say()
    say("  GUARDRAIL 3 (distance from production + update dynamics):")
    say(f"    alpha:        {prod_params['alpha']} -> {win_params['alpha']} "
        f"({win_params['alpha'] - prod_params['alpha']:+.3f}; "
        f"adaptation speed x{win_params['alpha'] / prod_params['alpha']:.2f})")
    say(f"    beta:         {prod_params['beta']} -> {win_params['beta']} "
        f"({win_params['beta'] - prod_params['beta']:+.2f}; "
        f"xG-vs-goals weight in the update numerator)")
    say(f"    money_weight: {prod_params['money_weight']} -> "
        f"{win_params['money_weight']} "
        f"({win_params['money_weight'] - prod_params['money_weight']:+.2f}; "
        f"prior pull at each season boundary)")
    base_dyn = rating_dynamics(prod_params, df, ave_full, prior_map)
    win_dyn = rating_dynamics(win_params, df, ave_full, prior_map)
    say(f"    mean |rating step| per update (full history): attack "
        f"{base_dyn['mean_step_atk']:.5f} -> {win_dyn['mean_step_atk']:.5f} "
        f"(x{win_dyn['mean_step_atk'] / base_dyn['mean_step_atk']:.2f}), defense "
        f"{base_dyn['mean_step_def']:.5f} -> {win_dyn['mean_step_def']:.5f} "
        f"(x{win_dyn['mean_step_def'] / base_dyn['mean_step_def']:.2f})")

    # Winner vs baseline CS reliability
    win_per, win_pooled_f, win_probs, win_events, _ = \
        evaluate_combo(win_params, ctx, prior_map)
    say()
    say("  clean-sheet reliability, pooled held-out (production -> winner):")
    say(f"    {'P(CS) bin':<18s}{'n':>7s}{'pred':>9s}{'obs':>9s}   "
        f"{'pred':>9s}{'obs':>9s}")
    base_bins = cs_bin_table(base_probs, base_events)
    win_bins = cs_bin_table(win_probs, win_events)
    for rb, rw in zip(base_bins, win_bins):
        say(f"    {rb['bin']:<18s}{rb['n']:>7d}{rb['pred']:>9.3f}{rb['obs']:>9.3f}"
            f"   {rw['pred']:>9.3f}{rw['obs']:>9.3f}")

    # Pre-registration scorecard
    say()
    say("PRE-REGISTRATION SCORECARD (honest):")
    pred1 = 0.04 <= win_params["alpha"] <= 0.051
    say(f"  1. 'optimum near production alpha (0.04-0.05)': winner alpha="
        f"{win_params['alpha']} -> {'HELD' if pred1 else 'FAILED'}")
    d_alpha = abs(win_params["alpha"] - prod_params["alpha"])
    d_beta = abs(win_params["beta"] - prod_params["beta"])
    d_mw = abs(win_params["money_weight"] - prod_params["money_weight"])
    biggest = max((d_alpha, "alpha"), (d_beta, "beta"), (d_mw, "money_weight"))[1]
    pred2 = biggest == "money_weight"
    say(f"  2. 'money_weight most likely to move materially': largest shift is "
        f"{biggest} (dAlpha={d_alpha:.3f}, dBeta={d_beta:.2f}, "
        f"dMw={d_mw:.2f}) -> {'HELD' if pred2 else 'FAILED'}")
    pred3 = not boundary
    say(f"  3. 'boundary winner is a flag, not a ship': "
        f"{'no boundary values -- rule not triggered' if pred3 else 'BOUNDARY WINNER -- rule triggered'}")

    # Verdict (guardrail 4: never ships automatically)
    say()
    say("-" * 60)
    say("VERDICT (guardrail 4: nothing is shipped automatically)")
    say("-" * 60)
    if improvement <= 0:
        say("  NO grid combination beats the production values on the primary")
        say("  metric. Recommendation: keep alpha=0.045, beta=0.8,")
        say("  money_weight=0.3 unchanged. (Consistent with the §20 alpha=0.05")
        say("  probe: the incumbent sits at/near a local optimum.)")
        verdict = "no_improvement"
    elif boundary:
        say("  Best combination sits at a grid boundary -- per guardrail 2 this")
        say("  is a range-extension signal, NOT a shipping candidate. Extend")
        say("  the grid on the flagged axis/axes and re-run before concluding.")
        verdict = "boundary_flag"
    else:
        say("  Candidate improvement found (interior of the grid). Review the")
        say("  per-cutoff consistency, CS calibration, walk-forward agreement,")
        say("  dynamics and near-tie structure above, then decide. Shipping =")
        say("  editing the three constants in update_team_ratings.py and")
        say("  re-running the harness baseline (--label baseline) for the new")
        say("  reference point.")
        verdict = "candidate_interior"
    say()
    say("  update_team_ratings.py is UNCHANGED by this script.")

    # ── Save artifacts ──
    total_time = time.perf_counter() - t_start
    say()
    say(f"  total wall time: {total_time:.0f}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = OUT_DIR / "tuning_grid.csv"
    grid.to_csv(grid_path, index=False)
    report_path = OUT_DIR / "tuning_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_REPORT_LINES) + "\n")

    top10 = grid.head(10).to_dict("records")
    summary = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "data": data_meta,
        "grid": {"alphas": ALPHAS, "betas": BETAS, "mweights": MWEIGHTS,
                 "n_combinations": n_done},
        "cutoffs": list(CUTOFFS),
        "primary_metric": "pooled frozen Poisson LL per team-match "
                          "(sample-size-weighted across cutoffs)",
        "baseline": {k: base_row[k] for k in base_row},
        "baseline_rank": base_rank,
        "winner": {k: (float(winner[k]) if isinstance(winner[k], (int, float, np.floating))
                       else winner[k]) for k in winner.index},
        "improvement_ll_per_tm": float(improvement),
        "cutoffs_improved": improved,
        "boundary_flags": boundary,
        "mean_cut_metric_agrees": bool(agree),
        "near_ties_within_0p001": int(len(near)),
        "dynamics": {"production": base_dyn, "winner": win_dyn},
        "preregistration": {"alpha_near_production": bool(pred1),
                            "money_weight_biggest_shift": bool(pred2),
                            "no_boundary_winner": bool(pred3)},
        "verdict": verdict,
        "runtimes_s": {"guard": round(guard_time, 1), "grid": round(grid_time, 1),
                       "total": round(total_time, 1)},
        "top10": top10,
    }
    summary_path = OUT_DIR / "tuning_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    say(f"  Saved {grid_path}")
    say(f"  Saved {report_path}")
    say(f"  Saved {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
