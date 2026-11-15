"""
Season-boundary rebalancing experiment for the team ratings model --
implements TEAM_RATINGS_REBALANCING.md (experiment 3 of the three
planned in TEAM_RATINGS_OOT.md) against the §20 OOT harness, using the
§21 fast engine (ratings_fast.py) EXTENDED with the rebalancing step,
verified against a NEW slow verbatim reference
(evaluate_team_ratings_oot.run_sequential_rebalanced) with the same
bit-equality discipline as the tuning run. update_team_ratings.py is
NOT modified: this script only measures and reports; shipping is a
separate human decision (guardrail 4).

Mechanism under test (measured earlier in this project, notably the
2024->2025 transition): relegated teams exit seasons with
genuinely-earned low ratings while promoted teams enter at
100%-money-implied ratings that are structurally more optimistic, so
the league-wide Attack average runs above the Defense average and the
asymmetry compounds season over season. The fix under test rescales at
the start of EVERY season (2017 included -- there the table is pure
money priors, so the step rescales the prior scale itself), after the
prune + money-prior blend and before the season's first match, over
ALL 20 season teams (newcomers at pure priors).

Design decision flagged per the doc: M = (avg_Attack + avg_Defense)/2
(midpoint) is a CHOICE, not an obvious default -- anchoring one side
instead is equally defensible. The defense-anchored variant (hold the
Defense average fixed, rescale Attack only) runs here as a sensitivity
check and is reported alongside, not as the primary.

Sequential, path-dependent model -> full-history re-run through every
boundary, never a partial or forward-only application.

Pre-registered prediction (TEAM_RATINGS_REBALANCING.md, written before
this implementation; the doc IS the pre-registration): if the
compounding-drift mechanism is real, the improvement should GROW with
cutoff recency -- 2026's held-out performance benefits more than
2023's. Operationalized before the run:
  * improvement(c) = baseline_LL(c) - rebalanced_LL(c), frozen mode
    (the primary metric), positive = rebalancing better;
  * HELD     if improvement(2026) > improvement(2023) AND the four
             improvements are non-decreasing across 2023->2026;
  * PARTIAL  if improvement(2026) > improvement(2023) but the
             sequence is not monotone (pattern present but noisy --
             2026 has only 19 held-out matches);
  * FAILED   if improvement(2026) <= improvement(2023), or the
             pattern is flat/reversed, even if the aggregate improves.
  The walk-forward pattern is reported alongside for interpretation.

Guardrails (same discipline as §21, plus the tuning-specific one):
  1. Baseline (production, no rebalancing) scored first on this exact
     harness -- never assumed neutral.
  2. Small aggregate improvements are checked against the §21 noise
     band (292 of 1001 tuning combos sat within 0.001 LL/tm of the
     winner -- deltas inside that band are not distinguishable from
     noise on this harness) and must survive walk-forward mode.
  3. Per-cutoff results reported individually (required for the
     recency prediction, not optional).
  4. Nothing ships automatically; update_team_ratings.py UNCHANGED.

Out of scope (per the doc): hyperparameter re-tuning (production
values stay), team-specific home/away (separate later experiment).
Note only: rebalancing changes rating levels entering update
denominators, so the §21 optimum could shift under rebalancing -- not
tested here.

Outputs (data/current/team_ratings_oot/): rebalancing_report.txt,
rebalancing_summary.json.

Run locally:
    python rebalance_team_ratings.py
"""
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
from research.team_ratings.tune_team_ratings import CUTOFFS, build_context, fmt_params

# §21 noise-band reference points, fixed for the guardrail-2 assessment.
NOISE_BAND_LL = 0.001   # 292/1001 tuning combos within this of the winner
NOISE_BAND_N = 292

_REPORT_LINES = []


def say(line=""):
    print(line)
    _REPORT_LINES.append(line)


def evaluate(params, ctx, prior_map, rebalance=None):
    """Score one configuration (params + rebalance mode) across all
    cutoffs with the fast engine. Mirrors tune_team_ratings's scoring
    path, plus per-cutoff walk-forward summaries and per-boundary gap
    records for the mechanism check."""
    per_f, per_w, gaps = {}, {}, {}
    f_ph_all, f_pa_all, yh_all, ya_all, w_ph_all, w_pa_all = [], [], [], [], [], []
    for c in CUTOFFS:
        frame = ctx["frames"][c]
        ph, pa, snaps, _, gaps_c = fast_run(
            frame, ctx["ave"][c], params, prior_map,
            snapshot_seasons={c}, rebalance=rebalance,
            collect_boundary_gaps=True)
        m = ctx["masks"][c]
        fph, fpa = frozen_predict_fast(ctx["val"][c], snaps[c], prior_map,
                                       ctx["ave"][c], params)
        s_f, _, _ = summarize_predictions(fph, fpa, ctx["yh"][c], ctx["ya"][c])
        s_w, _, _ = summarize_predictions(ph[m], pa[m], ctx["yh"][c], ctx["ya"][c])
        per_f[c] = s_f
        per_w[c] = s_w
        gaps[c] = gaps_c
        f_ph_all.append(fph); f_pa_all.append(fpa)
        yh_all.append(ctx["yh"][c]); ya_all.append(ctx["ya"][c])
        w_ph_all.append(ph[m]); w_pa_all.append(pa[m])
    pooled_f, probs, events = summarize_predictions(
        np.concatenate(f_ph_all), np.concatenate(f_pa_all),
        np.concatenate(yh_all), np.concatenate(ya_all))
    pooled_w, _, _ = summarize_predictions(
        np.concatenate(w_ph_all), np.concatenate(w_pa_all),
        np.concatenate(yh_all), np.concatenate(ya_all))
    return {"frozen": per_f, "wf": per_w, "pooled_frozen": pooled_f,
            "pooled_wf": pooled_w, "cs_probs": probs, "cs_events": events,
            "gaps": gaps}


def ll(res, mode, c=None):
    if c is None:
        return res[f"pooled_{mode}"]["poisson_ll_per_tm"]
    return res[mode][c]["poisson_ll_per_tm"]


def main():
    t_start = time.perf_counter()
    df, prior_map, data_meta = load_inputs()
    prod = {"alpha": alpha, "beta": beta,
            "home_advantage": home_advantage, "money_weight": money_weight}
    ave_full = float(pd.concat([df["Home_NPG"], df["Away_NPG"]]).mean())
    ctx = build_context(df)

    say("=" * 60)
    say("TEAM RATINGS SEASON-BOUNDARY REBALANCING EXPERIMENT")
    say("=" * 60)
    say(f"  data: match_results.csv {data_meta['match_results']['rows']} rows "
        f"(md5 {data_meta['match_results']['md5'][:10]}..., "
        f"{data_meta['match_results']['mtime']}), "
        f"money_priors {data_meta['money_priors']['rows']} rows")
    say(f"  cutoffs: {list(CUTOFFS)} (rolling origin, same as §20/§21)")
    say(f"  params: {fmt_params(prod)}, home_advantage={home_advantage:.9f} "
        f"(production, fixed -- no re-tuning in this pass)")
    say("  rebalancing: at the start of EVERY season (2017-2026), after prune +")
    say("    money blend, before the season's first match, over ALL season teams")
    say("    (newcomers at pure priors); anchor = MIDPOINT M=(avg_A+avg_D)/2")
    say("    -- A DESIGN DECISION, not an obvious default; defense-anchored")
    say("    sensitivity variant reported below.")
    say("  metrics: pooled frozen Poisson LL/tm primary; walk-forward, MAE,")
    say("    clean-sheet LL alongside (same as §21).")
    say()
    say("PRE-REGISTERED PREDICTION (TEAM_RATINGS_REBALANCING.md, written before")
    say("this implementation): if the compounding-drift mechanism is real, the")
    say("improvement GROWS with cutoff recency (2026 benefits more than 2023).")
    say("Flat or reversed pattern = mechanism not operating as theorized, even")
    say("if the aggregate improves. Scoring rule fixed above in this script's")
    say("docstring before the run.")

    # ── Verification guard: fast engine (both modes) vs references ──
    say()
    say("-" * 60)
    say("VERIFICATION GUARD: fast engine vs verbatim references")
    say("-" * 60)
    checks = [(
        "default path unchanged: production params, cutoff 2026 frame "
        f"({len(ctx['frames'][2026])} rows; walk-forward + snapshot + frozen)",
        ctx["frames"][2026], ctx["ave"][2026], prod, 2026, None)]
    for anchor in ("midpoint", "defense"):
        for c in CUTOFFS:
            checks.append((
                f"rebalanced ({anchor}), cutoff {c} frame "
                f"({len(ctx['frames'][c])} rows; walk-forward + snapshot + frozen)",
                ctx["frames"][c], ctx["ave"][c], prod, c, anchor))
        checks.append((
            f"rebalanced ({anchor}), full history ({len(df)} rows; walk-forward)",
            df, ave_full, prod, None, anchor))
    t_guard = time.perf_counter()
    guard_ok = ratings_fast.verify_against_reference(checks, prior_map, log=say)
    guard_time = time.perf_counter() - t_guard
    say(f"  guard wall time: {guard_time:.0f}s")
    if not guard_ok:
        say()
        say("  GUARD FAILED -- aborting. Numbers would be untrustworthy; fix")
        say("  the fast engine / reference divergence before drawing conclusions.")
        return 1

    # ── Guardrail 1: baseline first ──
    say()
    say("-" * 60)
    say("GUARDRAIL 1: BASELINE (production, no rebalancing) ON THIS HARNESS")
    say("-" * 60)
    base = evaluate(prod, ctx, prior_map, rebalance=None)
    say(f"  {fmt_params(prod)}, rebalancing OFF")
    say(f"  {'cutoff':<8s}{'LL/tm':>9s}{'MAE':>9s}{'CS LL':>9s}   (frozen -- primary)")
    for c in CUTOFFS:
        s = base["frozen"][c]
        say(f"  {c:<8d}{s['poisson_ll_per_tm']:>9.4f}{s['mae']:>9.4f}{s['cs_ll']:>9.4f}")
    pf = base["pooled_frozen"]
    pw = base["pooled_wf"]
    say(f"  {'pooled':<8s}{pf['poisson_ll_per_tm']:>9.4f}{pf['mae']:>9.4f}"
        f"{pf['cs_ll']:>9.4f}   ({pf['n_matches']} held-out matches)")
    say(f"  walk-forward pooled: LL/tm {pw['poisson_ll_per_tm']:.4f} | "
        f"MAE {pw['mae']:.4f} | CS LL {pw['cs_ll']:.4f}")

    baseline_json = OUT_DIR / "baseline.json"
    if baseline_json.exists():
        with open(baseline_json, encoding="utf-8") as f:
            stored = json.load(f)
        same_md5 = (stored["data"]["match_results"]["md5"]
                    == data_meta["match_results"]["md5"])
        if same_md5:
            ref = stored["pooled"]["frozen"]["poisson_ll_per_tm"]
            d = abs(ref - pf["poisson_ll_per_tm"])
            say(f"  cross-check vs stored harness baseline.json: "
                f"{ref:.10f} vs {pf['poisson_ll_per_tm']:.10f} "
                f"(|diff| = {d:.2e}) -> {'MATCH' if d < 1e-9 else 'MISMATCH'}")
        else:
            say("  cross-check vs stored baseline.json: SKIPPED "
                "(match_results.csv changed since that run)")

    # ── Mechanism check: the gap rebalancing corrects ──
    mid = evaluate(prod, ctx, prior_map, rebalance="midpoint")
    dfa = evaluate(prod, ctx, prior_map, rebalance="defense")

    say()
    say("-" * 60)
    say("MECHANISM CHECK: the Attack/Defense league-average gap at every")
    say("season boundary (cutoff-2026 frame, the longest history)")
    say("-" * 60)
    say("  baseline gap = what the uncorrected process carries into each season")
    say("  (compounds if the drift mechanism is real); rebal pre-gap = the gap")
    say("  still present at that boundary AFTER all prior seasons' corrections;")
    say("  k_a/k_d = the rescale factors applied (midpoint anchor).")
    say(f"  {'season':<8s}{'base gap':>10s}{'rebal gap':>11s}{'k_a':>8s}{'k_d':>8s}")
    base_gaps = {g["season"]: g for g in base["gaps"][2026]}
    for g in mid["gaps"][2026]:
        s = g["season"]
        bg = base_gaps[s]["gap"]
        say(f"  {s:<8d}{bg:>10.4f}{g['gap']:>11.4f}"
            f"{g['k_attack']:>8.4f}{g['k_defense']:>8.4f}")
    b_last = base_gaps[2026]["gap"]
    say(f"  baseline gap trajectory: {base_gaps[2017]['gap']:+.4f} (2017) -> "
        f"{b_last:+.4f} (2026) -- "
        f"{'grows with seasons (compounding visible)' if b_last > base_gaps[2017]['gap'] else 'does NOT grow monotonically'}")
    r_first = mid["gaps"][2026][0]["gap"]
    say(f"  rebalanced pre-gap trajectory: {r_first:+.4f} (2017) -> "
        f"{mid['gaps'][2026][-1]['gap']:+.4f} (2026) -- "
        f"{'stays controlled vs baseline' if abs(mid['gaps'][2026][-1]['gap']) < abs(b_last) else 'NOT controlled relative to baseline'}")

    say()
    say("  gap entering each cutoff's held-out season (the boundary directly")
    say("  before the matches being scored; baseline = uncorrected, rebal =")
    say("  pre-rebalance gap of the rebalanced process, corrected to ~0 after):")
    for c in CUTOFFS:
        bg = base["gaps"][c][-1]
        rg = mid["gaps"][c][-1]
        say(f"    cutoff {c}: baseline {bg['gap']:+.4f} -> rebalanced pre-gap "
            f"{rg['gap']:+.4f} (k_a {rg['k_attack']:.4f}, k_d {rg['k_defense']:.4f})")

    # ── Primary comparison: rebalanced (midpoint) vs baseline ──
    say()
    say("-" * 60)
    say("REBALANCING (midpoint anchor) vs BASELINE")
    say("-" * 60)
    for mode, label in (("frozen", "frozen -- PRIMARY"),
                        ("wf", "walk-forward -- secondary")):
        b_p = base[f"pooled_{mode}"]
        r_p = mid[f"pooled_{mode}"]
        say(f"  [{label}]")
        say(f"    pooled Poisson LL/tm: {b_p['poisson_ll_per_tm']:.4f} -> "
            f"{r_p['poisson_ll_per_tm']:.4f} "
            f"({r_p['poisson_ll_per_tm'] - b_p['poisson_ll_per_tm']:+.4f}) | "
            f"MAE {b_p['mae']:.4f} -> {r_p['mae']:.4f} "
            f"({r_p['mae'] - b_p['mae']:+.4f}) | "
            f"CS LL {b_p['cs_ll']:.4f} -> {r_p['cs_ll']:.4f} "
            f"({r_p['cs_ll'] - b_p['cs_ll']:+.4f})")
        parts, improved = [], 0
        for c in CUTOFFS:
            d = ll(mid, mode, c) - ll(base, mode, c)
            parts.append(f"{c} {d:+.4f}")
            improved += int(d < 0)
        say(f"    per-cutoff LL deltas (negative = rebalancing better): "
            f"{' | '.join(parts)}")
        say(f"    cutoffs improved: {improved}/{len(CUTOFFS)}")

    # ── Pre-registration scorecard ──
    improvements = {c: ll(base, "frozen", c) - ll(mid, "frozen", c)
                    for c in CUTOFFS}
    seq = [improvements[c] for c in CUTOFFS]
    monotone = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
    recency_holds = improvements[2026] > improvements[2023]
    if recency_holds and monotone:
        prediction = "HELD"
    elif recency_holds:
        prediction = "PARTIAL"
    else:
        prediction = "FAILED"
    say()
    say("  PRE-REGISTRATION SCORECARD (improvement = baseline LL - rebalanced LL,")
    say("  frozen mode; positive = rebalancing better; grows with recency?):")
    say(f"    2023 {improvements[2023]:+.4f} | 2024 {improvements[2024]:+.4f} | "
        f"2025 {improvements[2025]:+.4f} | 2026 {improvements[2026]:+.4f} "
        f"(19 matches -- low power)")
    say(f"    improvement(2026) > improvement(2023): "
        f"{'YES' if recency_holds else 'NO'}; monotone across cutoffs: "
        f"{'YES' if monotone else 'NO'}")
    say(f"    -> {prediction}"
        + ("" if prediction == "HELD" else
           " -- pattern present but noisy" if prediction == "PARTIAL" else
           " -- flat/reversed: mechanism not operating as theorized"))

    # ── Guardrail 2: noise band + walk-forward survival ──
    d_frozen = mid["pooled_frozen"]["poisson_ll_per_tm"] - base["pooled_frozen"]["poisson_ll_per_tm"]
    d_wf = mid["pooled_wf"]["poisson_ll_per_tm"] - base["pooled_wf"]["poisson_ll_per_tm"]
    gain = -d_frozen
    if gain <= 0:
        band = "no improvement to assess"
    elif gain < NOISE_BAND_LL:
        band = (f"WITHIN the §21 noise band (< {NOISE_BAND_LL} LL/tm; {NOISE_BAND_N} of "
                f"1001 tuning combos sat inside it) -- not distinguishable from noise")
    else:
        band = (f"OUTSIDE the §21 noise band (>= {NOISE_BAND_LL} LL/tm) -- "
                f"larger than typical plateau noise on this harness")
    say()
    say("  GUARDRAIL 2 (§21 noise-band skepticism):")
    say(f"    pooled frozen improvement: {gain:+.4f} LL/tm -> {band}")
    if gain <= 0:
        say(f"    pooled walk-forward delta: {d_wf:+.4f} LL/tm (no frozen improvement")
        say("    to carry over)")
    else:
        say(f"    pooled walk-forward delta: {d_wf:+.4f} LL/tm -> frozen improvement "
            f"{'SURVIVES' if d_wf < 0 else 'DOES NOT survive'} walk-forward mode")
    if prediction == "HELD" and gain <= 0:
        say("    NOTE: the scorecard pattern is technically monotone, but every")
        say("    magnitude is far below the noise band and the pooled aggregate")
        say("    is negative -- the pattern carries no practical weight here.")

    # ── Clean-sheet reliability ──
    say()
    say("  clean-sheet reliability, pooled held-out frozen (baseline -> rebalanced):")
    say(f"    {'P(CS) bin':<18s}{'n':>7s}{'pred':>9s}{'obs':>9s}   "
        f"{'pred':>9s}{'obs':>9s}")
    base_bins = cs_bin_table(base["cs_probs"], base["cs_events"])
    mid_bins = cs_bin_table(mid["cs_probs"], mid["cs_events"])
    for rb, rm in zip(base_bins, mid_bins):
        say(f"    {rb['bin']:<18s}{rb['n']:>7d}{rb['pred']:>9.3f}{rb['obs']:>9.3f}"
            f"   {rm['pred']:>9.3f}{rm['obs']:>9.3f}")

    # ── Sensitivity: defense-anchored variant ──
    say()
    say("-" * 60)
    say("SENSITIVITY: defense-anchored variant (the M-choice flag)")
    say("-" * 60)
    say("  M=midpoint is a design decision; the equally defensible alternative")
    say("  holds the Defense average fixed and rescales Attack only. If the two")
    say("  anchors disagree in direction, the conclusion is anchor-dependent.")
    for mode, label in (("frozen", "frozen"), ("wf", "walk-forward")):
        d = dfa[f"pooled_{mode}"]["poisson_ll_per_tm"] - base[f"pooled_{mode}"]["poisson_ll_per_tm"]
        d_mid = mid[f"pooled_{mode}"]["poisson_ll_per_tm"] - base[f"pooled_{mode}"]["poisson_ll_per_tm"]
        say(f"  [{label}] pooled LL/tm delta vs baseline: {d:+.4f} "
            f"(midpoint was {d_mid:+.4f})")
        parts, improved = [], 0
        for c in CUTOFFS:
            dc = ll(dfa, mode, c) - ll(base, mode, c)
            parts.append(f"{c} {dc:+.4f}")
            improved += int(dc < 0)
        say(f"    per-cutoff: {' | '.join(parts)} | improved {improved}/{len(CUTOFFS)}")
    agree = ((mid["pooled_frozen"]["poisson_ll_per_tm"] - base["pooled_frozen"]["poisson_ll_per_tm"]) < 0) == \
            ((dfa["pooled_frozen"]["poisson_ll_per_tm"] - base["pooled_frozen"]["poisson_ll_per_tm"]) < 0)
    say(f"  direction agreement (frozen pooled): "
        f"{'YES -- conclusion not anchor-dependent' if agree else 'NO -- conclusion FLIPS with the anchor choice (flag)'}")

    # ── Verdict (guardrail 4) ──
    say()
    say("-" * 60)
    say("VERDICT (guardrail 4: nothing is shipped automatically)")
    say("-" * 60)
    if gain <= 0:
        say("  Rebalancing does NOT improve the primary metric. Recommendation:")
        say("  keep update_team_ratings.py's season-boundary logic unchanged.")
        verdict = "no_improvement"
    elif gain < NOISE_BAND_LL:
        say(f"  Improvement ({gain:+.4f} LL/tm frozen) is WITHIN the §21 noise band")
        say(f"  ({NOISE_BAND_N} of 1001 tuning combos sat within {NOISE_BAND_LL}).")
        say("  Per guardrail 2 this is not shippable evidence, whatever the")
        say("  per-cutoff pattern looks like.")
        verdict = "within_noise_band"
    else:
        say("  Improvement is outside the §21 noise band. Weigh the evidence")
        say("  above before deciding: per-cutoff recency pattern "
            f"({prediction}), walk-forward survival "
            f"({'yes' if d_wf < 0 else 'no'}), CS calibration, anchor")
        say("  sensitivity. Shipping = adding the rebalancing step to")
        say("  update_team_ratings.py's season-boundary block and re-running")
        say("  the harness baseline (--label baseline) for the new reference.")
        verdict = ("candidate_outside_noise_band"
                   if prediction == "HELD" and d_wf < 0
                   else "candidate_with_caveats")
    say()
    say("  Out-of-scope note: rebalancing changes rating levels entering")
    say("  update denominators, so the §21 hyperparameter optimum could shift")
    say("  under rebalancing -- noted, not acted on (per the doc).")
    say()
    say("  update_team_ratings.py is UNCHANGED by this script.")

    # ── Save artifacts ──
    total_time = time.perf_counter() - t_start
    say()
    say(f"  total wall time: {total_time:.0f}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "rebalancing_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_REPORT_LINES) + "\n")

    def cuts(res):
        return {str(c): {"frozen": res["frozen"][c], "wf": res["wf"][c]}
                for c in CUTOFFS}

    summary = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "data": data_meta,
        "params": prod,
        "cutoffs": list(CUTOFFS),
        "guard": {"passed": bool(guard_ok), "n_checks": len(checks),
                  "wall_time_s": round(guard_time, 1)},
        "rebalance_spec": {
            "applied_at": "start of every season 2017-2026, after prune + "
                          "money blend, before the season's first match",
            "team_set": "all season teams incl. newcomers at pure priors",
            "anchor_primary": "midpoint (M = (avg_Attack+avg_Defense)/2) -- "
                              "design decision, flagged",
            "anchor_sensitivity": "defense (k_d=1, k_a=avg_Defense/avg_Attack)",
        },
        "baseline": {"pooled_frozen": base["pooled_frozen"],
                     "pooled_wf": base["pooled_wf"], "splits": cuts(base)},
        "rebalanced_midpoint": {"pooled_frozen": mid["pooled_frozen"],
                                "pooled_wf": mid["pooled_wf"],
                                "splits": cuts(mid)},
        "rebalanced_defense": {"pooled_frozen": dfa["pooled_frozen"],
                               "pooled_wf": dfa["pooled_wf"],
                               "splits": cuts(dfa)},
        "deltas_vs_baseline": {
            "frozen_pooled_ll": float(d_frozen),
            "wf_pooled_ll": float(d_wf),
            "frozen_pooled_mae": float(mid["pooled_frozen"]["mae"]
                                       - base["pooled_frozen"]["mae"]),
            "frozen_pooled_cs_ll": float(mid["pooled_frozen"]["cs_ll"]
                                         - base["pooled_frozen"]["cs_ll"]),
            "frozen_per_cutoff": {str(c): float(ll(mid, "frozen", c)
                                                - ll(base, "frozen", c))
                                  for c in CUTOFFS},
            "wf_per_cutoff": {str(c): float(ll(mid, "wf", c) - ll(base, "wf", c))
                              for c in CUTOFFS},
            "defense_frozen_pooled_ll": float(
                dfa["pooled_frozen"]["poisson_ll_per_tm"]
                - base["pooled_frozen"]["poisson_ll_per_tm"]),
        },
        "boundary_gaps_baseline_2026frame": base["gaps"][2026],
        "boundary_gaps_midpoint_2026frame": mid["gaps"][2026],
        "boundary_gaps_per_cutoff": {
            str(c): {"baseline_last": base["gaps"][c][-1],
                     "midpoint_last": mid["gaps"][c][-1]} for c in CUTOFFS},
        "preregistration": {
            "prediction": "improvement grows with cutoff recency (2026 > 2023)",
            "improvements_frozen": {str(c): float(improvements[c]) for c in CUTOFFS},
            "recency_holds": bool(recency_holds),
            "monotone": bool(monotone),
            "score": prediction,
        },
        "noise_band": {"gain_frozen_ll": float(gain),
                       "band_ll": NOISE_BAND_LL, "band_n_combos": NOISE_BAND_N,
                       "assessment": band,
                       "wf_survives": bool(d_wf < 0)},
        "anchor_direction_agrees": bool(agree),
        "verdict": verdict,
        "runtimes_s": {"guard": round(guard_time, 1),
                       "total": round(total_time, 1)},
    }
    summary_path = OUT_DIR / "rebalancing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    say(f"  Saved {report_path}")
    say(f"  Saved {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
