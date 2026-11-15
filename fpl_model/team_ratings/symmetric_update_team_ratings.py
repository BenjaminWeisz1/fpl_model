"""
Symmetric log-space update in isolation -- the §24 follow-up. The Kalman
experiment (§24) cleared its pre-registered bar marginally (no-leakage
pooled frozen -0.0012 LL/tm) but its gain was uniform across tiers, so
the adaptive-uncertainty MECHANISM was not evidenced; the §24 analysis
decomposed the gain as roughly -0.0003 from the ES-calibrated reference
("the Kalman dressed as the incumbent" -- the symmetric log-space update
plus a calibrated lam-dependent gain) and the rest from the variance
machinery's re-profiled learning schedule. The user's request: test the
symmetric-update component ALONE. If it reproduces most of the -0.0012,
it is the simpler, more defensible thing to ship; if it does not, the
variance-tracking machinery is earning its complexity -- either result
settles whether the modest gain is about the update's functional form
or genuinely about uncertainty tracking.

The ONE change (everything else verbatim from the incumbent engine):
    incumbent:  rating *= (1 - alpha + alpha * num / pred)
    symmetric:  rating *= exp(alpha * (num - pred) / pred)
These are identical to first order: the incumbent's per-update log move
is log(1 + alpha(r-1)) with r = num/pred, the symmetric form's is
alpha(r-1), and the difference is z - log(1+z) = z^2/2 - z^3/3 with
z = alpha(r-1) -- second order (hand-checked in STEP 0a). The two forms
differ materially only in extreme overshoots (r >> 1), where the
symmetric form moves further (exponential vs linear in the signal
ratio): at r = 8 the per-update factors differ by ~4%; that tail
behavior is part of what the A/B measures, not a bug. This is the
first-order skeleton of the §24 Kalman update (x += K(num-lam) at
steady-state gain), so the ladder below is a genuine decomposition.

Design: an A/B with ZERO hyperparameter selection -- production alpha,
beta, money_weight, home_advantage; only the update expression differs.
The primary decision number is therefore leakage-free by construction
(nothing is tuned on anything). Secondary: an alpha sensitivity sweep
(the form change slightly increases effective step size, and the §21
optimum 0.045 was tuned for the INCUMBENT form), reported both as the
in-sample-flattered global surface and via the §23/§24 no-leakage
discipline (per cutoff, alpha chosen by walk-forward LL on Season < C
only, then applied to score C).

Decomposition ladder (all four arms re-run here on the same context, so
the comparison is internally consistent):
    incumbent (multiplicative form)
    -> symmetric form @ alpha 0.045           [THE A/B: the form alone]
    -> ES-calibrated Kalman (§24 reference)   [+ calibrated lam-dependent
                                                gain, a_eff ~ 0.025-0.065
                                                across typical lam]
    -> §24 no-leakage Kalman (selected)       [+ entry/boundary P
                                                inflation + r_scale; the
                                                full machinery]
with the §24 selected arm cross-checked against the stored §24 summary
(reproduction guard).

Correctness (before any comparison; the §24 pattern -- the new engine
has no verbatim twin because there is no production code to mirror; the
incumbent engine's standard guard secures the baseline):
  STEP 0a  hand-checked 12-step toy: the two forms track each other to
           second order (same toy sequence as §24; expected max diff on
           the scale of §24's KF-vs-ES toy, 0.0031 log units; gate 0.02);
  STEP 0b  STRUCTURAL: with alpha = 0 BOTH engines are exact no-ops
           (x * 1.0 and x * exp(0.0) are bit-exact identity multipliers),
           so the symmetric engine must match the incumbent engine
           BIT-EXACTLY over the full history -- validating the entire
           loop copy (season transition, money blend, lazy adds,
           snapshot protocol) independent of the update expression;
  STEP 0c  production-alpha invariants: match-1 prediction and the 2017
           boundary snapshot bit-exact (no update has intervened
           differently yet), team sets identical, all predictions finite
           and positive, and the walk-forward trajectory divergence vs
           the incumbent reported (the second-order difference
           accumulating over 2017-2026).

PRE-REGISTERED PREDICTION (fixed in this docstring 2026-09-04, BEFORE
any harness comparison was run; scored honestly after):
  1. PRIMARY (symmetric @ production alpha, pooled frozen LL/tm, all
     four cutoffs): improvement expected in [-0.0005, +0.0008] with a
     central estimate ~+0.0003 -- i.e. it reproduces roughly a QUARTER
     of §24's -0.0012, not most of it, and the predicted verdict is
     MACHINERY_EARNS_IT. Rationale: §24's ES-calibrated reference
     (symmetric update + calibrated lam-dependent gain) scored -0.0003;
     the pure form removes even that gain variation. This registers a
     (respectful) disagreement with the motivating framing ("likely
     responsible for most of the real gain") -- the §24 anchor says a
     quarter; this experiment settles it. The genuinely uncertain
     component is whether the ES-reference's lam-scaled gain (larger
     effective steps on higher-scoring matches) helped or hurt; if it
     hurt, the pure form could reach ~+0.0006.
  2. ALPHA SENSITIVITY: the surface will be flat around 0.045 (§21
     found the incumbent's alpha surface flat and 0.045 optimal; the
     form change shifts effective step size by less than the
     second-order residual); no-leakage selected alphas within
     {0.040, 0.045, 0.050}; the selected-alpha delta vs fixed 0.045
     under 0.0005.
  3. TIERS: deltas uniform across early/late/prom/rest -- the form
     change has no uncertainty semantics, so no early-round locus
     (contrast §24's mechanism prediction, which also failed to show).
  4. LADDER: roughly monotone in machinery -- incumbent, then symmetric
     ~= ES-calibrated Kalman, then the §24 selected Kalman clearly
     ahead; the selected arm reproduces §24's -0.0012 (same engine,
     same hyperparameters, same data -> bit-identical).
Scoring rule (fixed before the run; improvement = incumbent minus
symmetric on pooled frozen LL/tm, so positive = better):
  FORM_EXPLAINS_MOST      improvement >= 0.0008 (two-thirds of §24's
                          -0.0012) AND walk-forward degradation
                          < 0.001 -> the symmetric update alone is the
                          ship candidate; the variance machinery adds
                          complexity without most of the measured gain;
  MACHINERY_EARNS_IT      improvement < 0.0008 (including any
                          degradation inside the noise band) -> most of
                          the measured gain requires the variance state;
                          the full model remains the only vehicle for
                          it (adoption still a separate review);
  FORM_HURTS_CHECK_ALPHA  improvement <= -0.001 (degradation beyond the
                          noise band) -> read the alpha surface before
                          concluding the form itself is bad (a step-size
                          mismatch artifact would look identical);
  (edge case)             frozen qualifies as FORM_EXPLAINS_MOST but
                          walk-forward degrades >= 0.001 ->
                          FORM_EXPLAINS_BUT_WF_DAMAGE: report, ship
                          nothing.
Whatever the verdict, update_team_ratings.py is NOT touched -- shipping
any change is a separate human review decision (guardrail, unchanged
since §20).

Out of scope: re-tuning beta/money_weight/home_advantage, promoted-team
entry changes, rebuilding any Kalman component on top of the symmetric
engine (that would be §24 again), shipping without review.

Outputs (data/current/team_ratings_oot/): sym_report.txt,
sym_summary.json, sym_alpha_grid.csv.

Run locally:
    python symmetric_update_team_ratings.py
"""
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.team_ratings.update_team_ratings import alpha, beta, home_advantage, money_weight

from fpl_model.team_ratings import ratings_fast
from fpl_model.team_ratings.evaluate_team_ratings_oot import OUT_DIR, load_inputs, summarize_predictions
from fpl_model.team_ratings.ratings_fast import fast_run, frozen_predict_fast  # noqa: F401 (frozen_predict_fast via evaluate)
from fpl_model.team_ratings.tune_team_ratings import CUTOFFS, build_context, fmt_params
from fpl_model.team_ratings.kalman_team_ratings import NOISE_BAND_LL, TOY_LAM0, TOY_NUMS, add_match_indices, build_season_teams, es_calibrated_reference, evaluate, incumbent_runner, kalman_runner

EPS = 1e-6  # mirrors the production loop's eps exactly (ratings_fast._EPS)

ALPHA_SENS_GRID = [0.035, 0.040, 0.045, 0.050, 0.055, 0.060]
FORM_EXPLAINS_THRESHOLD = 0.0008  # two-thirds of §24's -0.0012

# §24's no-leakage Kalman selections (fallback if kalman_summary.json is
# missing or covers a different data snapshot; values from §24's run).
K24_FALLBACK = {
    2023: {"q_within": 0.002, "q_boundary": 0.02, "p_promoted": 0.1, "r_scale": 2.0},
    2024: {"q_within": 0.002, "q_boundary": 0.02, "p_promoted": 0.1, "r_scale": 2.0},
    2025: {"q_within": 0.001, "q_boundary": 0.005, "p_promoted": 0.1, "r_scale": 1.0},
    2026: {"q_within": 0.002, "q_boundary": 0.005, "p_promoted": 0.1, "r_scale": 2.0},
}

_REPORT_LINES = []


def say(line=""):
    print(line)
    _REPORT_LINES.append(line)


# ── The engine: the incumbent with ONE expression changed ───────────

def sym_run(df, ave_goals, params, prior_map, snapshot_seasons=()):
    """The incumbent engine (a verbatim copy of fast_run's loop:
    season transition, money blend, lazy adds, snapshot protocol,
    prediction formula, eps guard) with the fixed-alpha multiplicative
    update replaced by its symmetric log-space equivalent. Signal/pred
    pairings are IDENTICAL to the incumbent:
        h_atk <- num_h/xh   h_def <- num_a/xa
        a_atk <- num_a/xa   a_def <- num_h/xh
    Returns (pred_home, pred_away, snapshots) in fast_run's formats."""
    a = params["alpha"]
    b_ = params["beta"]
    ha = params["home_advantage"]
    mw = params["money_weight"]
    two_ha = 2 - ha

    season = df["Season"].values
    home = df["Home"].values
    away = df["Away"].values
    h_npxg = df["Home_NPxG_Adj"].values.astype(float)
    a_npxg = df["Away_NPxG_Adj"].values.astype(float)
    h_npg = df["Home_NPG"].values.astype(float)
    a_npg = df["Away_NPG"].values.astype(float)

    season_teams = {}
    for s in pd.unique(season):
        m = season == s
        season_teams[s] = set(home[m]) | set(away[m])

    ratings = {}
    current_season = None
    n = len(df)
    ph = np.empty(n)
    pa = np.empty(n)
    snaps = {}
    eps = EPS

    for i in range(n):
        s = season[i]
        if s != current_season:
            current_season = s
            teams = season_teams[s]
            ratings = {t: v for t, v in ratings.items() if t in teams}
            for t, v in ratings.items():
                pr = prior_map.get((s, t))
                if pr is not None:
                    m_atk, m_def = pr
                    ratings[t] = [(1 - mw) * v[0] + mw * m_atk,
                                  (1 - mw) * v[1] + mw * m_def]
            if s in snapshot_seasons and s not in snaps:
                snap = {t: list(v) for t, v in ratings.items()}
                for t in sorted(teams):
                    if t not in snap:
                        m_atk, m_def = prior_map[(s, t)]
                        snap[t] = [m_atk, m_def]
                snaps[s] = snap
        hi, ai = home[i], away[i]
        if hi not in ratings:
            m_atk, m_def = prior_map[(s, hi)]
            ratings[hi] = [m_atk, m_def]
        if ai not in ratings:
            m_atk, m_def = prior_map[(s, ai)]
            ratings[ai] = [m_atk, m_def]
        h_atk, h_def = ratings[hi]
        a_atk, a_def = ratings[ai]

        xh = h_atk * (a_def / ave_goals) * ha
        xa = a_atk * (h_def / ave_goals) * two_ha
        ph[i] = xh
        pa[i] = xa

        num_h = h_npxg[i] * b_ + h_npg[i] * (1 - b_)
        num_a = a_npxg[i] * b_ + a_npg[i] * (1 - b_)
        # THE change: symmetric log-space update (identical pairings to
        # the incumbent's four multiplicative lines; see module docstring)
        nh_atk = h_atk * math.exp(a * ((num_h - xh) / max(xh, eps)))
        nh_def = h_def * math.exp(a * ((num_a - xa) / max(xa, eps)))
        na_atk = a_atk * math.exp(a * ((num_a - xa) / max(xa, eps)))
        na_def = a_def * math.exp(a * ((num_h - xh) / max(xh, eps)))
        ratings[hi] = [nh_atk, nh_def]
        ratings[ai] = [na_atk, na_def]

    return ph, pa, snaps


def sym_runner(params, prior_map):
    def r(c, frame, ave):
        return sym_run(frame, ave, params, prior_map, snapshot_seasons={c})
    return r


def sym_runner_by_alpha(alpha_by_cutoff, prior_map, prod):
    def r(c, frame, ave):
        return sym_run(frame, ave, dict(prod, alpha=alpha_by_cutoff[c]),
                       prior_map, snapshot_seasons={c})
    return r


def sym_wf_train_ll(params, train_df, ave, prior_map):
    """Walk-forward Poisson LL/tm over a TRAINING frame (sequential
    predict-then-update) -- the no-leakage alpha-selection criterion."""
    ph, pa, _ = sym_run(train_df, ave, params, prior_map)
    s, _, _ = summarize_predictions(ph, pa,
                                    train_df["Home_NPG"].values,
                                    train_df["Away_NPG"].values)
    return s["poisson_ll_per_tm"]


# ── Correctness: the §24 pattern, before any comparison ─────────────

def sym_toy(say):
    """STEP 0a. Returns (ok, max_diff)."""
    a_ = alpha
    lam0 = TOY_LAM0
    say()
    say("  Hand-checked toy (single rating, §24's 12-step sequence,")
    say(f"  x0 = log({lam0})):")
    say("    step 1 by hand (num = 2, lam = 1.3):")
    say("      incumbent move  = log(1 - 0.045 + 0.045*2/1.3)")
    say("                     = log(1.0242308)  = 0.0239419")
    say("      symmetric move  = 0.045*(2 - 1.3)/1.3 = 0.0242308")
    say("      diff = 0.000289 = z - log(1+z) = z^2/2 - z^3/3 with")
    say("      z = 0.0242308 -- second order, confirmed by hand.")
    say(f"    {'step':<6}{'num':>4}{'lam':>8}{'x_inc':>10}{'x_sym':>10}{'diff':>10}")
    x_inc = math.log(lam0)
    x_sym = math.log(lam0)
    max_diff = 0.0
    for t, num in enumerate(TOY_NUMS, 1):
        lam_i = math.exp(x_inc)
        x_inc += math.log(1 - a_ + a_ * num / lam_i)
        lam_s = math.exp(x_sym)
        x_sym += a_ * (num - lam_s) / lam_s
        d = x_sym - x_inc
        max_diff = max(max_diff, abs(d))
        say(f"    {t:<6}{num:>4}{lam_s:>8.4f}{x_inc:>10.5f}{x_sym:>10.5f}"
            f"{d:>10.6f}")
    say(f"    max |x_sym - x_inc| over {len(TOY_NUMS)} steps = "
        f"{max_diff:.6f} log units (§24's KF-vs-ES toy on the same "
        f"sequence: 0.003139; gate 0.02)")
    ok = max_diff <= 0.02
    if not ok:
        say("    GATE FAILED -- the forms are not second-order equivalent;")
        say("    do not trust anything below.")
    return ok, max_diff


def sym_invariants(prod, prior_map, df, ave_full, say):
    """STEP 0b/0c. Returns True only if every gate passes."""
    ok = True
    say()
    say("  Structural check (alpha = 0: x*(1-0+0*r) and x*exp(0*z) are")
    say("  both bit-exact identity multipliers, so both engines are exact")
    say("  no-ops and MUST agree bit-for-bit):")
    p0 = dict(prod, alpha=0.0)
    ph_s, pa_s, sn_s = sym_run(df, ave_full, p0, prior_map,
                               snapshot_seasons={2017, 2026})
    ph_f, pa_f, sn_f = fast_run(df, ave_full, p0, prior_map,
                                snapshot_seasons={2017, 2026})
    d0 = float(max(np.max(np.abs(ph_s - ph_f)), np.max(np.abs(pa_s - pa_f))))
    say(f"    full-history walk-forward predictions: max |diff| = {d0:.2e} "
        f"({'PASS' if d0 == 0.0 else 'FAIL'})")
    ok = ok and d0 == 0.0
    d_snap = 0.0
    for ssn in (2017, 2026):
        for t, v in sn_f[ssn].items():
            k = sn_s[ssn][t]
            d_snap = max(d_snap, abs(k[0] - v[0]), abs(k[1] - v[1]))
    say(f"    boundary snapshots (2017, 2026): max |diff| = {d_snap:.2e} "
        f"({'PASS' if d_snap == 0.0 else 'FAIL'})")
    ok = ok and d_snap == 0.0

    say()
    say("  Production-alpha invariants (full history):")
    ph_s, pa_s, sn_s = sym_run(df, ave_full, prod, prior_map,
                               snapshot_seasons={2017})
    ph_f, pa_f, sn_f = fast_run(df, ave_full, prod, prior_map,
                                snapshot_seasons={2017})
    d1 = float(max(abs(ph_s[0] - ph_f[0]), abs(pa_s[0] - pa_f[0])))
    say(f"    match-1 prediction (identical expression, no update yet): "
        f"max |diff| = {d1:.2e} ({'PASS' if d1 == 0.0 else 'FAIL'})")
    ok = ok and d1 == 0.0
    d2 = 0.0
    for t, v in sn_f[2017].items():
        k = sn_s[2017][t]
        d2 = max(d2, abs(k[0] - v[0]), abs(k[1] - v[1]))
    same = set(sn_s[2017]) == set(sn_f[2017])
    say(f"    2017 boundary snapshot: max |diff| = {d2:.2e}; team sets "
        f"identical: {'PASS' if (d2 == 0.0 and same) else 'FAIL'}")
    ok = ok and d2 == 0.0 and same
    fin = bool(np.isfinite(ph_s).all() and np.isfinite(pa_s).all()
               and (ph_s > 0).all() and (pa_s > 0).all())
    say(f"    all predictions finite and > 0: {'PASS' if fin else 'FAIL'}")
    ok = ok and fin
    rel = np.concatenate([np.abs(ph_s - ph_f) / ph_f,
                          np.abs(pa_s - pa_f) / pa_f])
    say(f"    walk-forward divergence vs incumbent over 2017-2026 (the")
    say(f"    second-order form difference accumulating): relative |diff| "
        f"median {np.median(rel):.2e}, mean {rel.mean():.2e}, "
        f"max {rel.max():.2e}")
    return ok


def load_k24_selection(data_meta, say):
    """§24's no-leakage Kalman selections: read from
    kalman_summary.json when it covers the same data snapshot (single
    source of truth); otherwise the recorded fallback values. Returns
    (hyper_by_cutoff, stored_summary_or_None)."""
    path = OUT_DIR / "kalman_summary.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            stored = json.load(f)
        if stored["data"]["match_results"]["md5"] == \
                data_meta["match_results"]["md5"]:
            sel = {int(c): dict(v["hyper"])
                   for c, v in stored["no_leakage_selection"].items()}
            say("  §24 selections read from kalman_summary.json "
                "(same data snapshot)")
            return sel, stored
        say("  kalman_summary.json covers a different data snapshot -- "
            "using the recorded §24 fallback hyperparameters")
    else:
        say("  kalman_summary.json not found -- using the recorded §24 "
            "fallback hyperparameters")
        stored = None
    return {c: dict(v) for c, v in K24_FALLBACK.items()}, stored


# ── Main ─────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()
    df, prior_map, data_meta = load_inputs()
    prod = {"alpha": alpha, "beta": beta,
            "home_advantage": home_advantage, "money_weight": money_weight}
    ave_full = float(pd.concat([df["Home_NPG"], df["Away_NPG"]]).mean())
    df = add_match_indices(df)
    ctx = build_context(df)
    seasons, team_sets = build_season_teams(df)

    say("=" * 60)
    say("SYMMETRIC LOG-SPACE UPDATE IN ISOLATION -- EXPERIMENT 5 (§24 follow-up)")
    say("=" * 60)
    say(f"  data: match_results.csv {data_meta['match_results']['rows']} rows "
        f"(md5 {data_meta['match_results']['md5'][:10]}..., "
        f"{data_meta['match_results']['mtime']}), "
        f"money_priors {data_meta['money_priors']['rows']} rows")
    say(f"  cutoffs: {list(CUTOFFS)} (rolling origin, same as §20-§24)")
    say(f"  everything at production: {fmt_params(prod)}, "
        f"home_advantage={home_advantage:.9f}")
    say("  THE change (one expression, four lines):")
    say("    incumbent: rating *= (1 - alpha + alpha * num / pred)")
    say("    symmetric: rating *= exp(alpha * (num - pred) / pred)")
    say("  identical to first order (diff = (alpha(r-1))^2/2, second")
    say("  order); materially different only in extreme overshoots, where")
    say("  the symmetric form moves further (exp vs linear in num/pred).")
    say("  ZERO hyperparameters selected in the primary A/B -- the")
    say("  decision number is leakage-free by construction. Alpha")
    say("  sensitivity secondary (global surface = flattered; no-leakage")
    say(f"  per-cutoff selection over {ALPHA_SENS_GRID} the honest read).")
    say("  decomposition ladder: incumbent -> symmetric -> ES-calibrated")
    say("  Kalman (§24 reference) -> §24 no-leakage Kalman (full")
    say("  machinery), all re-run here on the same context.")
    say()
    say("PRE-REGISTERED PREDICTION (docstring, fixed 2026-09-04 before any")
    say("harness comparison ran): symmetric @ production alpha improves")
    say("pooled frozen by [-0.0005, +0.0008], central ~+0.0003 -- about a")
    say("QUARTER of §24's -0.0012, not most of it (the §24 ES-reference")
    say("anchor; registers a respectful disagreement with the motivating")
    say("'most of the gain' framing -- this run settles it). Predicted")
    say("verdict MACHINERY_EARNS_IT. Alpha surface flat around 0.045;")
    say("tier deltas uniform (the form has no uncertainty semantics);")
    say("ladder monotone in machinery; §24 selected arm reproduces -0.0012.")
    say("Scoring: FORM_EXPLAINS_MOST if improvement >= 0.0008 with wf")
    say("degradation < 0.001; MACHINERY_EARNS_IT if < 0.0008; ")
    say("FORM_HURTS_CHECK_ALPHA if <= -0.001 (read the alpha surface")
    say("before concluding the form is bad).")

    # ── STEP 0: correctness before any comparison ───────────────────
    say()
    say("-" * 60)
    say("STEP 0: CORRECTNESS -- second-order equivalence + engine identity")
    say("-" * 60)
    toy_ok, toy_max = sym_toy(say)
    inv_ok = sym_invariants(prod, prior_map, df, ave_full, say)
    if not (toy_ok and inv_ok):
        say()
        say("  STEP 0 FAILED -- aborting (implementation bug risk; nothing")
        say("  below would be trustworthy).")
        return 1

    # ── STEP 1: guard + baseline ────────────────────────────────────
    say()
    say("-" * 60)
    say("STEP 1: VERIFICATION GUARD (incumbent engine) + BASELINE")
    say("-" * 60)
    checks = [(
        "default path: production params, cutoff 2026 frame "
        f"({len(ctx['frames'][2026])} rows; walk-forward + snapshot + frozen)",
        ctx["frames"][2026], ctx["ave"][2026], prod, 2026),
        ("default path: full history "
         f"({len(df)} rows; walk-forward)",
         df, ave_full, prod, None)]
    t_guard = time.perf_counter()
    guard_ok = ratings_fast.verify_against_reference(checks, prior_map, log=say)
    say(f"  guard wall time: {time.perf_counter() - t_guard:.0f}s "
        f"(the symmetric engine has no verbatim twin -- STEP 0 is its")
    say(f"  correctness story; this guard secures the incumbent baseline)")
    if not guard_ok:
        say("  GUARD FAILED -- aborting.")
        return 1

    base = evaluate(incumbent_runner(prod, prior_map), ctx, prior_map,
                    prod, team_sets)
    for key in ("all", "early", "late", "prom", "rest"):
        s_f, s_w = base["pooled"][key]["frozen"], base["pooled"][key]["wf"]
        if s_f is None:
            say(f"  baseline pooled {key:<6}: no held-out matches")
            continue
        say(f"  baseline pooled {key:<6}: frozen {s_f['poisson_ll_per_tm']:.4f}"
            f" (n={s_f['n_matches']}) | wf {s_w['poisson_ll_per_tm']:.4f} | "
            f"MAE {s_f['mae']:.4f} | CS LL {s_f['cs_ll']:.4f}")
    baseline_json = OUT_DIR / "baseline.json"
    if baseline_json.exists():
        with open(baseline_json, encoding="utf-8") as f:
            stored = json.load(f)
        if stored["data"]["match_results"]["md5"] == \
                data_meta["match_results"]["md5"]:
            ref = stored["pooled"]["frozen"]["poisson_ll_per_tm"]
            cur = base["pooled"]["all"]["frozen"]["poisson_ll_per_tm"]
            d = abs(ref - cur)
            say(f"  cross-check vs stored baseline.json (ALL matches): "
                f"{ref:.10f} vs {cur:.10f} (|diff| = {d:.2e}) -> "
                f"{'MATCH' if d < 1e-9 else 'MISMATCH'}")
        else:
            say("  cross-check vs stored baseline.json: SKIPPED "
                "(match_results.csv changed since that run)")

    # ── STEP 2: THE A/B + decomposition ladder ──────────────────────
    say()
    say("-" * 60)
    say("STEP 2: THE A/B (production alpha, form swapped, nothing else)")
    say("-" * 60)
    symres = evaluate(sym_runner(prod, prior_map), ctx, prior_map,
                      prod, team_sets)
    for c in CUTOFFS:
        b, v = base["per"][c]["tiers"], symres["per"][c]["tiers"]
        say(f"  cutoff {c}:")
        for key in ("all", "early", "late", "prom", "rest"):
            tb, tv = b[key], v[key]
            if tb["frozen"] is None or tv["frozen"] is None:
                say(f"    {key:<6}: no held-out matches")
                continue
            d_f = tv["frozen"]["poisson_ll_per_tm"] - \
                tb["frozen"]["poisson_ll_per_tm"]
            d_w = tv["wf"]["poisson_ll_per_tm"] - tb["wf"]["poisson_ll_per_tm"]
            say(f"    {key:<6} n={tb['n']:>3}: frozen "
                f"{tb['frozen']['poisson_ll_per_tm']:.4f} -> "
                f"{tv['frozen']['poisson_ll_per_tm']:.4f} ({d_f:+.4f}) | wf "
                f"{tb['wf']['poisson_ll_per_tm']:.4f} -> "
                f"{tv['wf']['poisson_ll_per_tm']:.4f} ({d_w:+.4f})")
    say()
    say("  POOLED across cutoffs (match-count-weighted):")
    for key in ("all", "early", "late", "prom", "rest"):
        bf = base["pooled"][key]["frozen"]
        vf = symres["pooled"][key]["frozen"]
        if bf is None or vf is None:
            say(f"    {key:<6}: no held-out matches")
            continue
        bw = base["pooled"][key]["wf"]
        vw = symres["pooled"][key]["wf"]
        d_f = vf["poisson_ll_per_tm"] - bf["poisson_ll_per_tm"]
        d_w = vw["poisson_ll_per_tm"] - bw["poisson_ll_per_tm"]
        say(f"    {key:<6}: frozen {bf['poisson_ll_per_tm']:.4f} -> "
            f"{vf['poisson_ll_per_tm']:.4f} ({d_f:+.4f}) | wf "
            f"{bw['poisson_ll_per_tm']:.4f} -> {vw['poisson_ll_per_tm']:.4f} "
            f"({d_w:+.4f}) | MAE {bf['mae']:.4f} -> {vf['mae']:.4f} | CS LL "
            f"{bf['cs_ll']:.4f} -> {vf['cs_ll']:.4f} (n={bf['n_matches']})")

    say()
    say("  Decomposition ladder (all arms re-run on this context; pooled")
    say("  frozen LL/tm, lower = better; deltas vs incumbent):")
    b_f = base["pooled"]["all"]["frozen"]["poisson_ll_per_tm"]
    b_w = base["pooled"]["all"]["wf"]["poisson_ll_per_tm"]

    es_ref = es_calibrated_reference(alpha, ave_full, say)
    es_arm = evaluate(kalman_runner({c: es_ref for c in CUTOFFS},
                                    prior_map, prod),
                      ctx, prior_map, prod, team_sets)
    say()
    sel24, stored24 = load_k24_selection(data_meta, say)
    sel_arm = evaluate(kalman_runner(sel24, prior_map, prod), ctx,
                       prior_map, prod, team_sets)

    arms = [
        ("incumbent (multiplicative form)", base),
        ("symmetric form @ alpha 0.045", symres),
        ("ES-calibrated Kalman (§24 ref)", es_arm),
        ("§24 no-leakage Kalman (selected)", sel_arm),
    ]
    say(f"    {'arm':<36}{'frozen':>9}{'delta':>9}{'wf':>9}{'MAE':>8}{'CS':>8}")
    arm_pooled = {}
    for name, res in arms:
        f = res["pooled"]["all"]["frozen"]
        w = res["pooled"]["all"]["wf"]
        arm_pooled[name] = {"frozen": f, "wf": w}
        say(f"    {name:<36}{f['poisson_ll_per_tm']:>9.4f}"
            f"{f['poisson_ll_per_tm'] - b_f:>+9.4f}"
            f"{w['poisson_ll_per_tm']:>9.4f}{f['mae']:>8.4f}"
            f"{f['cs_ll']:>8.4f}")
    sym_f = symres["pooled"]["all"]["frozen"]["poisson_ll_per_tm"]
    es_f = es_arm["pooled"]["all"]["frozen"]["poisson_ll_per_tm"]
    sel_f = sel_arm["pooled"]["all"]["frozen"]["poisson_ll_per_tm"]
    k24_imp = b_f - sel_f
    sym_imp = b_f - sym_f
    frac = sym_imp / k24_imp if k24_imp > 0 else float("nan")
    say(f"    -> the symmetric form alone reproduces {frac:.0%} of the §24")
    say(f"       gain ({sym_imp:+.4f} of {k24_imp:+.4f}); ES-ref-vs-symmetric")
    say(f"       isolates the calibrated lam-dependent gain ({es_f - sym_f:+.4f});")
    say(f"       selected-vs-ES-ref isolates entry/boundary P inflation +")
    say(f"       r_scale ({sel_f - es_f:+.4f}).")
    if stored24 is not None:
        ref_delta = stored24["scorecard"]["no_leakage_pooled_frozen_delta"]
        cur_delta = sel_f - b_f
        say(f"    §24 reproduction check: stored delta {ref_delta:+.6f} vs "
            f"recomputed {cur_delta:+.6f} "
            f"(|diff| {abs(ref_delta - cur_delta):.1e}) -> "
            f"{'MATCH' if abs(ref_delta - cur_delta) < 1e-9 else 'MISMATCH'}")

    # ── STEP 3: alpha sensitivity ───────────────────────────────────
    say()
    say("-" * 60)
    say("STEP 3: ALPHA SENSITIVITY (the form slightly increases effective")
    say("step size; §21's 0.045 was tuned for the incumbent form)")
    say("-" * 60)
    t_a = time.perf_counter()
    surf = []
    for a_ in ALPHA_SENS_GRID:
        res = evaluate(sym_runner(dict(prod, alpha=a_), prior_map),
                       ctx, prior_map, prod, team_sets)
        surf.append({"alpha": a_,
                     "pooled_frozen_ll": res["pooled"]["all"]["frozen"]["poisson_ll_per_tm"],
                     "pooled_wf_ll": res["pooled"]["all"]["wf"]["poisson_ll_per_tm"]})
    say(f"  scored {len(surf)} alphas in {time.perf_counter() - t_a:.0f}s")
    say(f"    {'alpha':>7}{'frozen':>9}{'wf':>9}")
    for r in surf:
        marker = "  <- production" if r["alpha"] == alpha else ""
        say(f"    {r['alpha']:>7.3f}{r['pooled_frozen_ll']:>9.4f}"
            f"{r['pooled_wf_ll']:>9.4f}{marker}")
    say(f"    (incumbent baseline for reference: frozen {b_f:.4f} | "
        f"wf {b_w:.4f})")

    say()
    say("  No-leakage alpha selection (per cutoff, by walk-forward LL on")
    say("  Season < C only, then applied to score C -- the honest read):")
    sel_alpha = {}
    sel_alpha_detail = {}
    for c in CUTOFFS:
        train = df[df["Season"] < c]
        best = None
        for a_ in ALPHA_SENS_GRID:
            ll = sym_wf_train_ll(dict(prod, alpha=a_), train,
                                 ctx["ave"][c], prior_map)
            if best is None or ll < best[0]:
                best = (ll, a_)
        sel_alpha[c] = best[1]
        sel_alpha_detail[c] = best[0]
        say(f"    cutoff {c}: selected alpha {best[1]:.3f} (train wf LL/tm "
            f"{best[0]:.4f}; train rows {len(train)})")
    selres = evaluate(sym_runner_by_alpha(sel_alpha, prior_map, prod),
                      ctx, prior_map, prod, team_sets)
    sa_f = selres["pooled"]["all"]["frozen"]["poisson_ll_per_tm"]
    sa_w = selres["pooled"]["all"]["wf"]["poisson_ll_per_tm"]
    say(f"    honest pooled with per-cutoff selected alphas: frozen "
        f"{sa_f:.4f} ({sa_f - b_f:+.4f} vs incumbent; {sa_f - sym_f:+.4f} "
        f"vs fixed 0.045) | wf {sa_w:.4f} ({sa_w - b_w:+.4f})")

    # ── STEP 4: scorecard + verdict ─────────────────────────────────
    say()
    say("-" * 60)
    say("SCORECARD vs PRE-REGISTERED PREDICTION")
    say("-" * 60)
    s_f = sym_f - b_f
    s_wf = symres["pooled"]["all"]["wf"]["poisson_ll_per_tm"] - b_w
    improvement = -s_f
    e_b = base["pooled"]["early"]["frozen"]["poisson_ll_per_tm"]
    e_s = symres["pooled"]["early"]["frozen"]["poisson_ll_per_tm"]
    l_b = base["pooled"]["late"]["frozen"]["poisson_ll_per_tm"]
    l_s = symres["pooled"]["late"]["frozen"]["poisson_ll_per_tm"]
    pr_b = base["pooled"]["prom"]["frozen"]["poisson_ll_per_tm"]
    pr_s = symres["pooled"]["prom"]["frozen"]["poisson_ll_per_tm"]
    say(f"  symmetric @ 0.045 pooled frozen: {s_f:+.4f} LL/tm "
        f"(improvement {improvement:.4f}; noise band {NOISE_BAND_LL}; "
        f"{frac:.0%} of the §24 gain)")
    say(f"  tier deltas: early {e_s - e_b:+.4f} | late {l_s - l_b:+.4f} | "
        f"prom {pr_s - pr_b:+.4f} (uniformity check)")
    say(f"  pooled walk-forward: {s_wf:+.4f} LL/tm")
    say(f"  no-leakage alpha: {sa_f - b_f:+.4f} vs incumbent "
        f"({sa_f - sym_f:+.4f} vs fixed 0.045)")
    say(f"  ladder: ES-ref {es_f - b_f:+.4f} | §24 selected "
        f"{sel_f - b_f:+.4f} (reproduces §24)")

    say()
    say("-" * 60)
    say("VERDICT")
    say("-" * 60)
    if improvement >= FORM_EXPLAINS_THRESHOLD:
        if s_wf < NOISE_BAND_LL:
            verdict = "form_explains_most"
            say("  FORM_EXPLAINS_MOST: the symmetric update alone "
                f"reproduces {frac:.0%} of the")
            say(f"  §24 gain ({improvement:.4f} LL/tm, >= "
                f"{FORM_EXPLAINS_THRESHOLD:.4f} threshold) with no")
            say("  walk-forward damage -- the simpler, more defensible ship")
            say("  candidate; the variance machinery adds complexity without")
            say("  most of the measured gain. ONE line of production code.")
        else:
            verdict = "form_explains_but_wf_damage"
            say("  FORM_EXPLAINS_BUT_WF_DAMAGE: the frozen-mode gain "
                f"({improvement:.4f}) clears the")
            say(f"  threshold but walk-forward degrades {s_wf:+.4f} >= "
                f"{NOISE_BAND_LL:.3f} --")
            say("  not shippable as-is; the frozen/wf split needs reading")
            say("  before any decision.")
    elif improvement <= -NOISE_BAND_LL:
        verdict = "form_hurts_check_alpha"
        say("  FORM_HURTS_CHECK_ALPHA: the symmetric form at production "
            f"alpha degrades ({s_f:+.4f},")
        say("  beyond the noise band). Read the STEP 3 surface before")
        say("  concluding the form itself is bad -- a step-size mismatch")
        say("  artifact (the symmetric form's slightly larger effective")
        say("  moves at the same alpha) would look identical.")
    else:
        verdict = "machinery_earns_it"
        say("  MACHINERY_EARNS_IT: the symmetric form alone does NOT "
            "reproduce most of")
        say(f"  the §24 gain ({improvement:+.4f} of {k24_imp:+.4f} = "
            f"{frac:.0%}; threshold")
        say(f"  {FORM_EXPLAINS_THRESHOLD:.4f}). Most of the measured gain "
            "requires the")
        say("  variance-tracking machinery -- per §24's read, it functions "
            "as a")
        say("  re-profiled learning schedule (slower steady-state, much "
            "faster")
        say("  entry/boundary). If the team-ratings gain matters enough to "
            "ship")
        say("  anything, the full Kalman remains the only vehicle for it; "
            "the")
        say("  symmetric form is not a shortcut to the same result.")
    say()
    say("  update_team_ratings.py UNCHANGED (measured, not shipped);")
    say("  adoption of anything here is a separate review decision.")

    # ── Artifacts ───────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_rows = [dict(r, note=None) for r in surf] + [
        {"alpha": None, "pooled_frozen_ll": b_f, "pooled_wf_ll": b_w,
         "note": "incumbent baseline (multiplicative form, alpha 0.045)"},
        {"alpha": None, "pooled_frozen_ll": es_f, "pooled_wf_ll":
         es_arm["pooled"]["all"]["wf"]["poisson_ll_per_tm"],
         "note": "ES-calibrated Kalman (§24 reference)"},
        {"alpha": None, "pooled_frozen_ll": sel_f, "pooled_wf_ll":
         sel_arm["pooled"]["all"]["wf"]["poisson_ll_per_tm"],
         "note": "§24 no-leakage Kalman (selected)"},
        {"alpha": None, "pooled_frozen_ll": sa_f, "pooled_wf_ll": sa_w,
         "note": "symmetric, no-leakage per-cutoff alphas"},
    ]
    pd.DataFrame(grid_rows).to_csv(OUT_DIR / "sym_alpha_grid.csv",
                                   index=False)
    report = OUT_DIR / "sym_report.txt"
    report.write_text("\n".join(_REPORT_LINES) + "\n", encoding="utf-8")
    summary = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "data": data_meta,
        "params_fixed": prod,
        "design": {
            "change": "update form only: rating *= exp(alpha*(num-pred)/"
                      "pred) replacing rating *= (1-alpha+alpha*num/pred); "
                      "identical pairings, transition, blend, snapshots",
            "primary_ab": "production alpha, zero hyperparameter "
                          "selection (leakage-free by construction)",
            "form_explains_threshold": FORM_EXPLAINS_THRESHOLD,
            "noise_band_ll": NOISE_BAND_LL,
        },
        "correctness": {
            "toy_max_diff_log_units": toy_max,
            "toy_gate": 0.02,
            "invariants_pass": inv_ok,
        },
        "baseline_pooled": {k: base["pooled"][k] for k in
                            ("all", "early", "late", "prom", "rest")},
        "sym_pooled": {k: symres["pooled"][k] for k in
                       ("all", "early", "late", "prom", "rest")},
        "sym_per_cutoff": {str(c): symres["per"][c]["tiers"] for c in CUTOFFS},
        "ladder": {name: {"pooled_frozen_ll":
                          res["pooled"]["all"]["frozen"]["poisson_ll_per_tm"],
                          "pooled_wf_ll":
                          res["pooled"]["all"]["wf"]["poisson_ll_per_tm"],
                          "pooled_mae": res["pooled"]["all"]["frozen"]["mae"],
                          "pooled_cs_ll": res["pooled"]["all"]["frozen"]["cs_ll"]}
                   for name, res in arms},
        "alpha_surface": surf,
        "no_leakage_alpha": {str(c): {"alpha": sel_alpha[c],
                                      "train_wf_ll": sel_alpha_detail[c]}
                             for c in CUTOFFS},
        "no_leakage_alpha_pooled": {
            "frozen_ll": sa_f, "wf_ll": sa_w,
            "delta_vs_baseline": sa_f - b_f,
            "delta_vs_fixed_045": sa_f - sym_f},
        "k24_reproduction": (None if stored24 is None else {
            "stored_delta": stored24["scorecard"]["no_leakage_pooled_frozen_delta"],
            "recomputed_delta": sel_f - b_f,
            "match": bool(abs(stored24["scorecard"]["no_leakage_pooled_frozen_delta"]
                              - (sel_f - b_f)) < 1e-9)}),
        "scorecard": {
            "sym_fixed_delta": s_f,
            "improvement": improvement,
            "fraction_of_k24_gain": frac,
            "early_delta": e_s - e_b,
            "late_delta": l_s - l_b,
            "prom_delta": pr_s - pr_b,
            "wf_delta": s_wf,
            "es_ref_delta": es_f - b_f,
            "k24_sel_delta": sel_f - b_f,
            "no_leakage_alpha_delta": sa_f - b_f,
        },
        "verdict": verdict,
        "update_team_ratings_changed": False,
    }
    (OUT_DIR / "sym_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    say()
    say(f"  wall time: {time.perf_counter() - t0:.0f}s")
    say(f"  artifacts: {report.name}, sym_summary.json, sym_alpha_grid.csv "
        f"-> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
