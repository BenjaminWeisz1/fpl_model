"""
State-space (Kalman-filter) reformulation of the team ratings model --
implements KALMAN_FILTER.md (a STRUCTURAL experiment, the fourth
generation after tuning §21 / rebalancing §22 / promoted discount §23,
all null) against the §20 OOT harness. Built as a SEPARATE, PARALLEL
implementation: update_team_ratings.py is never touched; adoption is a
separate human decision after review. Given the series' 0/3 record,
this goes in expecting it may not clear the bar either -- the value is
as much "know for certain" as "expect to ship" (the doc's own framing).

The conceptual shift
--------------------
The incumbent updates every rating by the same fixed alpha regardless
of how much is currently known about that team. This model tracks, per
team per side, a log-rating x AND its variance P; the per-match move
(the Kalman gain) falls out of P mathematically. Higher effective
learning rate early / after boundaries / for promoted entries is an
emergent property, not a schedule.

Formulation (per the doc's recommendations):
- LOG-SPACE state: x_atk = log(Attack_Rating), x_def = log(Defense_
  Rating). The multiplicative prediction becomes additive; exp() only
  at prediction time. The prediction formula itself is UNCHANGED:
  lam_h = exp(x_atk[home]) * (exp(x_def[away]) / ave) * ha.
- DIAGONAL: per team, per side, independent. No cross-team, no
  attack/defense cross-covariance (the incumbent has none either).
- Process model: P += Q_within per match played (the incumbent's own
  time unit); P += Q_boundary at every season boundary (squad turnover
  -- a real discrete event); promoted entries initialize at
  P_promoted > the continuing teams' post-inflation P (greater genuine
  uncertainty in a money-implied-only prior). 2017 pool initialization
  uses the same P_promoted (same epistemic situation: pure priors, no
  in-pool evidence).
- Observation model: the incumbent's signal num = NPxG_adj*beta +
  NPG*(1-beta) (UNCHANGED, beta fixed at production) is a noisy
  observation of the true rate lambda, with Poisson variance
  approximated as R = r_scale * lambda (the doc's option (a); the
  exp() observation function is linearized at the current estimate --
  the trivial one-step Jacobian, no iteration).
- Update (replaces the alpha multiplicative update entirely): the
  match's home signal informs (x_atk[home], x_def[away]) jointly, the
  away signal informs (x_atk[away], x_def[home]) -- two DISJOINT pairs
  (the incumbent couples exactly the same pairs), so the two scalar
  updates are order-independent. Per pair, with H = (lam, lam):
      S = lam^2 (P1 + P2) + R        (innovation variance)
      K1 = lam P1 / S,  K2 = lam P2 / S
      x1 += K1 (num - lam),  x2 += K2 (num - lam)
      P1 <- P1 (1 - lam K1),  P2 <- P2 (1 - lam K2)
  These are the EXACT marginal posteriors of the joint 2-state update;
  the induced cross-covariance (-lam^2 P1 P2 / S) is discarded --
  that is the "keep it diagonal" instruction, stated not hidden.
- Season boundary keeps the incumbent's money-prior BLEND on the mean
  (rate-space arithmetic, identical form: x <- log((1-mw) exp(x) +
  mw prior)) -- this experiment isolates the UPDATE mechanism; mw
  stays production 0.3 (the §21 grid optimum). The blend does not
  shrink P (the incumbent's blend has no uncertainty semantics; the
  boundary's genuine new uncertainty enters via Q_boundary).

Reduction to the incumbent (the doc's validation point 1)
---------------------------------------------------------
Single-state special case (opponent side certain, P2 = 0): gain
K = lam P / (lam^2 P + R). Matching the incumbent's per-rating move
(alpha per match, to first order) requires lam*K = alpha, i.e. steady
state P** = alpha R / (lam^2 (1 - 2 alpha)) (two-state matched form;
P* = alpha R / (lam^2 (1 - alpha)) single-state), sustained by
Q** = alpha P** per match. Then the update is x += (alpha/lam)(num -
lam) vs the incumbent's x += log(1 - alpha + alpha num/lam) --
identical to SECOND order in (num - lam)/lam (the diff is
(alpha d / lam)^2 / 2). With Q = 0 the same P** matches step 1 exactly
(then P decays geometrically and the filter freezes -- process noise
is what sustains the ES-equivalent operating point). The script's
STEP 0 demonstrates this numerically on a hand-checked example using
the ACTUAL _kf_update code path, plus engine invariants:
  * match-1 prediction == the incumbent engine's (prior-based product,
    exp/log roundtrip only);
  * the 2017 boundary snapshot == the pure money priors;
  * all P finite and > 0 after a full-history run.
Plus one end-to-end anchor: an ES-CALIBRATED REFERENCE configuration
(Q=Q**, Q_boundary=0, P_promoted=P**, r_scale=1 -- "the Kalman dressed
as the incumbent") should score within ~0.005 pooled frozen LL of the
incumbent; a large deviation is an implementation-bug flag (investigate
before trusting anything).

Leakage discipline (the doc's validation point 4)
--------------------------------------------------
Hyperparameters (Q_within, Q_boundary, P_promoted, r_scale) are
selected TWO ways, both reported:
  * GLOBAL GRID (the ceiling): 300 combinations scored on all four
    cutoffs -- like §21, this SELECTS ON THE SCORED SEASONS and is
    therefore in-sample-flattered; reported for surface structure only.
  * NO-LEAKAGE SELECTION (the decision number): for each cutoff C, the
    combination is chosen by walk-forward Poisson LL/tm over matches
    with Season < C ONLY (the filter's native sequential tracking
    quality, using cutoff C's train-only ave_goals -- exactly the
    estimate production could have made at time C), then applied to
    score C. Mirrors §23's rolling-origin discipline.
beta, money_weight, home_advantage stay at production values
throughout -- only the four Kalman hyperparameters vary.

PRE-REGISTERED PREDICTION (fixed in this docstring 2026-09-04, BEFORE
any harness comparison was run; scored honestly after):
  1. PRIMARY (pooled frozen Poisson LL/tm, no-leakage selection, all
     four cutoffs): expected within +/-0.003 of the incumbent 0.8821.
     If the adaptive-uncertainty mechanism genuinely helps, a small
     improvement of order 0.001-0.003 (1-3x the §21 noise band of
     0.001 LL/tm) is the realistic ceiling -- §20-§23 all indicate the
     incumbent sits near this data's ceiling.
  2. LOCUS: any improvement should concentrate in (i) early-season
     matches (either team's first 5 of the season -- post-boundary P
     inflated, high gain) and (ii) walk-forward mode (continuous
     adaptation). Early-round frozen improvement >= 0.005 LL/tm with
     late-round ~0 would be clear mechanism evidence.
  3. SECONDARY (promoted-team matches): per the design doc, NOT framed
     as a promoted-team fix -- higher initial P speeds convergence but
     does not move the entry point estimate, and the §23 discount
     failed on cohort-varying optimism, not slow correction. Expect no
     reliable promoted-specific gain beyond the early-round effect.
  4. ALSO PRE-REGISTERED: the global-grid winner will look better than
     the no-leakage selection (§23 flattery pattern); the no-leakage
     number is the decision number.
Scoring rule (fixed before the run):
  SUPPORTED     no-leakage pooled frozen improves by > 0.001 LL/tm
                (beyond the noise band) AND early-round improvement
                exceeds late-round AND pooled walk-forward does not
                degrade by > 0.001;
  WEAK/PARTIAL  direction right but the pooled improvement is inside
                the noise band, or the locus is absent;
  FAILED        no improvement, degradation, or the gain exists only
                in the global-grid number (flattery-only).
If FAILED, that is FOUR consecutive nulls (three parametric, one
structural) -- a convergent signal that the incumbent is close to this
data's actual ceiling, to be treated as real information about the
model's limits (the doc's closing note).

Out of scope: joint attack/defense or cross-team covariance (possible
later refinement if this validates), full EKF/UKP iteration, re-tuning
beta/money_weight/home_advantage, replacing update_team_ratings.py.

Outputs (data/current/team_ratings_oot/): kalman_report.txt,
kalman_summary.json, kalman_grid.csv (301 rows incl. the ES reference).

Run locally:
    python kalman_team_ratings.py
"""
import itertools
import json
import math
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from fpl_model.team_ratings.update_team_ratings import alpha, beta, home_advantage, money_weight
from research.team_ratings import ratings_fast
from research.team_ratings.evaluate_team_ratings_oot import OUT_DIR, load_inputs, summarize_predictions
from research.team_ratings.ratings_fast import fast_run, frozen_predict_fast
from research.team_ratings.tune_team_ratings import CUTOFFS, build_context, fmt_params

# §21 noise-band reference points, fixed for the skepticism assessment.
NOISE_BAND_LL = 0.001
NOISE_BAND_N = 292

EARLY_N = 5  # per-team match number within season: "early" = first 5

Q_WITHIN_GRID = [0.0, 0.0005, 0.001, 0.002, 0.004]
Q_BOUNDARY_GRID = [0.005, 0.02, 0.05, 0.1, 0.2]
P_PROMOTED_GRID = [0.05, 0.1, 0.2, 0.4]
R_SCALE_GRID = [0.5, 1.0, 2.0]

TOY_NUMS = [2, 0, 1, 3, 1, 2, 0, 1, 4, 1, 2, 1]
TOY_LAM0 = 1.3

_REPORT_LINES = []


def say(line=""):
    print(line)
    _REPORT_LINES.append(line)


# ── The Kalman engine ────────────────────────────────────────────────

def _kf_update(x1, P1, x2, P2, lam, num, r_scale):
    """One diagonal Kalman update step. Observation num (count scale)
    of lam = exp(x1 + x2 + const), linearized at the current estimate;
    R = r_scale * lam (Gaussian approximation to the Poisson
    likelihood, variance ~= mean). Returns updated (x1, P1, x2, P2) --
    the EXACT marginal posteriors; the induced cross-covariance is
    discarded (diagonal filter, per the design doc). With P2 = 0 the
    second state is certain and untouched (the reduction-check case)."""
    R = r_scale * lam
    S = lam * lam * (P1 + P2) + R
    K1 = lam * P1 / S
    K2 = lam * P2 / S
    inn = num - lam
    x1 += K1 * inn
    x2 += K2 * inn
    P1 = P1 - lam * K1 * P1
    P2 = P2 - lam * K2 * P2
    return x1, P1, x2, P2


def kalman_run(df, ave_goals, hyper, prior_map, beta_, mw, ha,
               snapshot_seasons=(), return_state=False):
    """Log-space diagonal Kalman engine over a match frame. Mirrors the
    harness protocol exactly: season transition (prune + money blend +
    Q_boundary inflation + promoted entry at P_promoted) happens at the
    first row of each season; the snapshot is captured after it; each
    match predicts BEFORE updating (walk-forward predictions returned
    for every row). Returns (pred_home, pred_away, snapshots); with
    return_state=True also the final state dict {team: [x_a, x_d,
    P_a, P_d]} for correctness checks."""
    q_w = hyper["q_within"]
    q_b = hyper["q_boundary"]
    p_pr = hyper["p_promoted"]
    r_s = hyper["r_scale"]

    season = df["Season"].values
    home = df["Home"].values
    away = df["Away"].values
    h_npxg = df["Home_NPxG_Adj"].values.astype(float)
    a_npxg = df["Away_NPxG_Adj"].values.astype(float)
    h_npg = df["Home_NPG"].values.astype(float)
    a_npg = df["Away_NPG"].values.astype(float)
    first_season = int(season[0])

    season_teams = {}
    for s in pd.unique(season):
        m = season == s
        season_teams[s] = set(home[m]) | set(away[m])

    st = {}          # team -> [x_atk, x_def, P_atk, P_def]
    cur = None
    n = len(df)
    ph = np.empty(n)
    pa = np.empty(n)
    snaps = {}

    for i in range(n):
        s = season[i]
        if s != cur:
            cur = s
            teams = season_teams[s]
            st = {t: v for t, v in st.items() if t in teams}
            if s == first_season:
                for t in teams:
                    if (s, t) not in prior_map:
                        raise KeyError(f"No money prior for ({s}, {t!r})")
                    m_a, m_d = prior_map[(s, t)]
                    st[t] = [math.log(m_a), math.log(m_d), p_pr, p_pr]
            else:
                for t in sorted(teams):
                    if (s, t) not in prior_map:
                        raise KeyError(f"No money prior for ({s}, {t!r})")
                    m_a, m_d = prior_map[(s, t)]
                    if t in st:
                        v = st[t]
                        # incumbent money blend, rate space (form identical)
                        v[0] = math.log((1 - mw) * math.exp(v[0]) + mw * m_a)
                        v[1] = math.log((1 - mw) * math.exp(v[1]) + mw * m_d)
                        v[2] += q_b
                        v[3] += q_b
                    else:
                        st[t] = [math.log(m_a), math.log(m_d), p_pr, p_pr]
            if s in snapshot_seasons and s not in snaps:
                snaps[s] = {t: [math.exp(v[0]), math.exp(v[1])]
                            for t, v in st.items()}
        hi, ai = home[i], away[i]
        vh = st[hi]
        va = st[ai]
        atk_h = math.exp(vh[0])
        def_h = math.exp(vh[1])
        atk_a = math.exp(va[0])
        def_a = math.exp(va[1])
        lam_h = atk_h * (def_a / ave_goals) * ha
        lam_a = atk_a * (def_h / ave_goals) * (2 - ha)
        ph[i] = lam_h
        pa[i] = lam_a

        num_h = h_npxg[i] * beta_ + h_npg[i] * (1 - beta_)
        num_a = a_npxg[i] * beta_ + a_npg[i] * (1 - beta_)

        # predict step: within-season process noise for the 4 states
        vh[2] += q_w
        vh[3] += q_w
        va[2] += q_w
        va[3] += q_w

        # home signal -> (x_atk[home], x_def[away]); away signal ->
        # (x_atk[away], x_def[home]): disjoint pairs, order-independent
        vh[0], vh[2], va[1], va[3] = _kf_update(
            vh[0], vh[2], va[1], va[3], lam_h, num_h, r_s)
        va[0], va[2], vh[1], vh[3] = _kf_update(
            va[0], va[2], vh[1], vh[3], lam_a, num_a, r_s)

    if return_state:
        return ph, pa, snaps, st
    return ph, pa, snaps


def es_calibrated_reference(alpha_, ave_full, say=None):
    """The 'Kalman dressed as incumbent' configuration: steady-state
    P**/Q** matched so each rating's per-match move equals the
    incumbent's alpha to first order (derivation in the module
    docstring), Q_boundary = 0 (the incumbent has no boundary
    uncertainty event), P_promoted = P** (no extra promoted
    uncertainty), r_scale = 1 (pure Poisson variance)."""
    lam = float(ave_full)
    p_ss = alpha_ / (lam * (1 - 2 * alpha_))     # two-state matched
    q_ss = alpha_ * p_ss                          # sustaining process noise
    ref = {"q_within": q_ss, "q_boundary": 0.0,
           "p_promoted": p_ss, "r_scale": 1.0}
    if say is not None:
        say(f"  ES-calibrated reference (matched at lam = ave_full = "
            f"{lam:.4f}, alpha = {alpha_}):")
        say(f"    P** = alpha/(lam(1-2 alpha)) = {p_ss:.5f}, "
            f"Q** = alpha P** = {q_ss:.6f}")
        say(f"    -> hyper = {ref} (Q_boundary = 0: the incumbent has no "
            f"boundary uncertainty event; P_promoted = P**: no extra "
            f"promoted uncertainty)")
    return ref


# ── Correctness: the doc's validation point 1 ───────────────────────

def reduction_check(prod, prior_map, df, ave_full, es_ref, say):
    """STEP 0. (1) Analytic derivation printed; (2) hand-checked toy:
    the sustained-Q** Kalman tracks the incumbent ES update to second
    order, using the ACTUAL _kf_update code path; (3) engine
    invariants vs the incumbent engine. Returns True only if all
    numeric gates pass."""
    a_ = prod["alpha"]
    lam0 = TOY_LAM0
    R = lam0  # r_scale = 1
    p_star = a_ * R / (lam0 * lam0 * (1 - a_))        # single-state match
    p_ss = a_ * R / (lam0 * lam0 * (1 - 2 * a_))      # two-state match
    q_sustain = a_ * p_star
    ok = True

    say("-" * 60)
    say("STEP 0: CORRECTNESS -- reduction to the incumbent update")
    say("-" * 60)
    say("  Single-state reduction (opponent certain, P2 = 0):")
    say("    K = lam P / (lam^2 P + R); the incumbent's per-rating move is")
    say("    alpha to first order iff lam*K = alpha, i.e. steady state")
    say(f"    P* = alpha R / (lam^2 (1-alpha)) = {p_star:.5f} at lam = {lam0},")
    say(f"    sustained by Q = alpha P* = {q_sustain:.6f} per match. The")
    say("    update becomes x += (alpha/lam)(num-lam) vs the incumbent's")
    say("    x += log(1 - alpha + alpha num/lam) -- identical to SECOND")
    say("    order in (num-lam)/lam; the residual is (alpha d/lam)^2 / 2.")
    say("    With Q = 0 the same P* matches step 1 exactly, then P decays")
    say("    geometrically and the filter freezes: process noise is what")
    say("    sustains the ES-equivalent operating point.")
    say()
    say("  Hand-checked toy (shared _kf_update code path, P2 = 0,")
    say(f"  x0 = log({lam0}), R = {R}, P0 = P*, Q = alpha P*):")
    say(f"    {'step':<6}{'num':>4}{'lam':>8}{'x_ES':>10}{'x_KF':>10}{'diff':>10}")
    x_es = math.log(lam0)
    x_kf = math.log(lam0)
    P = p_star
    max_diff = 0.0
    for t, num in enumerate(TOY_NUMS, 1):
        lam_es = math.exp(x_es)
        lam_kf = math.exp(x_kf)
        x_es += math.log(1 - a_ + a_ * num / lam_es)
        x_kf, P, _, _ = _kf_update(x_kf, P, 0.0, 0.0, lam_kf, float(num), 1.0)
        P += q_sustain
        d = x_kf - x_es
        max_diff = max(max_diff, abs(d))
        say(f"    {t:<6}{num:>4}{lam_kf:>8.4f}{x_es:>10.5f}{x_kf:>10.5f}"
            f"{d:>10.6f}")
    say(f"    max |x_KF - x_ES| over {len(TOY_NUMS)} steps = {max_diff:.6f} "
        f"log units (expected scale: (alpha d/lam)^2/2 ~ 0.0003; gate 0.02)")
    if max_diff > 0.02:
        ok = False
        say("    GATE FAILED -- the update does not reduce to ES; do not "
            "trust anything below.")

    say()
    say("  Engine invariants vs the incumbent engine (full history,")
    say("  ES-reference hyperparameters):")
    ph_ref, pa_ref, snaps_ref, st_final = kalman_run(
        df, ave_full, es_ref, prior_map, prod["beta"],
        prod["money_weight"], prod["home_advantage"],
        snapshot_seasons={2017}, return_state=True)
    ph_inc, pa_inc, snaps_inc = fast_run(
        df, ave_full, prod, prior_map, snapshot_seasons={2017})
    d0 = max(abs(ph_ref[0] - ph_inc[0]), abs(pa_ref[0] - pa_inc[0]))
    say(f"    match-1 prediction (prior-based product; only exp/log "
        f"roundtrip may differ): max |diff| = {d0:.2e} "
        f"({'PASS' if d0 < 1e-12 else 'FAIL'})")
    ok = ok and d0 < 1e-12
    d_snap = 0.0
    for t, v in snaps_inc[2017].items():
        k = snaps_ref[2017][t]
        d_snap = max(d_snap, abs(k[0] - v[0]), abs(k[1] - v[1]))
    say(f"    2017 boundary snapshot vs pure money priors: max |diff| = "
        f"{d_snap:.2e} ({'PASS' if d_snap < 1e-12 else 'FAIL'})")
    ok = ok and d_snap < 1e-12
    same_teams = set(snaps_ref[2017]) == set(snaps_inc[2017])
    say(f"    2017 snapshot team sets identical: "
        f"{'PASS' if same_teams else 'FAIL'}")
    ok = ok and same_teams
    p_vals = [v[2] for v in st_final.values()] + \
             [v[3] for v in st_final.values()]
    p_ok = all(math.isfinite(p) and p > 0 for p in p_vals)
    say(f"    final P over all teams/sides: finite and > 0 "
        f"(range {min(p_vals):.5f}..{max(p_vals):.5f}): "
        f"{'PASS' if p_ok else 'FAIL'}")
    ok = ok and p_ok
    return ok


# ── Evaluation ───────────────────────────────────────────────────────

def build_season_teams(df):
    seasons = [int(s) for s in sorted(df["Season"].unique())]
    team_sets = {}
    for s in seasons:
        d = df[df["Season"] == s]
        team_sets[s] = set(d["Home"]) | set(d["Away"])
    return seasons, team_sets


def add_match_indices(df):
    """Per-team match number within season (1-based) for home/away --
    the early/late split's clock. Columns added BEFORE build_context so
    every cutoff frame carries them."""
    season = df["Season"].values
    home = df["Home"].values
    away = df["Away"].values
    h_idx = np.empty(len(df), dtype=int)
    a_idx = np.empty(len(df), dtype=int)
    counts = {}
    for i in range(len(df)):
        kh = (season[i], home[i])
        ka = (season[i], away[i])
        counts[kh] = counts.get(kh, 0) + 1
        counts[ka] = counts.get(ka, 0) + 1
        h_idx[i] = counts[kh]
        a_idx[i] = counts[ka]
    df = df.copy()
    df["_h_idx"] = h_idx
    df["_a_idx"] = a_idx
    return df


def evaluate(runner, ctx, prior_map, prod, team_sets):
    """Score one engine configuration across the cutoffs with the
    harness protocol. runner(c, frame, ave) -> (ph, pa, snaps). Tiers:
    'all'; 'early' (either team's first EARLY_N matches of the season)
    and 'late' (partition of all); 'prom' (a held-out match involves a
    promoted team of that season) and 'rest' (partition of all) --
    early/late and prom/rest are two independent partitions, all
    diagnostic slices of the same predictions. Frozen primary,
    walk-forward secondary, per cutoff and pooled."""
    per = {}
    tiers = ("all", "early", "late", "prom", "rest")
    acc = {k: {"f_ph": [], "f_pa": [], "yh": [], "ya": [],
               "w_ph": [], "w_pa": []} for k in tiers}
    for c in CUTOFFS:
        frame = ctx["frames"][c]
        ph, pa, snaps = runner(c, frame, ctx["ave"][c])
        m = ctx["masks"][c]
        val = ctx["val"][c]
        fph, fpa = frozen_predict_fast(val, snaps[c], prior_map,
                                       ctx["ave"][c], prod)
        yh, ya = ctx["yh"][c], ctx["ya"][c]
        wph, wpa = ph[m], pa[m]
        promoted = team_sets[c] - team_sets[c - 1]
        prom_mask = (val["Home"].isin(promoted)
                     | val["Away"].isin(promoted)).values
        early_mask = ((val["_h_idx"] <= EARLY_N)
                      | (val["_a_idx"] <= EARLY_N)).values
        masks = {"all": np.ones(len(val), dtype=bool),
                 "early": early_mask, "late": ~early_mask,
                 "prom": prom_mask, "rest": ~prom_mask}
        entry = {}
        for key in tiers:
            mask = masks[key]
            if int(mask.sum()) == 0:
                entry[key] = {"frozen": None, "wf": None, "n": 0}
                continue
            s_f, _, _ = summarize_predictions(fph[mask], fpa[mask],
                                              yh[mask], ya[mask])
            s_w, _, _ = summarize_predictions(wph[mask], wpa[mask],
                                              yh[mask], ya[mask])
            entry[key] = {"frozen": s_f, "wf": s_w, "n": int(mask.sum())}
            acc[key]["f_ph"].append(fph[mask])
            acc[key]["f_pa"].append(fpa[mask])
            acc[key]["yh"].append(yh[mask])
            acc[key]["ya"].append(ya[mask])
            acc[key]["w_ph"].append(wph[mask])
            acc[key]["w_pa"].append(wpa[mask])
        per[c] = {"promoted": sorted(promoted), "tiers": entry}
    pooled = {}
    for key in tiers:
        if not acc[key]["f_ph"]:
            pooled[key] = {"frozen": None, "wf": None}
            continue
        s_f, _, _ = summarize_predictions(
            np.concatenate(acc[key]["f_ph"]),
            np.concatenate(acc[key]["f_pa"]),
            np.concatenate(acc[key]["yh"]),
            np.concatenate(acc[key]["ya"]))
        s_w, _, _ = summarize_predictions(
            np.concatenate(acc[key]["w_ph"]),
            np.concatenate(acc[key]["w_pa"]),
            np.concatenate(acc[key]["yh"]),
            np.concatenate(acc[key]["ya"]))
        pooled[key] = {"frozen": s_f, "wf": s_w}
    return {"per": per, "pooled": pooled}


def incumbent_runner(prod, prior_map):
    def r(c, frame, ave):
        return fast_run(frame, ave, prod, prior_map,
                        snapshot_seasons={c})
    return r


def kalman_runner(hyper_by_cutoff, prior_map, prod):
    def r(c, frame, ave):
        return kalman_run(frame, ave, hyper_by_cutoff[c], prior_map,
                          prod["beta"], prod["money_weight"],
                          prod["home_advantage"], snapshot_seasons={c})
    return r


def grid_score(hyper, ctx, prior_map, prod):
    """Slim scorer for the grid: pooled frozen + pooled walk-forward
    Poisson LL/tm across the four cutoffs (the 'all' tier of
    evaluate())."""
    f_ph, f_pa, yh, ya, w_ph, w_pa = [], [], [], [], [], []
    for c in CUTOFFS:
        frame = ctx["frames"][c]
        ph, pa, snaps = kalman_run(frame, ctx["ave"][c], hyper, prior_map,
                                   prod["beta"], prod["money_weight"],
                                   prod["home_advantage"],
                                   snapshot_seasons={c})
        m = ctx["masks"][c]
        fph, fpa = frozen_predict_fast(ctx["val"][c], snaps[c], prior_map,
                                       ctx["ave"][c], prod)
        f_ph.append(fph)
        f_pa.append(fpa)
        yh.append(ctx["yh"][c])
        ya.append(ctx["ya"][c])
        w_ph.append(ph[m])
        w_pa.append(pa[m])
    s_f, _, _ = summarize_predictions(np.concatenate(f_ph),
                                      np.concatenate(f_pa),
                                      np.concatenate(yh),
                                      np.concatenate(ya))
    s_w, _, _ = summarize_predictions(np.concatenate(w_ph),
                                      np.concatenate(w_pa),
                                      np.concatenate(yh),
                                      np.concatenate(ya))
    return (s_f["poisson_ll_per_tm"], s_w["poisson_ll_per_tm"])


def wf_train_ll(hyper, train_df, ave, prior_map, prod):
    """Walk-forward Poisson LL/tm over a TRAINING frame (all rows,
    sequential predict-then-update) -- the no-leakage selection
    criterion: the filter's native tracking quality on data before the
    cutoff."""
    ph, pa, _ = kalman_run(train_df, ave, hyper, prior_map, prod["beta"],
                           prod["money_weight"], prod["home_advantage"])
    s, _, _ = summarize_predictions(ph, pa,
                                    train_df["Home_NPG"].values,
                                    train_df["Away_NPG"].values)
    return s["poisson_ll_per_tm"]


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
    say("TEAM RATINGS STATE-SPACE (KALMAN) REFORMULATION -- EXPERIMENT 4")
    say("=" * 60)
    say(f"  data: match_results.csv {data_meta['match_results']['rows']} rows "
        f"(md5 {data_meta['match_results']['md5'][:10]}..., "
        f"{data_meta['match_results']['mtime']}), "
        f"money_priors {data_meta['money_priors']['rows']} rows")
    say(f"  cutoffs: {list(CUTOFFS)} (rolling origin, same as §20-§23)")
    say(f"  fixed at production: {fmt_params(prod)}, "
        f"home_advantage={home_advantage:.9f}")
    say("  state under test: per team per side, log-rating x and variance P;")
    say("    per-match Kalman gain replaces the fixed-alpha multiplicative")
    say("    update entirely; money blend at season boundaries kept (rate")
    say("    space, identical form); Q_within per match played, Q_boundary")
    say("    per season, P_promoted at entry (2017 pool init uses the same),")
    say("    R = r_scale * predicted goals (Gaussian approx to Poisson).")
    say("  hyperparameters tuned (never eyeballed): Q_within x Q_boundary x")
    say(f"    P_promoted x r_scale = {len(Q_WITHIN_GRID)}x{len(Q_BOUNDARY_GRID)}"
        f"x{len(P_PROMOTED_GRID)}x{len(R_SCALE_GRID)} = "
        f"{len(Q_WITHIN_GRID) * len(Q_BOUNDARY_GRID) * len(P_PROMOTED_GRID) * len(R_SCALE_GRID)}"
        f" combinations; global grid (ceiling, in-sample-flattered) AND")
    say("    no-leakage per-cutoff selection (decision number, §23 discipline)")
    say("  tiers: early/late (either team's first 5 matches of the season) and")
    say("    promoted/rest partitions -- diagnostic slices; pooled figures are")
    say("    continuity, never the headline. Frozen primary, wf secondary.")
    say()
    say("PRE-REGISTERED PREDICTION (docstring, fixed 2026-09-04 before any")
    say("harness comparison ran): primary within +/-0.003 of the incumbent")
    say("0.8821, at best +0.001-0.003 if the mechanism helps; any improvement")
    say("concentrated in early rounds (>= 0.005 there would be clear evidence)")
    say("and walk-forward; no reliable promoted-specific gain expected (NOT a")
    say("promoted fix -- the doc's framing); global grid will flatter vs the")
    say("no-leakage selection. Scoring rule: SUPPORTED if no-leakage pooled")
    say("frozen improves > 0.001 with the early-round locus and no wf damage;")
    say("WEAK/PARTIAL if direction right but inside the noise band; FAILED")
    say("otherwise (incl. flattery-only). Four nulls in a row would be a")
    say("convergent signal about the data's ceiling -- real information.")

    # ── STEP 0: correctness before any comparison ───────────────────
    say()
    es_ref = es_calibrated_reference(alpha, ave_full, say)
    if not reduction_check(prod, prior_map, df, ave_full, es_ref, say):
        say()
        say("  STEP 0 FAILED -- aborting (implementation bug risk; nothing")
        say("  below would be trustworthy).")
        return 1

    # ── STEP 1: guard + baseline ────────────────────────────────────
    say()
    say("-" * 60)
    say("STEP 1: VERIFICATION GUARD (incumbent engine) + BASELINE, TIERED")
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
        f"(the Kalman engine has no verbatim reference -- STEP 0 is its")
    say("  correctness story; this guard secures the incumbent baseline)")
    if not guard_ok:
        say("  GUARD FAILED -- aborting.")
        return 1

    base = evaluate(incumbent_runner(prod, prior_map), ctx, prior_map,
                    prod, team_sets)
    for c in CUTOFFS:
        p = base["per"][c]
        say(f"  baseline cutoff {c}: promoted {', '.join(p['promoted'])}")
        for key in ("all", "early", "late", "prom", "rest"):
            t = p["tiers"][key]
            if t["frozen"] is None:
                say(f"    {key:<6}: no held-out matches")
                continue
            say(f"    {key:<6} n={t['n']:>3}: frozen LL/tm "
                f"{t['frozen']['poisson_ll_per_tm']:.4f} | wf "
                f"{t['wf']['poisson_ll_per_tm']:.4f}")
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

    # ── STEP 2: global grid (the ceiling) ───────────────────────────
    say()
    say("-" * 60)
    say("STEP 2: GLOBAL GRID (300 combinations -- the CEILING; selection on")
    say("the scored seasons flatters, §21-style; surface structure only)")
    say("-" * 60)
    grid = list(itertools.product(Q_WITHIN_GRID, Q_BOUNDARY_GRID,
                                  P_PROMOTED_GRID, R_SCALE_GRID))
    t_grid = time.perf_counter()
    rows = []
    for (q_w, q_b, p_pr, r_s) in grid:
        hyper = {"q_within": q_w, "q_boundary": q_b,
                 "p_promoted": p_pr, "r_scale": r_s}
        f_ll, w_ll = grid_score(hyper, ctx, prior_map, prod)
        rows.append({"q_within": q_w, "q_boundary": q_b,
                     "p_promoted": p_pr, "r_scale": r_s,
                     "pooled_frozen_ll": f_ll, "pooled_wf_ll": w_ll})
    grid_time = time.perf_counter() - t_grid
    f_ref, w_ref = grid_score(es_ref, ctx, prior_map, prod)
    b_f = base["pooled"]["all"]["frozen"]["poisson_ll_per_tm"]
    b_w = base["pooled"]["all"]["wf"]["poisson_ll_per_tm"]
    say(f"  scored {len(rows)} combinations in {grid_time:.0f}s")
    say()
    say(f"  incumbent baseline:            frozen {b_f:.4f} | wf {b_w:.4f}")
    say(f"  ES-calibrated reference (KF as incumbent): frozen {f_ref:.4f} | "
        f"wf {w_ref:.4f}")
    say(f"    anchor: expected within ~0.005 of the incumbent -> "
        f"{'OK' if abs(f_ref - b_f) < 0.005 else 'FLAG -- investigate before trusting the grid'} "
        f"(delta {f_ref - b_f:+.4f})")
    say()
    ranked = sorted(rows, key=lambda r: r["pooled_frozen_ll"])
    say("  top 10 by pooled frozen LL/tm (lower = better):")
    say(f"    {'q_within':>9}{'q_boundary':>11}{'p_prom':>8}{'r_scale':>8}"
        f"{'frozen':>9}{'wf':>9}")
    for r in ranked[:10]:
        say(f"    {r['q_within']:>9.4f}{r['q_boundary']:>11.3f}"
            f"{r['p_promoted']:>8.2f}{r['r_scale']:>8.1f}"
            f"{r['pooled_frozen_ll']:>9.4f}{r['pooled_wf_ll']:>9.4f}")
    grid_best = ranked[0]
    say()
    say("  surface structure (best pooled frozen per level):")
    for name, vals, idx in (("q_within", Q_WITHIN_GRID, 0),
                            ("q_boundary", Q_BOUNDARY_GRID, 1),
                            ("p_promoted", P_PROMOTED_GRID, 2),
                            ("r_scale", R_SCALE_GRID, 3)):
        parts = []
        for v in vals:
            sub = [r for r in rows if r[name] == v]
            best_v = min(r["pooled_frozen_ll"] for r in sub)
            parts.append(f"{v:g}: {best_v:.4f}")
        say(f"    {name:<11} " + " | ".join(parts))
    n_beat = sum(1 for r in rows if r["pooled_frozen_ll"] < b_f)
    say(f"  combinations beating the incumbent on pooled frozen: "
        f"{n_beat}/{len(rows)}")

    # ── STEP 3: no-leakage selection (the decision number) ─────────
    say()
    say("-" * 60)
    say("STEP 3: NO-LEAKAGE SELECTION -- per cutoff, hyperparameters chosen")
    say("by walk-forward LL on Season < C ONLY, then applied to score C")
    say("-" * 60)
    t_sel = time.perf_counter()
    sel_hyper = {}
    sel_detail = {}
    for c in CUTOFFS:
        train = df[df["Season"] < c]
        best = None
        for (q_w, q_b, p_pr, r_s) in grid:
            hyper = {"q_within": q_w, "q_boundary": q_b,
                     "p_promoted": p_pr, "r_scale": r_s}
            ll = wf_train_ll(hyper, train, ctx["ave"][c], prior_map, prod)
            if best is None or ll < best[0]:
                best = (ll, hyper)
        sel_hyper[c] = best[1]
        sel_detail[c] = best[0]
        say(f"  cutoff {c}: selected {best[1]} (train wf LL/tm "
            f"{best[0]:.4f}; train rows {len(train)})")
    say(f"  selection wall time: {time.perf_counter() - t_sel:.0f}s")

    say()
    say("  HONEST EVALUATION of the no-leakage selections vs BASELINE:")
    sel = evaluate(kalman_runner(sel_hyper, prior_map, prod), ctx,
                   prior_map, prod, team_sets)
    for c in CUTOFFS:
        b, v = base["per"][c]["tiers"], sel["per"][c]["tiers"]
        h = sel_hyper[c]
        say(f"  cutoff {c} (Q_w {h['q_within']:.4f}, Q_b {h['q_boundary']:.3f},"
            f" P_pr {h['p_promoted']:.2f}, R_s {h['r_scale']:.1f}; promoted: "
            f"{', '.join(base['per'][c]['promoted'])}):")
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
        vf = sel["pooled"][key]["frozen"]
        if bf is None or vf is None:
            say(f"    {key:<6}: no held-out matches")
            continue
        bw = base["pooled"][key]["wf"]
        vw = sel["pooled"][key]["wf"]
        d_f = vf["poisson_ll_per_tm"] - bf["poisson_ll_per_tm"]
        d_w = vw["poisson_ll_per_tm"] - bw["poisson_ll_per_tm"]
        say(f"    {key:<6}: frozen {bf['poisson_ll_per_tm']:.4f} -> "
            f"{vf['poisson_ll_per_tm']:.4f} ({d_f:+.4f}) | wf "
            f"{bw['poisson_ll_per_tm']:.4f} -> {vw['poisson_ll_per_tm']:.4f} "
            f"({d_w:+.4f}) | MAE {bf['mae']:.4f} -> {vf['mae']:.4f} | CS LL "
            f"{bf['cs_ll']:.4f} -> {vf['cs_ll']:.4f} (n={bf['n_matches']})")

    # ── STEP 4: flattery check + locus summary ──────────────────────
    say()
    say("-" * 60)
    say("STEP 4: FLATTERY CHECK (global winner, no-leakage PRIMARY) + LOCUS")
    say("-" * 60)
    glob_hyper = {"q_within": grid_best["q_within"],
                  "q_boundary": grid_best["q_boundary"],
                  "p_promoted": grid_best["p_promoted"],
                  "r_scale": grid_best["r_scale"]}
    glob = evaluate(kalman_runner({c: glob_hyper for c in CUTOFFS}, prior_map,
                                  prod), ctx, prior_map, prod, team_sets)
    g_f = glob["pooled"]["all"]["frozen"]["poisson_ll_per_tm"] - b_f
    s_f = sel["pooled"]["all"]["frozen"]["poisson_ll_per_tm"] - b_f
    say(f"  pooled frozen delta vs incumbent: global-grid winner {g_f:+.4f}"
        f" vs no-leakage selection {s_f:+.4f} (flattery "
        f"{g_f - s_f:+.4f})")
    say("    (pre-registered: the global number flatters; the no-leakage")
    say("     number is the decision number)")
    e_b = base["pooled"]["early"]["frozen"]["poisson_ll_per_tm"]
    e_s = sel["pooled"]["early"]["frozen"]["poisson_ll_per_tm"]
    l_b = base["pooled"]["late"]["frozen"]["poisson_ll_per_tm"]
    l_s = sel["pooled"]["late"]["frozen"]["poisson_ll_per_tm"]
    if l_b is not None and l_s is not None:
        say(f"  locus: early {e_b:.4f} -> {e_s:.4f} "
            f"({e_s - e_b:+.4f}) | late {l_b:.4f} -> {l_s:.4f} "
            f"({l_s - l_b:+.4f})")
    else:
        say(f"  locus: early {e_b:.4f} -> {e_s:.4f} "
            f"({e_s - e_b:+.4f}) | late: no held-out matches")

    # ── STEP 5: scorecard + verdict ─────────────────────────────────
    say()
    say("-" * 60)
    say("SCORECARD vs PRE-REGISTERED PREDICTION")
    say("-" * 60)
    s_wf = sel["pooled"]["all"]["wf"]["poisson_ll_per_tm"] - b_w
    improvement = -s_f
    say(f"  no-leakage pooled frozen: {s_f:+.4f} LL/tm "
        f"(improvement {improvement:.4f}; noise band {NOISE_BAND_LL})")
    say(f"  early-round delta {e_s - e_b:+.4f} vs late-round delta "
        f"{(l_s - l_b) if (l_b is not None and l_s is not None) else float('nan'):+.4f}")
    say(f"  pooled walk-forward: {s_wf:+.4f} LL/tm")
    pr_b = base["pooled"]["prom"]["frozen"]["poisson_ll_per_tm"]
    pr_s = sel["pooled"]["prom"]["frozen"]["poisson_ll_per_tm"]
    say(f"  promoted-matches delta (secondary, NOT the framing): "
        f"{pr_s - pr_b:+.4f} LL/tm")
    say(f"  flattery: global {g_f:+.4f} vs no-leakage {s_f:+.4f}")

    say()
    say("-" * 60)
    say("VERDICT")
    say("-" * 60)
    d_early = e_s - e_b
    d_late = (l_s - l_b) if (l_b is not None and l_s is not None) else 0.0
    if improvement > NOISE_BAND_LL and d_early < d_late and s_wf < 0.001:
        verdict = "supported"
        say("  SUPPORTED: adaptive uncertainty improves prediction quality")
        say(f"  beyond the noise band ({improvement:.4f} LL/tm), with the")
        say("  pre-registered early-round locus and no walk-forward damage.")
    elif improvement > 0 and d_early < d_late:
        verdict = "weak_partial"
        say("  WEAK/PARTIAL: direction matches but the pooled improvement is")
        say("  inside the noise band -- §22's lesson applies: mechanism")
        say("  plausibility is not predictive evidence.")
    else:
        verdict = "failed"
        say("  FAILED: no improvement, degradation, or the gain exists only")
        say("  in the flattered global-grid number.")
    if verdict == "failed":
        say()
        say("  FOUR CONSECUTIVE NULLS (tuning §21, rebalancing §22, promoted")
        say("  discount §23, structural Kalman §24 -- three parametric, one")
        say("  structural): a convergent signal that the incumbent is close")
        say("  to this data's actual ceiling. Per the design doc, that is")
        say("  real information about the model's limits, not just another")
        say("  negative result to move past.")
    say()
    say("  update_team_ratings.py UNCHANGED (measured, not shipped);")
    say("  adoption is a separate review decision. Shipping candidate")
    say("  hyperparameters if adopted: the no-leakage selections above.")

    # ── Artifacts ───────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_rows = rows + [{"q_within": es_ref["q_within"],
                         "q_boundary": es_ref["q_boundary"],
                         "p_promoted": es_ref["p_promoted"],
                         "r_scale": es_ref["r_scale"],
                         "pooled_frozen_ll": f_ref,
                         "pooled_wf_ll": w_ref,
                         "note": "ES-calibrated reference"},
                        {"q_within": None, "q_boundary": None,
                         "p_promoted": None, "r_scale": None,
                         "pooled_frozen_ll": b_f,
                         "pooled_wf_ll": b_w,
                         "note": "incumbent baseline"}]
    pd.DataFrame(grid_rows).to_csv(OUT_DIR / "kalman_grid.csv", index=False)
    report = OUT_DIR / "kalman_report.txt"
    report.write_text("\n".join(_REPORT_LINES) + "\n", encoding="utf-8")
    summary = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "data": data_meta,
        "params_fixed": prod,
        "design": {
            "state": "per team per side: log-rating x, variance P",
            "observation": "num = NPxG_adj*beta + NPG*(1-beta), R = "
                           "r_scale * predicted goals (Gaussian approx to "
                           "Poisson)",
            "update": "diagonal 2-state per signal; exact marginals, "
                      "cross-covariance discarded; disjoint signal pairs",
            "boundary": "money blend kept (rate space) + Q_boundary; "
                        "promoted at P_promoted; 2017 init at P_promoted",
            "noise_band_ll": NOISE_BAND_LL,
            "early_n": EARLY_N,
        },
        "es_reference": {"hyper": es_ref,
                         "pooled_frozen_ll": f_ref,
                         "pooled_wf_ll": w_ref,
                         "anchor_ok": bool(abs(f_ref - b_f) < 0.005)},
        "grid": {"n_combos": len(rows),
                 "best": grid_best,
                 "n_beating_incumbent": n_beat},
        "no_leakage_selection": {str(c): {"hyper": sel_hyper[c],
                                          "train_wf_ll": sel_detail[c]}
                                 for c in CUTOFFS},
        "baseline_pooled": {k: base["pooled"][k] for k in
                            ("all", "early", "late", "prom", "rest")},
        "selected_pooled": {k: sel["pooled"][k] for k in
                            ("all", "early", "late", "prom", "rest")},
        "selected_per_cutoff": {
            str(c): sel["per"][c]["tiers"] for c in CUTOFFS},
        "global_winner_pooled": glob["pooled"]["all"],
        "scorecard": {
            "no_leakage_pooled_frozen_delta": s_f,
            "improvement": improvement,
            "early_delta": e_s - e_b,
            "late_delta": (l_s - l_b) if (l_b is not None
                                          and l_s is not None) else None,
            "wf_delta": s_wf,
            "prom_delta": pr_s - pr_b,
            "flattery": g_f - s_f,
        },
        "verdict": verdict,
        "update_team_ratings_changed": False,
    }
    (OUT_DIR / "kalman_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    say()
    say(f"  wall time: {time.perf_counter() - t0:.0f}s")
    say(f"  artifacts: {report.name}, kalman_summary.json, kalman_grid.csv "
        f"-> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
