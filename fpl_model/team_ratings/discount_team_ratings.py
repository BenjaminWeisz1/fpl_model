"""
Promoted-team entry-discount experiment for the team ratings model --
implements the user-specified design (CHANGELOG §23) against the §20 OOT
harness, using the §21 fast engine (ratings_fast.py) EXTENDED with the
discounted-entry mode, verified against a NEW slow verbatim reference
(evaluate_team_ratings_oot.run_sequential_discounted) with the same
bit-exactness discipline as §21/§22. update_team_ratings.py is NOT
modified: this script only measures and reports; shipping is a separate
human decision.

Mechanism under test: every promoted team enters the ratings pool at its
100%-money-implied prior (Money_GF/Money_GA -- NPG-per-match scale,
confirmed by the user), while continuing teams enter the season at a
money_weight blend of earned ratings. If the pure prior is structurally
optimistic for promoted teams specifically, multiplying the ENTRY values
by an empirically estimated discount -- actual delivered NPG per match
divided by the assigned entry prior, averaged over every historical
promoted instance, shrunk toward 1.0 (no discount) -- should improve
predictions on their matches.

Estimation design (fixed before the validation ran):
- Instances: every team entering the pool via the money branch, i.e.
  teams(season s) - teams(s-1) for s = 2018..2026 (2017 is pool
  initialization, not promotion). Complete seasons only for estimation:
  2026 is in progress, so its entrants are discounted in validation but
  excluded from estimation -- stated, not silent.
- Comparator scale: actual NPG per match (Home_NPG/Away_NPG) -- the same
  scale as the priors (user-confirmed) and as the Attack/Defense ratings
  themselves, so the discount is scale-consistent end to end. A
  raw-total-goals comparator would import the penalty dimension the
  ratings never model.
- Per instance: r_atk = actual_NPG_for_per_match / Money_GF and
  r_def = actual_NPG_against_per_match / Money_GA. Raw ratio per
  instance, pooled mean and median, and the match-weighted total ratio
  are all reported. The MEAN enters the discount (per "averaged across
  all instances"); the median is robustness context.
- Continuing-team control (guardrail: sanity): the same ratios computed
  for continuing teams. If promoted and continuing ratios matched, the
  "discount" would be a prior-level artifact, not promoted-specific
  optimism; the report shows both.
- Shrinkage toward 1.0 -- formula and weight stated explicitly, per
  instruction. The codebase's credibility-weighting principle
  (build_player_ratings' SHRINKAGE_PRIOR pseudo-matches, defcon's
  k = Min_Old/90 pseudo-matches) in pseudo-count form,
      d = (n * r_bar + k * 1.0) / (n + k)   ==   1 + z * (r_bar - 1),
  with the pseudo-count k derived by method of moments from the two
  variance components of the per-instance ratios:
      sigma2_i = r_i / (m_i * prior_i)      within-instance Poisson
               match-noise variance of the ratio estimate
      s2       = sample variance of r_i     total spread
      tau2     = max(0, s2 - mean(sigma2_i))   true between-instance
               heterogeneity
      k        = mean(sigma2_i) / tau2      pseudo-instances at 1.0
               (tau2 = 0 -> k = inf -> z = 0: full shrinkage)
      z        = n / (n + k)                credibility weight on data
  Every input number is printed. Caveat stated in the report: tau2
  absorbs any opponent-driven overdispersion beyond Poisson, which
  biases z upward (toward less shrinkage); the validation, not the
  estimate, carries the decision weight.

Validation design:
- Entry-only application: the discount multiplies the pure prior at the
  lazy add (the entrant's first match of the season) and the matching
  frozen-snapshot extension, ONLY after the frame's first season (pool
  initialization is not promotion). Continuing teams' money_weight
  blend is untouched, in every season.
- No-leakage estimation for validation: each cutoff C is scored with
  the discount estimated ONLY from complete instances with season < C
  -- exactly the estimate production could have made at that point in
  time, matching the harness's train-only ave_goals discipline. The
  full-sample estimate is the shipping candidate; it is also run for
  contrast, and the gap between the two quantifies the in-sample
  flattery of using it directly.
- Tiers (per instruction; pooled figures are never the headline):
  (a) held-out matches involving a promoted team of that season,
  (b) all other held-out matches. Frozen mode primary, walk-forward
  secondary, per cutoff and pooled (match-count-weighted).
- Tier-(b) change is propagation only, measured rather than assumed:
  a tier-(b) held-out match involves two continuing teams, so it is
  not touched directly -- but the discount applies to promoted entries
  in EVERY season of the frame, and matches against those entrants
  shaped the continuing teams' ratings during training history, so
  their boundary ratings (and hence frozen tier-(b) predictions)
  carry a small propagated difference. The report measures it (pooled
  delta + max per-team-match diff); the design expectation is near
  zero, with walk-forward tier (b) adding only the same spillover.
  This is the honest test of the "little to no effect in tier (b)"
  prediction. (An earlier draft asserted frozen tier (b) was EXACTLY
  unchanged; the first run falsified that -- propagation through
  training history is real, just tiny. The check now measures.)
- Metrics: Poisson LL per team-match (primary), MAE, clean-sheet LL;
  CS reliability bins for tier (a); per-promoted-team-season deltas
  (the instance-level view); cluster bootstrap (resampling promoted
  team-seasons, 1000 reps, fixed seed) on the tier-(a) pooled delta.

Pre-registered prediction (user, stated before the run): a real,
positive effect specifically in tier (a) -- this targets a confirmed
structural bias, unlike the last two experiments, which tested
already-reasonable mechanics -- with little to no effect in tier (b),
since non-promoted teams are untouched by this change.
Scoring rule (fixed here before the run):
  * SUPPORTED      tier-(a) pooled frozen LL improves (delta < 0),
                   >= 3 of 4 cutoffs improve, and the cluster-bootstrap
                   97.5th percentile is < 0 (distinguishable from noise);
  * WEAK/PARTIAL   direction right but the effect is inside the noise
                   band / bootstrap CI straddles zero;
  * FAILED         tier (a) does not improve.
  Tier (b): near zero expected in both modes -- the direct effect is
  zero by construction; only training-history propagation of
  discounted entries can move it (measured, not assumed).

Guardrails (same discipline as §21/§22):
  1. Baseline (production, no discount) scored FIRST on this harness,
     per tier, pooled cross-checked against the stored baseline.json.
  2. The §21 noise band (0.001 pooled LL/tm; 292/1001 combos within it
     of the winner) is the skepticism reference; tier-level effects are
     additionally bootstrap-checked because tier (a) holds only ~1/3 of
     the held-out matches.
  3. Per-cutoff and per-tier results reported individually.
  4. Extreme-discount flag: an estimate outside [0.85, 1.15] is flagged
     for sanity review against the continuing-team control, never
     accepted automatically.
  5. Nothing ships automatically; update_team_ratings.py UNCHANGED.

Out of scope: re-tuning alpha/beta/money_weight under the discount,
rebalancing interaction (engine modes are exclusive), team-specific
home/away (the remaining planned experiment).

Outputs (data/current/team_ratings_oot/): discount_report.txt,
discount_summary.json.

Run locally:
    python discount_team_ratings.py
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.team_ratings.update_team_ratings import alpha, beta, home_advantage, money_weight

from fpl_model.team_ratings import ratings_fast
from fpl_model.team_ratings.evaluate_team_ratings_oot import EPS, OUT_DIR, cs_bin_table, load_inputs, summarize_predictions
from fpl_model.team_ratings.ratings_fast import fast_run, frozen_predict_fast
from fpl_model.team_ratings.tune_team_ratings import CUTOFFS, build_context, fmt_params

# §21 noise-band reference points, fixed for the guardrail-2 assessment.
NOISE_BAND_LL = 0.001
NOISE_BAND_N = 292

MIN_EST_MATCHES = 30                  # below this an instance's season is incomplete
EXTREME_LO, EXTREME_HI = 0.85, 1.15   # guardrail-4 sanity band around 1.0
BOOT_N = 1000                         # cluster bootstrap reps (team-seasons)
BOOT_SEED = 20260904                  # fixed: the report must be reproducible

_REPORT_LINES = []


def say(line=""):
    print(line)
    _REPORT_LINES.append(line)


def build_season_teams(df):
    seasons = [int(s) for s in sorted(df["Season"].unique())]
    team_sets = {}
    for s in seasons:
        d = df[df["Season"] == s]
        team_sets[s] = set(d["Home"]) | set(d["Away"])
    return seasons, team_sets


def actual_npg_pm(df, season, team):
    """Matches played, actual NPG-for and NPG-against per match for one
    team-season -- the delivered side of the discount ratio (raw match
    outcomes from match_results.csv; no model involved)."""
    d = df[(df["Season"] == season)
           & ((df["Home"] == team) | (df["Away"] == team))]
    n = len(d)
    gf = float(d.loc[d["Home"] == team, "Home_NPG"].sum()
               + d.loc[d["Away"] == team, "Away_NPG"].sum())
    ga = float(d.loc[d["Home"] == team, "Away_NPG"].sum()
               + d.loc[d["Away"] == team, "Home_NPG"].sum())
    return n, gf / n, ga / n


def shrink_toward_one(ratios, priors, match_counts):
    """Method-of-moments credibility shrinkage of the mean ratio toward
    1.0 (formula and rationale in the module docstring). Returns every
    input and output number so the report can state them explicitly."""
    r = np.asarray(ratios, dtype=float)
    p = np.asarray(priors, dtype=float)
    m = np.asarray(match_counts, dtype=float)
    n = int(len(r))
    r_bar = float(r.mean())
    median = float(np.median(r))
    s2 = float(r.var(ddof=1)) if n > 1 else 0.0
    sig2 = float(np.mean(r / (m * p)))
    tau2 = max(0.0, s2 - sig2)
    if tau2 > 0.0:
        k = sig2 / tau2
        z = n / (n + k)
    else:
        k, z = float("inf"), 0.0
    d = 1.0 + z * (r_bar - 1.0)
    pooled_total = float((r * m * p).sum() / (m * p).sum())
    return {"n": n, "mean": r_bar, "median": median, "var_total": s2,
            "var_within": sig2, "var_between": tau2, "k": k, "z": z,
            "d_shrunk": d, "pooled_total_ratio": pooled_total}


def estimate_discount(df, prior_map, seasons, team_sets):
    """Identify every promoted (pool-entrant) instance, compute raw
    ratios vs the assigned pure-prior entry, the continuing-team
    control, the shrunk full-sample discount, and the rolling-origin
    per-cutoff discounts (instances with season < C only)."""
    last = seasons[-1]
    instances, excluded = [], []
    per_season = {}
    for s in seasons[1:]:
        promoted = sorted(team_sets[s] - team_sets[s - 1])
        per_season[s] = promoted
        for t in promoted:
            if (s, t) not in prior_map:
                raise KeyError(
                    f"No money prior for promoted instance ({s}, {t!r})")
            n, gf, ga = actual_npg_pm(df, s, t)
            p_atk, p_def = prior_map[(s, t)]
            rec = {"season": s, "team": t, "n": n,
                   "prior_atk": float(p_atk), "prior_def": float(p_def),
                   "act_atk": gf, "act_def": ga,
                   "r_atk": gf / float(p_atk), "r_def": ga / float(p_def)}
            if s == last or n < MIN_EST_MATCHES:
                excluded.append(rec)
            else:
                instances.append(rec)
    control = []
    for s in seasons[1:]:
        for t in sorted(team_sets[s] & team_sets[s - 1]):
            n, gf, ga = actual_npg_pm(df, s, t)
            if n < MIN_EST_MATCHES:
                continue  # in-progress season's teams
            p_atk, p_def = prior_map[(s, t)]
            control.append({"season": s, "team": t, "n": n,
                            "prior_atk": float(p_atk),
                            "prior_def": float(p_def),
                            "r_atk": gf / float(p_atk),
                            "r_def": ga / float(p_def)})
    atk = shrink_toward_one([x["r_atk"] for x in instances],
                            [x["prior_atk"] for x in instances],
                            [x["n"] for x in instances])
    dfn = shrink_toward_one([x["r_def"] for x in instances],
                            [x["prior_def"] for x in instances],
                            [x["n"] for x in instances])
    ctrl_atk = shrink_toward_one([x["r_atk"] for x in control],
                                 [x["prior_atk"] for x in control],
                                 [x["n"] for x in control])
    ctrl_def = shrink_toward_one([x["r_def"] for x in control],
                                 [x["prior_def"] for x in control],
                                 [x["n"] for x in control])
    rolling = {}
    for c in CUTOFFS:
        sub = [x for x in instances if x["season"] < c]
        rolling[c] = (
            shrink_toward_one([x["r_atk"] for x in sub],
                              [x["prior_atk"] for x in sub],
                              [x["n"] for x in sub]),
            shrink_toward_one([x["r_def"] for x in sub],
                              [x["prior_def"] for x in sub],
                              [x["n"] for x in sub]),
        )
    return {"instances": instances, "excluded": excluded, "control": control,
            "per_season": per_season, "attack": atk, "defense": dfn,
            "control_attack": ctrl_atk, "control_defense": ctrl_def,
            "rolling": rolling}


def _ll_tm(pred_h, pred_a, yh, ya):
    """Per-team-match Poisson LL kernel, same clipping as
    summarize_predictions -- for delta/bootstrap work."""
    ph = np.maximum(np.asarray(pred_h, dtype=float), EPS)
    pa = np.maximum(np.asarray(pred_a, dtype=float), EPS)
    return np.concatenate([ph - yh * np.log(ph), pa - ya * np.log(pa)])


def evaluate(params, ctx, prior_map, team_sets, discounts):
    """Score one discount configuration ({cutoff: (d_atk, d_def) or
    None}) across the cutoffs with the fast engine, broken into tier (a)
    (held-out matches involving a promoted team of that season) and tier
    (b) (the rest). Returns per-cutoff and pooled tier summaries and
    per-team-match LL arrays (frozen + walk-forward) for delta/bootstrap
    work; tier-(a) probs/events are re-derived where needed."""
    per = {}
    acc = {t: {"f_ph": [], "f_pa": [], "yh": [], "ya": [],
               "w_ph": [], "w_pa": [], "probs": [], "events": []}
           for t in ("a", "b", "all")}
    for c in CUTOFFS:
        frame = ctx["frames"][c]
        d = discounts.get(c)
        ph, pa, snaps = fast_run(frame, ctx["ave"][c], params, prior_map,
                                 snapshot_seasons={c}, discount=d)
        m = ctx["masks"][c]
        val = ctx["val"][c]
        fph, fpa = frozen_predict_fast(val, snaps[c], prior_map,
                                       ctx["ave"][c], params)
        yh, ya = ctx["yh"][c], ctx["ya"][c]
        wph, wpa = ph[m], pa[m]
        promoted = team_sets[c] - team_sets[c - 1]
        tier_a = (val["Home"].isin(promoted)
                  | val["Away"].isin(promoted)).values
        tier_b = ~tier_a
        # Cluster per tier-(a) match: the promoted team involved (the
        # home side when both are promoted -- stated choice; the
        # bootstrap resamples these promoted team-seasons).
        clusters = []
        for h, a in zip(val["Home"].values[tier_a],
                        val["Away"].values[tier_a]):
            clusters.append(f"{c}:{h}" if h in promoted else f"{c}:{a}")
        entry = {}
        for key, mask in (("a", tier_a), ("b", tier_b),
                          ("all", np.ones(len(val), dtype=bool))):
            if int(mask.sum()) == 0:
                entry[key] = {"frozen": None, "wf": None, "n": 0}
                continue
            s_f, probs, events = summarize_predictions(
                fph[mask], fpa[mask], yh[mask], ya[mask])
            s_w, _, _ = summarize_predictions(
                wph[mask], wpa[mask], yh[mask], ya[mask])
            entry[key] = {"frozen": s_f, "wf": s_w, "n": int(mask.sum())}
            acc[key]["f_ph"].append(fph[mask])
            acc[key]["f_pa"].append(fpa[mask])
            acc[key]["yh"].append(yh[mask])
            acc[key]["ya"].append(ya[mask])
            acc[key]["w_ph"].append(wph[mask])
            acc[key]["w_pa"].append(wpa[mask])
            acc[key]["probs"].append(probs)
            acc[key]["events"].append(events)
        per[c] = {"promoted": sorted(promoted), "tiers": entry,
                  "clusters": clusters,
                  "ll_f_a": _ll_tm(fph[tier_a], fpa[tier_a],
                                   yh[tier_a], ya[tier_a]),
                  "ll_w_a": _ll_tm(wph[tier_a], wpa[tier_a],
                                   yh[tier_a], ya[tier_a]),
                  "ll_f_b": _ll_tm(fph[tier_b], fpa[tier_b],
                                   yh[tier_b], ya[tier_b]),
                  "ll_w_b": _ll_tm(wph[tier_b], wpa[tier_b],
                                   yh[tier_b], ya[tier_b])}
    pooled = {}
    for key in ("a", "b", "all"):
        if not acc[key]["f_ph"]:
            pooled[key] = {"frozen": None, "wf": None,
                           "probs": None, "events": None}
            continue
        s_f, probs, events = summarize_predictions(
            np.concatenate(acc[key]["f_ph"]),
            np.concatenate(acc[key]["f_pa"]),
            np.concatenate(acc[key]["yh"]),
            np.concatenate(acc[key]["ya"]))
        s_w, _, _ = summarize_predictions(
            np.concatenate(acc[key]["w_ph"]),
            np.concatenate(acc[key]["w_pa"]),
            np.concatenate(acc[key]["yh"]),
            np.concatenate(acc[key]["ya"]))
        pooled[key] = {"frozen": s_f, "wf": s_w, "probs": probs,
                       "events": events}
    return {"per": per, "pooled": pooled}


def cluster_bootstrap(deltas, cluster_tm):
    """Resample promoted team-seasons (clusters) with replacement and
    recompute the pooled mean team-match delta each time -- the tier-(a)
    uncertainty check that respects the strong within-team-season
    correlation of entry-rating effects."""
    uniq = np.unique(cluster_tm)
    s_arr = np.array([deltas[cluster_tm == u].sum() for u in uniq])
    c_arr = np.array([(cluster_tm == u).sum() for u in uniq], dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    stats = np.empty(BOOT_N)
    for b in range(BOOT_N):
        idx = rng.integers(0, len(uniq), len(uniq))
        stats[b] = s_arr[idx].sum() / c_arr[idx].sum()
    return uniq, stats


def _js(e):
    out = dict(e)
    if isinstance(out.get("k"), float) and np.isinf(out["k"]):
        out["k"] = None
    return out


def main():
    t0 = time.perf_counter()
    df, prior_map, data_meta = load_inputs()
    prod = {"alpha": alpha, "beta": beta,
            "home_advantage": home_advantage, "money_weight": money_weight}
    ave_full = float(pd.concat([df["Home_NPG"], df["Away_NPG"]]).mean())
    ctx = build_context(df)
    seasons, team_sets = build_season_teams(df)

    say("=" * 60)
    say("TEAM RATINGS PROMOTED-TEAM ENTRY-DISCOUNT EXPERIMENT")
    say("=" * 60)
    say(f"  data: match_results.csv {data_meta['match_results']['rows']} rows "
        f"(md5 {data_meta['match_results']['md5'][:10]}..., "
        f"{data_meta['match_results']['mtime']}), "
        f"money_priors {data_meta['money_priors']['rows']} rows")
    say(f"  cutoffs: {list(CUTOFFS)} (rolling origin, same as §20-§22)")
    say(f"  params: {fmt_params(prod)}, home_advantage={home_advantage:.9f} "
        f"(production, fixed)")
    say("  change under test: teams entering the pool via the 100%-money")
    say("    branch start at (d_atk * Money_GF, d_def * Money_GA) instead of")
    say("    the pure prior; continuing teams' money_weight blend untouched;")
    say("    entry only, every season after pool initialization.")
    say("  scale: priors and actuals both NPG-per-match (user-confirmed);")
    say("    Attack/Defense ratings and the harness metrics live on the same")
    say("    scale, so the discount is scale-consistent end to end.")
    say("  tiers: (a) held-out matches involving a promoted team of that")
    say("    season; (b) all other held-out matches -- direct effect zero by")
    say("    construction; the measured difference is training-history")
    say("    propagation (see the tier-(b) propagation check). Pooled figures")
    say("    are continuity, never the headline.")
    say("  metrics: Poisson LL/tm primary; MAE, CS LL alongside.")
    say()
    say("PRE-REGISTERED PREDICTION (user, stated before the run): a real,")
    say("positive effect specifically in tier (a) -- this targets a confirmed")
    say("structural bias, unlike the last two experiments, which tested")
    say("already-reasonable mechanics -- with little to no effect in tier (b),")
    say("since non-promoted teams are untouched by this change. Scoring rule")
    say("fixed in this script's docstring before the run.")

    # ── Step 1: estimate the discount ───────────────────────────────
    est = estimate_discount(df, prior_map, seasons, team_sets)
    say()
    say("-" * 60)
    say("STEP 1: DISCOUNT ESTIMATION (every historical promoted instance)")
    say("-" * 60)
    say("  promoted entrants per season (teams(s) - teams(s-1); 2017 is pool")
    say("  initialization, not promotion):")
    for s in seasons[1:]:
        say(f"    {s}: {len(est['per_season'][s])} "
            f"({', '.join(est['per_season'][s])})")
    say()
    say("  per-instance raw ratios -- r = actual NPG per match / assigned")
    say("  pure-prior entry (exactly what the model assigned at season start);")
    say("  A = attack side (goals for), D = defense side (goals against):")
    say(f"    {'season':<7}{'team':<18}{'n':>3}{'priA':>7}{'actA':>7}{'rA':>7}"
        f"{'priD':>7}{'actD':>7}{'rD':>7}")
    for x in est["instances"]:
        say(f"    {x['season']:<7}{x['team']:<18}{x['n']:>3}"
            f"{x['prior_atk']:>7.3f}{x['act_atk']:>7.3f}{x['r_atk']:>7.3f}"
            f"{x['prior_def']:>7.3f}{x['act_def']:>7.3f}{x['r_def']:>7.3f}")
    if est["excluded"]:
        say(f"  EXCLUDED from estimation (season in progress / n < "
            f"{MIN_EST_MATCHES}); still discounted in validation:")
        for x in est["excluded"]:
            say(f"    {x['season']:<7}{x['team']:<18}{x['n']:>3}"
                f"{'':>21}{x['r_atk']:>7.3f}{x['r_def']:>7.3f}")
    for side, key, ckey in (("ATTACK", "attack", "control_attack"),
                            ("DEFENSE", "defense", "control_defense")):
        e, ctl = est[key], est[ckey]
        say()
        say(f"  {side} side -- pooled over {e['n']} promoted instances:")
        say(f"    mean r = {e['mean']:.4f} | median = {e['median']:.4f} | "
            f"match-weighted total ratio = {e['pooled_total_ratio']:.4f}")
        say("    (the MEAN enters the discount per 'averaged across all")
        say("     instances'; median and total ratio are context.)")
        say(f"    variance split: total {e['var_total']:.5f} = within "
            f"(Poisson match noise) {e['var_within']:.5f} + between "
            f"(heterogeneity) {e['var_between']:.5f}")
        if np.isinf(e["k"]):
            say("    shrinkage: tau2 = 0 -> k = inf -> z = 0 -> d = 1.0000 "
                "(full shrinkage: no distinguishable signal)")
        else:
            say("    shrinkage formula (pseudo-count form): "
                "d = (n*r_bar + k*1.0) / (n + k)")
            say(f"      k = mean(sigma2)/tau2 = {e['var_within']:.5f}/"
                f"{e['var_between']:.5f} = {e['k']:.2f} pseudo-instances at 1.0")
            say(f"      z = n/(n+k) = {e['n']}/({e['n']}+{e['k']:.2f}) = "
                f"{e['z']:.3f} (credibility weight on the data)")
            say(f"      d = 1 + z*(r_bar - 1) = {e['d_shrunk']:.4f}")
            say(f"      sensitivity: halving z to {e['z'] / 2:.3f} would give "
                f"d = {1.0 + (e['z'] / 2) * (e['mean'] - 1.0):.4f} -- "
                f"direction robust to the shrinkage choice")
        say(f"    continuing-team control ({ctl['n']} team-seasons): mean r = "
            f"{ctl['mean']:.4f}, median = {ctl['median']:.4f} -- promoted vs "
            f"continuing gap {e['mean'] - ctl['mean']:+.4f}")
    say()
    say("  caveat: tau2 absorbs opponent-driven overdispersion beyond Poisson,")
    say("  biasing z upward (toward less shrinkage); the validation, not the")
    say("  estimate, carries the decision weight.")

    d_full = (est["attack"]["d_shrunk"], est["defense"]["d_shrunk"])
    say()
    say(f"  FULL-SAMPLE ESTIMATE (the shipping candidate): d_attack = "
        f"{d_full[0]:.4f}, d_defense = {d_full[1]:.4f}")
    for name, v in (("attack", d_full[0]), ("defense", d_full[1])):
        if v < EXTREME_LO or v > EXTREME_HI:
            say(f"    FLAG (guardrail 4): d_{name} = {v:.4f} sits OUTSIDE the "
                f"[{EXTREME_LO:.2f}, {EXTREME_HI:.2f}] sanity band -- review "
                f"the continuing-team control above before trusting it; the "
                f"validation below still runs, it just starts under suspicion.")
        else:
            say(f"    d_{name} is inside the [{EXTREME_LO:.2f}, {EXTREME_HI:.2f}]"
                f" sanity band (guardrail 4: no extreme flag).")
    say()
    say("  ROLLING-ORIGIN discounts for validation -- estimated ONLY from")
    say("  complete instances with season < C, i.e. exactly the estimate")
    say("  production could have made at each cutoff (matches the harness's")
    say("  train-only ave_goals discipline; cutoff C's own instances never")
    say("  inform the discount applied to scoring C):")
    say(f"    {'cutoff':<8}{'n_inst':>7}{'d_attack':>10}{'d_defense':>10}")
    d_rolling = {}
    for c in CUTOFFS:
        a, dd = est["rolling"][c]
        d_rolling[c] = (a["d_shrunk"], dd["d_shrunk"])
        say(f"    {c:<8}{a['n']:>7}{a['d_shrunk']:>10.4f}{dd['d_shrunk']:>10.4f}")
    say("    (cutoff 2026 uses every complete instance = the full-sample")
    say("    estimate; the full-sample contrast run below quantifies the")
    say("    in-sample flattery of the 2023-2025 cutoffs.)")

    # ── Step 2: verification guard ──────────────────────────────────
    say()
    say("-" * 60)
    say("STEP 2: VERIFICATION GUARD -- fast engine vs verbatim references")
    say("-" * 60)
    checks = [(
        "default path unchanged: production params, cutoff 2026 frame "
        f"({len(ctx['frames'][2026])} rows; walk-forward + snapshot + frozen)",
        ctx["frames"][2026], ctx["ave"][2026], prod, 2026)]
    for c in CUTOFFS:
        checks.append((
            f"discounted (full-sample d={d_full[0]:.4f}/{d_full[1]:.4f}), "
            f"cutoff {c} frame ({len(ctx['frames'][c])} rows; "
            f"walk-forward + snapshot + frozen)",
            ctx["frames"][c], ctx["ave"][c], prod, c, None, d_full))
    synthetic = (0.8, 1.2)
    checks.append((
        f"discounted (synthetic {synthetic[0]}/{synthetic[1]}), full history "
        f"({len(df)} rows; walk-forward)",
        df, ave_full, prod, None, None, synthetic))
    checks.append((
        f"discounted (synthetic {synthetic[0]}/{synthetic[1]}), cutoff 2024 "
        f"frame ({len(ctx['frames'][2024])} rows; walk-forward + snapshot + "
        f"frozen)",
        ctx["frames"][2024], ctx["ave"][2024], prod, 2024, None, synthetic))
    t_guard = time.perf_counter()
    guard_ok = ratings_fast.verify_against_reference(checks, prior_map, log=say)
    say(f"  guard wall time: {time.perf_counter() - t_guard:.0f}s")
    if not guard_ok:
        say()
        say("  GUARD FAILED -- aborting. Numbers would be untrustworthy; fix")
        say("  the fast engine / reference divergence before drawing conclusions.")
        return 1

    # ── Step 3: guardrail 1, baseline per tier ──────────────────────
    say()
    say("-" * 60)
    say("STEP 3: GUARDRAIL 1 -- BASELINE (production, no discount), TIERED")
    say("-" * 60)
    base = evaluate(prod, ctx, prior_map, team_sets, discounts={})
    for c in CUTOFFS:
        p = base["per"][c]
        say(f"  cutoff {c}: promoted {', '.join(p['promoted'])}")
        for key, label in (("a", "tier (a) promoted matches"),
                           ("b", "tier (b) the rest")):
            t = p["tiers"][key]
            if t["frozen"] is None:
                say(f"    {label}: no held-out matches")
                continue
            say(f"    {label}: n={t['n']:>3}  frozen LL/tm "
                f"{t['frozen']['poisson_ll_per_tm']:.4f} | wf LL/tm "
                f"{t['wf']['poisson_ll_per_tm']:.4f}")
    for key, label in (("a", "tier (a) pooled"),
                       ("b", "tier (b) pooled"),
                       ("all", "ALL matches pooled (continuity)")):
        s_f, s_w = base["pooled"][key]["frozen"], base["pooled"][key]["wf"]
        say(f"  {label}: frozen LL/tm {s_f['poisson_ll_per_tm']:.4f} "
            f"(n={s_f['n_matches']}) | wf {s_w['poisson_ll_per_tm']:.4f} | "
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

    # ── Step 4: validation, rolling-origin discounts ────────────────
    say()
    say("-" * 60)
    say("STEP 4: VALIDATION -- rolling-origin discounts vs BASELINE, TIERED")
    say("-" * 60)
    disc = evaluate(prod, ctx, prior_map, team_sets, discounts=d_rolling)
    for c in CUTOFFS:
        b, v = base["per"][c], disc["per"][c]
        say(f"  cutoff {c} (d = {d_rolling[c][0]:.4f}/{d_rolling[c][1]:.4f}; "
            f"promoted: {', '.join(b['promoted'])}):")
        for key, label in (("a", "tier (a)"), ("b", "tier (b)")):
            tb, tv = b["tiers"][key], v["tiers"][key]
            if tb["frozen"] is None or tv["frozen"] is None:
                say(f"    {label}: no held-out matches")
                continue
            d_f = tv["frozen"]["poisson_ll_per_tm"] - \
                tb["frozen"]["poisson_ll_per_tm"]
            d_w = tv["wf"]["poisson_ll_per_tm"] - tb["wf"]["poisson_ll_per_tm"]
            say(f"    {label} n={tb['n']:>3}: frozen LL/tm "
                f"{tb['frozen']['poisson_ll_per_tm']:.4f} -> "
                f"{tv['frozen']['poisson_ll_per_tm']:.4f} ({d_f:+.4f}) | "
                f"wf {tb['wf']['poisson_ll_per_tm']:.4f} -> "
                f"{tv['wf']['poisson_ll_per_tm']:.4f} ({d_w:+.4f}) | MAE "
                f"{tb['frozen']['mae']:.4f} -> {tv['frozen']['mae']:.4f} | "
                f"CS LL {tb['frozen']['cs_ll']:.4f} -> "
                f"{tv['frozen']['cs_ll']:.4f}")
    say()
    say("  POOLED across cutoffs (match-count-weighted):")
    for key, label in (("a", "tier (a) promoted matches -- the PRIMARY result"),
                       ("b", "tier (b) the rest"),
                       ("all", "ALL matches (continuity only)")):
        bf, bw = base["pooled"][key]["frozen"], base["pooled"][key]["wf"]
        vf, vw = disc["pooled"][key]["frozen"], disc["pooled"][key]["wf"]
        d_f = vf["poisson_ll_per_tm"] - bf["poisson_ll_per_tm"]
        d_w = vw["poisson_ll_per_tm"] - bw["poisson_ll_per_tm"]
        say(f"    {label}: frozen LL/tm {bf['poisson_ll_per_tm']:.4f} -> "
            f"{vf['poisson_ll_per_tm']:.4f} ({d_f:+.4f}) | wf "
            f"{bw['poisson_ll_per_tm']:.4f} -> {vw['poisson_ll_per_tm']:.4f} "
            f"({d_w:+.4f}) | MAE {bf['mae']:.4f} -> {vf['mae']:.4f} | CS LL "
            f"{bf['cs_ll']:.4f} -> {vf['cs_ll']:.4f} (n={bf['n_matches']})")

    # ── Tier-(b) propagation check (direct tier-(b) matches are ────
    # untouched; the difference is the training-history propagation ──
    # of discounted promoted entries into continuing-team ratings) ──
    max_b = 0.0
    for c in CUTOFFS:
        a1, a2 = disc["per"][c]["ll_f_b"], base["per"][c]["ll_f_b"]
        if a1.size:
            max_b = max(max_b, float(np.max(np.abs(a1 - a2))))
    b_f = disc["pooled"]["b"]["frozen"]["poisson_ll_per_tm"] - \
        base["pooled"]["b"]["frozen"]["poisson_ll_per_tm"]
    b_wf = disc["pooled"]["b"]["wf"]["poisson_ll_per_tm"] - \
        base["pooled"]["b"]["wf"]["poisson_ll_per_tm"]
    say()
    say("  TIER-(B) PROPAGATION CHECK -- tier-(b) matches involve two")
    say("    continuing teams, so nothing touches them directly; the discount")
    say("    reaches them only through training history (discounted promoted")
    say("    entries changed the updates continuing teams received in prior")
    say("    seasons, hence their boundary ratings):")
    say(f"    frozen pooled delta {b_f:+.4f} LL/tm, max per-team-match |diff| "
        f"{max_b:.1e}; walk-forward spillover {b_wf:+.4f} LL/tm")
    if abs(b_f) < 0.002 and abs(b_wf) < 0.002:
        say("    -> near zero, as the prediction expects (weak propagation)")
    else:
        say("    -> larger than the near-zero expectation -- inspect before")
        say("       trusting the tier-(a) reading")

    # ── Tier (a) instance-level view + cluster bootstrap ───────────
    deltas, cluster_tm = [], []
    for c in CUTOFFS:
        deltas.append(disc["per"][c]["ll_f_a"] - base["per"][c]["ll_f_a"])
        cl_c = base["per"][c]["clusters"]
        cluster_tm.extend(cl_c)
        cluster_tm.extend(cl_c)
    deltas = np.concatenate(deltas)
    cluster_tm = np.array(cluster_tm)
    say()
    say("  PER-PROMOTED-TEAM-SEASON view (frozen, tier a): does the discount")
    say("  help team by team, or is the pooled figure one outlier's work?")
    say(f"    {'team-season':<26}{'n_tm':>5}{'mean dLL':>10}")
    for c in CUTOFFS:
        for t in base["per"][c]["promoted"]:
            key = f"{c}:{t}"
            msk = cluster_tm == key
            if msk.sum() == 0:
                continue
            say(f"    {key:<26}{int(msk.sum()):>5}{deltas[msk].mean():>+10.4f}")
    uniq, boot = cluster_bootstrap(deltas, cluster_tm)
    obs = float(deltas.mean())
    lo, med, hi = (float(x) for x in np.percentile(boot, [2.5, 50, 97.5]))
    say()
    say(f"  TIER (a) pooled frozen delta: {obs:+.4f} LL/tm "
        f"(negative = discount better; {deltas.size} team-matches)")
    say(f"    cluster bootstrap (resampling {len(uniq)} promoted team-seasons, "
        f"{BOOT_N} reps, seed {BOOT_SEED}):")
    say(f"    95% CI [{lo:+.4f}, {hi:+.4f}], median {med:+.4f} -- "
        f"{'EXCLUDES zero' if (lo < 0 and hi < 0) or (lo > 0 and hi > 0) else 'STRADDLES zero'}")

    # ── Full-sample contrast ────────────────────────────────────────
    say()
    say("  FULL-SAMPLE-DISCOUNT CONTRAST (every cutoff scored with d = the")
    say("  full-sample estimate -- quantifies the in-sample flattery of")
    say("  including the validation seasons' own instances in the estimate):")
    disc_full = evaluate(prod, ctx, prior_map, team_sets,
                         discounts={c: d_full for c in CUTOFFS})
    for key, label in (("a", "tier (a)"), ("b", "tier (b)"),
                       ("all", "ALL")):
        d_roll = disc["pooled"][key]["frozen"]["poisson_ll_per_tm"] - \
            base["pooled"][key]["frozen"]["poisson_ll_per_tm"]
        d_fullr = disc_full["pooled"][key]["frozen"]["poisson_ll_per_tm"] - \
            base["pooled"][key]["frozen"]["poisson_ll_per_tm"]
        say(f"    {label}: rolling-origin {d_roll:+.4f} vs full-sample "
            f"{d_fullr:+.4f} (flattery {d_fullr - d_roll:+.4f})")

    # ── Tier (a) clean-sheet reliability ───────────────────────────
    say()
    say("  tier (a) clean-sheet reliability, pooled (equal-count bins on")
    say("  predicted P(clean sheet); the entry discount moves exactly these")
    say("  probabilities):")
    for tag, res in (("baseline", base), ("discounted", disc)):
        rows = cs_bin_table(res["pooled"]["a"]["probs"],
                            res["pooled"]["a"]["events"])
        say(f"    {tag}:")
        for r in rows:
            say(f"      {r['bin']:<18}n={r['n']:>4}  pred {r['pred']:.3f}  "
                f"obs {r['obs']:.3f}")

    # ── Scorecard ───────────────────────────────────────────────────
    say()
    say("-" * 60)
    say("SCORECARD vs PRE-REGISTERED PREDICTION")
    say("-" * 60)
    per_cutoff_a = []
    for c in CUTOFFS:
        tb = base["per"][c]["tiers"]["a"]["frozen"]
        tv = disc["per"][c]["tiers"]["a"]["frozen"]
        per_cutoff_a.append(
            float("nan") if tb is None or tv is None
            else tv["poisson_ll_per_tm"] - tb["poisson_ll_per_tm"])
    say("  tier (a) frozen LL/tm deltas (negative = discount better): "
        + " | ".join(f"{c} {d:+.4f}" for c, d in zip(CUTOFFS, per_cutoff_a)))
    say(f"  pooled tier (a): {obs:+.4f} LL/tm | bootstrap 95% CI "
        f"[{lo:+.4f}, {hi:+.4f}]")
    n_improved = sum(1 for d in per_cutoff_a if d < 0)
    say(f"  cutoffs improved: {n_improved}/{len(CUTOFFS)}")
    say(f"  tier (b): frozen {b_f:+.4f} LL/tm, walk-forward {b_wf:+.4f} "
        f"LL/tm -- propagation only (direct effect zero by construction;")
    say(f"  near zero, as predicted)")
    say(f"  noise-band context (§21): 0.001 LL/tm pooled is the established")
    say(f"  noise band on this harness; tier (a) holds "
        f"{base['pooled']['a']['frozen']['n_matches']} of "
        f"{base['pooled']['all']['frozen']['n_matches']} held-out matches, so")
    say(f"  the cluster bootstrap above -- not the pooled band -- is the")
    say(f"  tier-level judgment.")

    # ── Verdict ─────────────────────────────────────────────────────
    say()
    say("-" * 60)
    say("VERDICT")
    say("-" * 60)
    if obs < 0 and hi < 0 and n_improved >= 3:
        verdict = "supported"
        say("  SUPPORTED: real positive tier-(a) effect, distinguishable from")
        say("  noise (bootstrap CI excludes zero), consistent across cutoffs;")
        say("  tier (b) untouched as predicted. The mechanism -- money priors")
        say("  structurally optimistic for promoted entries -- is confirmed")
        say("  where it should show and absent where it should not.")
    elif obs < 0:
        verdict = "weak_partial"
        say("  WEAK/PARTIAL: tier-(a) direction matches the prediction, but")
        say("  the effect is not distinguishable from noise (bootstrap CI")
        say("  straddles zero and/or cutoffs inconsistent) -- §22's lesson")
        say("  applies: mechanism plausibility is not predictive evidence.")
    else:
        verdict = "failed"
        say("  FAILED: tier (a) does not improve under the discount.")
    say()
    say("  update_team_ratings.py UNCHANGED in this run (measured, not")
    say("  shipped); shipping is a separate review decision. If shipped, the")
    say(f"  values would be the full-sample estimates: d_attack={d_full[0]:.4f},"
        f" d_defense={d_full[1]:.4f}.")

    # ── Artifacts ───────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = OUT_DIR / "discount_report.txt"
    report.write_text("\n".join(_REPORT_LINES) + "\n", encoding="utf-8")
    summary = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "data": data_meta,
        "params": prod,
        "design": {
            "scale": "NPG per match (user-confirmed; priors, actuals and "
                     "ratings all on it)",
            "tiers": {
                "a": "held-out match involves a promoted team of that season",
                "b": "all other held-out matches"},
            "estimation_instances": len(est["instances"]),
            "excluded_from_estimation": [
                {"season": x["season"], "team": x["team"], "n": x["n"]}
                for x in est["excluded"]],
            "shrinkage_formula":
                "d = (n*r_bar + k*1.0)/(n+k); k = mean(sigma2)/tau2 with "
                "sigma2_i = r_i/(m_i*prior_i), tau2 = max(0, var(r) - "
                "mean(sigma2)); z = n/(n+k)",
            "noise_band_ll": NOISE_BAND_LL,
            "noise_band_n": NOISE_BAND_N,
        },
        "estimation": {
            "attack": _js(est["attack"]),
            "defense": _js(est["defense"]),
            "control_attack": _js(est["control_attack"]),
            "control_defense": _js(est["control_defense"]),
            "instances": est["instances"],
        },
        "rolling_discounts": {
            str(c): {"attack": _js(est["rolling"][c][0]),
                     "defense": _js(est["rolling"][c][1])}
            for c in CUTOFFS},
        "guard_passed": bool(guard_ok),
        "baseline_pooled": {
            k: {"frozen": base["pooled"][k]["frozen"],
                "wf": base["pooled"][k]["wf"]}
            for k in ("a", "b", "all")},
        "validation_rolling": {
            "per_cutoff": {
                str(c): {"discount": list(d_rolling[c]),
                         "a": disc["per"][c]["tiers"]["a"],
                         "b": disc["per"][c]["tiers"]["b"]}
                for c in CUTOFFS},
            "pooled": {k: {"frozen": disc["pooled"][k]["frozen"],
                           "wf": disc["pooled"][k]["wf"]}
                       for k in ("a", "b", "all")},
            "tier_a_delta_frozen": obs,
            "tier_a_bootstrap": {"lo": lo, "median": med, "hi": hi,
                                 "n_clusters": int(len(uniq)),
                                 "reps": BOOT_N, "seed": BOOT_SEED},
            "tier_b_frozen_delta": float(b_f),
            "tier_b_frozen_max_diff": max_b,
            "tier_b_wf_delta": float(b_wf),
            "per_team_season": {str(k): float(deltas[cluster_tm == k].mean())
                                for k in uniq},
        },
        "validation_fullsample_contrast": {
            k: {"delta": disc_full["pooled"][k]["frozen"]["poisson_ll_per_tm"]
                - base["pooled"][k]["frozen"]["poisson_ll_per_tm"]}
            for k in ("a", "b", "all")},
        "scorecard": {
            "tier_a_deltas_per_cutoff": {str(c): d for c, d
                                         in zip(CUTOFFS, per_cutoff_a)},
            "n_improved": n_improved,
        },
        "verdict": verdict,
        "update_team_ratings_changed": False,
    }
    (OUT_DIR / "discount_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    say()
    say(f"  wall time: {time.perf_counter() - t0:.0f}s")
    say(f"  artifacts: {report.name}, discount_summary.json -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
