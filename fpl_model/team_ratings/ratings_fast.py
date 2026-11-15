"""
Fast dict-based port of update_team_ratings.py's sequential rating loop.

Permanent engine for team-ratings tuning and experiment work: identical
operations in identical order to the production loop, but with the team
table held in a plain dict instead of a pandas DataFrame -- ~800x faster
(0.006 ms/row vs 4.9 ms/row; full 3,439-match pass ~0.02s vs ~17s),
which turns thousand-combination grid searches into minutes-long jobs.

Correctness contract: fast_run() is BIT-EXACT against the verbatim
reference implementation, evaluate_team_ratings_oot.run_sequential()
(measured max |diff| = 0.0 on the full history, and re-verified on every
consumer run via verify_against_reference() below). The verbatim loop is
kept permanently in the harness file precisely for this equivalence
check: if update_team_ratings.py's rating logic ever changes, BOTH the
harness copy and this port must be updated together and re-verified.

The expressions here deliberately mirror run_sequential() line-for-line
(including the precomputed 2 - home_advantage, which is bit-identical to
recomputing it per match). Resist "cleaning them up" -- reordering float
operations can break bit-exactness silently.

Rebalancing extension (TEAM_RATINGS_REBALANCING.md): fast_run(rebalance=
"midpoint"|"defense") applies the season-boundary rebalancing step at
the start of EVERY season (2017 included), after prune + money blend
and before the season's first match: extend the table to all season
teams (newcomers at pure priors), then rescale so the Attack/Defense
league averages meet at M=(avg_Attack+avg_Defense)/2 ("midpoint" -- a
real design decision, not an obvious default) or the Defense average
stays fixed ("defense"). The reference for this mode is the harness's
run_sequential_rebalanced() twin; the averages accumulate in sorted
team order in BOTH engines so the rescale factors stay bit-exact.
collect_boundary_gaps=True records the pre-rebalance league-average
gap at every season boundary (the mechanism's checkable side-output)
-- with rebalance=None it only measures (virtual extension, ratings
untouched), so the default path's predictions are unaffected.

Promoted-team discount extension (discount_team_ratings.py):
fast_run(discount=(d_attack, d_defense)) multiplies the pure money
prior by (d_attack, d_defense) for every team entering the pool via
the 100%-money-implied branch (the lazy add at a season entrant's
first match, and the matching snapshot extension) -- applied only
after the frame's first season (pool initialization is not promotion);
continuing teams' money_weight blend is untouched. The reference for
this mode is the harness's run_sequential_discounted() twin;
multiplication order (discount * prior) mirrors it exactly.
rebalance and discount are EXCLUSIVE (one experimental change at a
time).

Consumers MUST run verify_against_reference() before trusting results;
tune_team_ratings.py runs it unconditionally at startup and aborts on
failure. Checks are (label, frame, ave_goals, params, snapshot_season)
tuples, optionally with a 6th element rebalance=<anchor> to verify the
rebalanced path against its twin, or a 7th element discount=(d_atk,
d_def) to verify the discounted path against its twin.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.team_ratings.evaluate_team_ratings_oot import frozen_predict, run_sequential, run_sequential_discounted, run_sequential_rebalanced

_EPS = 1e-6      # mirrors the production loop's eps exactly
_EQUIV_TOL = 1e-9  # same threshold class as the harness's history check


def fast_run(df, ave_goals, params, prior_map, snapshot_seasons=(),
             collect_dynamics=False, rebalance=None,
             collect_boundary_gaps=False, discount=None):
    """Dict-based mirror of run_sequential(). Returns (pred_home, pred_away,
    snapshots); with collect_dynamics=True also a 4th result
    {"mean_step_atk", "mean_step_def", "n_updates"} -- mean |rating change|
    per rating update over the run (4 updates per match), the update-
    dynamics summary used to sanity-check tuned parameters; with
    collect_boundary_gaps=True a 5th result -- the per-season-boundary
    gap records (see module docstring; when both dynamics and gaps are
    requested the 4th result is the dynamics dict, the 5th the gaps
    list, otherwise the 4th is None).

    rebalance: None (default -- production behavior, bit-for-bit the
    existing code path) or "midpoint"/"defense" (the season-boundary
    rebalancing step, mirrored from run_sequential_rebalanced()).
    discount: None (default) or (d_attack, d_defense) -- the
    promoted-team entry discount, mirrored from
    run_sequential_discounted(). rebalance and discount are EXCLUSIVE.

    snapshots: {season: {team: [attack, defense]}} captured AFTER the
    season transition (prune + money-prior blend [+ rebalancing when
    enabled]) and extended with every season team still missing at its
    pure prior values -- identical to the harness's boundary snapshot
    (with the discount applied to those extension values when
    enabled)."""
    if rebalance is not None and discount is not None:
        raise ValueError("rebalance and discount are exclusive: one "
                         "experimental change at a time")
    a = params["alpha"]
    b_ = params["beta"]
    ha = params["home_advantage"]
    mw = params["money_weight"]
    two_ha = 2 - ha
    d_atk = d_def = None
    if discount is not None:
        d_atk, d_def = discount

    season = df["Season"].values
    home = df["Home"].values
    away = df["Away"].values
    h_npxg = df["Home_NPxG_Adj"].values.astype(float)
    a_npxg = df["Away_NPxG_Adj"].values.astype(float)
    h_npg = df["Home_NPG"].values.astype(float)
    a_npg = df["Away_NPG"].values.astype(float)
    first_season = int(season[0])  # pool init season: discount never applies

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
    dyn = {"step_atk": 0.0, "step_def": 0.0, "n": 0} if collect_dynamics else None
    gaps = [] if collect_boundary_gaps else None
    eps = _EPS

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
            if rebalance is not None or collect_boundary_gaps:
                if rebalance is not None:
                    # Newcomers join the table at pure priors (their
                    # post-blend state) so the rescale reaches them too.
                    for t in teams:
                        if t not in ratings:
                            ratings[t] = list(prior_map[(s, t)])
                # League averages over ALL season teams, accumulated in
                # sorted team order -- must mirror run_sequential_
                # rebalanced() operation-for-operation (bit-exactness).
                # With rebalance=None this is measurement only: missing
                # teams are read virtually and the table is untouched.
                sum_atk = 0.0
                sum_def = 0.0
                for t in sorted(teams):
                    v = ratings.get(t)
                    if v is None:
                        m_atk, m_def = prior_map[(s, t)]
                        v = (m_atk, m_def)
                    sum_atk += v[0]
                    sum_def += v[1]
                n_t = len(teams)
                avg_atk = sum_atk / n_t
                avg_def = sum_def / n_t
                rec = {"season": int(s), "n_teams": n_t,
                       "avg_attack": avg_atk, "avg_defense": avg_def,
                       "gap": avg_atk - avg_def}
                if rebalance is not None:
                    if rebalance == "midpoint":
                        m_ = (avg_atk + avg_def) / 2
                        k_a = m_ / avg_atk
                        k_d = m_ / avg_def
                    elif rebalance == "defense":
                        k_a = avg_def / avg_atk
                        k_d = 1.0
                    else:
                        raise ValueError(
                            f"unknown anchor {rebalance!r} (expected 'midpoint' or 'defense')")
                    for t in teams:
                        ratings[t] = [ratings[t][0] * k_a, ratings[t][1] * k_d]
                    rec.update({"anchor": rebalance,
                                "k_attack": k_a, "k_defense": k_d})
                if collect_boundary_gaps:
                    if rebalance is not None:
                        post_a = post_d = 0.0
                        for t in sorted(teams):
                            post_a += ratings[t][0]
                            post_d += ratings[t][1]
                        rec["avg_attack_post"] = post_a / n_t
                        rec["avg_defense_post"] = post_d / n_t
                    gaps.append(rec)
            if s in snapshot_seasons and s not in snaps:
                snap = {t: list(v) for t, v in ratings.items()}
                for t in sorted(teams):
                    if t not in snap:
                        m_atk, m_def = prior_map[(s, t)]
                        if d_atk is not None and s > first_season:
                            snap[t] = [d_atk * m_atk, d_def * m_def]
                        else:
                            snap[t] = [m_atk, m_def]
                snaps[s] = snap
        hi, ai = home[i], away[i]
        if hi not in ratings:
            m_atk, m_def = prior_map[(s, hi)]
            if d_atk is not None and s > first_season:
                ratings[hi] = [d_atk * m_atk, d_def * m_def]
            else:
                ratings[hi] = [m_atk, m_def]
        if ai not in ratings:
            m_atk, m_def = prior_map[(s, ai)]
            if d_atk is not None and s > first_season:
                ratings[ai] = [d_atk * m_atk, d_def * m_def]
            else:
                ratings[ai] = [m_atk, m_def]
        h_atk, h_def = ratings[hi]
        a_atk, a_def = ratings[ai]

        xh = h_atk * (a_def / ave_goals) * ha
        xa = a_atk * (h_def / ave_goals) * two_ha
        ph[i] = xh
        pa[i] = xa

        num_h = h_npxg[i] * b_ + h_npg[i] * (1 - b_)
        num_a = a_npxg[i] * b_ + a_npg[i] * (1 - b_)
        nh_atk = h_atk * (1 - a + a * (num_h / max(xh, eps)))
        nh_def = h_def * (1 - a + a * (num_a / max(xa, eps)))
        na_atk = a_atk * (1 - a + a * (num_a / max(xa, eps)))
        na_def = a_def * (1 - a + a * (num_h / max(xh, eps)))
        ratings[hi] = [nh_atk, nh_def]
        ratings[ai] = [na_atk, na_def]
        if collect_dynamics:
            dyn["step_atk"] += abs(nh_atk - h_atk) + abs(na_atk - a_atk)
            dyn["step_def"] += abs(nh_def - h_def) + abs(na_def - a_def)
            dyn["n"] += 2

    if collect_dynamics:
        dynamics = {
            "mean_step_atk": float(dyn["step_atk"] / dyn["n"]) if dyn["n"] else 0.0,
            "mean_step_def": float(dyn["step_def"] / dyn["n"]) if dyn["n"] else 0.0,
            "n_updates": int(dyn["n"]),
        }
    else:
        dynamics = None
    if collect_boundary_gaps:
        return ph, pa, snaps, dynamics, gaps
    if collect_dynamics:
        return ph, pa, snaps, dynamics
    return ph, pa, snaps


def frozen_predict_fast(val_df, snap, prior_map, ave_goals, params):
    """Dict-based mirror of evaluate_team_ratings_oot.frozen_predict():
    predict held-out matches from a frozen boundary snapshot (dict form)."""
    ha = params["home_advantage"]
    two_ha = 2 - ha
    pred_home, pred_away = [], []
    for _, row in val_df.iterrows():
        home, away, season = row["Home"], row["Away"], row["Season"]
        h = snap.get(home)
        if h is None:
            h_atk, h_def = prior_map[(season, home)]
            h = [h_atk, h_def]
        a = snap.get(away)
        if a is None:
            a_atk, a_def = prior_map[(season, away)]
            a = [a_atk, a_def]
        pred_home.append(h[0] * (a[1] / ave_goals) * ha)
        pred_away.append(a[0] * (h[1] / ave_goals) * two_ha)
    return np.asarray(pred_home, dtype=float), np.asarray(pred_away, dtype=float)


def verify_against_reference(checks, prior_map, log=print):
    """Equivalence guard: run the fast engine and the verbatim reference
    on the same frame/params and require agreement.

    checks: iterable of (label, frame, ave_goals, params, snapshot_season)
    tuples, optionally extended with a 6th element rebalance=<anchor>
    and/or a 7th element discount=(d_atk, d_def) -- at most one of the
    two may be set (one experimental change at a time).
      no 6th/7th          -> reference is run_sequential() (plain production)
      rebalance=<anchor>  -> reference is run_sequential_rebalanced()
      discount=(d_a, d_d) -> reference is run_sequential_discounted()
      snapshot_season None  -> verify walk-forward predictions only
      snapshot_season = C   -> also verify the boundary snapshot values AND
                               the frozen predictions end-to-end (the full
                               scoring path the harness uses).

    A check passes when every compared quantity agrees within _EQUIV_TOL
    (bit-exact in practice; the tolerance only guards against future
    pandas float-handling changes). Returns True only if ALL checks pass.
    """
    all_ok = True
    for check in checks:
        if len(check) == 7:
            (label, frame, ave_goals, params, snap_season, rebalance,
             discount) = check
        elif len(check) == 6:
            label, frame, ave_goals, params, snap_season, rebalance = check
            discount = None
        else:
            label, frame, ave_goals, params, snap_season = check
            rebalance = discount = None
        if rebalance is not None and discount is not None:
            raise ValueError("rebalance and discount checks are exclusive")
        snap_set = {snap_season} if snap_season is not None else set()
        if rebalance is not None:
            ref_out, ref_snaps = run_sequential_rebalanced(
                frame, ave_goals, params, prior_map, snapshot_seasons=snap_set,
                anchor=rebalance)
        elif discount is not None:
            ref_out, ref_snaps = run_sequential_discounted(
                frame, ave_goals, params, prior_map, snapshot_seasons=snap_set,
                discount=discount)
        else:
            ref_out, ref_snaps = run_sequential(
                frame, ave_goals, params, prior_map, snapshot_seasons=snap_set)
        ph, pa, snaps = fast_run(
            frame, ave_goals, params, prior_map, snapshot_seasons=snap_set,
            rebalance=rebalance, discount=discount)

        diffs = {
            "walk_forward_predictions": float(max(
                np.max(np.abs(ph - ref_out["Home_Prediction"].values)),
                np.max(np.abs(pa - ref_out["Away_Prediction"].values)))),
        }
        team_mismatch = False
        if snap_season is not None:
            ref_snap = ref_snaps[snap_season]
            ref_lookup = {t: (atk, dfn) for t, atk, dfn in
                          zip(ref_snap["Team"], ref_snap["Attack_Rating"],
                              ref_snap["Defense_Rating"])}
            fast_snap = snaps[snap_season]
            team_mismatch = set(ref_lookup) != set(fast_snap)
            d_snap = 0.0
            for t, (atk, dfn) in ref_lookup.items():
                if t in fast_snap:
                    d_snap = max(d_snap, abs(fast_snap[t][0] - atk),
                                 abs(fast_snap[t][1] - dfn))
            diffs["snapshot_values"] = float(d_snap)
            val = frame[frame["Season"] == snap_season]
            vh, va = frozen_predict(val, ref_snap, prior_map, ave_goals, params)
            fh, fa = frozen_predict_fast(val, fast_snap, prior_map, ave_goals, params)
            diffs["frozen_predictions"] = float(max(
                np.max(np.abs(vh - fh)), np.max(np.abs(va - fa))))

        ok = (not team_mismatch) and all(v < _EQUIV_TOL for v in diffs.values())
        parts = [f"{k} max|diff|={v:.3e}" for k, v in diffs.items()]
        if team_mismatch:
            parts.append("SNAPSHOT TEAM SETS DIFFER")
        log(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        for p in parts:
            log(f"         {p}")
        all_ok = all_ok and ok
    return all_ok
