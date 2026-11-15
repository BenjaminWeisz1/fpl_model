"""Defcon model: out-of-time validation metrics (log loss, AUC, Brier, reliability by evidence bucket).

Split out of predict.py (2026-11-16 cleanup) without changes. Reporting only: nothing here changes a trained model.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def stratified_reliability(proba, y, matches_played, label, n_bins=5, min_cell=25):
    """Reliability check stratified by MatchesPlayed (individual-evidence strength).

    Aggregate ECE can hide evidence-dependent miscalibration: a low-evidence
    player pushed up toward the mean and a high-evidence player pushed down
    toward the mean land in the same predicted-probability bin and cancel
    each other out. This checks calibration, prediction spread, and
    discrimination WITHIN evidence buckets, cross-tabulates observed rates
    by (probability bin x evidence bucket), and formally tests
    (likelihood-ratio) whether P(target | prediction) depends on
    MatchesPlayed. The compression signature is observed rates RISING with
    MatchesPlayed in the top probability bins and FALLING in the bottom
    bins -- i.e. the same prediction meaning different things depending on
    how much individual evidence backs it."""
    p = np.asarray(proba, dtype=float)
    yv = np.asarray(y)
    mp = np.asarray(matches_played, dtype=float)

    bucket_labels = ["0-9", "10-29", "30-79", "80-149", "150+"]
    buckets = pd.cut(mp, bins=[-0.5, 9.5, 29.5, 79.5, 149.5, np.inf], labels=bucket_labels)
    bucket_masks = {lab: np.asarray(buckets == lab) for lab in bucket_labels}
    active = [lab for lab in bucket_labels if bucket_masks[lab].sum() > 0]

    # Per-bucket calibration, prediction spread, within-bucket discrimination.
    # A model that properly trusts individual evidence should show BOTH:
    # prediction spread (p_std) growing with MatchesPlayed, and within-bucket
    # AUC growing with MatchesPlayed (more evidence -> more discrimination).
    summary = {"buckets": {}, "lrt": None}
    print(f"\n  Reliability by MatchesPlayed bucket:")
    print(f"  {'bucket':<10s} {'n':>6s} {'obs':>6s} {'pred':>6s} {'gap':>6s} {'p_std':>6s} {'AUC':>6s} {'Brier':>7s}")
    for lab in active:
        m = bucket_masks[lab]
        obs, pred = yv[m].mean(), p[m].mean()
        summary["buckets"][lab] = {"n": int(m.sum()), "obs": float(obs), "pred": float(pred)}
        try:
            auc = roc_auc_score(yv[m], p[m])
        except ValueError:
            auc = float("nan")
        print(f"  {lab:<10s} {m.sum():>6d} {obs:>6.3f} {pred:>6.3f} {pred - obs:>+6.3f} "
              f"{p[m].std():>6.3f} {auc:>6.3f} {brier_score_loss(yv[m], p[m]):>7.3f}")

    # Observed rate by (probability bin x evidence bucket): the compression
    # signature lives HERE, not in the aggregate numbers above. If the model
    # under-trusts strong evidence, high-evidence players inside a given
    # probability bin will hit MORE often than low-evidence players in the
    # same bin (and vice versa in the bottom bins).
    interior = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)[1:-1]))
    if len(interior) > 0:
        pbin = np.digitize(p, interior)
        nb = len(interior) + 1
        print(f"\n  Observed target rate by (predicted-probability bin x MatchesPlayed bucket):")
        hdr = f"  {'prob bin':<18s} {'n':>5s} {'pred':>6s} {'all':>6s}"
        hdr += "".join(f" {lab:>9s}" for lab in active)
        print(hdr)
        bin_ranges = {}
        for b in range(nb):
            m = pbin == b
            if m.sum() == 0:
                continue
            lo = interior[b - 1] if b > 0 else p.min()
            hi = interior[b] if b < len(interior) else p.max()
            bin_ranges[b] = (lo, hi)
            row = f"  [{lo:.3f}, {hi:.3f})".ljust(19) + f"{m.sum():>5d} {p[m].mean():>6.3f} {yv[m].mean():>6.3f}"
            for lab in active:
                mm = m & bucket_masks[lab]
                row += f" {yv[mm].mean():>9.3f}" if mm.sum() >= min_cell else f" {'--':>9s}"
            print(row)

        # Direct compression test: within each probability bin, compare
        # observed rates for lowest- vs highest-evidence players. Same bin,
        # different evidence -- a consistent nonzero gap means predictions
        # carry different meaning depending on sample size.
        lo_lab, hi_lab = active[0], active[-1]
        print(f"\n  Within-bin evidence gaps (observed rate, {hi_lab} minus {lo_lab} MatchesPlayed):")
        for b, (lo, hi) in bin_ranges.items():
            m = pbin == b
            lo_m = m & bucket_masks[lo_lab]
            hi_m = m & bucket_masks[hi_lab]
            if lo_m.sum() < min_cell or hi_m.sum() < min_cell:
                continue
            gap = yv[hi_m].mean() - yv[lo_m].mean()
            print(f"    [{lo:.3f}, {hi:.3f}): {yv[hi_m].mean():.3f} (n={hi_m.sum():>4d}) vs "
                  f"{yv[lo_m].mean():.3f} (n={lo_m.sum():>4d}) -> gap {gap:+.3f}")

    # Formal omnibus test: LRT of y ~ logit(p) against y ~ logit(p) * bucket.
    # (Caveat: rows cluster within players and matches, so the p-value is
    # anti-conservative -- treat a small p as a flag, not proof.)
    # Skipped when y has only one class (e.g. forward 2026 with 0 positives)
    # — logistic regression requires at least 2 classes.
    if len(active) > 1 and len(np.unique(yv)) > 1:
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        lp = np.log(pc / (1 - pc))
        lp = (lp - lp.mean()) / (lp.std() + 1e-12)
        dummies = np.column_stack([bucket_masks[lab].astype(float) for lab in active])
        X_red = np.column_stack([np.ones(len(p)), lp])
        extra = [dummies[:, j] for j in range(1, len(active))]
        extra += [lp * dummies[:, j] for j in range(1, len(active))]
        X_full = np.column_stack([X_red] + extra)

        def _loglik(X):
            clf = LogisticRegression(C=1e8, max_iter=5000)
            clf.fit(X, yv)
            ph = np.clip(clf.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
            return float(np.sum(yv * np.log(ph) + (1 - yv) * np.log(1 - ph)))

        lrt = 2.0 * (_loglik(X_full) - _loglik(X_red))
        dof = X_full.shape[1] - X_red.shape[1]
        p_val = float(1.0 - chi2.cdf(lrt, dof))
        print(f"\n  LRT (calibration independent of MatchesPlayed): chi2={lrt:.2f}, df={dof}, p={p_val:.4f}")
        if p_val < 0.05:
            print("  -> P(target | pred) differs by evidence level: evidence-dependent bias present.")
        else:
            print("  -> No detectable evidence-dependent miscalibration at the 5% level.")
        summary["lrt"] = {"chi2": lrt, "dof": dof, "p": p_val}
    return summary


def report_metrics(proba, y_val, X_val, feature_cols, label):
    """Metrics + MatchesPlayed-stratified reliability for a scored validation
    frame -- shared by the freshly-trained and already-trained (ablation)
    validation paths. Returns the stratified-reliability summary, if any."""
    ll = log_loss(y_val, proba, labels=[0, 1])
    # AUC is undefined when y_true has only one class (e.g. forward 2026
    # with 0 positives among a tiny validation set); guard with NaN.
    try:
        auc = roc_auc_score(y_val, proba)
    except ValueError:
        auc = float("nan")
    brier = brier_score_loss(y_val, proba)

    # Expected calibration error (10 equal-width bins)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (proba >= bin_edges[b]) & (proba < bin_edges[b + 1])
        if b == n_bins - 1:
            mask = mask | (proba == bin_edges[b + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_val[mask].mean() - proba[mask].mean())
    ece /= len(y_val)

    print(f"\n{'=' * 60}")
    print(f"OUT-OF-TIME VALIDATION: {label}")
    print(f"{'=' * 60}")
    print(f"  Val: {len(y_val)} records | target rate: {y_val.mean():.3f}")
    print(f"  Log loss:  {ll:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Brier:     {brier:.4f}")
    print(f"  ECE:       {ece:.4f}")

    # Stratified reliability check -- aggregate ECE can hide compression
    # (a low-evidence player pushed up toward the mean and a high-evidence
    # player pushed down toward the mean land in the same probability bin
    # and cancel out), so check calibration within MatchesPlayed buckets.
    summary = None
    if "MatchesPlayed" in feature_cols:
        mp_idx = feature_cols.index("MatchesPlayed")
        summary = stratified_reliability(proba, y_val, X_val[:, mp_idx], label)
    return summary


def score_validation(calibrated, X_val, y_val, feature_cols, label):
    """Score a validation frame with an already-trained model -- used for
    the priors-ON vs priors-OFF ablation, where training rows are identical
    under both configs and only the validation features differ.
    The CalibratedModel wrapper handles scaling internally."""
    proba = calibrated.predict_proba(X_val)[:, 1]
    return report_metrics(proba, y_val, X_val, feature_cols, label)


# Compact falsifiable-test comparison: the two numbers the mechanism claim
# makes a directional prediction about.
def mech_gap(s):
    b = (s or {}).get("buckets", {}).get("0-9")
    return f"gap {b['pred'] - b['obs']:+.3f} (n={b['n']})" if b else "no 0-9 rows"

def mech_lrt(s):
    lrt = (s or {}).get("lrt")
    return f"LRT chi2={lrt['chi2']:.2f}, p={lrt['p']:.4f}" if lrt else "LRT n/a"
