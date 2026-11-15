"""
Post-hoc look at the pre-registered test in validate_kalman_other_leagues.py.
Exploratory, written after that gate was read; decides nothing.
1. Level-matched frozen delta: each arm rescaled to the actual goals of each
   league-season, so only the ordering of matches counts.
2. Incumbent at other alphas (full-sample comparison, flattered), to see if a
   plain learning-rate change reaches the Kalman arm.
3. Premier League controls with paired match-level 95% intervals. Season
   clusters are too few there, and match-level intervals run too narrow.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.team_ratings import validate_kalman_other_leagues as v
from fpl_model.team_ratings.evaluate_team_ratings_oot import load_inputs

OUT = v.OUT
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def level_gain(m, arm):
    """Frozen log loss per team-match after the best per-league-season scale,
    from aggregates: LL(k) - LL(1) = (k - 1) sum(lam) - sum(y) log k."""
    g = m.groupby(["Dataset", "Season"]).agg(
        lam=(f"{arm}_pred_goals", "sum"), y=("actual_goals", "sum"),
        n=("actual_goals", "size"))
    k = g["y"] / g["lam"]
    return float(((k - 1) * g["lam"] - g["y"] * np.log(k)).sum()
                 / (2 * g["n"].sum())), g


def match_ci(m, metric):
    d = (m[f"kal_{metric}"] - m[f"inc_{metric}"]).values / 2
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - 1.96 * se, d.mean() + 1.96 * se


def main():
    m = pd.read_csv(OUT / "match_scores.csv.gz")
    gi, _ = level_gain(m, "inc")
    gk, _ = level_gain(m, "kal")
    raw = v.delta(m, "frozen")
    say("1. Level-matched (per league-season), frozen")
    say(f"   raw delta {raw:+.4f}; level gain inc {gi:+.5f}, kal {gk:+.5f}; "
        f"level-matched delta {raw + gk - gi:+.4f}")
    for lg, g in m.groupby("Dataset"):
        a, _ = level_gain(g, "inc")
        b, _ = level_gain(g, "kal")
        say(f"   {lg:<12} {v.delta(g, 'frozen') + b - a:+.4f}")

    say()
    say("2. Incumbent at other alphas v Kalman, five leagues pooled")
    say("   alpha  inc frozen  inc wf    kal frozen  kal wf")
    base_alpha = v.alpha
    for a in (0.025, 0.035, 0.045, 0.055):
        v.alpha = a
        parts = []
        for lg in v.LEAGUES:
            df, _, _ = v.load_league(lg)
            prior, _ = v.signal_prior_map(df)
            parts.append(v.run_arms(df, prior, v.LEAGUE_CUTOFFS, "goals",
                                    v.money_weight, lg))
        r = pd.concat(parts, ignore_index=True)
        say(f"   {a:.3f}  {r['inc_frozen'].mean() / 2:.4f}      "
            f"{r['inc_wf'].mean() / 2:.4f}    "
            f"{r['kal_frozen'].mean() / 2:.4f}      "
            f"{r['kal_wf'].mean() / 2:.4f}")
    v.alpha = base_alpha

    say()
    say("3. Premier League controls, paired match-level 95% (mean, low, high)")
    prod, money, _ = load_inputs()
    a = v.run_arms(prod.copy(), money, v.SECTION24_CUTOFFS, "npg",
                   v.money_weight, "A")
    epl, _, _ = v.load_league("EPL")
    prior, _ = v.signal_prior_map(epl)
    d = v.run_arms(epl, prior, v.EPL_CUTOFFS, "goals", v.money_weight, "D")
    for lab, x in (("A production, money prior", a), ("D Understat EPL", d)):
        for k in ("frozen", "wf", "cs", "cs_wf"):
            say(f"   {lab:<27}{k:<7}" + " ".join(
                f"{q:+.4f}" for q in match_ci(x, k)))
        for s, g in x.groupby("Season"):
            say(f"      {s}: frozen {v.delta(g, 'frozen'):+.4f}, "
                f"wf {v.delta(g, 'wf'):+.4f}")
    (OUT / "posthoc_report.txt").write_text("\n".join(lines) + "\n",
                                            encoding="utf-8")


if __name__ == "__main__":
    main()
