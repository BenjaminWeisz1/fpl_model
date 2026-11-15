"""
Promoted-team priors from GW1 betting odds (AGENDA §2, CHANGELOG §32).

Mechanism under test: a promoted team (absent from the previous season;
returning clubs count as new) enters the ratings pool at its money prior.
This experiment replaces that entry value with a blend
    (1 - w) * money + w * k * odds_implied
where odds_implied is the team's GW1 market-implied goals for/against
inverted through the model's own prediction formula against the opponent's
preseason snapshot rating, and k is a level conversion (mean model GW1
prediction / mean odds GW1 NPG over earlier seasons). Established teams are
never touched: markets inform, never override.

No-leakage design (pre-registered in CHANGELOG §32):
  * For each evaluated season S, priors are rebuilt season by season
    (2018..S), because opponents' preseason snapshots depend on earlier
    promoted priors. ave_goals is train-only (Season < S), as in the harness.
  * k uses GW1 fixtures of seasons before the one being built only.
  * w for cutoff C is chosen on seasons 2018..C-1 (each scored under its own
    no-leakage pipeline) -- never on C itself (§23: full-sample selection
    flattered tier (a) by 7x).

Entry priors override prior_map[(season, team)] for promoted instances only.
Promoted teams are pruned at every season boundary, so prior_map is used
for them solely at entry (lazy add + snapshot extension) -- the production
loop itself is unchanged, and fast_run is verified against the verbatim
reference on an odds-modified prior_map at startup.

Run:
    python odds_prior_team_ratings.py
Writes data/current/team_ratings_oot/odds_prior_report.txt + _summary.json.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import poisson

_HERE = Path(__file__).resolve().parent
from research.team_ratings.evaluate_team_ratings_oot import CS_PEN_GOAL_MULT, DEFAULT_CUTOFFS, EPS, OUT_DIR, load_inputs, alpha, beta, home_advantage, money_weight
from research.team_ratings.ratings_fast import fast_run, frozen_predict_fast, verify_against_reference
from fpl_model.fetch.fetch_football_data import load_season

PARAMS = {"alpha": alpha, "beta": beta, "home_advantage": home_advantage,
          "money_weight": money_weight}
W_GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
EARLY_N = 10            # "first 10 matches" of a promoted team
N_BOOT = 1000
SEED = 42

_LINES = []


def say(line=""):
    print(line)
    _LINES.append(line)


# ── Odds -> implied goals ────────────────────────────────────────────

_G = np.arange(11)
_TOT = np.add.outer(_G, _G)


def _outcome_probs(lh, la):
    m = np.outer(poisson.pmf(_G, lh), poisson.pmf(_G, la))
    return [np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum(), 1 - m[_TOT <= 2].sum()]


def implied_goals(ph, pd_, pa, pov):
    """Independent-Poisson (lam_home, lam_away) best reproducing de-vigged
    1X2 + over-2.5 probabilities (least squares). All goals, not NPG."""
    r = least_squares(lambda x: np.array(_outcome_probs(*x)) - [ph, pd_, pa, pov],
                      [1.4, 1.1], bounds=([0.05, 0.05], [6, 6]))
    return float(r.x[0]), float(r.x[1]), float(np.abs(r.fun).max())


def _devig(*odds):
    inv = 1 / np.asarray(odds, dtype=float)
    return inv / inv.sum()


def gw1_odds(seasons):
    """One row per fixture that is BOTH teams' first match of the season,
    with odds-implied NPG for each side."""
    rows = []
    for s in seasons:
        fd = load_season(s).reset_index(drop=True)
        pre = "BbAv" if "BbAvH" in fd.columns else "Avg"
        seen = set()
        for _, r in fd.iterrows():
            h, a = r["HomeTeam"], r["AwayTeam"]
            first = h not in seen and a not in seen
            seen |= {h, a}
            if not first:
                continue
            ph, pd_, pa = _devig(r[f"{pre}H"], r[f"{pre}D"], r[f"{pre}A"])
            pov = _devig(r[f"{pre}>2.5"], r[f"{pre}<2.5"])[0]
            lh, la, err = implied_goals(ph, pd_, pa, pov)
            rows.append({"Season": s, "Home": h, "Away": a,
                         "NPG_H": lh / CS_PEN_GOAL_MULT, "NPG_A": la / CS_PEN_GOAL_MULT,
                         "fit_err": err})
    return pd.DataFrame(rows)


# ── Prior construction ───────────────────────────────────────────────

def odds_signal(odds, promoted):
    """Signal factory: GW1 odds inverted through the opponent's snapshot.
    Returns signal(s, ctx) -> {team: ((atk, def), info)} for season s's
    promoted teams; ctx carries this season's fast_run outputs."""
    ha, two_ha = PARAMS["home_advantage"], 2 - PARAMS["home_advantage"]

    def signal(s, ctx):
        frame, ph, pa, snap, ave = ctx["frame"], ctx["ph"], ctx["pa"], ctx["snap"], ctx["ave"]
        # level conversion k from earlier seasons' GW1 fixtures with no promoted team
        pos = {k: i for i, k in enumerate(zip(frame["Season"], frame["Home"], frame["Away"]))}
        prev = odds[(odds["Season"] < s)]
        prev = prev[[not (h in promoted.get(ss, ()) or a in promoted.get(ss, ()))
                     for ss, h, a in zip(prev["Season"], prev["Home"], prev["Away"])]]
        idx = [pos[k] for k in zip(prev["Season"], prev["Home"], prev["Away"])]
        model_mean = np.r_[ph[idx], pa[idx]].mean()
        odds_mean = np.r_[prev["NPG_H"].values, prev["NPG_A"].values].mean()
        k = model_mean / odds_mean
        so = odds[odds["Season"] == s]
        out = {}
        for t in sorted(promoted.get(s, ())):
            fx = so[(so["Home"] == t) | (so["Away"] == t)]
            if len(fx) != 1:
                raise SystemExit(f"no unique GW1 odds fixture for promoted ({s}, {t})")
            fx = fx.iloc[0]
            if fx["Home"] == t:
                opp = snap[fx["Away"]]
                lam_for, lam_agst = k * fx["NPG_H"], k * fx["NPG_A"]
                atk_o = lam_for * ave / (opp[1] * ha)
                def_o = lam_agst * ave / (opp[0] * two_ha)
            else:
                opp = snap[fx["Home"]]
                lam_for, lam_agst = k * fx["NPG_A"], k * fx["NPG_H"]
                atk_o = lam_for * ave / (opp[1] * two_ha)
                def_o = lam_agst * ave / (opp[0] * ha)
            out[t] = ((atk_o, def_o), {"k": k})
        return out
    return signal


def build_priors(df, prior_map, signal, promoted, ave, w, upto):
    """prior_map with promoted entries for seasons 2018..upto replaced by
    (1-w)*money + w*signal, built season by season (opponent snapshots depend
    on earlier promoted priors). Returns (prior_map, {(s,t): info})."""
    pm = dict(prior_map)
    info = {}
    first = int(df["Season"].min())
    for s in range(first + 1, upto + 1):
        frame = df[df["Season"] <= s]
        ph, pa, snaps = fast_run(frame, ave, PARAMS, pm, snapshot_seasons={s})
        ctx = {"frame": frame, "ph": ph, "pa": pa, "snap": snaps[s], "ave": ave}
        for t, ((atk_o, def_o), extra) in signal(s, ctx).items():
            m_atk, m_def = prior_map[(s, t)]
            pm[(s, t)] = ((1 - w) * m_atk + w * atk_o, (1 - w) * m_def + w * def_o)
            info[(s, t)] = {**extra, "money": (m_atk, m_def), "signal": (atk_o, def_o)}
    return pm, info


def score_season(df, prior_map, signal, promoted, S, w):
    """Score held-out season S under its own no-leakage pipeline. Returns
    per-team-match arrays plus tier masks."""
    ave = float(pd.concat([df[df["Season"] < S]["Home_NPG"], df[df["Season"] < S]["Away_NPG"]]).mean())
    pm, info = build_priors(df, prior_map, signal, promoted, ave, w, S)
    frame = df[df["Season"] <= S]
    ph, pa, snaps = fast_run(frame, ave, PARAMS, pm, snapshot_seasons={S})
    m = (frame["Season"] == S).values
    val = frame[m]
    fh, fa = frozen_predict_fast(val, snaps[S], pm, ave, PARAMS)
    prom = promoted.get(S, set())
    tier_a = (val["Home"].isin(prom) | val["Away"].isin(prom)).values
    # promoted team's match number within the season (for the early window)
    count, early = {}, []
    for h, a in zip(val["Home"], val["Away"]):
        n = []
        for t in (h, a):
            count[t] = count.get(t, 0) + 1
            if t in prom:
                n.append(count[t])
        early.append(bool(n) and min(n) <= EARLY_N)
    cluster = [f"{S}:{h if h in prom else a}" if (h in prom or a in prom) else ""
               for h, a in zip(val["Home"], val["Away"])]
    return {"S": S, "frozen": (fh, fa), "wf": (ph[m], pa[m]),
            "y": (val["Home_NPG"].values.astype(float), val["Away_NPG"].values.astype(float)),
            "tier_a": tier_a, "early": np.asarray(early), "cluster": np.asarray(cluster),
            "info": info, "ave": ave}


def ll_tm(pred, y):
    p = np.maximum(pred, EPS)
    return p - y * np.log(p)


def per_match_ll(res, mode):
    (ph, pa), (yh, ya) = res[mode], res["y"]
    return ll_tm(ph, yh) + ll_tm(pa, ya)   # sum of the match's two team-match LLs


def tier_mean(res_list, mode, tier):
    num = den = 0.0
    for r in res_list:
        mask = r[tier] if tier != "tier_b" else ~r["tier_a"]
        if tier == "early":
            mask = r["tier_a"] & r["early"]
        num += per_match_ll(r, mode)[mask].sum()
        den += 2 * mask.sum()
    return num / den if den else float("nan")


# ── Main ─────────────────────────────────────────────────────────────

def promoted_by_season(df):
    seasons = sorted(int(s) for s in df["Season"].unique())
    teams = {s: set(g["Home"]) | set(g["Away"]) for s, g in df.groupby("Season")}
    return seasons, {s: teams[s] - teams[s - 1] for s in seasons[1:]}


def evaluate(df, prior_map, promoted, seasons, signal, title, stem, signal_label, header=(),
             prereg=(0.005, 0.02)):
    """The full pre-registered evaluation for one entry-prior signal: guard,
    w=0 baseline check, rolling-origin w, two tiers + early window, cluster
    bootstrap, per-instance entry ratings. prereg = (min, max) pre-registered
    frozen tier-(a) improvement per team-match. Writes <stem>_report.txt/_summary.json."""
    _LINES.clear()
    say("=" * 64)
    say(title)
    say("=" * 64)
    say(f"  run {datetime.now().isoformat(timespec='seconds')} | params {PARAMS}")
    for line in header:
        say(line)
    say(f"  promoted instances: {sum(len(v) for v in promoted.values())} "
        f"({', '.join(f'{s}:{len(v)}' for s, v in promoted.items())})")

    # Guard: fast engine vs verbatim reference on an odds-modified prior_map.
    ave24 = float(pd.concat([df[df["Season"] < 2024]["Home_NPG"], df[df["Season"] < 2024]["Away_NPG"]]).mean())
    pm_g, _ = build_priors(df, prior_map, signal, promoted, ave24, 1.0, 2024)
    ok = verify_against_reference(
        [(f"{signal_label} w=1.0 prior_map, 2024 frame", df[df["Season"] <= 2024], ave24, PARAMS, 2024)],
        pm_g, log=say)
    if not ok:
        raise SystemExit("fast engine disagrees with the reference -- aborting")

    # Score every season 2018..last under every w (each under its own pipeline).
    say("\n  scoring seasons x w-grid ...")
    results = {w: {S: score_season(df, prior_map, signal, promoted, S, w) for S in seasons[1:]}
               for w in W_GRID}

    # Sanity: w=0 must equal the harness baseline exactly.
    base_json = json.loads((OUT_DIR / "baseline.json").read_text(encoding="utf-8"))
    for C in DEFAULT_CUTOFFS:
        r = results[0.0][C]
        v = (per_match_ll(r, "frozen").sum()) / (2 * len(r["tier_a"]))
        ref = base_json["splits"][str(C)]["frozen"]["poisson_ll_per_tm"]
        if abs(v - ref) > 1e-9:
            raise SystemExit(f"w=0 does not reproduce baseline for {C}: {v} vs {ref}")
    say("  w=0 reproduces baseline.json frozen Poisson LL for all cutoffs: OK")

    # Rolling-origin selection of w.
    say("\n--- w selection (tier-(a) frozen LL pooled over seasons before C) ---")
    chosen = {}
    for C in DEFAULT_CUTOFFS:
        prior_seasons = [S for S in seasons[1:] if S < C]
        scores = {w: tier_mean([results[w][S] for S in prior_seasons], "frozen", "tier_a") for w in W_GRID}
        chosen[C] = min(scores, key=scores.get)
        say(f"  C={C}: w*={chosen[C]}  | " + " ".join(f"{w}:{v:.4f}" for w, v in scores.items()))

    # Held-out evaluation at the rolling-origin w*.
    say(f"\n--- Held-out evaluation (per team-match Poisson LL; delta = {signal_label} - baseline) ---")
    rows = []
    for C in DEFAULT_CUTOFFS:
        b, e = results[0.0][C], results[chosen[C]][C]
        row = {"C": C, "w": chosen[C]}
        for mode in ("frozen", "wf"):
            for tier in ("tier_a", "early", "tier_b"):
                row[f"{mode}_{tier}_base"] = tier_mean([b], mode, tier)
                row[f"{mode}_{tier}_new"] = tier_mean([e], mode, tier)
        rows.append(row)
        say(f"  C={C} w*={chosen[C]}: frozen tier(a) {row['frozen_tier_a_base']:.4f} -> "
            f"{row['frozen_tier_a_new']:.4f} ({row['frozen_tier_a_new'] - row['frozen_tier_a_base']:+.4f}) | "
            f"tier(b) {row['frozen_tier_b_new'] - row['frozen_tier_b_base']:+.5f} | "
            f"wf tier(a) {row['wf_tier_a_new'] - row['wf_tier_a_base']:+.4f} | "
            f"wf early {row['wf_early_new'] - row['wf_early_base']:+.4f}")

    say("\n--- Pooled 2023-2026 at rolling-origin w* ---")
    pooled = {}
    for mode in ("frozen", "wf"):
        for tier in ("tier_a", "early", "tier_b"):
            bb = tier_mean([results[0.0][C] for C in DEFAULT_CUTOFFS], mode, tier)
            nn = tier_mean([results[chosen[C]][C] for C in DEFAULT_CUTOFFS], mode, tier)
            pooled[f"{mode}_{tier}"] = {"base": bb, "new": nn, "delta": nn - bb}
            say(f"  {mode:<6s} {tier:<7s} {bb:.4f} -> {nn:.4f}  delta {nn - bb:+.5f}")

    # Cluster bootstrap of the pooled frozen tier-(a) delta, by promoted team-season.
    d_by_cluster = {}
    for C in DEFAULT_CUTOFFS:
        b, e = results[0.0][C], results[chosen[C]][C]
        diff = per_match_ll(e, "frozen") - per_match_ll(b, "frozen")
        for cl in np.unique(b["cluster"][b["tier_a"]]):
            mk = b["cluster"] == cl
            d_by_cluster[cl] = (diff[mk].sum(), 2 * mk.sum())
    keys = sorted(d_by_cluster)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(keys), len(keys), replace=True)
        s = sum(d_by_cluster[keys[i]][0] for i in pick)
        n = sum(d_by_cluster[keys[i]][1] for i in pick)
        boots.append(s / n)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    say(f"\n  cluster bootstrap ({len(keys)} promoted team-seasons, {N_BOOT} reps): "
        f"frozen tier(a) delta 95% CI [{lo:+.4f}, {hi:+.4f}]")
    say("  per promoted team-season (frozen tier(a) delta per team-match):")
    for k_ in keys:
        s, n = d_by_cluster[k_]
        say(f"    {k_:<24s} {s / n:+.4f}  (n_tm={n})")

    # Instance view: money vs signal-implied entry ratings (at each season's own pipeline, w=1 info).
    say(f"\n--- Entry ratings per promoted instance (money vs {signal_label}-implied, own-season pipeline) ---")
    say(f"    {'instance':<26s}{'k':>7s}{'atk_money':>10s}{'atk_' + signal_label:>10s}"
        f"{'def_money':>10s}{'def_' + signal_label:>10s}")
    for S in seasons[1:]:
        for (s, t), inf in sorted(results[1.0][S]["info"].items()):
            if s == S:
                say(f"    {f'{s} {t}':<26s}{inf.get('k', float('nan')):>7.3f}{inf['money'][0]:>10.3f}"
                    f"{inf['signal'][0]:>10.3f}{inf['money'][1]:>10.3f}{inf['signal'][1]:>10.3f}")

    # Pre-registered verdict.
    d = pooled["frozen_tier_a"]["delta"]
    p_lo, p_hi = prereg
    verdict = ("SUPPORTED" if -p_hi <= d <= -p_lo else
               "IMPROVED (smaller than predicted)" if -p_lo < d < 0 else
               "IMPROVED (larger than predicted)" if d < -p_hi else "FAILED")
    say(f"\n  pre-registered: frozen tier(a) improves by {p_lo}-{p_hi}/tm -> observed {d:+.5f}: {verdict}"
        f"{'' if lo < 0 < hi else ' (CI excludes 0)'}")
    say(f"  tier(b) frozen delta {pooled['frozen_tier_b']['delta']:+.6f} (expected tiny, < 0.001)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{stem}_report.txt").write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    (OUT_DIR / f"{stem}_summary.json").write_text(json.dumps({
        "chosen_w": chosen, "per_cutoff": rows, "pooled": pooled,
        "bootstrap_ci": [lo, hi], "verdict": verdict,
        "per_cluster": {k_: v[0] / v[1] for k_, v in d_by_cluster.items()},
    }, indent=2, default=float), encoding="utf-8")
    say(f"\n  wrote {OUT_DIR / (stem + '_report.txt')}")


def main():
    df, prior_map, _ = load_inputs()
    seasons, promoted = promoted_by_season(df)
    odds = gw1_odds(seasons)
    evaluate(df, prior_map, promoted, seasons, odds_signal(odds, promoted),
             "PROMOTED-TEAM PRIORS FROM GW1 ODDS (CHANGELOG §32)", "odds_prior", "odds",
             header=[f"  GW1 odds fixtures: {len(odds)} (max 1X2/OU fit error {odds['fit_err'].max():.4f})"])


if __name__ == "__main__":
    main()
