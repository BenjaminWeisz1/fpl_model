"""
Faster learning for promoted teams early in the season (CHANGELOG §252).

A team entering the ratings pool from its money prior (absent the previous
season; never the frame's first season) updates with
    alpha_t = alpha * (1 + (m - 1) * max(0, 1 - k / N))
where k = its matches already played this season. Its opponent keeps alpha.
m = 1 is production exactly. update_team_ratings.py is untouched.

Primary score: walk-forward tier (a) -- held-out matches with a promoted team,
predicted one step ahead -- because the harness's frozen score cannot see an
in-season learning change. (m, N) chosen per cutoff from seasons before it.

Run:
    python -m research.team_ratings.promoted_learning_team_ratings
Writes data/current/team_ratings_oot/promoted_learning_report.txt + _summary.json.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
from research.team_ratings.evaluate_team_ratings_oot import DEFAULT_CUTOFFS, EPS, OUT_DIR, load_inputs  # noqa: E402
from research.team_ratings.odds_prior_team_ratings import EARLY_N, PARAMS, per_match_ll, promoted_by_season, tier_mean
from research.team_ratings.ratings_fast import fast_run, frozen_predict_fast  # noqa: E402

M_GRID = [1.0, 1.5, 2.0, 3.0, 4.0]
N_GRID = [5, 10, 19, 38]
N_BOOT = 1000
SEED = 42
TIER_B_TOLERANCE = 0.0005

_LINES = []


def say(line=""):
    print(line, flush=True)
    _LINES.append(line)


def boost_factor(m, n_cap, k):
    return 1.0 + (m - 1.0) * max(0.0, 1.0 - k / n_cap)


def boosted_run(df, ave_goals, params, prior_map, m, n_cap, snapshot_seasons=()):
    """fast_run's core loop (no rebalance/discount) with the entrant boost.
    Arithmetic order matches fast_run so that m = 1 is bit-identical."""
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
    first_season = int(season[0])
    season_teams = {s: set(home[season == s]) | set(away[season == s]) for s in pd.unique(season)}

    ratings, entrants, played = {}, set(), {}
    current_season = None
    n = len(df)
    ph, pa = np.empty(n), np.empty(n)
    snaps = {}
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
                    ratings[t] = [(1 - mw) * v[0] + mw * m_atk, (1 - mw) * v[1] + mw * m_def]
            entrants = set(teams) - set(ratings) if s > first_season else set()
            played = {}
            if s in snapshot_seasons and s not in snaps:
                snap = {t: list(v) for t, v in ratings.items()}
                for t in sorted(teams):
                    if t not in snap:
                        snap[t] = list(prior_map[(s, t)])
                snaps[s] = snap
        hi, ai = home[i], away[i]
        if hi not in ratings:
            ratings[hi] = list(prior_map[(s, hi)])
        if ai not in ratings:
            ratings[ai] = list(prior_map[(s, ai)])
        h_atk, h_def = ratings[hi]
        a_atk, a_def = ratings[ai]
        xh = h_atk * (a_def / ave_goals) * ha
        xa = a_atk * (h_def / ave_goals) * two_ha
        ph[i], pa[i] = xh, xa
        num_h = h_npxg[i] * b_ + h_npg[i] * (1 - b_)
        num_a = a_npxg[i] * b_ + a_npg[i] * (1 - b_)
        ah = a * boost_factor(m, n_cap, played.get(hi, 0)) if hi in entrants and m != 1.0 else a
        aa = a * boost_factor(m, n_cap, played.get(ai, 0)) if ai in entrants and m != 1.0 else a
        ratings[hi] = [h_atk * (1 - ah + ah * (num_h / max(xh, EPS))),
                       h_def * (1 - ah + ah * (num_a / max(xa, EPS)))]
        ratings[ai] = [a_atk * (1 - aa + aa * (num_a / max(xa, EPS))),
                       a_def * (1 - aa + aa * (num_h / max(xh, EPS)))]
        played[hi] = played.get(hi, 0) + 1
        played[ai] = played.get(ai, 0) + 1
    return ph, pa, snaps


def reference_run(df, ave_goals, params, prior_map, m, n_cap):
    """Independent, plain row-by-row reference (walk-forward predictions only)."""
    a, b_, ha, mw = params["alpha"], params["beta"], params["home_advantage"], params["money_weight"]
    first = int(df["Season"].iloc[0])
    ratings, prev_teams = {}, set()
    out_h, out_a = [], []
    for s, g in df.groupby("Season", sort=False):
        teams = set(g["Home"]) | set(g["Away"])
        ratings = {t: ((1 - mw) * v[0] + mw * prior_map[(s, t)][0],
                       (1 - mw) * v[1] + mw * prior_map[(s, t)][1])
                   for t, v in ratings.items() if t in teams}
        new = (teams - prev_teams) if s != first else set()
        k = dict.fromkeys(teams, 0)
        for r in g.itertuples(index=False):
            h, w = r.Home, r.Away
            hr = ratings.setdefault(h, prior_map[(s, h)])
            wr = ratings.setdefault(w, prior_map[(s, w)])
            lh = hr[0] * (wr[1] / ave_goals) * ha
            la = wr[0] * (hr[1] / ave_goals) * (2 - ha)
            out_h.append(lh)
            out_a.append(la)
            th = r.Home_NPxG_Adj * b_ + r.Home_NPG * (1 - b_)
            ta = r.Away_NPxG_Adj * b_ + r.Away_NPG * (1 - b_)
            al_h = a * (1 + (m - 1) * max(0.0, 1 - k[h] / n_cap)) if h in new else a
            al_a = a * (1 + (m - 1) * max(0.0, 1 - k[w] / n_cap)) if w in new else a
            ratings[h] = (hr[0] * (1 - al_h + al_h * th / max(lh, EPS)), hr[1] * (1 - al_h + al_h * ta / max(la, EPS)))
            ratings[w] = (wr[0] * (1 - al_a + al_a * ta / max(la, EPS)), wr[1] * (1 - al_a + al_a * th / max(lh, EPS)))
            k[h] += 1
            k[w] += 1
        prev_teams = teams
    return np.array(out_h), np.array(out_a)


def score_season(df, prior_map, promoted, S, m, n_cap):
    ave = float(pd.concat([df[df["Season"] < S]["Home_NPG"], df[df["Season"] < S]["Away_NPG"]]).mean())
    frame = df[df["Season"] <= S]
    ph, pa, snaps = boosted_run(frame, ave, PARAMS, prior_map, m, n_cap, snapshot_seasons={S})
    mask = (frame["Season"] == S).values
    val = frame[mask]
    fh, fa = frozen_predict_fast(val, snaps[S], prior_map, ave, PARAMS)
    prom = promoted.get(S, set())
    tier_a = (val["Home"].isin(prom) | val["Away"].isin(prom)).values
    count, early = {}, []
    for h, a in zip(val["Home"], val["Away"]):
        nums = []
        for t in (h, a):
            count[t] = count.get(t, 0) + 1
            if t in prom:
                nums.append(count[t])
        early.append(bool(nums) and min(nums) <= EARLY_N)
    cluster = [f"{S}:{h if h in prom else a}" if (h in prom or a in prom) else ""
               for h, a in zip(val["Home"], val["Away"])]
    return {"S": S, "frozen": (fh, fa), "wf": (ph[mask], pa[mask]),
            "y": (val["Home_NPG"].values.astype(float), val["Away_NPG"].values.astype(float)),
            "tier_a": tier_a, "early": np.asarray(early), "cluster": np.asarray(cluster)}


def main():
    df, prior_map, _ = load_inputs()
    seasons, promoted = promoted_by_season(df)
    say("=" * 64)
    say("FASTER LEARNING FOR PROMOTED TEAMS EARLY IN THE SEASON (CHANGELOG §252)")
    say("=" * 64)
    say(f"  run {datetime.now().isoformat(timespec='seconds')} | params {PARAMS}")
    say(f"  grid m {M_GRID} x N {N_GRID}; cutoffs {list(DEFAULT_CUTOFFS)}")

    # Guard 1: m = 1 is fast_run bit for bit (2024 frame and full history).
    for upto in (2024, int(df["Season"].max())):
        frame = df[df["Season"] <= upto]
        ave = float(pd.concat([df[df["Season"] < upto]["Home_NPG"], df[df["Season"] < upto]["Away_NPG"]]).mean())
        r0 = fast_run(frame, ave, PARAMS, prior_map, snapshot_seasons={upto})
        r1 = boosted_run(frame, ave, PARAMS, prior_map, 1.0, 10, snapshot_seasons={upto})
        d_wf = max(np.abs(r0[0] - r1[0]).max(), np.abs(r0[1] - r1[1]).max())
        d_sn = max(abs(r0[2][upto][t][j] - r1[2][upto][t][j]) for t in r0[2][upto] for j in (0, 1))
        ok = d_wf == 0 and d_sn == 0 and set(r0[2][upto]) == set(r1[2][upto])
        say(f"  [{'PASS' if ok else 'FAIL'}] m=1 v fast_run, frame to {upto}: wf max|diff| {d_wf:.1e}, snapshot {d_sn:.1e}")
        if not ok:
            raise SystemExit("boosted engine at m=1 differs from fast_run")

    # Guard 2: boosted engine v independent reference at m=2, N=10.
    frame = df[df["Season"] <= 2024]
    ave = float(pd.concat([df[df["Season"] < 2024]["Home_NPG"], df[df["Season"] < 2024]["Away_NPG"]]).mean())
    b = boosted_run(frame, ave, PARAMS, prior_map, 2.0, 10)
    r = reference_run(frame, ave, PARAMS, prior_map, 2.0, 10)
    d_ref = max(np.abs(b[0] - r[0]).max(), np.abs(b[1] - r[1]).max())
    d_base = max(np.abs(b[0] - fast_run(frame, ave, PARAMS, prior_map)[0]).max(), 0)
    ok = d_ref < 1e-12 and d_base > 0
    say(f"  [{'PASS' if ok else 'FAIL'}] m=2,N=10 v reference, frame to 2024: max|diff| {d_ref:.1e} "
        f"(v production {d_base:.3f}, so the boost is active)")
    if not ok:
        raise SystemExit("boosted engine disagrees with the reference")

    say("\n  scoring seasons x grid ...")
    grid = [(1.0, N_GRID[0])] + [(m, n) for m in M_GRID if m != 1.0 for n in N_GRID]
    results = {g: {S: score_season(df, prior_map, promoted, S, *g) for S in seasons[1:]} for g in grid}
    base = results[(1.0, N_GRID[0])]

    # Guard 3: baseline reproduces baseline.json (frozen all-match LL per cutoff).
    base_json = json.loads((OUT_DIR / "baseline.json").read_text(encoding="utf-8"))
    for C in DEFAULT_CUTOFFS:
        rr = base[C]
        v = per_match_ll(rr, "frozen").sum() / (2 * len(rr["tier_a"]))
        ref = base_json["splits"][str(C)]["frozen"]["poisson_ll_per_tm"]
        if abs(v - ref) > 1e-9:
            raise SystemExit(f"m=1 does not reproduce baseline.json for {C}: {v} vs {ref}")
    say("  m=1 reproduces baseline.json frozen LL for every cutoff: OK")

    say("\n--- (m, N) selection: walk-forward tier-(a) LL pooled over seasons 2018..C-1 ---")
    chosen = {}
    for C in DEFAULT_CUTOFFS:
        prior = [S for S in seasons[1:] if S < C]
        scores = {g: tier_mean([results[g][S] for S in prior], "wf", "tier_a") for g in grid}
        chosen[C] = min(scores, key=scores.get)
        best = sorted(scores.items(), key=lambda kv: kv[1])[:5]
        say(f"  C={C}: chosen m={chosen[C][0]}, N={chosen[C][1]} | base {scores[grid[0]]:.4f} | best five: "
            + "; ".join(f"({g[0]},{g[1]}) {v:.4f}" for g, v in best))

    say("\n--- Held-out (per team-match Poisson LL; delta = boosted - production) ---")
    rows = []
    for C in DEFAULT_CUTOFFS:
        b0, e0 = base[C], results[chosen[C]][C]
        row = {"C": C, "m": chosen[C][0], "N": chosen[C][1]}
        for mode in ("wf", "frozen"):
            for tier in ("tier_a", "early", "tier_b"):
                row[f"{mode}_{tier}"] = (tier_mean([b0], mode, tier), tier_mean([e0], mode, tier))
        rows.append(row)
        say(f"  C={C} (m={row['m']}, N={row['N']}): wf tier(a) {row['wf_tier_a'][0]:.4f} -> {row['wf_tier_a'][1]:.4f} "
            f"({row['wf_tier_a'][1] - row['wf_tier_a'][0]:+.4f}) | wf early {row['wf_early'][1] - row['wf_early'][0]:+.4f}"
            f" | wf tier(b) {row['wf_tier_b'][1] - row['wf_tier_b'][0]:+.5f}"
            f" | frozen tier(a) {row['frozen_tier_a'][1] - row['frozen_tier_a'][0]:+.5f}")

    say("\n--- Pooled 2023-2026 at the chosen (m, N) ---")
    pooled = {}
    for mode in ("wf", "frozen"):
        for tier in ("tier_a", "early", "tier_b"):
            bb = tier_mean([base[C] for C in DEFAULT_CUTOFFS], mode, tier)
            nn = tier_mean([results[chosen[C]][C] for C in DEFAULT_CUTOFFS], mode, tier)
            pooled[f"{mode}_{tier}"] = {"base": bb, "new": nn, "delta": nn - bb}
            say(f"  {mode:<6s} {tier:<7s} {bb:.4f} -> {nn:.4f}  delta {nn - bb:+.5f}")

    d_by_cluster = {}
    for C in DEFAULT_CUTOFFS:
        b0, e0 = base[C], results[chosen[C]][C]
        diff = per_match_ll(e0, "wf") - per_match_ll(b0, "wf")
        for cl in np.unique(b0["cluster"][b0["tier_a"]]):
            mk = b0["cluster"] == cl
            d_by_cluster[cl] = (diff[mk].sum(), 2 * mk.sum())
    keys = sorted(d_by_cluster)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(keys), len(keys), replace=True)
        boots.append(sum(d_by_cluster[keys[i]][0] for i in pick) / sum(d_by_cluster[keys[i]][1] for i in pick))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    say(f"\n  cluster bootstrap ({len(keys)} promoted team-seasons, {N_BOOT} reps): wf tier(a) delta 95% CI "
        f"[{lo:+.4f}, {hi:+.4f}]")
    say("  per promoted team-season (wf tier(a) delta per team-match):")
    for k_ in keys:
        s_, n_ = d_by_cluster[k_]
        say(f"    {k_:<24s} {s_ / n_:+.4f}  (n_tm={n_})")

    # Descriptive only (never used for selection): the whole grid on all held-out seasons.
    say("\n--- Descriptive: pooled 2023-2026 wf tier(a) delta for every (m, N) (hindsight; not used) ---")
    for g in grid[1:]:
        dd = tier_mean([results[g][C] for C in DEFAULT_CUTOFFS], "wf", "tier_a") - pooled["wf_tier_a"]["base"]
        say(f"    m={g[0]:<4} N={g[1]:<3} {dd:+.5f}")

    d = pooled["wf_tier_a"]["delta"]
    n_improved = sum(1 for r_ in rows if r_["wf_tier_a"][1] < r_["wf_tier_a"][0])
    tb = pooled["wf_tier_b"]["delta"]
    if d < 0 and hi < 0 and n_improved >= 3 and tb <= TIER_B_TOLERANCE:
        verdict = "SUPPORTED"
    elif d < 0:
        verdict = "WEAK/PARTIAL"
    else:
        verdict = "FAILED"
    say(f"\n  RULE: wf tier(a) {d:+.5f}, CI [{lo:+.4f}, {hi:+.4f}], cutoffs improved {n_improved}/4, "
        f"wf tier(b) {tb:+.5f} -> {verdict}")
    say("  update_team_ratings.py UNCHANGED (measured, not shipped).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "promoted_learning_report.txt").write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    (OUT_DIR / "promoted_learning_summary.json").write_text(json.dumps({
        "chosen": {str(C): list(g) for C, g in chosen.items()}, "per_cutoff": rows, "pooled": pooled,
        "bootstrap_ci": [lo, hi], "cutoffs_improved": n_improved, "verdict": verdict,
        "per_cluster": {k_: v[0] / v[1] for k_, v in d_by_cluster.items()},
    }, indent=2, default=float), encoding="utf-8")
    say(f"\n  wrote {OUT_DIR / 'promoted_learning_report.txt'}")


if __name__ == "__main__":
    main()
