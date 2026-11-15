"""
Other-league replication of §252's entrant learning boost (CHANGELOG §253).
Research only: reads Understat league files, writes only to
data/validation/promoted_learning_other_leagues_20261115/. Reuses §252's
boosted_run and §178's load_league / signal_prior_map unchanged.

Protocol, rule and prediction: PROTOCOL.md in that folder, fixed before any
arm was scored.

    python -u -X utf8 scripts/team_ratings/validate_promoted_learning_other_leagues.py
"""
from fpl_model import paths
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.team_ratings.update_team_ratings import alpha, beta, money_weight  # noqa: E402
from research.team_ratings.promoted_learning_team_ratings import M_GRID, N_GRID, boosted_run, reference_run  # noqa: E402
from research.team_ratings.ratings_fast import fast_run  # noqa: E402
from research.team_ratings.validate_kalman_other_leagues import cs_ll, load_league, signal_prior_map, team_ll  # noqa: E402

ROOT = paths.ROOT
OUT = paths.VALIDATION / "promoted_learning_other_leagues_20261115"
LEAGUES = ("Bundesliga", "La_liga", "Ligue_1", "RFPL", "Serie_A")
SCORED = tuple(range(2015, 2026))      # 2014 = each frame's first season
CUTOFFS = tuple(range(2017, 2026))     # held out
ARM_A = (2.0, 10)
EARLY_N = 10
N_BOOT = 2000
SEED = 20261115
TIER_B_TOLERANCE = 0.0005
GRID = [(1.0, N_GRID[0])] + [(m, n) for m in M_GRID if m != 1.0 for n in N_GRID]
BASE = GRID[0]

_LINES = []


def say(s=""):
    print(s, flush=True)
    _LINES.append(s)


def score_season(df, prior, lg, S, m, n_cap):
    """Walk-forward predictions of season S from the frame to S, train-only
    level, home advantage and goals/NPG scaling (§178's "goals" target)."""
    train = df[df["Season"] < S]
    ave = float(pd.concat([train["Home_NPG"], train["Away_NPG"]]).mean())
    ha = float(train["Home_NPG"].mean()) / ave
    mult = float(pd.concat([train["Home_Goals"], train["Away_Goals"]]).mean()) / ave
    params = {"alpha": alpha, "beta": beta, "home_advantage": ha, "money_weight": money_weight}
    frame = df[df["Season"] <= S].reset_index(drop=True)
    ph, pa, _ = boosted_run(frame, ave, params, prior, m, n_cap)
    mask = (frame["Season"] == S).values
    val = frame[mask]
    prev = set(df.loc[df["Season"] == S - 1, "Home"]) | set(df.loc[df["Season"] == S - 1, "Away"])
    hh, aa = val["Home"].values, val["Away"].values
    ent_h = ~np.isin(hh, list(prev))
    ent_a = ~np.isin(aa, list(prev))
    count, kh, ka = {}, [], []
    for h, a in zip(hh, aa):
        count[h] = count.get(h, 0) + 1
        count[a] = count.get(a, 0) + 1
        kh.append(count[h])
        ka.append(count[a])
    kh, ka = np.array(kh), np.array(ka)
    lh, la = mult * ph[mask], mult * pa[mask]
    yh = val["Home_Goals"].values.astype(float)
    ya = val["Away_Goals"].values.astype(float)
    early = (ent_h & (kh <= EARLY_N)) | (ent_a & (ka <= EARLY_N))
    cluster = np.where(ent_h, [f"{lg}:{S}:{t}" for t in hh],
                       np.where(ent_a, [f"{lg}:{S}:{t}" for t in aa], ""))
    return {"lg": lg, "S": S, "lh": lh, "la": la, "yh": yh, "ya": ya,
            "ll": team_ll(lh, yh) + team_ll(la, ya),
            "cs": cs_ll(la, ya) + cs_ll(lh, yh),
            "tier_a": ent_h | ent_a, "early": early, "cluster": cluster,
            "ent_h": ent_h, "ent_a": ent_a, "kh": kh, "ka": ka}


def tmean(rs, tier, key="ll"):
    num = den = 0.0
    for r in rs:
        mk = {"tier_a": r["tier_a"], "tier_b": ~r["tier_a"],
              "early": r["early"], "late": r["tier_a"] & ~r["early"]}[tier]
        num += r[key][mk].sum()
        den += 2 * mk.sum()
    return num / den if den else float("nan")


def bootstrap(base_rs, new_rs):
    d = {}
    for b, n in zip(base_rs, new_rs):
        diff = n["ll"] - b["ll"]
        for cl in np.unique(b["cluster"][b["tier_a"]]):
            mk = b["cluster"] == cl
            d[cl] = (diff[mk].sum(), 2 * mk.sum())
    keys = sorted(d)
    s = np.array([d[k][0] for k in keys])
    n = np.array([d[k][1] for k in keys])
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, len(keys), len(keys))
        boots[i] = s[pick].sum() / n[pick].sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi), len(keys)


def bias(rs, first_n=None):
    """Entrants' walk-forward bias, actual minus predicted goals (conceded, scored)."""
    con, sco = [], []
    for r in rs:
        for side, ent, k in (("h", r["ent_h"], r["kh"]), ("a", r["ent_a"], r["ka"])):
            mk = ent if first_n is None else ent & (k <= first_n)
            other = "a" if side == "h" else "h"
            con.append(r[f"y{other}"][mk] - r[f"l{other}"][mk])
            sco.append(r[f"y{side}"][mk] - r[f"l{side}"][mk])
    return float(np.concatenate(con).mean()), float(np.concatenate(sco).mean())


def verdict_of(base_rs, new_rs, by_league):
    d = tmean(new_rs, "tier_a") - tmean(base_rs, "tier_a")
    tb = tmean(new_rs, "tier_b") - tmean(base_rs, "tier_b")
    lo, hi, n = bootstrap(base_rs, new_rs)
    good = sum(1 for v in by_league.values() if v < 0)
    if d < 0 and hi < 0 and good >= 4 and tb <= TIER_B_TOLERANCE:
        v = "SUPPORTED"
    elif d < 0:
        v = "WEAK/PARTIAL"
    else:
        v = "FAILED"
    return {"d_tier_a": d, "d_tier_b": tb, "ci": [lo, hi], "clusters": n,
            "leagues_improved": good, "verdict": v}


def main():
    protocol = OUT / "PROTOCOL.md"
    if not protocol.exists():
        raise SystemExit("PROTOCOL.md must exist before any arm is scored")
    sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    say("=" * 70)
    say("FASTER LEARNING FOR PROMOTED TEAMS: OTHER LEAGUES (CHANGELOG §253)")
    say("=" * 70)
    say(f"  PROTOCOL.md sha256 {sha}")
    say(f"  alpha {alpha}, beta {beta}, money_weight {money_weight}; arm A m={ARM_A[0]}, N={ARM_A[1]}")

    data = {}
    for lg in LEAGUES:
        df, _, _ = load_league(lg)
        prior, _ = signal_prior_map(df)
        data[lg] = (df, prior)

    # Guards on Bundesliga to 2020: m = 1 equals fast_run; boost matches the reference.
    df, prior = data["Bundesliga"]
    frame = df[df["Season"] <= 2020].reset_index(drop=True)
    ave = float(pd.concat([frame["Home_NPG"], frame["Away_NPG"]]).mean())
    params = {"alpha": alpha, "beta": beta, "home_advantage": 1.08, "money_weight": money_weight}
    r0 = fast_run(frame, ave, params, prior)
    r1 = boosted_run(frame, ave, params, prior, 1.0, 10)
    d0 = max(np.abs(r0[0] - r1[0]).max(), np.abs(r0[1] - r1[1]).max())
    b = boosted_run(frame, ave, params, prior, *ARM_A)
    rf = reference_run(frame, ave, params, prior, *ARM_A)
    d1 = max(np.abs(b[0] - rf[0]).max(), np.abs(b[1] - rf[1]).max())
    d2 = np.abs(b[0] - r0[0]).max()
    ok = d0 == 0 and d1 < 1e-12 and d2 > 0
    say(f"  [{'PASS' if ok else 'FAIL'}] m=1 v fast_run {d0:.1e}; boost v reference {d1:.1e}; "
        f"boost v production {d2:.3f}")
    if not ok:
        raise SystemExit("guard failed")

    say("\n  scoring 5 leagues x 11 seasons x 17 settings ...")
    res = {g: {(lg, S): score_season(data[lg][0], data[lg][1], lg, S, *g)
               for lg in LEAGUES for S in SCORED} for g in GRID}
    held = [(lg, C) for lg in LEAGUES for C in CUTOFFS]
    base = [res[BASE][k] for k in held]
    n_ent = sum(len(np.unique(r["cluster"][r["tier_a"]])) for r in base)
    say(f"  held-out: {len(held)} league-seasons, {sum(len(r['ll']) for r in base)} matches, "
        f"{n_ent} entrant team-seasons, {int(sum(r['tier_a'].sum() for r in base))} tier-(a) matches")

    summary = {"protocol_sha256": sha}
    for arm in ("A", "B"):
        if arm == "A":
            pick = {C: ARM_A for C in CUTOFFS}
            say(f"\n--- ARM A (primary): fixed m={ARM_A[0]}, N={ARM_A[1]} ---")
        else:
            say("\n--- ARM B: (m, N) chosen per cutoff on the five leagues' seasons 2015..C-1 ---")
            pick = {}
            for C in CUTOFFS:
                sc = {g: tmean([res[g][(lg, S)] for lg in LEAGUES for S in SCORED if S < C], "tier_a")
                      for g in GRID}
                pick[C] = min(sc, key=sc.get)
                best = sorted(sc.items(), key=lambda kv: kv[1])[:3]
                say(f"  C={C}: chosen m={pick[C][0]}, N={pick[C][1]} (base {sc[BASE]:.4f}; best "
                    + "; ".join(f"({g[0]},{g[1]}) {v:.4f}" for g, v in best) + ")")
        new = [res[pick[C]][(lg, C)] for lg, C in held]
        say(f"  {'':<12}{'tier(a) base':>13}{'delta':>10}{'early':>10}{'late':>10}{'tier(b)':>10}{'CS (a)':>10}")
        by_lg = {}
        for lg in LEAGUES + ("POOLED",):
            ix = [i for i, (l, _) in enumerate(held) if lg in (l, "POOLED")]
            bb, nn = [base[i] for i in ix], [new[i] for i in ix]
            row = [tmean(bb, "tier_a"), tmean(nn, "tier_a") - tmean(bb, "tier_a"),
                   tmean(nn, "early") - tmean(bb, "early"), tmean(nn, "late") - tmean(bb, "late"),
                   tmean(nn, "tier_b") - tmean(bb, "tier_b"),
                   tmean(nn, "tier_a", "cs") - tmean(bb, "tier_a", "cs")]
            if lg != "POOLED":
                by_lg[lg] = row[1]
            say(f"  {lg:<12}{row[0]:>13.4f}" + "".join(f"{x:>+10.5f}" for x in row[1:]))
        say("  by season (pooled leagues), tier(a) delta:")
        say("   " + " ".join(f"{C}:{tmean([new[i] for i, (_, c) in enumerate(held) if c == C], 'tier_a') - tmean([base[i] for i, (_, c) in enumerate(held) if c == C], 'tier_a'):+.4f}"
                             for C in CUTOFFS))
        v = verdict_of(base, new, by_lg)
        say(f"  bootstrap ({v['clusters']} entrant team-seasons, {N_BOOT} reps): tier(a) 95% CI "
            f"[{v['ci'][0]:+.5f}, {v['ci'][1]:+.5f}]")
        say(f"  bias, entrants (actual - predicted goals; conceded / scored):")
        for lab, fn in (("whole season", None), (f"first {EARLY_N}", EARLY_N)):
            c0, s0 = bias(base, fn)
            c1, s1 = bias(new, fn)
            say(f"    {lab:<14} conceded {c0:+.3f} -> {c1:+.3f} | scored {s0:+.3f} -> {s1:+.3f}")
        say(f"  RULE: tier(a) {v['d_tier_a']:+.5f}, CI [{v['ci'][0]:+.5f}, {v['ci'][1]:+.5f}], "
            f"leagues improved {v['leagues_improved']}/5, tier(b) {v['d_tier_b']:+.5f} -> {v['verdict']}"
            + ("" if arm == "A" else " (secondary; not the verdict)"))
        summary[f"arm_{arm}"] = {**v, "by_league": by_lg,
                                 "chosen": {str(C): list(g) for C, g in pick.items()}}

    say("\n--- Descriptive (hindsight, no part in the verdict): every setting on all held-out seasons ---")
    hind = {}
    for g in GRID[1:]:
        nn = [res[g][k] for k in held]
        hind[f"{g[0]},{g[1]}"] = tmean(nn, "tier_a") - tmean(base, "tier_a")
    for k_, v_ in hind.items():
        say(f"    m={k_.split(',')[0]:<4} N={k_.split(',')[1]:<3} {v_:+.5f}")
    summary["hindsight_tier_a"] = hind
    say(f"\n  VERDICT (arm A): {summary['arm_A']['verdict']}. update_team_ratings.py UNCHANGED.")

    (OUT / "report.txt").write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
