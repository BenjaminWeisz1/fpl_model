"""
Fresh pre-registered test of §24's Kalman team ratings against the incumbent
on other leagues (AGENDA "Start here", 2026-11-02). Research only: reads
Understat league files and production inputs, writes only to
data/validation/kalman_other_leagues_20261102/. Reuses the §24 engine
(kalman_run) and the incumbent engine (fast_run) unchanged.

Protocol, gate and predictions: PROTOCOL.md in that folder, fixed before any
arm was scored. `--check` prints data coverage and scores no arm.

    python -u -X utf8 -m research.team_ratings.validate_kalman_other_leagues --check
    python -u -X utf8 -m research.team_ratings.validate_kalman_other_leagues
"""
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from fpl_model import paths
from fpl_model.team_ratings.update_team_ratings import alpha, beta, home_advantage, money_weight
from research.team_ratings.evaluate_team_ratings_oot import CS_PEN_GOAL_MULT, load_inputs
from research.team_ratings.kalman_team_ratings import add_match_indices, kalman_run
from research.team_ratings.ratings_fast import fast_run, frozen_predict_fast

ROOT = paths.ROOT
LEAGUE_DIR = paths.RAW / "understat" / "league"
OUT = paths.VALIDATION / "kalman_other_leagues_20261102"

LEAGUES = ("Bundesliga", "La_liga", "Ligue_1", "RFPL", "Serie_A")
LEAGUE_CUTOFFS = tuple(range(2017, 2026))      # 2014-2016 are burn-in
EPL_CUTOFFS = tuple(range(2020, 2027))         # 2017-2019 are burn-in
SECTION24_CUTOFFS = (2023, 2024, 2025, 2026)
HYPER = {"q_within": 0.002, "q_boundary": 0.02,   # §24 modal, frozen
         "p_promoted": 0.1, "r_scale": 2.0}
PEN_CONVERSION = 0.79   # project constant; --check prints the measured rate
EARLY_N = 5
N_BOOT = 10000
SEED = 20261102

# Gate, as fixed in PROTOCOL.md.
GATE_MIN_IMPROVEMENT = 0.0006
GATE_MIN_LEAGUES = 4
GATE_CS_TOLERANCE = 0.0002
GATE_WF_TOLERANCE = 0.001


def load_league(league):
    """One row per match from the two team histories, chronological. Team key
    is the Understat id (titles can change). Asserts every match pairs."""
    rows = []
    titles = {}
    for path in sorted(LEAGUE_DIR.glob(f"{league}_20*.json")):
        season = int(path.stem.rsplit("_", 1)[1])
        with open(path, encoding="utf-8") as f:
            teams = json.load(f)["teams"]
        side = {"h": [], "a": []}
        for tid, team in teams.items():
            titles.setdefault(tid, set()).add(team["title"])
            for r in team["history"]:
                side[r["h_a"]].append({"team": tid, **{
                    k: r[k] for k in ("date", "xG", "npxG", "scored", "missed")
                }, "xGA": r["xGA"], "npxGA": r["npxGA"]})
        h = pd.DataFrame(side["h"])
        a = pd.DataFrame(side["a"])
        m = h.merge(a, left_on=["date", "scored", "missed", "xG", "xGA"],
                    right_on=["date", "missed", "scored", "xGA", "xG"],
                    suffixes=("_h", "_a"))
        if not (len(m) == len(h) == len(a)):
            raise ValueError(f"{path.name}: {len(h)} home, {len(a)} away, "
                             f"{len(m)} paired")
        m["Season"] = season
        rows.append(m)
    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(["Season", "date", "team_h"], kind="stable")
    df = df.reset_index(drop=True)
    out = pd.DataFrame({
        "Season": df["Season"].astype(int), "Date": df["date"],
        "Home": df["team_h"].astype(str), "Away": df["team_a"].astype(str),
        "Home_Goals": df["scored_h"].astype(float),
        "Away_Goals": df["scored_a"].astype(float),
        "Home_NPxG_Adj": df["npxG_h"].astype(float),
        "Away_NPxG_Adj": df["npxG_a"].astype(float),
        "Home_xG": df["xG_h"].astype(float),
        "Away_xG": df["xG_a"].astype(float)})
    # Team-level penalty goals aren't published: attempts come from the
    # xG gap, whose unit is the modal positive gap.
    gaps = pd.concat([out["Home_xG"] - out["Home_NPxG_Adj"],
                      out["Away_xG"] - out["Away_NPxG_Adj"]])
    gaps = gaps[gaps > 0.01].round(3)
    unit = float(gaps.mode().iloc[0])
    for s in ("Home", "Away"):
        att = ((out[f"{s}_xG"] - out[f"{s}_NPxG_Adj"]) / unit).round()
        out[f"{s}_PKatt"] = att
        out[f"{s}_NPG"] = (out[f"{s}_Goals"]
                           - PEN_CONVERSION * att).clip(lower=0.0)
    return out, titles, unit


def signal_prior_map(df):
    """The substitute entry prior, shared by both arms and built from the data
    only, never from an arm's ratings. Continuing teams blend toward last
    season's league-mean signal; entrants take the mean per-match signal for
    and against of the teams that left. First season (burn-in, never scored)
    uses its own league mean."""
    b = beta
    sf_h = df["Home_NPxG_Adj"] * b + df["Home_NPG"] * (1 - b)
    sf_a = df["Away_NPxG_Adj"] * b + df["Away_NPG"] * (1 - b)
    long = pd.concat([
        pd.DataFrame({"Season": df["Season"], "Team": df["Home"],
                      "F": sf_h, "A": sf_a}),
        pd.DataFrame({"Season": df["Season"], "Team": df["Away"],
                      "F": sf_a, "A": sf_h})], ignore_index=True)
    seasons = sorted(int(s) for s in df["Season"].unique())
    teams = {s: set(long.loc[long["Season"] == s, "Team"]) for s in seasons}
    prior = {}
    log = []
    for i, s in enumerate(seasons):
        if i == 0:
            mean = long[long["Season"] == s]
            lvl = float(pd.concat([mean["F"], mean["A"]]).mean())
            for t in teams[s]:
                prior[(s, t)] = (lvl, lvl)
            continue
        p = seasons[i - 1]
        last = long[long["Season"] == p]
        lvl = float(pd.concat([last["F"], last["A"]]).mean())
        left = teams[p] - teams[s]
        new = teams[s] - teams[p]
        if new and not left:
            raise ValueError(f"season {s}: entrants but no leavers")
        lv = last[last["Team"].isin(left)]
        for t in teams[s]:
            if t in new:
                prior[(s, t)] = (float(lv["F"].mean()), float(lv["A"].mean()))
            else:
                prior[(s, t)] = (lvl, lvl)
        log.append({"season": s, "n_teams": len(teams[s]),
                    "entrants": len(new), "leavers": len(left)})
    return prior, log


def team_ll(lam, y):
    lam = np.maximum(lam, 1e-9)
    return lam - y * np.log(lam)


def cs_ll(lam_opp, conceded):
    p = np.clip(np.exp(-lam_opp), 1e-9, 1 - 1e-9)
    e = (conceded == 0).astype(float)
    return -(e * np.log(p) + (1 - e) * np.log(1 - p))


def run_arms(df, prior_map, cutoffs, target, mw, name):
    """Score both arms over one dataset. target "goals": Understat, scored
    against all goals, with rates scaled by the league's train-only goals/NPG
    ratio and home advantage computed train-only. target "npg": production
    match_results, production home advantage and clean-sheet multiplier.
    Returns one row per held-out match, carrying each arm's per-match sums."""
    df = add_match_indices(df.reset_index(drop=True))
    out = []
    for c in cutoffs:
        train = df[df["Season"] < c]
        frame = df[df["Season"] <= c].reset_index(drop=True)
        mask = frame["Season"].values == c
        val = frame[mask]
        ave = float(pd.concat([train["Home_NPG"], train["Away_NPG"]]).mean())
        if target == "goals":
            ha = float(train["Home_NPG"].mean()) / ave
            goals = pd.concat([train["Home_Goals"], train["Away_Goals"]])
            mult = float(goals.mean()) / ave
            cs_mult = mult
            yh, ya = (val["Home_Goals"].values.astype(float),
                      val["Away_Goals"].values.astype(float))
        else:
            ha, mult, cs_mult = home_advantage, 1.0, CS_PEN_GOAL_MULT
            yh, ya = (val["Home_NPG"].values.astype(float),
                      val["Away_NPG"].values.astype(float))
        gh = val["Home_Goals"].values.astype(float)
        ga = val["Away_Goals"].values.astype(float)
        params = {"alpha": alpha, "beta": beta, "home_advantage": ha,
                  "money_weight": mw}
        prev = set(df.loc[df["Season"] == c - 1, "Home"]) | set(
            df.loc[df["Season"] == c - 1, "Away"])
        prom = (~val["Home"].isin(prev) | ~val["Away"].isin(prev)).values
        early = ((val["_h_idx"] <= EARLY_N) | (val["_a_idx"] <= EARLY_N)).values
        rec = {"Dataset": name, "Season": c, "Home": val["Home"].values,
               "Away": val["Away"].values, "Early": early, "Promoted": prom}
        ph_i, pa_i, sn_i = fast_run(frame, ave, params, prior_map,
                                    snapshot_seasons={c})
        ph_k, pa_k, sn_k = kalman_run(frame, ave, HYPER, prior_map, beta, mw,
                                      ha, snapshot_seasons={c})
        for arm, ph, pa, sn in (("inc", ph_i, pa_i, sn_i),
                                ("kal", ph_k, pa_k, sn_k)):
            fh, fa = frozen_predict_fast(val, sn[c], prior_map, ave, params)
            wh, wa = ph[mask], pa[mask]
            rec[f"{arm}_frozen"] = (team_ll(mult * fh, yh)
                                    + team_ll(mult * fa, ya))
            rec[f"{arm}_wf"] = (team_ll(mult * wh, yh)
                                + team_ll(mult * wa, ya))
            rec[f"{arm}_cs"] = (cs_ll(cs_mult * fa, ga) + cs_ll(cs_mult * fh, gh))
            rec[f"{arm}_cs_wf"] = (cs_ll(cs_mult * wa, ga)
                                   + cs_ll(cs_mult * wh, gh))
            rec[f"{arm}_pred_goals"] = mult * (fh + fa)
        rec["actual_goals"] = yh + ya
        out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True)


def delta(m, metric):
    """Kalman minus incumbent, per team-match; each row holds two."""
    return float((m[f"kal_{metric}"] - m[f"inc_{metric}"]).mean() / 2)


def boot_ci(m, metric, seed=SEED):
    """League-season cluster bootstrap, stratified by dataset."""
    rng = np.random.default_rng(seed)
    g = m.assign(d=m[f"kal_{metric}"] - m[f"inc_{metric}"]).groupby(
        ["Dataset", "Season"])["d"].agg(["sum", "size"]).reset_index()
    strata = [x[["sum", "size"]].values for _, x in g.groupby("Dataset")]
    res = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = n = 0.0
        for arr in strata:
            pick = arr[rng.integers(0, len(arr), len(arr))]
            s += pick[:, 0].sum()
            n += pick[:, 1].sum()
        res[b] = s / n / 2
    return float(np.quantile(res, 0.025)), float(np.quantile(res, 0.975))


def line(label, m):
    return {"label": label, "matches": int(len(m)),
            "seasons": int(m[["Dataset", "Season"]].drop_duplicates().shape[0]),
            "inc_frozen": float(m["inc_frozen"].mean() / 2),
            "kal_frozen": float(m["kal_frozen"].mean() / 2),
            "d_frozen": delta(m, "frozen"), "d_wf": delta(m, "wf"),
            "d_cs": delta(m, "cs"), "d_cs_wf": delta(m, "cs_wf")}


def fmt(r):
    return (f"  {r['label']:<34}{r['matches']:>6} {r['inc_frozen']:.4f} "
            f"{r['kal_frozen']:.4f} {r['d_frozen']:+.4f} {r['d_wf']:+.4f} "
            f"{r['d_cs']:+.4f} {r['d_cs_wf']:+.4f}")


def check():
    for lg in LEAGUES + ("EPL",):
        df, titles, unit = load_league(lg)
        _, log = signal_prior_map(df)
        multi = {k: sorted(v) for k, v in titles.items() if len(v) > 1}
        print(f"{lg}: {len(df)} matches, seasons {df['Season'].min()}-"
              f"{df['Season'].max()}, penalty xG unit {unit}, teams "
              f"{len(titles)}, id with >1 title {multi}")
        print("   entrants/leavers by season: " + " ".join(
            f"{r['season']}:{r['entrants']}/{r['leavers']}" for r in log))
        print(f"   PK attempts per team-match "
              f"{pd.concat([df['Home_PKatt'], df['Away_PKatt']]).mean():.4f}; "
              f"mean goals {pd.concat([df['Home_Goals'], df['Away_Goals']]).mean():.4f}; "
              f"mean NPG est {pd.concat([df['Home_NPG'], df['Away_NPG']]).mean():.4f}")
        goals = npg = 0
        for p in LEAGUE_DIR.glob(f"{lg}_20*.json"):
            with open(p, encoding="utf-8") as f:
                for r in json.load(f)["players"]:
                    goals += int(r["goals"])
                    npg += int(r["npg"])
        att = pd.concat([df["Home_PKatt"], df["Away_PKatt"]]).sum()
        print(f"   measured penalty conversion {(goals - npg) / att:.3f}")
    # EPL Understat against production match_results, same matches.
    df, _, _ = load_league("EPL")
    with open(LEAGUE_DIR / "EPL_2025.json", encoding="utf-8") as f:
        ids = {k: v["title"] for k, v in json.load(f)["teams"].items()}
    prod, _, _ = load_inputs()
    print("production:", prod.groupby("Season").size().to_dict())
    print("Understat EPL:", df.groupby("Season").size().to_dict())
    print(f"mean total goals: production {prod[['Home_Goals', 'Away_Goals']].values.mean():.4f}, "
          f"Understat {df[['Home_Goals', 'Away_Goals']].values.mean():.4f}")
    print(f"mean npxG: production {prod[['Home_NPxG', 'Away_NPxG']].values.mean():.4f}, "
          f"Understat {df[['Home_NPxG_Adj', 'Away_NPxG_Adj']].values.mean():.4f}")
    print(f"sample ids {list(ids.items())[:3]}")


def main():
    protocol = OUT / "PROTOCOL.md"
    if not protocol.exists():
        raise SystemExit("PROTOCOL.md must exist before any arm is scored")
    sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    lines = [f"PROTOCOL.md sha256 {sha}", f"frozen settings {HYPER}", ""]

    def say(s=""):
        print(s)
        lines.append(s)

    # Control ladder: does the arm comparison survive each change of input?
    prod, money, _ = load_inputs()
    prod = prod.copy()
    ctrl = [run_arms(prod, money, SECTION24_CUTOFFS, "npg", money_weight,
                     "A production data, money prior")]
    sub, _ = signal_prior_map(prod)
    ctrl.append(run_arms(prod, sub, SECTION24_CUTOFFS, "npg", money_weight,
                         "B production data, substitute prior"))
    epl, _, _ = load_league("EPL")
    epl_prior, _ = signal_prior_map(epl)
    ctrl.append(run_arms(epl, epl_prior, SECTION24_CUTOFFS, "goals",
                         money_weight, "C Understat EPL, 2023-26"))
    ctrl.append(run_arms(epl, epl_prior, EPL_CUTOFFS, "goals", money_weight,
                         "D Understat EPL, 2020-26"))

    primary = []
    sens = []
    for lg in LEAGUES:
        df, _, _ = load_league(lg)
        prior, _ = signal_prior_map(df)
        primary.append(run_arms(df, prior, LEAGUE_CUTOFFS, "goals",
                                money_weight, lg))
        sens.append(run_arms(df, prior, LEAGUE_CUTOFFS, "goals", 0.0, lg))
    m = pd.concat(primary, ignore_index=True)
    ms = pd.concat(sens, ignore_index=True)

    head = (f"  {'':<34}{'match':>6} {'inc':>6} {'kal':>6} {'dLL':>7} "
            f"{'dLLwf':>7} {'dCS':>7} {'dCSwf':>7}")
    say("CONTROLS (Kalman minus incumbent, per team-match; §24's -0.0012 "
        "used per-cutoff selection on its data snapshot)")
    say(head)
    rows = {"controls": [], "leagues": [], "seasons": [], "tiers": []}
    for c in ctrl:
        r = line(c["Dataset"].iloc[0], c)
        rows["controls"].append(r)
        say(fmt(r))

    say()
    say("PRIMARY: five leagues, 2017-2025, frozen")
    say(head)
    for lg in LEAGUES:
        r = line(lg, m[m["Dataset"] == lg])
        rows["leagues"].append(r)
        say(fmt(r))
    pooled = line("POOLED", m)
    say(fmt(pooled))
    ci = {k: boot_ci(m, k) for k in ("frozen", "wf", "cs", "cs_wf")}
    for k, v in ci.items():
        say(f"  95% CI d_{k}: ({v[0]:+.4f}, {v[1]:+.4f})")
    say(f"  level: predicted goals per team-match "
        f"{m['inc_pred_goals'].mean() / 2:.4f} inc, "
        f"{m['kal_pred_goals'].mean() / 2:.4f} kal, actual "
        f"{m['actual_goals'].mean() / 2:.4f}")

    say()
    say("By season")
    for s, g in m.groupby("Season"):
        r = line(str(s), g)
        rows["seasons"].append(r)
        say(fmt(r))

    say()
    say("Diagnostic tiers")
    for lab, g in (("early", m[m["Early"]]), ("late", m[~m["Early"]]),
                   ("promoted", m[m["Promoted"]]), ("rest", m[~m["Promoted"]])):
        r = line(lab, g)
        rows["tiers"].append(r)
        say(fmt(r))

    say()
    say("Sensitivity: money_weight 0 (no boundary blend), not the gate")
    rs = line("POOLED mw=0", ms)
    say(fmt(rs))
    rows["sensitivity_mw0"] = rs

    d = pooled["d_frozen"]
    n_good = sum(1 for r in rows["leagues"] if r["d_frozen"] < 0)
    tests = {
        "1 pooled frozen CI below 0": ci["frozen"][1] < 0,
        f"2 improvement >= {GATE_MIN_IMPROVEMENT}": -d >= GATE_MIN_IMPROVEMENT,
        f"3 at least {GATE_MIN_LEAGUES} of 5 leagues improve":
            n_good >= GATE_MIN_LEAGUES,
        f"4 clean-sheet delta <= +{GATE_CS_TOLERANCE}":
            pooled["d_cs"] <= GATE_CS_TOLERANCE,
        f"5 walk-forward delta <= +{GATE_WF_TOLERANCE}":
            pooled["d_wf"] <= GATE_WF_TOLERANCE}
    say()
    say("GATE")
    for k, v in tests.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")
    verdict = "PASS" if all(tests.values()) else "FAIL"
    say(f"VERDICT: {verdict} (leagues improving {n_good}/5). Adopts nothing.")

    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(
        {"protocol_sha256": sha, "hyper": HYPER, "pooled": pooled, "ci": ci,
         "gate": {k: bool(v) for k, v in tests.items()}, "verdict": verdict,
         **rows}, indent=2) + "\n", encoding="utf-8")
    m.drop(columns=["Home", "Away"]).to_csv(
        OUT / "match_scores.csv.gz", index=False)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
    else:
        sys.exit(main())
