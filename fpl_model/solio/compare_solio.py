"""
Compares our forecasts with Solio's Baselines for one part-2 snapshot (or data/current), stat by stat.

Descriptive only: it shows where the two models differ, not who is right; scoring against outcomes is a
separate step once the rounds are played. Any change it suggests is pre-registered and tested on our own
history like every other model change (CLAUDE.md §5.4); never calibrate toward Solio (§236).

Solio's Baselines are one rate per player. Ours are forecasts per fixture, so each of our stats is turned into
a rate over the snapshot's rounds (all of them, or the first --rounds N), summed over fixtures and divided by
our expected minutes. Like for like (definitions checked on the GW11 capture, §236):
  * Goals/90 include penalties on both sides (ours: NP_Goals + PenGoals).
  * Assists/90 include FPL-only assists on both sides (ours: Assists, which carries them, §90).
  * DC/90 is a count of defensive contributions per 90: ours is Defcon_Mu90 on FPL's real scale
    (defcon_scaling.points_mu90, defenders x DEFENDER_POINTS_FACTOR), averaged over fixtures by xMins.
    Keepers are dropped (neither side forecasts it for them).
  * Goal and assist shares: Solio's G% is a share of NON-PENALTY goals and A% a share of all FPL assists (found
    on GW11: G/90 x Mins/M / 90 / G% is the same for every player of a team except its penalty takers, who sit
    above it; the assist version is constant to 0.1%). Ours: NP_Goals and Assists over the team's, summed over
    the rounds, among the players both sides list (we forecast fringe players Solio leaves out; the share over
    all of ours is kept as Ours_*_Share_All).
  * Solio's team non-penalty goals (assists) per match are that per-team constant, taken as the median over the
    team's players; a player's non-penalty G/90 is then G% x team NPG x 90 / Mins/M, and the rest of his G/90
    is penalty goals. Both are compared with ours (NP_Goals, PenGoals per 90).
  * Clean sheets: Solio gives one team value; ours is the team's CS_Prob_Team averaged over its fixtures,
    and, fixture-neutral, P(0 goals) against an average opponent from the team ratings. Unlike its goal
    rates, Solio's CS% follows fixtures: on GW11 it matched our mean over the first five rounds (r 0.92, the
    five gameweeks Solio shows) better than our neutral value (0.78). The printout finds the best window each
    week; compare clean sheets with --rounds set to it.
  * Minutes: Solio's Mins/M against our mean xMins_Raw (minutes.csv, FPL Review's) per fixture; our per-90
    rates use xMins (the recorded clock, §176), which is how our points are built.
  * Per-90 rates on Solio's side are blank under its minutes cut-off (about 18 expected minutes); those rows
    are left out of every per-90 comparison, as are rows where we forecast under MIN_XMINS_PER_MATCH.

Solio's baselines are fixture-neutral (the user's reading, 2026-11-13, and the GW11 data agree: Solio's team
non-penalty goals correlate 0.86 with our neutral ratings, 0.80 with our forecast over GW11-20), so team
comparisons lead with our neutral values. Our player rates come from the forecast over the rounds, where ten
fixtures largely average out; the printout keeps both team versions so a shorter --rounds can be judged.

Outputs in data/solio/comparisons/<snapshot name or "current">/:
    players.csv          one row per player: ours v Solio for every stat, and the differences
    summary.csv          by position and stat: players, means, ratio of means, correlation, slope (ours on Solio),
                         mean absolute difference; "regulars" (both sides >= 60 minutes a match) and "all"
    teams.csv            team goals per match and clean-sheet probability: Solio, ours neutral, ours forecast
    top_differences.csv  the largest gaps among regulars, per stat

Run locally:
    python -m fpl_model.solio.compare_solio                      # latest pre-deadline GW snapshot with usable Solio data
    python -m fpl_model.solio.compare_solio --snapshot <folder>
    python -m fpl_model.solio.compare_solio --current            # data/current, before a snapshot exists
    python -m fpl_model.solio.compare_solio --current --rounds 5 # only the first five rounds of our forecast
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model import paths

ROOT = paths.ROOT
from fpl_model.atomic_write import to_csv_atomic  # noqa: E402
from fpl_model.defcon_scaling import points_mu90  # noqa: E402
from fpl_model.league_rates import mean_npg, penalty_rates  # noqa: E402

DATA = paths.CURRENT
SNAPSHOTS = paths.SNAPSHOTS
OUT_ROOT = paths.SOLIO / "comparisons"

REGULAR_MINUTES = 60       # minutes a match on both sides for the "regulars" group
MIN_XMINS_PER_MATCH = 18   # our side's floor for a per-90 rate, matching Solio's own cut-off (§236)
TOP_N = 20

# stat -> (ours column, Solio column, per-90 rate?, positions compared)
STATS = {
    "pts90": ("Ours_Pts90", "Solio_Pts90", True, "GDMF"),
    "mins": ("Ours_Mins", "Solio_Mins", False, "GDMF"),
    "g90": ("Ours_G90", "Solio_G90", True, "DMF"),
    "npg90": ("Ours_NPG90", "Solio_NPG90", True, "DMF"),
    "pen_g90": ("Ours_PenG90", "Solio_PenG90", True, "DMF"),
    "a90": ("Ours_A90", "Solio_A90", True, "GDMF"),
    "goal_share": ("Ours_Goal_Share", "Solio_Goal_Share", False, "DMF"),
    "assist_share": ("Ours_Assist_Share", "Solio_Assist_Share", False, "DMF"),
    "bon90": ("Ours_Bon90", "Solio_Bon90", True, "GDMF"),
    "dc90": ("Ours_DC90", "Solio_DC90", True, "DMF"),
}


def find_inputs(args):
    if args.current:
        return DATA, DATA / "solio_baselines.csv", DATA / "solio_status.json", "current"
    if args.snapshot:
        snap = Path(args.snapshot)
    else:
        candidates = sorted(p.parent.parent for p in SNAPSHOTS.glob("*/GW*/*_pre_deadline_partall/solio/solio_status.json")
                            if json.loads(p.read_text(encoding="utf-8")).get("usable_for_comparison"))
        if not candidates:
            sys.exit("No pre-deadline snapshot with usable Solio data yet; use --current or run part 2.")
        snap = candidates[-1]
    return snap / "outputs", snap / "solio" / "solio_baselines.csv", snap / "solio" / "solio_status.json", snap.name


def load(args):
    outputs, solio_path, status_path, name = find_inputs(args)
    needed = [outputs / "predicted_components.csv", outputs / "simulation_features.csv",
              outputs / "team_ratings_current.csv", solio_path, status_path]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        sys.exit(f"Missing {', '.join(missing)}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not status.get("usable_for_comparison"):
        sys.exit(f"Solio data for {name} is not usable: {status.get('reasons')}")
    comp = pd.read_csv(outputs / "predicted_components.csv", encoding="utf-8-sig")
    sim = pd.read_csv(outputs / "simulation_features.csv", encoding="utf-8-sig")
    ratings = pd.read_csv(outputs / "team_ratings_current.csv", encoding="utf-8-sig")
    solio = pd.read_csv(solio_path, encoding="utf-8-sig")
    first = int(comp["Round"].min())
    if int(status["gameweek"]) != first:
        sys.exit(f"Solio was captured for GW{status['gameweek']}, but our forecast starts at GW{first}")
    if args.rounds:
        keep = sorted(comp["Round"].unique())[:args.rounds]
        comp, sim = comp[comp["Round"].isin(keep)], sim[sim["Round"].isin(keep)]
    return comp, sim, ratings, solio, status, name


def our_rates(comp, sim):
    """Our forecast over the rounds as one rate per player, on Solio's definitions."""
    keys = ["Player", "Team"]
    # simulation_features.csv keys players by full name in Player; predicted_components.csv by Web_Name.
    sim = sim.drop(columns="Player").rename(columns={"Web_Name": "Player"})
    teams = pd.read_csv(DATA / "players_current.csv", encoding="utf-8-sig").drop_duplicates("Team").set_index("Team")["Team_Short"]
    if not sim["Team"].isin(comp["Team"]).all():
        sim["Team"] = sim["Team"].map(teams)
    sim = sim.assign(DC_Mu90_Points=points_mu90(sim["Defcon_Mu90"], sim["Pos"]))
    sim["_dc_x_mins"] = sim["DC_Mu90_Points"] * sim["xMins"]
    dc = sim.groupby(keys).agg(_dc=("_dc_x_mins", "sum"), _dcw=("xMins", "sum"))
    o = comp.groupby(keys + ["Pos", "Price"], as_index=False).agg(
        Fixtures=("Opponent", "count"), xMins=("xMins", "sum"), xMins_Raw=("xMins_Raw", "sum"),
        Goals=("Goals", "sum"), NP_Goals=("NP_Goals", "sum"), PenGoals=("PenGoals", "sum"), Assists=("Assists", "sum"),
        Bonus=("BonusPts", "sum"), Points=("TotalPts", "sum"))
    o = o.merge(dc, left_on=keys, right_index=True, how="left")
    if o.loc[o["Pos"].ne("G") & o["xMins"].gt(0), "_dc"].isna().any():
        raise ValueError("simulation_features.csv doesn't cover every outfield player in predicted_components.csv")
    per90 = 90 / o["xMins"].where(o["xMins"] > 0)
    o["Ours_Mins"] = o["xMins_Raw"] / o["Fixtures"]
    o["Ours_Pts90"] = o["Points"] * per90
    o["Ours_G90"] = o["Goals"] * per90
    o["Ours_NPG90"] = o["NP_Goals"] * per90
    o["Ours_PenG90"] = o["PenGoals"] * per90
    o["Ours_A90"] = o["Assists"] * per90
    o["Ours_Bon90"] = o["Bonus"] * per90
    o["Ours_DC90"] = (o["_dc"] / o["_dcw"].where(o["_dcw"] > 0)).where(o["Pos"].ne("G"))
    team = o.groupby("Team")[["NP_Goals", "Assists"]].transform("sum")
    o["Ours_Goal_Share"] = o["NP_Goals"] / team["NP_Goals"]
    o["Ours_Assist_Share"] = o["Assists"] / team["Assists"]
    return o.drop(columns=["_dc", "_dcw"])


def solio_team_rates(solio):
    """Solio's team non-penalty goals and assists per match: G/90 x Mins/M / 90 / share, median over the team.

    Penalty takers sit above the goal constant (their G/90 includes penalties, G% doesn't), so the median is
    taken; the spread column shows how far the highest player sits above it.
    """
    s = solio[~solio["Solio_Per90_Suppressed"] & solio["Team_Short"].notna()]
    out = {}
    for stat, rate, share in (("NPG", "Solio_G90", "Solio_Goal_Share"), ("Assists", "Solio_A90", "Solio_Assist_Share")):
        ok = s[s[share].gt(0.02) & s[rate].gt(0)]
        implied = (ok[rate] * ok["Solio_Mins"] / 90 / ok[share]).groupby(ok["Team_Short"])
        out[f"Solio_Team_{stat}"] = implied.median()
        out[f"Solio_Team_{stat}_Max_Over"] = implied.max() - implied.median()
    return pd.DataFrame(out)


def solio_split_goals(solio):
    """Add Solio's non-penalty and penalty goals per 90 (blank where its per-90 values are suppressed)."""
    team = solio_team_rates(solio)
    s = solio.copy()
    npg_team = s["Team_Short"].map(team["Solio_Team_NPG"])
    s["Solio_NPG90"] = (s["Solio_Goal_Share"] * npg_team * 90 / s["Solio_Mins"]).where(~s["Solio_Per90_Suppressed"])
    s["Solio_PenG90"] = (s["Solio_G90"] - s["Solio_NPG90"]).clip(lower=0)
    return s


def join(ours, solio):
    s = solio.dropna(subset=["Web_Name"]).rename(columns={"Web_Name": "Player", "Team_Short": "Team"})
    s = s[["Player", "Team", "Solio_Pos", "Solio_Status", "Solio_Per90_Suppressed", *[v[1] for v in STATS.values()],
           "Solio_Team_CS"]]
    joined = ours.merge(s, on=["Player", "Team"], how="outer", indicator=True, validate="one_to_one")
    joined["Source"] = joined.pop("_merge").astype(str).map({"both": "both", "left_only": "ours_only", "right_only": "solio_only"})
    both = joined["Source"].eq("both")
    # Shares over the players both sides list: we forecast fringe players Solio leaves out, and they would take
    # part of our team totals. Ours_*_Share_All keeps the share over every player we forecast.
    for share, col in (("Ours_Goal_Share", "NP_Goals"), ("Ours_Assist_Share", "Assists")):
        joined[f"{share}_All"] = joined[share]
        team = joined[both].groupby("Team")[col].transform("sum")
        joined[share] = (joined.loc[both, col] / team).reindex(joined.index)
    suppressed = joined["Solio_Per90_Suppressed"].astype("boolean").fillna(True).astype(bool)
    joined["Rate_OK"] = both & ~suppressed & (joined["Ours_Mins"] >= MIN_XMINS_PER_MATCH)
    joined["Regular"] = joined["Rate_OK"] & (joined["Ours_Mins"] >= REGULAR_MINUTES) & (joined["Solio_Mins"] >= REGULAR_MINUTES)
    for stat, (o, s, _, _) in STATS.items():
        joined[f"Diff_{stat}"] = joined[o] - joined[s]
    return joined


def summarise(j):
    rows = []
    for group, mask in (("regulars", j["Regular"]), ("all", j["Source"].eq("both"))):
        for stat, (o, s, rate, positions) in STATS.items():
            base = j[mask & j["Pos"].isin(list(positions))]
            if rate:
                base = base[base["Rate_OK"]]
            for pos, g in [*base.groupby("Pos"), ("all", base)]:
                g = g.dropna(subset=[o, s])
                if len(g) < 3:
                    continue
                a, b = g[o], g[s]
                slope = float(np.polyfit(b, a, 1)[0]) if b.std() > 0 else np.nan
                rows.append({"Group": group, "Stat": stat, "Pos": pos, "Players": len(g),
                             "Ours_Mean": a.mean(), "Solio_Mean": b.mean(),
                             "Ratio": a.mean() / b.mean() if b.mean() else np.nan,
                             "Corr": a.corr(b) if a.std() > 0 and b.std() > 0 else np.nan,
                             "Slope_Ours_on_Solio": slope, "Mean_Abs_Diff": (a - b).abs().mean()})
    order = {"G": 0, "D": 1, "M": 2, "F": 3, "all": 4}
    out = pd.DataFrame(rows)
    out["_o"] = out["Pos"].map(order)
    out["_s"] = out["Stat"].map({k: i for i, k in enumerate(STATS)})
    return out.sort_values(["Group", "_s", "_o"], ascending=[False, True, True]).drop(columns=["_o", "_s"]).round(4)


def teams(comp, ratings, solio, players):
    """Team non-penalty goals and assists per match and clean sheets: Solio, ours fixture-neutral, ours over the rounds."""
    t = solio_team_rates(solio)
    t["Solio_CS"] = solio.groupby("Team_Short")["Solio_Team_CS"].first()

    # Ours, fixture-neutral: non-penalty goals v an average opponent at a neutral venue (predict_fixture_goals.py's
    # formula without home advantage); clean sheet = P(0 goals) with penalties added at the league rate, as
    # CS_Prob_Team counts them.
    attempts_per_npg, conversion = penalty_rates()
    pen = 1 + attempts_per_npg * conversion
    avg = mean_npg()
    r = ratings.copy()
    r["Ours_Neutral_NPG"] = r["Attack_Rating"] * r["Defense_Rating"].mean() / avg
    r["Ours_Neutral_CS"] = np.exp(-r["Attack_Rating"].mean() * r["Defense_Rating"] / avg * pen)
    short = players.drop_duplicates("Team").set_index("Team")["Team_Short"]
    r["Team_Short"] = r["Team"].map(short)
    t = t.join(r.set_index("Team_Short")[["Ours_Neutral_NPG", "Ours_Neutral_CS"]])

    # Ours over the rounds: summed player non-penalty goals and assists per fixture, CS_Prob_Team per fixture.
    fixtures = comp.drop_duplicates(["Team", "Round", "Opponent", "Venue"])
    n = fixtures.groupby("Team").size()
    t["Ours_Forecast_NPG"] = comp.groupby("Team")["NP_Goals"].sum() / n
    t["Ours_Forecast_Assists"] = comp.groupby("Team")["Assists"].sum() / n
    t["Ours_Forecast_CS"] = fixtures.groupby("Team")["CS_Prob_Team"].mean()
    return t.reset_index(names="Team").round(4)


def tops(j):
    reg = j[j["Regular"]]
    out = []
    for stat, (o, s, _, positions) in STATS.items():
        d = reg[reg["Pos"].isin(list(positions))].dropna(subset=[o, s])
        d = d.reindex(d[f"Diff_{stat}"].abs().sort_values(ascending=False).index).head(TOP_N)
        out.append(d.assign(Stat=stat, Ours=d[o], Solio=d[s], Diff=d[f"Diff_{stat}"])
                   [["Stat", "Player", "Team", "Pos", "Price", "Ours_Mins", "Solio_Mins", "Ours", "Solio", "Diff"]])
    return pd.concat(out).round(4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot")
    parser.add_argument("--current", action="store_true")
    parser.add_argument("--rounds", type=int, help="compare only the first N rounds of our forecast")
    args = parser.parse_args()
    comp, sim, ratings, solio, status, name = load(args)
    players = pd.read_csv(DATA / "players_current.csv", encoding="utf-8-sig")
    solio = solio_split_goals(solio)
    j = join(our_rates(comp, sim), solio)
    summary = summarise(j)
    team_table = teams(comp, ratings, solio, players)
    top = tops(j)

    rounds = sorted(comp["Round"].unique())
    out_dir = OUT_ROOT / (name if not args.rounds else f"{name}_first{args.rounds}")
    out_dir.mkdir(parents=True, exist_ok=True)
    to_csv_atomic(j.round(4), out_dir / "players.csv", index=False, encoding="utf-8-sig")
    to_csv_atomic(summary, out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    to_csv_atomic(team_table, out_dir / "teams.csv", index=False, encoding="utf-8-sig")
    to_csv_atomic(top, out_dir / "top_differences.csv", index=False, encoding="utf-8-sig")

    counts = j["Source"].value_counts().to_dict()
    print(f"Solio GW{status['gameweek']} v ours, {name}, rounds GW{rounds[0]}-GW{rounds[-1]}")
    print(f"  players: both {counts.get('both', 0)}, ours only {counts.get('ours_only', 0)}, "
          f"Solio only {counts.get('solio_only', 0)}; regulars (>= {REGULAR_MINUTES} min a match on both sides) "
          f"{int(j['Regular'].sum())}")
    tc = team_table.dropna()
    print("\nTeams, Solio (fixture-neutral) v ours neutral and over the rounds; correlations:")
    print(f"  non-penalty goals a match: fixture-neutral {tc['Solio_Team_NPG'].corr(tc['Ours_Neutral_NPG']):.3f}, "
          f"over the rounds {tc['Solio_Team_NPG'].corr(tc['Ours_Forecast_NPG']):.3f}; "
          f"clean sheets: neutral {tc['Solio_CS'].corr(tc['Ours_Neutral_CS']):.3f}, "
          f"over the rounds {tc['Solio_CS'].corr(tc['Ours_Forecast_CS']):.3f}")
    print(f"  means: NPG Solio {tc['Solio_Team_NPG'].mean():.3f}, ours neutral {tc['Ours_Neutral_NPG'].mean():.3f}, "
          f"over the rounds {tc['Ours_Forecast_NPG'].mean():.3f}; assists Solio {tc['Solio_Team_Assists'].mean():.3f}, "
          f"ours {tc['Ours_Forecast_Assists'].mean():.3f}; CS Solio {tc['Solio_CS'].mean():.3f}, "
          f"ours neutral {tc['Ours_Neutral_CS'].mean():.3f}, over the rounds {tc['Ours_Forecast_CS'].mean():.3f}")
    # Solio's CS% isn't fixture-neutral like its goal rates: on GW11 it tracked our mean over the first five rounds
    # (r 0.92) far better than our neutral value (0.78) or any single round. Find the window each week.
    by_window = {}
    fx = comp.drop_duplicates(["Team", "Round", "Opponent", "Venue"])
    for k in range(1, len(rounds) + 1):
        x = fx[fx["Round"].isin(rounds[:k])].groupby("Team")["CS_Prob_Team"].mean()
        by_window[k] = team_table.set_index("Team")["Solio_CS"].corr(x)
    best = max(by_window, key=by_window.get)
    print("  Solio CS% v our mean CS over the first k rounds, r by k: "
          + ", ".join(f"{k}: {v:.2f}" for k, v in by_window.items()) + f"; closest at k = {best}")
    def slope(a, b):
        return float(np.polyfit(np.log(tc[b]), np.log(tc[a]), 1)[0])
    print(f"  team spread, log slope ours on Solio (below 1 = ours narrower): NPG neutral {slope('Ours_Neutral_NPG', 'Solio_Team_NPG'):.2f}, "
          f"over the rounds {slope('Ours_Forecast_NPG', 'Solio_Team_NPG'):.2f}; CS neutral {slope('Ours_Neutral_CS', 'Solio_CS'):.2f}, "
          f"over the rounds {slope('Ours_Forecast_CS', 'Solio_CS'):.2f}")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\nRegulars, by stat (ratio = our mean / Solio's; slope = ours on Solio, below 1 = ours narrower):")
        print(summary[summary["Group"].eq("regulars")].drop(columns="Group").to_string(index=False))
        print("\nTeams:")
        print(team_table.to_string(index=False))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
