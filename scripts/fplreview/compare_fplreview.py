"""
Compares our forecasts with FPL Review's for one part-2 snapshot (or data/current), stat by stat.

Descriptive only: it shows where the two models differ. Who is right is a separate question,
answered by scoring both against outcomes once the rounds are played. Any change it suggests is
pre-registered and tested on our own history like every other model change (CLAUDE.md §5.4);
never calibrate toward FPL Review.

Like for like:
  * Goals include penalties on both sides; assists include FPL-only assists on both sides.
  * Clean sheets are P(clean sheet and 60+ minutes): ours is CS_Prob_Team x P60. Forwards earn no
    clean-sheet points and FPL Review shows them as 0, so forwards are dropped from that stat.
  * Defcon is the threshold probability; keepers are excluded (they can't earn it).
  * Minutes: FPL Review projects at its own xMins (minutes.csv, our xMins_Raw); we rescale them by
    the recorded clock (about 0.95, §176). "ratio_total" compares totals; "ratio_per90" divides each
    side by its own minutes, so it isolates the rate. Their gap is the clock.
  * Double gameweeks: both sides are summed per player and round.

Outputs in data/fplreview/comparisons/<snapshot name or "current">/:
    player_rounds.csv   one row per player and round, ours v theirs for every stat
    summary.csv         by position, stat and horizon (next round, later rounds)
    teams.csv           team goals per round: our team model, our players summed, FPL Review's players summed
    top_differences.csv the largest per-player gaps in points, goals and assists over the snapshot's rounds
    relevant_players.csv FPL-relevant players (mean xMins >= 60 and >= 3 points per round over the snapshot's rounds in
                        either model), the points gap split into goals, assists, clean sheets, defcon
                        and other (appearance, bonus, saves, goals conceded, cards; FPL Review's
                        "other" is its total minus the four stats in FPL points)
    relevant_segments.csv the same gaps averaged by position, xMins band, price band, PL evidence
                        (career Premier League minutes) and team
    team_strengths.csv  our team ratings v FPL Review's team table (GF, GA, GD per match, neutral venue,
                        average opponent, penalties included), when Team Strengths.xlsx was pasted
    team_spread.csv     per team: both sides' goals and clean-sheet probability; the printed slopes
                        (log scale, ours on theirs) say how much narrower our team spread is

Run locally:
    python compare_fplreview.py                      # latest GW snapshot with FPL Review data
    python compare_fplreview.py --snapshot <folder>
    python compare_fplreview.py --current            # data/current, before a snapshot exists
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from atomic_write import to_csv_atomic  # noqa: E402
from league_rates import mean_npg, penalty_rates  # noqa: E402

DATA = ROOT / "data" / "current"
SNAPSHOTS = ROOT / "data" / "snapshots"
OUT_ROOT = ROOT / "data" / "fplreview" / "comparisons"

STATS = {  # ours column -> FPL Review column
    "TotalPts": "Points", "Goals": "Goals", "Assists": "Assists", "CS_Prob": "CleanSheet", "Defcon_Prob": "Defcon",
}
PTS_PER_GOAL = {"G": 10, "D": 6, "M": 5, "F": 4}
CS_PTS = {"G": 4, "D": 4, "M": 1, "F": 0}
RELEVANT_XMINS = 60     # mean over the snapshot's rounds
# Total in either model, per round of the snapshot: 15 over five rounds, 30 over ten (horizon 10 from GW11, §215).
RELEVANT_POINTS_PER_ROUND = 3
PARTS = ("goal", "assist", "cs", "defcon", "other")
LABEL = {"TotalPts": "points", "Goals": "goals", "Assists": "assists", "CS_Prob": "clean_sheet", "Defcon_Prob": "defcon"}


def find_inputs(args):
    if args.current:
        return DATA / "predicted_components.csv", DATA / "fplreview_projections.csv", DATA / "fplreview_status.json", \
            DATA / "fixture_goal_predictions.csv", DATA / "penalty_predictions.csv", "current"
    if args.snapshot:
        snap = Path(args.snapshot)
    else:
        candidates = sorted(p.parent for p in SNAPSHOTS.glob("*/GW*/*_pre_deadline_partall/fplreview/fplreview_status.json"))
        if not candidates:
            sys.exit("No pre-deadline snapshot with FPL Review data yet; use --current or run part 2.")
        snap = candidates[-1].parent
    return snap / "outputs" / "predicted_components.csv", snap / "fplreview" / "fplreview_projections.csv", \
        snap / "fplreview" / "fplreview_status.json", snap / "outputs" / "fixture_goal_predictions.csv", \
        snap / "outputs" / "penalty_predictions.csv", snap.name


def load(args):
    ours_path, theirs_path, status_path, fixtures_path, pens_path, name = find_inputs(args)
    for p in (ours_path, theirs_path, status_path):
        if not p.is_file():
            sys.exit(f"Missing {p}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not status.get("usable_for_comparison"):
        sys.exit(f"FPL Review data for {name} is not usable: {status.get('reasons')}")
    ours = pd.read_csv(ours_path, encoding="utf-8-sig")
    theirs = pd.read_csv(theirs_path, encoding="utf-8-sig")
    status["_dir"] = status_path.parent
    return ours, theirs, status, fixtures_path, pens_path, name


def player_rounds(ours, theirs):
    keys = ["Player", "Team", "Round"]
    o = ours.groupby(keys + ["Pos", "Price"], as_index=False).agg(
        **{c: (c, "sum") for c in [*STATS, "xMins", "xMins_Raw", "PenGoals"]}, Fixtures=("Opponent", "count"))
    t = theirs.dropna(subset=["Web_Name"]).rename(columns={"Web_Name": "Player", "Team_Short": "Team"})
    t = t[["Player", "Team", "Round", *STATS.values(), "FPLR_xMins_First"]]
    t = t.rename(columns={v: f"FPLR_{v}" for v in STATS.values()})
    joined = o.merge(t, on=keys, how="outer", indicator=True, validate="one_to_one")
    first = joined["Round"].min()
    joined["Horizon"] = np.where(joined["Round"].eq(first), "next", "later")
    return joined


def summarise(pr):
    both = pr[pr["_merge"].eq("both")]
    rows = []
    for (pos, horizon), g in [*both.groupby(["Pos", "Horizon"]), *[((pos, "all"), g) for pos, g in both.groupby("Pos")]]:
        for ours_col, theirs_col in STATS.items():
            if (ours_col == "CS_Prob" and pos == "F") or (ours_col == "Defcon_Prob" and pos == "G"):
                continue
            a, b = g[ours_col], g[f"FPLR_{theirs_col}"]
            live = g[(g["xMins_Raw"] > 0)]
            # Per 90 at each side's own minutes: ours at xMins, theirs at FPL Review's xMins (= xMins_Raw).
            ours90 = live[ours_col].sum() / live["xMins"].sum() * 90 if live["xMins"].sum() else np.nan
            theirs90 = live[f"FPLR_{theirs_col}"].sum() / live["xMins_Raw"].sum() * 90 if live["xMins_Raw"].sum() else np.nan
            rows.append({
                "Pos": pos, "Horizon": horizon, "Stat": LABEL[ours_col], "Player_Rounds": len(g),
                "Ours_Sum": round(a.sum(), 2), "FPLR_Sum": round(b.sum(), 2),
                "ratio_total": round(a.sum() / b.sum(), 3) if b.sum() else np.nan,
                "ratio_per90": round(ours90 / theirs90, 3) if theirs90 else np.nan,
                "corr": round(a.corr(b), 3) if a.std() and b.std() else np.nan,
                "mean_abs_diff": round((a - b).abs().mean(), 4),
            })
    out = pd.DataFrame(rows)
    order = {"G": 0, "D": 1, "M": 2, "F": 3}
    hz = {"next": 0, "later": 1, "all": 2}
    return out.sort_values(["Stat", "Horizon", "Pos"], key=lambda s: s.map(order) if s.name == "Pos" else
                           s.map(hz) if s.name == "Horizon" else s).reset_index(drop=True)


def team_goals(pr, fixtures_path, pens_path, ours):
    teams = pr[pr["_merge"].eq("both")].groupby(["Team", "Round"], as_index=False).agg(
        Ours_Players_Goals=("Goals", "sum"), FPLR_Players_Goals=("FPLR_Goals", "sum"), Ours_PenGoals=("PenGoals", "sum"))
    if fixtures_path.is_file():
        fx = pd.read_csv(fixtures_path, encoding="utf-8-sig")
        long = pd.concat([fx.rename(columns={"Home": "Team_Full", "Home_Pred_NPG": "Team_Model_NPG"})[["Round", "Team_Full", "Team_Model_NPG"]],
                          fx.rename(columns={"Away": "Team_Full", "Away_Pred_NPG": "Team_Model_NPG"})[["Round", "Team_Full", "Team_Model_NPG"]]])
        long = long.groupby(["Round", "Team_Full"], as_index=False)["Team_Model_NPG"].sum()
        players = pd.read_csv(DATA / "players_current.csv", encoding="utf-8-sig")
        long["Team"] = long["Team_Full"].map(dict(zip(players["Team"], players["Team_Short"])))
        teams = teams.merge(long[["Team", "Round", "Team_Model_NPG"]], on=["Team", "Round"], how="left")
    return teams.round(3)


def stat_points(d, goals, assists, cs, defcon, total, prefix):
    """FPL points of each compared stat; 'other' is what the total leaves."""
    d[f"{prefix}goal"] = d[goals] * d["Pos"].map(PTS_PER_GOAL)
    d[f"{prefix}assist"] = d[assists] * 3
    d[f"{prefix}cs"] = d[cs] * d["Pos"].map(CS_PTS)
    d[f"{prefix}defcon"] = np.where(d["Pos"].eq("G"), 0.0, 2 * d[defcon])
    d[f"{prefix}other"] = d[total] - d[[f"{prefix}{c}" for c in PARTS[:-1]]].sum(axis=1)


def relevant_players(pr):
    both = pr[pr["_merge"].eq("both")].copy()
    stat_points(both, "Goals", "Assists", "CS_Prob", "Defcon_Prob", "TotalPts", "o_")
    stat_points(both, "FPLR_Goals", "FPLR_Assists", "FPLR_CleanSheet", "FPLR_Defcon", "FPLR_Points", "f_")
    cols = ["TotalPts", "FPLR_Points", "xMins_Raw", *[f"{s}{c}" for s in ("o_", "f_") for c in PARTS]]
    f = both.groupby(["Player", "Team", "Pos", "Price"], as_index=False)[cols].sum()
    f["Mean_xMins"] = f["xMins_Raw"] / both["Round"].nunique()
    f = f[(f["Mean_xMins"] >= RELEVANT_XMINS) & (f[["TotalPts", "FPLR_Points"]].max(axis=1) >= RELEVANT_POINTS_PER_ROUND * both["Round"].nunique())].copy()
    f["Gap"] = f["TotalPts"] - f["FPLR_Points"]
    f["Gap_Pct"] = f["Gap"] / f["FPLR_Points"]
    for c in PARTS:
        f[f"Gap_{c}"] = f[f"o_{c}"] - f[f"f_{c}"]
    # Premier League evidence: career minutes in player_match_stats.csv, keyed through players_current.csv.
    players = pd.read_csv(DATA / "players_current.csv", encoding="utf-8-sig")
    full = players.drop_duplicates(["Web_Name", "Team_Short"]).set_index(["Web_Name", "Team_Short"])["Player"]
    stats = pd.read_csv(DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False, usecols=["Player", "Minutes"])
    minutes = stats.groupby("Player")["Minutes"].sum()
    f["PL_Minutes"] = [minutes.get(full.get((p, t)), 0.0) for p, t in zip(f["Player"], f["Team"])]
    f["xMins_Band"] = pd.cut(f["Mean_xMins"], [RELEVANT_XMINS - 1e-9, 70, 80, 88, 200],
                             labels=["60-70", "70-80", "80-88", "88+"]).astype(str)
    f["Price_Band"] = pd.cut(f["Price"], [0, 5.0, 6.0, 8.0, 99], labels=["<=5.0", "5.1-6.0", "6.1-8.0", "8.1+"]).astype(str)
    f["Evidence"] = pd.cut(f["PL_Minutes"], [-1, 1500, 5000, 15000, 1e9], labels=["<1.5k", "1.5-5k", "5-15k", "15k+"]).astype(str)
    keep = ["Player", "Team", "Pos", "Price", "Mean_xMins", "PL_Minutes", "TotalPts", "FPLR_Points", "Gap", "Gap_Pct",
            *[f"Gap_{c}" for c in PARTS], "xMins_Band", "Price_Band", "Evidence"]
    f = f[keep].sort_values("Gap").reset_index(drop=True)
    segments = []
    for by in ("Pos", "xMins_Band", "Price_Band", "Evidence", "Team"):
        g = f.groupby(by).agg(Players=("Gap", "size"), Gap=("Gap", "mean"), Gap_Pct=("Gap_Pct", "mean"),
                              **{f"Gap_{c}": (f"Gap_{c}", "mean") for c in PARTS})
        segments.append(g.reset_index().rename(columns={by: "Value"}).assign(Segment=by))
    segments = pd.concat(segments)[["Segment", "Value", "Players", "Gap", "Gap_Pct", *[f"Gap_{c}" for c in PARTS]]]
    return f.round(3), segments.round(3)


def team_spread(pr, teams, ours):
    """Team goals and clean-sheet probability per team, both sides; log-scale slopes (1 = same spread)."""
    t = teams.groupby("Team").sum(numeric_only=True)
    rows = pr[pr["_merge"].eq("both") & pr["Fixtures"].eq(1) & pr["Pos"].isin(["G", "D", "M"])].merge(
        ours[["Player", "Team", "Round", "CS_Prob_Team", "P60"]], on=["Player", "Team", "Round"])
    rows = rows[rows["P60"] >= 0.9]  # for nailed players FPL Review's CS / P60 is close to its team CS
    rows = rows.assign(FPLR_Team_CS=rows["FPLR_CleanSheet"] / rows["P60"])
    cs = rows.groupby(["Team", "Round"]).agg(Ours_Team_CS=("CS_Prob_Team", "first"), FPLR_Team_CS=("FPLR_Team_CS", "median"))
    out = pd.DataFrame({"Ours_Players_Goals": t["Ours_Players_Goals"], "FPLR_Players_Goals": t["FPLR_Players_Goals"]})
    if "Team_Model_NPG" in t:
        out["Ours_Team_Model_Goals"] = t["Team_Model_NPG"] + t["Ours_PenGoals"]
    out = out.join(cs.groupby("Team").mean()).reset_index()
    slopes = {}
    x = np.log(out["FPLR_Players_Goals"])
    for col in ("Ours_Players_Goals", "Ours_Team_Model_Goals"):
        if col in out:
            slopes[f"{col} on FPLR_Players_Goals"] = float(np.polyfit(x, np.log(out[col]), 1)[0])
    ok = out["FPLR_Team_CS"].gt(0) & out["Ours_Team_CS"].gt(0)
    slopes["Ours_Team_CS on FPLR_Team_CS"] = float(np.polyfit(np.log(out.loc[ok, "FPLR_Team_CS"]),
                                                              np.log(out.loc[ok, "Ours_Team_CS"]), 1)[0])
    return out.round(3), slopes


def team_strengths(ratings_path, fplr_path):
    """Our team ratings against FPL Review's team table, both as goals per match v an average opponent at a neutral venue.

    Ours: Attack x mean(Defense) / league mean NPG for goals for, mean(Attack) x Defense / league mean for goals
    against (predict_fixture_goals.py's formula without home advantage), then penalties added at the league
    rate (attempts per NPG x conversion), since FPL Review's GF and GA are total goals.
    """
    if not (ratings_path.is_file() and fplr_path.is_file()):
        return None, {}
    ratings = pd.read_csv(ratings_path, encoding="utf-8-sig")
    fplr = pd.read_csv(fplr_path, encoding="utf-8-sig")
    attempts_per_npg, conversion = penalty_rates()
    pen = 1 + attempts_per_npg * conversion
    avg = mean_npg()
    ratings["Ours_GF"] = ratings["Attack_Rating"] * ratings["Defense_Rating"].mean() / avg * pen
    ratings["Ours_GA"] = ratings["Attack_Rating"].mean() * ratings["Defense_Rating"] / avg * pen
    ratings["Ours_GD"] = ratings["Ours_GF"] - ratings["Ours_GA"]
    t = fplr.merge(ratings[["Team", "Ours_GF", "Ours_GA", "Ours_GD"]], on="Team", how="left", validate="one_to_one")
    if t["Ours_GF"].isna().any():
        raise ValueError(f"teams missing from our ratings: {t.loc[t['Ours_GF'].isna(), 'Team'].tolist()}")
    for c in ("GF", "GA", "GD"):
        t[f"{c}_Diff"] = t[f"Ours_{c}"] - t[f"FPLR_{c}"]
    t["Rank_Ours"] = t["Ours_GD"].rank(ascending=False).astype(int)
    t["Rank_FPLR"] = t["FPLR_GD"].rank(ascending=False, method="min").astype(int)
    summary = {
        "mean GF ours / FPLR": t["Ours_GF"].mean() / t["FPLR_GF"].mean(),
        "mean GA ours / FPLR": t["Ours_GA"].mean() / t["FPLR_GA"].mean(),
        "log slope GF ours on FPLR": float(np.polyfit(np.log(t["FPLR_GF"]), np.log(t["Ours_GF"]), 1)[0]),
        "log slope GA ours on FPLR": float(np.polyfit(np.log(t["FPLR_GA"]), np.log(t["Ours_GA"]), 1)[0]),
        "slope GD ours on FPLR": float(np.polyfit(t["FPLR_GD"], t["Ours_GD"], 1)[0]),
        "corr GF": t["Ours_GF"].corr(t["FPLR_GF"]), "corr GA": t["Ours_GA"].corr(t["FPLR_GA"]),
        "corr GD": t["Ours_GD"].corr(t["FPLR_GD"]),
    }
    cols = ["Team", "Team_Short", "Ours_GF", "FPLR_GF", "GF_Diff", "Ours_GA", "FPLR_GA", "GA_Diff",
            "Ours_GD", "FPLR_GD", "GD_Diff", "Rank_Ours", "Rank_FPLR"]
    return t[cols].sort_values("FPLR_GD", ascending=False).round(3), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot")
    parser.add_argument("--current", action="store_true")
    args = parser.parse_args()
    ours, theirs, status, fixtures_path, pens_path, name = load(args)
    pr = player_rounds(ours, theirs)
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    only_ours = pr[pr["_merge"].eq("left_only")]
    only_theirs = pr[pr["_merge"].eq("right_only")]
    summary = summarise(pr)
    teams = team_goals(pr, fixtures_path, pens_path, ours)
    both = pr[pr["_merge"].eq("both")]
    five = both.groupby(["Player", "Team", "Pos", "Price"], as_index=False)[
        [*STATS, *[f"FPLR_{v}" for v in STATS.values()], "xMins", "xMins_Raw"]].sum()
    tops = []
    for ours_col in ("TotalPts", "Goals", "Assists"):
        d = five.assign(Stat=LABEL[ours_col], Ours=five[ours_col], FPLR=five[f"FPLR_{STATS[ours_col]}"])
        d["Diff"] = d["Ours"] - d["FPLR"]
        d = d.reindex(d["Diff"].abs().sort_values(ascending=False).index).head(25)
        tops.append(d[["Stat", "Player", "Team", "Pos", "Price", "xMins_Raw", "Ours", "FPLR", "Diff"]])
    tops = pd.concat(tops).round(3)

    to_csv_atomic(pr.drop(columns="_merge").assign(Source=pr["_merge"].astype(str).map(
        {"both": "both", "left_only": "ours_only", "right_only": "fplreview_only"})).round(4),
        out_dir / "player_rounds.csv", index=False, encoding="utf-8-sig")
    to_csv_atomic(summary, out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    to_csv_atomic(teams, out_dir / "teams.csv", index=False, encoding="utf-8-sig")
    to_csv_atomic(tops, out_dir / "top_differences.csv", index=False, encoding="utf-8-sig")
    relevant, segments = relevant_players(pr)
    spread, slopes = team_spread(pr, teams, ours)
    to_csv_atomic(relevant, out_dir / "relevant_players.csv", index=False, encoding="utf-8-sig")
    to_csv_atomic(segments, out_dir / "relevant_segments.csv", index=False, encoding="utf-8-sig")
    to_csv_atomic(spread, out_dir / "team_spread.csv", index=False, encoding="utf-8-sig")
    strengths, strength_summary = team_strengths(fixtures_path.parent / "team_ratings_current.csv",
                                                 status["_dir"] / "fplreview_team_strengths.csv")
    if strengths is not None:
        to_csv_atomic(strengths, out_dir / "team_strengths.csv", index=False, encoding="utf-8-sig")

    print(f"Comparison of {name} (rounds {status['rounds']}): {both[['Player', 'Team']].drop_duplicates().shape[0]} players in both")
    print(f"  ours only: {only_ours[['Player', 'Team']].drop_duplicates().shape[0]} players "
          f"({only_ours['TotalPts'].sum():.1f} pts); FPL Review only: {only_theirs[['Player', 'Team']].drop_duplicates().shape[0]}")
    print(f"  xMins clock (ours / FPL Review minutes): {both['xMins'].sum() / both['xMins_Raw'].sum():.3f}")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary[summary["Horizon"].eq("all")].drop(columns="Horizon").to_string(index=False))
        tm = teams.groupby("Team")[["Team_Model_NPG", "Ours_PenGoals", "Ours_Players_Goals", "FPLR_Players_Goals"]].sum() \
            if "Team_Model_NPG" in teams else teams.groupby("Team")[["Ours_Players_Goals", "FPLR_Players_Goals"]].sum()
        tm["ratio"] = tm["Ours_Players_Goals"] / tm["FPLR_Players_Goals"]
        print("\nTeam goals over the snapshot's rounds:")
        print(tm.sort_values("ratio").round(2).to_string())
        show = ["Player", "Team", "Pos", "Price", "Mean_xMins", "TotalPts", "FPLR_Points", "Gap",
                *[f"Gap_{c}" for c in PARTS]]
        print(f"\nFPL-relevant players: {len(relevant)} (mean xMins >= {RELEVANT_XMINS}, >= {RELEVANT_POINTS_PER_ROUND} points per round "
              f"in either model); mean gap {relevant['Gap'].mean():+.2f}, SD {relevant['Gap'].std():.2f}")
        print("We are lower:")
        print(relevant.head(12)[show].to_string(index=False))
        print("We are higher:")
        print(relevant.tail(12).iloc[::-1][show].to_string(index=False))
        print("\nBy segment (mean gap over the snapshot's rounds):")
        print(segments[segments["Segment"].ne("Team")].to_string(index=False))
    print("\nTeam spread, log slope ours on FPL Review (below 1 = ours narrower):")
    for name, value in slopes.items():
        print(f"  {name}: {value:.2f}")
    if strengths is not None:
        print("\nTeam strengths, goals per match v an average opponent at a neutral venue (penalties included):")
        with pd.option_context("display.width", 200):
            print(strengths.drop(columns="Team").to_string(index=False))
        for key, value in strength_summary.items():
            print(f"  {key}: {value:.3f}")
    else:
        print("\nNo FPL Review team strengths for this snapshot.")
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
