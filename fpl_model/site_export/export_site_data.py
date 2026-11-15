"""Part 3, stage 1: export the personal website's data (website_plan.md) to site/data/*.json.

    python scripts/site_export/export_site_data.py

Reads data/current outputs only; never fetches, fits or writes anything outside site/data/. Refuses stale
outputs with save_gameweek_outputs.snapshot_plan, the check a full snapshot uses: predicted_points.csv must be
newer than its inputs and start at the next gameweek. The site shows the latest week only.

Players are those in player_ratings.csv (website_plan.md "Players"). The window is forecast_horizon.HORIZON,
read from the data's own round columns, never set here.

projections.json (Projections page), per player, one entry per round of the window (null = no fixture):
    points   TotalPts                 summed over a double gameweek's fixtures
    minutes  xMins_Raw (FPL Review)   summed
    goals    Goals (penalties incl.)  summed
    assists  Assists (FPL-only incl.) summed
    cs       CS_Prob                  summed: the player's clean-sheet chance, P60 x the team's (user, 2026-11-10)
    defcon   Defcon_Prob              summed (at the forecast minutes)
    pens     penalty_share.csv's share: in a double gameweek the average of its games (user, 2026-11-10). The file
             holds one share per gameweek, so each game carries that week's share and their average equals it.

ratings.json (Player ratings page), one row per player, each value per 90 v an average opponent at a neutral venue:
    minutes     minutes.csv's HORIZON rounds (FPL Review xMins), averaged with weights 1, TIME_DECAY, TIME_DECAY^2, ...
                (forecast_horizon.week_weights): the more distant the gameweek, the lower its weight (user, 2026-11-11)
    share       mean of penalty_share.csv's HORIZON weekly shares
    npg         player_ratings.csv NPG;  assists: Assists;  defcon: Defcon (chance of the threshold in 90 minutes)
    team_att90  the team's penalty attempts per 90 = Attack_Rating x PEN_ATTEMPTS_PER_NPG. predict_fixture_goals.py's
                goals = Attack x (opponent Defense / mean_npg) x venue factor; an average opponent's Defense is
                mean_npg and a neutral venue's factor is 1, so the team's NPG v an average opponent is its Attack_Rating
    conversion  build_penalty_predictions.build_conversion_rates (0.79 without history, as there)
    cs          exp(-xGC) with xGC = Defense_Rating x (1 + PEN_ATTEMPTS_PER_NPG x conversion), the formula of
                build_simulation_features.py with an average opponent (whose NPG against this team is Defense_Rating)
The page recomputes goals /90 = npg + share x team_att90 x conversion, so share edits need nothing from the server.

teams.json (Team ratings and Fixtures pages):
    ratings   team_ratings_current.csv: overall = Rating (Attack - Defense), attack, defense (lower = better)
    table     predicted_table.csv as it is
    fixtures  per team, one list per round of the window (empty = blank, two entries = double), the schedule of
              fixture_goal_predictions.csv: opponent code, venue, and the opposition's strength as the fixture-ticker
              images rate it (user, 2026-11-10): gf and ga = the goals an average team would score and concede v
              that opponent at that venue, from fixture_ticker.build_difficulty (no penalties, as the images)
"""
from __future__ import annotations
from fpl_model import paths

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = paths.ROOT
DATA = paths.CURRENT
OUT = paths.SITE / "data"
from fpl_model.forecast_horizon import HORIZON, week_weights  # noqa: E402
from save_gameweek_outputs import CURRENT_SEASON, snapshot_plan  # noqa: E402
from fpl_model.minutes_penalties_cs.build_penalty_predictions import build_conversion_rates, resolve_names  # noqa: E402
from fpl_model.league_rates import penalty_rates  # noqa: E402

# Component column -> exported stat key. Summed over a gameweek's fixtures.
SUMMED_STATS = {
    "TotalPts": "points", "xMins_Raw": "minutes", "Goals": "goals", "Assists": "assists",
    "CS_Prob": "cs", "Defcon_Prob": "defcon",
}
DECIMALS = {"points": 3, "minutes": 1, "goals": 4, "assists": 4, "cs": 4, "defcon": 4, "pens": 4}


def read(name):
    return pd.read_csv(DATA / name, encoding="utf-8-sig")


def rated_players(ratings):
    """(Web_Name, Team_Short) of every player with a rating, in file order."""
    if ratings.duplicated(["Player", "Team"]).any():
        raise ValueError("player_ratings.csv has duplicate (Player, Team) rows")
    return ratings[["Player", "Pos", "Team", "Price"]]


def window_rounds(points):
    rounds = sorted(int(c) for c in points.columns if str(c).isdigit())
    if len(rounds) != HORIZON or rounds != list(range(rounds[0], rounds[0] + HORIZON)):
        raise ValueError(f"predicted_points.csv rounds {rounds} are not {HORIZON} consecutive rounds (HORIZON)")
    return rounds


def weekly_components(components, rounds):
    """One row per (Player, Team, Round): summed stats, fixture count and opponents."""
    unknown = sorted(set(components["Round"]) - set(rounds))
    if unknown:
        raise ValueError(f"predicted_components.csv has rounds outside the window: {unknown}")
    components = components.assign(_opp=components["Opponent"] + " (" + components["Venue"].str[0] + ")")
    grouped = components.groupby(["Player", "Team", "Round"], sort=False)
    weekly = grouped[list(SUMMED_STATS)].sum().rename(columns=SUMMED_STATS)
    weekly["fixtures"] = grouped.size()
    weekly["opponents"] = grouped["_opp"].agg(list)
    return weekly


def weekly_shares(shares, players_current, rounds):
    """(Web_Name, Team_Short) -> list of HORIZON shares from penalty_share.csv (GW1 = first round)."""
    columns = [f"GW{i}" for i in range(1, len(rounds) + 1)]
    missing = [c for c in columns if c not in shares]
    if missing:
        raise ValueError(f"penalty_share.csv lacks {missing}")
    shares = shares.copy()
    shares["Web_Name"] = resolve_names(shares, players_current, "Player")
    shares[columns] = shares[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    shares = shares.drop_duplicates(["Web_Name", "Team"])
    return {(r.Web_Name, r.Team): [float(getattr(r, c)) for c in columns] for r in shares.itertuples()}


def number(value, decimals):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), decimals)


def build_projections(data_files=None):
    """The projections.json payload. data_files maps file names to frames (tests); default reads DATA."""
    frames = data_files or {}
    get = lambda name: frames[name] if name in frames else read(name)  # noqa: E731
    points = get("predicted_points.csv")
    rounds = window_rounds(points)
    players = rated_players(get("player_ratings.csv"))
    players_current = get("players_current.csv")
    weekly = weekly_components(get("predicted_components.csv"), rounds)
    shares = weekly_shares(get("penalty_share.csv"), players_current, rounds)

    keys = set(zip(players["Player"], players["Team"]))
    unmatched = keys - set(zip(weekly.index.get_level_values(0), weekly.index.get_level_values(1)))
    if unmatched:
        raise ValueError(f"{len(unmatched)} rated players have no forecast rows, e.g. {sorted(unmatched)[:3]}")

    weekly_dict = weekly.to_dict("index")
    rows = []
    for p in players.itertuples(index=False):
        row = {"name": p.Player, "pos": p.Pos, "team": p.Team, "price": float(p.Price),
               "fixtures": [], "opponents": [], **{k: [] for k in DECIMALS}}
        share = shares.get((p.Player, p.Team), [0.0] * len(rounds))
        for i, rnd in enumerate(rounds):
            week = weekly_dict.get((p.Player, p.Team, rnd))
            row["fixtures"].append(int(week["fixtures"]) if week else 0)
            row["opponents"].append(week["opponents"] if week else [])
            for stat in SUMMED_STATS.values():
                row[stat].append(number(week[stat], DECIMALS[stat]) if week else None)
            row["pens"].append(number(share[i], DECIMALS["pens"]) if week else None)
        rows.append(row)
    return {
        "meta": {"season": CURRENT_SEASON, "rounds": rounds, "horizon": HORIZON,
                 "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "teams": sorted(players["Team"].unique().tolist()),
        "players": rows,
    }


def team_strengths(team_ratings, players_current, pen_attempts_per_npg, pen_conversion):
    """Team_Short -> (penalty attempts per 90, clean-sheet chance) v an average opponent at a neutral venue."""
    short = dict(zip(players_current["Team"], players_current["Team_Short"]))
    missing = sorted(set(team_ratings["Team"]) - set(short))
    if missing:
        raise ValueError(f"team_ratings_current.csv teams without a code in players_current.csv: {missing}")
    out = {}
    for r in team_ratings.itertuples():
        conceded = r.Defense_Rating * (1 + pen_attempts_per_npg * pen_conversion)
        out[short[r.Team]] = (r.Attack_Rating * pen_attempts_per_npg, math.exp(-conceded))
    return out


def mean_minutes(minutes, players_current, rounds):
    """(Web_Name, Team_Short) -> FPL Review xMins over the window's rounds, a weighted mean with week_weights (a
    blank week counts 0)."""
    columns = [f"{r}_xMins" for r in rounds]
    missing = [c for c in columns if c not in minutes]
    if missing:
        raise ValueError(f"minutes.csv lacks {missing}")
    minutes = minutes.copy()
    minutes["Web_Name"] = resolve_names(minutes, players_current, "Name")
    weights = pd.Series(week_weights(len(columns)), index=columns)
    values = minutes[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    minutes["_mean"] = values.mul(weights, axis=1).sum(axis=1) / weights.sum()
    return dict(zip(zip(minutes["Web_Name"], minutes["Team"]), minutes["_mean"]))


def build_ratings(data_files=None, rates=None, conversion_rates=None):
    """The ratings.json payload. Tests pass frames, (attempts per NPG, conversion) and conversion rates."""
    frames = data_files or {}
    get = lambda name: frames[name] if name in frames else read(name)  # noqa: E731
    rounds = window_rounds(get("predicted_points.csv"))
    ratings = get("player_ratings.csv")
    players = rated_players(ratings)
    players_current = get("players_current.csv")
    pen_attempts_per_npg, pen_conversion = rates or penalty_rates()
    if conversion_rates is None:
        conversion_rates = build_conversion_rates(
            read("penalty_log.csv"), pd.read_csv(DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False))
    teams = team_strengths(get("team_ratings_current.csv"), players_current, pen_attempts_per_npg, pen_conversion)
    shares = weekly_shares(get("penalty_share.csv"), players_current, rounds)
    minutes = mean_minutes(get("minutes.csv"), players_current, rounds)
    full_name = dict(zip(zip(players_current["Web_Name"], players_current["Team_Short"]), players_current["Player"]))
    unknown = sorted(set(players["Team"]) - set(teams))
    if unknown:
        raise ValueError(f"rated players' teams missing from team_ratings_current.csv: {unknown}")

    rows = []
    for p in ratings.itertuples(index=False):
        key = (p.Player, p.Team)
        share = shares.get(key, [0.0] * len(rounds))
        team_att90, cs = teams[p.Team]
        rows.append({
            "name": p.Player, "pos": p.Pos, "team": p.Team, "price": float(p.Price),
            "minutes": number(minutes.get(key, 0.0), 1),
            "share": number(sum(share) / len(share), 4),
            "npg": number(p.NPG, 4), "assists": number(p.Assists, 4), "defcon": number(p.Defcon, 4),
            "team_att90": number(team_att90, 5),
            "conversion": number(conversion_rates.get(full_name.get(key), 0.79), 4),
            "cs": number(cs, 4),
        })
    return {"meta": {"season": CURRENT_SEASON, "rounds": rounds, "horizon": HORIZON},
            "teams": sorted(players["Team"].unique().tolist()), "players": rows}


def opposition_strength(codes):
    """(Opponent code, Venue) -> (goals for, goals against) of an average team: the fixture ticker's own table."""
    from fpl_model.optimizer.fixture_ticker import build_difficulty  # heavy imports (sklearn, matplotlib): only when exporting teams
    table = build_difficulty(sorted(codes))
    return {(r.Opponent, r.Venue): (r.Goals_For, r.Goals_Against) for r in table.itertuples()}


def build_teams(data_files=None, strength=None):
    """The teams.json payload. Tests pass frames and an opposition_strength() lookup."""
    frames = data_files or {}
    get = lambda name: frames[name] if name in frames else read(name)  # noqa: E731
    rounds = window_rounds(get("predicted_points.csv"))
    players_current = get("players_current.csv")
    short = dict(zip(players_current["Team"], players_current["Team_Short"]))

    team_ratings = get("team_ratings_current.csv")
    fixtures = get("fixture_goal_predictions.csv")
    table = get("predicted_table.csv")
    names = set(team_ratings["Team"]) | set(fixtures["Home"]) | set(fixtures["Away"]) | set(table["Team"])
    missing = sorted(names - set(short))
    if missing:
        raise ValueError(f"teams without a code in players_current.csv: {missing}")
    outside = sorted(set(fixtures["Round"]) - set(rounds))
    if outside:
        raise ValueError(f"fixture_goal_predictions.csv has rounds outside the window: {outside}")

    strength = strength or opposition_strength({short[t] for t in team_ratings["Team"]})
    weeks = {short[t]: [[] for _ in rounds] for t in team_ratings["Team"]}
    for f in fixtures.sort_values(["Round", "MatchID"]).itertuples():
        i = rounds.index(int(f.Round))
        for team, opp, venue in ((f.Home, f.Away, "H"), (f.Away, f.Home, "A")):
            gf, ga = strength[(short[opp], venue)]
            weeks[short[team]][i].append({"opp": short[opp], "venue": venue, "gf": number(gf, 4), "ga": number(ga, 4)})
    ratings = [{"team": short[r.Team], "name": r.Team, "overall": number(r.Rating, 4),
                "attack": number(r.Attack_Rating, 4), "defense": number(r.Defense_Rating, 4)}
               for r in team_ratings.itertuples()]
    table_rows = [{"team": short[r["Team"]], "name": r["Team"], **{k: (number(v, 3) if k != "Team" else v)
                   for k, v in r.items() if k != "Team"}} for r in table.to_dict("records")]
    return {"meta": {"season": CURRENT_SEASON, "rounds": rounds, "horizon": HORIZON},
            "ratings": ratings, "table": table_rows,
            "fixtures": [{"team": t, "weeks": w} for t, w in weeks.items()]}


def write_json(payload, name):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{name}.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(OUT / name)


def main():
    plan = snapshot_plan(DATA, CURRENT_SEASON, part="all")  # raises on stale outputs
    projections = build_projections()
    if projections["meta"]["rounds"][0] != plan["origin"]:
        raise ValueError(f"Forecast starts at GW{projections['meta']['rounds'][0]}, next gameweek is GW{plan['origin']}")
    write_json(projections, "projections.json")
    ratings = build_ratings()
    ratings["meta"]["generated_utc"] = projections["meta"]["generated_utc"]
    write_json(ratings, "ratings.json")
    teams = build_teams()
    write_json(teams, "teams.json")
    meta = {**projections["meta"], "next_gameweek": plan["origin"]}
    write_json(meta, "meta.json")
    rounds = projections["meta"]["rounds"]
    print(f"Wrote projections.json: {len(projections['players'])} players, GW{rounds[0]}-GW{rounds[-1]}")
    print(f"Wrote ratings.json: {len(ratings['players'])} players")
    print(f"Wrote teams.json: {len(teams['ratings'])} teams, {sum(len(f) for t in teams['fixtures'] for f in t['weeks'])} team-fixtures")


if __name__ == "__main__":
    main()
