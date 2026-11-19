"""
Cleans the weekly FPL Review projection pastes into one table, for comparing against our forecasts.

Benchmark data only: no model stage reads the output. FPL Review is a reference for finding where
our model may be weak, never a target to calibrate to (the "markets inform, never override" rule
applies in spirit).

Input: one Excel paste per stat from FPL Review's projections page, saved by the user to
    data/raw/fplreview/<season>/GW<nn>/{points,goals,assists,cs,defcon}.xlsx
just before part 2, at the same time as minutes.csv and review_pk_share.xlsx. The user overwrites
them on each refresh; the part-2 snapshot copies them, so each snapshot keeps the pastes it was run
against.

FPL Review's definitions (stated by the user, 2026-11-04): goals include penalties; assists include
FPL-only assists; a clean sheet counts only if the player plays 60+ minutes; defcon is the
probability of reaching the defensive-contribution threshold.

Layout, the same as the penalty-share paste (clean_penalty_share_paste.py): a settings panel, a
header row ("Player", "Price", "GW11".."GW20", "Total", "Elite Own%"), then three rows per player:
    stats row  -- team code, price ("£15.8mSV: 15.6▲"), HORIZON gameweek values (10 since GW11, 5 before),
                  total, elite own%
    name row   -- "{Raw_Web_Name}{TEAM} • {POS}"
    xMins row  -- FPL Review's rounded xMins for the first gameweek, or "-" (checked against minutes.csv)

Output (data/current/sourced/):
    fplreview_projections.csv  one row per player and round: Points, Goals, Assists, CleanSheet,
                               Defcon, plus FPLR_xMins_First on the first round's rows
    fplreview_status.json      usable_for_comparison and the reasons; the comparison skips a
                               snapshot whose status is not usable
    fplreview_team_strengths.csv  optional, from "Team Strengths.xlsx" in the same folder: each team's
                               GF, GA and GD per match from FPL Review's team model, whose Source
                               Blend (GW10: Markets 0, elevenify 100%) is kept in the status file

Never halts the pipeline: any failure is written to the status file and the script exits 0, and a
failed run deletes the previous run's projections so a stale table can't be snapshotted.

Run locally:
    python -m fpl_model.fplreview.clean_fplreview_paste [--gameweek N]
"""
import argparse
import json
import re
import sys
import traceback
from datetime import UTC, datetime

import pandas as pd

from fpl_model import paths

ROOT = paths.ROOT
from fpl_model.atomic_write import to_csv_atomic  # noqa: E402
from fpl_model.forecast_horizon import HORIZON  # noqa: E402  (the pastes must cover the pipeline's window)
from fpl_model.league_rates import current_season  # noqa: E402

DATA = paths.CURRENT
RAW = paths.RAW / "fplreview"
OUT_FILE = paths.current_file("fplreview_projections.csv", root=paths.CURRENT)
STATUS_FILE = paths.current_file("fplreview_status.json", root=paths.CURRENT)
TEAM_FILE = paths.current_file("fplreview_team_strengths.csv", root=paths.CURRENT)

# file stem -> output column; the stat name FPL Review prints above the header row
STATS = {
    "points": ("Points", "Points"),
    "goals": ("Goals", "Goals"),
    "assists": ("Assists", "Assists"),
    "cs": ("CleanSheet", "CleanSheets"),
    "defcon": ("Defcon", "CBIT / CBITR"),
}
POS_MAP = {"GKP": "G", "DEF": "D", "MID": "M", "FWD": "F"}
PRICE = re.compile(r"£\s*(\d+(?:\.\d+)?)m")
XMINS_ROW = re.compile(r"^(\d+(\.\d+)?|-)$")  # "-" for some fringe or doubtful players
# Paste and minutes.csv must agree to within rounding for nearly every player, or one of them is stale.
XMINS_TOLERANCE = 1.0
XMINS_MIN_AGREEMENT = 0.95


def parse_paste(path, expected_stat):
    """Rows of (FPLR_Name, Team_Short, FPLR_Pos, Price, round -> value, FPLR_xMins_First) from one paste."""
    df = pd.read_excel(path, header=None, dtype=object)
    header = df.index[df[0].astype(str).str.strip().eq("Player") & df[1].astype(str).str.strip().eq("Price")]
    if len(header) != 1:
        raise ValueError(f"{path.name}: expected one header row ('Player', 'Price'), found {len(header)}")
    h = header[0]
    stat_label = next((str(v).strip() for v in df.iloc[:h, 0][::-1] if str(v).strip() not in ("Own%", "Secondary Stat", "nan")), "")
    if stat_label != expected_stat:
        raise ValueError(f"{path.name}: the paste shows stat '{stat_label}', expected '{expected_stat}' "
                         f"(wrong stat selected, or file saved under the wrong name)")
    labels = [str(v).strip() for v in df.iloc[h, 2:].tolist()]
    rounds = [int(lab[2:]) for lab in labels if re.fullmatch(r"GW\d+", lab)]
    if len(rounds) != HORIZON or labels[HORIZON] != "Total":
        raise ValueError(f"{path.name}: expected {HORIZON} gameweeks then 'Total' (forecast_horizon.HORIZON), "
                         f"found header {labels}")
    body = df.iloc[h + 1:].reset_index(drop=True)
    if len(body) % 3:
        raise ValueError(f"{path.name}: {len(body)} rows after the header, not a multiple of three")
    rows = []
    for i in range(0, len(body), 3):
        stats, name, xmins = body.iloc[i], body.iloc[i + 1], body.iloc[i + 2]
        team = str(stats[0]).strip()
        price = PRICE.search(str(stats[1]))
        name_pos = str(name[0])
        xm = str(xmins[0]).strip()
        if not price or " • " not in name_pos or not XMINS_ROW.match(xm) or not name.iloc[1:].isna().all():
            raise ValueError(f"{path.name}: player block at row {h + 2 + i} doesn't match the three-row layout: "
                             f"{[str(stats[0]), str(stats[1]), name_pos, xm]}")
        name_team, pos = name_pos.rsplit(" • ", 1)
        if not name_team.endswith(team) or pos not in POS_MAP:
            raise ValueError(f"{path.name}: can't split name, team and position from '{name_pos}' (team {team})")
        values = pd.to_numeric(pd.Series(stats[2:2 + HORIZON].tolist()), errors="raise").tolist()
        total = float(stats[2 + HORIZON])
        # Points show one decimal, other stats two: HORIZON + 1 half-units of rounding at most. Catches shifted columns.
        if abs(sum(values) - total) > (HORIZON + 1) * 0.05 + 0.01:
            raise ValueError(f"{path.name}: {name_pos} gameweeks sum to {sum(values):.2f}, total says {total}")
        rows.append({"FPLR_Name": name_team[:-len(team)], "Team_Short": team, "FPLR_Pos": POS_MAP[pos],
                     "Price": float(price.group(1)), "values": dict(zip(rounds, values)),
                     "FPLR_xMins_First": float("nan") if xm == "-" else float(xm)})
    return rounds, rows


def link_players(frame, players):
    """FPL Review shows FPL's raw web name; Raw_Web_Name is not unique, so join on (raw name, team) (CLAUDE.md §5.1)."""
    lookup = players.drop_duplicates(["Raw_Web_Name", "Team_Short"], keep=False)
    linked = frame.merge(lookup[["Raw_Web_Name", "Team_Short", "Web_Name", "Pos"]],
                         left_on=["FPLR_Name", "Team_Short"], right_on=["Raw_Web_Name", "Team_Short"],
                         how="left", validate="many_to_one").drop(columns="Raw_Web_Name")
    return linked


def minutes_check(frame, first_round):
    """Compare the paste's xMins row with minutes.csv's first round: a mismatch means one of them is stale."""
    minutes = pd.read_csv(paths.current_file("minutes.csv", root=DATA), encoding="utf-8-sig")
    col = f"{first_round}_xMins"
    if col not in minutes:
        return {"ok": False, "reason": f"minutes.csv has no {col}; it covers other rounds than the paste"}
    m = minutes[["Name", "Team", col]].rename(columns={"Name": "FPLR_Name", "Team": "Team_Short", col: "_mins"})
    m = m.drop_duplicates(["FPLR_Name", "Team_Short"], keep=False)
    joined = frame.drop_duplicates(["FPLR_Name", "Team_Short"]).merge(m, on=["FPLR_Name", "Team_Short"], how="inner")
    played = joined[(joined["_mins"] > 0) | (joined["FPLR_xMins_First"] > 0)]
    agree = float(((played["_mins"] - played["FPLR_xMins_First"]).abs() <= XMINS_TOLERANCE).mean()) if len(played) else 0.0
    return {"ok": agree >= XMINS_MIN_AGREEMENT, "players_compared": int(len(played)), "agreement": round(agree, 4),
            "reason": None if agree >= XMINS_MIN_AGREEMENT else
            f"only {agree:.1%} of players' xMins match minutes.csv {col} within {XMINS_TOLERANCE:g}; "
            f"refresh the pastes and minutes.csv together"}


def build(gameweek, season):
    folder = RAW / str(season) / f"GW{gameweek:02d}"
    if not folder.is_dir():
        folder = RAW / str(season) / f"GW{gameweek}"
    missing = [f"{stem}.xlsx" for stem in STATS if not (folder / f"{stem}.xlsx").is_file()]
    if missing:
        raise FileNotFoundError(f"{folder} lacks {', '.join(missing)}")
    long, rounds_seen, sizes = [], {}, {}
    for stem, (column, label) in STATS.items():
        rounds, rows = parse_paste(folder / f"{stem}.xlsx", label)
        rounds_seen[stem], sizes[stem] = rounds, len(rows)
        for r in rows:
            for rnd, value in r["values"].items():
                long.append({"FPLR_Name": r["FPLR_Name"], "Team_Short": r["Team_Short"], "FPLR_Pos": r["FPLR_Pos"],
                             "Price": r["Price"], "Round": rnd, "Stat": column, "Value": value,
                             "FPLR_xMins_First": r["FPLR_xMins_First"]})
    if len({tuple(r) for r in rounds_seen.values()}) != 1:
        raise ValueError(f"pastes cover different rounds: {rounds_seen}")
    rounds = rounds_seen["points"]
    if rounds[0] != gameweek:
        raise ValueError(f"pastes start at GW{rounds[0]}, expected GW{gameweek}")
    long = pd.DataFrame(long)
    keys = ["FPLR_Name", "Team_Short", "FPLR_Pos", "Round"]
    if long.duplicated(keys + ["Stat"]).any():
        dup = long[long.duplicated(keys + ["Stat"], keep=False)].drop_duplicates(["FPLR_Name", "Team_Short"])
        raise ValueError(f"duplicate players in a paste: {dup[['FPLR_Name', 'Team_Short']].values.tolist()}")
    xmins = long.groupby(["FPLR_Name", "Team_Short"])["FPLR_xMins_First"].nunique(dropna=False)
    if (xmins > 1).any():
        raise ValueError(f"the pastes disagree on xMins for {list(xmins[xmins > 1].index)[:5]}; they were taken at different times")
    wide = long.pivot_table(index=keys + ["Price"], columns="Stat", values="Value", aggfunc="first").reset_index()
    wide.columns.name = None
    missing_stats = wide[[c for c, _ in STATS.values()]].isna().any(axis=1)
    first = long.drop_duplicates(["FPLR_Name", "Team_Short"])[["FPLR_Name", "Team_Short", "FPLR_xMins_First"]]
    wide = wide.merge(first, on=["FPLR_Name", "Team_Short"], how="left")
    wide.loc[wide["Round"] != rounds[0], "FPLR_xMins_First"] = pd.NA

    players = pd.read_csv(paths.current_file("players_current.csv", root=DATA), encoding="utf-8-sig")
    wide = link_players(wide, players)
    wide.insert(0, "Season", season)
    unmatched = wide[wide["Web_Name"].isna()].drop_duplicates(["FPLR_Name", "Team_Short"])
    pos_diff = wide[wide["Web_Name"].notna() & wide["Pos"].ne(wide["FPLR_Pos"])].drop_duplicates(["FPLR_Name", "Team_Short"])
    check = minutes_check(wide, rounds[0])
    reasons = [check["reason"]] if check["reason"] else []
    if missing_stats.any():
        reasons.append(f"{int(missing_stats.sum())} player-rounds miss a stat; the pastes list different players")
    status = {
        "usable_for_comparison": not reasons, "reasons": reasons, "gameweek": gameweek, "season": season,
        "rounds": rounds, "players_per_paste": sizes, "unmatched_players": unmatched[["FPLR_Name", "Team_Short"]].values.tolist(),
        "position_differences": pos_diff[["FPLR_Name", "Team_Short", "FPLR_Pos", "Pos"]].values.tolist(),
        "minutes_check": check, "source_folder": str(folder.relative_to(ROOT)).replace("\\", "/"),
        "source_modified_utc": {f"{s}.xlsx": datetime.fromtimestamp((folder / f"{s}.xlsx").stat().st_mtime, UTC).isoformat()
                                for s in STATS},
        "built_utc": datetime.now(UTC).isoformat(),
    }
    cols = ["Season", "Round", "Web_Name", "Team_Short", "Pos", "FPLR_Name", "FPLR_Pos", "Price",
            "Points", "Goals", "Assists", "CleanSheet", "Defcon", "FPLR_xMins_First"]
    return wide[cols].sort_values(["Round", "Points"], ascending=[True, False]), status


def parse_team_strengths(path):
    """FPL Review's team table: five rows per team (code, name, GF, GD, GA), goals per match.

    The pasted header reads "AttackGF, GADefence, GD", but the values come as GF, GD, GA
    (Arsenal 1.84, 1.05, 0.79). GF - GA = GD is checked for every team, and since that identity can't
    tell GD from GA, the GD column must also average about 0 and every GA be positive. The block ends at the
    "Data provided for free by" line. Returns the table and the Source Blend text above it.
    """
    col = pd.read_excel(path, header=None, dtype=object)[0].tolist()
    cells = [str(v).strip() for v in col if not pd.isna(v)]
    if "Team Strengths" not in cells:
        raise ValueError(f"{path.name}: no 'Team Strengths' heading")
    i = cells.index("Team Strengths")
    blend = " ".join(cells[cells.index("Source Blend", i) + 1:cells.index("Team", i)]) if "Source Blend" in cells[i:] else ""
    try:
        start = cells.index("GD", i) + 1
        end = next(j for j in range(start, len(cells)) if cells[j].startswith("Data provided"))
    except (ValueError, StopIteration) as err:
        raise ValueError(f"{path.name}: can't find the team table between 'GD' and 'Data provided for free by'") from err
    block = cells[start:end]
    if len(block) % 5:
        raise ValueError(f"{path.name}: team table has {len(block)} cells, not five per team")
    rows = []
    for k in range(0, len(block), 5):
        code, name, gf, gd, ga = block[k:k + 5]
        gf, gd, ga = float(gf), float(gd), float(ga)
        if not re.fullmatch(r"[A-Z]{3}", code) or abs(gf - ga - gd) > 0.015:
            raise ValueError(f"{path.name}: team row {block[k:k + 5]} isn't code, name, GF, GD, GA")
        rows.append({"Team_Short": code, "FPLR_Team_Name": name, "FPLR_GF": gf, "FPLR_GD": gd, "FPLR_GA": ga})
    table = pd.DataFrame(rows)
    if len(table) != 20 or table["Team_Short"].duplicated().any():
        raise ValueError(f"{path.name}: expected 20 distinct teams, found {table['Team_Short'].nunique()}")
    # GF = GD + GA can't tell GD from GA; goal differences average about 0 and goals against are all positive.
    if abs(table["FPLR_GD"].mean()) > 0.15 or (table["FPLR_GA"] <= 0).any():
        raise ValueError(f"{path.name}: the GD column averages {table['FPLR_GD'].mean():.2f} and GA runs "
                         f"{table['FPLR_GA'].min():.2f}-{table['FPLR_GA'].max():.2f}; GD and GA look swapped")
    return table, blend


def build_team_strengths(folder, season):
    """Optional: Team Strengths.xlsx beside the player pastes. Returns (table or None, status entry)."""
    paths = sorted(folder.glob("*[Ss]trength*.xlsx"))
    if not paths:
        return None, {"ok": False, "reason": "no Team Strengths.xlsx in the paste folder"}
    try:
        table, blend = parse_team_strengths(paths[0])
        players = pd.read_csv(paths.current_file("players_current.csv", root=DATA), encoding="utf-8-sig")
        team_names = players.drop_duplicates("Team_Short").set_index("Team_Short")["Team"]
        table.insert(0, "Team", table["Team_Short"].map(team_names))
        missing = table.loc[table["Team"].isna(), "Team_Short"].tolist()
        if missing:
            raise ValueError(f"team codes not in players_current.csv: {missing}")
        table.insert(0, "Season", season)
        return table, {"ok": True, "reason": None, "source_blend": blend, "file": paths[0].name,
                       "modified_utc": datetime.fromtimestamp(paths[0].stat().st_mtime, UTC).isoformat()}
    except Exception as error:
        return None, {"ok": False, "reason": f"{type(error).__name__}: {error}"}


def default_gameweek():
    """The first forecast round in minutes.csv, which part 2 is about to use."""
    minutes = pd.read_csv(paths.current_file("minutes.csv", root=DATA), encoding="utf-8-sig", nrows=1)
    return min(int(c.split("_")[0]) for c in minutes.columns if re.fullmatch(r"\d+_xMins", c))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gameweek", type=int)
    parser.add_argument("--season", type=int, default=current_season())
    args = parser.parse_args()
    try:
        gameweek = args.gameweek or default_gameweek()
        table, status = build(gameweek, args.season)
        to_csv_atomic(table, OUT_FILE, index=False, encoding="utf-8-sig")
        print(f"Wrote {table[['FPLR_Name', 'Team_Short']].drop_duplicates().shape[0]} players x {table.Round.nunique()} rounds to {OUT_FILE.name}")
    except Exception as error:  # benchmark only: never stop part 2
        OUT_FILE.unlink(missing_ok=True)
        status = {"usable_for_comparison": False, "reasons": [f"{type(error).__name__}: {error}"],
                  "built_utc": datetime.now(UTC).isoformat()}
        traceback.print_exc()
    # Team strengths are independent of the player pastes: a problem with one never spoils the other.
    TEAM_FILE.unlink(missing_ok=True)
    try:
        gameweek = args.gameweek or default_gameweek()
        folder = RAW / str(args.season) / f"GW{gameweek:02d}"
        teams, team_status = build_team_strengths(folder if folder.is_dir() else RAW / str(args.season) / f"GW{gameweek}", args.season)
    except Exception as error:
        teams, team_status = None, {"ok": False, "reason": f"{type(error).__name__}: {error}"}
    if teams is not None:
        to_csv_atomic(teams, TEAM_FILE, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(teams)} team strengths to {TEAM_FILE.name} (source blend: {team_status['source_blend']})")
    else:
        print(f"WARNING: team strengths not written: {team_status['reason']}")
    status["team_strengths"] = team_status
    STATUS_FILE.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"usable_for_comparison: {status['usable_for_comparison']}")
    for reason in status["reasons"]:
        print(f"WARNING: {reason}")
    for name, team in status.get("unmatched_players", []):
        print(f"WARNING: no players_current.csv match for {name} ({team})")
    if status.get("position_differences"):
        print(f"Note: {len(status['position_differences'])} players have a different position in FPL Review")
    return 0


if __name__ == "__main__":
    sys.exit(main())
