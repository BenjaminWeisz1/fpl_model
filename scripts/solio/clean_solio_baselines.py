"""
Cleans the weekly Solio "Baselines" table into one linked table, for comparing against our player ratings.

Benchmark data only: no model stage reads the output. Solio (fpl.solioanalytics.com, free tier) replaced
FPL Review as the weekly benchmark on 2026-11-13, by user decision, because its Baselines view matches the
format of player_ratings.csv. A reference for finding weaknesses, never a target ("markets inform, never
override" applies in spirit). minutes.csv and the penalty share still come from FPL Review.

Input, captured in Chrome each week before part 2 (routine in scripts/solio/SOLIO.md):
    data/raw/solio/<season>/GW<nn>/solio_baselines.csv   one row per player, Solio's full-precision values
    data/raw/solio/<season>/GW<nn>/capture.json          when it was captured and the first gameweek shown
Raw files are never rewritten; a recapture overwrites them, and the part-2 snapshot keeps the copy it ran on.

Solio's columns (checked on the GW11 capture, 2026-11-13):
    pts90           expected FPL points per 90
    mins_per_match  expected minutes per match
    g90, a90        goals per 90 including penalties; FPL assists per 90 (FPL-only assists included)
    g_pct, a_pct    share of the team's goals / assists (each sums to 1 per team)
    cs_pct          the team's clean-sheet probability (one value per team)
    bon90, dc90     bonus points per 90; defensive contributions per 90 (a count, not the threshold chance)
Solio sets g90, a90, bon90 and dc90 to 0 (and leaves them blank on screen) for players under a minutes
cut-off (about 18 expected minutes at GW11). Those four become blank here; the shares and CS stay. The
cut-off is read from the data each run, never set here.

Output (data/current/):
    solio_baselines.csv     one row per Solio player, linked to Web_Name / Team_Short / Pos through the
                            FPL code (fpl_player_codes.csv Opta_Code -> players_current.csv)
    solio_status.json       usable_for_comparison and the reasons; a comparison skips an unusable week

Never halts the pipeline: any failure is written to the status file and the script exits 0, and a
failed run deletes the previous run's table so a stale one can't be snapshotted.

Run locally:
    python clean_solio_baselines.py [--gameweek N]
"""
import argparse
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from atomic_write import to_csv_atomic  # noqa: E402
from league_rates import current_season  # noqa: E402

DATA = ROOT / "data" / "current"
RAW = ROOT / "data" / "raw" / "solio"
SOURCE_NAME = "solio_baselines.csv"
CAPTURE_NAME = "capture.json"
OUT_FILE = DATA / "solio_baselines.csv"
STATUS_FILE = DATA / "solio_status.json"

RAW_COLUMNS = ["fpl_id", "fpl_code", "name", "position", "solio_team_id", "now_cost", "status",
               "pts90", "mins_per_match", "g90", "a90", "g_pct", "a_pct", "cs_pct", "bon90", "dc90"]
# raw column -> output column
RENAME = {"pts90": "Solio_Pts90", "mins_per_match": "Solio_Mins", "g90": "Solio_G90", "a90": "Solio_A90",
          "g_pct": "Solio_Goal_Share", "a_pct": "Solio_Assist_Share", "cs_pct": "Solio_Team_CS",
          "bon90": "Solio_Bon90", "dc90": "Solio_DC90"}
PER90_SUPPRESSED = ["g90", "a90", "bon90", "dc90"]
POS_MAP = {"GKP": "G", "DEF": "D", "MID": "M", "FWD": "F"}
SHARE_TOLERANCE = 0.01        # goal and assist shares must sum to 1 per team
MIN_LINKED = 0.98             # nearly every Solio player must link to players_current.csv
MIN_POSITION_AGREEMENT = 0.98


def folder_for(season, gameweek):
    folder = RAW / str(season) / f"GW{gameweek:02d}"
    return folder if folder.is_dir() else RAW / str(season) / f"GW{gameweek}"


def read_raw(path):
    raw = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in RAW_COLUMNS if c not in raw]
    if missing:
        raise ValueError(f"{path.name} lacks columns {missing}; recapture with the routine in SOLIO.md")
    if raw["fpl_code"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate FPL codes {raw.loc[raw.fpl_code.duplicated(), 'name'].tolist()[:5]}")
    unknown = sorted(set(raw["position"]) - set(POS_MAP))
    if unknown:
        raise ValueError(f"{path.name}: unknown positions {unknown}")
    values = raw[list(RENAME)]
    if values.isna().any().any():
        raise ValueError(f"{path.name}: blank values in {values.columns[values.isna().any()].tolist()}")
    return raw


def suppression(raw):
    """Rows where Solio zeroed all four per-90 columns, and the minutes cut-off that implies."""
    suppressed = raw[PER90_SUPPRESSED].eq(0).all(axis=1)
    shown = raw.loc[~suppressed, "mins_per_match"]
    hidden = raw.loc[suppressed, "mins_per_match"]
    overlap = bool(len(hidden) and len(shown) and hidden.max() >= shown.min())
    return suppressed, {"rows": int(suppressed.sum()),
                        "max_mins_suppressed": None if hidden.empty else round(float(hidden.max()), 3),
                        "min_mins_shown": None if shown.empty else round(float(shown.min()), 3),
                        "clean_cut": not overlap}


def share_check(raw):
    """Goal and assist shares sum to 1 per team, and CS is one value per team: a partial capture breaks both."""
    sums = raw.groupby("solio_team_id")[["g_pct", "a_pct"]].sum()
    bad_share = sums[(sums - 1).abs().gt(SHARE_TOLERANCE).any(axis=1)]
    cs_values = raw.groupby("solio_team_id")["cs_pct"].nunique()
    return {"teams": int(len(sums)), "share_sum_off": bad_share.round(3).reset_index().values.tolist(),
            "teams_with_several_cs": cs_values[cs_values > 1].index.tolist()}


def link_players(raw):
    """FPL code -> full name (fpl_player_codes.csv) -> Web_Name, Team_Short, Pos (players_current.csv)."""
    codes = pd.read_csv(DATA / "fpl_player_codes.csv", encoding="utf-8-sig")
    players = pd.read_csv(DATA / "players_current.csv", encoding="utf-8-sig")
    codes = codes.drop_duplicates("Opta_Code", keep=False)[["Opta_Code", "Player"]]
    players = players.drop_duplicates("Player", keep=False)[["Player", "Web_Name", "Team_Short", "Pos", "Price"]]
    linked = (raw.merge(codes, left_on="fpl_code", right_on="Opta_Code", how="left", validate="one_to_one")
                 .merge(players, on="Player", how="left", validate="many_to_one"))
    return linked.drop(columns=["Opta_Code"])


def team_check(linked):
    """Solio's team ID must map to one of our teams for every linked player; a mismatch is a mid-week transfer
    or a stale players_current.csv."""
    found = linked.dropna(subset=["Team_Short"])
    mapping = found.groupby("solio_team_id")["Team_Short"].agg(lambda s: s.mode().iloc[0])
    off = found[found["Team_Short"].ne(found["solio_team_id"].map(mapping))]
    return mapping, off[["name", "solio_team_id", "Team_Short"]].values.tolist()


def capture_info(folder, source):
    path = folder / CAPTURE_NAME
    if path.is_file():
        info = json.loads(path.read_text(encoding="utf-8"))
        return info, None
    stamp = datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat()
    return {"captured_utc": stamp}, f"no {CAPTURE_NAME}; capture time taken from the file's modified time"


def deadline_for(gameweek):
    path = DATA / "gameweek_deadlines.csv"
    if not path.is_file():
        return None
    d = pd.read_csv(path, encoding="utf-8-sig")
    row = d[d["Gameweek"].eq(gameweek)]
    return None if row.empty else pd.Timestamp(row["Deadline_UTC"].iloc[0])


def build(gameweek, season):
    folder = folder_for(season, gameweek)
    source = folder / SOURCE_NAME
    if not source.is_file():
        raise FileNotFoundError(f"no {SOURCE_NAME} in {folder}; capture Solio's Baselines first (scripts/solio/SOLIO.md)")
    raw = read_raw(source)
    capture, capture_note = capture_info(folder, source)
    reasons, notes = [], [capture_note] if capture_note else []

    shown_gw = capture.get("first_gameweek_shown")
    if shown_gw is not None and int(shown_gw) != gameweek:
        reasons.append(f"Solio showed GW{shown_gw} first, expected GW{gameweek}; the capture is from another week")
    deadline = deadline_for(gameweek)
    captured = pd.Timestamp(capture["captured_utc"])
    if deadline is not None and captured >= deadline:
        reasons.append(f"captured {captured.isoformat()}, after the GW{gameweek} deadline {deadline.isoformat()}")

    suppressed, cut = suppression(raw)
    if not cut["clean_cut"]:
        notes.append("zeroed per-90 rows and shown rows overlap in minutes; the cut-off isn't by minutes alone")
    shares = share_check(raw)
    if shares["share_sum_off"] or shares["teams_with_several_cs"] or shares["teams"] != 20:
        reasons.append(f"team checks failed ({shares['teams']} teams; shares off {shares['share_sum_off']}; "
                       f"several CS values {shares['teams_with_several_cs']}): a partial or filtered capture")

    linked = link_players(raw)
    unmatched = linked[linked["Web_Name"].isna()]
    linked_share = 1 - len(unmatched) / len(linked)
    if linked_share < MIN_LINKED:
        reasons.append(f"only {linked_share:.1%} of Solio players link to players_current.csv; run part 1 first")
    mapping, team_off = team_check(linked)
    if mapping.duplicated().any():
        reasons.append(f"two Solio teams map to one of ours: {mapping[mapping.duplicated(keep=False)].to_dict()}")
    solio_pos = linked["position"].map(POS_MAP)
    both = linked["Pos"].notna()
    pos_agree = float(solio_pos[both].eq(linked.loc[both, "Pos"]).mean()) if both.any() else 0.0
    if pos_agree < MIN_POSITION_AGREEMENT:
        reasons.append(f"positions agree for only {pos_agree:.1%} of linked players")
    price_agree = float(((linked.loc[both, "now_cost"] / 10) - linked.loc[both, "Price"]).abs().lt(0.05).mean()) if both.any() else 0.0

    out = linked.rename(columns=RENAME)
    for col in PER90_SUPPRESSED:
        out.loc[suppressed.values, RENAME[col]] = pd.NA
    out["Solio_Per90_Suppressed"] = suppressed.values
    out["Solio_Pos"] = solio_pos
    out["Solio_Team_Short"] = out["solio_team_id"].map(mapping)
    out["Solio_Price"] = out["now_cost"] / 10
    out.insert(0, "Gameweek", gameweek)
    out.insert(0, "Season", season)
    cols = ["Season", "Gameweek", "Web_Name", "Team_Short", "Pos", "fpl_id", "fpl_code", "name", "Solio_Pos",
            "Solio_Team_Short", "Solio_Price", "status", *RENAME.values(), "Solio_Per90_Suppressed"]
    out = out[cols].rename(columns={"fpl_id": "FPL_ID", "fpl_code": "FPL_Code", "name": "Solio_Name", "status": "Solio_Status"})

    status = {
        "usable_for_comparison": not reasons, "reasons": reasons, "notes": notes,
        "gameweek": gameweek, "season": season, "players": int(len(out)),
        "linked_share": round(linked_share, 4),
        "unmatched_players": unmatched[["name", "fpl_code"]].values.tolist(),
        "team_differences": team_off,
        "position_agreement": round(pos_agree, 4),
        "position_differences": linked.loc[both & solio_pos.ne(linked["Pos"]), ["Web_Name", "position", "Pos"]].values.tolist(),
        "price_agreement": round(price_agree, 4),
        "per90_suppression": cut, "team_checks": shares,
        "capture": capture, "deadline_utc": None if deadline is None else deadline.isoformat(),
        "source_folder": str(folder.relative_to(ROOT)).replace("\\", "/"),
        "source_modified_utc": datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat(),
        "built_utc": datetime.now(timezone.utc).isoformat(),
    }
    return out.sort_values("Solio_Pts90", ascending=False), status


def default_gameweek():
    """The first forecast round in minutes.csv, which part 2 is about to use."""
    minutes = pd.read_csv(DATA / "minutes.csv", encoding="utf-8-sig", nrows=1)
    return min(int(c.split("_")[0]) for c in minutes.columns if re.fullmatch(r"\d+_xMins", c))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gameweek", type=int)
    parser.add_argument("--season", type=int, default=current_season())
    args = parser.parse_args()
    gameweek = None
    try:
        gameweek = args.gameweek or default_gameweek()
        table, status = build(gameweek, args.season)
        to_csv_atomic(table, OUT_FILE, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(table)} Solio players for GW{gameweek} to {OUT_FILE.name}")
    except Exception as error:  # benchmark only: never stop part 2
        OUT_FILE.unlink(missing_ok=True)
        # The gameweek, when known, lets the snapshot keep this status as the record of why Solio is missing.
        status = {"usable_for_comparison": False, "reasons": [f"{type(error).__name__}: {error}"],
                  "gameweek": gameweek, "season": args.season, "built_utc": datetime.now(timezone.utc).isoformat()}
        traceback.print_exc()
    STATUS_FILE.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"usable_for_comparison: {status['usable_for_comparison']}")
    for reason in status["reasons"]:
        print(f"WARNING: {reason}")
    for note in status.get("notes", []):
        print(f"Note: {note}")
    for name, code in status.get("unmatched_players", []):
        print(f"WARNING: no players_current.csv match for {name} (FPL code {code})")
    if status.get("position_differences"):
        print(f"Note: {len(status['position_differences'])} players have a different position in Solio")
    if "per90_suppression" in status:
        cut = status["per90_suppression"]
        print(f"Per-90 values blank for {cut['rows']} players (Solio's cut-off: under {cut['min_mins_shown']} expected minutes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
