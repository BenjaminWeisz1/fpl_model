"""
Fills penalty_log.csv automatically from Understat shot data (CHANGELOG §75), then
reconciles player_match_stats.csv and match_results.csv with the log (§73).

Understat's cached getMatchData payloads (data/raw/understat/matches/, downloaded
each week by fetch_understat.py) list every shot with its `situation`; a penalty is
situation == "Penalty", scored if result == "Goal". On 2025/26 this reproduced the
manual log exactly (92 attempts, 77 scored, player by player), and in 2026/27 it
found two scored penalties the log lacked (Saka v Sunderland, Groß v Coventry),
both corroborated by FPL goals and Sofascore penalties won.

Players and sides are mapped to project names through understat_match_positions.csv
(match, Understat player -> team/opponent/venue) and player_match_positions.csv
(-> project Player), so this runs after the position crosswalk stage.

The log stays the single source of truth and is only ever appended to:
  * a penalty Understat has and the log lacks is appended (same columns, BOM, CRLF)
    and recorded in penalty_log_understat_additions.csv;
  * a log row Understat lacks is kept and printed as a warning (a manual entry, or
    Understat not yet updated);
  * a penalty whose player cannot be mapped is printed and skipped.

From 2026 (CHANGELOG §197) the penalties of those seasons come from FotMob instead (fotmob_shots.csv: situation
"Penalty", scored if the event is a goal; players already keyed to project names), with the same append-only rules;
FotMob additions are audited in penalty_log_fotmob_additions.csv. Understat files of those seasons are not read.

Run locally (after fetch_fotmob_matches.py and build_position_crosswalk.py):
    python penalty_log_understat.py [--dry-run]
"""
from fpl_model import paths
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fpl_model.fetch import penalty_reconcile
from fpl_model.fetch.fetch_fotmob_matches import FIRST_SEASON as FOTMOB_FIRST_SEASON, SHOTS_OUT as FOTMOB_SHOTS

_ROOT = paths.ROOT
_DATA = paths.CURRENT
MATCH_CACHE = paths.RAW / "understat" / "matches"
POSITIONS_FILE = paths.CURRENT / "understat_match_positions.csv"
CROSSWALK_FILE = paths.CURRENT / "player_match_positions.csv"
ADDITIONS_FILE = paths.CURRENT / "penalty_log_understat_additions.csv"
FOTMOB_ADDITIONS_FILE = paths.CURRENT / "penalty_log_fotmob_additions.csv"
KEY = ["Season", "Team", "Opponent", "Venue", "Player", "Scored"]


def understat_penalties(first_season=penalty_reconcile.FIRST_FPL_SEASON):
    """Every Understat penalty shot from first_season on, with project names where mappable."""
    positions = pd.read_csv(POSITIONS_FILE, encoding="utf-8-sig", low_memory=False)
    positions = positions[(positions["Season"] >= first_season) & (positions["Season"] < FOTMOB_FIRST_SEASON)]
    sides = positions[["Season", "MatchID_Understat", "Player_ID_Understat", "Team", "Opponent", "Venue"]] \
        .drop_duplicates(["MatchID_Understat", "Player_ID_Understat"])
    rows = []
    for match_id in sorted(positions["MatchID_Understat"].unique()):
        path = MATCH_CACHE / f"{match_id}.json"
        if not path.exists():
            continue
        shots = json.loads(path.read_text(encoding="utf-8"))["shots"]
        for s in shots["h"] + shots["a"]:
            if s["situation"] == "Penalty":
                rows.append({"MatchID_Understat": int(match_id), "Player_ID_Understat": int(s["player_id"]),
                             "Understat_Player": s["player"], "Minute": int(s["minute"]),
                             "Scored": s["result"] == "Goal"})
    pens = pd.DataFrame(rows, columns=["MatchID_Understat", "Player_ID_Understat", "Understat_Player", "Minute", "Scored"])
    pens = pens.merge(sides, on=["MatchID_Understat", "Player_ID_Understat"], how="left")
    crosswalk = pd.read_csv(CROSSWALK_FILE, encoding="utf-8-sig", low_memory=False)
    crosswalk = crosswalk[["Season", "Team", "Opponent", "Venue", "Player_ID_Understat", "Player"]] \
        .drop_duplicates(["Season", "Team", "Opponent", "Venue", "Player_ID_Understat"])
    return pens.merge(crosswalk, on=["Season", "Team", "Opponent", "Venue", "Player_ID_Understat"], how="left")


def fotmob_penalties():
    """Every FotMob penalty shot from FOTMOB_FIRST_SEASON on (players keyed to project names by fetch_fotmob_matches)."""
    s = pd.read_csv(FOTMOB_SHOTS, encoding="utf-8-sig", low_memory=False)
    s = s[(s["Season"] >= FOTMOB_FIRST_SEASON) & (s["Situation"] == "Penalty")]
    return pd.DataFrame({"MatchID_FotMob": s["MatchID_FotMob"].astype(int), "FotMob_Player": s["Player_FotMob"],
                         "Minute": s["Minute"].astype(float).astype(int), "Scored": s["Event_Type"] == "Goal",
                         "Season": s["Season"], "Team": s["Team"], "Opponent": s["Opponent"], "Venue": s["Venue"],
                         "Player": s["Player"]})


def compare(log, pens, id_col="MatchID_Understat", name_col="Understat_Player"):
    """(rows to append, log rows the source lacks, unmapped penalties), counting repeats."""
    log = log.copy()
    log["Player"] = log["Player"].replace(penalty_reconcile.ALIASES)
    seasons = set(pens["Season"].dropna())
    log = log[log["Season"].isin(seasons)]
    unmapped = pens[pens["Player"].isna()]
    pens = pens.dropna(subset=["Player"]).copy()
    pens["Season"] = pens["Season"].astype(int)
    have = log.groupby(KEY).size()
    want = pens.groupby(KEY).size()
    counts = pd.concat([have.rename("log"), want.rename("understat")], axis=1).fillna(0).astype(int)
    add, extra = [], []
    for key, row in counts.iterrows():
        source = pens.set_index(KEY).loc[[key]] if row["understat"] else None
        for i in range(row["understat"] - row["log"]):
            add.append(dict(zip(KEY, key)) | {id_col: int(source[id_col].iloc[i]),
                                               "Minute": int(source["Minute"].iloc[i]),
                                               name_col: source[name_col].iloc[i]})
        if row["log"] > row["understat"]:
            extra.append(dict(zip(KEY, key)) | {"log": int(row["log"]), "understat": int(row["understat"])})
    return pd.DataFrame(add), extra, unmapped


def append_to_log(add, path=penalty_reconcile.PENALTY_LOG_FILE, id_col="MatchID_Understat"):
    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    if not raw.endswith(newline.encode()):
        raw += newline.encode()
        path.write_bytes(raw)
    lines = "".join(f"{int(r.Season)},{r.Venue},{r.Player},{r.Team},{r.Opponent},{'TRUE' if r.Scored else 'FALSE'}{newline}"
                    for r in add.sort_values(["Season", id_col, "Minute"]).itertuples())
    with open(path, "ab") as h:
        h.write(lines.encode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    log = pd.read_csv(penalty_reconcile.PENALTY_LOG_FILE, encoding="utf-8-sig").rename(columns={"HomeAway": "Venue"})
    log["Scored"] = log["Scored"].astype(str).str.strip().str.upper() == "TRUE"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for source, pens, id_col, name_col, audit_file in (
            ("Understat", understat_penalties(), "MatchID_Understat", "Understat_Player", ADDITIONS_FILE),
            ("FotMob", fotmob_penalties(), "MatchID_FotMob", "FotMob_Player", FOTMOB_ADDITIONS_FILE)):
        add, extra, unmapped = compare(log, pens, id_col, name_col)
        print(f"{source} penalties found: {len(pens)}; to append to penalty_log.csv: {len(add)}")
        for r in add.itertuples():
            print(f"  + {r.Season} {r.Venue} {r.Player} ({r.Team} v {r.Opponent}), {'scored' if r.Scored else 'missed'}, {r.Minute}'")
        for r in extra:
            print(f"WARNING: penalty_log row not in {source} (kept): {r}")
        for r in unmapped.itertuples():
            print(f"WARNING: {source} penalty with no project player mapping (skipped): "
                  f"{getattr(r, name_col)}, match {getattr(r, id_col)}, {r.Minute}'")
        if args.dry_run or not len(add):
            continue
        append_to_log(add, id_col=id_col)
        add.assign(Added_UTC=stamp).to_csv(audit_file, mode="a", header=not audit_file.exists(), index=False,
                                           encoding="utf-8-sig")
    if args.dry_run:
        return
    penalty_reconcile.reconcile_files(apply=True)


if __name__ == "__main__":
    main()
