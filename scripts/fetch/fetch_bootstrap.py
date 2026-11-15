"""
Pulls the current player list from the FPL API's bootstrap-static endpoint,
automatically disambiguating any players who share a raw web_name (e.g. two
different players both named "Sangare" would otherwise collide in every
downstream join keyed on Web_Name).
"""
import sys
import requests
import pandas as pd
from pathlib import Path
from team_name_map import to_fbref_name

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atomic_write import to_csv_atomic  # noqa: E402

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
_DATA = Path(__file__).resolve().parents[2] / "data" / "current"
OUT_FILE = _DATA / "players_current.csv"
OVERRIDES_FILE = _DATA / "web_name_overrides.csv"
DEADLINES_FILE = _DATA / "gameweek_deadlines.csv"

POSITION_MAP = {1: "G", 2: "D", 3: "M", 4: "F"}


def write_deadlines(events, path=DEADLINES_FILE):
    """Each gameweek's FPL deadline, so snapshots can tell before from after it without a fetch."""
    rows = [{"Gameweek": e["id"], "Deadline_UTC": e["deadline_time"]} for e in events]
    to_csv_atomic(pd.DataFrame(rows), path, index=False, encoding="utf-8-sig")


SET_PIECE_FILE = _DATA / "set_piece_orders.csv"
SET_PIECE_FIELDS = ["corners_and_indirect_freekicks_order", "direct_freekicks_order", "penalties_order"]


def append_set_piece_orders(data, path=SET_PIECE_FILE):
    """Append FPL's set-piece pecking order, so a history builds up (CHANGELOG §196). FPL keeps only the current order.

    One row per player with any order set, tagged with the next gameweek; a row is new when its (Next_GW, FPL_ID,
    orders) haven't been seen, so a change within a week is also kept. A player absent from a gameweek's rows had no
    duty in that fetch. Never rewrites old rows; never stops the pipeline.
    """
    try:
        nxt = next((e["id"] for e in data["events"] if e.get("is_next")), None)
        teams = {t["id"]: to_fbref_name(t["name"]) for t in data["teams"]}
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = []
        for p in data["elements"]:
            orders = {f: p.get(f) for f in SET_PIECE_FIELDS}
            if all(v is None for v in orders.values()):
                continue
            rows.append({"Next_GW": nxt, "Fetched_UTC": stamp, "FPL_ID": p["id"],
                         "Player": f'{p["first_name"]} {p["second_name"]}'.strip(), "Web_Name": p["web_name"],
                         "Team": teams.get(p["team"], "UNKNOWN"), **orders})
        new = pd.DataFrame(rows)
        if new.empty:
            print("Set-piece orders: none set in bootstrap-static")
            return
        key = ["Next_GW", "FPL_ID"] + SET_PIECE_FIELDS
        ints = {f: "Int64" for f in key}
        new = new.astype(ints)
        if path.exists():
            old = pd.read_csv(path, encoding="utf-8-sig", dtype=ints)
            seen = set(old[key].astype(str).agg("|".join, axis=1))
            new = new[~new[key].astype(str).agg("|".join, axis=1).isin(seen)]
            out = pd.concat([old, new], ignore_index=True)
        else:
            out = new
        if len(new):
            to_csv_atomic(out, path, index=False, encoding="utf-8-sig")
        print(f"Set-piece orders: {len(new)} new rows for GW{nxt} ({len(out)} total)")
    except Exception as exc:  # a snapshot must never stop part 1
        print(f"WARNING: set-piece orders not saved: {exc!r}")


CODES_FILE = _DATA / "fpl_player_codes.csv"


def append_player_codes(data, path=CODES_FILE):
    """Append-only (Opta_Code, Player) pairs: FPL's `code` is the Opta player ID that FotMob pages carry as
    optaId (CHANGELOG §197). Kept across seasons so a player who leaves still resolves."""
    rows = pd.DataFrame([{"Opta_Code": str(p["code"]), "Player": f'{p["first_name"]} {p["second_name"]}'.strip(),
                          "First_Seen_GW": next((e["id"] for e in data["events"] if e.get("is_next")), None)}
                         for p in data["elements"]])
    old = pd.read_csv(path, encoding="utf-8-sig", dtype={"Opta_Code": str}) if path.exists() \
        else pd.DataFrame(columns=rows.columns)
    seen = set(zip(old["Opta_Code"], old["Player"]))
    new = rows[[(c, p) not in seen for c, p in zip(rows["Opta_Code"], rows["Player"])]]
    if len(new):
        to_csv_atomic(pd.concat([old, new], ignore_index=True), path, index=False, encoding="utf-8-sig")
    print(f"fpl_player_codes.csv: {len(new)} new code(s), {len(old) + len(new)} in total")


def main():
    resp = requests.get(BOOTSTRAP_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    write_deadlines(data["events"])
    append_set_piece_orders(data)
    append_player_codes(data)

    team_lookup = {team["id"]: to_fbref_name(team["name"]) for team in data["teams"]}
    team_short_lookup = {team["id"]: team["short_name"] for team in data["teams"]}

    rows = []
    for p in data["elements"]:
        rows.append({
            "Player": f'{p["first_name"]} {p["second_name"]}'.strip(),
            "First_Name": p["first_name"],
            "Raw_Web_Name": p["web_name"],
            "Pos": POSITION_MAP.get(p["element_type"], "UNKNOWN"),
            "Team": team_lookup.get(p["team"], "UNKNOWN"),
            "Team_Short": team_short_lookup.get(p["team"], "UNKNOWN"),
            "Price": p["now_cost"] / 10,
            "Status": p["status"],
        })
    df = pd.DataFrame(rows)

    if OVERRIDES_FILE.exists():
        overrides = pd.read_csv(OVERRIDES_FILE, encoding="utf-8-sig")
        overrides_map = dict(zip(overrides["Player"], overrides["Web_Name"]))
    else:
        overrides = pd.DataFrame(columns=["Player", "Web_Name"])
        overrides_map = {}

    # start from each player's persisted override if one exists, else their raw name
    df["Web_Name"] = df.apply(
        lambda r: overrides_map.get(r["Player"], r["Raw_Web_Name"]), axis=1)

    # only look for NEW collisions among players not already covered by a
    # persisted override -- an already-resolved player's name never gets
    # recomputed just because the league's current roster changed
    unresolved = df[~df["Player"].isin(overrides_map)]
    new_overrides = []

    dupe_names = unresolved["Web_Name"][unresolved["Web_Name"].duplicated(keep=False)].unique()
    for name in dupe_names:
        colliding = unresolved[unresolved["Web_Name"] == name]
        for idx, row in colliding.iterrows():
            candidate = f'{row["First_Name"][0]}. {row["Raw_Web_Name"]}'
            df.at[idx, "Web_Name"] = candidate
            new_overrides.append({"Player": row["Player"], "Web_Name": candidate})

    # re-check: does the first-initial fix itself still collide for anyone?
    # (two players sharing both first initial AND surname)
    still_dupe = df["Web_Name"][df["Web_Name"].duplicated(keep=False)].unique()
    for name in still_dupe:
        colliding = df[df["Web_Name"] == name]
        for idx, row in colliding.iterrows():
            df.at[idx, "Web_Name"] = row["Player"]  # fall back to full name
            new_overrides = [o for o in new_overrides if o["Player"] != row["Player"]]
            new_overrides.append({"Player": row["Player"], "Web_Name": row["Player"]})

    if new_overrides:
        print(f"New disambiguation applied for {len(new_overrides)} player(s):")
        for o in new_overrides:
            print(f"  {o['Player']} -> {o['Web_Name']}")
        overrides = pd.concat([overrides, pd.DataFrame(new_overrides)], ignore_index=True)
        overrides = overrides.drop_duplicates(subset=["Player"], keep="last")
        to_csv_atomic(overrides, OVERRIDES_FILE, index=False, encoding="utf-8-sig")
    else:
        print("No new web_name collisions found.")

    out_cols = ["Player", "Web_Name", "Raw_Web_Name", "Pos", "Team", "Team_Short", "Price", "Status"]
    to_csv_atomic(df[out_cols], OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(df)} players to {OUT_FILE}")


if __name__ == "__main__":
    main()