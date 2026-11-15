"""
Second audit of arrivals' transfers.csv positions against Transfermarkt starting positions (CHANGELOG §248).

    python audit_arrival_positions_v2.py          # proposals -> data/validation/position_audit_20261114/
    python audit_arrival_positions_v2.py --apply  # also rewrites the differing Pos cells of transfers.csv (text level,
                                                  # backup in data/archive/)

Why a second audit (user decision 2026-11-14): §239 changed a listing only when its rate moved by 0.05 or more, so it
left 37 listings whose first position was plainly wrong (Manzambi DM,CM,AM against 4% DM starts). The user's rule is
that a listing must match where the player played before arriving, whatever the rate effect.

Same source, window and identity match as §239 (`audit_arrival_positions.py`): starting-lineup rows, the three seasons
before the arrival season, any club and competition. What changes:
  * Transfermarkt's labels are read with the team's formation in that match (games.csv `*_club_formation`), so
    they mean what the project's own slots (Understat's) mean:
      - "Left/Right Midfield" is a wing-back (WB) when the team plays a back three or five (formation starts with 3 or
        5), a central midfielder (CM) in a diamond (4-4-2 Diamond, 4-3-1-2), otherwise a wide midfielder (WM);
      - "Left-Back/Right-Back" in a back five (formation starts with 5) is a wing-back;
      - "Second Striker" is a forward (FW) when the team plays two up front (formation ends in 2), otherwise AM;
      - with no formation recorded, the §239 mapping (WM; SS read as AM).
  * The listing built from the starts (positions with at least 20% of starts, most first, at most three) replaces
    the current one whenever they differ, for every arrival with at least 15 starts. No rate threshold.
Arrivals with fewer than 15 starts, or no Transfermarkt match, keep their listing (no evidence).
"""
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from fpl_model.player_ratings import audit_arrival_positions as v1  # noqa: E402

ROOT = v1.ROOT
DATA = v1.DATA
TM = v1.TM
OUT = ROOT / "data" / "validation" / "position_audit_20261114"
ARCHIVE = ROOT / "data" / "archive"

MIN_STARTS = v1.MIN_STARTS
SHARE_MIN = v1.SHARE_MIN
DIAMOND = {"4-4-2 Diamond", "4-3-1-2"}
PLAIN = {k: v for k, v in v1.TM_POS.items() if k not in ("Left Midfield", "Right Midfield", "Left-Back", "Right-Back",
                                                           "Second Striker")}


def slot(position, formation):
    f = str(formation).replace("Starting Line-up: ", "").strip() if isinstance(formation, str) else ""
    back = f[:1]
    if position in ("Left Midfield", "Right Midfield"):
        if back in ("3", "5"):
            return "WB"
        return "CM" if f in DIAMOND else "WM"
    if position in ("Left-Back", "Right-Back"):
        return "WB" if back == "5" else "FB"
    if position == "Second Striker":
        if not f:
            return "AM"
        return "FW" if f.split()[0].split("-")[-1] == "2" else "AM"
    return PLAIN.get(position)


def lineups():
    lu = pd.read_csv(TM / "game_lineups.csv.gz", usecols=["date", "game_id", "player_id", "club_id", "type", "position"])
    lu = lu[lu["type"].eq("starting_lineup")].copy()
    g = pd.read_csv(TM / "games.csv.gz", usecols=["game_id", "home_club_id", "away_club_id", "home_club_formation",
                                                  "away_club_formation"])
    lu = lu.merge(g, on="game_id", how="left", validate="many_to_one")
    lu["formation"] = lu["home_club_formation"].where(lu["club_id"].eq(lu["home_club_id"]),
                                                      lu["away_club_formation"].where(lu["club_id"].eq(lu["away_club_id"])))
    lu["date"] = pd.to_datetime(lu["date"])
    lu["slot"] = [slot(p, f) for p, f in zip(lu["position"], lu["formation"])]
    return lu


def audit():
    t = pd.read_csv(DATA / "transfers.csv", encoding="utf-8-sig", usecols=["Player", "Season", "Pos", "League", "Notes"])
    arr = t[t["Notes"].isna() & t["League"].ne("Premier League") & t["Pos"].notna() & t["Season"].ge(2017)
            & ~t["Pos"].astype(str).str.contains("GK")].copy()
    ids = v1.tm_player_ids(arr)
    lu = lineups()
    by_pid = dict(tuple(lu.groupby("player_id")))
    rate = v1.rate_fn()
    rows = []
    for r in arr.itertuples():
        pid, how = ids.get((r.Player, r.Season), (None, None))
        row = {"Player": r.Player, "Season": r.Season, "League": r.League, "Listed": r.Pos, "TM_player_id": pid,
               "Match": how, "TM_starts": 0}
        if pid is not None and pid in by_pid:
            g = by_pid[pid]
            lo, hi = pd.Timestamp(r.Season - 3, 8, 1), pd.Timestamp(r.Season, 7, 31)
            g = g[(g["date"] >= lo) & (g["date"] <= hi) & g["slot"].notna()]
            n = len(g)
            row["TM_starts"] = n
            row["No_formation_share"] = float(g["formation"].isna().mean()) if n else None
            if n:
                share = g["slot"].value_counts(normalize=True)
                row["TM_shares"] = "; ".join(f"{k} {v:.0%}" for k, v in share.items())
                raw = g["position"].value_counts(normalize=True)
                row["TM_labels"] = "; ".join(f"{k} {v:.0%}" for k, v in raw.items())
                if share.get("GK", 0) > 0.5:
                    row["Match"] = f"{how}; rejected: Transfermarkt player is a keeper"
                    rows.append(row)
                    continue
                derived = [k for k, v in share.items() if v >= SHARE_MIN and k != "GK"][:3] or [share.index[0]]
                row["Proposed"] = ",".join(derived)
                listed = ",".join(p.strip() for p in str(r.Pos).split(","))
                row["Change"] = bool(n >= MIN_STARTS and row["Proposed"] != listed)
                for col, lab in (("NPG/90", "npxG"), ("Assists/90", "xAG")):
                    row[f"{lab}_rate_listed"] = rate(r.Pos, col)
                    row[f"{lab}_rate_proposed"] = rate(row["Proposed"], col)
        rows.append(row)
    a = pd.DataFrame(rows)
    a["Change"] = a.get("Change", pd.Series(False, index=a.index)).fillna(False).astype(bool)
    return a


def apply(changes):
    """Rewrite only the Pos cells of the changed rows, at text level (CLAUDE.md §5.2)."""
    path = DATA / "transfers.csv"
    backup = ARCHIVE / f"transfers_before_position_audit2_{date.today():%Y%m%d}.csv"
    if not backup.exists():
        shutil.copy2(path, backup)
    t = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    key = {(p, str(s)): new for p, s, new in zip(changes["Player"], changes["Season"], changes["Proposed"])}
    hit = 0
    for i, (p, s, notes, league) in enumerate(zip(t["Player"], t["Season"], t["Notes"], t["League"])):
        if (p, s) in key and notes == "" and league != "Premier League":
            t.at[i, "Pos"] = key[(p, s)]
            hit += 1
    if hit != len(key):
        raise SystemExit(f"matched {hit} rows for {len(key)} changes; nothing written")
    tmp = path.with_suffix(".csv.tmp")
    t.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)
    return backup, hit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = audit()
    a.to_csv(OUT / "audit.csv", index=False, encoding="utf-8-sig")
    ch = a[a["Change"]].sort_values(["Season", "Player"])
    ch.to_csv(OUT / "changes.csv", index=False, encoding="utf-8-sig")
    summary = {"arrivals": int(len(a)), "matched": int(a["TM_player_id"].notna().sum()),
               "with_15_starts": int((a["TM_starts"] >= MIN_STARTS).sum()), "changes": int(len(ch)),
               "changes_first_position": int(sum(x.split(",")[0] != y.split(",")[0].strip()
                                                 for x, y in zip(ch["Proposed"], ch["Listed"]))),
               "match_methods_of_changes": ch["Match"].value_counts().to_dict(),
               "rule": {"min_starts": MIN_STARTS, "share_min": SHARE_MIN, "diamond": sorted(DIAMOND)}}
    if "--apply" in sys.argv:
        backup, hit = apply(ch)
        summary["applied"] = {"rows": hit, "backup": str(backup.relative_to(ROOT))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
