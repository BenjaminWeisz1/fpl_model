"""Exports what sofascore_player_match_stats.csv leaves out of the frozen raw files (AGENDA 2026-10-31, item 5).

Writes data/current/sofascore_player_extra.csv.gz (every statistics field, one row per listed player and match) and
data/current/sofascore_missing_players.csv (the source's missingPlayers lists: who was out, with its type and reason).
Reads data/raw/sofascore/matches/ only; no network; not a pipeline stage.
Run: python -u -X utf8 scripts/fetch/export_sofascore_extra.py
"""
from fpl_model import paths
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = paths.ROOT
RAW = paths.RAW / "sofascore" / "matches"
DATA = paths.CURRENT
SKIP = {"ratingVersions", "statisticsType"}


def main():
    rows, missing, files, seen, by_season = [], [], 0, Counter(), Counter()
    for season_dir in sorted(RAW.iterdir()):
        season = int(season_dir.name)
        for path in sorted(season_dir.glob("*.json")):
            d = json.loads(path.read_text(encoding="utf-8"))
            files += 1
            event_id = int(path.stem)
            for side in ("home", "away"):
                block = d["payload"].get(side) or {}
                for p in block.get("players", []):
                    stats = p.get("statistics") or {}
                    row = {"Season": season, "Event_ID_Sofascore": event_id, "Side": side, "Player_ID_Sofascore": p["player"]["id"],
                           "Substitute": p.get("substitute"), "Captain": p.get("captain"), "Position": p.get("position")}
                    for k, v in stats.items():
                        if k not in SKIP and not isinstance(v, (dict, list)):
                            row[k] = v
                            seen[(season, k)] += 1
                    rows.append(row)
                for m in block.get("missingPlayers") or []:
                    by_season[season] += 1
                    pl = m.get("player") or {}
                    missing.append({"Season": season, "Event_ID_Sofascore": event_id, "Side": side, "Player_ID_Sofascore": pl.get("id"),
                                    "Player": pl.get("name"), "Position": pl.get("position"), "Type": m.get("type"), "Reason": m.get("reason"),
                                    "Description": m.get("description"), "External_Type": m.get("externalType"),
                                    "Expected_End": m.get("expectedEndDate")})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "sofascore_player_extra.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    mp = pd.DataFrame(missing)
    mp.to_csv(DATA / "sofascore_missing_players.csv", index=False, encoding="utf-8-sig")
    print("files", files, "rows", len(out), "fields", out.shape[1], "missing rows", len(mp))
    cov = out.drop(columns=["Event_ID_Sofascore", "Side", "Player_ID_Sofascore"]).groupby("Season").agg(lambda s: s.notna().mean())
    played = out[out.get("minutesPlayed").notna()]
    cov_p = played.drop(columns=["Event_ID_Sofascore", "Side", "Player_ID_Sofascore"]).groupby("Season").agg(lambda s: s.notna().mean()).T.round(2)
    cov_p.to_csv(DATA.parent / "validation" / "sofascore_extra_coverage_20261031.csv", encoding="utf-8-sig")
    print(cov_p.loc[[c for c in ("rating", "expectedAssists", "goalsPrevented", "minutesPlayed", "touches") if c in cov_p.index]].to_string())
    print("missing rows by season", dict(sorted(by_season.items())))
    if len(mp):
        print(mp.groupby(["Type", "Reason"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
