"""
One-off backfill of historical FPL gameweek data from the vaastav/Fantasy-Premier-League archive
(CHANGELOG §94). Frozen raw output: data/raw/vaastav/<YYYY-YY>/, never rewritten (an existing
file is skipped), with SHA256 hashes in data/raw/vaastav/manifest.json.

Per season 2017-18 .. 2024-25 (project Seasons 2017-2024): gws/merged_gw.csv (one row per player
per fixture, FPL-definition assists, minutes, bonus, bps, opponent_team id, was_home,
kickoff_time), fixtures.csv, players_raw.csv and raw.json (the season's bootstrap snapshot, which
holds the team names). Personal, non-commercial use; plain requests with a 1 s pause.

Run locally:
    python fetch_vaastav.py
"""
import hashlib
import json
import time
from pathlib import Path

import requests

_RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "vaastav"
BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2017, 2025)]
FILES = ["gws/merged_gw.csv", "fixtures.csv", "players_raw.csv", "raw.json"]
OPTIONAL = {"fixtures.csv", "raw.json"}  # absent for 2017-18 in the archive
SHARED = ["master_team_list.csv"]  # season, team id -> team name, every season
PAUSE = 1.0


def main():
    manifest_path = _RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    try:
        _fetch(manifest)
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Manifest: {len(manifest)} files -> {manifest_path}")


def _fetch(manifest):
    jobs = [("", name) for name in SHARED] + [(season, name) for season in SEASONS for name in FILES]
    for season, name in jobs:
        target = _RAW / season / name if season else _RAW / name
        key = f"{season}/{name}" if season else name
        url = f"{BASE}/{season}/{name}" if season else f"{BASE}/{name}"
        if target.exists():
            if key not in manifest:  # frozen by an interrupted earlier run
                data = target.read_bytes()
                manifest[key] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                                 "url": url, "retrieved_utc": "earlier run"}
            print(f"  {key}: already frozen, skipped")
            continue
        r = requests.get(url, timeout=60)
        if r.status_code == 404 and name in OPTIONAL:
            manifest[key] = {"missing_in_archive": True}
            print(f"  {key}: not in the archive (optional)")
            time.sleep(PAUSE)
            continue
        r.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(r.content)
        manifest[key] = {"sha256": hashlib.sha256(r.content).hexdigest(), "bytes": len(r.content),
                         "url": url,
                         "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(f"  {key}: {len(r.content):,} bytes")
        time.sleep(PAUSE)


if __name__ == "__main__":
    main()
