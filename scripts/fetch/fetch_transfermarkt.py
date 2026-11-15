"""One-off backfill: dcaribou/transfermarkt-datasets (CC0) tables, frozen to data/raw/transfermarkt/ (AGENDA 2026-10-31).

Existing files are never replaced. A manifest records each file's SHA256, size and retrieval time. The source ends
6 July 2026. Run: python -u -X utf8 scripts/fetch/fetch_transfermarkt.py
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
FILES = ["games.csv.gz", "appearances.csv.gz", "player_valuations.csv.gz", "players.csv.gz", "clubs.csv.gz", "game_lineups.csv.gz",
         "transfers.csv.gz"]  # transfers added 2026-11-09 (§212)
OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "transfermarkt"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in FILES:
        path = OUT / name
        if path.exists():
            print("kept", name)
            continue
        r = requests.get(BASE + name, timeout=300)
        if r.status_code == 404:
            print("not published:", name)
            continue
        r.raise_for_status()
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(r.content)
        tmp.replace(path)
        manifest[name] = {"sha256": hashlib.sha256(r.content).hexdigest(), "bytes": len(r.content), "url": BASE + name,
                          "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        print("saved", name, len(r.content))
        time.sleep(3)
    manifest_path.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
