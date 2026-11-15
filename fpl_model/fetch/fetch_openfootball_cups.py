"""One-off backfill: EFL Cup results 2018/19-2024/25 from openfootball/england (CC0), frozen to data/raw/openfootball/.
Fills Transfermarkt's League Cup gap (AGENDA 2026-10-31). Existing files are never replaced.
Run: python -u -X utf8 scripts/fetch/fetch_openfootball_cups.py
"""
from fpl_model import paths
import hashlib
import json
import time
from pathlib import Path

import requests

BASE = "https://raw.githubusercontent.com/openfootball/england/master/{}/eflcup.txt"
SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
OUT = paths.RAW / "openfootball"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for s in SEASONS:
        path = OUT / f"eflcup_{s}.txt"
        if not path.exists():
            r = requests.get(BASE.format(s), timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content)
            time.sleep(2)
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(len(manifest), "files")


if __name__ == "__main__":
    main()
