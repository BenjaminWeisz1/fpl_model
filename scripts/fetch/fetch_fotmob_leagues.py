"""
Freeze FotMob league-season player stat lists for new-player priors (CHANGELOG §115, §116).

    python fetch_fotmob_leagues.py                 # every league in LEAGUES, seasons FIRST_SEASON .. CURRENT_SEASON-1
    python fetch_fotmob_leagues.py --league Championship --season 2025
    python fetch_fotmob_leagues.py --refresh-pages # re-read league pages (new season IDs)

Source rules (CHANGELOG §115): FotMob's robots.txt disallows www.fotmob.com/api/* for generic agents, so this
reads only the public league stats page (www.fotmob.com/leagues/<id>/stats/<slug>, for its season IDs) and the
stat files it links on data.fotmob.com. A missing stat file answers 403 AccessDenied from S3 and is recorded as
missing; any other 403 or a 429 stops the run. At least 6 s between requests. Personal, non-commercial use.

Output (frozen, never rewritten): data/raw/fotmob/league/<fotmob_id>_<season>_<stat>.json, the league pages in
data/raw/fotmob/pages/, and data/raw/fotmob/manifest.json (url, status, UTC time, SHA256 per file).
Season = the year the season starts (2025 = 2025/26; calendar-year leagues: 2025 = the 2025 season).
FotMob lists are penalty-inclusive: xG's sub-stat is goals, goals' sub-stat is penalty goals.
"""
import sys
import argparse
import hashlib
import json
import random
import re
import shutil
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "fotmob"
LEAGUE_DIR = RAW / "league"
PAGE_DIR = RAW / "pages"
MANIFEST = RAW / "manifest.json"
PROBE = ROOT / "data" / "validation" / "fotmob_coverage_20261029"   # §115 probe cache, reused when present
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from league_rates import current_season  # noqa: E402

CURRENT_SEASON = current_season()
FIRST_SEASON = 2020   # complete lists start in 2020/21 (§115); 2018/19-2019/20 are fragments
STATS = ["expected_goals", "expected_assists", "goals", "goal_assist", "mins_played"]
MIN_GAP = 6.0
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")

# transfers.csv League -> (FotMob league id, page slug)
LEAGUES = {
    "Championship": (48, "championship"), "Eredivisie": (57, "eredivisie"),
    "Primeira Liga": (61, "liga-portugal"), "Belgian Pro League": (40, "first-division-a"),
    "League One": (108, "league-one"), "Brazilian Série A": (268, "serie-a"),
    "Scottish Premiership": (64, "premiership"), "Turkish Süper Lig": (71, "super-lig"),
    "League Two": (109, "league-two"), "Liga Argentina": (112, "liga-profesional"),
    "Allsvenskan": (67, "allsvenskan"), "Swiss Super Lg": (69, "super-league"),
    "Austrian Bundesliga": (38, "bundesliga"), "Danish Superliga": (46, "superligaen"),
    "MLS": (130, "mls"), "Greek Super League": (135, "super-league-1"),
    "Ukrainian Premier League": (441, "premier-league"), "Ligue 2": (110, "ligue-2"),
    "National League": (117, "national-league"), "2. Bundesliga": (146, "2-bundesliga"),
    "Czech First League": (122, "1-liga"), "Eliteserien": (59, "eliteserien"),
}


# Calendar-year leagues: an arrival in PL season S last played FotMob season S (he leaves mid-season), so these
# also need the current year. Its file covers the season to the fetch date, which includes the whole stint of
# anyone who has already left.
CALENDAR_LEAGUES = {"Brazilian Série A", "Liga Argentina", "MLS", "Allsvenskan", "Eliteserien"}


class Blocked(Exception):
    pass


class Fetcher:
    def __init__(self):
        self.last = 0.0
        self.n = 0
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
        self.probe = {}
        log = PROBE / "requests_log.csv"
        if log.exists():
            for line in log.read_text(encoding="utf-8").splitlines()[1:]:
                parts = line.split(",")
                if len(parts) >= 6 and parts[3] == "200":
                    self.probe[parts[2]] = PROBE / "raw" / parts[5]

    def save_manifest(self):
        MANIFEST.write_text(json.dumps(self.manifest, indent=1, sort_keys=True), encoding="utf-8")

    def get(self, url, dest):
        """Text of url, saved to dest (frozen). None when the file does not exist at the source."""
        assert "fotmob.com/api" not in url, url
        rel = dest.relative_to(RAW).as_posix()
        if dest.exists():
            return dest.read_text(encoding="utf-8")
        if self.manifest.get(rel, {}).get("status") == 403:
            return None
        if url in self.probe and self.probe[url].exists():
            shutil.copyfile(self.probe[url], dest)
            status, when = 200, "probe-cache"
        else:
            wait = MIN_GAP + random.uniform(0.3, 1.5) - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "en-GB,en;q=0.9"}, timeout=40)
            self.last = time.time()
            self.n += 1
            when = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            print(f"[{self.n}] {r.status_code} {len(r.content)} {url}", flush=True)
            if r.status_code == 429:
                raise Blocked(f"429 on {url}")
            if r.status_code == 403:
                if "data.fotmob.com" in url and "<Code>AccessDenied</Code>" in r.text:
                    self.manifest[rel] = {"url": url, "status": 403, "utc": when}
                    self.save_manifest()
                    return None
                raise Blocked(f"non-S3 403 on {url}")
            r.raise_for_status()
            dest.write_text(r.text, encoding="utf-8")
            status = 200
        self.manifest[rel] = {"url": url, "status": status, "utc": when,
                              "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
        self.save_manifest()
        return dest.read_text(encoding="utf-8")


def season_links(f, league, refresh):
    fid, slug = LEAGUES[league]
    dest = PAGE_DIR / f"{fid}.html"
    if refresh and dest.exists():
        dest.unlink()
    html = f.get(f"https://www.fotmob.com/leagues/{fid}/stats/{slug}", dest)
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.S)
    stats = json.loads(m.group(1))["props"]["pageProps"].get("stats") or {}
    out = {}
    for s in stats.get("seasonStatLinks") or []:
        start = int(str(s["Name"])[:4])
        out.setdefault(start, s["TournamentId"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", action="append")
    ap.add_argument("--season", type=int, action="append")
    ap.add_argument("--refresh-pages", action="store_true")
    a = ap.parse_args()
    for d in (LEAGUE_DIR, PAGE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    seasons = a.season or list(range(FIRST_SEASON, CURRENT_SEASON))
    f = Fetcher()
    try:
        for league in a.league or LEAGUES:
            fid = LEAGUES[league][0]
            links = season_links(f, league, a.refresh_pages)
            league_seasons = seasons if a.season or league not in CALENDAR_LEAGUES else seasons + [CURRENT_SEASON]
            for s in league_seasons:
                tid = links.get(s)
                if tid is None:
                    print(f"{league} {s}: no season at FotMob", flush=True)
                    continue
                for stat in STATS:
                    f.get(f"https://data.fotmob.com/stats/{fid}/season/{tid}/{stat}.json",
                          LEAGUE_DIR / f"{fid}_{s}_{stat}.json")
    except Blocked as e:
        print(f"STOPPED: {e}", flush=True)
        raise SystemExit(2)
    print(f"done: {f.n} network requests", flush=True)


if __name__ == "__main__":
    main()
