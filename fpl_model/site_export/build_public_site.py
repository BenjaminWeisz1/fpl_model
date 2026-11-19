"""Part 3, optional: build a copy of the website for Netlify without the Solver page.

    python -m fpl_model.site_export.build_public_site

Run after run_site_export.py (it reads site/data/*.json, which part 3 writes). Writes data/site_public/, a plain
static folder: index.html without the Solver tab, section and script, the three other assets, data/*.json, plus
Netlify's _headers (no caching of the data, no search indexing) and robots.txt. Drag that folder onto Netlify, or
run `netlify deploy --prod --dir data/site_public`. The Solver needs site/server.py and stays local.

Fails loudly if index.html no longer has the Solver pieces it removes, so a layout change can't ship them silently.
"""
import re
import shutil
import sys

from fpl_model import paths

OUT = paths.DATA / "site_public"
ASSETS = ["style.css", "heatmap.js", "app.js"]

# (label, pattern) pairs; each must match exactly once in index.html.
SOLVER_PARTS = [
    ("navbar link", re.compile(r'[ \t]*<a href="#solver"[^>]*>Solver</a>\n')),
    ("view section", re.compile(r'[ \t]*<section id="view-solver".*?</section>\n', re.S)),
    ("script tag", re.compile(r'[ \t]*<script src="solver\.js"></script>\n')),
]

HEADERS = """/data/*
  Cache-Control: no-cache
/*
  X-Robots-Tag: noindex, nofollow
"""

ROBOTS = "User-agent: *\nDisallow: /\n"


def strip_solver(html):
    """Return index.html without the Solver tab, section and script."""
    for label, pattern in SOLVER_PARTS:
        html, n = pattern.subn("", html)
        if n != 1:
            raise SystemExit(f"index.html: expected one Solver {label}, found {n}; update SOLVER_PARTS")
    return html


def main():
    src = paths.SITE
    data_files = sorted((src / "data").glob("*.json"))
    if not data_files:
        raise SystemExit("site/data has no JSON: run `python run_site_export.py --no-serve` first")
    # Overwrite in place: the repo sits in OneDrive, which locks a just-emptied folder while it syncs, so
    # deleting the tree fails (WinError 5). Only files left over from an earlier build are removed.
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    keep = {"index.html", "_headers", "robots.txt", *ASSETS, *(f"data/{f.name}" for f in data_files)}
    for old in OUT.rglob("*"):
        if old.is_file() and old.relative_to(OUT).as_posix() not in keep:
            old.unlink()
    html = (src / "index.html").read_text(encoding="utf-8")
    (OUT / "index.html").write_text(strip_solver(html), encoding="utf-8", newline="\n")
    for name in ASSETS:
        shutil.copy2(src / name, OUT / name)
    for f in data_files:
        shutil.copy2(f, OUT / "data" / f.name)
    (OUT / "_headers").write_text(HEADERS, encoding="utf-8", newline="\n")
    (OUT / "robots.txt").write_text(ROBOTS, encoding="utf-8", newline="\n")
    print(f"Wrote {OUT} ({len(data_files)} data files). Deploy: netlify deploy --prod --dir \"{OUT}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
