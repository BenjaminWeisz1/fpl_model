"""
Every folder the pipeline reads or writes, declared once.

Modules import these constants instead of rebuilding paths from ``__file__``, so moving a module
never changes where it finds its data. ``data/`` is local only and never committed.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "fpl_model"

DATA = ROOT / "data"
CURRENT = DATA / "current"          # live pipeline inputs and outputs, refreshed weekly
RAW = DATA / "raw"                  # frozen downloads, never rewritten
SNAPSHOTS = DATA / "snapshots"      # immutable per-gameweek output archives
VALIDATION = DATA / "validation"    # experiment evidence and frozen protocols
ARCHIVE = DATA / "archive"          # superseded data kept for traceability
OPTIMIZER = DATA / "optimizer"      # transfer optimizer runs and plans
SOLIO = DATA / "solio"
FPLREVIEW = DATA / "fplreview"

CACHE = ROOT / "cache"              # regenerable model cache
IMAGES = ROOT / "images"            # weekly charts, published in the README
SITE = ROOT / "site"                # the local website; site/data is written by part 3
