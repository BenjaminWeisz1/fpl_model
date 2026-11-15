"""Reopen the personal website without re-exporting its data (part 3's server only).

    python run_site.py [--no-browser]
"""
import sys

from run_site_export import serve

if __name__ == "__main__":
    sys.exit(serve([a for a in sys.argv[1:] if a == "--no-browser"]))
