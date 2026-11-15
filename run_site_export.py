"""
Part 3 of the weekly pipeline -- run after part 2 (run_pre_deadline.py): export the website's data, then
start the local site (website_plan.md).

    export site data (site/data/*.json, refuses stale part-2 outputs) -> local server (opens the browser)

Run locally (from anywhere -- paths are resolved relative to this file):
    python run_site_export.py               # export, then serve
    python run_site_export.py --no-serve    # export only
    python run_site_export.py --serve-only  # reopen the site without re-exporting (same as python run_site.py)
The server runs until Ctrl+C. No fetches, no model fits.
"""
import subprocess
import sys

from pipeline_stages import ROOT, SCRIPTS, run_pipeline

STAGES = [
    {
        "label": "Export website data",
        "script": SCRIPTS / "site_export" / "export_site_data.py",
        "summary_files": [],
    },
]

SERVER = ROOT / "site" / "server.py"


def serve(extra_args=()):
    """Run the server in the foreground with its output shown (a captured stage would hide it until Ctrl+C)."""
    try:
        return subprocess.call([sys.executable, "-u", "-X", "utf8", str(SERVER), *extra_args])
    except KeyboardInterrupt:
        return 0


def main(argv):
    passthrough = [a for a in argv if a in ("--no-browser",)]
    if "--serve-only" in argv:
        return serve(passthrough)
    code = run_pipeline("FPL Pipeline -- Part 3: website export", STAGES, argv)
    if code or "--no-serve" in argv:
        return code
    return serve(passthrough)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
