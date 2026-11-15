"""Local server for the personal website (website_plan.md): serves site/ on 127.0.0.1 and opens the browser.

    python site/server.py [--no-browser] [--port N]

Serves the static pages, plus the Solver page's API (§228), which runs solves through
scripts/optimizer/solver_jobs.py (one at a time, in the background):
    GET  /api/solver/state   weeks, modes, players for the pickers, the last result
    POST /api/solver/team    {"id": n} | {"json": text} | {"file": true} -> the squad, rebuilt or checked
    POST /api/solver/solve   squad, horizon, mode, chips, hold / avoid / moves -> starts the job
    GET  /api/solver/job     the job's status and progress (polled)
    POST /api/solver/cancel
    GET  /solver/plans       the last run's replay-checked plans.html
site/data/ is written by part 3 (run_site_export.py) and never edited by hand. Ctrl+C stops the server.
Run it by path: this folder is named "site", so it is not imported as a package (it would shadow Python's
own site module).
"""
import json
import sys
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # the repo root, so the fpl_model package imports
from fpl_model.optimizer import solver_jobs  # noqa: E402
from fpl_model.optimizer import team_source  # noqa: E402
PORT = 8766  # the optimizer page uses 8765


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
                      ".css": "text/css; charset=utf-8", ".html": "text/html; charset=utf-8"}

    def end_headers(self):
        # Each part 3 rewrites site/data; never let the browser show last week's numbers.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, value, status=200):
        data = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.endswith(".py"):
            self.send_error(404)
        elif path == "/api/solver/state":
            self._json(solver_jobs.page_state())
        elif path == "/api/solver/job":
            self._json(solver_jobs.JOB.status())
        elif path == "/solver/plans":
            plans = solver_jobs.FOLDER / "plans.html"
            if not plans.exists():
                self.send_error(404)
                return
            data = plans.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            super().do_GET()

    def do_POST(self):
        # A JSON content type can't be sent cross-site without a preflight, which this server refuses.
        if self.headers.get("Content-Type") != "application/json":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        path = self.path.split("?")[0]
        if path == "/api/solver/team":
            try:
                self._json({"ok": True, **solver_jobs.load_team(body)})
            except (team_source.TeamError, ValueError, OSError, KeyError) as error:
                self._json({"ok": False, "message": str(error) or type(error).__name__})
        elif path == "/api/solver/solve":
            problem = solver_jobs.JOB.start(body)
            self._json({"ok": problem is None, "message": problem})
        elif path == "/api/solver/cancel":
            problem = solver_jobs.JOB.cancel()
            self._json({"ok": problem is None, "message": problem})
        else:
            self.send_error(404)

    def log_message(self, *_args):
        pass


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else PORT
    if not (HERE / "data" / "projections.json").exists():
        print("site/data/projections.json is missing; run python run_site_export.py first")
        return 1
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(HERE)))
    url = f"http://127.0.0.1:{port}/"
    print(f"Website: {url}  (Ctrl+C to stop)", flush=True)
    if "--no-browser" not in argv:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
