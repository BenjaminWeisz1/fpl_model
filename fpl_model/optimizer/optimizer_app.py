"""Local page for the transfer optimizer: set forced decisions, press Start Solve, see the three plans.

    python scripts/optimizer/optimizer_app.py

Serves app.html on 127.0.0.1 only and opens it in the browser. Start Solve runs run_optimizer.py (in the solve mode
picked beside the button; solve_modes.py) and replay_plans.py,
so the plans shown are the replay-checked ones. A new solve replaces that gameweek's folder. Ctrl+C stops the server.
"""
import json
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fpl_model.optimizer import replay_plans
from fpl_model.optimizer import run_optimizer
from fpl_model.optimizer import solve_modes

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_REQUEST_FILE = _ROOT / "data" / "optimizer" / "constraints_request.json"

PORT = 8765
SOLVE_TIMEOUT_SECONDS = 2 * max(solve_modes.solver_options(m)["secs"] * solve_modes.solver_options(m)["num_iterations"]
                                for m in solve_modes.MODES)
ERROR_LINES = 6

_solving = threading.Lock()


def page_state():
    """What the page needs to draw itself, or the reason a solve cannot start yet."""
    try:
        fpl, _, team, gws, projection = run_optimizer.load_state()
    except (AssertionError, ValueError, OSError, KeyError) as error:
        return {"error": str(error) or type(error).__name__}
    live = {e["id"]: e for e in fpl["elements"]}
    owned = {p["element"] for p in team["picks"]}
    totals = projection[[f"{w}_Pts" for w in gws]].sum(axis=1)
    players = [
        {"id": int(row["ID"]), "name": live[row["ID"]]["web_name"], "team": row["Team"], "pos": row["Pos"],
         "price": row["Value"], "xp": round(float(total), 1), "owned": row["ID"] in owned}
        for row, total in zip(projection.to_dict("records"), totals, strict=True)
    ]
    players.sort(key=lambda p: (not p["owned"], -p["xp"]))
    transfers = team["transfers"]
    modes = [{"name": m, "label": f"{m.title()}: {solve_modes.MEASURED[m]}" if m in solve_modes.MEASURED else f"{m.title()}: proven optimum"}
             for m in solve_modes.MODES]
    return {"gws": gws, "fts": max(0, transfers["limit"] - transfers["made"]), "bank": transfers["bank"] / 10, "players": players,
            "modes": modes, "default_mode": solve_modes.DEFAULT_MODE}


def last_lines(text):
    return "\n".join(text.strip().splitlines()[-ERROR_LINES:])


def run_step(script, *args):
    module = "fpl_model.optimizer." + Path(script).stem
    command = [sys.executable, "-W", "ignore", "-u", "-X", "utf8", "-m", module, *args]
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=_ROOT, timeout=SOLVE_TIMEOUT_SECONDS, check=False)


def solve(constraints):
    mode = constraints.pop("mode", solve_modes.DEFAULT_MODE)
    if mode not in solve_modes.MODES:
        return {"ok": False, "message": f"Unknown solve mode {mode}."}
    _REQUEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    _REQUEST_FILE.write_text(json.dumps(constraints), encoding="utf-8")
    solved = run_step("run_optimizer.py", "--constraints", str(_REQUEST_FILE), "--mode", mode)
    if solved.returncode == run_optimizer.INFEASIBLE_EXIT:
        return {"ok": False, "message": "No plan satisfies these decisions. Check for a planned move that clashes with a hold, the budget, the 3-per-club limit, or a player you do not own in that week."}
    if solved.returncode != 0:
        return {"ok": False, "message": last_lines(solved.stderr or solved.stdout)}
    replayed = run_step("replay_plans.py")
    if replayed.returncode != 0:
        return {"ok": False, "message": "The plans failed the replay check:\n" + last_lines(replayed.stderr or replayed.stdout)}
    return {"ok": True}


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, content_type="application/json", status=200):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send((_HERE / "app.html").read_bytes(), "text/html")
        elif path == "/api/state":
            self._send(json.dumps(page_state()))
        elif path == "/plans":
            self._send((replay_plans.latest_folder() / "plans.html").read_bytes(), "text/html")
        else:
            self._send("Not found", "text/plain", 404)

    def do_POST(self):
        # A JSON content type cannot be sent cross-site without a preflight, which this server refuses.
        if self.path != "/api/solve" or self.headers.get("Content-Type") != "application/json":
            self._send("Not found", "text/plain", 404)
            return
        constraints = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not _solving.acquire(blocking=False):
            self._send(json.dumps({"ok": False, "message": "A solve is already running."}))
            return
        try:
            self._send(json.dumps(solve(constraints)))
        except subprocess.TimeoutExpired:
            self._send(json.dumps({"ok": False, "message": "The solve timed out."}))
        finally:
            _solving.release()

    def log_message(self, *_args):
        pass


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Optimizer page: {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
