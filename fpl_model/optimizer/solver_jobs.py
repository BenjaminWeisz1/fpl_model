"""Solve jobs for the website's Solver page (website_plan.md step 5, §228): one at a time, in the background.

A job writes the request's squad and forced decisions to data/optimizer/site/, runs run_optimizer.py (--team,
--horizon, --mode, --constraints, --out) and then replay_plans.py on that folder, so the plans the page shows are the
replay-checked ones. Each run replaces the folder's plans. The page polls status(); the solver's [progress] lines
(solve_modes.py) say which plan is being solved. A job is stopped at solve_modes.time_limit(mode), or by cancel().
"""
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fpl_model.optimizer import chips as chip_rules
from fpl_model.optimizer import run_optimizer
from fpl_model.optimizer import solve_modes
from fpl_model.optimizer import team_source
from fpl_model.forecast_horizon import HORIZON

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
FOLDER = _ROOT / "data" / "optimizer" / "site"
TEAM_FILE = FOLDER / "request_team.json"
CONSTRAINTS_FILE = FOLDER / "request_constraints.json"
KEEP_LINES = 40
STATE_CACHE_SECONDS = 300
# Files a run writes; cleared before each run so an older run's plan_3 never sits beside a one-plan run
RUN_FILES = ("plan_*.csv", "solver_summary.json", "plans.md", "plans.html", "verification.json", "constraints.json",
             "projection.csv", "bootstrap_static.json", "team.json")


def mode_info():
    return [{"name": m, "measured": solve_modes.MEASURED[m],
             "plans": solve_modes.solver_options(m)["num_iterations"], "limit_s": solve_modes.time_limit(m)}
            for m in solve_modes.MODES]


_cache = {"at": 0, "value": None}


def page_state():
    """Weeks, modes and the player list for the hold / avoid / move pickers (no squad needed)."""
    if _cache["value"] and time.time() - _cache["at"] < STATE_CACHE_SECONDS:
        return _cache["value"]
    try:
        fpl = run_optimizer.fetch(f"{run_optimizer.BASE_URL}/bootstrap-static/")
        gws = run_optimizer.plan_weeks(fpl)
        projection = run_optimizer.build_projection(fpl, gws)
    except (AssertionError, ValueError, OSError, KeyError) as error:
        return {"error": str(error) or type(error).__name__}
    live = {e["id"]: e for e in fpl["elements"]}
    totals = projection[[f"{w}_Pts" for w in gws]].sum(axis=1)
    players = [{"id": int(r["ID"]), "name": live[r["ID"]]["web_name"], "team": r["Team"], "pos": r["Pos"],
                "price": float(r["Value"]), "xp": round(float(t), 1)} for r, t in zip(projection.to_dict("records"), totals, strict=True)]
    players.sort(key=lambda p: -p["xp"])
    value = {"gws": gws, "horizon": HORIZON, "modes": mode_info(), "default_mode": solve_modes.DEFAULT_MODE,
             "chips": [{"key": c, "label": chip_rules.LABELS[c]} for c in chip_rules.CHIPS], "players": players,
             "last": last_result()}
    _cache.update(at=time.time(), value=value)
    return value


def load_team(request):
    """{"id": n} | {"json": text} | {"file": true} -> the squad, a summary for the page, and the chip weeks open."""
    fpl = run_optimizer.fetch(f"{run_optimizer.BASE_URL}/bootstrap-static/")
    gws = run_optimizer.plan_weeks(fpl)
    if request.get("id") not in (None, ""):
        if not str(request["id"]).isdigit():
            raise team_source.TeamError("An FPL team ID is a number (the one in your team's points-page URL).")
        team = team_source.team_from_id(int(request["id"]), run_optimizer.fetch)
        source = f"FPL team {int(request['id'])}, rebuilt from public data"
    elif request.get("json"):
        try:
            team = json.loads(request["json"])
        except json.JSONDecodeError as error:
            raise team_source.TeamError(f"Not valid JSON: {error}") from error
        team = team_source.check_shape(team)
        source = "pasted JSON"
    elif request.get("file"):
        team = json.loads(run_optimizer.TEAM_FILE.read_text(encoding="utf-8-sig"))
        run_optimizer.check_team_fresh(run_optimizer.team_saved_utc(), fpl["events"])
        source = f"team_data.json, saved {run_optimizer.team_saved_utc():%Y-%m-%d %H:%M} UTC"
    else:
        raise team_source.TeamError("Give a team ID, paste JSON, or use team_data.json.")
    projection = run_optimizer.build_projection(fpl, gws)
    try:
        run_optimizer.check_team(team, fpl, projection)
    except AssertionError as error:
        raise team_source.TeamError(f"The squad fails a check: {error}") from error
    live = {e["id"]: e for e in fpl["elements"]}
    clubs = {t["id"]: t["short_name"] for t in fpl["teams"]}
    xp = dict(zip(projection["ID"], projection[[f"{w}_Pts" for w in gws]].sum(axis=1), strict=True))
    squad = [{"id": p["element"], "name": live[p["element"]]["web_name"], "team": clubs[live[p["element"]]["team"]],
              "pos": "GDMF"[live[p["element"]]["element_type"] - 1], "sell": p["selling_price"] / 10,
              "now": live[p["element"]]["now_cost"] / 10, "xp": round(float(xp[p["element"]]), 1)} for p in team["picks"]]
    squad.sort(key=lambda p: ("GDMF".index(p["pos"]), -p["xp"]))
    transfers = team["transfers"]
    return {"team": team, "source": source, "squad": squad, "bank": transfers["bank"] / 10,
            "fts": max(0, transfers["limit"] - transfers["made"]), "made": transfers["made"],
            "chip_weeks": team_source.chip_weeks(team, fpl, gws)}


class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.state = {"status": "idle"}

    def status(self):
        with self.lock:
            state = dict(self.state)
        if state.get("started"):
            state["elapsed_s"] = round((state.get("finished") or time.time()) - state["started"])
        return state

    def _set(self, **values):
        with self.lock:
            self.state.update(values)

    def _line(self, line):
        with self.lock:
            lines = self.state.setdefault("lines", [])
            lines.append(line)
            del lines[:-KEEP_LINES]
            if line.startswith(solve_modes.PROGRESS):
                self.state["progress"] = line[len(solve_modes.PROGRESS):].strip()

    def start(self, request):
        """Validate the request and start the run; returns None, or the reason it can't start."""
        mode = request.get("mode", solve_modes.DEFAULT_MODE)
        if mode not in solve_modes.MODES:
            return f"Unknown solve mode {mode}."
        horizon = request.get("horizon", HORIZON)
        if not isinstance(horizon, int) or not 1 <= horizon <= HORIZON:
            return f"Horizon must be 1-{HORIZON}."
        team = request.get("team")
        if not team:
            return "Load a team first."
        constraints = {"hold": request.get("hold", []), "avoid": request.get("avoid", []),
                       "moves": request.get("moves", []), "chips": request.get("chips", {})}
        with self.lock:
            if self.state["status"] == "running":
                return "A solve is already running."
            self.state = {"status": "running", "mode": mode, "horizon": horizon, "started": time.time(),
                          "limit_s": solve_modes.time_limit(mode), "plans": solve_modes.solver_options(mode)["num_iterations"],
                          "progress": "fetching FPL data and building the model", "lines": []}
        FOLDER.mkdir(parents=True, exist_ok=True)
        for pattern in RUN_FILES:
            for path in FOLDER.glob(pattern):
                path.unlink()
        TEAM_FILE.write_text(json.dumps(team, indent=1), encoding="utf-8")
        CONSTRAINTS_FILE.write_text(json.dumps(constraints), encoding="utf-8")
        threading.Thread(target=self._run, args=(mode, horizon), daemon=True).start()
        return None

    def _step(self, args, deadline):
        command = [sys.executable, "-W", "ignore", "-u", "-X", "utf8", *args]
        self.process = subprocess.Popen(command, cwd=_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace")
        timer = threading.Timer(max(1.0, deadline - time.time()), self._kill, args=("time limit",))
        timer.start()
        try:
            for line in self.process.stdout:
                self._line(line.rstrip())
            return self.process.wait()
        finally:
            timer.cancel()

    def _run(self, mode, horizon):
        deadline = time.time() + solve_modes.time_limit(mode)
        try:
            code = self._step(["-m", "fpl_model.optimizer.run_optimizer", "--team", str(TEAM_FILE), "--horizon", str(horizon),
                               "--mode", mode, "--constraints", str(CONSTRAINTS_FILE), "--out", str(FOLDER)], deadline)
            if self.state.get("stopped"):
                return self._set(status="failed", message=f"Stopped ({self.state['stopped']}).", finished=time.time())
            if code == run_optimizer.INFEASIBLE_EXIT:
                return self._set(status="failed", finished=time.time(), message=(
                    "No plan satisfies these decisions. Check for a planned move that clashes with a hold, the budget, "
                    "the 3-per-club limit, or a player you don't own in that week."))
            if code != 0:
                return self._set(status="failed", finished=time.time(), message="The solve failed: " + self._tail())
            self._set(progress="checking the plans (replay)")
            code = self._step(["-m", "fpl_model.optimizer.replay_plans", str(FOLDER)], deadline)
            if code != 0:
                return self._set(status="failed", finished=time.time(), message="The plans failed the replay check:\n" + self._tail())
            self._set(status="done", finished=time.time(), progress="done", last=last_result())
            _cache["value"] = None
        except Exception as error:  # noqa: BLE001  (the page must hear about any failure)
            self._set(status="failed", finished=time.time(), message=f"{type(error).__name__}: {error}")
        finally:
            self.process = None

    def _tail(self, n=8):
        """A Python error's own message when the step ended with one (e.g. a chip week outside the horizon), else the
        last lines of output."""
        with self.lock:
            lines = [line for line in self.state.get("lines", []) if line.strip() and not line.startswith(solve_modes.PROGRESS)]
        error = re.match(r"^(?:\w+\.)*(?:ValueError|AssertionError|TeamError): (.*)$", lines[-1]) if lines else None
        return error.group(1) if error else "\n".join(lines[-n:])

    def _kill(self, why):
        process = self.process
        if process and process.poll() is None:
            self._set(stopped=why)
            # the whole tree: a process-per-plan mode has children
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)

    def cancel(self):
        if self.state.get("status") != "running":
            return "No solve is running."
        self._kill("cancelled")
        return None


def last_result():
    """The latest replay-checked run in FOLDER, or None."""
    summary, html = FOLDER / "solver_summary.json", FOLDER / "plans.html"
    if not (summary.exists() and html.exists()) or html.stat().st_mtime < summary.stat().st_mtime:
        return None
    run = json.loads(summary.read_text(encoding="utf-8"))
    projection_weeks = sorted(int(c.split("_")[0]) for c in (FOLDER / "projection.csv").read_text(encoding="utf-8").split("\n", 1)[0].split(",") if c.endswith("_Pts"))
    return {"mode": run.get("mode"), "gws": [projection_weeks[0], projection_weeks[-1]], "plans": len(run["plans"]),
            "solved_utc": datetime.fromtimestamp(html.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M")}


JOB = Job()
