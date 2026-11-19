"""Solve the multi-week transfer plan with the pinned open-fpl-solver (brief: optimizer_brief.md).

Builds the projection from data/current/forecasts/predicted_points.csv (points) and data/current/manual/minutes.csv (IDs and xMins), with live
bootstrap-static prices, then writes three plans to data/optimizer/GW<nn>/. Reads production data, never writes it.
The gameweek comes from bootstrap-static (is_next); free transfers and bank come from team_data.json. Each week: run
part 2, refresh team_data.json, then run this and replay_plans.py. The plans the replay reports are the checked ones.

    python -u -X utf8 -m fpl_model.optimizer.run_optimizer [--mode fast|standard|deep] [--full-pool] [--constraints file.json]
        [--team file.json] [--horizon N] [--out folder]

--mode picks the solve mode (solve_modes.py, §227): how long the solver searches and how close to the optimum it must
get. The default is solve_modes.DEFAULT_MODE.

--team takes the squad from another file in team_data.json's format (the website's Solver page, team_source.py);
--horizon solves fewer weeks than HORIZON; --out writes the plans to another folder. The squad used is copied to the
plan folder as team.json, which replay_plans.py checks against.

--full-pool skips upstream's EV-per-price filter (a robustness check); output goes to a full_pool/ subfolder.
--constraints takes forced decisions, usually from optimizer_app.py: {"hold": [ids], "avoid": [ids],
"moves": [{"gw": 11, "out": id or null, "in": id or null}], "chips": {"wc": gw or null, "fh": .., "bb": .., "tc": ..}}.
See apply_constraints; chips are fixed weeks, never picked by the solver (chips.py, §226). Exits 3 if no plan satisfies
them.
"""
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from fpl_model import paths

_HERE = Path(__file__).resolve().parent
_ROOT = paths.ROOT
_DATA = paths.CURRENT
_SOLVER = _HERE / "open_fpl_solver"
sys.path.insert(0, str(_SOLVER))  # vendored solver: imported as `dev.solver`, not part of the package

from dev import solver  # noqa: E402

from fpl_model.forecast_horizon import (  # noqa: E402  (the pipeline's forecast window and week weighting, set in one place)
    HORIZON,
    TIME_DECAY,
)
from fpl_model.optimizer import chips as chip_rules  # noqa: E402
from fpl_model.optimizer import solve_modes  # noqa: E402

BASE_URL = "https://fantasy.premierleague.com/api"
LAST_GW = 38
# Upstream's minimum total xMins for the player pool is 300 over its default 8-week horizon; the brief scales it
# to the horizon (190 over 5 weeks, §148). Kept at that rate per week: 375 over 10.
XMIN_LB_PER_GW = 37.5
NOT_IN_FPL_BV = 99.9
INFEASIBLE_EXIT = 3
NO_CONSTRAINTS = {"hold": [], "avoid": [], "moves": [], "chips": dict(chip_rules.NO_CHIPS)}

# Every setting the brief fixes; the rest stays at upstream's comprehensive_settings.json.
OVERRIDES = {
    "horizon": HORIZON,
    "objective": "decay",
    "decay_base": TIME_DECAY,
    "ft_value": 1.5,
    "ft_value_list": {},
    "weekly_hit_limit": None,
    "no_transfer_last_gws": 0,
    "ft_use_penalty": 0,
    "itb_value": 0,
    "xmin_lb": round(XMIN_LB_PER_GW * HORIZON),
    "chip_limits": {"bb": 0, "wc": 0, "fh": 0, "tc": 0},
    "iteration_criteria": "this_gw_transfer_in_out",
    "bench_weights": {"0": 0.03, "1": 0.21, "2": 0.06, "3": 0.002},
    "vcap_weight": 0.1,
    "verbose": False,
}  # gap, secs and num_iterations come from the solve mode (solve_modes.MODES)


def fetch(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def build_projection(fpl, gws):
    """predicted_points (Player, Team) -> players_current -> (Raw_Web_Name, Team_Short) -> bootstrap ID."""
    points = pd.read_csv(paths.current_file("predicted_points.csv", root=_DATA), encoding="utf-8-sig")
    assert not points.duplicated(["Player", "Team"]).any()
    weeks = [int(c) for c in points.columns if c.isdigit()]
    if weeks[0] != gws[0] or not set(gws) <= set(weeks):
        raise ValueError(f"predicted_points.csv covers GW{weeks}, the plan needs GW{gws}: run part 2 first.")

    current = pd.read_csv(paths.current_file("players_current.csv", root=_DATA), encoding="utf-8-sig")
    current = current[["Web_Name", "Raw_Web_Name", "Team_Short"]].drop_duplicates(["Web_Name", "Team_Short"])
    merged = points.merge(current, left_on=["Player", "Team"], right_on=["Web_Name", "Team_Short"], how="left", validate="1:1")

    teams = {t["id"]: t["short_name"] for t in fpl["teams"]}
    types = {1: "G", 2: "D", 3: "M", 4: "F"}
    bootstrap = pd.DataFrame(
        [
            {"ID": e["id"], "Raw_Web_Name": e["web_name"], "Team_Short": teams[e["team"]], "Live_Pos": types[e["element_type"]], "now_cost": e["now_cost"]}
            for e in fpl["elements"]
        ]
    )
    merged = merged.merge(bootstrap, on=["Raw_Web_Name", "Team_Short"], how="left", validate="1:1")
    assert merged["ID"].notna().all(), merged.loc[merged["ID"].isna(), "Player"].tolist()
    merged["ID"] = merged["ID"].astype(int)
    assert merged["ID"].is_unique and (merged["Pos"] == merged["Live_Pos"]).all()

    minutes = pd.read_csv(paths.current_file("minutes.csv", root=_DATA), encoding="utf-8-sig")
    minutes = minutes[minutes["BV"] != NOT_IN_FPL_BV]
    assert len(minutes) == len(points) and minutes["ID"].is_unique
    merged = merged.merge(minutes[["ID", "Team", *[f"{w}_xMins" for w in gws]]], on="ID", how="left", suffixes=("", "_minutes"), validate="1:1")
    assert (merged["Team"] == merged["Team_minutes"]).all()

    moved = merged[(merged["Price"] * 10).round().astype(int) != merged["now_cost"]]
    print(f"Prices: {len(moved)} players moved from predicted_points.csv; live prices used.")

    out = pd.DataFrame({"ID": merged["ID"], "Name": merged["Player"], "Pos": merged["Pos"], "Team": merged["Team"], "Value": merged["now_cost"] / 10})
    for w in gws:
        out[f"{w}_Pts"] = merged[str(w)]
        out[f"{w}_xMins"] = merged[f"{w}_xMins"]
    assert out.notna().all().all()
    return out


def check_team(team, fpl, projection):
    costs = {e["id"]: e["now_cost"] for e in fpl["elements"]}
    for pick in team["picks"]:
        cost, bought = costs[pick["element"]], pick["purchase_price"]
        expected = bought + (cost - bought) // 2 if cost > bought else cost
        assert expected == pick["selling_price"], f"{pick['element']}: rule {expected}, file {pick['selling_price']}; refresh team_data.json"
    assert {p["element"] for p in team["picks"]} <= set(projection["ID"])
    active = [c["name"] for c in team["chips"] if c.get("status_for_entry") == "active"]
    assert team["transfers"]["limit"] is not None and not active, f"Chip active {active}: fix it as a chip week instead."


TEAM_FILE = paths.ROOT / "team_data.json"


def team_saved_utc(team_file=TEAM_FILE):
    """When the team file was saved: its modified time. picks_last_updated is when the picks changed."""
    return datetime.fromtimestamp(Path(team_file).stat().st_mtime, UTC)


def check_team_fresh(saved, events, now=None):
    """Free transfers, bank and picks all change at a deadline, so a file from before the latest one is stale."""
    now = now or datetime.now(UTC)
    deadlines = [datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00")) for e in events]
    passed = [d for d in deadlines if d <= now]
    if passed and saved < max(passed):
        raise ValueError(f"team_data.json was saved {saved:%Y-%m-%d %H:%M} UTC, before the {max(passed):%Y-%m-%d %H:%M} "
                         f"UTC deadline; refresh it.")


def plan_weeks(fpl, horizon=HORIZON):
    if not 1 <= horizon <= HORIZON:
        raise ValueError(f"Horizon {horizon} is outside 1-{HORIZON}.")
    next_gw = next(e["id"] for e in fpl["events"] if e["is_next"])
    return list(range(next_gw, min(next_gw + horizon, LAST_GW + 1)))


def load_state(team_file=TEAM_FILE, horizon=HORIZON):
    fpl = fetch(f"{BASE_URL}/bootstrap-static/")
    fixtures = fetch(f"{BASE_URL}/fixtures/")
    gws = plan_weeks(fpl, horizon)
    with open(team_file, encoding="utf-8-sig") as f:
        team = json.load(f)
    check_team_fresh(team_saved_utc(team_file), fpl["events"])
    projection = build_projection(fpl, gws)
    check_team(team, fpl, projection)
    return fpl, fixtures, team, gws, projection


def validate_constraints(constraints, gws, projection, fpl=None, team=None):
    """Checks the forced decisions, normalising constraints["chips"] in place (fpl and team are needed for chips)."""
    constraints["chips"] = chip_rules.normalise(constraints.get("chips"))
    if any(constraints["chips"].values()):
        constraints["chips"] = chip_rules.validate(constraints["chips"], gws, fpl, team)
    free_hit = constraints["chips"]["fh"]
    known = set(projection["ID"])
    moves = constraints["moves"]
    ids = [*constraints["hold"], *constraints["avoid"], *[m[side] for m in moves for side in ("out", "in") if m[side] is not None]]
    if not set(ids) <= known:
        raise ValueError(f"Unknown player IDs: {sorted(set(ids) - known)}")
    if set(constraints["hold"]) & set(constraints["avoid"]):
        raise ValueError("A player cannot be both held and avoided.")
    for move in moves:
        if move["gw"] not in gws or (move["out"] is None and move["in"] is None):
            raise ValueError(f"Bad planned move: {move}")
        if move["gw"] == free_hit:
            raise ValueError(f"GW{free_hit} is the Free Hit week; its moves aren't transfers: {move}")


def apply_constraints(options, constraints, gws, owned):
    """Hold: owned stays every week, unowned is in the squad by the last GW. Avoid: owned is gone by the last GW, unowned is never bought."""
    options["locked"] = [p for p in constraints["hold"] if p in owned]
    options["locked_next_gw"] = [[p, gws[-1]] for p in constraints["hold"] if p not in owned]
    options["banned"] = [p for p in constraints["avoid"] if p not in owned]
    options["banned_next_gw"] = [[p, gws[-1]] for p in constraints["avoid"] if p in owned]
    options["booked_transfers"] = [{"gw": m["gw"], "transfer_out": m["out"], "transfer_in": m["in"]} for m in constraints["moves"]]
    options.update(chip_rules.solver_options(constraints.get("chips")))

    # The plans must differ in a week still open to choice, or a planned move in the first GW forces pointless extra moves.
    booked = {m["gw"] for m in constraints["moves"]}
    if gws[0] in booked:
        options["iteration_criteria"] = "target_gws_transfer_in"
        options["iteration_target"] = [next((w for w in gws if w not in booked), gws[-1])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-pool", action="store_true")
    parser.add_argument("--constraints", type=Path)
    parser.add_argument("--mode", choices=list(solve_modes.MODES), default=solve_modes.DEFAULT_MODE)
    parser.add_argument("--team", type=Path, default=TEAM_FILE)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    fpl, fixtures, team, gws, projection = load_state(args.team, args.horizon)
    constraints = json.loads(args.constraints.read_text(encoding="utf-8")) if args.constraints else NO_CONSTRAINTS
    constraints = {**NO_CONSTRAINTS, **constraints}
    validate_constraints(constraints, gws, projection, fpl, team)
    print(f"Chips: {chip_rules.describe(constraints['chips'])}")

    out_dir = args.out or _ROOT / "data" / "optimizer" / f"GW{gws[0]:02d}"
    if args.full_pool:
        out_dir = out_dir / "full_pool"
    out_dir.mkdir(parents=True, exist_ok=True)

    transfers = team["transfers"]
    print(f"GW{gws[0]}-{gws[-1]}: {max(0, transfers['limit'] - transfers['made'])} free transfers, bank {transfers['bank'] / 10:.1f}")
    projection.to_csv(out_dir / "projection.csv", index=False, encoding="utf-8")
    (out_dir / "bootstrap_static.json").write_text(json.dumps(fpl), encoding="utf-8")
    (out_dir / "team.json").write_text(json.dumps(team, indent=1), encoding="utf-8")

    with open(_SOLVER / "comprehensive_settings.json") as f:
        options = {**json.load(f), **OVERRIDES, "horizon": len(gws), "xmin_lb": round(XMIN_LB_PER_GW * len(gws))}
    options.update(solve_modes.solver_options(args.mode))
    solve_modes.install(solver, args.mode)
    print(f"Solve mode: {args.mode}" + (f" (measured: {solve_modes.MEASURED[args.mode]})" if args.mode in solve_modes.MEASURED else ""))
    if args.full_pool:
        options["ev_per_price_cutoff"] = 0
    apply_constraints(options, constraints, gws, {p["element"] for p in team["picks"]})

    # Upstream's lookups go through these two; feed them the snapshot just saved so the solve and the replay see one state.
    responses = {f"{BASE_URL}/bootstrap-static/": fpl, f"{BASE_URL}/fixtures/": fixtures}
    solver.cached_request = responses.__getitem__
    solver.read_data = lambda _options: projection.copy()

    data = solver.prep_data(team, options)
    results = solver.solve_multi_period_fpl(data, options)
    if not results:
        print("No plan satisfies these constraints.")
        sys.exit(INFEASIBLE_EXIT)

    (out_dir / "constraints.json").write_text(json.dumps(constraints, indent=2), encoding="utf-8")
    summary = []
    for rank, result in enumerate(sorted(results, key=lambda r: -r["score"]), start=1):
        result["picks"].to_csv(out_dir / f"plan_{rank}.csv", index=False, encoding="utf-8")
        row = {k: result[k] for k in ("score", "solve_status", "mip_gap", "ft_end", "buy", "sell", "chip")}
        summary.append({"plan": rank, **row})
        print(row)
    run = {"mode": args.mode, "measured": solve_modes.MEASURED.get(args.mode), "pool_size": len(data["merged_data"]), "team_file": str(args.team), "team_data_saved_utc": team_saved_utc(args.team).isoformat(),
           "settings": options, "plans": summary}
    (out_dir / "solver_summary.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
