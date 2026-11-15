"""Check the solver's plans without the solver, score the no-transfer baseline and write plans.md.

Replays each plan_<n>.csv against the rules in optimizer_brief.md, recomputing squad legality, money (in integer
tenths), free transfers, hits, raw points and the decayed objective from team_data.json and predicted_points.csv,
and that any forced decisions in the folder's constraints.json were honoured. Any broken rule stops the script. The numbers in plans.md are this replay's, not the solver's.

Chips (§226, rules in optimizer_brief.md) are replayed from the FPL rules, not from the solver's model: Wildcard (free,
permanent moves), Free Hit (a separate legal squad within bank + selling value; the real squad returns the next week),
Bench Boost (all 15 score, no bench weights), Triple Captain (captain x3). After a Wildcard or Free Hit the free
transfers carry over unchanged. The chip weeks must be exactly the ones fixed in constraints.json.

    python -u -X utf8 scripts/optimizer/replay_plans.py [plan_folder]

With no folder it uses the GW folder solved most recently. The gameweeks come from that folder's projection.csv.
"""
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from render_plans_html import write_html

import chips as chip_rules

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
from forecast_horizon import TIME_DECAY as DECAY  # noqa: E402  (the solver's decay_base, declared once)
FT_VALUE = 1.5
HIT_COST = 4
VICE_WEIGHT = 0.1
BENCH_WEIGHTS = [0.03, 0.21, 0.06, 0.002]
MAX_FT = 5
MAX_PER_CLUB = 3
TOLERANCE = 1e-6


def weight(gw, first_gw):
    return DECAY ** (gw - first_gw)


def latest_folder():
    summaries = (_ROOT / "data" / "optimizer").glob("GW*/solver_summary.json")
    return max(summaries, key=lambda path: path.stat().st_mtime).parent


def load_inputs(folder, team_file=None, check_source=True):
    """team_file: the squad to replay from (default the folder's team.json, written by run_optimizer.py; folders
    solved before §228 have none and use the repo's team_data.json). check_source: the projection must equal
    predicted_points.csv (off for benchmark instances built from older forecasts)."""
    fpl = json.loads((folder / "bootstrap_static.json").read_text(encoding="utf-8"))
    team_file = team_file or (folder / "team.json" if (folder / "team.json").exists() else _ROOT / "team_data.json")
    team = json.loads(team_file.read_text(encoding="utf-8-sig"))
    live = {e["id"]: e for e in fpl["elements"]}
    club = {t["id"]: t["short_name"] for t in fpl["teams"]}
    types = {t["id"]: t for t in fpl["element_types"]}

    projection = pd.read_csv(folder / "projection.csv", encoding="utf-8")
    gws = sorted(int(c.split("_")[0]) for c in projection.columns if c.endswith("_Pts"))
    if check_source:
        source = pd.read_csv(_ROOT / "data" / "current" / "predicted_points.csv", encoding="utf-8-sig")
        source = source.set_index(["Player", "Team"])
    points = {}
    for values in projection.to_dict("records"):
        for gw in gws:
            if check_source:
                assert values[f"{gw}_Pts"] == source.loc[(values["Name"], values["Team"]), str(gw)], values["Name"]
            points[values["ID"], gw] = values[f"{gw}_Pts"]
    names = {pid: live[pid]["web_name"] for pid in projection["ID"]}
    return {"gws": gws, "live": live, "club": club, "types": types, "team": team, "points": points, "names": names}


def opening_state(inputs):
    held = {}
    for pick in inputs["team"]["picks"]:
        cost, bought = inputs["live"][pick["element"]]["now_cost"], pick["purchase_price"]
        held[pick["element"]] = bought + (cost - bought) // 2 if cost > bought else cost
        assert held[pick["element"]] == pick["selling_price"]
    transfers = inputs["team"]["transfers"]
    return held, transfers["bank"], max(0, transfers["limit"] - transfers["made"])


def check_squad(squad, inputs):
    live = inputs["live"]
    assert len(squad) == 15
    counts = Counter(live[p]["element_type"] for p in squad)
    for type_id, spec in inputs["types"].items():
        assert counts[type_id] == spec["squad_select"], counts
    assert max(Counter(live[p]["team"] for p in squad).values()) <= MAX_PER_CLUB


def check_selection(lineup, bench, captain, vice, squad, inputs, bench_boost=False):
    """A legal 11 and ordered bench of 4 with the keeper first; on a Bench Boost all 15 play and there is no bench."""
    live = inputs["live"]
    assert set(lineup) | set(bench) == set(squad) and not set(lineup) & set(bench)
    assert captain in lineup and vice in lineup and captain != vice
    if bench_boost:
        assert len(lineup) == 15 and not bench
        return
    assert len(lineup) == 11 and len(bench) == 4
    counts = Counter(live[p]["element_type"] for p in lineup)
    for type_id, spec in inputs["types"].items():
        assert spec["squad_min_play"] <= counts[type_id] <= spec["squad_max_play"], counts
    assert live[bench[0]]["element_type"] == 1


def score_week(gw, lineup, bench, captain, vice, inputs, triple=False):
    """(points, tie-breakers): the lineup plus the captain once more (twice on a Triple Captain); 0.1 x vice and the
    bench weights are tie-breakers, not points."""
    pts = lambda p: inputs["points"][p, gw]  # noqa: E731
    lineup_xp = sum(pts(p) for p in lineup)
    # a Bench Boost week has no bench (all 15 in the lineup), so no bench tie-breakers, as in the solver
    extras = VICE_WEIGHT * pts(vice) + (sum(w * pts(p) for w, p in zip(BENCH_WEIGHTS, bench, strict=True)) if bench else 0)
    return lineup_xp + (2 if triple else 1) * pts(captain), extras


def next_fts(fts, transfers):
    return min(MAX_FT, max(1, fts - transfers + 1))


def week_chip(week):
    """The chip the solver's picks show for a week (wc/fh/bb/tc or None). WC/FH/BB mark every row, TC the captain."""
    text = set(week["chip"].dropna().astype(str)) - {""}
    assert len(text) <= 1, text
    by_text = {v: k for k, v in chip_rules.SOLVER_TEXT.items()}
    return by_text[text.pop()] if text else None


def opening_fts(inputs, chips):
    """FTs in the first week; a Wildcard there makes the week's transfers already made free (FT count = the limit)."""
    transfers = inputs["team"]["transfers"]
    if chip_rules.normalise(chips)["wc"] == inputs["gws"][0]:
        return transfers["limit"]
    return max(0, transfers["limit"] - transfers["made"])


def free_hit_squad(week, held, bank, inputs):
    """The Free Hit week's squad, checked against FPL's budget: bank + selling value of the real squad. A current
    player kept costs his selling price, any other player his current price."""
    squad = sorted(int(p) for p in week.loc[week["squad"] == 1, "id"])
    check_squad(squad, inputs)
    cost = sum(held[p] if p in held else inputs["live"][p]["now_cost"] for p in squad)
    assert cost <= bank + sum(held.values()), f"Free Hit squad costs {cost}, budget {bank + sum(held.values())}"
    return squad


def replay_plan(folder, n, inputs, chips=None):
    picks = pd.read_csv(folder / f"plan_{n}.csv", encoding="utf-8")
    held, bank, _ = opening_state(inputs)
    fts = opening_fts(inputs, chips)
    expected = chip_rules.fixed(chips)
    live = inputs["live"]
    first = inputs["gws"][0]
    weeks = []
    objective = 0.0
    for gw in inputs["gws"]:
        week = picks[picks["week"] == gw]
        chip = week_chip(week)
        assert chip == expected.get(gw), f"GW{gw}: plan plays {chip}, fixed {expected.get(gw)}"
        bought = [int(p) for p in week.loc[week["transfer_in"] == 1, "id"]]
        sold = [int(p) for p in week.loc[week["transfer_out"] == 1, "id"]]
        assert len(bought) == len(sold) and not set(bought) & set(sold)
        assert set(sold) <= set(held) and not set(bought) & set(held)
        if chip == "fh":
            assert not bought, f"GW{gw}: Free Hit week with permanent transfers"
        bank_before = bank
        for p in sold:
            bank += held.pop(p)
        for p in bought:
            bank -= live[p]["now_cost"]
            held[p] = live[p]["now_cost"]  # a rebought player sells at the buy price
        assert bank >= 0, f"GW{gw} bank {bank}"
        check_squad(held, inputs)
        playing = free_hit_squad(week, held, bank, inputs) if chip == "fh" else sorted(held)

        lineup = [int(p) for p in week.loc[week["lineup"] == 1, "id"]]
        benched = week[week["bench"] >= 0].sort_values("bench")
        assert benched["bench"].tolist() == ([] if chip == "bb" else [0, 1, 2, 3])
        bench = [int(p) for p in benched["id"]]
        captain = int(week.loc[week["captain"] == 1, "id"].item())
        vice = int(week.loc[week["vicecaptain"] == 1, "id"].item())
        assert set(week.loc[week["squad"] == 1, "id"]) == set(playing)
        check_selection(lineup, bench, captain, vice, playing, inputs, bench_boost=chip == "bb")

        if chip in ("wc", "fh"):  # unlimited free moves; the FT count carries over unchanged (optimizer_brief.md)
            hits, after = 0, fts
        else:
            hits, after = max(0, len(bought) - fts), next_fts(fts, len(bought))
        ft_credit = FT_VALUE * (after - fts)
        points, extras = score_week(gw, lineup, bench, captain, vice, inputs, triple=chip == "tc")
        raw = points - HIT_COST * hits
        objective += weight(gw, first) * (raw + extras) + weight(gw + 1, first) * ft_credit
        weeks.append(
            {"gw": gw, "chip": chip, "fts": fts, "transfers": len(bought), "hits": hits, "bank_before": bank_before, "bank": bank,
             "in": bought, "out": sold, "squad": sorted(held), "playing": playing, "lineup": lineup, "bench": bench,
             "captain": captain, "vice": vice, "raw": raw, "decayed": weight(gw, first) * raw, "extras": extras,
             "ft_credit": ft_credit}
        )
        fts = after
    return {"weeks": weeks, "objective": objective, "ft_end": fts}


def check_constraints(plan, constraints, inputs):
    owned = set(opening_state(inputs)[0])
    weeks = plan["weeks"]
    # On a Free Hit week the held player must be in the Free Hit squad ("playing"), as the solver's locked rule.
    for p in constraints["hold"]:
        assert all(p in w["playing"] for w in weeks) if p in owned else p in weeks[-1]["playing"], f"Hold broken: {p}"
    for p in constraints["avoid"]:
        assert p not in weeks[-1]["playing"] if p in owned else not any(p in w["playing"] for w in weeks), f"Avoid broken: {p}"
    by_gw = {w["gw"]: w for w in weeks}
    for move in constraints["moves"]:
        week = by_gw[move["gw"]]
        assert move["out"] is None or move["out"] in week["out"], f"Planned move broken: {move}"
        assert move["in"] is None or move["in"] in week["in"], f"Planned move broken: {move}"


def best_selection(gw, squad, inputs):
    """Best legal XI by the brief's weekly value, by enumeration. Bench credits are small but count."""
    live, pts = inputs["live"], lambda p: inputs["points"][p, gw]
    keepers = [p for p in squad if live[p]["element_type"] == 1]
    outfield = [p for p in squad if live[p]["element_type"] != 1]
    best = None
    for keeper in keepers:
        spare = next(p for p in keepers if p != keeper)
        for chosen in itertools.combinations(outfield, 10):
            counts = Counter(live[p]["element_type"] for p in chosen)
            if not all(inputs["types"][t]["squad_min_play"] <= counts[t] <= inputs["types"][t]["squad_max_play"] for t in (2, 3, 4)):
                continue
            lineup = [keeper, *chosen]
            ranked = sorted(lineup, key=pts, reverse=True)
            subs = sorted((p for p in outfield if p not in chosen), key=pts, reverse=True)
            bench = [spare, *subs]
            points, extras = score_week(gw, lineup, bench, ranked[0], ranked[1], inputs)
            if best is None or points + extras > best[0] + best[1]:
                best = (points, extras, lineup, bench, ranked[0], ranked[1])
    return best


def baseline(inputs):
    held, _, fts = opening_state(inputs)
    first = inputs["gws"][0]
    weeks, objective = [], 0.0
    for gw in inputs["gws"]:
        points, extras, _, _, captain, _ = best_selection(gw, held, inputs)
        after = next_fts(fts, 0)
        credit = FT_VALUE * (after - fts)
        objective += weight(gw, first) * (points + extras) + weight(gw + 1, first) * credit
        weeks.append({"gw": gw, "raw": points, "decayed": weight(gw, first) * points, "captain": captain, "fts": fts})
        fts = after
    return {"weeks": weeks, "objective": objective}


def label(p, gw, inputs):
    e = inputs["live"][p]
    return f"{e['web_name']} ({inputs['club'][e['team']]} {inputs['points'][p, gw]:.2f})"


def render_plan(n, plan, base, inputs):
    names = inputs["names"]
    first = inputs["gws"][0]
    raw = sum(w["raw"] for w in plan["weeks"])
    decayed = sum(w["decayed"] for w in plan["weeks"])
    credit = sum(weight(w["gw"] + 1, first) * w["ft_credit"] for w in plan["weeks"])
    extras = sum(weight(w["gw"], first) * w["extras"] for w in plan["weeks"])
    lines = [
        f"## Plan {n}", "",
        f"Objective **{plan['objective']:.2f}** ({plan['objective'] - base['objective']:+.2f} vs no transfers). "
        f"Raw {raw:.2f}, decayed {decayed:.2f}, bench and vice tie-breakers {extras:+.2f}, "
        f"FT credit {credit:+.2f} (not points scored).", "",
        "| GW | Chip | FT | Moves | Hits | Bank | Raw xP after hits | Decayed |", "|---|---|---|---|---|---|---|---|",
    ]
    for w in plan["weeks"]:
        moves = ", ".join(f"{names[o]} → {names[i]}" for o, i in zip(w["out"], w["in"], strict=True)) or "Roll"
        if w["chip"] == "fh":
            moves = "Free Hit squad (real squad returns next week)"
        chip = chip_rules.LABELS[w["chip"]] if w["chip"] else ""
        lines.append(f"| {w['gw']} | {chip} | {w['fts']} | {moves} | {w['hits']} | {w['bank'] / 10:.1f} | {w['raw']:.2f} | {w['decayed']:.2f} |")
    lines.append(f"| Total | | | | {sum(w['hits'] for w in plan['weeks'])} | | {raw:.2f} | {decayed:.2f} |")
    lines.append("")
    for w in plan["weeks"]:
        gw = w["gw"]
        lines += [
            f"**GW{gw}.** Captain {names[w['captain']]}, vice {names[w['vice']]}.",
            "- XI: " + ", ".join(label(p, gw, inputs) for p in w["lineup"]),
            "- Bench: " + ", ".join(f"{i}. {label(p, gw, inputs)}" for i, p in enumerate(w["bench"])),
            "",
        ]
    return lines


def solver_constant(inputs, chips):
    """Upstream's score carries 1.5 x (opening FTs + 1); see PINNED.md."""
    return FT_VALUE * (opening_fts(inputs, chips) + 1)


def check_against_solver(plan, summary, inputs, chips, require_optimal=True):
    """The replayed objective must equal the solver's, and its end-of-horizon FTs too."""
    difference = plan["objective"] - (summary["score"] - solver_constant(inputs, chips))
    assert abs(difference) < TOLERANCE, f"Plan {summary['plan']}: replay and solver differ by {difference}"
    assert plan["ft_end"] == round(summary["ft_end"])
    if require_optimal:
        assert summary["solve_status"] == "Optimal", summary["solve_status"]


def read_constraints(folder):
    constraints_file = folder / "constraints.json"
    constraints = json.loads(constraints_file.read_text(encoding="utf-8")) if constraints_file.exists() else {}
    return {"hold": [], "avoid": [], "moves": [], **constraints, "chips": chip_rules.normalise(constraints.get("chips"))}


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_folder()
    inputs = load_inputs(folder)
    run = json.loads((folder / "solver_summary.json").read_text(encoding="utf-8"))
    solver_plans = run["plans"]
    # Deep must prove each optimum; the approximate modes (solve_modes.py, §227) may stop at their time limit with a
    # feasible plan, which the replay still checks in full
    mode = run.get("mode", "deep")
    base = baseline(inputs)
    constraints = read_constraints(folder)
    chips = constraints["chips"]
    print(f"No-transfer baseline: objective {base['objective']:.4f}, raw {sum(w['raw'] for w in base['weeks']):.2f}, "
          f"decayed {sum(w['decayed'] for w in base['weeks']):.2f}")

    plans = {}
    for summary in solver_plans:
        n = summary["plan"]
        plans[n] = replay_plan(folder, n, inputs, chips)
        check_constraints(plans[n], constraints, inputs)
        check_against_solver(plans[n], summary, inputs, chips, require_optimal=mode == "deep")
        print(f"Plan {n}: replay objective {plans[n]['objective']:.4f} = solver {summary['score'] - solver_constant(inputs, chips):.4f}; "
              f"gap {summary['mip_gap']}")
    moves = [tuple((frozenset(w["in"]), frozenset(w["out"])) for w in p["weeks"]) for p in plans.values()]
    if chips["fh"] != inputs["gws"][0]:  # a first-week Free Hit makes the plans differ in that week's lineup instead
        assert len(set(moves)) == len(moves), "Two plans make identical moves"
    assert all(a["objective"] >= b["objective"] for a, b in itertools.pairwise(plans.values()))

    lines = [f"# GW{inputs['gws'][0]}-{inputs['gws'][-1]} plans", "", "All figures come from `replay_plans.py`. Points are from `predicted_points.csv`; prices are fixed at the saved live snapshot.", ""]
    lines += [f"Chips: {chip_rules.describe(chips)}.", ""]
    lines += [f"Solve mode: {mode}" + ("." if mode == "deep" else f", an approximation; measured cost against Deep: {run.get('measured', 'not recorded')}."), ""]
    lines += [
        f"No-transfer baseline (no chips): objective {base['objective']:.2f}, raw {sum(w['raw'] for w in base['weeks']):.2f}, "
        f"decayed {sum(w['decayed'] for w in base['weeks']):.2f}. "
        "Captains: " + ", ".join(f"GW{w['gw']} {inputs['names'][w['captain']]}" for w in base["weeks"]) + ".", "",
    ]
    for n, plan in plans.items():
        lines += render_plan(n, plan, base, inputs)
    (folder / "plans.md").write_text("\n".join(lines), encoding="utf-8")
    (folder / "verification.json").write_text(json.dumps({"baseline": base, "plans": plans}, indent=2), encoding="utf-8")
    write_html(folder / "plans.html", plans, base, inputs, constraints, mode, run.get("measured"))
    print(f"Wrote {folder / 'plans.html'}")


if __name__ == "__main__":
    main()
