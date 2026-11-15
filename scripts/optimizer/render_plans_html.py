"""Write plans.html: one column per plan, one row per gameweek, in the style of FPL Review's optimizer view.

Called by replay_plans.py, so the page only ever shows replayed numbers. Self-contained: no libraries, no network.
"""
from html import escape

PLAN_COLOURS = ["#22d39a", "#2aa8ff", "#b07cff"]
CHIP_LABELS = {"wc": "Wildcard", "fh": "Free Hit", "bb": "Bench Boost", "tc": "Triple Captain"}  # as chips.LABELS
POSITION_ORDER = [1, 2, 3, 4]

STYLE = """
:root { --bg: #0b1622; --panel: #0f1d2b; --line: #1d3142; --text: #d8e4ee; --muted: #7f95a8; --out: #ff5f7e; --in: #22d39a; --hit: #ffb547; }
* { box-sizing: border-box; }
body { margin: 0; padding: 6px; background: var(--bg); color: var(--text); font: 14px/1.3 "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 18px; margin: 0 0 2px; }
.sub { color: var(--muted); font-size: 13px; margin: 0 0 10px; }
table { border-collapse: collapse; width: 100%; table-layout: fixed; background: var(--panel); }
th, td { border: 1px solid var(--line); padding: 8px 2px; vertical-align: top; text-align: center; overflow-wrap: break-word; }
col.gw { width: 2.8em; }
thead th { font-weight: 400; }
.plan-name { font-size: 14px; font-weight: 700; }
.plan-score { font-size: 22px; font-weight: 700; line-height: 1.15; }
.plan-meta { color: var(--muted); font-size: 12px; }
th.gw { color: var(--in); vertical-align: middle; font-size: 12px; padding: 4px 0; white-space: nowrap; }
.status { color: var(--muted); font-size: 12px; }
.status .hit { color: var(--hit); font-weight: 700; }
.status .chip { color: var(--hit); font-weight: 700; }
.moves { margin: 8px 0 6px; font-size: clamp(16px, 1.9vw, 21px); font-weight: 600; }
.move + .move { margin-top: 10px; }
.out, .in { display: inline-block; }
.out { color: var(--out); }
.in { color: var(--in); }
.roll { display: inline-block; border: 1px solid var(--line); color: var(--muted); padding: 3px 6px; border-radius: 4px; font: 400 clamp(11px, 1.1vw, 14px) Consolas, "Cascadia Mono", Menlo, monospace; white-space: nowrap; }
details { margin-top: 4px; font-size: 13px; color: var(--muted); text-align: left; }
summary { cursor: pointer; text-align: center; }
details p { margin: 4px 0; }
details b { color: var(--text); font-weight: 400; }
"""


def _names(players, inputs, gw, captain=None, vice=None):
    parts = []
    for p in players:
        tag = " (C)" if p == captain else " (V)" if p == vice else ""
        parts.append(f"<b>{escape(inputs['names'][p])}{tag}</b> {inputs['points'][p, gw]:.1f}")
    return ", ".join(parts)


def _cell(week, inputs):
    names, gw = inputs["names"], week["gw"]
    hits = f' <span class="hit">-{week["hits"] * 4}</span>' if week["hits"] else ""
    status = f'{week["transfers"]}/{week["fts"]} FT{hits} &middot; &pound;{week["bank"] / 10:.1f}M'
    if week.get("chip"):
        status = f'<span class="chip">{escape(CHIP_LABELS[week["chip"]].upper())}</span> ' + status
    if week.get("chip") == "fh":
        moves = '<span class="roll">FREE HIT SQUAD</span>'
    elif week["out"]:
        moves = "".join(
            f'<div class="move"><span class="out">{escape(names[o])}</span> <span class="in">&raquo; {escape(names[i])}</span></div>'
            for o, i in zip(week["out"], week["in"], strict=True)
        )
    else:
        moves = '<span class="roll">ROLL TRANSFER</span>'
    rows = "".join(
        f"<p>{_names([p for p in week['lineup'] if inputs['live'][p]['element_type'] == t], inputs, gw, week['captain'], week['vice'])}</p>"
        for t in POSITION_ORDER
    )
    return (
        f'<td><div class="status">{status}</div><div class="moves">{moves}</div>'
        f'<details><summary>Team</summary><p>C <b>{escape(names[week["captain"]])}</b>, V <b>{escape(names[week["vice"]])}</b>, <b>{week["raw"]:.1f}</b> xP</p>'
        f'{rows}<p>Bench: {_names(week["bench"], inputs, gw) or "none (Bench Boost)"}</p></details></td>'
    )


def _constraints_line(constraints, inputs):
    names = inputs["names"]
    parts = []
    if constraints["hold"]:
        parts.append("Hold/target: " + ", ".join(names[p] for p in constraints["hold"]))
    if constraints["avoid"]:
        parts.append("Avoid/sell: " + ", ".join(names[p] for p in constraints["avoid"]))
    for m in constraints["moves"]:
        out, bought = (names[m[s]] if m[s] is not None else "any" for s in ("out", "in"))
        parts.append(f"GW{m['gw']}: {out} &raquo; {bought}")
    return escape(" | ".join(parts), quote=False).replace("&amp;raquo;", "&raquo;")


def write_html(path, plans, base, inputs, constraints, mode="deep", measured=None):
    gws = inputs["gws"]
    rows = sorted(plans)
    colours = {n: PLAN_COLOURS[i % len(PLAN_COLOURS)] for i, n in enumerate(rows)}

    heads = ""
    for n in rows:
        plan = plans[n]
        gain = plan["objective"] - base["objective"]
        heads += (
            f'<th><div style="color:{colours[n]}"><div class="plan-name">PLAN {n}</div><div class="plan-score">{plan["objective"]:.1f}</div></div>'
            f'<div class="plan-meta">{gain:+.1f} vs no moves</div></th>'
        )
    body = ""
    for i, gw in enumerate(gws):
        cells = "".join(_cell(plans[n]["weeks"][i], inputs) for n in rows)
        body += f'<tr><th class="gw">GW{gw}</th>{cells}</tr>'

    title = f"GW{gws[0]}-{gws[-1]} plans"
    forced = _constraints_line(constraints, inputs)
    subtitle = f"Forced: {forced}." if forced else f"No forced decisions. No transfers scores {base['objective']:.1f}."
    if mode != "deep":  # an approximate solve mode shows its measured cost (solve_modes.py, §227)
        subtitle += escape(f" Solve mode {mode}: an approximation. Measured against Deep: {measured or 'not recorded'}.")
    page = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title><style>{STYLE}</style></head><body>"
        f'<div class="wrap"><h1>{title}</h1><p class="sub">{subtitle}</p>'
        f'<table><colgroup><col class="gw">{"<col>" * len(rows)}</colgroup>'
        f"<thead><tr><th></th>{heads}</tr></thead><tbody>{body}</tbody></table></div></body></html>"
    )
    path.write_text(page, encoding="utf-8")
