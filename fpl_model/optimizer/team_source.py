"""The squad the solver starts from, in team_data.json's format (FPL's /my-team/ endpoint), from one of three sources:
an FPL team ID (public endpoints only), pasted JSON, or the repo's team_data.json. Used by the website's Solver page
(website_plan.md step 5, §228).

From an ID, the squad is rebuilt as upstream's generate_team_json does (open_fpl_solver/dev/solver.py): the first
gameweek's picks at start prices, then every transfer outside Free Hit weeks; free transfers follow
optimizer_brief.md (+1 a week, capped at 5, carried unchanged after a Wildcard or Free Hit week). Differences from
upstream, each because upstream's version gives a wrong team in that case:
- transfers already made for the next gameweek count as `made` (upstream fails on them);
- chips are the instances in bootstrap-static, marked played from the entry's history, so chips.validate can read them;
- the result is checked against what FPL itself reports: the bank and the 15 players after the latest finished
  gameweek, and the bank at the first gameweek (which catches a squad picked after GW1 prices moved). A mismatch is an
  error that says to paste the JSON instead.
Not visible publicly: extra free transfers FPL hands out to everyone, and a chip activated for the next gameweek.
"""
from fpl_model.optimizer.chips import CHIPS, FPL_NAMES

BASE_URL = "https://fantasy.premierleague.com/api"
START_BUDGET = 1000  # £100.0m in tenths
MAX_FT = 5
SQUAD_SIZE = 15


class TeamError(ValueError):
    """The team can't be built or doesn't pass a check; the message says what to do."""


def chip_instances(static, played):
    """team_data.json-style chip list: each bootstrap-static instance, with status from the entry's played chips
    ({name: [events]})."""
    out = []
    for c in static["chips"]:
        events = [e for e in played.get(c["name"], []) if c["start_event"] <= e <= c["stop_event"]]
        out.append({"name": c["name"], "number": c.get("number"), "start_event": c["start_event"], "stop_event": c["stop_event"],
                    "status_for_entry": "played" if events else "available", "played_by_entry": events})
    return out


HIT_COST = 4


def free_transfers(counts, first_gw, next_gw, chip_weeks, charged=None):
    """FTs for next_gw. counts: {gw: transfers made in that gw}; chip_weeks: Wildcard and Free Hit weeks.
    charged ({gw: points FPL deducted for hits}, from the entry's history) checks each week: the hits the rebuilt FT
    count implies must be the ones FPL charged, or FPL gave free transfers this rule doesn't know about."""
    fts = 1  # the week after the first gameweek starts with one
    for gw in range(first_gw + 1, next_gw):
        if gw in chip_weeks:
            continue
        if charged is not None and charged.get(gw, 0) != HIT_COST * max(0, counts.get(gw, 0) - fts):
            raise TeamError(f"GW{gw}: {counts.get(gw, 0)} transfers with {fts} free transfers rebuilt should cost "
                            f"{HIT_COST * max(0, counts.get(gw, 0) - fts)} points, but FPL charged {charged.get(gw, 0)}; "
                            f"FPL may have given extra free transfers. Paste the JSON instead.")
        fts = min(max(fts - counts.get(gw, 0), 0) + 1, MAX_FT)
    return fts


def selling_price(purchase, now):
    return purchase + (now - purchase) // 2 if now > purchase else now


def team_from_id(team_id, fetch):
    """fetch(url) -> parsed JSON. Returns the team dict; raises TeamError when FPL's own figures disagree."""
    static = fetch(f"{BASE_URL}/bootstrap-static/")
    elements = {e["id"]: e for e in static["elements"]}
    next_gw = next((e["id"] for e in static["events"] if e["is_next"]), None)
    if next_gw is None:
        raise TeamError("FPL shows no next gameweek (the season is over).")
    try:
        history = fetch(f"{BASE_URL}/entry/{team_id}/history/")
    except Exception as error:
        raise TeamError(f"FPL has no team {team_id} ({error}).") from error
    if not history.get("current"):
        raise TeamError(f"Team {team_id} has not played a gameweek yet; paste its JSON instead.")
    transfers = sorted(fetch(f"{BASE_URL}/entry/{team_id}/transfers/"), key=lambda t: t["time"])
    played = {}
    for c in history["chips"]:
        played.setdefault(c["name"], []).append(c["event"])
    fh_weeks = set(played.get(FPL_NAMES["fh"], []))
    chip_weeks = fh_weeks | set(played.get(FPL_NAMES["wc"], []))

    first_gw = history["current"][0]["event"]
    first = fetch(f"{BASE_URL}/entry/{team_id}/event/{first_gw}/picks/")
    squad = {p["element"]: elements[p["element"]]["now_cost"] - elements[p["element"]]["cost_change_start"] for p in first["picks"]}
    bank = START_BUDGET - sum(squad.values())
    if first["entry_history"]["bank"] != bank:
        raise TeamError(f"The first gameweek's (GW{first_gw}) bank is £{first['entry_history']['bank'] / 10:.1f}m, but the "
                        f"start prices give £{bank / 10:.1f}m: prices had moved before this squad was picked. Paste the JSON instead.")

    counts, pending = {}, 0
    last_gw = history["current"][-1]["event"]
    for t in transfers:
        if t["event"] in fh_weeks:
            continue
        if t["event"] == next_gw:
            pending += 1
        elif t["event"] > last_gw:
            raise TeamError(f"A transfer for GW{t['event']} is after the next gameweek GW{next_gw}.")
        counts[t["event"]] = counts.get(t["event"], 0) + 1
        bank += t["element_out_cost"] - t["element_in_cost"]
        squad.pop(t["element_out"])
        squad[t["element_in"]] = t["element_in_cost"]
    pending_cost = sum(t["element_out_cost"] - t["element_in_cost"] for t in transfers if t["event"] == next_gw)

    # FPL's own figures after the latest finished gameweek (a Free Hit week shows the temporary squad: skip that check)
    if last_gw not in fh_weeks:
        latest = fetch(f"{BASE_URL}/entry/{team_id}/event/{last_gw}/picks/")
        held = set(squad)  # undo the next gameweek's transfers to compare with the latest picks
        for t in reversed([t for t in transfers if t["event"] == next_gw]):
            held.discard(t["element_in"])
            held.add(t["element_out"])
        if held != {p["element"] for p in latest["picks"]}:
            raise TeamError(f"Rebuilt squad differs from FPL's GW{last_gw} picks; paste the JSON instead.")
        if bank - pending_cost != latest["entry_history"]["bank"]:
            raise TeamError(f"Rebuilt bank £{(bank - pending_cost) / 10:.1f}m differs from FPL's GW{last_gw} bank "
                            f"£{latest['entry_history']['bank'] / 10:.1f}m; paste the JSON instead.")
    if len(squad) != SQUAD_SIZE:
        raise TeamError(f"Rebuilt squad has {len(squad)} players.")

    picks = [{"element": pid, "purchase_price": bought, "selling_price": selling_price(bought, elements[pid]["now_cost"]),
              "element_type": elements[pid]["element_type"]} for pid, bought in squad.items()]
    picks.sort(key=lambda p: (p["element_type"], -p["selling_price"]))
    return {"picks": picks, "chips": chip_instances(static, played),
            "transfers": {"bank": bank, "limit": free_transfers(counts, first_gw, next_gw, chip_weeks,
                                                                  {h["event"]: h["event_transfers_cost"] for h in history["current"]}), "made": pending},
            "team_id": int(team_id)}


def check_shape(team):
    """Pasted JSON: the fields the solver and the replay read must be there."""
    if not isinstance(team, dict) or not isinstance(team.get("picks"), list) or not isinstance(team.get("transfers"), dict):
        raise TeamError("The JSON needs 'picks' (a list) and 'transfers' (bank, limit, made), as in team_data.json.")
    if len(team["picks"]) != SQUAD_SIZE:
        raise TeamError(f"The JSON has {len(team['picks'])} picks; a squad has {SQUAD_SIZE}.")
    for key in ("element", "purchase_price", "selling_price"):
        if not all(isinstance(p.get(key), int) for p in team["picks"]):
            raise TeamError(f"Every pick needs an integer '{key}' (prices in tenths of a million).")
    for key in ("bank", "limit", "made"):
        if not isinstance(team["transfers"].get(key), int):
            raise TeamError(f"transfers.{key} must be an integer.")
    team.setdefault("chips", [])
    return team


def chip_weeks(team, static, gws):
    """Weeks in gws each chip may be fixed in for this team (chips.validate's rules, per chip on its own)."""
    from fpl_model.optimizer.chips import validate
    out = {}
    for chip in CHIPS:
        weeks = []
        for gw in gws:
            try:
                validate({chip: gw}, gws, static, team)
                weeks.append(gw)
            except ValueError:
                pass
        out[chip] = weeks
    return out
