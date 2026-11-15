"""Fixed chip weeks for the solver: validation against FPL's chip rules and the solver options (optimizer_brief.md,
"Chip rules for fixed chip weeks", §226).

A chip plan is {"wc": gw or None, "fh": ..., "bb": ..., "tc": ...}. The user fixes the weeks; the solver never picks a
chip or a week, so every chip not fixed keeps chip_limits 0.
"""

CHIPS = ("wc", "fh", "bb", "tc")
FPL_NAMES = {"wc": "wildcard", "fh": "freehit", "bb": "bboost", "tc": "3xc"}
LABELS = {"wc": "Wildcard", "fh": "Free Hit", "bb": "Bench Boost", "tc": "Triple Captain"}
SOLVER_TEXT = {"wc": "WC", "fh": "FH", "bb": "BB", "tc": "TC"}  # the chip column of the solver's picks
NO_CHIPS = dict.fromkeys(CHIPS)


def normalise(chips):
    """Fill missing chips with None and reject unknown keys."""
    chips = chips or {}
    unknown = set(chips) - set(CHIPS)
    if unknown:
        raise ValueError(f"Unknown chips: {sorted(unknown)}; use {list(CHIPS)}.")
    return {c: (None if chips.get(c) is None else int(chips[c])) for c in CHIPS}


def fixed(chips):
    """{gw: chip} for the chips that are played."""
    return {gw: c for c, gw in normalise(chips).items() if gw is not None}


def windows(fpl, chip):
    """(start, stop) of every instance of a chip in bootstrap-static."""
    return [(c["start_event"], c["stop_event"]) for c in fpl["chips"] if c["name"] == FPL_NAMES[chip]]


def played_windows(team, chip):
    """Windows of the instances team_data.json shows as no longer available (played or otherwise unavailable)."""
    return {(c["start_event"], c["stop_event"]) for c in team.get("chips", [])
            if c["name"] == FPL_NAMES[chip] and (c.get("status_for_entry") != "available" or c.get("played_by_entry"))}


def validate(chips, gws, fpl, team):
    """Raise ValueError unless every fixed chip is legal: in the horizon, inside an unplayed instance's window, one
    chip per week, no Free Hit in GW1. Returns the normalised plan."""
    chips = normalise(chips)
    weeks = [gw for gw in chips.values() if gw is not None]
    if len(weeks) != len(set(weeks)):
        raise ValueError("Only one chip can be played in a gameweek.")
    for chip, gw in chips.items():
        if gw is None:
            continue
        if gw not in gws:
            raise ValueError(f"{LABELS[chip]} in GW{gw} is outside the horizon GW{gws[0]}-{gws[-1]}.")
        window = next((w for w in windows(fpl, chip) if w[0] <= gw <= w[1]), None)
        if window is None:
            raise ValueError(f"No {LABELS[chip]} covers GW{gw}.")
        if window in played_windows(team, chip):
            raise ValueError(f"The {LABELS[chip]} for GW{window[0]}-{window[1]} has been played.")
        if chip == "fh" and gw == 1:
            raise ValueError("The Free Hit can't be played in GW1.")
    return chips


def solver_options(chips):
    """Options for the pinned solver: each fixed chip forced in its week (use_<chip>, which also sets its limit to 1);
    every other chip limited to 0."""
    chips = normalise(chips)
    options = {"chip_limits": {c: 0 for c in CHIPS}}
    for chip, gw in chips.items():
        options[f"use_{chip}"] = [] if gw is None else [gw]
    return options


def describe(chips):
    return ", ".join(f"{LABELS[c]} GW{gw}" for gw, c in sorted(fixed(chips).items())) or "no chips"
