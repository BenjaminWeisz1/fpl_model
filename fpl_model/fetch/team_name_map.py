FPL_TO_FBREF_TEAM_NAME = {
    "Spurs": "Tottenham",
    "Man Utd": "Manchester Utd",
    "Man City": "Manchester City",  # inferred by pattern, not yet directly observed -- verify
    "Newcastle": "Newcastle Utd",
    "Nott'm Forest": "Nott'ham Forest",
    "Leeds": "Leeds United",
}


def to_fbref_name(fpl_name: str) -> str:
    return FPL_TO_FBREF_TEAM_NAME.get(fpl_name, fpl_name)


if __name__ == "__main__":
    # Verification helper: run this file directly to print every current
    # team's raw FPL name alongside what it maps to, so you can check the
    # FULL list against fbref's actual naming rather than just the teams
    # that happened to appear in one fetch.
    import requests
    resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=30)
    resp.raise_for_status()
    teams = resp.json()["teams"]
    print(f"{'FPL name':<20} -> {'Mapped to':<20} {'(changed?)' if False else ''}")
    for t in sorted(teams, key=lambda x: x["name"]):
        mapped = to_fbref_name(t["name"])
        flag = "  <-- MAPPED" if mapped != t["name"] else ""
        print(f"{t['name']:<20} -> {mapped:<20}{flag}")
