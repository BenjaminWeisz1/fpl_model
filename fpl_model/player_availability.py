"""Who is really an option: the one place that combines FPL's status with expected minutes (CHANGELOG §168).

players_current.csv keeps every player FPL lists, including those who have left (status "u"). Keep it
that way: it is also the identity and position lookup for their history. Filter lists of options with
this instead. minutes.csv outranks FPL flags, so a player is Departed only when both agree:
status "u" and no expected minutes. Injured players (status "i") stay.
"""
import pandas as pd


def with_availability(players, minutes):
    """players plus Total_xMins and Departed. minutes.csv is keyed on (raw name, team code)."""
    xmins = [c for c in minutes.columns if c.endswith("_xMins")]
    totals = minutes.assign(_total=minutes[xmins].sum(axis=1)).groupby(["Name", "Team"])["_total"].sum()
    def lookup(names):
        keys = pd.MultiIndex.from_arrays([names, players["Team_Short"]])
        return pd.Series(totals.reindex(keys).to_numpy(), index=players.index)
    total = lookup(players["Raw_Web_Name"]).fillna(lookup(players["Web_Name"])).fillna(0.0)
    return players.assign(Total_xMins=total, Departed=players["Status"].eq("u") & total.eq(0))


def status_conflicts(players, minutes):
    """Players FPL lists as unavailable to the league whom minutes.csv still has playing."""
    frame = with_availability(players, minutes)
    return frame[frame["Status"].eq("u") & frame["Total_xMins"].gt(0)]
