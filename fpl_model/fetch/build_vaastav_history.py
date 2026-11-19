"""
Builds data/current/sourced/vaastav_player_match_history.csv from the frozen vaastav archive in
data/raw/vaastav/ (CHANGELOG §96): one row per player per played fixture, 2017-18 .. 2024-25
(project Seasons 2017-2024), with FPL-definition bps, bonus, assists and the other scoring fields,
for training the bonus model on pre-2025 seasons later.

- Teams: the season's `team` column from 2020-21; earlier, the fixture's other side of
  `opponent_team`, named through master_team_list.csv. Mapped to this project's names.
- Duplicated (element, fixture) rows (2019-20's postponed fixtures appear twice: a 0-minute
  placeholder and the replayed match) keep the latest kickoff.
- Identity: each row is matched to the Sofascore played row of the same (Season, Team, Opponent,
  Venue) by optimal assignment on name similarity with minutes within 10 (as §94), adding
  Canonical_Player_ID and Local_Player (the player_match_stats.csv name). Unmatched rows are kept
  with blank identity, so every fixture's full played roster (and its bonus) is present.
- Reviewed identities in vaastav_name_overrides.csv then replace what name matching got wrong: FPL
  full names it couldn't match (Fabio Henrique Tavares, Fabinho) and teammates it swapped. The
  build prints every played player_match_stats row left without a partner (CHANGELOG §167).
- Stat fields a season does not publish stay blank (NULL), never filled.
BPS rules changed over the seasons; see "Rules Verification: BPS Values by Season" in
"Data Sources for Reconstructing the FPL Bonus Points System.md" before training on it.

Run locally (after fetch_vaastav.py; no network):
    python -m fpl_model.fetch.build_vaastav_history
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from fpl_model import paths
from fpl_model.fetch.build_position_crosswalk import name_similarity

_ROOT = paths.ROOT
RAW = paths.RAW / "vaastav"
_DATA = paths.CURRENT
OUT_FILE = paths.current_file("vaastav_player_match_history.csv", root=paths.CURRENT)
OVERRIDES_FILE = paths.current_file("vaastav_name_overrides.csv", root=paths.CURRENT)
SEASONS = list(range(2017, 2025))
TEAM = {"Cardiff": "Cardiff City", "Leeds": "Leeds United", "Leicester": "Leicester City", "Luton": "Luton Town",
        "Man City": "Manchester City", "Man Utd": "Manchester Utd", "Newcastle": "Newcastle Utd",
        "Norwich": "Norwich City", "Nott'm Forest": "Nott'ham Forest", "Spurs": "Tottenham", "Stoke": "Stoke City",
        "Swansea": "Swansea City", "Ipswich": "Ipswich Town", "Hull": "Hull City"}
# every scoring / BPS-relevant field any season publishes; absent ones stay NULL
FIELDS = ["minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "own_goals", "penalties_missed",
          "penalties_saved", "saves", "yellow_cards", "red_cards", "bonus", "bps", "total_points",
          "big_chances_created", "big_chances_missed", "clearances_blocks_interceptions", "key_passes", "tackles",
          "recoveries", "errors_leading_to_goal", "errors_leading_to_goal_attempt", "attempted_passes",
          "completed_passes", "dribbles", "fouls", "offside", "open_play_crosses", "penalties_conceded", "tackled",
          "target_missed", "winning_goals", "expected_goals", "expected_assists", "starts", "influence",
          "creativity", "threat", "ict_index", "team_h_score", "team_a_score"]
MIN_TOL = 10
NAME_MIN = 0.5


def read_csv_any(path):
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def load_season(season, teams):
    tag = f"{season}-{str(season + 1)[2:]}"
    g = read_csv_any(RAW / tag / "gws" / "merged_gw.csv")
    g = g.sort_values("kickoff_time").drop_duplicates(["element", "fixture"], keep="last")
    g = g[g["minutes"] > 0].copy()
    if "team" in g.columns:
        g["team_name"] = g["team"]
        other = g.groupby("fixture")["team_name"].agg(lambda s: sorted(set(s)))
        g["opp_name"] = [next((t for t in other[f] if t != tm), None) for f, tm in zip(g["fixture"], g["team_name"])]
    else:
        ids = g.groupby("fixture")["opponent_team"].agg(lambda s: sorted(set(s)))
        team_id = [next((t for t in ids[f] if t != o), None) for f, o in zip(g["fixture"], g["opponent_team"])]
        names = teams[teams["season"] == tag].set_index("team")["team_name"]
        g["team_name"], g["opp_name"] = pd.Series(team_id, index=g.index).map(names), g["opponent_team"].map(names)
    out = pd.DataFrame({
        "Season": season, "Season_Tag": tag, "GW": g.get("GW", g.get("round")), "Fixture_FPL": g["fixture"],
        "Element_FPL": g["element"], "Kickoff": g["kickoff_time"],
        "Name_FPL": g["name"].astype(str).str.replace(r"_\d+$", "", regex=True).str.replace("_", " "),
        "Team": g["team_name"].map(lambda t: TEAM.get(t, t)), "Opponent": g["opp_name"].map(lambda t: TEAM.get(t, t)),
        "Venue": np.where(g["was_home"].astype(str).str.lower().isin(["true", "1"]), "Home", "Away")})
    for f in FIELDS:
        out[f] = g[f] if f in g.columns else np.nan
    return out


def sofascore_rows():
    so = pd.read_csv(paths.current_file("sofascore_player_match_stats.csv", root=_DATA), encoding="utf-8-sig", low_memory=False,
                     usecols=["Season", "Team", "Opponent", "Venue", "Player_Sofascore", "Local_Player",
                              "Canonical_Player_ID", "Appearance_Status", "Sofascore_minutesPlayed"])
    so = so[so["Appearance_Status"].eq("played") & so["Season"].between(SEASONS[0], SEASONS[-1])].copy()
    so["mins"] = pd.to_numeric(so["Sofascore_minutesPlayed"], errors="coerce").fillna(0)
    return so


def attach_identity(v, so):
    key = ["Season", "Team", "Opponent", "Venue"]
    groups = {k: g for k, g in so.groupby(key)}
    for col in ("Canonical_Player_ID", "Local_Player", "Player_Sofascore"):
        v[col] = None
    v["Sofascore_Minutes"], v["Match_Score"] = np.nan, np.nan
    for k, g in v.groupby(key):
        s = groups.get(k)
        if s is None:
            continue
        sim = np.array([[name_similarity(a, b) if abs(ma - mb) <= MIN_TOL else 0.0
                         for b, mb in zip(s["Player_Sofascore"], s["mins"])] for a, ma in zip(g["Name_FPL"], g["minutes"])])
        for i, j in zip(*linear_sum_assignment(sim, maximize=True)):
            if sim[i, j] >= NAME_MIN:
                vi, sr = g.index[i], s.iloc[j]
                v.loc[vi, ["Canonical_Player_ID", "Local_Player", "Player_Sofascore"]] = [
                    sr["Canonical_Player_ID"], sr["Local_Player"], sr["Player_Sofascore"]]
                v.loc[vi, ["Sofascore_Minutes", "Match_Score"]] = [sr["mins"], round(float(sim[i, j]), 4)]
    return v


def apply_overrides(v, so, overrides):
    """Set reviewed identities, then take Sofascore IDs from the same match for those and for blanks."""
    key = ["Season", "Team", "Opponent", "Venue"]
    changed = pd.Series(False, index=v.index)
    for r in overrides.itertuples():
        rows = v["Name_FPL"].eq(r.Name_FPL) & (v["Team"].eq(r.Team) if r.Team else True)
        if not rows.any():
            raise ValueError(f"Override matches no rows: {r.Name_FPL} {r.Team}")
        v.loc[rows, "Local_Player"] = r.Local_Player
        changed |= rows
    target = changed | v["Canonical_Player_ID"].isna()
    name = v.loc[target, "Local_Player"].fillna(v.loc[target, "Name_FPL"])
    named = so.dropna(subset=["Local_Player"])
    slot = named.drop_duplicates(key + ["Local_Player"]).set_index(key + ["Local_Player"])
    found = slot.reindex(pd.MultiIndex.from_arrays([*(v.loc[target, c] for c in key), name]))
    ids = named.groupby("Local_Player")["Canonical_Player_ID"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else None)
    # A row keeps what it has unless this match gives a better answer: Sofascore's own name for a
    # player can differ from player_match_stats (the two Josh Kings, CHANGELOG §125).
    def at(col):
        return pd.Series(found[col].to_numpy(), index=name.index)
    v.loc[target, "Canonical_Player_ID"] = at("Canonical_Player_ID").fillna(name.map(ids)).fillna(v.loc[target, "Canonical_Player_ID"])
    v.loc[target, "Player_Sofascore"] = at("Player_Sofascore").fillna(v.loc[target, "Player_Sofascore"])
    v.loc[target, "Sofascore_Minutes"] = at("mins").fillna(v.loc[target, "Sofascore_Minutes"])
    return v


def unpartnered_stats_rows(v):
    """Played player_match_stats rows that no vaastav row claims, and duplicate claims."""
    key = ["Season", "Team", "Opponent", "Venue", "Player"]
    p = pd.read_csv(paths.current_file("player_match_stats.csv", root=_DATA), encoding="utf-8-sig", low_memory=False, usecols=key + ["Minutes"])
    p = p[p["Season"].between(SEASONS[0], SEASONS[-1]) & p["Minutes"].gt(0)]
    claimed = v.assign(Player=v["Local_Player"].fillna(v["Name_FPL"]))[key]
    left = p.merge(claimed.drop_duplicates().assign(_hit=1), on=key, how="left")
    return left[left["_hit"].isna()][key + ["Minutes"]], int(claimed.duplicated().sum())


def main():
    teams = pd.read_csv(RAW / "master_team_list.csv")
    v = pd.concat([load_season(s, teams) for s in SEASONS], ignore_index=True)
    assert v["Team"].notna().all() and v["Opponent"].notna().all(), "unresolved team names"
    assert not v.duplicated(["Season", "Element_FPL", "Fixture_FPL"]).any()
    so = sofascore_rows()
    v = attach_identity(v, so)
    v = apply_overrides(v, so, pd.read_csv(OVERRIDES_FILE, encoding="utf-8-sig", keep_default_na=False))
    v.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    left, duplicates = unpartnered_stats_rows(v)
    print(f"Identity check: {len(left)} played stats rows without a vaastav partner, {duplicates} duplicate claims")
    if len(left):
        print(left.to_string(index=False))
    matched = v["Canonical_Player_ID"].notna()
    fx = v.groupby(["Season", "Fixture_FPL"])["bonus"].sum()
    summary = v.assign(m=matched).groupby("Season").agg(rows=("m", "size"), matched=("m", "mean"), bonus=("bonus", "sum"),
                                                        bps_mean=("bps", "mean"), fixtures=("Fixture_FPL", "nunique"))
    summary["bonus_per_fixture"] = summary["bonus"] / summary["fixtures"]
    print(f"Wrote {len(v):,} played player-fixture rows -> {OUT_FILE.name}; matched to Sofascore {matched.mean():.2%} "
          f"of rows, {v.loc[matched, 'minutes'].sum() / v['minutes'].sum():.2%} of minutes; "
          f"Sofascore played minutes covered {v.loc[matched, 'Sofascore_Minutes'].sum() / so['mins'].sum():.2%}")
    print(summary.round(3).to_string())
    print(f"fixtures with bonus total < 6: {int((fx < 6).sum())} of {len(fx)}")


if __name__ == "__main__":
    main()
