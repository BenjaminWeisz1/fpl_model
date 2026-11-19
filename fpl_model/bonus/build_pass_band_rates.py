"""
Per-player pass-band probabilities for each upcoming fixture (CHANGELOG §68).

For every outfield and goalkeeper in players_current.csv and each round in
fixture_goal_predictions.csv: shrunk band distribution from the player's last 38
Sofascore 60+ appearances (position prior when he has none), tilted by this
season's opponent effect and the venue effect. When pass bands are inactive
(see pass_bands.active) it writes the header only, and the simulator falls back
to the §64 model.

Run locally (after build_residual_bps_baseline.py):
    python -m fpl_model.bonus.build_pass_band_rates
"""
import pandas as pd

from fpl_model import paths
from fpl_model.bonus import pass_bands as pb

_DATA = pb._DATA
OUT_FILE = paths.current_file("pass_band_rates.csv", root=_DATA)
COLUMNS = ["Player", "Team", "Round", "Opponent", "Venue", "Sofascore_Appearances",
           "PB_P0", "PB_P2", "PB_P4", "PB_P6", "PB_Expected"]
from fpl_model.league_rates import current_season  # noqa: E402

CURRENT_SEASON = current_season()


def main():
    on, reason = pb.active()
    if not on:
        pd.DataFrame(columns=COLUMNS).to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
        print(f"Pass bands inactive ({reason}); wrote empty {OUT_FILE.name}, bonus uses the §64 model")
        return
    so = pb.load_sofascore()
    counts, pmean = pb.history(so)
    opp = pb.opponent_effects(so, CURRENT_SEASON, pmean)
    # A player's Sofascore identity: the canonical ID of his latest named appearance.
    named = so.dropna(subset=["Local_Player"]).sort_values("Kickoff")
    canonical = named.groupby("Local_Player")["Canonical_Player_ID"].last()

    players = pd.read_csv(paths.current_file("players_current.csv", root=_DATA), encoding="utf-8-sig")
    players = players[players["Pos"].isin(list(pb.PRIOR))]
    fixtures = pd.read_csv(paths.current_file("fixture_goal_predictions.csv", root=_DATA), encoding="utf-8-sig")
    sides = pd.concat([
        fixtures[["Round", "Home", "Away"]].rename(columns={"Home": "Team", "Away": "Opponent"}).assign(Venue="Home"),
        fixtures[["Round", "Away", "Home"]].rename(columns={"Away": "Team", "Home": "Opponent"}).assign(Venue="Away"),
    ]).drop_duplicates(["Round", "Team"])  # first fixture per round, as build_simulation_features

    rows = []
    for p in players.itertuples():
        cid = canonical.get(p.Player)
        c = counts.loc[cid].to_numpy() if cid in counts.index else pb.BANDS * 0
        base = pb.shrunk(c, p.Pos)
        for fx in sides[sides["Team"] == p.Team].itertuples():
            q = pb.tilt(base, opp.get(fx.Opponent, 0.0) + pb.VENUE[fx.Venue])
            rows.append({"Player": p.Player, "Team": p.Team, "Round": fx.Round, "Opponent": fx.Opponent,
                         "Venue": fx.Venue, "Sofascore_Appearances": int(c.sum()),
                         "PB_P0": q[0], "PB_P2": q[1], "PB_P4": q[2], "PB_P6": q[3], "PB_Expected": q @ pb.BANDS})
    out = pd.DataFrame(rows, columns=COLUMNS)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    with_history = (out.drop_duplicates("Player")["Sofascore_Appearances"] > 0).mean()
    print(f"Pass bands active: {len(out)} player-fixture rows, {with_history:.1%} of players with Sofascore history")
    print(out.sort_values("PB_Expected", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
