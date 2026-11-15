"""Team-goal coherence in simulate_match (CHANGELOG §145).

Run locally:
    python -m fpl_model.bonus.test_coherence
"""
import numpy as np
import pandas as pd

from fpl_model.bonus import simulate_bonus


def match_players():
    rows = []
    for team in ("A", "B"):
        for pos, npg in (("G", 0.0), ("D", 0.05), ("M", 0.2), ("F", 0.5)):
            rows.append({"Pos": pos, "Team": team, "P1": 1.0, "P60": 1.0, "NPG90_scaled": npg, "Ast90_scaled": 0.1,
                         "PenAttempts": 0.1 if pos == "F" else 0.0, "Defcon_Mu90": 0.0, "ResidualBPS_Mean": 5.0,
                         "ResidualBPS_Std": 3.0, "PB_P0": 1.0, "PB_P2": 0.0, "PB_P4": 0.0, "PB_P6": 0.0})
    return pd.DataFrame(rows)


def run(players, exp_gc, seed=7, n=4000):
    simulate_bonus.rng = np.random.default_rng(seed)
    return simulate_bonus.simulate_match(players, exp_gc, exp_gc, "A", n)[0]


def test_scorers_take_the_clean_sheet_away():
    # With expected goals conceded 0, goals conceded come only from the opponents' simulated goals, so team A's
    # back line earns more bonus when team B's players cannot score.
    players = match_players()
    blunt = players.copy()
    team_b = blunt["Team"] == "B"
    blunt.loc[team_b, ["NPG90_scaled", "PenAttempts"]] = 0.0
    back_a = ((players["Team"] == "A") & players["Pos"].isin(["G", "D"])).to_numpy()
    assert run(players, 0.0)[back_a].sum() < run(blunt, 0.0)[back_a].sum()


def test_seed_reproducible():
    assert (run(match_players(), 1.2) == run(match_players(), 1.2)).all()


if __name__ == "__main__":
    test_scorers_take_the_clean_sheet_away()
    test_seed_reproducible()
    print("2 tests pass")
