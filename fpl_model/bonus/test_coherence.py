"""Team-goal coherence in simulate_match (CHANGELOG §145).

Run locally:
    python test_coherence.py
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


def run(coherent, exp_gc, seed=7, n=4000):
    original = simulate_bonus.COHERENCE
    simulate_bonus.COHERENCE = coherent
    simulate_bonus.rng = np.random.default_rng(seed)
    try:
        return simulate_bonus.simulate_match(match_players(), exp_gc, exp_gc, "A", n)[0]
    finally:
        simulate_bonus.COHERENCE = original


def test_scorers_take_the_clean_sheet_away():
    # exp_gc = 0: independent draws give both sides a clean sheet every time, coherence only when no opponent scores.
    on, off = run(True, 0.0), run(False, 0.0)
    back = match_players()["Pos"].isin(["G", "D"]).to_numpy()
    assert on[back].sum() < off[back].sum()


def test_seed_reproducible():
    assert (run(True, 1.2) == run(True, 1.2)).all()


if __name__ == "__main__":
    test_scorers_take_the_clean_sheet_away()
    test_seed_reproducible()
    print("2 tests pass")
