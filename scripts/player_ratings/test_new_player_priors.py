"""Tests for the new-player priors (CHANGELOG §103-§104). Run: python test_new_player_priors.py (or pytest)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import new_player_priors as npp  # noqa: E402
import build_new_player_priors as bnp  # noqa: E402


def _toy():
    arr = pd.DataFrame({"pos_npxG": [0.3, 0.1], "pos_xAG": [0.1, 0.1], "Team_Attack_Rating": [1.32023, 2.64046],
                        "Age": [25.0, 30.0]})
    slots = {"m": np.array([[900.0, 0, 0], [0.0, 0, 0]]), "z": np.zeros((2, 3)),
             "league": np.array([["U:La_liga", "", ""], ["M:Championship", "", ""]], dtype=object),
             "x": {"npxG": np.array([[3.0, 0, 0], [0.0, 0, 0]]), "xAG": np.zeros((2, 3))}}
    p = {"d": 0.5, "beta": 0.0, "b": 1.0, "M0": 900.0, "gamma": 0.0, "alpha": 0.0, "mu_U": 0.0, "mu_M": 0.0, "u": {}}
    return arr, slots, p


def test_prior_formula_by_hand():
    arr, slots, p = _toy()
    pr = npp.PriorModel.predict_with(p, arr, slots, "npxG")
    # (3 + 0.3 * 10) / (10 + 10) = 0.3 ; no data -> the position prior exactly
    assert np.allclose(pr, [0.3, 0.1])


def test_team_factor_and_age():
    arr, slots, p = _toy()
    p.update(gamma=1.0, alpha=0.1)
    pr = npp.PriorModel.predict_with(p, arr, slots, "npxG")
    assert np.isclose(pr[1], 0.1 * 2.0 * np.exp(0.5))


def test_position_rule_matches_build_player_ratings():
    pp = npp.position_priors(["FW", "W,AM", "AM,CM,DM"])
    single = pd.read_csv(npp.DATA / "position_combinations.csv").set_index("Pos")
    assert np.isclose(pp["W,AM"]["NPG/90"], (2 * single.loc["W", "NPG/90"] + single.loc["AM", "NPG/90"]) / 3)
    assert np.isclose(pp["AM,CM,DM"]["Assists/90"],
                      (4 * single.loc["AM", "Assists/90"] + 2 * single.loc["CM", "Assists/90"] + single.loc["DM", "Assists/90"]) / 7)


def test_name_match_unique_only():
    up = pd.DataFrame({"ULeague": ["Ligue_1"] * 3, "Season": [2025] * 3, "Understat_ID": [1, 2, 3],
                       "Name": ["Emanuel Emegha", "Jonathan David", "David Silva"]})
    arr = pd.DataFrame({"Player": ["Emmanuel Emegha", "David", "Nobody Here"], "League": ["Ligue 1"] * 3,
                        "Season": [2026] * 3})
    found = bnp.name_match_ids(arr, up, {})
    assert found == {"Emmanuel Emegha": 1}  # "David" is ambiguous (two hits), "Nobody Here" has none


def test_production_file():
    d = pd.read_csv(npp.DATA / "new_player_priors.csv", encoding="utf-8-sig")
    adopted = set()   # §103 groups, plus the FotMob-path (§116) and refit (§127, §139) groups
    for f in (bnp.PARAMS, bnp.PARAMS_FOTMOB, bnp.PARAMS_REFIT):
        if f.exists():
            adopted |= set(json.loads(f.read_text(encoding="utf-8"))["adopted_groups"])
    assert d[["Initial_NPG_Rating", "Initial_Assist_Rating"]].notna().all().all()
    assert (d[["Initial_NPG_Rating", "Initial_Assist_Rating"]] > 0).all().all()
    assert not d.duplicated(["Player", "Season"]).any()
    assert set(d.loc[d["Adopted"], "Group"]) == adopted


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
