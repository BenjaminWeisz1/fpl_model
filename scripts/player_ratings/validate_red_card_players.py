"""
Player-level red-card adjustment in the goal/assist loop (CHANGELOG §136): score the two pre-match logs.

    GOAL_MODEL_RED_CARD=0|1 GOAL_MODEL_PREMATCH_LOG=.../log_rc{0,1}.csv build_player_ratings.py   (run first)
    python validate_red_card_players.py    # -> data/validation/red_card_20261030/player/results.json

MSE of match npxG / xAG (weighted, as validate_opp_beta.py) on rows from matches without a red card, where the targets
are identical in both runs; goalkeepers excluded. Rule: adopt unless either target's 2023-2026 MSE worsens by > 0.05%.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_prior_k import boot_diff  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "validation" / "red_card_20261030" / "player"
TARGETS = {"npxG": ("npxg90", "pred_goals_90"), "xAG": ("xag90", "pred_ast_90")}
LIMIT = 0.0005


def main():
    base = pd.read_csv(OUT / "log_rc0.csv", encoding="utf-8-sig", low_memory=False)
    cand = pd.read_csv(OUT / "log_rc1.csv", encoding="utf-8-sig", low_memory=False)
    assert len(base) == len(cand) and (base["Player"].to_numpy() == cand["Player"].to_numpy()).all()
    rc = pd.read_csv(ROOT / "data" / "current" / "red_card_factors.csv", encoding="utf-8-sig")
    red = set(zip(rc["Season"], rc["Home"], rc["Away"]))
    home = np.where(base["Venue"] == "Home", base["Team"], base["Opponent"])
    away = np.where(base["Venue"] == "Home", base["Opponent"], base["Team"])
    in_red = np.array([(s, h, a) in red for s, h, a in zip(base["Season"], home, away)])
    for t, (a, _) in TARGETS.items():   # targets must be identical outside red-card matches
        assert np.allclose(base.loc[~in_red, a], cand.loc[~in_red, a]), t
    gk = set(pd.read_csv(ROOT / "data" / "current" / "transfers.csv", encoding="utf-8-sig").query("Pos == 'GK'")["Player"])
    keep = (base["Season"] >= 2018).to_numpy() & ~base["Player"].isin(gk).to_numpy() & ~in_red
    masks = {"2018_2022": keep & (base["Season"] <= 2022).to_numpy(), "2023_2026": keep & (base["Season"] >= 2023).to_numpy(),
             "2025plus": keep & (base["Season"] >= 2025).to_numpy()}
    w = base["weight"].to_numpy()
    players = base["Player"].to_numpy()
    res = {"rows": {k: int(m.sum()) for k, m in masks.items()}, "red_rows_excluded": int(in_red.sum())}
    for t, (a, p) in TARGETS.items():
        e0 = ((base[p].to_numpy() - base[a].to_numpy()) * w) ** 2
        e1 = ((cand[p].to_numpy() - cand[a].to_numpy()) * w) ** 2
        res[t] = {}
        for k, m in masks.items():
            ch, ci = boot_diff(e0[m], e1[m], players[m])
            res[t][k] = {"change_pct": round(100 * ch, 4), "ci95_pct": [round(100 * ci[0], 4), round(100 * ci[1], 4)],
                         "calibration_base": round(float((base[a].to_numpy()[m] * w[m]).sum() / (base[p].to_numpy()[m] * w[m]).sum()), 4),
                         "calibration_cand": round(float((cand[a].to_numpy()[m] * w[m]).sum() / (cand[p].to_numpy()[m] * w[m]).sum()), 4)}
    res["adopt"] = bool(all(res[t]["2023_2026"]["change_pct"] <= 100 * LIMIT for t in TARGETS))
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
