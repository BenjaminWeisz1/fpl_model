"""
Every setting the package reads from the environment, declared once.

Each setting has a name (the environment variable), a type, a default, a one-line meaning and the CHANGELOG section
that introduced it. Revert switches (settings that restore an older model) also name the pending check that decides
whether they can go. Modules read a setting with `config.get(NAME)` at import time, exactly when they used to call
`os.environ.get`, so a research script that sets the variable before importing (or in a subprocess's environment)
still controls it. `tests/test_config.py` checks that no other module in the package reads `os.environ` directly.

    python -m fpl_model.config        # print every setting with its current value

Production uses the defaults: the weekly orchestrators never set any of these.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_BOOL = {"1": True, "0": False}


@dataclass(frozen=True)
class Setting:
    name: str
    kind: str                       # "bool", "int", "float", "str", "path" or "choice"
    default: Any                    # a value, or a function of no arguments for a default that depends on others
    meaning: str
    changelog: str
    pending: str = ""               # revert switches: the check that decides whether the switch can go
    choices: tuple = field(default=())

    def parse(self, raw: str) -> Any:
        if self.kind == "bool":
            if raw not in _BOOL:
                raise ValueError(f"{self.name} must be 0 or 1, not {raw!r}")
            return _BOOL[raw]
        if self.kind == "int":
            return int(raw)
        if self.kind == "float":
            return float(raw)
        if self.kind == "path":
            return Path(raw)
        if self.kind == "choice" and raw not in self.choices:
            raise ValueError(f"{self.name} must be one of {', '.join(self.choices)}, not {raw!r}")
        return raw


def _alpha_npg_default():
    return get("GOAL_MODEL_ALPHA") if is_set("GOAL_MODEL_ALPHA") else 0.035


def _alpha_npg_cb_default():
    # An experiment that moves the general or npxG floor without naming this one moves centre-backs too (pre-§235).
    return get("GOAL_MODEL_ALPHA_NPG") if is_set("GOAL_MODEL_ALPHA_NPG") or is_set("GOAL_MODEL_ALPHA") else 0.018


def _prior_k_youth_default():
    return get("GOAL_MODEL_PRIOR_K")


_SETTINGS = [
    # ---- goal/assist ratings (fpl_model/player_ratings/build_player_ratings.py) ----
    Setting("GOAL_MODEL_ALPHA", "float", 0.025,
            "learning-rate floor of the goal/assist loop (xAG, and npxG unless set)", "§196"),
    Setting("GOAL_MODEL_ALPHA_NPG", "float", _alpha_npg_default, "npxG learning-rate floor", "§233",
            pending="adopted by user decision; 0.025 reverts"),
    Setting("GOAL_MODEL_ALPHA_NPG_CB", "float", _alpha_npg_cb_default,
            "npxG learning-rate floor for centre-back matches", "§235",
            pending="adopted by user decision; 0.035 reverts"),
    Setting("GOAL_MODEL_PRIOR_K", "float", 16.0, "matches the starting rating counts as", "§112"),
    Setting("GOAL_MODEL_PRIOR_K_YOUTH", "float", _prior_k_youth_default,
            "matches a youth row's starting rating counts as", "§126"),
    Setting("GOAL_MODEL_RETURN_H", "float", 5000.0, "returners' blend discount, minutes per season away", "§114"),
    Setting("GOAL_MODEL_OPP_BETA_NPG", "float", 0.7, "dampening exponent on the opponent factor, npxG", "§123"),
    Setting("GOAL_MODEL_OPP_BETA_AST", "float", 0.8, "dampening exponent on the opponent factor, xAG", "§123"),
    Setting("GOAL_MODEL_YOUTH_NPG", "float", None, "research: one fitted youth npxG multiplier for every position",
            "§113"),
    Setting("GOAL_MODEL_YOUTH_AST", "float", None, "research: one fitted youth xAG multiplier for every position",
            "§113"),
    Setting("GOAL_MODEL_XAG_CALIBRATION", "choice", "fw_cb", "expected assists calibrated to assists by match slot",
            "§243-§244", pending="season-end check, data/validation/xag_calibration_seasonend_2027/protocol.json",
            choices=("off", "fw_cb", "all", "both")),
    Setting("GOAL_MODEL_POS_UPDATE", "bool", True, "entrant position update in the loop", "§247, §249",
            pending="season-end re-score (AGENDA 'Arrivals and roles'); 0 reverts"),
    Setting("GOAL_MODEL_POS_UPDATE_C_NPG", "float", 0.4, "entrant position update strength, npxG", "§249"),
    Setting("GOAL_MODEL_POS_UPDATE_C_AST", "float", 0.35, "entrant position update strength, xAG", "§249"),
    Setting("GOAL_MODEL_POSITION_RATES", "choice", "file",
            "research: single-position rates from the file or from PL starts' slots", "§246",
            choices=("file", "slots", "slots3")),
    Setting("GOAL_MODEL_PREMATCH_LOG", "str", "",
            "research: write every row's pre-match rating here (ratings then go to <path>.ratings.csv)", "§72, §74"),
    Setting("GOAL_MODEL_NEW_PRIORS_FILE", "path", None, "research: new-player priors file (default the production one)",
            "§103"),
    Setting("GOAL_MODEL_RETURN_PRIORS_FILE", "str", "", "research: returners' new-league prior from a file", "§128"),
    Setting("GOAL_MODEL_RED_CARD_FILE", "path", None,
            "research: red-card player factors file (default the production one)", "§136"),
    Setting("GOAL_MODEL_MOVE_TABLE", "str", "", "research: same-player move factors (needs GOAL_MODEL_PREMATCH_LOG)",
            "§173"),
    Setting("GOAL_MODEL_ROLE_K_NPG_UP", "float", None, "research: role factor exponent (needs the pre-match log)",
            "§172"),
    Setting("GOAL_MODEL_ROLE_K_NPG_DOWN", "float", None, "research: role factor exponent", "§172"),
    Setting("GOAL_MODEL_ROLE_K_AST_UP", "float", None, "research: role factor exponent", "§172"),
    Setting("GOAL_MODEL_ROLE_K_AST_DOWN", "float", None, "research: role factor exponent", "§172"),
    # ---- points, bonus and cards ----
    Setting("BONUS_SAVE_FIXTURE", "bool", True, "keeper saves in the bonus simulation follow the fixture", "§165",
            pending="adopted by user decision against the rule; keeper-save refit at season end"),
    Setting("CARD_ROLE_MIX", "choice", "forecast", "how a fixture's minutes split into starts and cameos for cards",
            "§193, §195", choices=("forecast", "trailing")),
    Setting("DEFCON_POINTS_CALIBRATION", "bool", True, "defender Defcon_Mu90 x 1.083 for defcon points", "§153, §159",
            pending="refit when a season ends"),
    # ---- defcon model (fpl_model/defcon) ----
    Setting("DEFCON_TUNE", "bool", False, "re-run the ~5 h hyper-parameter search instead of the locked configs",
            "§19, §27"),
    Setting("DEFCON_MODEL_CACHE", "bool", True, "reuse cached defcon models (keyed on training data and source)",
            "§27"),
    Setting("DEFCON_RF_N_JOBS", "int", -1, "random-forest fit parallelism (-1 = all cores; results do not change)",
            "§27"),
    # ---- fetchers ----
    Setting("FOTMOB_ALLOW_MISSING", "bool", False, "continue part 1 when a FotMob match page is missing", "§197"),
    Setting("UNDERSTAT_THROTTLE", "float", 6.0, "seconds between Understat requests", "§37"),
    Setting("CLUBELO_SOURCE", "choice", "mirror", "ClubElo ratings from the GitHub mirror or the (gone) API", "§250",
            choices=("mirror", "api")),
]
SETTINGS: dict[str, Setting] = {s.name: s for s in _SETTINGS}


def is_set(name: str) -> bool:
    """True when the environment gives the setting a non-blank value."""
    if name not in SETTINGS:
        raise KeyError(f"{name} is not declared in fpl_model/config.py")
    return os.environ.get(name, "") != ""


def get(name: str) -> Any:
    """The setting's value: the environment's if set (and not blank), else the declared default."""
    if is_set(name):
        return SETTINGS[name].parse(os.environ[name])
    s = SETTINGS[name]
    return s.default() if callable(s.default) else s.default


def main() -> None:
    for s in _SETTINGS:
        tag = "set" if is_set(s.name) else "default"
        print(f"{s.name:32} {get(s.name)!r:12} ({tag})  {s.meaning} [{s.changelog}]"
              + (f"; pending: {s.pending}" if s.pending else ""))


if __name__ == "__main__":
    main()
