"""FPL's BPS weights, in one place for simulate_bonus.py and build_residual_bps_baseline.py (CHANGELOG §181).

These are game rules, not estimates, so they can't be computed: check them against FPL's rules each summer.
A rule change needs a new season entry, as CBI_ONE_PER_THREE_SEASON.
"""
GOAL_BPS = {"D": 12, "M": 18, "F": 24, "G": 12}
CLEAN_SHEET_BPS = 12            # defenders and goalkeepers, 60+ minutes
PEN_GOAL_BPS = 12
ASSIST_BPS = 9
MISS_PEN_BPS = -6
SAVE_BPS = 2
PEN_SAVE_BPS = 7                # 8 less the big-chance bonus already counted
GC_BPS = -4                     # per goal conceded, D and G, any minutes
CBI_ONE_PER_THREE_SEASON = 2026  # first season at 1 BPS per 3 CBI; earlier FPL seasons paid 1 per 2
