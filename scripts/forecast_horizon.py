"""How many upcoming gameweeks the pipeline forecasts: the one place to change it.

FPL Review's expected minutes, penalty shares and projection tables cover this many rounds, starting
at the next one. Every stage reads its window from here (CHANGELOG §215): fixture goals, minutes
probabilities, penalties, defcon (part 1), simulation features, predicted points, the FPL Review
paste cleaners, the snapshot and the transfer optimizer. Intermediate files label the rounds
GW1..GW<HORIZON> (GW1 = the next round); predicted_points.csv uses the real round numbers.

Was 5 until GW10 of 2026/27; 10 from GW11 (2026-11-10), when FPL Review began supplying 10 gameweeks.
"""
HORIZON = 10

# Weight of a week relative to the week before it, wherever weeks of the window are combined into one number: the
# transfer optimizer's objective (decay_base, the user's choice in optimizer_brief.md) and the website's Player
# ratings minutes (user, 2026-11-11: the more distant the gameweek, the lower its weight).
TIME_DECAY = 0.85


def week_weights(n):
    """Weights of the first n weeks: 1, TIME_DECAY, TIME_DECAY**2, ..."""
    return [TIME_DECAY**i for i in range(n)]

# Relative labels used in intermediate files: GW1 is the next round.
GAMEWEEKS = list(range(1, HORIZON + 1))
