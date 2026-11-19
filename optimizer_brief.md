# Optimizer brief

Status: not started. Everything below was checked on 2026-10-30, the day before the GW9 deadline
(2026-10-31 11:00 UTC). Re-check prices and `team_data.json` if a later session runs after that.

## Task

**Since GW11 (2026-11-10, §215) the horizon is 10 gameweeks**, set by `scripts/forecast_horizon.py`, and `xmin_lb`
is 37.5 per gameweek (375). The text below is the original GW9 brief.


Maximise my time-decayed FPL points over the next 5 gameweeks (GW9–13) from my current squad. Predicted points come from
`data/current/forecasts/predicted_points.csv`, and from nowhere else.

Output 3 plans. Each plan lists, week by week:
- transfers in and out, hits taken, free transfers available, bank
- starting 11, bench order, captain and vice-captain
- expected points per week and in total, after hits, both raw and decayed

Do not run the weekly housekeeping or change production data or the prediction models.

## FPL rules

### Squad
- Budget 100.0M at the start of the season. From now on the limit is `bank + selling prices`, not 100.0M.
- 15 players: 2 goalkeepers, 5 defenders, 5 midfielders, 3 forwards.
- At most 3 players from one club.

### Starting 11
- Exactly 1 goalkeeper, 3–5 defenders, 2–5 midfielders, 1–3 forwards.
- The captain scores double. If the captain does not play, the vice-captain does instead.
- Bench players come on automatically if a starter does not play, in bench order. The bench keeper is first.

### Transfers
- Each gameweek gives +1 free transfer, up to a maximum of 5. You always have at least 1 the next week.
- Each transfer beyond your free transfers costs −4 points.
- Example: 1 FT, no transfers → 2 FT next week. 2 FT, 3 transfers → −4, then 1 FT next week.

### Prices
- You buy at the current price (CP).
- If CP is above your purchase price (PP), you keep half the rise, rounded down to 0.1M:
  `SP = PP + floor((CP − PP) / 0.2) × 0.1`.
  Example: PP 11.5, CP 11.9 or 12.0 → SP 11.7.
- If CP is at or below PP, then `SP = CP`. Example: PP 7.5, CP 7.4 → SP 7.4.

### Chips
Each half of the season has its own set (GW1–19 and GW20–38), and only one chip per gameweek.

| Chip | Status for GW9–13 |
|---|---|
| Bench Boost | played GW1, unavailable |
| Triple Captain | played GW3, unavailable |
| Wildcard | played GW4, unavailable |
| Free Hit | available, but **not to be used in GW9–13** (my decision) |

The second-half chips start at GW20, outside the horizon. No chips are used in any plan.

### Chip rules for fixed chip weeks (2026/27; written 2026-11-10 for the website's solver, §226)
The user fixes the weeks; the solver never picks a chip or a week (`chip_limits` stay 0 for every chip not fixed).
Implemented in `scripts/optimizer/chips.py` and checked independently by `replay_plans.py`.

- **Windows** (from `bootstrap-static` `chips`, read at run time): first set Wildcard and Free Hit GW2–19, Bench Boost and
  Triple Captain GW1–19; second set all four GW20–38. First-half chips don't carry over. A chip is available in a week
  if `bootstrap-static` has an instance covering it and `team_data.json` doesn't show that instance played.
- **One chip per gameweek.** Free Hit not in GW1, and not in GW19 and GW20 both (Premier League, "What's happening with
  FPL chips in 2026/27"). One fixed week per chip type per solve.
- **Wildcard:** unlimited free transfers that week, permanent, no hits. Banked free transfers are kept (Premier League
  chip guide). If played in the first week, transfers already made that week become free too (upstream sets the FT
  count to the `limit`).
- **Free Hit:** unlimited free transfers for one week; the squad reverts at the next deadline, and that week's
  transfers don't count. Budget: bank plus the selling value of the current squad. A current player kept on the Free
  Hit costs his selling price, any other his current price.
- **Free transfers after a Wildcard or Free Hit:** the count is carried unchanged: k free transfers in week N give k in
  N+1, and the week's +1 isn't earned (e.g. 1 FT, Free Hit, 1 FT the next week). Confirmed by the user 2026-11-10; it
  is also upstream's model. The official 2026/27 pages don't state it. The FT cap is 5 (`max_extra_free_transfers` 4).
- **Bench Boost:** all 15 score, so the bench weights don't apply that week. **Triple Captain:** the captain scores ×3.
  Vice-captain cover isn't modelled (no substitutions; the 0.1 vice weight stays).

## Inputs

### `team_data.json`
Saved 2026-10-30 15:05 in the format of FPL's `/my-team/` endpoint. Prices are in tenths of a million.
`picks_last_updated` (23 Oct) is when the picks last changed, not when the file was saved.

- Bank 2.1M (`bank: 21`), 1 free transfer for GW9 (`limit: 1`, `made: 0`).
- `transfers.cost: 4` is the hit price, not a hit already taken.
- `value: 1007` is the sum of current prices. Spending power is 98.8M selling prices + 2.1M bank = 100.9M.
- Checked: all 15 selling prices match the rule above against live prices, the squad is 2/5/5/3, and no club has more than 2.

| ID | Player | Club | Pos | PP | SP | CP | xP GW9–13 |
|---|---|---|---|---|---|---|---|
| 1 | Raya | ARS | G | 6.0 | 6.1 | 6.2 | 20.01 |
| 251 | Lecomte | FUL | G | 4.0 | 4.0 | 4.0 | 0.35 |
| 334 | Muharemović | LEE | D | 5.0 | 5.0 | 5.1 | 18.93 |
| 277 | Egan | HUL | D | 4.0 | 4.0 | 4.1 | 14.55 |
| 449 | Hall | NEW | D | 5.2 | 5.2 | 5.3 | 16.27 |
| 31 | Konsa | ARS | D | 4.4 | 4.5 | 4.7 | 13.91 |
| 115 | De Cuyper | BHA | D | 5.1 | 5.2 | 5.3 | 15.73 |
| 236 | Dewsbury-Hall | EVE | M | 6.5 | 6.5 | 6.6 | 20.82 |
| 154 | Palmer | CHE | M | 9.7 | 9.7 | 9.8 | 22.58 |
| 40 | Rogers | CHE | M | 7.5 | 7.7 | 8.0 | 21.74 |
| 427 | Mbeumo | MUN | M | 7.8 | 7.8 | 7.9 | 23.99 |
| 124 | Groß | BHA | M | 5.6 | 5.9 | 6.2 | 22.28 |
| 249 | Barry | EVE | F | 5.6 | 5.6 | 5.6 | 22.01 |
| 411 | Haaland | MCI | F | 15.5 | 15.6 | 15.8 | 33.64 |
| 346 | Calvert-Lewin | LEE | F | 6.0 | 6.0 | 6.1 | 20.11 |

### Projections and player IDs
- `data/current/forecasts/predicted_points.csv` is the points source: columns `Player, Pos, Team, Price, 9..13, Total`; 669 rows. Columns
  `9`..`13` are each player's expected points for that gameweek. `Player` is `Web_Name` and has no FPL ID.
- `data/current/manual/minutes.csv` supplies IDs and expected minutes only, never points. FPL Review format: `ID` is the FPL element ID,
  with `Name` (raw web name), `BV`, `SV`, `Team` and `9_xMins`..`13_xMins`. It has 680 rows; drop the 11 that are not in FPL
  (`BV` 99.9).
- **Join:** `predicted_points (Player, Team)` → `players_current (Web_Name, Team_Short)` → `(Raw_Web_Name, Team_Short)` →
  bootstrap `(web_name, short team)`. This gave 669 of 669 unique IDs. A direct name join loses 38 players whose names
  are disambiguated in our files (e.g. FPL "Palmer" is our "C. Palmer").
- **Prices come from live `bootstrap-static` `now_cost`, not `Price`:** 7 players had already moved from the CSV on 30 Oct.
- All 20 clubs have one fixture in each of GW9–13, so there are no blank or double gameweeks.

## Optimizer rules to apply

Sources: https://github.com/sertalpbilal/fpl_optimized and https://github.com/solioanalytics/open-fpl-solver.

`fpl_optimized` has no transfer model: its `src/solve.py` builds one-off squads for the website. Its squad rules are all included
below. The transfer model is `open-fpl-solver` (`dev/solver.py`, `solve_multi_period_fpl`, solved with HiGHS), formerly
`sertalpbilal/FPL-Optimization-Tools`, free for personal use under Apache 2.0. Use it as the engine.

### Rules from both repositories
- 15 players at the FPL position counts, at most 3 per club.
- A legal 11 every week, with position limits read from `bootstrap-static`.
- One captain, inside the 11.
- A keeper first on the bench, and every squad player either starts or sits on the bench.

### Rules from `open-fpl-solver`
- **Decisions per gameweek:** squad, starting 11, bench order 0–3, captain, vice-captain, transfers in and out.
- **Vice-captain:** inside the 11 and not the captain.
- **Squad flow:** `squad[w] = squad[w−1] + in − out`. A player cannot be bought and sold in the same week.
- **Bank:** `bank[w] = bank[w−1] + sales − purchases`, and it cannot go negative.
- **Selling:** the first sale of a current player uses their real selling price. If rebought and sold later, they sell at the
  buy price, so buying back earns no profit.
- **Prices are fixed for the horizon.** Rises and falls are not predicted.
- **Free transfers:** `next FT = FT − transfers + 1`, clamped to 1–5.
- **Hits:** `hits ≥ transfers − FT`, costing 4 points each.
- **Substitutions are not simulated.** Bench weights stand in for them: 0.03 for the bench keeper, then 0.21, 0.06 and 0.002.
- **Alternatives:** after each solve, add a cut that forbids the previous GW9 moves and solve again. Three solves give the 3 plans.

### Not applied from `fpl_optimized`
- The 100.0M limit: the budget here is bank plus selling prices.
- The flat 0.1 bench weight: `open-fpl-solver`'s ordered weights are used instead.
- "Every squad player must start at least once" (`solve_iterative_squads`): it would force my backup keeper to start.

### Objective
Maximise the sum over GW9–13 of `0.85^(w − 9) × weekly value`, where:

`weekly value = lineup xP + captain xP + 0.1 × vice xP + weighted bench xP − 4 × hits + 1.5 × FT gain`

- **Time decay 0.85** is FPL Review's setting, which discounts future points for uncertainty:

| GW | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|
| Weight | 100% | 85% | 72.25% | 61.41% | 52.20% |

- **Each free transfer saved is worth 1.5 points.** FT gain is the change in the free-transfer count from the previous week. Rolling
  takes 1 → 2 for +1.5, using 2 takes 2 → 1 for −1.5, and rolling at the cap of 5 gains nothing. The solver's `ft_value: 1.5` with
  an empty `ft_value_list` does this, crediting the gain in the week the extra transfer arrives.
- **Required patch:** the solver has no FT variable beyond the last gameweek, so a transfer saved in GW13 earns nothing and GW13
  transfers look free. Add `fts[14]` to the local copy with the same clamping rule, and credit its gain at GW14's weight (44.37%).

### Settings to use
Upstream's defaults do not match this brief. Set these explicitly:

| Option | Upstream default | Use | Why |
|---|---|---|---|
| `horizon` | 8 | 5 | GW9–13 |
| `objective` | decay | decay | Time-decayed points |
| `decay_base` | 0.9 | 0.85 | FPL Review's value |
| `ft_value` | 1.5 | 1.5 | Each saved FT is worth 1.5 |
| `ft_value_list` | 2, 1.6, 1.3, 1.1 | `{}` | Flat 1.5 for every saved FT |
| `weekly_hit_limit` | 0 (no hits) | `null` | The brief allows hits |
| `no_transfer_last_gws` | 2 | 0 | Otherwise GW12–13 transfers are banned |
| `ft_use_penalty` | 0.2 | 0 | Not in the objective |
| `itb_value` | 0.08 | 0 | Not in the objective |
| `xmin_lb` | 300 | 190 | Scaled from 8 to 5 weeks |
| `chip_limits` | all 0 | all 0 | No chips, including Free Hit |
| `use_fh`, `use_wc`, `use_bb`, `use_tc` | empty | empty | No chips |
| `num_iterations` | 1 | 3 | 3 plans |
| `iteration_criteria` | `this_gw_transfer_in_out` | same | Plans differ in their GW9 moves |
| `bench_weights`, `vcap_weight` | as above | as above | Tie-breakers for bench order and vice-captain |

Rank plans by the decayed objective. Report it alongside raw points (lineup xP + captain xP − hits), and show the 1.5-point FT credit
separately so it is not mistaken for points scored.

## Implementation plan

1. Add `scripts/optimizer/`. Pin open-fpl-solver to a specific commit outside `data/`, and add `highspy` to `requirements.txt`.
2. Build a projection CSV in the solver's format (`ID, Name, Pos, Team, Value, {w}_Pts, {w}_xMins`, with `Pos` G/D/M/F) using the join above.
   Assert 669 rows, unique IDs and all 15 squad IDs present. Check `run/solve.py` and `data/team.json.sample` for how it loads
   the projection file and `team_data.json`, which I have not run.
3. Apply the GW13 free-transfer patch above, then solve with the settings above.
4. Check each plan independently of the solver: replay it against the rules above and recompute budget, FTs, hits, raw points and
   the decayed objective. Also score the no-transfer baseline.
5. Write the 3 plans to `data/optimizer/GW09/`. Record the work in `CHANGELOG.md` (§148) and update `AGENDA.md` and `CLAUDE.md`.

## Decisions
- **Time decay 0.85 and 1.5 points per saved free transfer:** my decisions.
- **No Free Hit in GW9–13:** my decision.
- **Plans differ in their GW9 moves,** because that is the decision to act on now. Allowing any difference across the 5 weeks gives
  near-copies of the same plan. "Roll the transfer" counts as a valid GW9 move.
