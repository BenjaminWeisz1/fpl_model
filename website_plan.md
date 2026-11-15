# Website plan

Status: **plan agreed by the user, 2026-11-10; all five steps built (§216 part 3, server, Projections; §217 Player ratings; §218 Team ratings, Fixtures; §226–§227 solver research; §228 Solver page; later changes §219–§225, §229–§230).** A build session starts from this file and follows
the build order below. Record work in `CHANGELOG.md` as new §-entries and keep AGENDA "Website" current.

A personal website, for the user only, that shows the model's weekly output and runs the transfer solver. It runs
on the user's PC as a local app: free, the fastest machine available for the solver, and a local server can call the
FPL API (a browser page can't, because of CORS). FPL Review's minutes and penalty shares are shown as-is because
nobody else sees the site.

## Decisions (user, 2026-11-10)

| Topic | Decision |
|---|---|
| Audience | The user only. No logins, no public hosting, phone layout not needed. |
| Hosting | Local app on the user's PC (Python server + static pages). |
| Navbar | Projections, Player ratings, Team ratings, Fixtures, Solver. |
| Publishing | A new **part 3** exports the site's data after part 2; the site shows only the latest week (no history). |
| Players | Every player with a non-null player rating (`player_ratings.csv`). |
| Horizon | The pipeline's window, `HORIZON` in `scripts/forecast_horizon.py` (10). Never hardcoded in the site. |
| Probability totals | Clean-sheet and defcon % totals are **sums** over the shown weeks (expected count). |
| Penalty-share total | The weekly values are `penalty_share.csv`'s per-week shares; Total is their **average** over the shown weeks (a summed share has no meaning). Agreed. |
| Penalty-share editing | On Player ratings only: updates goal rating and goal+assist rating. Not Projections, not the solver. |
| Solver | Built and sped up separately first, then integrated. Fixed chip weeks only; the solver never picks them. |

## Architecture

```
run_site_export.py   (part 3)  data/current outputs  ->  site/data/*.json  ->  starts the local server, opens the browser
site/
  server.py          stdlib ThreadingHTTPServer (as scripts/optimizer/optimizer_app.py): serves site/, proxies FPL API
                     calls for the solver, runs solver jobs in a background thread
  index.html         one page, navbar switches views; no build step, no framework
  app.js, heatmap.js, style.css
  data/              written by part 3 only, never edited by hand
```

- **No framework.** About 670 players × 10 weeks is small enough for plain DOM rendering; nothing to install or
  build. Re-evaluate only if a page feels slow.
- **Part 3** is a third orchestrator on `pipeline_stages.run_pipeline`, like parts 1 and 2. Stages: export JSON;
  then launch the server. It refuses stale data the way `save_gameweek_outputs.py` does: `predicted_points.csv` must
  be newer than its inputs and start at the next gameweek. `python run_site.py` (or `run_site_export.py --serve-only`)
  reopens the site without re-exporting.
- **One heatmap component** for every table. Colours are the images' map: `#F8696B` (worst) → `#FFEB84` → `#63BE7B`
  (best), min–max scaled. Scale all week columns together, so a green week is green relative to the other weeks too;
  Total gets its own scale. Columns where lower is better (xGA, Defense rating, fixture difficulty) invert the map.
  Every recolour follows the rows and columns currently shown (filters, removed weeks).

## Shared gameweek controls (Projections and Fixtures)

- **Horizon:** first round to last round (default all HORIZON rounds).
- **Remove weeks:** a toggle per round. A removed week disappears from view and from Total (e.g. GW11–20 without GW14
  when free-hitting it).
- Total = sum of the shown weeks (average for penalty share); the table re-sorts by Total.
- A blank week shows "–" (counts 0); a double week shows the summed value with a small "×2" mark.

## Pages

### 1. Projections

One row per player; per-round values from `predicted_components.csv` (per fixture; summed for doubles).

- **Columns:** Player, Pos, Team, Price, ~~**xMins**~~ (removed by user decision, §222; the rest of this bullet is historical) (always the first stat column, whatever the stat dropdown says),
  then one column per shown round, then Total.
  - xMins = FPL Review's `xMins_Raw`: mean over the shown weeks in the xMins column. The weekly stats themselves use the
    model's minutes as now.
- **Stat dropdown** (default Points): Points (`TotalPts`), Minutes (`xMins_Raw` per week), Goals (penalties
  included), Assists (FPL-only included), Penalties (`penalty_share.csv`, per week), Clean sheets (the team's chance
  in that fixture, `CS_Prob_Team`), Defcon % (`Defcon_Prob`, the probability at the forecast minutes).
- **Filters:** team (one team at a time since §219), position, price (two-handle slider, min–max from the data), name search.
- Sorted by Total, descending; clicking another header sorts by it.

### 2. Player ratings

One row per player, one value each (no weeks).

| Column | Definition | Source |
|---|---|---|
| Minutes | xMins over the HORIZON weeks (FPL Review), weighted 1, 0.85, 0.85², … so nearer weeks count more (user, 2026-11-11; `forecast_horizon.TIME_DECAY`, the solver's decay) | `minutes.csv` |
| Penalty share | mean of the HORIZON weekly shares | `penalty_share.csv` |
| Non-pen goals /90 | goal rating | `player_ratings.csv` `NPG` |
| Goals /90 | non-pen goals /90 + penalty goals /90 | computed below |
| Assists /90 | assist rating | `player_ratings.csv` `Assists` |
| Goals + assists /90 | goals /90 + assists /90 | computed |
| CS % | the team's clean-sheet chance, 90 minutes, average opponent, neutral venue | computed below |
| Defcon % | chance of reaching the threshold in 90 minutes v an average opponent | `player_ratings.csv` `Defcon` |

- **Penalty goals /90** = share × team penalty attempts per 90 × player's conversion rate. Team attempts per 90 =
  the team's non-penalty goals v an average opponent at a neutral venue × `PEN_ATTEMPTS_PER_NPG`. The rate is
  `penalty_rates()`; conversion is `build_conversion_rates()` from `build_penalty_predictions.py`. Part 3 exports each
  piece (share, team attempts, conversion), so the page can recompute.
- **Editing the share:** each player's share cell is editable. An edit updates Goals /90 and Goals + assists /90 for
  that player and recolours the columns; a "reset" restores the model's values. Edits stay in the browser
  (localStorage) until the next part 3, which clears them (a new week, new shares). They never reach Projections
  or the solver.
- **CS %:** exp(−expected goals conceded v average opponent, penalties included), with the same formula as
  `build_simulation_features.py`; identical for teammates, by design.
- **Filters:** position, price slider, team, search, minutes (over 30 / 45 / 60, §230). **Hide columns:** a × above each header since §219 (saved in localStorage).
  Click a header to sort; default sort is Goals + assists /90.

### 3. Team ratings

- **Table:** Team, Overall, Attack, Defense (`team_ratings_current.csv`), heatmap coloured; sorted by Overall; click
  a header to sort by Attack or Defense (Defense coloured lower = better).
- **Below it:** the predicted final table (`predicted_table.csv`: played, points, expected W/D/L remaining, predicted
  points), sorted by predicted points.

### 4. Fixtures (fixture ticker)

- Rows = teams, columns = rounds; each cell is the opponent ("ars (H)") coloured by difficulty.
- **Mode filter:** Overall (expected goal difference), Attack (expected goals for), Defense (expected goals against,
  lower = better), as in `scripts/optimizer/fixture_ticker.py`.
- Same horizon and remove-week controls as Projections. Teams sorted best → worst by the total over the shown weeks;
  any filter change re-sorts.
- **Differences from the images:** they use 8 weeks and a 0.85 decay; the page uses HORIZON weeks and a plain sum (as
  Projections). Values come from `fixture_goal_predictions.csv` (penalties added as in the clean-sheet formula), so
  the page matches the forecast exactly.

### 5. Solver (phase 2, after the solver work below)

- **Load the team:** either an FPL team ID (the server rebuilds picks, bank, free transfers and selling prices from
  public endpoints; upstream's `generate_team_json`), or paste JSON in `team_data.json`'s format (after a transfer,
  or when the ID route can't see something). Selling prices are checked as in `run_optimizer.check_team`.
- **Inputs:** horizon (1..HORIZON); chip weeks for wildcard, free hit, bench boost, triple captain (each "none" or
  a round in the horizon; FPL's chip rules enforced); later, hold / avoid / planned moves as in `optimizer_app.py`.
- **Run:** a background job with a progress line and a time limit; results show the replay-checked plans
  (`replay_plans.py`) as the plan pages do now.

## Solver work (separate, before integration)

### Chips

- Map the fixed weeks to upstream's `forced_chip_gws` with `chip_limits` 1 for each chip used (today all 0).
- Extend `replay_plans.py`'s independent check to chips: free hit (squad reverts the next week, transfers that week
  don't touch the FT count), wildcard (unlimited free moves that week, FTs carry as FPL rules), bench boost (all 15
  score), triple captain (×3). No chip plan is trusted until the replay agrees with the solver's score.
- Free-transfer rules around chips (FTs kept, not gained, on WC/FH weeks) need checking against FPL's 2026/27 rules.
  Write them down in `optimizer_brief.md` first.

### Speed: research plan

Measure before changing anything. **Benchmark:** GW11 projection, horizon 5 and 10, no chips and two chip
scenarios. Report the time to the first plan, to optimal, and the objective at each time limit. Production now uses
`gap` 0 (prove optimality), 3 iterations (three plans), `secs` 1800, upstream's pool filters
(`ev_per_price_cutoff` 30, `keep_top_ev_percent` 5) and `xmin_lb` 375. Levers to test one at a time:

1. **MIP gap:** `mip_rel_gap` 0 → 0.001–0.005. Usually the largest saving; the cost is at most that share of the
   objective (about 0.1–0.5 points over 10 weeks). The replay then reports the real score.
2. **Fewer plans by default:** one plan for a quick answer, three on request; or solve the alternatives in parallel
   processes.
3. **Pool size:** a tighter `ev_per_price_cutoff` / `keep_top_ev_percent`, or a cap of N players per position by
   projected points. Check it never removes a player in an optimal plan from the benchmark runs.
4. **Warm start:** give HiGHS a feasible start (the current squad with no transfers, or last week's plan rolled
   forward) through `setSolution`.
5. **HiGHS settings:** threads and parallel (already on), presolve, MIP heuristic effort; test the newest highspy
   against the pinned 1.15.1.
6. **Upstream changes:** compare the pinned commit (ec65f5e, 2026-09-15) with the current
   [open-fpl-solver](https://github.com/solioanalytics/open-fpl-solver) for model or speed changes; re-pin only with
   `PINNED.md` and `gw14_ft.patch` re-applied and the replay passing.
7. **Other free solvers:** SCIP (Apache 2.0 since v9) and CBC, through the same model, only if HiGHS stays slow.
8. **Model size:** horizon and `no_transfer_last_gws` trade speed for foresight; report the trade, don't change it
   silently.

**Adoption rule (pre-registered):** a lever is kept if the benchmark's replay-checked score stays within 0.1 points
of the gap-0 optimum over the horizon and the GW-now transfers are unchanged in at least 4 of 5 test weeks, or the
user accepts the trade.

**Three solve modes (user request, 2026-11-10): Fast, Standard (the default) and Deep.**
- **Deep** is today's production solve: gap 0, three plans, `secs` 1800.
- **Standard** combines the levers that pass the adoption rule above.
- **Fast** answers in well under a minute: one plan, plus whatever gap, pool and time limit the benchmark shows.
  It may break the adoption rule, but the page shows its measured cost: the objective lost against Deep, and
  whether the GW-now transfers match Deep's.

Each mode is one named settings set, declared once in `scripts/optimizer/`. The benchmark's improving-solution
trajectories set each mode's time limit.

**Built overnight 2026-11-10/11 (§227), in `scripts/optimizer/solve_modes.py`:**
- **Deep:** unchanged; no exact speed-up beat the ±10% noise. 25–28 min and 2.4–2.5 GB at horizon 10.
- **Standard (default):** plan 1 is Deep's solve, so it is the proven optimum. Plans 2–3 come from a model that allows
  transfers only in the first 5 weeks: 0–1.1 points below Deep's alternatives. About 10–11 min at horizon 10; peak
  memory is plan 1's (1.1–2.0 GB).
- **Fast:** one plan, transfers only in the first 4 weeks, gap 0.05%, 60 s limit.
  - 34–63 s and under 350 MB.
  - 0–1.15 points below Deep's best plan: 0.63 for GW11 at horizon 10.
  - Its GW-now moves differed from Deep's on 1 of 6 instances.

The Solver page (step 5) shows each mode's measured cost from `solve_modes.MEASURED` (already in `plans.html`).
Five test weeks aren't saved, so the benchmark uses GW9 and GW10 at horizon 5, and GW11 at horizons 5 and 10 with
and without chips. "4 of 5 weeks" therefore reads as all instances but at most one.

## Build order

1. Part 3 export + server + Projections (the shared heatmap and gameweek controls are built here).
2. Player ratings (with share editing).
3. Team ratings and Fixtures.
4. Solver research (benchmark, chips, speed), separately in `scripts/optimizer/`. Chips and benchmark done (§226); the
   speed session and the three modes are done (§227, `scripts/optimizer/SOLVER_SPEED_PLAN.md`).
5. Solver page. Done (§228): `site/solver.js`, `scripts/optimizer/solver_jobs.py`, `team_source.py`.

Each step: check the page's numbers against the CSVs it came from (an automated test per exporter) and look at it in
the browser.

## Settled details (user, 2026-11-10)

- Penalty-share Total on Projections: the average over the shown weeks.
- Penalty share in a double gameweek on Projections: the average of its games (user, 2026-11-10; one share per week
  in `penalty_share.csv`, so it equals the week's share).
- Projections xMins (column and Minutes stat): FPL Review's raw minutes (`xMins_Raw`), not the model's rescaled ones
  (§177).
- Defcon % on Projections uses the forecast minutes, while Player ratings uses 90 minutes; the labels must say which.
  Superseded by the user (§219): labels carry no brackets; the 90-minute basis is Player ratings' whole premise.

## Layout changes after review (user, 2026-11-10, §219)

These replace the matching details above.
- Team and position filters are single-choice dropdowns, not multi-select. Stat labels have no brackets.
- Weeks (Projections, Fixtures) and Player ratings columns are hidden with a × above the header; no from/to horizon
  selects, no checkbox list. No subtitles above the Projections and Player ratings tables; nothing in the navbar
  but the five pages.
- Projections: values centred, the week's highest points among the rows shown in bold (captain pick).
- Player ratings labels: Minutes, Penalty share, NPG, Goals, Assists, Goals + assists, CS %, Defcon %.
- Team ratings: rank 1–20 first, Overall the right-most column; predicted final table: rank, Team, Predicted
  points only.
- (§220) Projections' Clean sheet % is the player's: P60 × the team's chance that gameweek (`CS_Prob`), not the
  team's. Tables use the full width. Fixtures' Team column shows full names.
- (§221) Fixtures rate the opposition's strength as the fixture-ticker images do (an average team's goals v that
  opponent at that venue, `fixture_ticker.build_difficulty`), not the team's own forecast goals; this replaces
  "Values come from fixture_goal_predictions.csv" above (it still supplies the schedule). Search sits first in the
  filter rows; Team ratings has no subtitles and its predicted final table is not sortable.
- (§222) Projections has no xMins column (replaces "xMins always the first stat column"); minutes remain the
  Minutes stat. Pos, Team and Price widen with the space, about 0.6 of a gameweek column.
- (§223) Bold marks each week's captain over all players (shown only when he is on screen). A rank column leads
  Projections and Player ratings. Any change of stat, filter or weeks re-sorts Projections by Total. Hidden weeks
  and columns come back one at a time (a chip each) or all together.
- (§224) No Pos column while one position is selected (Projections and Player ratings).
- (§225) No Team column while one team is selected (Projections and Player ratings).
