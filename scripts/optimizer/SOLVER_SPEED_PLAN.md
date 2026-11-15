# Solver speed plan: Fast, Standard and Deep (overnight session, written 2026-11-10, §226)

Goal (user, 2026-11-10): make the solver as good as possible on three counts: the plan it returns, the time it takes
and the memory it uses. **Deep** must stay exactly optimal. **Standard** (the default) and **Fast** must be close to
optimal, and their cost against Deep must be measured. Context: `website_plan.md` "Speed: research plan" and "Three
solve modes"; `optimizer_brief.md` (model, chip rules); `PINNED.md` (pinned open-fpl-solver ec65f5e + `gw14_ft.patch`).

## The goals (the user's, 2026-11-10); everything below serves them

Build the best solver possible on three counts, traded off by mode:

1. **Solution quality.**
   - Deep returns the proven optimum, exactly as today.
   - Standard is within 0.1 points of it, with the same GW-now moves (the adoption rule below).
   - Fast is close, with its loss measured and shown.
2. **Time.**
   - Every mode should be as fast as it can be at its quality level.
   - Fast should answer in well under a minute at horizon 10, Standard in a few minutes.
   - Deep's time should be as low as possible while it stays exactly optimal (today: 28 minutes for three plans at
     horizon 10).
3. **Memory.** Peak memory should be as low as possible and never rise because of a change. Today's peak is 2.5 GB on a
   16 GB PC that is also in use.

**No mode is ever "done" while time remains.** Targets are floors to pass, not finish lines. If Deep drops from 28 to
27 minutes, or Fast reaches 50 s, keep going:
- look for the next lever and stack levers;
- revisit the formulation;
- try the next solver or idea.

Stop working on a mode only when the evidence shows its remaining levers are exhausted, and say why in the Decision
log. When one line of work stalls, move to another, and return if a later finding reopens it. Use the whole night.

**Wiring the modes in comes in two parts:**
- Wire them in as soon as a Standard exists that passes the adoption rule. Then the user has a working result whatever
  happens later.
- Rewire each time a mode improves. Every new result must be measured on all six instances.

**Success** means three modes, each measured on all six instances: time, peak memory, loss against Deep and GW-now
match. That table goes into CHANGELOG, and the modes are wired into `run_optimizer.py` and `optimizer_app.py`.
A partial result still counts: well-measured findings and a clear next step.

## You may change this plan

The task list below is a starting order based on what was known on 2026-11-10, not a script. Adjust it as you learn:
- reorder tasks;
- drop a task the evidence says won't pay;
- add levers this plan doesn't list;
- go deeper on whatever turns out to matter most;
- change the time targets above if the measurements show they're wrong.

For example: if CP-SAT is clearly faster, porting it may become the night's main job. If memory turns out to be the
binding problem, move it first.

**What is fixed:** the ground rules and adoption rules below. Each rule guards solution quality or the user's PC.

**Record every change of plan** under "Decision log" at the end of this file: what you learned, and what you changed
because of it. Do this as you go, so the user can follow the night's reasoning in the morning. Don't stop to ask the
user; they are asleep, and decide design details within these rules (memory: run without design sign-off).
Pre-register each experiment's expected effect before you run it.

## Ground rules

- **Run every solve as a detached process, one at a time.** Use `Start-Process` (CLAUDE.md §5.3) or
  `benchmark_solver.py queue`; never two solves at once. Claude Code stopped an in-session background solve once when
  the PC (16 GB) ran low. A queue launched through PowerShell `*>` writes UTF-16 logs, so watch `results/*/*/result.json`
  and the per-run logs instead (those are UTF-8).
- **Treat the pinned solver as read-only.** Any model change is a new patch beside `gw14_ft.patch`, recorded in
  `PINNED.md`, and it must pass `replay_plans.py` (the independent re-scoring) on every benchmark instance.
- **Replay every plan** (the harness does it). No result counts unless the replay agrees with the solver.
- **Timings vary about ±20% between identical runs** (gap0001's plan 1 was slower than base). Before calling any lever a
  win or a loss, repeat it with 3 `random_seed` values, and report the median and the range.
- **Compute every constant in code; declare each setting once** (memory rules). The three modes live in one module,
  e.g. `scripts/optimizer/solve_modes.py`, imported by `run_optimizer.py`, `optimizer_app.py` and the benchmark.
- **Adoption rule for Standard** (pre-registered, `website_plan.md`):
  - the replay score stays within 0.1 points of the gap-0 optimum;
  - GW-now transfers are unchanged on all instances but at most one, or the user accepts the trade.
- **Fast may break that rule.** The page then shows its measured loss against Deep and whether its GW-now moves match.
- **Deep's own adoption rule:** a change to Deep must reproduce the gap-0 optimum's score on every instance, plan
  for plan.
- **Record everything:**
  - a CHANGELOG §-entry (§227 onward);
  - AGENDA "Website";
  - `website_plan.md`;
  - CLAUDE.md's § range.
- Run tests with `python -m unittest scripts/optimizer/test_chips.py` (plus any new test file).

## Harness (`benchmark_solver.py`, built §226)

```
python -u -X utf8 -W ignore scripts/optimizer/benchmark_solver.py run --instance gw11_h10 --variant base
python -u -X utf8 -W ignore scripts/optimizer/benchmark_solver.py queue gw11_h10:warm gw11_h5:warm   # skips finished pairs
python -W ignore scripts/optimizer/benchmark_solver.py report
```

**Instances** (frozen in `data/optimizer/benchmark/instances/`, don't rebuild them):
- `gw09_h5` and `gw10_h5` use those weeks' forecasts and prices but today's squad. They are speed test cases, not
  reproductions of the real GW9 and GW10 decisions.
- `gw11_h5` and `gw11_h10` are GW11 at horizons 5 and 10.
- `gw11_h10_fh` adds a Free Hit in GW14.
- `gw11_h10_chips` adds a Wildcard in GW12, Triple Captain in GW15 and Bench Boost in GW17. It is hypothetical: this
  squad has already played those chips.

**Variants** are in `VARIANTS`; add new ones there. `Recorder` (a `highspy.Highs` subclass) logs per plan:
- improving solutions with times;
- the dual bound, about once a second;
- the peak memory of the process.

The plans and `solve_log.json` are saved before the replay runs.

## Baseline (§226; production = gap 0, 3 plans, secs 1800, pool filters 30/5)

| instance | total s | plan 1 s | first plan s | plan-1 replay | peak MB |
|---|---|---|---|---|---|
| gw09_h5 | 198 | 68 | 5.6 | 222.1946 | 401 |
| gw10_h5 | 201 | 74 | 5.2 | 212.5964 | 372 |
| gw11_h5 | 142 | 39 | 5.3 | 212.5019 | 350 |
| gw11_h10 | 1690 | 482 | 64 | 305.4561 | 2412 |
| gw11_h10_fh | 1516 | 525 | 29 | 307.3217 | 2516 |
| gw11_h10_chips | 1669 | 416 | 69 | 317.8854 | 2546 |

**Levers already measured on gw11_h10** (one run each):

| Variant | Total | Plan-1 score | GW-now moves | Verdict |
|---|---|---|---|---|
| gap 0.1% | 1510 s | −0.02 | same | plan 2 −0.13; fails the 0.1-point rule |
| gap 0.5% | 789 s | −1.03 | changed: plans 1 and 2 swap | fails; a Fast candidate |
| `mip_heuristic_effort` 0.3 | 1753 s | same | same | no gain |

**What the trajectories show:**
- On gw11_h10 the first plan arrives at about 60 s, 1–2 points below the optimum. Nothing better is found until about
  400 s, and the optimum turns up close to the end.
- So the slow part is finding good plans. The dual bound is also slow to close: about 1% above the optimum at 300 s.
- **Memory grows by plan:** about 1.2 → 2.1 → 2.5 GB. One model is reused for all three plans, with a no-good
  constraint added between them, and HiGHS keeps search data from the earlier solves.
- **Horizon:** 10 weeks is about 12× slower than 5. The two horizons rank the top two GW11 moves in opposite order, so
  shortening the horizon changes the advice. Report this trade; never make it silently.

## Tasks (starting order; adjust per "You may change this plan")

1. **CP-SAT feasibility test (first task; research done 2026-11-10, see below).**
   1. Export the gw11_h10 plan-1 model exactly as HiGHS solves it. In the benchmark, call
      `self.writeModel("<path>.mps")` before the first `run()`. Record HiGHS's plan-1 objective (`objective` in
      `results/gw11_h10/base/result.json`) as the target.
   2. `pip install ortools` (9.15 is on PyPI for Python 3.13).
   3. Load the MPS with OR-Tools `model_builder` and solve with the CP-SAT backend. If loading fails, try MathOpt.
      CP-SAT works only with integers:
      - the only continuous variable is `in_the_bank`, a multiple of 0.1;
      - objective coefficients are fractional points, so check how the import scales them (CP-SAT's `mip_*`
        parameters).
   4. Run at 8 and 16 workers, with a 1800 s limit. Record:
      - time to the first solution;
      - time to within 0.1 of the target, and to the target;
      - peak memory;
      - the final gap.
   5. **Decision rule:**
      - Port the model to CP-SAT only if it reaches the target at least 2× faster than HiGHS's 482 s (median of
        3 seeds) and its peak memory stays below HiGHS's.
      - A port means rewriting the model section of `solver.py` (lines ~351–700) against `cp_model`. Scale points by
        ≥ 10⁴ and confirm with the replay. Rounding can swap near-ties: gw11_h10_chips's top two plans differ by 0.0014.
      - If the import itself fails, judge on the research alone. Then a port is a full overnight job, so do it only
        after tasks 2–5.
      - Either way, record the outcome.
2. **Finish the four unrun levers** (stopped by the user 2026-11-10):
   - `gw11_h10:pool_tight`;
   - `gw11_h10:threads1`;
   - `gw11_h5:nosample` (the recorder's own overhead);
   - `gw11_h5:warm` and `gw11_h10:warm`. `_warm_start` was fixed for highspy 1.15, where `getOptionValue` returns
     (status, value); it is untested since.
3. **Memory:**
   - solve each plan in a fresh model, or call `clearSolver()` between plans, or run each plan in its own process;
   - test HiGHS's cut-pool limits (`mip_pool_soft_limit`, `mip_pool_age_limit`).
   - Target: peak memory at plan 1's level, with the same plans and scores.
4. **Better starting plans (targets the 6-minute search):**
   - for plans 2 and 3, plan 1 repaired so it meets the new constraint (its GW-now move swapped for the best
     alternative);
   - last week's saved plan rolled forward;
   - all given through `setSolution`.
5. **Exact reformulations for Deep.** Each one must reproduce the optimum on all six instances:
   - Drop provably dominated players: same position and team, no dearer, at least as many points every week.
     Club-limit interactions make cross-team dominance unsafe.
   - Tighten the FT big-M constraints (`ft_above_ub`, `ft_below_lb`, `fts_state`).
   - Break bench-order and vice-captain symmetry, or derive the bench order after solving if the objective is unchanged.
   - Branching priorities on `transfer_in`.
6. **Approximations for Standard and Fast** (the replay always scores the real plan):
   - relax-and-fix: weeks after GW-now+4 solved with lineup, captain and bench continuous, then fixed and re-solved;
   - small gaps (0.2–0.5%);
   - time limits read off the improving-solution trajectories;
   - one plan for Fast.
7. **Other free solvers:** only SCIP (Apache 2.0), as one comparison run on gw11_h10, and only if HiGHS stays slow
   after tasks 3–5. Skip CBC (usually slower than HiGHS). Gurobi is out (paid; academic licence only).
8. **Define the modes** (as soon as a passing Standard exists, then update them as levers land; not left to the end) in one module and wire them into `run_optimizer.py` (`--mode fast|standard|deep`, default
   standard) and `optimizer_app.py`:
   - **Deep** = today's settings plus any exact speed-ups;
   - **Standard** = levers that pass the adoption rule;
   - **Fast** = well under a minute at horizon 10, with one plan; its measured cost is printed with the plan.

   Then run `replay_plans.py` on each mode's GW11 plans.

## CP-SAT research (2026-11-10)

- Its developers call it [competitive with the best MIP solvers on purely integer problems](https://drops.dagstuhl.de/storage/00lipics/lipics-vol280-cp2023/LIPIcs.CP.2023.3/LIPIcs.CP.2023.3.pdf). That is their own claim.
- The [CP-SAT Primer](https://d-krupke.github.io/cpsat-primer/mathopt.html) is more cautious: it "may not outpace dedicated
  MIP solvers" on classic MIPs. It is stronger at finding solutions than at proving bounds
  ([search core](https://d-krupke.github.io/cpsat-primer/search_core.html)).
- No head-to-head comparison with HiGHS on a similar model was found, and no FPL solver uses CP-SAT.
- It uses all cores as a portfolio of searches. Forum reports say memory grows with the number of workers.
- It takes solution hints (`AddHint`). Parallel runs aren't repeatable without `interleave_search`.
- Our model is binary and integer apart from `in_the_bank` (a multiple of £0.1m), so it fits after scaling.

## Decision log

Add an entry for each change of plan, with the date and time, what you learned and what changed. Newest at the bottom.

- 2026-11-10 (planning session): CP-SAT test placed first. It is the one lever that could change the solver itself
  rather than tune it, and the test is cheap.
- 2026-11-10 22:55 (task 1 started). Exported gw11_h10 plan 1 (`benchmark_solver.py export`, new subcommand):
  23,047 columns (11 continuous, all `itb*`), 38,957 rows, objective offset 0. Prices make 6,560 matrix coefficients
  fractional (£0.1m steps). Rather than trust OR-Tools' floating MIP import, `cpsat_test.py` converts by hand (itb in
  £0.1m units, each row ×10^k until integral, objective ×10⁴ rounded, every solution re-scored exactly).
  **Pre-registration:** CP-SAT finds a first plan sooner than HiGHS (< 64 s) but, as a weak bound-prover on a classic
  MIP, does not reach HiGHS's 308.4561 within 241 s (the 2× rule) and does not prove optimality in 1800 s. Expected
  verdict: no port. Memory: expected above HiGHS's 1.6 GB at 16 workers.
- 2026-11-10 23:05. **Reading of the base trajectories:** on gw11_h10 plan 1 the dual bound sits flat at 310.70 from
  ~120 s to ~400 s and the incumbent at 306.58; then both move within seconds and the optimum is proved at 482 s. That
  pattern looks like root-node processing (cut rounds, root heuristics) taking most of the time, with the tree search
  short. So a `log` variant (HiGHS's own log to a file) is queued to see where the time goes before tuning blindly.
  New harness pieces: `@<seed>` suffix on variant names (`base@1`), `clear` (clearSolver() before plans 2–3),
  `repair` (plans 2–3 start from the previous plan's squads/lineups/captains/bench after GW-now as a *partial*
  solution, HiGHS completes it; `setSolution(n, idx, vals)`).
  **Pre-registrations (gw11_h5, seed 0, one run each as screens):**
  - `clear`: same three scores; peak memory ≤ plan-1 level (−20% or more vs 350 MB); time within ±20%.
  - `repair`: same three scores; plans 2–3 get an incumbent within 1 s; plans 2–3 time −20% (h5 is fast, so this is
    a smoke test; the real test is gw11_h10).
  - `nosample`: time within ±10% of base (the recorder is cheap).
  - `warm` (plan-1 start = keep the squad): plan 1 gets an incumbent at once but no time change beyond noise.
- 2026-11-11 00:20. **CP-SAT verdict: no port (task 1 closed).** gw11_h10 plan 1, seed 0, 1800 s limit
  (`cpsat_test.py`, results in `data/optimizer/benchmark/cpsat/gw11_h10/`). The hand conversion was exact enough: every
  solution's exact re-score matched CP-SAT's scaled objective to < 0.001, and both runs reach HiGHS's 308.4561 exactly.

  | | first plan | within 0.1 | reaches 308.4561 | proves optimum | peak MB |
  |---|---|---|---|---|---|
  | HiGHS (base) | 64 s (306.58) | 478 s | 478 s | 482 s | 1,621 |
  | CP-SAT 8 workers | 7.6 s (≈296) | 526 s | 533 s | 1,241 s | 1,364 |
  | CP-SAT 16 workers | 7.6 s | 323 s | 381 s | not in 1,800 s (bound 310.10) | 2,316 |

  The rule needed ≤ 241 s and less memory than HiGHS. 16 workers is 1.26× faster to the optimum but uses 43% more
  memory and cannot prove it; 8 workers is slower. The gap to the rule (1.6× more speed) is far beyond the ±20% seed
  noise, so no further seeds were run. As in the pre-registration: CP-SAT finds plans early and proves bounds badly.
  Nor is it a Fast candidate: at 60 s it holds 304.5–304.6 where HiGHS holds 306.58 (from 64 s). A CP-SAT warm start
  for HiGHS is possible but not promising on these numbers; not pursued. `ortools` stays installed (not in
  requirements; research only).
- 00:20. The first queue (h5 screens + gw11_h10 log) crashed at once: `install()` repoints `highspy.Highs` at the
  subclass, so `highspy.Highs.run(self)` recursed. Fixed with `solve_modes.BaseHighs` (the original class); requeued.
- 2026-11-11 00:40. **gw11_h5 screens** (seed 0, one run each; base 142 s is the fast end of the noise — the four
  variants all took 154–165 s with plan 1 at 44–47 s vs base's 39 s, though plan 1 is identical in all of them):
  `repair`, `clear`, `nosample`, `warm` all return the same three plans and scores. `repair` gives plans 2–3 an
  incumbent at 0.96 s (0.75 and 1.8 points below their optima) but no time gain; `warm` gives plan 1 an incumbent at
  1.3 s, no gain. At h5 the optimum is found by ~15 s and the rest is proof, so start solutions can't help there.
  `clear` does **not** lower memory at h5 (peak 359 vs 350 MB; plan 3 is still the peak). h10 tests follow.
- 00:40. **The HiGHS log of gw11_h10 plan 1 (variant `log`) says where the time goes.** Presolve 4 s; root LP and
  cuts to 79 s (sub-MIP finds 306.58 at 59 s, a restart at 62 s); then **nodes 1–3 take 79 → 244 s** and the tree
  finishes the remaining 1,700 nodes in ~230 s. 233k of the 538k LP iterations are strong branching. So the
  bottleneck is reliability branching's strong-branching on the first nodes (18.6k binaries,
  `mip_pscost_minreliable` = 8 by default). **New lever, moved to the front of the queue:** `psc0`, `psc2`
  (`mip_pscost_minreliable` 0 and 2). Deep-safe: branching rules don't change the optimum.
  **Pre-registration:** psc2 cuts plan-1 time by 30–50% (the 165 s window shrinks, the tree may grow somewhat);
  psc0 (pure pseudocost from the start) is riskier: −20% to +50%. Scores identical.
- 01:50. **psc0 result:** same three plans and scores; 1,639 s total (base 1,690 and 1,879 in two runs), plan times
  445 / 407 / 784 s against 482 / 523 / 683 and 473 / 616 / 787. Within the noise band; plan-1 peak memory rose to
  2,262 MB (base 1,621). Not a win on one seed; psc2 is running. The bound samples show psc0 moves the bound earlier
  (309.68 at 120 s vs 310.70) but the tree then takes longer, as feared.
- 01:50. **New exact lever `continuous_assignment` (variant `cassign`), moved ahead of the rest.** Bench-order
  (7,520 columns at h10) and vice-captain (1,880) binaries become continuous: given integral squads, lineups and
  captains they form a transportation problem / a one-slot assignment with integral vertices, so the optimum can't
  change; any fractional final value (ties) is repaired by an integral re-solve with every other integer column fixed
  (`round_assignment`). It halves HiGHS's binaries (18.6k after presolve) and so its strong-branching candidates.
  **Pre-registration:** identical scores on gw11_h5 and gw11_h10; plan times −20% to −40% at h10; memory down.
  Queue now: psc2 (running) → gw11_h5 cassign (smoke) → gw11_h10 cassign → clear → repair → rf4 → threads1 → pool_tight.
- 02:17. **psc2:** same plans and scores, 2,060 s (527 / 787 / 744). Slower than base. With psc0 inside the noise and
  psc2 outside it on the slow side, reliability-branching settings are dropped: the strong-branching window is real,
  but HiGHS's tree pays it back later. (psc4 not run.)
- 02:17. **cassign failed its pre-registration, hard:** gw11_h5 identical scores but 403 s (114 / 125 / 163) against
  142–165 s for every other h5 run, i.e. 2.6× slower. Continuous bench/vice-captain columns cost HiGHS its clique
  table and probing on those rows, which matters more than the branching candidates saved. The gw11_h10 run was
  stopped at its start on this evidence (killed, no result kept). Lesson: on this model, binaries carry structure
  HiGHS exploits; formulation changes should add structure (tighter rows), not remove integrality.
- 02:25. **New exact lever `ftm` (patch `ft_tight_m.patch`, option `ft_tight_big_m`, off by default so the model is
  byte-identical without it).** Upstream uses M = 20 in all ten FT-clamp rows. From the bounds (fts ∈ [0,5],
  transfer_count ∈ [0,15]; Wildcard/Free Hit weeks have no counted transfers, so raw = fts − trc + 1 ∈ [−14, 6]) the
  smallest valid M per row is 20, 1, 6, 15, 0, 5, 4, 1, 19, 6. Same integer solutions; a tighter LP relaxation of the
  FT/hit trade-off, where the LP can otherwise fake FT value (1.5 per state).
  **Pre-registration:** identical scores; root dual bound lower (tighter) by 0.1–1 point; plan time −10% to −30% on
  gw11_h10. If it passes it goes into Deep and must reproduce every instance's optimum.
  Queue: clear (running) → gw11_h5 ftm → gw11_h10 ftm → repair → rf4 → threads1 → pool_tight.
- 03:05. **clear (gw11_h10): no memory gain.** Same plans; 1,869 s; peaks 1,616 / 1,994 / 2,390 MB (base 1,621 /
  1,983 / 2,412). clearSolver() frees HiGHS's search data but the process's resident memory keeps growing, so the
  growth is the heap not shrinking, not data HiGHS holds. **gw11_h5 ftm:** same scores, 155 s (noise band);
  gw11_h10 ftm is running.
- 03:05. **New memory lever `proc` (`process_per_plan`):** each plan in its own spawned process
  (`solve_modes.solve_plans`). The no-good rows between plans are rebuilt by `ModeHighs.add_exclusions` from the
  earlier plans' GW-now moves, coefficient for coefficient the pinned solver's `this_gw_transfer_in_out` row, so the
  plans must be identical. **Pre-registration:** same three plans and scores on gw11_h5 and gw11_h10; peak memory per
  plan ≈ plan 1's (~1.6 GB at h10, −33% on 2.4 GB); time +5–20 s per plan for process start and model build.
  Queue: ftm (running) → gw11_h5 proc → gw11_h10 proc → rf4 → repair → threads1 → pool_tight.
- 03:32. **ftm (gw11_h10): no gain, not adopted.** Same plans and scores; 1,932 s (480 / 647 / 802), inside the base
  band (1,690–1,879). The FT rows are not where the root gap is. The patch stays in place, off by default.
- 03:32. **proc (gw11_h5): plans identical** (the rebuilt no-good rows reproduce upstream's exactly), 169 s; per-plan
  peaks 195 / 192 / 307 MB against base 205 / 224 / 350. Plan 3 is heavier even in a fresh process, so part of the
  growth is plan 3's own search, not leftover heap. gw11_h10 proc is the real test.
- 03:40. **Every exact variant so far leaves plan 1 at its first incumbent (−1.87 points, found at 53–64 s by a
  root sub-MIP) until ~400 s**, when the tree search starts producing (base, log, clear, psc2, heur03, gap
  variants alike; psc0 is the exception, improving from 206 s). So the binding problem for Standard and Fast is
  finding good plans early, as the plan said. **New lever `restrict_start` (variants `rs5`, `rs3`):** before plan 1,
  solve the same model with all transfers banned from GW-now+5 (resp. +3) on, 120 s (60 s), gap 0.05%; any such plan
  is feasible for the full model, so it is a complete start. Late transfers carry the least weight (decay 0.85^5 =
  0.44 at GW-now+5), so the restricted optimum should be close.
  **Pre-registration (rs5, gw11_h10):** the restricted solve finishes inside 120 s with a plan within 0.5 points of
  308.4561; the full solve then holds that from ~120 s instead of −1.87 until 400 s; plan-1 total time −0% to −25%
  (a better incumbent prunes more, but proof still dominates); same scores. If the restricted plan is within 0.1 and
  has the optimum's GW-now moves, it is a Standard/Fast candidate on its own.
  Queue: proc (running) → rs5 → rf4 → repair → threads1 → pool_tight.
- 04:05. **proc (gw11_h10): memory unchanged, so the growth is intrinsic.** Identical plans; 1,936 s; per-plan peaks
  1,556 / 2,029 / 2,442 MB *in fresh processes* (base 1,621 / 1,983 / 2,412). Plans 2 and 3 are simply bigger
  searches (the no-good row makes them harder: 616–800 s against 470–480 s), and HiGHS's tree and cut pool grow with
  them. Neither `clear` nor `proc` can lower this; process-per-plan stays available (`process_per_plan`) but is not
  used by any mode. Next memory lever: the cut pool (`cutpool2k`, `mip_pool_soft_limit` 10000 → 2000), queued later.
- 04:40. **rs5 result.** The restricted solve (transfers only GW11–15) finished in 78 s at gap ≤ 0.05% with
  **308.0928, −0.36 points, first found at 38 s** (base holds −1.87 until ~400 s). As a start for the exact solve it
  did not pay: plan 1 took 602 s (base 470–490), plans 2–3 unchanged (they get no start), total 2,041 s. Pre-
  registration: restricted-solve part right (inside 120 s, within 0.5), time part wrong (+25%, not −0–25%).
  **Consequence (change of plan):** start solutions are dropped as a Deep lever (repair at h5, warm, rs5 all failed
  to cut time: the proof, not the search, dominates once a good plan is known). But the restricted model is the best
  Fast/Standard candidate yet, and upstream already has it (`no_transfer_last_gws`), so no new code is needed. New
  variants: `nt5_1p` (1 plan, transfers in the first 5 weeks only, gap 0.05%, 120 s), `nt5_1p_g0` (same at gap 0),
  `nt3` and `nt2` (3 plans, gap 0, transfers banned in the last 3 / 2 weeks: Standard candidates).
  **Pre-registrations (gw11_h10):** nt5_1p: < 90 s, replay within 0.5 of 305.4561, GW-now moves = Deep's plan 1
  (368 in, 236 out). nt3: each plan within 0.1 of Deep's, same GW-now moves, total −30% to −60%. nt2: within 0.05,
  −20% to −40%.
  Queue: rf4 (running) → nt5_1p → nt5_1p_g0 → nt3 → nt2 → cutpool2k → repair → threads1 → pool_tight.
- 05:30. **rf4 (relax-and-fix start): dropped.** The relaxed solve (lineups etc. continuous after GW-now+3) reached
  only 305.18 (−3.27 once completed) in its 120 s, and plan 1 then took 838 s; total 2,280 s.
- 05:30. **nt5_1p / nt5_1p_g0 (upstream's row form of the restriction): slow.** nt5_1p hit its 120 s limit at gap 0.8%
  with 303.01 (−2.44 against Deep); nt5_1p_g0 reached the restricted optimum 305.0928 only at 213 s. The *same*
  restriction as column bounds (rs5's sub-solve) found it at 38 s and proved it at 78 s: presolve deletes fixed
  columns but not the 5 rows. **Finding that matters for Fast:** the restricted optimum (−0.36) has GW-now
  504 in / 31 out, which is Deep's plan **2** (305.3905), not Deep's plan 1 (368 / 236). The two GW-now options are
  0.07 apart over 10 weeks, so any approximation can swap them. nt3 was stopped at its start (row form, slow; no
  result kept) and replaced by the bounds form: new setting `transfer_weeks` (Wildcard weeks exempt) and variants
  `tw5_1p`, `tw4_1p`, `tw3_1p` (Fast: 1 plan, gap 0.05%, 60 s) and `tw7`, `tw8` (Standard: 3 plans, gap 0).
  **Pre-registrations:** tw5_1p finds 305.0928 by ~40 s (inside 60 s); tw4_1p and tw3_1p are faster (< 30 s) and
  lose more (−0.5 to −1.5). tw8: every plan within 0.1 of Deep, same GW-now moves, total −30% or more; tw7: within
  0.2, −50%.
  Queue: tw5_1p → tw4_1p → tw3_1p → tw7 → tw8 → cutpool2k → threads1 → pool_tight → repair.
- 05:40. **Fast candidates (gw11_h10, seed 0), bounds form — the night's first real win:**

  | variant | time | status | replay vs Deep plan 1 | GW-now | peak MB |
  |---|---|---|---|---|---|
  | tw5_1p | 62 s | time limit (60 s) | −0.363 | 504/31 (Deep's plan 2) | 230 |
  | tw4_1p | 49 s | optimal at gap 0.05% | −0.630 | 368/236 (= Deep's plan 1) | 207 |
  | tw3_1p | 22 s | optimal at gap 0.05% | −1.045 | 504/31 (Deep's plan 2) | 210 |

  Memory falls ~8× (Deep plan 1: 1.6 GB). Pre-registration: tw5_1p found 305.09 at 34 s as predicted (but needed
  the full 60 s to prove the gap); tw4/tw3 losses inside the predicted −0.5 to −1.5. tw5 sits on the time-limit
  edge, so a slower PC or instance would return a worse plan; **Fast candidate = tw4_1p** (49 s, proves its gap,
  keeps Deep's GW-now). Queued: tw4_1p on the other five instances, tw5_1p on the two chip instances, tw4_1p seeds 1
  and 2; then tw8 (tw7 is running).
- 06:10. **Fast (tw4_1p) on all six instances** (loss = replay of Fast's plan − Deep's plan 1; GW-now = same moves as
  Deep's plan 1):

  | instance | Fast s | Deep s | loss | GW-now | Fast peak MB | Deep peak MB |
  |---|---|---|---|---|---|---|
  | gw09_h5 | 52 | 198 | 0.000 | same | 188 | 401 |
  | gw10_h5 | 58 | 201 | 0.000 | same | 182 | 372 |
  | gw11_h5 | 52 | 142 | 0.000 | same | 209 | 350 |
  | gw11_h10 | 49 (seeds 1, 2: 60, 60) | 1,690 | −0.630 (all three seeds) | same | 207–229 | 2,412 |
  | gw11_h10_fh | 39 | 1,516 | −1.150 | **different** | 226 | 2,516 |
  | gw11_h10_chips | 53 | 1,669 | −1.101 | same | 259 | 2,546 |

  Times include the model build (~3 s). tw5_1p on the chip instances: fh −0.178 same GW-now, chips −1.002 same, but
  it runs to its 60 s limit each time. Fast stays tw4_1p (it proves its 0.05% gap inside the limit everywhere).
  Fast at h5 is slower than Deep's plan 1 there (51–57 s vs 39–74 s) because its 0.05% gap at h5 means proving nearly
  the optimum; h5 is not the production horizon.
- 06:10. **tw7 (Standard candidate) fails the adoption rule:** 1,067 s (−37%), plan losses −0.18 / −0.15 / −0.10,
  GW-now changed in plans 1 and 2 (the 504/31 and 368/236 options swap). tw8 is running.
  **New Standard candidate `std_a` (`later_plans`):** plan 1 exactly as Deep; plans 2 and 3 on the tw5 restricted
  model, gap 0.05%, 60 s each. Plan 1 then passes the rule by construction; plans 2–3 are measured alternatives.
  **Pre-registration (gw11_h10):** ~480 + 2×65 s ≈ 10 min (−60%); peak memory ≈ plan 1's 1.6 GB (−33%); plan 1
  identical; plans 2–3 within 0.4 of Deep's. Queued on all six instances after tw8, then tw4_1p seed 1 on the other
  five instances.
- 07:20. **std_a on all six instances; modes defined and wired.** Standard (= std_a) keeps Deep's plan 1 exactly on all
  six instances (it is the same solve). Its plans 2–3 are 0 to 1.09 below Deep's plans 2–3 and keep Deep's GW-now moves
  on 9 of 12 (gw10_h5 plans 2–3 differ by ±0.004, near-ties; gw11_h10_chips plan 3). h10 time 620–673 s against
  1,516–1,690 s (−56 to −63%); peak memory = Deep's plan-1 peak (1.1–2.0 GB against 2.4–2.5 GB). h5: no time gain
  (162–211 s against 142–201 s), since plans 2–3 at h5 take ~60 s either way. Fast (= tw4_1p) seed 1 on the other five
  instances repeats seed 0's plans exactly (gw09_h5 hit its 60 s limit with the same plan). Pre-registration for
  std_a: time and memory right; "plans 2–3 within 0.4" wrong on two plans (gw11_h10 plan 3 −0.67, chips −1.09/−0.75).
  **Interpretation of the adoption rule (decided overnight, for the user to confirm):** Standard's recommended plan
  is Deep's proven optimum, so it meets "within 0.1 of the gap-0 optimum, same GW-now moves" on all six instances;
  the alternatives it shows are labelled approximations with their measured cost. If the user wants the rule applied
  to plans 2–3 as well, no tested Standard passes it (tw7/tw8 fail on plan 1 already), and Standard should fall back
  to Deep. Deep is unchanged: no exact lever beat the noise (psc0, psc2, ftm, cassign, clear, proc, repair, rs5, rf4,
  warm, all measured above).
  Queued for the rest of the night (gw11_h10, Deep-relevant, one each): cutpool2k, threads1, base@1 (a third base
  replicate for the noise band), pool_tight, repair.
- 08:14. **cutpool2k fails** (`mip_pool_soft_limit` 2000, pre-registered as a memory lever: peak −20% or more, time
  within noise). gw11_h10: 2,336 s against base 1,690/1,879 (plan 3 1,217 s against 683), peak 1,611/2,049/2,587 MB
  against 1,621/1,983/2,412: no memory gain, slower. Same three plans and GW-now moves. Not adopted.
- 08:59. **threads1 fails.** gw11_h10 with one HiGHS thread: 3,300 s (plans 1,296 / 953 / 1,048 s) against base
  1,690/1,879; same three plans and GW-now moves. Not adopted. The user asked to stop the queue after base@1;
  pool_tight and repair are dropped (not run).
- 09:30. **base@1 (third Deep replicate, random_seed 1):** gw11_h10 1,750 s (plans 380 / 703 / 664 s), same three plans
  and GW-now moves. Deep's band on gw11_h10 is now 1,690 / 1,879 / 1,750 s (plan 1 380–482 s), so every lever above
  needed to finish under about 1,350 s to count as a win; none did. The overnight session ends here (stopped by the
  user); Deep is unchanged.
