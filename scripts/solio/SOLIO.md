# Solio baselines — weekly capture

Solio (https://fpl.solioanalytics.com/, free tier) is the weekly benchmark since 2026-11-13, by user decision
(CHANGELOG §236). It replaced the FPL Review pastes for comparison; `minutes.csv` and the penalty share still
come from FPL Review. Benchmark only, never a model input or target.

## Each week, before part 2

1. **The user** signs in to Solio in Chrome (Google sign-in, never done by Claude) and opens
   Projections → view dropdown → **Baselines**. Check the Points view's first column is the coming gameweek.
2. **Claude**, in that tab (Claude in Chrome), runs the snippet below with `javascript_tool`. It checks every
   row's name against its source record and puts the CSV on the clipboard.
3. Save it from PowerShell as UTF-8 (a `>` redirect writes UTF-16), into the gameweek's folder:

   ```powershell
   $gw = "GW12"; $dir = "data/raw/solio/2026/$gw"; New-Item -ItemType Directory -Force $dir | Out-Null
   [IO.File]::WriteAllText((Resolve-Path $dir).Path + "\solio_baselines.csv", (Get-Clipboard -Raw), (New-Object Text.UTF8Encoding $false))
   ```

4. Write `capture.json` beside it: `{"captured_utc": "<now, ISO UTC>", "first_gameweek_shown": <N>, "view": "Projections > Baselines"}`.
5. Run `python scripts/solio/clean_solio_baselines.py` (part 2 also runs it) and check it prints
   `usable_for_comparison: True`.

Recapturing overwrites the two raw files; each part-2 snapshot keeps the copy it ran on under `solio/raw/`.

## Why a snippet and not copy-paste

The table is a Glide Data Grid drawn on a canvas, so there is no HTML table to copy, and Solio syncs its data
through Rocicorp Zero, so there is no JSON request to save. The snippet reads the grid's own cell function
(full precision) through React's fiber tree. `javascript_tool` truncates output at about 1,000 characters, hence
the clipboard. If Solio ships a new build and the fiber depths (5 and 14) or hook index (17) move, the snippet
stops with an error rather than writing wrong rows: find the DataEditor props (`getCellContent`, `columns`,
`rows`) and the parent with a 566-ish `data` array, and the hook holding the display order.

```js
const el = document.querySelector('.dvn-scroller');
let f = el[Object.keys(el).find(k => k.startsWith('__reactFiber'))], g = f;
for (let i = 0; i < 5; i++) g = g.return;
for (let i = 0; i < 14; i++) f = f.return;
const grid = g.memoizedProps, data = f.memoizedProps.data;
if (typeof grid.getCellContent !== 'function' || !Array.isArray(data)) throw new Error('grid or data not found: fiber depths moved');
let h = f.memoizedState; for (let i = 0; i < 17; i++) h = h.next;
const order = h.memoizedState[0];
const cols = grid.columns.map(c => c.title);
const want = ['Player','Pts/90','Mins/M','G/90','A/90','G%','A%','CS%','Bon/90','DC/90'];
if (JSON.stringify(cols) !== JSON.stringify(want)) throw new Error('not the Baselines view, or columns changed: ' + cols);
if (order.length !== grid.rows) throw new Error('display order hook moved');
const rows = [];
for (let r = 0; r < grid.rows; r++) {
  const e = data[order[r]].element;
  if (grid.getCellContent([0, r]).data !== e.name) throw new Error('row ' + r + ' name mismatch: order hook moved');
  const vals = cols.slice(1).map((c, i) => +Number(grid.getCellContent([i + 1, r]).data).toPrecision(7));
  const name = /[,"]/.test(e.name) ? '"' + e.name.replace(/"/g, '""') + '"' : e.name;
  rows.push([e.fplId, e.fplCode, name, e.position, e.teamId, e.nowCost, e.status, ...vals].join(','));
}
const hdr = 'fpl_id,fpl_code,name,position,solio_team_id,now_cost,status,pts90,mins_per_match,g90,a90,g_pct,a_pct,cs_pct,bon90,dc90';
await navigator.clipboard.writeText(hdr + '\n' + rows.join('\n') + '\n');
'copied ' + rows.length + ' rows'
```

## What the columns are (checked on GW11)

| Column | Meaning | Compare with |
|---|---|---|
| `pts90` | expected FPL points per 90 | our points per 90 |
| `mins_per_match` | expected minutes per match | `minutes.csv` xMins |
| `g90` | goals per 90, **penalties included** | our npxG rating + penalty goals per 90 |
| `a90` | FPL assists per 90, **FPL-only assists included** | our xAG rating + FPL-only assist rate |
| `g_pct` | share of the team's **non-penalty** goals (sums to 1 per team) | NP goal share |
| `a_pct` | share of the team's FPL assists (sums to 1 per team) | assist share |
| `cs_pct` | the team's clean-sheet probability, one value per team; follows fixtures (GW11: closest to our mean over the first five rounds) | `CS_Prob_Team` over that window |
| `bon90` | bonus points per 90 | our bonus per 90 |
| `dc90` | defensive contributions per 90 (a count) | `Defcon_Mu90`, not the threshold probability |

Solio zeroes `g90`, `a90`, `bon90` and `dc90` (blank on screen) below about 18 expected minutes; the cleaner
blanks them and reads the cut-off from the data each week.

Identities (GW11): `g90 × mins / 90 / g_pct` is the same for every player of a team (Solio's team non-penalty
goals per match) except its penalty takers, who sit above it by their penalty goals; `a90 × mins / 90 / a_pct`
is constant to 0.1% (team assists per match). `dc90` looks like a historical average (Haaland 3.078, our
`AvgPer90` 3.0780). Goal rates are fixture-neutral; CS% is not.

## Comparing

`python scripts/solio/compare_solio.py` (latest pre-deadline snapshot with usable Solio data), `--current`, or
`--rounds N`. Writes `data/solio/comparisons/<snapshot>/`: players, summary by position and stat, teams, top
differences. Descriptive only; never calibrate toward Solio.
