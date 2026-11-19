# Sofascore supplementary statistics

## Weekly collection (browser downloads, CHANGELOG §66)

Ordinary requests are refused, so each round is captured in Chrome and saved as
downloads, then imported offline by `sofascore_weekly.py`. It never makes HTTP requests.

**When:** after the gameweek, at least ~2 days after its last match and before
`run_pre_deadline.py`. Stats were unchanged between ~1.5 and ~4.5 days after kickoff
(`data/validation/sofascore_revision_check_20261028/`); earlier revisions are untested.

**Routine:**
1. `python -u -X utf8 -m fpl_model.fetch.sofascore_weekly plan` lists the round pages
   (and any known lineups) still needed for completed FPL fixtures.
2. Ask Claude to "collect Sofascore round N". In Chrome it opens
   `/api/v1/unique-tournament/17/season/{season_id}/events/round/N`, downloads a capture,
   reads the finished event IDs from that page, then opens each `/api/v1/event/{id}/lineups`
   and downloads it. Navigation starts are at least 6.1 s apart. Each capture is
   `sofascore_capture_<kind>_<id>_<UTC>.json` in Downloads, holding the URL, retrieval time
   and the page's exact `document.body.innerText`. Check that all files have arrived before
   importing; a download can still be in progress, or Chrome may have blocked it.
3. `sofascore_weekly.py import` validates the round pages first (refreshing current-season
   discovery, with old copies kept as `.revision.` files), then lineups for discovered
   finished events, into the frozen raw cache. Valid files move to
   `data/raw/sofascore/weekly/imported/<UTC>/` and the DOM text to `weekly/dom/`.
   Invalid files stay in Downloads and are listed. Re-capturing a cached match is a
   no-op (`lineup_unchanged`) when only follower counts (`userCount`) differ; any other
   change is refused (`cache_revision`) and never overwrites the cache.
4. Run `run_pre_deadline.py`. Its first stage, `sofascore_weekly.py pipeline`, does steps 3
   and `build` itself. It imports from Downloads and rebuilds only when cached lineups are
   missing from the output (about 70 s; otherwise skipped). It writes
   `data/current/sourced/sofascore_status.json` and never halts: import or build problems are recorded
   there as warnings or errors. Model stages may use Sofascore only when `usable_for_model`
   is true: no errors, every cached lineup is built, and the cache is at most 2 rounds behind
   FPL. Otherwise they keep the §64 baseline. Steps 3 and `build` can still be run by hand.

**Manual fallback (without Claude):** open each URL from `plan`, save the page with
Ctrl+S into Downloads under the `manual_filename` that `plan` prints
(`sofascore__round__<n>.json`, `sofascore__lineups__<event>.json`), then run step 4.
For a new round, import the round page first and rerun `plan` to get its lineup URLs.

**Chrome setting:** a site's second automatic download is blocked until it is allowed.
In Chrome, allow *Automatic downloads* for `www.sofascore.com` (Site settings), or click
Allow when Chrome asks.

Tests: `python -u -X utf8 -m unittest discover -s fpl_model/fetch -p test_sofascore_weekly.py -v`.

**Full history completed and published (2026-10-27, CHANGELOG §62).**
All 3,500 source-discovered completed EPL fixtures are cached under
`data/raw/sofascore/`: 380 per Season 2017–2025 and 80 for 2026, with the current
season rechecked at the end. Thirty validated pilot imports retain original retrieval,
transport, payload and DOM provenance; 3,470 new lineups were collected through normal
connected Chrome navigation. All seasons reconcile exactly against independent local
fixture teams/dates. Missing/deferred/unavailable/discovery gaps: zero.

Published `data/current/sourced/sofascore_player_match_stats.csv` (135,136 rows) and
`data/archive/current_unused_20261118/sofascore_field_coverage.csv`, plus compatible additions to the shared
player/alias/fixture registries. The identity registries and overrides are stored in
`data/archive/current_unused_20261118/`. All original registry rows and accepted IDs remain unchanged;
Every source row has project IDs. After the §63 identity repair (matcher fix plus 18
reviewed overrides), every played appearance is resolved. 544 source-only IDs / 2,664
rows remain, all unused substitutes. Event 8243397 retains
its reviewed legacy source-only fixture ID; it was not merged into a different ID.

Evidence: `data/validation/sofascore_full_backfill_20261026/REPORT.md`,
`completion_audit.json`, `data_audit.json`, `publication.json`,
`published_registry_compatibility.json`, `rebuild_idempotence.json` and
`preservation_final.json`. All 34 focused tests passed. Fresh-collector restart and
all-3,500-fixture zero-navigation rerun passed; two settled all-cache builds produced
identical bytes for all six outputs, with all raw hashes preserved.

The offline coordinator `sofascore_browser_backfill.py` supports explicit `--raw`,
`--data`, `--run`, `--pilot` paths and prepare/import-pilot/sync/plan/checkpoint/build/
reconcile/audit/normalize-fixture-aliases actions. `plan --refresh-current` refreshes
current discovery without refetching cached lineups. `checkpoint.complete` describes
source collection; the run's completion audit also verifies derived output and publication.
The saved `collector_functions.js` uses the supported browser tool, bounded batches,
atomic DOM saves, cache skipping and at most two deferred retry passes. Keep imports
paused until any live audit process has actually exited. Exact recovery/reproduction
instructions are in the run README. No unattended browser runner exists.

Actual attempts: 3,901 endpoint navigations (3,470 lineups, 431 metadata), no failed
source responses or repeated lineup attempts. HTTP status/headers were not exposed and
remain unknown. Three timer starts were recorded at 5.999 seconds; a rechecked
6.1-second target was subsequently verified on all 655 remaining navigations. Full
technical collection success does not establish unattended access or bulk permission.
Do not repeat ordinary-requests probes that previously returned 403, inject cookies or
headers, switch hosts/proxies, or bypass a challenge.

Coverage is field-specific: xG/xA are absent in 2017–2021; key passes and some other
fields are sparse. Preserve nulls and all raw fields, including signed goalsPrevented
and progression/value metrics. Keep native source xG/xA and total/won tackles separate;
accurate crosses are not proven open-play crosses. There are 100,669 positive-minute
rows and 34,467 bench rows with unknown minutes, not confirmed unused status. The
31-player pilot comparison's nine minute differences are retained; FPL minutes remain
authoritative for FPL scoring.

The completed pilot evidence is immutable under
`data/validation/sofascore_browser_pilot_20261026/`; its 1,152-row result and 26-test
counts are historical. Its fixed-30 `sync` is not the full-history coordinator.
Understat archives/positions, existing production player_match_stats, model/pipeline
code, manual penalty log, forecasts and frozen studies remain unchanged. Weekly wiring,
incidents, shot/heat maps, paid sources and model/stat repairs were not performed.

The following pilot implementation details remain historical reference.

## Browser pilot collection/import path

The original Python network client supports requests only. The new
`sofascore_browser_pilot.py` offers offline `prepare`, `sync`, `build` and
`verify-preservation` actions with an explicit `--pilot` path. It uses the existing
Client validator/store/builder and a shadow registry. **No unattended browser runner
exists.** Supported Chrome navigation remains tool-driven; actual collector functions
and bounded resume instructions are saved in the pilot directory. Do not present the
ordinary requests commands below as browser collection.
The successful read/save method was supported browser tooling with normal URL navigation,
read-only DOM `document.body.innerText`, JSON parsing and local file saving. A
small resumable collector/import adapter around that supported path was implemented,
with the explicit 30-event selection, six-second spacing, checkpoint/queue and cache
skipping. Existing and new focused recovery tests passed before the batch; the final
26-test suite adds a regression preserving signed extra `goalsPrevented` statistics.

Use `Client(raw=pilot_raw)`, `Registry(data=shadow_current)` and
`build(client, registry, data=shadow_current)` for an isolated pilot; copy required existing
identity/local-history inputs into the shadow directory. Record retrieval URL/time,
transport and original DOM text provenance; HTTP status is unknown unless actually exposed.
Do not invent fixture metadata for old saved samples: acquire supporting real event
metadata before importing them. Technical browser success does not establish sustained
bulk permission; record that separately and do not fabricate the access-evidence file.
The controlled restart, final zero-fetch rerun, 31-player real-payload comparison and
preservation checks are now complete. Its historical stop-at-pilot boundary does not cap
the next full-history task. Missing fields remain
missing; total/won tackles and source xG/xA remain separate. The two new source aliases
`Liverpool FC` and `West Ham United` map to local team names. Source-stat/model logic is
unchanged; settled builds preserve raw/registry/derived bytes. The 31-player audit agrees
31/31 for CBI/recoveries/saves/goals/assists/total tackles, 24/31 for won tackles and
22/31 for minutes. FPL minutes remain authoritative. Initial seasons preflight timing
was not instrumented; status/headers remain unknown. No access-evidence flag was created.

```powershell
python -u -X utf8 -m fpl_model.fetch.sofascore_browser_pilot sync
python -u -X utf8 -m fpl_model.fetch.sofascore_browser_pilot build
python -u -X utf8 -m unittest discover -s fpl_model/fetch -p test_sofascore_browser.py -v
python -u -X utf8 data/validation/sofascore_browser_pilot_20261026/validate_pilot.py
```

## Existing Python commands and bulk access gate (not a browser runner)

```powershell
python -u -X utf8 -m fpl_model.fetch.fetch_sofascore --probe
python -u -X utf8 -m fpl_model.fetch.player_identity
python -u -X utf8 -m fpl_model.fetch.fetch_sofascore --build-only
python -u -X utf8 -m unittest discover -s fpl_model/fetch -p test_sofascore.py -v
```

Do not repeatedly probe unchanged HTTP denial; `--probe` above uses requests, not Chrome.
Bulk requests require an access-evidence
JSON file with `permitted_method`, `permission_reference` and
`unattended_pilot_passed: true`. This records external evidence; setting the flag does
not establish permission or fix 403. The current client supports ordinary requests only.
No browser cookie harvesting, challenge bypass, proxies or alternate hosts are implemented.
The official FAQ offers no public data API; the earlier main-site terms review documented
extraction restrictions. Sustained bulk access remains unestablished; this does not negate
the successful bounded browser pilot or authorize a full backfill.

Deferred follow-up commands, only after separately established permitted access:

```powershell
python -u -X utf8 -m fpl_model.fetch.fetch_sofascore --access-evidence <actual-evidence.json> --season 2017 --limit 4
python -u -X utf8 -m fpl_model.fetch.fetch_sofascore --access-evidence <actual-evidence.json>
python -u -X utf8 -m fpl_model.fetch.fetch_sofascore --weekly --access-evidence <actual-evidence.json>
```

`--season` can repeat. `--limit` caps new successful lineup downloads, including repairs,
not cached reads. Default spacing is six seconds, a conservative configuration, not a
published source rate limit. Up to four attempts/request, two deferred match passes,
and three consecutive systemic failures per discovery/download phase stop identical
failure cascades. 401/403 and non-JSON challenges are not retried for that request;
404/410 or empty lineups remain unavailable and can be revisited next run. Retry-After
is honored for 429. Transient transport/408/5xx errors use bounded exponential backoff.

Standalone exit 2 means incomplete, including build-only on incomplete history; exit 1
means integrity/schema failure. Weekly returns 0 for access/incomplete outcomes with an
explicit INCOMPLETE manifest, but integrity failures still fail. `run_post_gameweek.py`
is deliberately **not wired yet**: AGENDA requires the real standalone pilot to pass first.
The eventual insertion point is after the Understat position crosswalk, retaining all
existing stages. The CLI weekly path is tested offline; live increment remains unverified.

## Cache and output contracts

`data/raw/sofascore/matches/<season>/<event>.json` contains the complete unmodified
decoded response as `payload`, canonical-JSON SHA256, source URL, retrieval timestamp,
event metadata, event ID and season. Payload keys/values are preserved; hashes describe
canonical JSON, not the original wire bytes. Supporting rounds/events metadata and
season discovery have the same envelope. Atomic replacement prevents truncated writes.
Successful match responses are frozen. Explicit metadata refresh retains dated prior
versions; invalid cached files are quarantined with an `.invalid.<timestamp>` suffix.
Build-only surfaces invalid cache and does not overwrite good enrichment output. A network
resume can repair a quarantined match. Current-season discovery refreshes each run,
including weekly; successful historical discovery is frozen unless `--refresh-metadata`.
Pagination follows `hasNextPage`; unexpected schema fails visibly. These paths are mocked,
not validated against live Sofascore enumeration yet.

`manifest.json` separates expected/cached/missing, deferred/unavailable, discovery failures,
and known-local counts by season. Unknown expected/missing counts are null. `queue.json`
persists failed requests. All-cache rebuilds preserve earlier seasons. Empty rebuilds
retain previous good player-stat output. `access_probe.json` stores request status, body
hash, timing and errors, never fabricated successful responses.

`data/current/sourced/sofascore_player_match_stats.csv` is emitted only when actual valid cached
matches exist. It retains original source IDs/names, project IDs, local names when known,
fixture home/away membership, kickoff, starter flag, played/unused/unknown bench state,
coarse source position/formation and raw provenance. All numeric exports are prefixed
`Sofascore_` and preserve native field names. `Present_Fields` records original keys;
`sofascore_field_coverage.csv` reports nonnull presence by season and appearance state.
Missing values remain missing. An absent bench minutes field gives `bench_unknown`;
only explicit zero plus substitute status establishes `unused_bench`.

`totalTackle` and `wonTackle` are distinct. `accurateCross` is not validated open-play
crosses. No dedicated big-chance-save derivation exists. FPL minutes remain authoritative
for FPL scoring. Source xG/xA is separate. No existing player-stat/model/forecast column
is filled or changed. No incidents, shot maps or penalty-log automation are included.

## Identity schemas

Identity tables are UTF-8-BOM CSV under `data/archive/current_unused_20261118/`, written atomically, with IDs as
strings. The published Sofascore player-match table is in `data/current/sourced/`.

| Table | Columns and invariant |
|---|---|
| `player_identity_registry.csv` | `Canonical_Player_ID,Display_Name`; deterministic immutable project identity, never a guessed FPL code |
| `player_identity_aliases.csv` | `Provider,Provider_Player_ID,Season,Team,Name,Canonical_Player_ID,Evidence`; each provider ID maps to one player; legacy PMS names key on season/team/name |
| `fixture_identity_registry.csv` | `Provider,Event_ID,Canonical_Fixture_ID,Season,Home,Away,Kickoff`; provider/event unique; cross-provider candidates require same season/home/away and date within one day |
| `identity_overrides.csv` | `Provider,Provider_Player_ID,Canonical_Player_ID,Reason`; reviewed source-only mappings can be resolved; conflicting accepted identities fail and require an explicit audited migration |

Seed preserves accepted `player_match_positions.csv` and Understat crosswalk assignments,
all Understat provider IDs, historical fixture context, and current web names only with
confirmed season/team context. Current inputs contain no stable FPL IDs; none are invented.
Accepted provider IDs are reused before name matching. New and source-only IDs use
one-to-one roster assignment (CHANGELOG §63): only players with positive minutes may
claim a local row, local and source minutes must agree within 10, the name score must
be >=.90 (token containment such as Alisson / Alisson Becker scores .95), and the best
assignment must beat the best assignment without that pair by >=.10. An identity
already held by another ID from the same provider is never taken. Ambiguity remains
`unresolved_cross_provider`, with its own durable source-only project ID and provider
alias, but is retried in every later fixture rather than frozen at its first failure
(an unused substitute has no local row). A later match promotes all of that ID's alias
rows (`promoted_from_source_only:`) and the build re-points its earlier rows. Reviewed
overrides still connect names that cannot match; unresolved history is never discarded. Transfers use the fixture side.
Fixture ID reuse preserves a reschedule; a repeated distinct event on a different date
gets its own fixture identity instead of joining on gameweek or local PMS MatchID.

Understat now consults the same registry for accepted contextual identities and matches
only unresolved rows. Its position mapping/substitute resolution is unchanged. Full
current-history before/after validation produced byte-identical position and crosswalk
outputs (99,920 player-match rows). Production artifacts were not regenerated.

## Completed history and later weekly work

The full-history discovery, collection, source comparison, restart, whole-cache audit,
settled rebuild, preservation and publication gates have passed (section 62). The
35-season listing is discovery evidence; only Seasons 2017-2026 were in this task.
The immutable pilot remains evidence for its original bounded trial. Unattended access
and bulk permission are not established by technical tool-driven success. Weekly
enrichment, late-lineup automation and model integration remain separate later tasks.
