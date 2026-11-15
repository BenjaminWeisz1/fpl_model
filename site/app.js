// Personal website (website_plan.md). One page; the navbar switches views. Data comes from site/data/,
// written by part 3 (run_site_export.py). No framework, no build step.
"use strict";

const fetchJson = (path) => fetch(path, { cache: "no-store" }).then((r) => {
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
});

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const norm = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
const store = {
  get(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage off: lasts the session */ } },
};

// ── Navigation ───────────────────────────────────────────────────────────────
function showView() {
  const name = (location.hash || "#projections").slice(1);
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  document.querySelectorAll(".navbar a").forEach((a) => a.classList.toggle("active", a.dataset.view === name));
}
window.addEventListener("hashchange", showView);

// ── Hideable columns (weeks on Projections and Fixtures, stat columns on Player ratings) ────────────────────
// Each hideable header carries a small × above its label. While any are hidden, the restore box lists them as
// chips ("+ GW15"): click one to bring that one back, or "show all". Hidden weeks drop out of Total.
// labelOf(key) names a key; storageKey (optional) remembers the choice in this browser.
function hideable(keys, restoreBox, labelOf, onChange, storageKey) {
  const hidden = new Set(storageKey ? (store.get(storageKey) || []).filter((k) => keys.includes(k)) : []);
  const save = () => { if (storageKey) store.set(storageKey, [...hidden]); };
  function drawRestore() {
    const list = keys.filter((k) => hidden.has(k));
    restoreBox.innerHTML = list.length
      ? `<span class="label">Hidden:</span>` +
        list.map((k) => `<button class="chip restore" data-show="${esc(k)}" title="Show ${esc(labelOf(k))} again">+ ${esc(labelOf(k))}</button>`).join("") +
        (list.length > 1 ? `<button class="link" data-show-all>show all</button>` : "")
      : "";
  }
  restoreBox.addEventListener("click", (e) => {
    const one = e.target.closest("[data-show]");
    if (one) hidden.delete(one.dataset.show);
    else if (e.target.hasAttribute("data-show-all")) hidden.clear();
    else return;
    save(); drawRestore(); onChange();
  });
  drawRestore();
  return {
    shown: () => keys.filter((k) => !hidden.has(k)),
    x: (k, what) => `<button class="x" data-hide="${esc(k)}" title="Hide ${esc(what)}">×</button>`,
    // Call from the table's click handler; true when the click was a ×.
    click(e) {
      const b = e.target.closest("[data-hide]");
      if (!b) return false;
      hidden.add(b.dataset.hide); save(); drawRestore(); onChange();
      return true;
    },
  };
}

// ── Filters: one row of dropdowns, a price slider and a search box ───────────
function selectFilter(select, options, onChange) {
  select.innerHTML = options.map(([v, label]) => `<option value="${esc(v)}">${esc(label)}</option>`).join("");
  select.addEventListener("change", onChange);
  return { has: (value) => select.value === "" || value === select.value, all: () => select.value === "" };
}

const POSITIONS = [["", "All positions"], ["G", "Goalkeepers"], ["D", "Defenders"], ["M", "Midfielders"], ["F", "Forwards"]];
const teamOptions = (teams) => [["", "All teams"], ...teams.map((t) => [t, t])];

function priceSlider(container, min, max, onChange) {
  const step = 0.1;
  container.innerHTML =
    `<span class="val" data-v="lo"></span>` +
    `<span class="dual"><span class="track"></span><span class="fill"></span>` +
    `<input type="range" data-h="lo" min="${min}" max="${max}" step="${step}" value="${min}" aria-label="Minimum price">` +
    `<input type="range" data-h="hi" min="${min}" max="${max}" step="${step}" value="${max}" aria-label="Maximum price"></span>` +
    `<span class="val" data-v="hi"></span>`;
  const lo = container.querySelector('[data-h="lo"]'), hi = container.querySelector('[data-h="hi"]');
  const fill = container.querySelector(".fill");
  function sync(e) {
    if (Number(lo.value) > Number(hi.value)) {
      if (e && e.target === lo) lo.value = hi.value; else hi.value = lo.value;
    }
    const pct = (v) => (max > min ? (v - min) / (max - min) * 100 : 0);
    fill.style.left = `${pct(Number(lo.value))}%`;
    fill.style.width = `${pct(Number(hi.value)) - pct(Number(lo.value))}%`;
    container.querySelector('[data-v="lo"]').textContent = `£${Number(lo.value).toFixed(1)}m`;
    container.querySelector('[data-v="hi"]').textContent = `£${Number(hi.value).toFixed(1)}m`;
  }
  container.addEventListener("input", (e) => { sync(e); onChange(); });
  sync();
  return { range: () => [Number(lo.value) - 1e-9, Number(hi.value) + 1e-9] };
}

// Header cell: sortable unless noSort; `extra` goes above the label (a hide ×).
function headerCell(state, key, label, { cls = "", extra = "", style = "" } = {}) {
  const sorted = state.sortKey === key ? ` sorted${state.sortAsc ? " asc" : ""}` : "";
  return `<th data-key="${esc(key)}" class="${cls}${sorted}"${style ? ` style="${style}"` : ""}>${extra}<span class="h">${label}</span></th>`;
}

// ── Projections ──────────────────────────────────────────────────────────────
// Probabilities show as % per week; their Total is the expected count (a sum). Penalty share: average.
const STATS = {
  points:  { week: (v) => v.toFixed(1), total: (v) => v.toFixed(1), totalLabel: "Total" },
  minutes: { week: (v) => v.toFixed(0), total: (v) => v.toFixed(0), totalLabel: "Total" },
  goals:   { week: (v) => v.toFixed(2), total: (v) => v.toFixed(2), totalLabel: "Total" },
  assists: { week: (v) => v.toFixed(2), total: (v) => v.toFixed(2), totalLabel: "Total" },
  pens:    { week: (v) => v.toFixed(2), total: (v) => v.toFixed(2), totalLabel: "Average", average: true },
  cs:      { week: (v) => (v * 100).toFixed(0), total: (v) => v.toFixed(2), totalLabel: "Total" },
  defcon:  { week: (v) => (v * 100).toFixed(0), total: (v) => v.toFixed(2), totalLabel: "Total" },
};

function projectionsPage(data) {
  const rounds = data.meta.rounds;
  const players = data.players;
  const prices = players.map((p) => p.price);
  const state = { stat: "points", search: "", sortKey: "total", sortAsc: false };

  const render = () => draw();
  const weekKeys = rounds.map(String);
  // Any change of stat, filter or weeks returns the table to its default order, Total descending.
  const resort = () => { state.sortKey = "total"; state.sortAsc = false; draw(); };
  const weeks = hideable(weekKeys, document.getElementById("p-restore"), (k) => `GW${k}`, resort);
  const team = selectFilter(document.getElementById("p-team"), teamOptions(data.teams), resort);
  const pos = selectFilter(document.getElementById("p-pos"), POSITIONS, resort);
  const price = priceSlider(document.getElementById("p-price"), Math.min(...prices), Math.max(...prices), resort);
  document.getElementById("p-stat").addEventListener("change", (e) => { state.stat = e.target.value; resort(); });
  document.getElementById("p-search").addEventListener("input", (e) => { state.search = norm(e.target.value.trim()); resort(); });

  const table = document.getElementById("p-table");
  table.addEventListener("click", (e) => {
    if (weeks.click(e)) return;
    const th = e.target.closest("th[data-key]");
    if (!th) return;
    const key = th.dataset.key;
    if (state.sortKey === key) state.sortAsc = !state.sortAsc;
    else { state.sortKey = key; state.sortAsc = ["name", "pos", "team"].includes(key); }
    render();
  });

  function draw() {
    const shown = weeks.shown().map((r) => rounds.indexOf(Number(r)));
    const fmt = STATS[state.stat];
    const [pLo, pHi] = price.range();

    const rows = [];
    for (const p of players) {
      if (!team.has(p.team) || !pos.has(p.pos) || p.price < pLo || p.price > pHi) continue;
      if (state.search && !norm(p.name).includes(state.search)) continue;
      const values = shown.map((i) => p[state.stat][i]);
      const sum = values.reduce((a, v) => a + (v ?? 0), 0);
      const total = shown.length ? (fmt.average ? sum / shown.length : sum) : 0;
      rows.push({ p, values, total });
    }

    const key = state.sortKey;
    const sortValue = (r) => key === "total" ? r.total : key === "price" ? r.p.price
      : key === "name" || key === "pos" || key === "team" ? r.p[key] : r.values[shown.indexOf(rounds.indexOf(Number(key)))] ?? -Infinity;
    rows.sort((a, b) => {
      const x = sortValue(a), y = sortValue(b);
      const c = typeof x === "string" ? x.localeCompare(y) : x - y;
      return (state.sortAsc ? c : -c) || b.total - a.total;
    });

    // Scales follow the rows and columns shown: all week columns together, Total separately.
    const weekScale = Heatmap.scale(rows.flatMap((r) => r.values));
    const totalScale = Heatmap.scale(rows.map((r) => r.total));
    // Points: bold each week's highest projection among all players (the captain pick). It is bold only when that
    // player is on screen, so a filtered group (defenders, one team) shows no bold unless it contains the captain.
    const best = state.stat === "points"
      ? shown.map((i) => Math.max(...players.map((p) => p.points[i] ?? -Infinity))) : [];

    // Pos, Team and Price grow with the space: about 0.6 of a value column's width, whatever number of weeks is shown.
    // Pos and Team are left out when one position or one team is selected (they would repeat the filter).
    const showPos = pos.all(), showTeam = team.all();
    const idCols = 1 + showPos + showTeam;
    const idWidth = `width:${Math.min(9, 52.8 / (shown.length + 1 + 0.6 * idCols)).toFixed(1)}%`;
    const idCell = (k, label) => headerCell(state, k, label, { cls: "id", style: idWidth });
    let html = `<thead><tr><th class="rank">#</th>` + headerCell(state, "name", "Player", { cls: "left name" }) + (showPos ? idCell("pos", "Pos") : "") +
      (showTeam ? idCell("team", "Team") : "") + idCell("price", "Price") +
      shown.map((i) => headerCell(state, String(rounds[i]), `GW${rounds[i]}`, { cls: "wk", extra: weeks.x(String(rounds[i]), `GW${rounds[i]}`) })).join("") +
      headerCell(state, "total", fmt.totalLabel, { cls: "wk" }) + "</tr></thead><tbody>";
    rows.forEach((r, n) => {
      const p = r.p;
      html += `<tr><td class="rank">${n + 1}</td><td class="left name">${esc(p.name)}</td>${showPos ? `<td>${p.pos}</td>` : ""}${showTeam ? `<td>${p.team}</td>` : ""}` +
        `<td>${p.price.toFixed(1)}</td>`;
      shown.forEach((i, j) => {
        const v = r.values[j];
        if (v === null || p.fixtures[i] === 0) { html += `<td class="blank" title="No fixture">–</td>`; return; }
        const dgw = p.fixtures[i] > 1 ? `<sup class="dgw">×${p.fixtures[i]}</sup>` : "";
        const top = best.length && v === best[j] ? ' class="best"' : "";
        html += `<td${top} title="${esc(p.opponents[i].join(", "))}" style="background:${Heatmap.colour(v, weekScale)}">${fmt.week(v)}${dgw}</td>`;
      });
      html += `<td class="total" style="background:${Heatmap.colour(r.total, totalScale)}">${fmt.total(r.total)}</td></tr>`;
    });
    table.innerHTML = html + "</tbody>";
  }

  draw();
}

// ── Player ratings ───────────────────────────────────────────────────────────
// Goals = NPG + share × team penalty attempts /90 × conversion, all per 90; the page recomputes it after a share
// edit. Edits live in localStorage until the next part 3 (a new generated_utc), then are cleared.
const EDITS_KEY = "fpl-site-share-edits", HIDDEN_KEY = "fpl-site-ratings-hidden-cols";

const RATING_COLUMNS = [
  { key: "minutes", label: "Minutes", fmt: (v) => v.toFixed(0) },
  { key: "share", label: "Penalty share", fmt: (v) => v.toFixed(2) },
  { key: "npg", label: "NPG", fmt: (v) => v.toFixed(2) },
  { key: "goals", label: "Goals", fmt: (v) => v.toFixed(2) },
  { key: "assists", label: "Assists", fmt: (v) => v.toFixed(2) },
  { key: "ga", label: "Goals + assists", fmt: (v) => v.toFixed(2) },
  { key: "cs", label: "CS %", fmt: (v) => (v * 100).toFixed(0) },
  { key: "defcon", label: "Defcon %", fmt: (v) => (v * 100).toFixed(0) },
];

// Goals per 90 with a given share (website_plan.md "Player ratings"); test_export_site_data.py checks the pieces.
function goalsPer90(p, share) { return p.npg + share * p.team_att90 * p.conversion; }

const MINUTE_THRESHOLDS = [0, 30, 45, 60];  // 0 = no filter

function ratingsPage(data) {
  const players = data.players;
  const id = (p) => `${p.name}|${p.team}`;
  const saved = store.get(EDITS_KEY);
  const fresh = saved && saved.generated === data.meta.generated_utc;
  const edits = fresh ? saved.edits : {};
  if (!fresh) store.set(EDITS_KEY, { generated: data.meta.generated_utc, edits });
  const state = { search: "", sortKey: "ga", sortAsc: false };
  const prices = players.map((p) => p.price);

  const render = () => draw();
  const labels = Object.fromEntries(RATING_COLUMNS.map((c) => [c.key, c.label]));
  const cols = hideable(RATING_COLUMNS.map((c) => c.key), document.getElementById("r-restore"), (k) => labels[k], render, HIDDEN_KEY);
  const team = selectFilter(document.getElementById("r-team"), teamOptions(data.teams), render);
  const pos = selectFilter(document.getElementById("r-pos"), POSITIONS, render);
  const price = priceSlider(document.getElementById("r-price"), Math.min(...prices), Math.max(...prices), render);
  // Minutes filter (user, 2026-11-11): keep players whose expected minutes (the Minutes column) are above the threshold
  const minutesSelect = document.getElementById("r-mins");
  minutesSelect.innerHTML = MINUTE_THRESHOLDS.map((m) => `<option value="${m}">${m ? `Minutes over ${m}` : "All minutes"}</option>`).join("");
  minutesSelect.addEventListener("change", render);
  document.getElementById("r-search").addEventListener("input", (e) => { state.search = norm(e.target.value.trim()); render(); });
  const save = () => store.set(EDITS_KEY, { generated: data.meta.generated_utc, edits });
  const reset = document.getElementById("r-reset");
  reset.addEventListener("click", () => { for (const k of Object.keys(edits)) delete edits[k]; save(); render(); });

  const table = document.getElementById("r-table");
  table.addEventListener("click", (e) => {
    if (cols.click(e) || e.target.closest("input")) return;
    const th = e.target.closest("th[data-key]");
    if (!th) return;
    const key = th.dataset.key;
    if (state.sortKey === key) state.sortAsc = !state.sortAsc;
    else { state.sortKey = key; state.sortAsc = ["name", "pos", "team"].includes(key); }
    render();
  });
  table.addEventListener("change", (e) => {
    const input = e.target.closest("input[data-player]");
    if (!input) return;
    const p = players.find((x) => id(x) === input.dataset.player);
    const v = Number(input.value.replace(",", "."));  // a text field, so the browser locale never adds a comma
    if (input.value.trim() === "" || !Number.isFinite(v) || Math.abs(v - p.share) < 1e-9) delete edits[id(p)];
    else edits[id(p)] = Math.max(0, v);
    save(); render();
  });

  function values(p) {
    const share = edits[id(p)] ?? p.share;
    const goals = goalsPer90(p, share);
    return { minutes: p.minutes, share, npg: p.npg, goals, assists: p.assists, ga: goals + p.assists, cs: p.cs, defcon: p.defcon };
  }

  function draw() {
    reset.hidden = Object.keys(edits).length === 0;
    const [pLo, pHi] = price.range();
    const rows = players
      .filter((p) => team.has(p.team) && pos.has(p.pos) && p.price >= pLo && p.price <= pHi && (!state.search || norm(p.name).includes(state.search))
        && (!Number(minutesSelect.value) || p.minutes > Number(minutesSelect.value)))
      .map((p) => ({ p, v: values(p) }));

    if (labels[state.sortKey] && !cols.shown().includes(state.sortKey)) { state.sortKey = "ga"; state.sortAsc = false; }
    const key = state.sortKey;
    const sortValue = (r) => ["name", "pos", "team", "price"].includes(key) ? r.p[key] : r.v[key] ?? -Infinity;
    rows.sort((a, b) => {
      const x = sortValue(a), y = sortValue(b);
      const c = typeof x === "string" ? x.localeCompare(y) : x - y;
      return (state.sortAsc ? c : -c) || b.v.ga - a.v.ga;
    });

    // Each column on its own scale, over the rows shown.
    const shownKeys = new Set(cols.shown());
    const shownCols = RATING_COLUMNS.filter((c) => shownKeys.has(c.key));
    const scales = Object.fromEntries(shownCols.map((c) => [c.key, Heatmap.scale(rows.map((r) => r.v[c.key]))]));
    const showPos = pos.all(), showTeam = team.all();  // no Pos / Team column when one position / team is selected
    let html = `<thead><tr><th class="rank">#</th>` + headerCell(state, "name", "Player", { cls: "left name" }) + (showPos ? headerCell(state, "pos", "Pos") : "") +
      (showTeam ? headerCell(state, "team", "Team") : "") + headerCell(state, "price", "Price") +
      shownCols.map((c) => headerCell(state, c.key, c.label, { cls: "stat", extra: cols.x(c.key, c.label) })).join("") +
      "</tr></thead><tbody>";
    rows.forEach(({ p, v }, n) => {
      html += `<tr><td class="rank">${n + 1}</td><td class="left name">${esc(p.name)}</td>${showPos ? `<td>${p.pos}</td>` : ""}${showTeam ? `<td>${p.team}</td>` : ""}<td>${p.price.toFixed(1)}</td>`;
      for (const c of shownCols) {
        const value = v[c.key];
        const bg = Heatmap.colour(value, scales[c.key]);
        if (c.key === "share") {
          const edited = id(p) in edits;
          html += `<td class="edit${edited ? " edited" : ""}" style="background:${bg}" title="${edited ? `Model: ${p.share.toFixed(2)}` : "Edit to test a different share"}">` +
            `<input type="text" inputmode="decimal" value="${value.toFixed(2)}" data-player="${esc(id(p))}"></td>`;
        } else if (value === null || value === undefined) {
          html += `<td class="blank">–</td>`;
        } else {
          html += `<td style="background:${bg}">${c.fmt(value)}</td>`;
        }
      }
      html += "</tr>";
    });
    table.innerHTML = html + "</tbody>";
  }

  draw();
}

// ── Team ratings ─────────────────────────────────────────────────────────────
// A small sortable heatmap table with a rank column (1 = first row in the current order). Columns are
// {key, label, fmt, lowerBetter, heat}; each heat column on its own scale. Sorting a column puts its best first.
function rankedTable(table, rows, columns, sortKey, sortable = true) {
  const state = { key: sortKey, asc: false };
  const textKeys = new Set(columns.filter((c) => !c.fmt).map((c) => c.key));
  function draw() {
    const col = columns.find((c) => c.key === state.key);
    const better = col.lowerBetter ? 1 : -1;
    const sorted = [...rows].sort((a, b) => {
      const x = a[state.key], y = b[state.key];
      const c = textKeys.has(state.key) ? String(x).localeCompare(String(y)) : (x - y) * better;
      return state.asc ? -c : c;
    });
    const scales = Object.fromEntries(columns.filter((c) => c.heat).map((c) => [c.key, Heatmap.scale(rows.map((r) => r[c.key]), c.lowerBetter)]));
    let html = `<thead><tr><th class="rank">#</th>` + columns.map((c) =>
      `<th data-key="${c.key}" class="${c.fmt ? "" : "left"}${sortable ? "" : " fixed"}${sortable && state.key === c.key ? " sorted" + (state.asc ? " asc" : "") : ""}"><span class="h">${c.label}</span></th>`).join("") +
      "</tr></thead><tbody>";
    sorted.forEach((r, n) => {
      html += `<tr><td class="rank">${n + 1}</td>` + columns.map((c) => !c.fmt ? `<td class="left">${esc(r[c.key])}</td>`
        : `<td${c.heat ? ` style="background:${Heatmap.colour(r[c.key], scales[c.key])}"` : ""}>${c.fmt(r[c.key])}</td>`).join("") + "</tr>";
    });
    table.innerHTML = html + "</tbody>";
  }
  if (sortable) table.addEventListener("click", (e) => {
    const th = e.target.closest("th[data-key]");
    if (!th) return;
    if (state.key === th.dataset.key) state.asc = !state.asc;
    else { state.key = th.dataset.key; state.asc = textKeys.has(state.key); }
    draw();
  });
  draw();
}

function teamsPage(data) {
  const f2 = (v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2), f1 = (v) => v.toFixed(1);
  rankedTable(document.getElementById("t-ratings"), data.ratings, [
    { key: "name", label: "Team" },
    { key: "attack", label: "Attack", fmt: f2, heat: true },
    { key: "defense", label: "Defense", fmt: f2, heat: true, lowerBetter: true },
    { key: "overall", label: "Overall", fmt: f2, heat: true },
  ], "overall");
  rankedTable(document.getElementById("t-table"), data.table, [
    { key: "name", label: "Team" },
    { key: "Predicted_Points", label: "Predicted points", fmt: f1, heat: true },
  ], "Predicted_Points", false);
}

// ── Fixtures (fixture ticker) ────────────────────────────────────────────────
// Each cell's value: Overall = gf − ga, Attack = gf, Defense = ga (lower is better); doubles summed.
const FIXTURE_MODES = {
  overall: { label: "Overall", value: (f) => f.gf - f.ga, lowerBetter: false,
             note: "Opposition strength, as the fixture-ticker images: the goal difference an average team would expect v that opponent at that venue." },
  attack: { label: "Attack", value: (f) => f.gf, lowerBetter: false,
            note: "Opposition defence: the goals an average team would score v that opponent at that venue (fixtures for attackers)." },
  defense: { label: "Defense", value: (f) => f.ga, lowerBetter: true,
             note: "Opposition attack: the goals an average team would concede v that opponent at that venue, lower is easier (fixtures for defenders and keepers)." },
};

function fixturesPage(data) {
  const rounds = data.meta.rounds;
  const names = Object.fromEntries(data.ratings.map((r) => [r.team, r.name]));
  const modeSelect = document.getElementById("f-mode");
  modeSelect.innerHTML = Object.entries(FIXTURE_MODES).map(([k, m]) => `<option value="${k}">${m.label}</option>`).join("");
  modeSelect.addEventListener("change", () => draw());
  const weeks = hideable(rounds.map(String), document.getElementById("f-restore"), (k) => `GW${k}`, () => draw());
  const table = document.getElementById("f-table");
  table.addEventListener("click", (e) => weeks.click(e));

  function draw() {
    const m = FIXTURE_MODES[modeSelect.value];
    const shown = weeks.shown().map((r) => rounds.indexOf(Number(r)));
    const rows = data.fixtures.map((t) => {
      const values = shown.map((i) => t.weeks[i].length ? t.weeks[i].reduce((a, f) => a + m.value(f), 0) : null);
      return { t, values, total: values.reduce((a, v) => a + (v ?? 0), 0) };
    });
    rows.sort((a, b) => m.lowerBetter ? a.total - b.total : b.total - a.total);
    const weekScale = Heatmap.scale(rows.flatMap((r) => r.values), m.lowerBetter);
    const totalScale = Heatmap.scale(rows.map((r) => r.total), m.lowerBetter);

    let html = `<thead><tr><th class="left name">Team</th>` +
      shown.map((i) => `<th class="wk">${weeks.x(String(rounds[i]), `GW${rounds[i]}`)}<span class="h">GW${rounds[i]}</span></th>`).join("") +
      `<th class="wk"><span class="h">Total</span></th></tr></thead><tbody>`;
    for (const r of rows) {
      html += `<tr><td class="left name">${esc(names[r.t.team] || r.t.team)}</td>`;
      shown.forEach((i, j) => {
        const games = r.t.weeks[i];
        if (!games.length) { html += `<td class="fx blank" title="Blank gameweek">–</td>`; return; }
        const label = games.map((f) => `<span class="opp">${f.opp.toLowerCase()} (${f.venue})</span>`).join("");
        const tip = games.map((f) => `${f.opp} (${f.venue}), average team: ${f.gf.toFixed(2)} for, ${f.ga.toFixed(2)} against`).join("\n");
        const dgw = games.length > 1 ? `<sup class="dgw">×${games.length}</sup>` : "";
        html += `<td class="fx" title="${esc(tip)}" style="background:${Heatmap.colour(r.values[j], weekScale)}">${label}${dgw}</td>`;
      });
      html += `<td class="total" style="background:${Heatmap.colour(r.total, totalScale)}">${r.total.toFixed(2)}</td></tr>`;
    }
    table.innerHTML = html + "</tbody>";
  }
  draw();
}

// ── Start ────────────────────────────────────────────────────────────────────
showView();
const loadError = (ids, what) => (err) => {
  for (const id of ids) {
    document.getElementById(id).innerHTML =
      `<p class="note">Could not load the ${what} (${esc(err.message)}). Run <code>python run_site_export.py</code>.</p>`;
  }
};
fetchJson("data/projections.json").then(projectionsPage).catch(loadError(["view-projections"], "projections"));
fetchJson("data/ratings.json").then(ratingsPage).catch(loadError(["view-ratings"], "player ratings"));
fetchJson("data/teams.json").then((data) => { teamsPage(data); fixturesPage(data); })
  .catch(loadError(["view-teams", "view-fixtures"], "team data"));
