// Solver page (website_plan.md step 5, §228). Talks to site/server.py's /api/solver/* routes, which run
// scripts/optimizer/solver_jobs.py: load a squad (FPL team ID, pasted JSON or team_data.json), pick the horizon, the
// solve mode and any fixed chip weeks or forced decisions, run one background solve, and show the replay-checked plans.
"use strict";

(function () {
  const root = document.getElementById("view-solver");
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const plain = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  const two = (n) => String(Math.floor(n)).padStart(2, "0");
  const clock = (s) => (s >= 3600 ? `${Math.floor(s / 3600)}:${two((s % 3600) / 60)}` : `${Math.floor(s / 60)}`) + `:${two(s % 60)}`;
  const keep = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage off */ } },
  };
  const post = (path, body) => fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
    .then((r) => r.json());

  let state = null;       // /api/solver/state
  let loaded = null;      // /api/solver/team result
  let source = keep.get("solver.source") || "id";
  const hold = [], avoid = [], moves = [];
  let polling = null;

  const player = (id) => state.players.find((p) => p.id === id);
  const label = (p) => `${p.name} (${p.team} ${p.pos})`;
  const weeks = () => state.gws.slice(0, Number($("s-horizon").value));

  function notice(id, message, kind = "error") {
    const box = $(id);
    box.textContent = message || "";
    box.className = `s-notice ${kind}`;
    box.hidden = !message;
  }

  // ── Player picker (hold / avoid / moves) ──────────────────────────────────
  function picker(placeholder, onPick, initial) {
    const wrap = document.createElement("div");
    wrap.className = "s-pick";
    wrap.innerHTML = `<input type="search" placeholder="${esc(placeholder)}"><div class="s-list"></div>`;
    const input = wrap.querySelector("input"), list = wrap.querySelector(".s-list");
    if (initial) input.value = label(initial);
    let shown = [], active = -1;
    const owned = () => new Set(loaded ? loaded.squad.map((p) => p.id) : []);
    function draw() {
      const term = plain(input.value.trim()), mine = owned();
      shown = state.players.filter((p) => !term || plain(`${p.name} ${p.team}`).includes(term));
      shown.sort((a, b) => (mine.has(b.id) - mine.has(a.id)) || b.xp - a.xp);
      shown = shown.slice(0, 40);
      active = -1;
      list.innerHTML = shown.map((p, i) => `<div class="s-opt" data-i="${i}"><span>${esc(p.name)}${mine.has(p.id) ? " ★" : ""}</span>` +
        `<small>${esc(p.team)} ${p.pos} £${p.price.toFixed(1)} · ${p.xp.toFixed(1)} xP</small></div>`).join("");
      list.classList.toggle("open", shown.length > 0);
    }
    function choose(p) { list.classList.remove("open"); onPick(p, input); }
    function mark() {
      [...list.children].forEach((row, i) => row.classList.toggle("active", i === active));
      if (list.children[active]) list.children[active].scrollIntoView({ block: "nearest" });
    }
    list.addEventListener("mousedown", (e) => {
      const row = e.target.closest(".s-opt");
      if (row) { e.preventDefault(); choose(shown[Number(row.dataset.i)]); }
    });
    input.addEventListener("focus", () => { input.select(); draw(); });
    input.addEventListener("input", () => { if (input.value === "") onPick(null, input); draw(); });
    input.addEventListener("blur", () => list.classList.remove("open"));
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { active = Math.min(active + 1, shown.length - 1); mark(); e.preventDefault(); }
      else if (e.key === "ArrowUp") { active = Math.max(active - 1, 0); mark(); e.preventDefault(); }
      else if (e.key === "Enter" && shown.length) { choose(shown[Math.max(active, 0)]); e.preventDefault(); }
      else if (e.key === "Escape") list.classList.remove("open");
    });
    return wrap;
  }

  function drawChosen(box, ids) {
    box.innerHTML = ids.length ? ids.map((id) => `<span class="s-tag">${esc(label(player(id)))}<button data-id="${id}" title="Remove">×</button></span>`).join("")
      : `<span class="label">None</span>`;
  }
  function addTo(ids, other, box, otherBox) {
    return (p, input) => {
      if (!p) return;
      if (!ids.includes(p.id)) ids.push(p.id);
      if (other.includes(p.id)) other.splice(other.indexOf(p.id), 1);
      input.value = "";
      drawChosen(box, ids); drawChosen(otherBox, other);
    };
  }

  function drawMoves() {
    const box = $("s-moves");
    if (!moves.length) { box.innerHTML = `<span class="label">No planned moves.</span>`; return; }
    box.replaceChildren(...moves.map((move) => {
      const row = document.createElement("div");
      row.className = "s-move";
      const gw = document.createElement("select");
      gw.innerHTML = weeks().map((w) => `<option value="${w}" ${w === move.gw ? "selected" : ""}>GW${w}</option>`).join("");
      if (!weeks().includes(move.gw)) move.gw = weeks()[0];
      gw.addEventListener("change", () => { move.gw = Number(gw.value); });
      const side = (key, text) => picker(text, (p, input) => { move[key] = p ? p.id : null; if (p) input.value = label(p); },
        move[key] ? player(move[key]) : null);
      const x = document.createElement("button");
      x.className = "link"; x.textContent = "remove";
      x.addEventListener("click", () => { moves.splice(moves.indexOf(move), 1); drawMoves(); });
      row.append(gw, side("out", "Transfer out"), side("in", "Transfer in"), x);
      return row;
    }));
  }

  // ── Squad ─────────────────────────────────────────────────────────────────
  function drawSource() {
    root.querySelectorAll("[data-source]").forEach((b) => b.classList.toggle("on", b.dataset.source === source));
    $("s-src-id").hidden = source !== "id";
    $("s-src-json").hidden = source !== "json";
    $("s-src-file").hidden = source !== "file";
  }

  async function loadTeam() {
    notice("s-team-msg", "Loading…", "info");
    const body = source === "id" ? { id: $("s-id").value.trim() } : source === "json" ? { json: $("s-json").value } : { file: true };
    if (source === "id") keep.set("solver.id", body.id);
    try {
      const result = await post("/api/solver/team", body);
      if (!result.ok) { notice("s-team-msg", result.message); return; }
      loaded = result;
      notice("s-team-msg", "");
      drawSquad();
      drawChips();
      $("s-run").disabled = false;
    } catch (error) {
      notice("s-team-msg", `Lost contact with the server: ${error.message}`);
    }
  }

  function drawSquad() {
    const s = loaded;
    $("s-squad-head").innerHTML = `<b>${esc(s.source)}</b> · ${s.fts} free transfer${s.fts === 1 ? "" : "s"}` +
      (s.made ? ` (${s.made} already made this week)` : "") + ` · bank £${s.bank.toFixed(1)}m · ` +
      `<button class="link" id="s-edit-json">edit as JSON</button>`;
    $("s-edit-json").addEventListener("click", () => {
      source = "json"; keep.set("solver.source", source); drawSource();
      $("s-json").value = JSON.stringify(s.team, null, 1);
    });
    $("s-squad").innerHTML = `<thead><tr><th class="left">Pos</th><th class="left">Player</th><th class="left">Team</th>` +
      `<th>Sell</th><th>Now</th><th>xP GW${state.gws[0]}–${state.gws[state.gws.length - 1]}</th></tr></thead><tbody>` +
      s.squad.map((p) => `<tr><td class="left">${p.pos}</td><td class="left">${esc(p.name)}</td><td class="left">${esc(p.team)}</td>` +
        `<td>${p.sell.toFixed(1)}</td><td>${p.now.toFixed(1)}</td><td>${p.xp.toFixed(1)}</td></tr>`).join("") + "</tbody>";
    $("s-squad-wrap").hidden = false;
  }

  // ── Settings ──────────────────────────────────────────────────────────────
  function drawChips() {
    const open = loaded ? loaded.chip_weeks : {};
    const shown = new Set(weeks());
    $("s-chips").innerHTML = state.chips.map((c) => {
      const options = (open[c.key] || []).filter((w) => shown.has(w));
      const old = $(`s-chip-${c.key}`) ? $(`s-chip-${c.key}`).value : "";
      return `<label>${esc(c.label)} <select id="s-chip-${c.key}" ${options.length ? "" : "disabled"}>` +
        `<option value="">${options.length ? "none" : loaded ? "not available" : "load a squad"}</option>` +
        options.map((w) => `<option value="${w}" ${String(w) === old ? "selected" : ""}>GW${w}</option>`).join("") + `</select></label>`;
    }).join("");
  }

  function drawMode() {
    const m = state.modes.find((x) => x.name === $("s-mode").value);
    $("s-mode-note").textContent = `Measured: ${m.measured}. A run is stopped after ${clock(m.limit_s)}.`;
  }

  // ── Run, progress, results ────────────────────────────────────────────────
  async function run() {
    notice("s-run-msg", "");
    if (moves.some((m) => !m.out && !m.in)) { notice("s-run-msg", "Each planned move needs a transfer out or a transfer in."); return; }
    const chips = Object.fromEntries(state.chips.map((c) => [c.key, $(`s-chip-${c.key}`).value ? Number($(`s-chip-${c.key}`).value) : null]));
    const body = { team: loaded.team, horizon: Number($("s-horizon").value), mode: $("s-mode").value, chips,
                   hold, avoid, moves: moves.map((m) => ({ ...m })) };
    const result = await post("/api/solver/solve", body);
    if (!result.ok) { notice("s-run-msg", result.message); return; }
    $("s-result").hidden = true;
    poll();
    setTimeout(() => $("s-progress").scrollIntoView({ behavior: "smooth", block: "start" }), 300);
  }

  function poll() {
    clearTimeout(polling);
    fetch("/api/solver/job", { cache: "no-store" }).then((r) => r.json()).then((job) => {
      drawJob(job);
      if (job.status === "running") polling = setTimeout(poll, 1000);
    }).catch(() => { polling = setTimeout(poll, 3000); });
  }

  function drawJob(job) {
    const running = job.status === "running";
    $("s-run").disabled = running || !loaded;
    $("s-progress").hidden = job.status === "idle";
    if (job.status === "idle") return;
    const share = Math.min(1, (job.elapsed_s || 0) / job.limit_s);
    $("s-progress-text").innerHTML = running
      ? `<b>Solving</b> (${esc(job.mode)}, ${job.horizon} week${job.horizon === 1 ? "" : "s"}, ${job.plans} plan${job.plans === 1 ? "" : "s"}): ` +
        `${esc(job.progress || "")} · ${clock(job.elapsed_s || 0)} <span class="label">(stopped at ${clock(job.limit_s)})</span>`
      : job.status === "done" ? `<b>Done</b> in ${clock(job.elapsed_s)}: the plans below passed the replay check.`
      : `<b>Failed</b> after ${clock(job.elapsed_s || 0)}.`;
    $("s-bar").style.width = `${(running ? share : 1) * 100}%`;
    $("s-bar").className = `s-bar ${job.status}`;
    $("s-cancel").hidden = !running;
    $("s-log").textContent = (job.lines || []).join("\n");
    notice("s-run-msg", job.status === "failed" ? job.message : "");
    if (job.status === "done") showResult(job.last);
  }

  function showResult(last) {
    if (!last) { $("s-result").hidden = true; return; }
    $("s-result-head").textContent = `${last.mode[0].toUpperCase()}${last.mode.slice(1)} solve, GW${last.gws[0]}–${last.gws[1]}, ` +
      `${last.plans} plan${last.plans === 1 ? "" : "s"}, finished ${last.solved_utc} UTC.`;
    const frame = $("s-plans");
    // follow the plans page's height, which changes when a week's "Team" row is opened
    frame.onload = () => {
      const page = frame.contentDocument.body;
      const fit = () => { frame.style.height = `${page.scrollHeight + 24}px`; };
      fit();
      new ResizeObserver(fit).observe(page);
    };
    frame.src = `/solver/plans?t=${Date.now()}`;
    $("s-result").hidden = false;
  }

  // ── Start ─────────────────────────────────────────────────────────────────
  async function start() {
    try {
      state = await (await fetch("/api/solver/state", { cache: "no-store" })).json();
    } catch (error) {
      root.innerHTML = `<p class="note">The Solver needs the site's server (python run_site.py): ${esc(error.message)}</p>`;
      return;
    }
    if (state.error) { root.innerHTML = `<p class="note">The solver can't start: ${esc(state.error)}</p>`; return; }
    $("s-horizon").innerHTML = state.gws.map((w, i) => `<option value="${i + 1}" ${i + 1 === state.horizon ? "selected" : ""}>` +
      `${i + 1} (GW${state.gws[0]}${i ? `–${w}` : ""})</option>`).join("");
    $("s-mode").innerHTML = state.modes.map((m) => `<option value="${m.name}" ${m.name === state.default_mode ? "selected" : ""}>` +
      `${m.name[0].toUpperCase()}${m.name.slice(1)}</option>`).join("");
    $("s-id").value = keep.get("solver.id") || "";
    root.querySelectorAll("[data-source]").forEach((b) => b.addEventListener("click", () => {
      source = b.dataset.source; keep.set("solver.source", source); drawSource();
    }));
    root.querySelectorAll("[data-load]").forEach((b) => b.addEventListener("click", loadTeam));
    $("s-id").addEventListener("keydown", (e) => { if (e.key === "Enter") loadTeam(); });
    $("s-horizon").addEventListener("change", () => { drawChips(); drawMoves(); });
    $("s-mode").addEventListener("change", drawMode);
    $("s-hold-pick").append(picker("Add a player to target or hold", addTo(hold, avoid, $("s-hold"), $("s-avoid"))));
    $("s-avoid-pick").append(picker("Add a player to avoid or sell", addTo(avoid, hold, $("s-avoid"), $("s-hold"))));
    for (const [box, ids] of [[$("s-hold"), hold], [$("s-avoid"), avoid]]) {
      box.addEventListener("click", (e) => {
        const b = e.target.closest("[data-id]");
        if (b) { ids.splice(ids.indexOf(Number(b.dataset.id)), 1); drawChosen($("s-hold"), hold); drawChosen($("s-avoid"), avoid); }
      });
    }
    $("s-add-move").addEventListener("click", () => { moves.push({ gw: weeks()[0], out: null, in: null }); drawMoves(); });
    $("s-run").addEventListener("click", run);
    $("s-cancel").addEventListener("click", () => post("/api/solver/cancel", {}).then(poll));
    drawSource(); drawChips(); drawMode(); drawMoves();
    drawChosen($("s-hold"), hold); drawChosen($("s-avoid"), avoid);
    $("s-context").textContent = `Next deadline: GW${state.gws[0]}. Projections from the latest part 2.`;
    showResult(state.last);
    poll();  // a solve may already be running (e.g. after a page reload)
  }
  start();
})();
