/* Audit tab. Data-quality checklist for the selected agent + batch. */
(function () {
  window.ZP = window.ZP || { pages: {}, agent: "" };
  ZP.pages = ZP.pages || {};

  const esc = (s) => String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");

  const agentOf = (ctx) =>
    (ctx && ctx.agent) || ZP.agent || sessionStorage.getItem("zp_agent") || localStorage.getItem("zp-agent") || "";

  const A = { mode: "" };

  const runKey = (agent) => "zp_audit_run:" + (agent || "");

  function api(path, opt) {
    return fetch(ZP.apiUrl ? ZP.apiUrl(path) : path, { credentials: "omit", ...opt }).then((r) => r.json());
  }

  function auditUrl(agent, runId) {
    const packed = runId ? agent + "::" + runId : agent;
    let url = "/api/audit?agent=" + encodeURIComponent(packed);
    if (runId) url += "&run=" + encodeURIComponent(runId);
    return url;
  }

  function runTitle(r, agent) {
    if (ZP.runTitle) return ZP.runTitle(r, agent);
    const kind = ZP.kindLabel ? ZP.kindLabel(r) : (r.mode || "Batch");
    return `${agent} · ${kind} · ${r.n ?? 0} conversations`;
  }

  function batchMeta(r) {
    if (ZP.batchMeta) return ZP.batchMeta(r);
    return `${r.n1 || 0} Pass · ${r.n0 || 0} Fail`;
  }

  function kindClass(r) {
    if (ZP.kindClass) return ZP.kindClass(r);
    return "";
  }

  function wordOf(st) {
    if (st === "pass" || st === "ready") return "Ready";
    if (st === "warn" || st === "watch") return "Watch";
    if (st === "fail" || st === "not_ready") return "Not ready";
    return "";
  }

  function tone(st) {
    if (st === "pass" || st === "ready") return "pass";
    if (st === "warn" || st === "watch") return "watch";
    if (st === "fail" || st === "not_ready") return "fail";
    return "";
  }

  function tallyLine(d) {
    const nr = d.n_not_ready ?? d.n_fail ?? 0;
    const nw = d.n_watch ?? d.n_warn ?? 0;
    const ok = d.n_ready ?? d.n_pass ?? 0;
    const parts = [];
    if (nr) parts.push(nr + " not ready");
    if (nw) parts.push(nw + " watch");
    if (ok) parts.push(ok + " ready");
    return parts.join(" · ");
  }

  function packHtml(title, pack) {
    const st = (pack && pack.status) || "not_ready";
    const word = (pack && pack.word) || wordOf(st);
    const why = (pack && pack.why) || "";
    return `<div class="block ${tone(st)}">
      <div class="word ${tone(st)}">${esc(word)}</div>
      <div class="t">${esc(title)}</div>
      ${why ? `<div class="m">${esc(why)}</div>` : ""}
    </div>`;
  }

  function checkHtml(check) {
    const st = check.status || "idle";
    const word = check.word || wordOf(st);
    const long = check.id === "n_per_prompt";
    return `<div class="block ${tone(st)}" data-check="${esc(check.id)}">
      <div class="word ${tone(st)}">${esc(st === "idle" ? "" : word)}</div>
      <div class="t">${esc(check.title)}</div>
      <div class="${long ? "m" : "n"}" style="margin-top:12px">${esc(check.value || "")}</div>
    </div>`;
  }

  function matchRun(runs, runId) {
    if (!runId) return null;
    return (runs || []).find((r) => r.id === runId || r.stem === runId) ||
      (runs || []).find((r) => (r.stem || "").includes(runId) || (r.id || "").includes(runId)) ||
      null;
  }

  function runCards(runs, agent, selected) {
    if (A.mode === "unique") A.mode = "explore";
    const shown = (runs || []).filter((r) =>
      ZP.matchesKindFilter ? ZP.matchesKindFilter(r, A.mode) : true
    );
    const chips = ZP.kindFilterHtml
      ? `<div class="seg" id="mode-seg" style="margin:0 0 12px;width:fit-content">${ZP.kindFilterHtml(A.mode)}</div>`
      : "";
    const grid = !shown.length
      ? `<p class="empty">${(runs || []).length ? "No batches match this kind." : "No batches for this agent."}</p>`
      : `<div class="blocks">${shown.map((r) => {
          const id = r.id || r.stem;
          const on = id === selected || r.stem === selected;
          return `<button type="button" class="block ${kindClass(r)} ${on ? "on" : ""}" data-run="${esc(id)}">
        <div class="t">${esc(runTitle(r, agent))}</div>
        <div class="m">${esc(batchMeta(r))}</div>
      </button>`;
        }).join("")}</div>`;
    return `<div class="audit-picks">
      <p class="lab">This set of conversations</p>
      ${chips}
      ${grid}
    </div>`;
  }

  function paint(main, state) {
    const d = state.data || {};
    const agent = state.agent;
    const runs = d.runs || [];
    const selected = state.runId || "";
    const checks = d.checks || [];
    const ran = Boolean(checks.length);
    const packs = d.packs || {};

    const tally = ran ? tallyLine(d) : "";
    const batchNote = `These checks are about this whole batch, not each conversation. Ready means this file can be used for that pack. Not ready means this file is the wrong shape.${tally ? " " + tally + " counts checks, not rows." : ""}`;
    const headline = ran && d.headline
      ? `<p class="sub">${esc(d.headline)}</p>`
      : "";
    const repeats = ran && d.repeats_line
      ? `<p class="sub">${esc(d.repeats_line)}</p>`
      : "";
    const packRow = ran
      ? `<div class="packs">
          ${packHtml("Good behavior pack", packs.imitation)}
          ${packHtml("Contrasting repeats", packs.mixed)}
        </div>`
      : "";

    main.innerHTML = `<div class="page wide">
      <div class="row-head">
        <h1>Is this batch usable?</h1>
      </div>
      <p class="sub">${esc(batchNote)}</p>
      ${runCards(runs, agent, selected)}
      <p class="err" id="audit-err">${esc(state.error || d.error || "")}</p>
      ${state.loading && !runs.length ? `<p class="empty">Loading batches.</p>` : ""}
      ${state.loading && runs.length ? `<p class="empty">Checking this batch.</p>` : ""}
      ${headline}
      ${repeats}
      ${packRow}
      <div class="audit-checks" id="audit-list">
        ${ran ? checks.map((c) => checkHtml(c)).join("") : (state.loading ? "" : `<p class="empty">Pick a batch. Audit says whether you can download a good behavior pack or Contrasting repeats.</p>`)}
      </div>
    </div>`;

    main.onclick = async (e) => {
      const modeBtn = e.target.closest("#mode-seg button");
      if (modeBtn) {
        A.mode = modeBtn.dataset.mode === "unique" ? "explore" : (modeBtn.dataset.mode || "");
        paint(main, state);
        return;
      }
      const card = e.target.closest("[data-run]");
      if (!card) return;
      const id = card.dataset.run;
      if (!id || id === state.runId) return;
      state.runId = id;
      try { sessionStorage.setItem(runKey(agent), state.runId); } catch (_) {}
      await loadBatch(main, state);
    };
  }

  async function loadBatch(main, state) {
    const runId = state.runId;
    const listed = (state.data && state.data.runs) || [];
    if (!runId) {
      state.data = { ...(state.data || {}), runs: listed, checks: [], headline: "", packs: {} };
      state.error = "";
      state.loading = false;
      paint(main, state);
      return;
    }
    state.loading = true;
    state.open = "";
    paint(main, state);
    try {
      const out = await api(auditUrl(state.agent, runId));
      out.runs = listed;
      state.data = out;
      state.error = out.error || "";
    } catch (_) {
      state.error = "Could not check this batch.";
    }
    state.loading = false;
    paint(main, state);
  }

  async function renderAudit(ctx) {
    const main = ctx.main || document.getElementById("main");
    const agent = agentOf(ctx);
    if (!agent) {
      main.innerHTML = `<div class="page">
        <h1>Is this batch usable?</h1>
        <p class="empty">Pick an agent first.</p>
        <a class="go" href="#agents">All agents</a>
      </div>`;
      return;
    }
    const restRun = (ctx.rest && ctx.rest[0]) || "";
    let saved = "";
    try { saved = sessionStorage.getItem(runKey(agent)) || ""; } catch (_) {}
    const state = {
      agent,
      runId: restRun || saved,
      data: { runs: [], checks: [] },
      open: "",
      error: "",
      loading: true,
    };
    paint(main, state);
    try {
      const listed = await api("/api/runs?agent=" + encodeURIComponent(agent));
      state.data = { runs: listed.runs || [], checks: [], packs: {} };
      if (listed.error) state.error = listed.error;
    } catch (_) {
      state.error = "Could not load batches.";
      state.loading = false;
      paint(main, state);
      return;
    }
    const runs = state.data.runs || [];
    const hit = matchRun(runs, state.runId);
    state.runId = hit ? hit.id : ((runs[0] && runs[0].id) || "");
    await loadBatch(main, state);
  }

  ZP.pages.audit = renderAudit;
})();
