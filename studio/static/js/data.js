window.ZP = window.ZP || { pages: {} };

(function () {
  const D = (ZP._data = ZP._data || {
    tags: new Set(),
    mode: "",
    q: "",
    splitOnly: false,
    group: -1,
    roll: -1,
    openRolls: new Set(),
    shown: 20,
    run: null,
    bin: "",
    reason: "",
    trace: false,
    traceN: 8,
    picked: new Set(),
  });

  const STYLE = `
.zp-data .toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:0 0 18px}
.zp-data .toolbar .grow{flex:1}
.zp-data .stats{display:flex;flex-wrap:wrap;gap:18px;color:var(--slate);margin:0 0 16px;font-variant-numeric:tabular-nums}
.zp-data .filter{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 14px;align-items:center}
.zp-data .tags{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.zp-data .tag{border:1px solid var(--line);background:var(--white);padding:3px 8px;font-size:12px;color:var(--slate)}
.zp-data button.tag{cursor:pointer}
.zp-data .tag.on{border-color:var(--violet);background:var(--wash);color:var(--ink)}
.zp-data .tag .x{margin-left:6px;color:var(--mut);border:0;background:none;padding:0;font-size:12px}
.zp-data .ghost{border:1px solid var(--line);background:var(--white);padding:7px 14px}
.zp-data .seg{display:flex;border:1px solid var(--line);background:var(--white)}
.zp-data .seg button{border:0;background:none;padding:7px 12px;border-right:1px solid var(--hair)}
.zp-data .seg button:last-child{border-right:0}
.zp-data .seg button.on{background:var(--select);color:var(--ink)}
.zp-data input.slim{width:auto;min-width:160px;border:1px solid var(--line);background:var(--white);padding:7px 10px}
.zp-data input.tag-in{width:92px;border:0;border-bottom:1px solid var(--line);background:transparent;padding:2px 4px}
.zp-data .crumb{color:var(--mut);margin:0 0 10px;font-size:13px}
.zp-data .group{border:1px solid var(--line);background:var(--white);margin-bottom:8px}
.zp-data .group header{display:flex;gap:12px;padding:10px 12px;cursor:pointer;align-items:flex-start}
.zp-data .group header .p{flex:1}
.zp-data .group header.open .p{font-weight:700}
.zp-data .group header.open{background:none;box-shadow:none;border-bottom:1px solid var(--line)}
.zp-data .badge{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--violet-ink);border:1px solid var(--violet);padding:1px 6px}
.zp-data .situations{border-top:1px solid var(--line)}
.zp-data .sit-h,.zp-data .roll-h{display:flex;gap:10px;width:100%;text-align:left;background:none;border:0;border-bottom:1px solid var(--hair);padding:10px 4px;cursor:pointer;color:inherit;font:inherit;align-items:flex-start}
.zp-data .sit-h:hover,.zp-data .roll-h:hover{background:var(--hover)}
.zp-data .sit-h.open .p,.zp-data .roll-h.open{font-weight:700}
.zp-data .sit-h.open,.zp-data .roll-h.open,.zp-data .repeat-h.open{background:none;box-shadow:none}
.zp-data .chev{color:var(--mut);width:1em;flex:none}
.zp-data .thread{padding:12px;display:flex;flex-direction:column;gap:14px}
.zp-data .repeat{border:1px solid var(--line);background:var(--white)}
.zp-data .repeat-h{display:flex;gap:10px;width:100%;text-align:left;background:none;border:0;padding:12px 14px;cursor:pointer;color:inherit;font:inherit;align-items:baseline}
.zp-data .repeat-h:hover{background:var(--hover)}
.zp-data .repeat-k{font-weight:600;min-width:5.2em}
.zp-data .repeat-h.open .repeat-k{font-weight:700}
.zp-data .repeat-body{padding:4px 14px 14px;border-top:1px solid var(--line)}
.zp-data .writer-failed-label{margin:0;font-weight:600;color:var(--ink)}
.zp-data .turn{padding:10px 0;border-bottom:1px solid var(--hair);font-size:13px}
.zp-data .turn:last-child{border-bottom:0}
.zp-data .who{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin-bottom:4px}
.zp-data .who.agent{color:var(--ink);font-weight:600}
.zp-data pre.blob{margin:0;white-space:pre-wrap;word-break:break-word;font-family:var(--mono);font-size:12px;color:var(--slate)}
.zp-data .file{display:flex;align-items:center;gap:10px}
.zp-data .muted{color:var(--mut);font-size:12px}
.zp-data td.actions{white-space:nowrap}
.zp-data td.actions button{margin-right:8px}
`;

  function esc(s) {
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function blobOf(value) {
    if (value == null) return "";
    if (typeof value === "string") return value;
    try { return JSON.stringify(value); } catch (_) { return String(value); }
  }

  function isWriterFailed(value) {
    const t = blobOf(value).toLowerCase();
    if (!t) return false;
    if (t.includes("stressd-vllm")) return true;
    if (t.includes("lost track of input")) return true;
    if (t.includes("internalfailure") && t.includes("modal")) return true;
    if (t.includes("modal.run") && (t.includes("500") || t.includes("internalfailure"))) return true;
    return false;
  }

  function rollWriterFailed(roll) {
    if (!roll) return false;
    if (isWriterFailed(roll.final_text) || isWriterFailed(roll.reason) || isWriterFailed(roll.error)) return true;
    for (const m of roll.messages || []) {
      if (isWriterFailed(m && m.content) || isWriterFailed(m && m.error)) return true;
    }
    return false;
  }

  function writerFailedHtml() {
    return `<p class="writer-failed-label">Writer failed</p>`;
  }

  function agentOf(ctx) {
    let a = (ctx && ctx.agent) || (window.ZP && ZP.agent) ||
      localStorage.getItem("zp-agent") || sessionStorage.getItem("zp_agent") || "";
    a = String(a || "").trim();
    if (a.includes("@")) a = "";
    if (a) {
      try { localStorage.setItem("zp-agent", a); } catch (_) {}
      if (window.ZP) ZP.agent = a;
    }
    return a;
  }

  function cardDate(ts) {
    if (!ts) return "";
    const d = new Date(ts * 1000);
    if (Number.isNaN(d.getTime())) return "";
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    return `${d.getDate()} ${months[d.getMonth()]} ${d.getFullYear()}`;
  }

  async function api(path, opt) {
    const res = await fetch(ZP.apiUrl ? ZP.apiUrl(path) : path, { credentials: "omit", ...opt });
    try {
      return await res.json();
    } catch (_) {
      return { error: "bad response" };
    }
  }

  function downloadText(filename, text, mime) {
    const blob = new Blob([text], { type: mime || "application/octet-stream" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 500);
  }

  function pretty(value) {
    if (value == null || value === "") return "";
    if (typeof value === "string") {
      const t = value.trim();
      if ((t.startsWith("{") || t.startsWith("[")) && t.length < 8000) {
        try { return JSON.stringify(JSON.parse(t), null, 2); } catch (_) { return value; }
      }
      return value;
    }
    try { return JSON.stringify(value, null, 2); } catch (_) { return String(value); }
  }

  function needAgent(ctx, main) {
    const agent = agentOf(ctx);
    if (agent) return false;
    main.innerHTML = `<div class="page zp-data"><style>${STYLE}</style>
      <h1>Data</h1>
      <p class="empty">Pick an agent first. Batches live under that name.</p>
      <a class="go" href="#agents">All agents</a></div>`;
    return true;
  }

  function passRate(n0, n1) {
    const a = Number(n0) || 0;
    const b = Number(n1) || 0;
    if (a + b <= 0) return "—";
    return Math.round((100 * b) / (a + b)) + "%";
  }

  function batchKind(r) {
    if (ZP.batchKind) return ZP.batchKind(r);
    if (!r || typeof r !== "object") {
      const m = String(r || "").toLowerCase();
      return m === "unique" ? "explore" : m;
    }
    const stem = String(r.stem || r.name || "").toLowerCase();
    const source = String(r.source || "").toLowerCase();
    const tags = (r.tags || []).map((t) => String(t).toLowerCase());
    if (stem.includes("unique") || source.includes("unique") || tags.includes("unique")) {
      return "explore";
    }
    const mode = String(r.mode || "").toLowerCase();
    if (mode === "unique") return "explore";
    return mode;
  }

  function isExploreRun(r) {
    return batchKind(r) === "explore";
  }

  function kindLabel(modeOrRun) {
    if (ZP.kindLabel) return ZP.kindLabel(modeOrRun);
    const m = batchKind(modeOrRun);
    if (m === "rl" || m === "grpo") return "Repeats";
    if (m === "sft") return "Imitation";
    if (m === "explore") return "Unique situations";
    if (m === "adaptive") return "Adaptive";
    return m || "Batch";
  }

  function kindNote(modeOrRun) {
    if (ZP.kindNote) return ZP.kindNote(modeOrRun);
    const m = batchKind(modeOrRun);
    if (m === "explore") return "New situations, one try each.";
    if (m === "sft") return "Several wordings of the same ask.";
    if (m === "rl") return "Same wording, several independent tries.";
    if (m === "adaptive") return "Steer toward gaps.";
    return "";
  }

  function uniqueOrigin(r) {
    if (ZP.uniqueOrigin) return ZP.uniqueOrigin(r);
    if (!r || typeof r !== "object") return "";
    const stem = String(r.stem || r.name || "").toLowerCase();
    const source = String(r.source || "").toLowerCase();
    const tags = (r.tags || []).map((t) => String(t).toLowerCase());
    if (stem.includes("unique") || source.includes("unique") || tags.includes("unique")) {
      return "deduped";
    }
    return "";
  }

  function batchTitle(r, agent) {
    if (ZP.runTitle) return ZP.runTitle(r, agent);
    const who = agent || r.agent || "agent";
    return `${who} · ${kindLabel(r)} · ${r.n ?? 0} conversations`;
  }

  function meanLabel(v) {
    if (v == null || Number.isNaN(Number(v))) return "—";
    return Number(v).toFixed(2);
  }

  function gradeWord(bin) {
    if (bin === 1) return "Pass";
    if (bin === 0) return "Fail";
    return "Ungraded";
  }

  function kindClass(mode) {
    if (ZP.kindClass) return ZP.kindClass(mode);
    const m = String(mode || "").toLowerCase();
    if (m === "rl" || m === "grpo") return "kind-rl";
    if (m === "sft") return "kind-sft";
    if (m === "explore" || m === "unique") return "kind-explore";
    if (m === "adaptive") return "kind-adaptive";
    return "";
  }

  function situationCaption(group) {
    const bits = [];
    if (group.novelty != null) {
      const n = Number(group.novelty);
      if (n > 0.7) bits.push("unusual in this batch");
      else if (n > 0.4) bits.push("somewhat different from the rest");
      else bits.push("similar to other prompts here");
    }
    if (group.diversity != null) {
      const d = Number(group.diversity);
      if (d > 0.5) bits.push("answers vary");
      else bits.push("answers stay close");
    }
    return bits.join(" · ");
  }

  function situationHeader(group) {
    const n = group.n ?? (group.rollouts || []).length;
    const nPass = group.n1 ?? group.n_pass ?? 0;
    const nFail = group.n0 ?? group.n_fail ?? 0;
    return `This situation · ${n} repeats · ${nPass} Pass · ${nFail} Fail`;
  }

  function matchesBin(r) {
    if (D.bin === "pass") return r.bin === 1;
    if (D.bin === "fail") return r.bin === 0;
    return true;
  }

  function filterRuns(runs) {
    const tags = [...D.tags].map((t) => t.toLowerCase());
    const q = (D.q || "").trim().toLowerCase();
    return (runs || []).filter((r) => {
      if (D.mode && !(ZP.matchesKindFilter
        ? ZP.matchesKindFilter(r, D.mode)
        : batchKind(r) === D.mode || (D.mode === "explore" && String(batchKind(r)) === "unique"))) return false;
      const have = (r.tags || []).map((t) => String(t).toLowerCase());
      if (tags.length && !tags.every((t) => have.includes(t))) return false;
      if (q) {
        const blob = [r.stem, r.mode, r.source, kindLabel(r), ...(r.tags || [])]
          .join(" ").toLowerCase().replaceAll("_", " ");
        if (!blob.includes(q)) return false;
      }
      return true;
    });
  }

  function tagChips(all, selected) {
    const hint = `<span class="muted">Tags are labels you put on a batch so you can keep only those batches.</span>`;
    if (!all.length) {
      return `${hint} <span class="muted">No tags yet.</span>`;
    }
    const chips = all.map((t) => {
      const on = selected.has(t);
      return `<button type="button" class="tag ${on ? "on" : ""}" data-filter="${esc(t)}">${esc(t)}${
        on ? `<span class="x" data-unfilter="${esc(t)}" title="Remove from filter">x</span>` : ""
      }</button>`;
    }).join("");
    return `${hint} ${chips}`;
  }

  async function saveTags(id, tags) {
    return api("/api/tags", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, tags }),
    });
  }

  async function downloadRun(id) {
    const out = await api("/api/download/run?id=" + encodeURIComponent(id));
    if (out.error) return out;
    downloadText(out.filename || "run.jsonl", out.jsonl || "", "application/x-ndjson");
    return out;
  }

  async function downloadHarness(agent) {
    const out = await api("/api/download/agent?agent=" + encodeURIComponent(agent));
    if (out.error) return out;
    downloadText(out.filename || agent + "-harness.json", out.harness || "{}", "application/json");
    return out;
  }

  async function uploadJsonl(ctx, file) {
    const agent = agentOf(ctx);
    const text = await file.text();
    return api("/api/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent,
        jsonl: text,
        filename: file.name,
      }),
    });
  }

  async function mergeRuns(agent, ids) {
    return api("/api/merge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent, ids }),
    });
  }

  function syncMergeBtn() {
    const btn = document.getElementById("merge-batches");
    if (!btn) return;
    const n = (D.picked instanceof Set ? D.picked.size : 0);
    btn.disabled = n < 2;
    btn.className = n >= 2 ? "go" : "ghost";
  }

  function rowHtml(r, agent) {
    if (!(D.picked instanceof Set)) D.picked = new Set();
    const title = batchTitle(r, agent);
    const picked = D.picked.has(r.id);
    const rate = ZP.passRate ? ZP.passRate(r.n0, r.n1) : passRate(r.n0, r.n1);
    const n1 = Number(r.n1) || 0;
    const n0 = Number(r.n0) || 0;
    const counts = (n0 + n1) > 0 ? `${n1} Pass · ${n0} Fail` : "";
    const tags = (r.tags || []).map((t) =>
      `<span class="tag">${esc(t)}<button type="button" class="x" data-drop="${esc(r.id)}" data-tag="${esc(t)}" title="Remove">x</button></span>`
    ).join("");
    const href = `#data/${encodeURIComponent(agent)}/${encodeURIComponent(r.stem)}`;
    return `<div class="block has-pick batch-card ${kindClass(batchKind(r))} ${picked ? "picked" : ""}" data-id="${esc(r.id)}" data-stem="${esc(r.stem)}">
      <a href="${href}">
        <div class="batch-top">
          <span class="batch-date">${esc(cardDate(r.mtime) || "n/a")}</span>
          <span class="n batch-rate">${esc(rate)}</span>
        </div>
        <div class="t">${esc(title)}</div>
        ${counts ? `<div class="m batch-counts">${esc(counts)}</div>` : ""}
      </a>
      <div class="batch-foot">
        <div class="tags">${tags}
          <input class="tag-in" data-edit="${esc(r.id)}" placeholder="add tag"/>
        </div>
        <button type="button" class="link" data-dl="${esc(r.id)}">Download JSONL</button>
        <label class="merge-pick" data-pick="${esc(r.id)}" title="Merge">
          <input type="checkbox" ${picked ? "checked" : ""} aria-label="Select batch to merge"/>
        </label>
      </div>
    </div>`;
  }

  async function renderList(ctx) {
    const main = ctx.main;
    const agent = agentOf(ctx);
    main.innerHTML = `<div class="page wide zp-data"><style>${STYLE}</style>
      <h1>Data</h1>
      <p class="sub">Loading batches for ${esc(agent)}.</p></div>`;
    const data = await api("/api/runs?agent=" + encodeURIComponent(agent));
    if (data.error && !(data.runs || []).length) {
      main.innerHTML = `<div class="page zp-data"><style>${STYLE}</style>
        <h1>Data</h1>
        <p class="err">${esc(data.error)}</p></div>`;
      return;
    }
    const all = data.runs || [];
    const used = [...new Set(all.flatMap((r) => r.tags || []))];
    D.tags = new Set([...D.tags].filter((t) => used.includes(t)));
    const shown = filterRuns(all);
    const totals = data.totals || {};
    const ingest = data.ingest;
    if (D.mode === "unique") D.mode = "explore";
    const modeBtns = ZP.kindFilterHtml
      ? ZP.kindFilterHtml(D.mode)
      : [
          { v: "", l: "All" },
          { v: "explore", l: "Explore" },
          { v: "rl", l: "Repeats" },
          { v: "sft", l: "Imitation" },
          { v: "adaptive", l: "Adaptive" },
        ].map((m) =>
          `<button type="button" data-mode="${esc(m.v)}" class="${D.mode === m.v ? "on" : ""}">${m.l}</button>`
        ).join("");

    const empty = shown.length
      ? `<div class="blocks">${shown.map((r) => rowHtml(r, agent)).join("")}</div>`
      : `<p class="empty">${all.length
        ? "No batches match these filters."
        : "No batches for this agent. Upload JSONL or simulate a new file."}</p>`;

    main.innerHTML = `<div class="page wide zp-data"><style>${STYLE}</style>
      <div class="row-head">
        <h1>Data</h1>
        <button type="button" class="ghost" id="dl-harness">Download harness</button>
        <label class="file">
          <input type="file" id="up-jsonl" accept=".jsonl,.json,.txt,application/json"/>
          <span class="muted">Upload JSONL into ${esc(agent)}</span>
        </label>
      </div>
      <p class="sub">${esc(agent)} · ${shown.length} of ${all.length} batches · ${totals.rows || 0} conversations ·
        Pass rate ${passRate(totals.n0, totals.n1)} ·
        <span class="z">${totals.n0 || 0} Fail</span> / <span class="o">${totals.n1 || 0} Pass</span>
        ${data.harness ? "" : " · no harness.json yet"}</p>
      <p class="muted">A run is one Simulate (or upload) sitting on this agent.</p>
      <div class="toolbar">
        <div class="seg" id="mode-seg">${modeBtns}</div>
        <input class="slim" id="q" type="text" placeholder="Filter file or tag" value="${esc(D.q)}"/>
        <button type="button" class="${(D.picked instanceof Set ? D.picked.size : 0) >= 2 ? "go" : "ghost"}" id="merge-batches" ${(D.picked instanceof Set ? D.picked.size : 0) < 2 ? "disabled" : ""}>Merge into one batch</button>
        <span class="grow"></span>
        <span class="err" id="err"></span>
      </div>
      <div class="filter">${tagChips(used, D.tags)}</div>
      ${empty}
    </div>`;

    const err = document.getElementById("err");
    const setErr = (m) => { if (err) err.textContent = m || ""; };

    main.onclick = async (e) => {
      const pick = e.target.closest("[data-pick]");
      if (pick) {
        e.stopPropagation();
        if (!(D.picked instanceof Set)) D.picked = new Set();
        const id = pick.dataset.pick;
        const box = pick.matches("input") ? pick : pick.querySelector("input");
        if (box && box.checked) D.picked.add(id);
        else D.picked.delete(id);
        const card = pick.closest(".block");
        if (card) card.classList.toggle("picked", D.picked.has(id));
        syncMergeBtn();
        return;
      }
      if (e.target.id === "merge-batches") {
        if (!(D.picked instanceof Set) || D.picked.size < 2) return;
        if (!window.confirm("This copies conversations onto disk. It will use more space.")) return;
        const out = await mergeRuns(agent, [...D.picked]);
        setErr(out.error || "");
        if (out.error) return;
        D.picked = new Set();
        renderList(ctx);
        return;
      }
      const unfilter = e.target.closest("[data-unfilter]");
      if (unfilter) {
        e.preventDefault();
        e.stopPropagation();
        D.tags.delete(unfilter.dataset.unfilter);
        renderList(ctx);
        return;
      }
      const filter = e.target.closest("[data-filter]");
      if (filter) {
        const t = filter.dataset.filter;
        if (D.tags.has(t)) D.tags.delete(t); else D.tags.add(t);
        renderList(ctx);
        return;
      }
      const modeBtn = e.target.closest("#mode-seg button");
      if (modeBtn) {
        D.mode = modeBtn.dataset.mode === "unique" ? "explore" : (modeBtn.dataset.mode || "");
        renderList(ctx);
        return;
      }
      if (e.target.id === "dl-harness") {
        const out = await downloadHarness(agent);
        setErr(out.error || "");
        return;
      }
      const drop = e.target.closest("[data-drop]");
      if (drop) {
        e.preventDefault();
        e.stopPropagation();
        const card = drop.closest(".block");
        const keep = [...(card || document).querySelectorAll(".tag [data-tag]")].map((btn) => btn.dataset.tag)
          .filter((t) => t && t.toLowerCase() !== drop.dataset.tag.toLowerCase());
        const out = await saveTags(drop.dataset.drop, keep);
        setErr(out.error || "");
        renderList(ctx);
        return;
      }
      const dl = e.target.closest("[data-dl]");
      if (dl) {
        e.stopPropagation();
        const out = await downloadRun(dl.dataset.dl);
        setErr(out.error || "");
        return;
      }
      if (e.target.closest("input, select, label")) return;
      const tr = e.target.closest("tr.click");
      if (tr) location.hash = ctx.href("data", agent) + "/" + encodeURIComponent(tr.dataset.stem);
    };

    const q = document.getElementById("q");
    if (q) {
      q.oninput = () => { D.q = q.value; };
      q.onkeydown = (e) => { if (e.key === "Enter") renderList(ctx); };
    }
    main.onkeydown = async (e) => {
      if (e.key !== "Enter") return;
      const inp = e.target.closest("input[data-edit]");
      if (!inp) return;
      const card = inp.closest(".block");
      const current = [...(card || document).querySelectorAll(".tag [data-tag]")].map((x) => x.dataset.tag);
      const add = inp.value.trim();
      const tags = add ? current.concat(add) : current;
      const out = await saveTags(inp.dataset.edit, tags);
      setErr(out.error || "");
      renderList(ctx);
    };
    const up = document.getElementById("up-jsonl");
    if (up) {
      up.onchange = async (e) => {
        const file = e.target.files && e.target.files[0];
        if (!file) return;
        setErr("");
        const out = await uploadJsonl(ctx, file);
        up.value = "";
        if (out.error) { setErr(out.error); return; }
        D.tags.clear();
        renderList(ctx);
      };
    }
  }

  function whoLabel(role) {
    const r = String(role || "").toLowerCase();
    if (r === "user") return "User";
    if (r === "assistant") return "Agent";
    if (r === "system") return "System";
    return r || "message";
  }

  function isTalkMessage(m) {
    if (!m || typeof m !== "object") return false;
    const role = String(m.role || "").toLowerCase();
    if (role === "tool" || role === "function") return false;
    if (role !== "user" && role !== "assistant" && role !== "system") return false;
    const text = String(m.content || "").trim();
    if (role === "assistant" && (m.tool_calls && m.tool_calls.length) && !text) return false;
    return Boolean(text);
  }

  function conversationHtml(roll, prompt) {
    const userText = String(prompt || roll.prompt || "").trim();
    if (rollWriterFailed(roll)) {
      return `${userText ? `<div class="turn"><div class="who">User</div><pre class="blob">${esc(userText)}</pre></div>` : ""}
        <div class="turn"><div class="who agent">Agent</div>${writerFailedHtml()}</div>`;
    }
    const msgs = (roll.messages || []).filter(isTalkMessage);
    const parts = [];
    const hasUser = msgs.some((m) => String(m.role || "").toLowerCase() === "user");
    if (!hasUser && userText) {
      parts.push(`<div class="turn"><div class="who">User</div><pre class="blob">${esc(userText)}</pre></div>`);
    }
    for (const m of msgs) {
      const role = String(m.role || "").toLowerCase();
      const body = isWriterFailed(m.content)
        ? writerFailedHtml()
        : `<pre class="blob">${esc(m.content)}</pre>`;
      parts.push(`<div class="turn"><div class="who ${role === "assistant" ? "agent" : ""}">${esc(whoLabel(role))}</div>${body}</div>`);
    }
    const hasAgent = msgs.some((m) => String(m.role || "").toLowerCase() === "assistant");
    if (!hasAgent && roll.final_text) {
      const body = isWriterFailed(roll.final_text)
        ? writerFailedHtml()
        : `<pre class="blob">${esc(roll.final_text)}</pre>`;
      parts.push(`<div class="turn"><div class="who agent">Agent</div>${body}</div>`);
    }
    return parts.join("") || `<p class="muted">No agent response.</p>`;
  }

  function traceTurns(roll) {
    const msgs = roll.messages || [];
    const fromMsgs = msgs.filter((m) => {
      const role = String(m.role || "").toLowerCase();
      return role === "tool" || role === "function" || (m.tool_calls && m.tool_calls.length);
    }).map((m) => {
      const role = m.role || "tool";
      const name = m.name ? ` ${m.name}` : "";
      const calls = (m.tool_calls || []).map((c) =>
        `<div class="muted">${esc(c.name || "tool")}</div><pre class="blob">${esc(pretty(c.arguments))}</pre>`
      ).join("");
      const body = m.content
        ? (isWriterFailed(m.content) ? writerFailedHtml() : `<pre class="blob">${esc(pretty(m.content))}</pre>`)
        : "";
      return `<div class="turn">
        <div class="who tool">${esc(role)}${esc(name)}</div>
        ${body}${calls}
      </div>`;
    });
    if (fromMsgs.length) return fromMsgs;
    return (roll.steps || []).filter((s) => s.tool).map((s) =>
      `<div class="turn"><div class="who tool">tool ${esc(s.tool)}</div>
        <pre class="blob">${esc(pretty(s.arguments))}</pre>
        <pre class="blob">${esc(pretty(s.result))}</pre></div>`
    );
  }

  function paintRun(ctx, d) {
    const main = ctx.main;
    const agent = agentOf(ctx);
    if (!(D.openRolls instanceof Set)) D.openRolls = new Set();
    const groups = d.groups || [];
    const base = D.splitOnly ? groups.filter((g) => g.split) : groups;
    let nPass = 0;
    let nFail = 0;
    for (const g of base) {
      for (const r of g.rollouts || []) {
        if (r.bin === 1) nPass += 1;
        else if (r.bin === 0) nFail += 1;
      }
    }
    const visible = base.map((g) => {
      const rollouts = (g.rollouts || []).filter(matchesBin);
      if (D.bin && !rollouts.length) return null;
      const n1 = rollouts.filter((r) => r.bin === 1).length;
      const n0 = rollouts.filter((r) => r.bin === 0).length;
      return { ...g, rollouts, n: rollouts.length, n1, n0, n_pass: n1, n_fail: n0 };
    }).filter(Boolean);
    const slice = visible.slice(0, D.shown);

    const ghtml = slice.map((group, i) => {
      const open = i === D.group;
      const rolls = open ? (group.rollouts || []).map((r, j) => {
        const expanded = D.openRolls instanceof Set && D.openRolls.has(j);
        const mark = r.bin === 1 ? "o" : r.bin === 0 ? "z" : "";
        const bit = r.verdict || gradeWord(r.bin);
        const why = rollWriterFailed(r) ? "Writer failed" : (r.reason_label || r.reason || "Ungraded");
        const traceKey = i + ":" + j;
        return `<div class="repeat">
          <button type="button" class="repeat-h ${expanded ? "open" : ""}" data-g="${i}" data-roll="${j}">
            <span class="chev">${expanded ? "▾" : "▸"}</span>
            <span class="repeat-k">Repeat ${j + 1}</span>
            <span class="${mark}">${esc(bit)}</span>
            <span class="muted">${esc(why)}</span>
          </button>
          ${expanded ? `<div class="repeat-body">
            ${conversationHtml(r, group.prompt)}
            <details class="adv" data-trace-key="${esc(traceKey)}">
              <summary>Show agent trace</summary>
              <div class="trace-slot"></div>
            </details>
          </div>` : ""}
        </div>`;
      }).join("") : "";
      return `<div class="group">
        <button type="button" class="sit-h ${open ? "open" : ""}" data-sit="${i}">
          <span class="chev">${open ? "▾" : "▸"}</span>
          <div>
            <div class="p">${esc(situationHeader(group))}</div>
            <div class="muted">${esc(group.prompt)}</div>
          </div>
          ${group.split ? `<span class="badge">contrasting</span>` : ""}
        </button>
        ${open ? `<div class="thread">${rolls || `<p class="muted">No repeats.</p>`}</div>` : ""}
      </div>`;
    }).join("");

    const tagList = (d.tags || []).map((t) =>
      `<span class="tag">${esc(t)}<button type="button" class="x" data-untag="${esc(t)}">x</button></span>`
    ).join("");

    main.innerHTML = `<div class="page wide zp-data"><style>${STYLE}</style>
      <p class="crumb"><a class="link" href="${ctx.href("data", agent)}">Data</a> · ${esc(kindLabel(d))} · ${esc(d.stem)}</p>
      <div class="row-head">
        <h1>${esc(kindLabel(d))}</h1>
        <button type="button" class="ghost" id="dl-jsonl">Download JSONL</button>
        <button type="button" class="ghost" id="dl-harness">Download harness</button>
      </div>
      ${kindNote(d) ? `<p class="sub">${esc(kindNote(d))}</p>` : ""}
      <div class="stats">
        <span>${d.n} conversations</span>
        <span>mean ${meanLabel(d.mean_reward)}</span>
        <span class="z">${d.n0} Fail</span>
        <span class="o">${d.n1} Pass</span>
        <span>${d.n_split} contrasting situations of ${d.prompts}</span>
      </div>
      <div class="toolbar">
        <div class="tags">${tagList}
          <input class="tag-in" id="run-tag" placeholder="add tag"/></div>
        <label class="muted" title="Only situations that have both a Pass and a Fail."><input type="checkbox" id="split-only" ${D.splitOnly ? "checked" : ""}/> Contrasting only</label>
        <span class="muted">${isExploreRun(d) ? `${visible.length} situations` : `${visible.length} questions. Same ask, several conversations.`}</span>
        <span class="err" id="err"></span>
      </div>
      <div class="seg" id="bin-seg" style="margin:0 0 14px;width:fit-content">
        <button type="button" data-bin="" class="${D.bin === "" ? "on" : ""}">All</button>
        <button type="button" data-bin="pass" class="${D.bin === "pass" ? "on" : ""}">Pass (${nPass})</button>
        <button type="button" data-bin="fail" class="${D.bin === "fail" ? "on" : ""}">Fail (${nFail})</button>
      </div>
      <div class="situations">
        ${ghtml || `<p class="empty">${D.bin ? "No conversations in this filter." : "Empty file."}</p>`}
        ${visible.length > D.shown ? `<button type="button" class="see-more" data-more="1">See more (${visible.length - D.shown} left)</button>` : ""}
        ${uniqueOrigin(d) ? `<p class="muted">${uniqueOrigin(d) === "deduped" ? "Deduped." : "Generated."}</p>` : ""}
      </div>
    </div>`;

    const err = document.getElementById("err");
    const setErr = (m) => { if (err) err.textContent = m || ""; };

    main.onclick = async (e) => {
      if (e.target.closest("details")) return;
      if (e.target.id === "dl-jsonl") {
        const out = await downloadRun(d.id);
        setErr(out.error || "");
        return;
      }
      if (e.target.id === "dl-harness") {
        const out = await downloadHarness(agent);
        setErr(out.error || "");
        return;
      }
      const binBtn = e.target.closest("#bin-seg button");
      if (binBtn) {
        D.bin = binBtn.dataset.bin || "";
        D.group = -1;
        D.roll = -1;
        D.openRolls = new Set();
        paintRun(ctx, d);
        return;
      }
      const more = e.target.closest("[data-more]");
      if (more) {
        D.shown += 20;
        paintRun(ctx, d);
        return;
      }
      const untag = e.target.closest("[data-untag]");
      if (untag) {
        const tags = (d.tags || []).filter((t) => t !== untag.dataset.untag);
        const out = await saveTags(d.id, tags);
        setErr(out.error || "");
        D.run = null;
        renderRun(ctx, d.stem);
        return;
      }
      const rollEl = e.target.closest("[data-roll]");
      if (rollEl) {
        const i = Number(rollEl.dataset.g);
        const j = Number(rollEl.dataset.roll);
        if (!(D.openRolls instanceof Set)) D.openRolls = new Set();
        if (D.group !== i) {
          D.group = i;
          D.openRolls = new Set([j]);
        } else if (D.openRolls.has(j)) {
          D.openRolls.delete(j);
        } else {
          D.openRolls.add(j);
        }
        D.roll = j;
        paintRun(ctx, d);
        return;
      }
      const header = e.target.closest("[data-sit]");
      if (header) {
        const i = Number(header.dataset.sit);
        if (D.group === i) {
          D.group = -1;
          D.roll = -1;
          D.openRolls = new Set();
        } else {
          D.group = i;
          D.openRolls = new Set();
          const rolls = (visible[i] && visible[i].rollouts) || [];
          if (rolls.length === 1) D.openRolls.add(0);
        }
        paintRun(ctx, d);
      }
    };
    main.querySelectorAll("details[data-trace-key]").forEach((det) => {
      det.addEventListener("toggle", () => {
        if (!det.open) return;
        const slot = det.querySelector(".trace-slot");
        if (!slot || slot.dataset.ready) return;
        const [gi, ri] = String(det.dataset.traceKey || "").split(":").map(Number);
        const group = visible[gi];
        const roll = group && (group.rollouts || [])[ri];
        slot.innerHTML = roll
          ? (traceTurns(roll).join("") || `<p class="muted">No tool calls this try.</p>`)
          : `<p class="muted">No tool calls this try.</p>`;
        slot.dataset.ready = "1";
      });
    });
    const split = document.getElementById("split-only");
    if (split) {
      split.onchange = () => {
        D.splitOnly = split.checked;
        D.group = -1;
        D.roll = -1;
        D.openRolls = new Set();
        paintRun(ctx, d);
      };
    }
    const tagIn = document.getElementById("run-tag");
    if (tagIn) {
      tagIn.onkeydown = async (e) => {
        if (e.key !== "Enter") return;
        const add = tagIn.value.trim();
        if (!add) return;
        const out = await saveTags(d.id, (d.tags || []).concat(add));
        setErr(out.error || "");
        D.run = null;
        renderRun(ctx, d.stem);
      };
    }
  }

  async function renderRun(ctx, stem) {
    const main = ctx.main;
    const agent = agentOf(ctx);
    const id = `outputs/studio-runs/agents/${agent}/runs/${stem}.jsonl`;
    if (!D.run || D.run.stem !== stem || D.run.agent !== agent) {
      main.innerHTML = `<div class="page wide zp-data"><style>${STYLE}</style>
        <p class="crumb"><a class="link" href="${ctx.href("data", agent)}">Data</a></p>
        <h1>Run</h1><p class="sub">Loading ${esc(stem)}.</p></div>`;
      const d = await api("/api/run?id=" + encodeURIComponent(id));
      if (d.error) {
        main.innerHTML = `<div class="page zp-data"><style>${STYLE}</style>
          <p class="crumb"><a class="link" href="${ctx.href("data", agent)}">Data</a></p>
          <p class="err">${esc(d.error)}</p></div>`;
        return;
      }
      D.run = d;
      D.group = -1;
      D.roll = -1;
      D.openRolls = new Set();
      D.shown = 20;
      D.bin = "";
    }
    paintRun(ctx, D.run);
  }

  ZP.pages.data = async function (ctx) {
    const main = ctx.main;
    if (needAgent(ctx, main)) return;
    const rest = (ctx.rest || []).filter(Boolean);
    if (rest.length) {
      await renderRun(ctx, decodeURIComponent(rest.join("/")));
      return;
    }
    D.run = null;
    D.bin = "";
    await renderList(ctx);
  };
})();
