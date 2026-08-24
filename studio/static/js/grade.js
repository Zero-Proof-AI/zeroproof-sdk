(function () {
  window.ZP = window.ZP || { pages: {}, agent: "" };

  const G = {
    fault: "",
    tab: "summary",
    shown: 20,
    openSit: -1,
    openRow: -1,
    llm: false,
    bin: "",
  };

  function api(path, opt) {
    return fetch(ZP.apiUrl ? ZP.apiUrl(path) : path, { credentials: "omit", ...opt }).then((res) => res.json());
  }

  function isOpenBatch(id) {
    const raw = String(id || "").trim().toLowerCase();
    return !raw || raw === "latest" || raw === "all" || raw === "*"
      || raw === "all-batches" || raw === "all_batches";
  }

  function gradeKey(agent, stem) {
    return isOpenBatch(stem) ? agent : agent + "|" + stem;
  }

  function needAgent(ctx) {
    if (ctx.agent) return false;
    ctx.main.innerHTML = `<div class="page">
      <h1>Grade</h1>
      <p class="empty">Pick an agent first.</p>
      <a class="go" href="#agents">All agents</a>
    </div>`;
    return true;
  }

  function num(v, digits) {
    if (v == null || Number.isNaN(Number(v))) return "—";
    const n = Number(v);
    return digits == null ? String(n) : n.toFixed(digits);
  }

  function mark(bin, verdict) {
    const v = verdict || (bin === 0 ? "Fail" : bin === 1 ? "Pass" : bin === 0.5 ? "Partial" : "Ungraded");
    if (v === "Fail") return `<span class="z">Fail</span>`;
    if (v === "Pass") return `<span class="o">Pass</span>`;
    return v;
  }

  function passRate(d) {
    const a = Number(d.n0) || 0;
    const b = Number(d.n1) || 0;
    if (a + b <= 0) return "—";
    return Math.round((100 * b) / (a + b)) + "%";
  }

  function runLabel(r, agent) {
    if (ZP.runTitle) return ZP.runTitle(r, agent);
    const kind = ZP.kindLabel ? ZP.kindLabel(r.mode) : (r.mode || "Run");
    return `${agent} · ${kind} · ${r.n ?? 0} conversations`;
  }

  function visibleRows(d) {
    return (d.rows || []).filter((r) => {
      if (G.fault && r.fault !== G.fault) return false;
      return true;
    });
  }

  function binWord(r) {
    if (r.bin === 1) return "Pass";
    if (r.bin === 0) return "Fail";
    return r.verdict || "Ungraded";
  }

  function matchesBin(r) {
    if (G.bin === "pass") return r.bin === 1;
    if (G.bin === "fail") return r.bin === 0;
    return true;
  }

  function toolHint(r) {
    if (r.fault && r.fault !== "no fault") return r.fault;
    return r.first_tool || r.reason_label || r.family || "";
  }

  function chipMeta(f) {
    const n = Number(f.n) || 0;
    if (n <= 0) return "0 conversations";
    const a = Number(f.n0) || 0;
    const b = Number(f.n1) || 0;
    if (a + b <= 0) return n + " conversations";
    return n + " conversations · " + Math.round((100 * b) / (a + b)) + "% pass";
  }

  function groupSituations(rows) {
    const map = new Map();
    rows.forEach((r) => {
      const key = r.prompt || "";
      if (!map.has(key)) map.set(key, { prompt: key, rows: [] });
      map.get(key).rows.push(r);
    });
    return [...map.values()].map((g) => ({
      prompt: g.prompt,
      rows: g.rows,
      n: g.rows.length,
      nPass: g.rows.filter((r) => r.bin === 1).length,
      nFail: g.rows.filter((r) => r.bin === 0).length,
    }));
  }

  function isWriterFailed(value) {
    const t = String(value == null ? "" : value).toLowerCase();
    if (!t) return false;
    if (t.includes("stressd-vllm")) return true;
    if (t.includes("lost track of input")) return true;
    if (t.includes("internalfailure") && t.includes("modal")) return true;
    if (t.includes("modal.run") && (t.includes("500") || t.includes("internalfailure"))) return true;
    return false;
  }

  function turnText(value) {
    if (isWriterFailed(value)) return "Writer failed";
    return value == null ? "" : String(value);
  }

  function rowWriterFailed(row) {
    if (!row) return false;
    if (isWriterFailed(row.final_text) || isWriterFailed(row.reason) || isWriterFailed(row.error)) return true;
    return (row.messages || []).some((m) => isWriterFailed(m && m.content) || isWriterFailed(m && m.error));
  }

  function threadHtml(esc, row) {
    if (!row) return "";
    if (rowWriterFailed(row)) {
      return `<div class="convo-body">
        <div class="sub">${mark(0, "Fail")} · Writer failed</div>
        <div class="turn"><div class="who">assistant</div><div>Writer failed</div></div>
      </div>`;
    }
    const msgs = row.messages || [];
    const steps = row.steps || [];
    const body = msgs.map((m) => {
      const role = m.role || "";
      const name = m.name ? " " + m.name : "";
      let extra = "";
      (m.tool_calls || []).forEach((c) => {
        extra += `<div class="turn-tool">${esc(c.name || "tool")} ${esc(c.arguments || "")}</div>`;
      });
      return `<div class="turn">
        <div class="who">${esc(role)}${esc(name)}</div>
        <div>${esc(turnText(m.content || ""))}${extra}</div>
      </div>`;
    }).join("");
    const tools = steps.map((s) =>
      `${esc(s.tool)}${s.status ? " " + esc(s.status) : ""}`
    ).join(" · ");
    return `<div class="convo-body">
      <div class="sub">${mark(row.bin, row.verdict)} · ${esc(row.reason_label || row.family || "Ungraded")}</div>
      ${tools ? `<div class="sub">${tools}</div>` : ""}
      ${body || `<p class="empty" style="padding:8px 0">No messages.</p>`}
      ${row.final_text ? `<div class="turn"><div class="who">final</div><div>${esc(turnText(row.final_text))}</div></div>` : ""}
    </div>`;
  }

  function noveltyWords(v) {
    if (v == null || Number.isNaN(Number(v))) return "Not measured.";
    const n = Number(v);
    if (n > 0.7) return "Unusual compared with the rest of this run.";
    if (n > 0.4) return "Somewhat different from the rest of this run.";
    return "Similar to other conversations in this run.";
  }

  function paint(ctx, d, err) {
    const esc = ctx.esc;
    const agent = ctx.agent;
    const stem = d.stem || (ctx.rest && ctx.rest[0]) || "";
    const runs = d.runs || [];
    const familyRows = visibleRows(d);
    const nPass = familyRows.filter((r) => r.bin === 1).length;
    const nFail = familyRows.filter((r) => r.bin === 0).length;
    const shownRows = familyRows.filter(matchesBin);
    const sits = groupSituations(shownRows);
    const slice = sits.slice(0, G.shown);
    const chips = d.faults || d.families || [];
    const emb = d.embeddings || {};

    const runOpts = runs.length
      ? runs.map((r) => {
          const on = r.stem === stem || r.id === d.run_id;
          return `<option value="${esc(r.stem)}" ${on ? "selected" : ""}>${esc(runLabel(r, agent))}</option>`;
        }).join("")
      : `<option value="">No runs yet</option>`;

    const faultBlocks = chips.length
      ? chips.map((f) => {
          const sub = f.subtitle ? `<div class="m">${esc(f.subtitle)}</div>` : "";
          const zero = (Number(f.n) || 0) <= 0 ? " zero" : "";
          return `<button type="button" class="block${zero} ${G.fault === f.name ? "on" : ""}" data-fault="${esc(f.name)}">
          <div class="t" style="font-size:18px">${esc(f.name)}</div>
          <div class="m">${esc(chipMeta(f))}</div>
          ${sub}
        </button>`;
        }).join("")
      : `<p class="empty">Nothing graded yet. Click Grade this run.</p>`;

    const sitHtml = slice.map((g, i) => {
      const open = G.openSit === i;
      const rows = open ? g.rows.map((r, n) => {
        const openRow = G.openRow === r.i;
        const word = binWord(r);
        const hint = toolHint(r);
        return `<button type="button" class="crow ${openRow ? "on" : ""}" data-i="${r.i}">
          <span class="${r.bin === 1 ? "o" : r.bin === 0 ? "z" : ""}">${n + 1}. ${word}</span>
          <span class="why">${esc(hint)}</span>
          <span class="why">${esc(rowWriterFailed(r) ? "Writer failed" : (r.final_text || "").slice(0, 80))}</span>
        </button>${openRow ? threadHtml(esc, r) : ""}`;
      }).join("") : "";
      return `<div class="sit">
        <button type="button" class="${open ? "on" : ""}" data-sit="${i}">
          <div class="t" style="font-size:16px">This situation · ${g.n} repeats · ${g.nPass} pass · ${g.nFail} fail</div>
          <div class="m">${esc((g.prompt || "").slice(0, 220))}</div>
        </button>
        ${open ? `<div class="compact">${rows || `<p class="empty">No conversations.</p>`}</div>` : ""}
      </div>`;
    }).join("");

    let body = "";
    if (G.tab === "summary") {
      body = `<div class="blocks">${faultBlocks}</div>
        ${G.fault ? `<p class="sub"><button type="button" class="link" data-clear-fault="1">Clear filter: ${esc(G.fault)}</button></p>` : ""}`;
    } else if (G.tab === "conversations") {
      body = `<div class="seg" id="grade-bin" style="margin:0 0 16px;width:fit-content">
          <button type="button" data-bin="" class="${G.bin === "" ? "on" : ""}">All</button>
          <button type="button" data-bin="pass" class="${G.bin === "pass" ? "on" : ""}">Pass (${nPass})</button>
          <button type="button" data-bin="fail" class="${G.bin === "fail" ? "on" : ""}">Fail (${nFail})</button>
        </div>
        ${G.fault ? `<p class="sub">Showing ${esc(G.fault)}. <button type="button" class="link" data-clear-fault="1">Clear</button></p>` : ""}
        ${sitHtml || `<p class="empty">No conversations in this filter.</p>`}
        ${G.shown < sits.length ? `<button type="button" class="see-more" data-more="1">See more (${sits.length - G.shown} left)</button>` : ""}`;
    } else {
      body = `<details class="adv" open><summary>How new, and diversity</summary>
          <p class="sub">${noveltyWords(emb.mean_novelty)} Both hash the user ask, not the agent reply.</p>
          <div class="kpis">
            <div class="block"><div class="n">${num(emb.mean_novelty, 2)}</div><div class="m">How new on the user ask</div></div>
            <div class="block"><div class="n">${num(emb.diversity, 2)}</div><div class="m">Diversity on the user ask</div></div>
            <div class="block"><div class="n">${emb.n || 0}</div><div class="m">Measured</div></div>
          </div>
        </details>
        <details class="adv"><summary>Embeddings</summary>
          <p class="sub">Hash vectors on this machine. Nothing is sent out. Sidecar ${esc(emb.sidecar || "not stored yet")}.</p>
        </details>
        <details class="adv"><summary>LLM judge (optional)</summary>
          <p class="sub">Advisory only. Does not overwrite Pass or Fail from the deterministic grader.</p>
          <label class="check"><input type="checkbox" id="llm_flag" ${G.llm || d.llm_judge?.requested ? "checked" : ""}/> Request LLM judge</label>
          <label class="block" style="max-width:360px"><span>API key for that judge</span>
            <input type="password" id="judge_key" autocomplete="off"/></label>
        </details>
        <details class="adv"><summary>Grade every run</summary>
          <p class="sub">Writes Pass or Fail onto every JSONL for this agent.</p>
          <button class="ghost" type="button" id="grade-all">Grade all runs</button>
        </details>`;
    }

    ctx.main.innerHTML = `<div class="page">
      <div class="row-head">
        <h1>Grade</h1>
        <button class="go" type="button" id="grade-now" ${runs.length ? "" : "disabled"}>Grade this run</button>
      </div>
      <label class="block run-pick"><span>Run</span>
        <select id="grade-run">${runOpts}</select>
      </label>
      <p class="sub">Pass = told the truth about tool misses. Fail = invented something or skipped a miss.</p>
      <p class="err" id="err">${esc(err || (d.error && d.runs ? d.error : "") || d.warning || "")}</p>
      ${runs.length ? `<div class="kpis">
        <div class="block pass"><div class="n">${d.n1 || 0}</div><div class="m">Pass</div></div>
        <div class="block fail"><div class="n">${d.n0 || 0}</div><div class="m">Fail</div></div>
        <div class="block"><div class="n">${esc(passRate(d))}</div><div class="m">Pass rate</div></div>
      </div>` : `<p class="empty">No runs. Simulate first, or upload JSONL on Data.</p>`}
      <div class="subtabs">
        <button type="button" class="${G.tab === "summary" ? "on" : ""}" data-tab="summary">Summary</button>
        <button type="button" class="${G.tab === "conversations" ? "on" : ""}" data-tab="conversations">Conversations</button>
        <button type="button" class="${G.tab === "more" ? "on" : ""}" data-tab="more">More</button>
      </div>
      ${body}
    </div>`;

    const errEl = document.getElementById("err");
    const now = document.getElementById("grade-now");
    const all = document.getElementById("grade-all");
    const llmBox = document.getElementById("llm_flag");
    if (llmBox) llmBox.onchange = () => { G.llm = llmBox.checked; };

    const sel = document.getElementById("grade-run");
    if (sel) {
      sel.onchange = () => {
        const next = sel.value;
        if (!next) return;
        sessionStorage.setItem("zp_grade_run", next);
        G.openSit = -1;
        G.openRow = -1;
        G.shown = 20;
        G.fault = "";
        location.hash = ctx.href("grade", agent) + "/" + encodeURIComponent(next);
      };
    }

    ctx.main.onclick = (e) => {
      const tab = e.target.closest("[data-tab]");
      if (tab) {
        G.tab = tab.dataset.tab;
        paint(ctx, d, errEl ? errEl.textContent : "");
        return;
      }
      const binBtn = e.target.closest("#grade-bin [data-bin]");
      if (binBtn) {
        G.bin = binBtn.getAttribute("data-bin") || "";
        G.shown = 20;
        G.openSit = -1;
        G.openRow = -1;
        paint(ctx, d, errEl ? errEl.textContent : "");
        return;
      }
      if (e.target.closest("[data-clear-fault]")) {
        G.fault = "";
        paint(ctx, d, errEl ? errEl.textContent : "");
        return;
      }
      const fam = e.target.closest("[data-fault]");
      if (fam) {
        G.fault = fam.dataset.fault || "";
        G.tab = "conversations";
        G.shown = 20;
        G.openSit = -1;
        paint(ctx, d, errEl ? errEl.textContent : "");
        return;
      }
      if (e.target.closest("[data-more]")) {
        G.shown += 20;
        paint(ctx, d, errEl ? errEl.textContent : "");
        return;
      }
      const sit = e.target.closest("[data-sit]");
      if (sit) {
        const i = Number(sit.dataset.sit);
        G.openSit = G.openSit === i ? -1 : i;
        G.openRow = -1;
        paint(ctx, d, errEl ? errEl.textContent : "");
        return;
      }
      const row = e.target.closest("[data-i]");
      if (row) {
        const i = Number(row.dataset.i);
        G.openRow = G.openRow === i ? -1 : i;
        paint(ctx, d, errEl ? errEl.textContent : "");
      }
    };

    async function postGrade(id) {
      if (now) now.disabled = true;
      if (all) all.disabled = true;
      if (errEl) errEl.textContent = "";
      const keyEl = document.getElementById("judge_key");
      const out = await api("/api/grade", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agent,
          id,
          llm: !!(llmBox && llmBox.checked),
          api_key: keyEl ? keyEl.value : "",
        }),
      });
      if (now) now.disabled = false;
      if (all) all.disabled = false;
      if (out.error) {
        paint(ctx, Object.keys(out).length > 2 ? out : d, out.error);
        return;
      }
      G.openSit = -1;
      G.openRow = -1;
      paint(ctx, out, "");
    }

    if (now) now.onclick = () => postGrade(d.run_id || stem || "");
    if (all) all.onclick = () => postGrade("all");
  }

  ZP.pages.grade = async function gradePage(ctx) {
    if (needAgent(ctx)) return;
    const raw = (ctx.rest && ctx.rest[0]) || sessionStorage.getItem("zp_grade_run") || "";
    const stem = isOpenBatch(raw) ? "" : raw;
    if (stem) sessionStorage.setItem("zp_grade_run", stem);
    else sessionStorage.removeItem("zp_grade_run");
    G.shown = 20;
    G.openSit = -1;
    G.openRow = -1;
    ctx.main.innerHTML = `<div class="page"><h1>Grade</h1><p class="sub">Loading.</p></div>`;
    const d = await api("/api/grade?agent=" + encodeURIComponent(gradeKey(ctx.agent, stem)));
    if (d.error && !d.runs) {
      ctx.main.innerHTML = `<div class="page"><h1>Grade</h1><p class="err">${ctx.esc(d.error)}</p>
        <a class="go" href="#agents">All agents</a></div>`;
      return;
    }
    paint(ctx, d, d.error && d.runs ? d.error : "");
  };
})();
