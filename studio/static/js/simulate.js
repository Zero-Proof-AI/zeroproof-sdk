(function () {
  // Hosted Qwen uses VLLM_API_KEY on the API server. Do not require a paste.
  const ZP = (window.ZP = window.ZP || { pages: {}, agent: "" });
  ZP.pages = ZP.pages || {};

  const MODE_COPY = {
    explore: "New situations, one try each.",
    sft: "Several wordings of the same ask.",
    rl: "Same wording, several independent tries.",
    adaptive: "Steer toward gaps.",
  };

  const JOB_KEYS = ["zp-job", "zp_job"];

  const state = {
    source: "existing",
    brain: "hosted",
    stop: "time",
    mode: "rl",
    job: null,
    fromStore: false,
    misses: 0,
    poll: null,
    apiKey: "",
    baseUrl: "",
    model: "",
    setAgent: null,
    href: null,
  };

  const $ = (root, id) => (root && root.querySelector("#" + id)) || document.getElementById(id);

  function esc(s) {
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  async function api(path, opt) {
    const url = (ZP.apiUrl ? ZP.apiUrl(path) : path);
    const res = await fetch(url, { credentials: "omit", ...opt });
    let body;
    try { body = await res.json(); } catch (_) { body = { error: "bad response from " + path }; }
    if (res.status === 404 && body && !body.status) body.status = "not_found";
    return body;
  }

  function savedJob() {
    try {
      return localStorage.getItem("zp-job")
        || localStorage.getItem("zp_job")
        || sessionStorage.getItem("zp-job")
        || sessionStorage.getItem("zp_job")
        || "";
    } catch (_) { return ""; }
  }

  function rememberJob(id) {
    const next = String(id || "").trim();
    state.job = next || null;
    try {
      JOB_KEYS.forEach((key) => {
        if (next) {
          localStorage.setItem(key, next);
          sessionStorage.setItem(key, next);
        } else {
          localStorage.removeItem(key);
          sessionStorage.removeItem(key);
        }
      });
    } catch (_) {}
  }

  function jobGone(j) {
    if (!j) return true;
    if (j.status === "idle" || j.status === "not_found") return true;
    if (j.error === "unknown job") return true;
    return false;
  }

  function selectedAgent() {
    return String(ZP.agent || sessionStorage.getItem("zp_agent") || localStorage.getItem("zp-agent") || "");
  }

  function rememberOwn(root) {
    const own = $(root, "sim-api-key");
    const url = $(root, "sim-base-url");
    const model = $(root, "sim-model");
    if (own) state.apiKey = own.value;
    if (url) state.baseUrl = url.value;
    if (model) state.model = model.value;
  }

  function bindSeg(root, id, key, onChange) {
    const box = $(root, id);
    if (!box) return;
    box.onclick = (e) => {
      const b = e.target.closest("button[data-v]");
      if (!b) return;
      state[key] = b.dataset.v;
      box.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      rememberOwn(root);
      if (onChange) onChange();
    };
  }

  function toolCount(text) {
    try {
      const parsed = JSON.parse(text || "[]");
      const tools = Array.isArray(parsed) ? parsed : parsed.tools;
      return Array.isArray(tools) ? tools.length : 0;
    } catch (_) { return 0; }
  }

  function fillHarness(root, spec) {
    const tools = $(root, "sim-tools");
    const policy = $(root, "sim-policy");
    const list = (spec && (spec.tools || (spec.harness && spec.harness.tools))) || [];
    const prompt = (spec && (spec.policy || spec.system_prompt || (spec.harness && spec.harness.policy))) || "";
    if (tools) tools.value = JSON.stringify(list || [], null, 2);
    if (policy) policy.value = prompt || "";
  }

  async function loadAgentHarness(root, name) {
    if (!name) { fillHarness(root, { tools: [], policy: "" }); return; }
    const spec = await api("/api/agent?name=" + encodeURIComponent(name));
    if (spec.error) {
      const err = $(root, "sim-err");
      if (err) err.textContent = spec.error;
      return;
    }
    fillHarness(root, spec);
  }

  function passRate(j) {
    const n0 = Number(j.n0), n1 = Number(j.n1);
    if (Number.isFinite(n0) && Number.isFinite(n1) && n0 + n1 > 0) {
      return Math.round((100 * n1) / (n0 + n1)) + "%";
    }
    const tail = j.tail || [];
    const graded = tail.filter((t) => t.reward != null);
    if (!graded.length) return "—";
    const p = graded.filter((t) => t.reward >= 1).length;
    return Math.round((100 * p) / graded.length) + "%";
  }

  function gradeMark(reward) {
    if (reward >= 1) return `<span class="pf pass">Pass</span>`;
    if (reward != null && reward <= 0) return `<span class="pf fail">Fail</span>`;
    return `<span class="muted">Ungraded</span>`;
  }

  function endpointDown(msg) {
    const t = String(msg || "").toLowerCase();
    if (!t) return false;
    if (t.includes("endpoint isn't up")) return true;
    if (t.includes("lost track of input")) return true;
    if (t.includes("internalfailure")) return true;
    if (t.includes("stressd-vllm")) return true;
    if (t.includes("hosted qwen dropped")) return true;
    if (t.includes("hosted not connected")) return true;
    if (t.includes("modal.run") && (t.includes("500") || t.includes("502") || t.includes("503"))) return true;
    return false;
  }

  function failHeadline(j) {
    return endpointDown(j && j.error) ? "The endpoint isn't up" : "Stopped";
  }

  function href(page, agent) {
    if (typeof state.href === "function") return state.href(page, agent);
    const a = agent || selectedAgent();
    if (!a || page === "agents") return "#" + page;
    return "#" + page + "/" + encodeURIComponent(a);
  }

  function paintIdle(root) {
    const box = $(root, "sim-live");
    if (box) box.innerHTML = "";
  }

  function enableGo(root) {
    const go = $(root, "sim-go");
    if (go) go.disabled = !!state.job;
  }

  function paintLive(root, j) {
    const box = $(root, "sim-live");
    if (!box) return;
    if (jobGone(j) || (!state.job && j.status !== "done" && j.status !== "error")) {
      paintIdle(root);
      return;
    }
    const done = j.status === "done";
    const fail = j.status === "error";
    const queued = j.status === "queued";
    const rows = j.rows ?? 0;
    const cap = state.stop === "rows" ? Number($(root, "sim-budget")?.value || j.budget || 0) : 0;
    const tcap = state.stop === "time" ? Number($(root, "sim-time")?.value || j.time_budget || 0) : 0;
    let pct = 16;
    if (done || fail) pct = 100;
    else if (cap > 0) pct = Math.max(8, Math.min(96, Math.round((rows / cap) * 100)));
    else if (tcap > 0 && j.total_s != null) pct = Math.max(8, Math.min(96, Math.round((Number(j.total_s) / tcap) * 100)));
    const title = fail ? failHeadline(j) : done ? "Done" : queued ? "Queued" : "Running";
    const elapsed = j.total_s != null ? Number(j.total_s).toFixed(0) + "s" : "—";
    const agent = j.agent || selectedAgent();
    const tail = (j.tail || []).map((t) =>
      `<div class="msg">${gradeMark(t.reward)} ${esc((t.prompt || "").slice(0, 160))}</div>`
    ).join("");
    const after = done ? `<p style="margin-top:16px">
        <a class="link" href="${href("data", agent)}">Open in Data</a>
        ·
        <a class="link" href="${href("grade", agent)}">Open in Grade</a>
      </p>
      <p class="sub">${esc(ZP.runTitle ? ZP.runTitle({ mode: state.mode, n: j.n || rows, agent }, agent) : "")}
        ${j.n0 != null ? ` · ${j.n0} fail / ${j.n1} pass` : ""}
        ${j.output ? `<br/><code>${esc(j.output)}</code>` : ""}</p>` : "";
    box.innerHTML = `<h2>${esc(title)}</h2>
      <div class="sim-kpis">
        <div><div class="n">${rows}</div><div class="l">Conversations</div></div>
        <div><div class="n">${esc(passRate(j))}</div><div class="l">Pass rate</div></div>
        <div><div class="n">${esc(elapsed)}</div><div class="l">Elapsed</div></div>
      </div>
      <div class="meter ${done || fail ? "done" : ""}"><i style="width:${pct}%"></i></div>
      <p class="sub">${j.stage ? esc(j.stage) : (queued ? "Queued" : "Live")}
        ${j.inflight != null ? " · " + j.inflight + " in flight" : ""}</p>
      ${fail ? `<p class="err">${esc(endpointDown(j.error) ? "The endpoint isn't up" : (j.error || "run failed"))}</p>` : ""}
      <div class="sim-tail">${tail || (done ? "" : `<p class="empty" style="padding:8px 0">Waiting for the first conversation.</p>`)}</div>
      ${after}`;
  }

  function stopPoll() {
    if (state.poll) { clearInterval(state.poll); state.poll = null; }
  }

  async function pollLive(root) {
    if (!state.job) return;
    const j = await api("/api/job?id=" + encodeURIComponent(state.job));
    if (jobGone(j)) {
      state.misses = (state.misses || 0) + 1;
      if (state.misses < 8) return;
    } else {
      state.misses = 0;
    }
    const stale = jobGone(j) || (state.fromStore && (j.status === "done" || j.status === "error"));
    if (stale) {
      state.fromStore = false;
      rememberJob(null);
      stopPoll();
      paintIdle(root);
      enableGo(root);
      return;
    }
    state.fromStore = false;
    paintLive(root, j);
    if (j.status === "done" || j.status === "error") {
      rememberJob(null);
      stopPoll();
      enableGo(root);
      if (j.agent) {
        if (typeof state.setAgent === "function") state.setAgent(j.agent);
        else ZP.agent = j.agent;
      }
    }
  }

  function startPoll(root) {
    stopPoll();
    state.poll = setInterval(() => pollLive(root), 1000);
    pollLive(root);
  }

  function applyStop(root) {
    const timeField = $(root, "sim-time-field");
    const rowsField = $(root, "sim-rows-field");
    if (timeField) timeField.style.display = state.stop === "time" ? "" : "none";
    if (rowsField) rowsField.style.display = state.stop === "rows" ? "" : "none";
  }

  function applyBrain(root) {
    const own = $(root, "sim-own");
    if (own) own.style.display = state.brain === "own" ? "" : "none";
    const note = $(root, "sim-hosted-note");
    if (note) {
      note.textContent = state.brain === "hosted"
        ? "Uses VLLM_API_KEY on the API server. You do not paste it here."
        : "";
    }
    const go = $(root, "sim-go");
    if (go && !state.job) go.disabled = false;
    const conc = $(root, "sim-concurrency");
    const concNote = $(root, "sim-concurrency-note");
    if (conc) {
      conc.max = state.brain === "hosted" ? 32 : 192;
      if (state.brain === "hosted" && Number(conc.value) > 32) conc.value = 32;
    }
    if (concNote) {
      concNote.textContent = state.brain === "hosted"
        ? "16 leaves headroom if another hosted job is running. 32 is the max per job when you are the only client."
        : "";
    }
  }

  function applyMode(root) {
    const note = $(root, "sim-mode-note");
    if (note) note.textContent = MODE_COPY[state.mode] || "";
    root.querySelectorAll(".kind").forEach((el) => {
      el.classList.toggle("on", el.dataset.v === state.mode);
    });
    const kWrap = $(root, "sim-k-main");
    if (kWrap) kWrap.style.display = state.mode === "rl" ? "" : "none";
    const nWrap = $(root, "sim-n-main");
    if (nWrap) nWrap.style.display = state.mode === "sft" ? "" : "none";
  }

  async function onSubmit(root, e) {
    e.preventDefault();
    const err = $(root, "sim-err");
    if (err) err.textContent = "";
    rememberOwn(root);
    const agent = selectedAgent();
    if (!agent) {
      if (err) err.textContent = "Pick an agent in the dropdown.";
      return;
    }
    const tools = $(root, "sim-tools")?.value || "";
    const policy = $(root, "sim-policy")?.value || "";
    if (state.brain === "own" && !($(root, "sim-base-url")?.value || "").trim()) {
      if (err) err.textContent = "Base URL is required for your model.";
      return;
    }
    const go = $(root, "sim-go");
    if (go) go.disabled = true;
    const kVal = $(root, "sim-k")?.value;
    const nVal = $(root, "sim-phrasings")?.value;
    const body = {
      source: "existing",
      agent,
      name: agent,
      tools,
      system_prompt: policy,
      tags: $(root, "sim-tags")?.value || "",
      stop: state.stop,
      time_budget: Number($(root, "sim-time")?.value || 60),
      budget: Number($(root, "sim-budget")?.value || 200),
      mode: state.mode,
      brain: state.brain,
      base_url: state.brain === "own" ? ($(root, "sim-base-url")?.value || "") : "",
      model: state.brain === "own" ? ($(root, "sim-model")?.value || "") : "",
      api_key: state.brain === "own" ? state.apiKey : "",
      concurrency: Number($(root, "sim-concurrency")?.value || 16),
      fault_rate: Number($(root, "sim-fault")?.value),
      avg_turns: Number($(root, "sim-turns")?.value),
      phrasings: state.mode === "sft" && nVal ? Number(nVal) : null,
      repeats: state.mode === "rl" ? Number(kVal || 4) : null,
      k: state.mode === "rl" ? Number(kVal || 4) : null,
      grade: !!$(root, "sim-grade")?.checked,
    };
    const json = await api("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (json.error) {
      if (err) err.textContent = endpointDown(json.error) ? "The endpoint isn't up" : json.error;
      rememberJob(null);
      enableGo(root);
      return;
    }
    if (json.agent && typeof state.setAgent === "function") state.setAgent(json.agent);
    state.fromStore = false;
    rememberJob(json.id);
    paintLive(root, { status: "queued", rows: 0, stage: "setup", tail: [], agent: json.agent, output: json.output });
    startPoll(root);
  }

  function recentTable(runs, agent) {
    const rows = (runs || []).slice(0, 8);
    if (!rows.length) return `<p class="empty">No runs yet. Simulate writes the first file here.</p>`;
    return `<table class="db"><thead><tr>
      <th>Batch</th><th>Conversations</th><th>Fail</th><th>Pass</th><th>Contrasting repeats</th>
    </tr></thead><tbody>${rows.map((r) => {
      const title = ZP.runTitle ? ZP.runTitle(r, agent) : (r.stem || "");
      const note = ZP.kindNote ? ZP.kindNote(r) : "";
      return `<tr>
        <td>${esc(title)}${note ? `<div class="m">${esc(note)}</div>` : ""}<div><code>${esc(r.stem || "")}</code></div></td>
        <td>${r.n ?? 0}</td>
        <td class="z">${r.n0 ?? 0}</td>
        <td class="o">${r.n1 ?? 0}</td>
        <td>${r.n_split ?? 0}</td>
      </tr>`;
    }).join("")}</tbody></table>`;
  }

  async function render(ctx) {
    const root = (ctx && ctx.main) || document.getElementById("main");
    state.setAgent = ctx && ctx.setAgent;
    state.href = ctx && ctx.href;
    const current = (ctx && ctx.agent) || selectedAgent();
    const pack = await api("/api/agents");
    ZP.agentsList = pack.agents || pack || [];
    let runs = [];
    if (current) {
      const listed = await api("/api/runs?agent=" + encodeURIComponent(current));
      runs = listed.runs || [];
    }

    root.innerHTML = `<div class="page desk">
      <div class="sim-head">
        <h1>Simulate</h1>
      </div>
      <p class="sub">Generate conversations for the agent in the dropdown. Nothing starts until you click Simulate.</p>

      <div class="sim-canvas">
        <section class="sim-col">
          <h2>What you generate</h2>
          <p class="sub" id="sim-mode-note">${esc(MODE_COPY[state.mode])}</p>
          <form id="sim-form">
            <div class="kinds" id="sim-mode">
              <button type="button" class="kind kind-explore ${state.mode === "explore" ? "on" : ""}" data-v="explore">
                <span class="t">Explore</span>
                <span class="d">New situations, one try each.</span>
              </button>
              <button type="button" class="kind kind-sft ${state.mode === "sft" ? "on" : ""}" data-v="sft">
                <span class="t">Imitation</span>
                <span class="d">Several wordings of the same ask.</span>
              </button>
              <button type="button" class="kind kind-rl ${state.mode === "rl" ? "on" : ""}" data-v="rl">
                <span class="t">Repeats</span>
                <span class="d">Same wording, several independent tries.</span>
              </button>
              <button type="button" class="kind kind-adaptive ${state.mode === "adaptive" ? "on" : ""}" data-v="adaptive">
                <span class="t">Adaptive</span>
                <span class="d">Steer toward gaps.</span>
              </button>
            </div>

            <div class="lab">Cap</div>
            <div class="cap-row">
              <div class="seg" id="sim-stop">
                <button type="button" data-v="time" class="${state.stop === "time" ? "on" : ""}">Time</button>
                <button type="button" data-v="rows" class="${state.stop === "rows" ? "on" : ""}">Conversations</button>
              </div>
              <label class="block" id="sim-time-field" style="margin:0"><span>Seconds</span>
                <input type="number" id="sim-time" value="60" min="1"/></label>
              <label class="block" id="sim-rows-field" style="margin:0;display:none"><span>Conversation cap</span>
                <input type="number" id="sim-budget" value="200" min="1"/></label>
            </div>

            <div class="lab">Model</div>
            <div class="seg" id="sim-brain">
              <button type="button" data-v="hosted" class="${state.brain === "hosted" ? "on" : ""}">Hosted Qwen</button>
              <button type="button" data-v="own" class="${state.brain === "own" ? "on" : ""}">Your model</button>
            </div>
            <p class="sub" id="sim-hosted-note" style="margin-top:10px"></p>
            <div id="sim-own" style="display:none;margin-top:8px">
              <label class="block"><span>Base URL</span>
                <input type="url" id="sim-base-url" placeholder="https://.../v1"/></label>
              <div class="grid">
                <label class="block"><span>Model name</span><input type="text" id="sim-model"/></label>
                <label class="block"><span>API key (your endpoint only)</span>
                  <input type="password" id="sim-api-key" autocomplete="off"/></label>
              </div>
            </div>

            <label class="check">
              <input type="checkbox" id="sim-grade" checked/>
              Grade while generating. Each conversation gets Pass or Fail as it is written.
            </label>

            <details class="adv"><summary>Advanced</summary>
              <div class="grid" style="margin-top:10px">
                <label class="block" id="sim-k-main"><span>Repeats per prompt</span>
                  <input type="number" id="sim-k" value="4" min="2" max="32"/></label>
                <label class="block" id="sim-n-main" style="display:none"><span>Phrasings</span>
                  <input type="number" id="sim-phrasings" placeholder="3" min="1" max="32"/></label>
                <label class="block"><span>Concurrency</span>
                  <input type="number" id="sim-concurrency" value="16" min="1" max="32"/>
                  <p class="sub" id="sim-concurrency-note" style="margin-top:6px">16 leaves headroom if another hosted job is running. 32 is the max per job when you are the only client.</p></label>
                <label class="block"><span>Fault rate</span>
                  <input type="number" id="sim-fault" value="0.5" min="0" max="1" step="0.1"/></label>
                <label class="block"><span>Avg turns</span>
                  <input type="number" id="sim-turns" value="4" min="1" max="32"/></label>
                <label class="block"><span>Tags</span>
                  <input id="sim-tags" type="text" placeholder="holdout"/></label>
              </div>
            </details>

            <details class="harness"><summary>Harness</summary>
              <label class="block"><span>Tools</span>
                <textarea id="sim-tools"></textarea></label>
              <label class="block"><span>System prompt</span>
                <textarea id="sim-policy"></textarea></label>
            </details>

            <p class="err" id="sim-err"></p>
            <button class="go sim-go" type="submit" id="sim-go">Simulate</button>
          </form>
        </section>
      </div>

      <section class="sim-below">
        <div id="sim-live"></div>
        <div id="sim-recent" style="margin-top:18px">
          <h2>Last batches</h2>
          ${current ? recentTable(runs, current) : `<p class="empty">Pick an agent in the dropdown first.</p>`}
        </div>
      </section>
    </div>`;

    applyStop(root);
    applyBrain(root);
    applyMode(root);
    fetch(ZP.apiUrl ? ZP.apiUrl("/api/auth/status") : "/api/auth/status", { credentials: "omit" }).then((r) => r.json()).then((st) => {
      ZP.hasHostedKey = !!st.has_hosted_key;
      applyBrain(root);
    }).catch(() => {});

    const ownKey = $(root, "sim-api-key");
    const url = $(root, "sim-base-url");
    const model = $(root, "sim-model");
    if (ownKey) ownKey.value = state.apiKey;
    if (url) url.value = state.baseUrl;
    if (model) model.value = state.model;

    const kinds = $(root, "sim-mode");
    if (kinds) {
      kinds.onclick = (e) => {
        const b = e.target.closest(".kind");
        if (!b) return;
        state.mode = b.dataset.v;
        applyMode(root);
      };
    }
    bindSeg(root, "sim-stop", "stop", () => applyStop(root));
    bindSeg(root, "sim-brain", "brain", () => applyBrain(root));
    $(root, "sim-form").onsubmit = (ev) => onSubmit(root, ev);

    if (current) await loadAgentHarness(root, current);
    if (!state.job) {
      const saved = savedJob();
      if (saved) {
        state.fromStore = true;
        rememberJob(saved);
      }
    }
    if (state.job) {
      const go = $(root, "sim-go");
      if (go) go.disabled = true;
      startPoll(root);
    }
  }

  ZP.pages.simulate = render;
})();
