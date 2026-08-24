(function () {
  const ZP = (window.ZP = window.ZP || { pages: {}, agent: "" });
  ZP.pages = ZP.pages || {};

  const state = {
    agents: [],
    selected: "",
    view: "list",
    editing: null,
    q: "",
    error: "",
    loading: false,
    formError: "",
    saving: false,
  };

  function esc(s) {
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function readStored() {
    try {
      return localStorage.getItem("zp-agent")
        || sessionStorage.getItem("zp-agent")
        || sessionStorage.getItem("zp_agent")
        || "";
    } catch (_) {
      return "";
    }
  }

  function remember(id, ctx) {
    const name = String(id || "");
    ZP.agent = name;
    if (ctx && typeof ctx.setAgent === "function") ctx.setAgent(name);
    else ZP.agent = name;
    try {
      if (name) {
        localStorage.setItem("zp-agent", name);
        sessionStorage.setItem("zp_agent", name);
        sessionStorage.setItem("zp-agent", name);
      } else {
        localStorage.removeItem("zp-agent");
        sessionStorage.removeItem("zp_agent");
        sessionStorage.removeItem("zp-agent");
      }
    } catch (_) {}
    window.dispatchEvent(new CustomEvent("zp:agent", { detail: { id: name, name } }));
  }

  async function api(path, opt) {
    let res;
    try {
      res = await fetch(ZP.apiUrl ? ZP.apiUrl(path) : path, { credentials: "omit", ...opt });
    } catch (err) {
      return { error: "Could not reach the local server. Is studio/serve.py running?" };
    }
    let data = {};
    try {
      data = await res.json();
    } catch (_) {
      data = { error: res.ok ? "empty response" : "bad response from " + path };
    }
    if (!res.ok && !data.error) data.error = res.statusText || "request failed";
    return data;
  }

  function when(ts) {
    if (!ts) return "-";
    const d = new Date(Number(ts) * 1000);
    if (Number.isNaN(d.getTime())) return "-";
    const p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  }

  function findAgent(id) {
    const key = String(id || "");
    return (state.agents || []).find((a) => (a.id || a.name) === key)
      || (state.agents || []).find((a) => String(a.id || a.name).toLowerCase() === key.toLowerCase())
      || null;
  }

  function filtered() {
    const q = state.q.trim().toLowerCase();
    const rows = state.agents || [];
    if (!q) return rows;
    return rows.filter((a) => {
      const names = (a.tool_names || []).join(" ");
      const blob = [a.id, a.name, a.policy, names].join(" ").toLowerCase();
      return blob.includes(q);
    });
  }

  function href(ctx, page, agent) {
    if (ctx && typeof ctx.href === "function") return ctx.href(page, agent || state.selected);
    const a = agent || state.selected || ZP.agent;
    if (!a || page === "agents") return "#" + page;
    return "#" + page + "/" + encodeURIComponent(a);
  }

  function hubIdFromHash() {
    const raw = (location.hash || "").replace(/^#\/?/, "");
    const parts = raw.split("/").filter(Boolean);
    if (parts[0] === "agents" && parts[1] && parts[1] !== "new") {
      try { return decodeURIComponent(parts[1]); } catch (_) { return parts[1]; }
    }
    return "";
  }

  function paintList(ctx) {
    const e = (ctx && ctx.esc) || esc;
    const hubId = hubIdFromHash();
    if (hubId && findAgent(hubId) && state.view !== "form") {
      state.selected = hubId;
      paintHub(ctx, findAgent(hubId));
      return;
    }
    const rows = filtered();
    const cards = rows.map((a) => {
      const id = a.id || a.name;
      const nTools = a.n_tools ?? (a.tool_names || []).length;
      const nRows = a.n_rows ?? 0;
      const rate = (a.n0 != null || a.n1 != null) && ((a.n0 || 0) + (a.n1 || 0))
        ? ` · ${Math.round(100 * (a.n1 || 0) / ((a.n0 || 0) + (a.n1 || 0)))}% pass`
        : "";
      return `<button type="button" class="block" data-id="${e(id)}">
        <div class="t">${e(a.name || id)}</div>
        <div class="m">${nTools} tools · ${nRows} conversations${rate}</div>
      </button>`;
    }).join("");

    let body;
    if (state.loading) {
      body = `<p class="empty">Loading agents.</p>`;
    } else if (state.error && !(state.agents || []).length) {
      body = `<p class="err">${e(state.error)}</p>
        <p class="empty">Agents live in outputs/studio-runs/agents/{id}/ with harness.json.</p>
        <button type="button" class="go" data-act="reload">Retry</button>`;
    } else if (!(state.agents || []).length) {
      body = `<p class="empty">No agents yet. Add one: name plus a few basic tools.</p>
        <button type="button" class="go" data-act="add">Add agent</button>`;
    } else if (!cards) {
      body = `<p class="empty">No agents match "${e(state.q)}".</p>`;
    } else {
      body = `<div class="blocks">${cards}</div>`;
    }

    const warn = state.error && (state.agents || []).length
      ? `<p class="err">${e(state.error)}</p>` : "";
    const filter = (state.agents || []).length
      ? `<label class="block" style="max-width:320px">
        <span>Filter</span>
        <input id="agent-q" type="text" value="${e(state.q)}" placeholder="name or tool"/>
      </label>` : "";

    ctx.main.innerHTML = `<div class="page wide">
      <div class="row-head">
        <h1>All agents</h1>
        <button type="button" class="go" data-act="add">Add agent</button>
      </div>
      <p class="sub">Pick an agent. Tools and instructions for this product. Click a block to go in.</p>
      ${warn}
      ${filter}
      ${body}
    </div>`;
    bindList(ctx);
  }

  function paintHub(ctx, a) {
    const e = (ctx && ctx.esc) || esc;
    const id = a.id || a.name;
    const nTools = a.n_tools ?? (a.tool_names || []).length;
    const tools = (a.tools || []).map((t) => {
      const n = toolName(t);
      const d = toolDesc(t);
      return `<div class="block"><div class="t" style="font-size:16px">${e(n)}</div><div class="m">${e(d)}</div></div>`;
    }).join("");
    const policy = a.policy || "";
    ctx.main.innerHTML = `<div class="page">
      <p class="sub"><button type="button" class="link" data-act="grid">All agents</button></p>
      <h1>${e(a.name || id)}</h1>
      <p class="sub">Tools → simulate coverage → grade responses → pack data. No weight training here yet.</p>
      <p class="sub">${nTools} tools · ${a.n_runs || 0} batches · ${a.n_rows || 0} conversations.
        ${a.error ? `<span class="err">${e(a.error)}</span>` : ""}</p>
      <div class="hub">
        <a class="block" href="${href(ctx, "data", id)}">
          <div class="t">Data <span class="chev" aria-hidden="true"></span></div>
          <div class="m">Conversations already on this agent. Upload JSONL or open a run.</div>
        </a>
        <a class="block" href="${href(ctx, "simulate", id)}">
          <div class="t">Simulate <span class="chev" aria-hidden="true"></span></div>
          <div class="m">Generate new conversations. Nothing starts until you click Simulate.</div>
        </a>
        <a class="block" href="${href(ctx, "grade", id)}">
          <div class="t">Grade <span class="chev" aria-hidden="true"></span></div>
          <div class="m">Mark each conversation Pass or Fail.</div>
        </a>
        <div class="block soon" aria-disabled="true">
          <div class="t">Audit</div>
          <div class="m">Is this batch usable? Quality check on the conversations, not a score of the agent.</div>
        </div>
        <div class="block soon" aria-disabled="true">
          <div class="t">Train</div>
          <div class="m">Download packs. You can download a good behavior pack. Contrasting repeats needs several tries of one ask with both Pass and Fail. This app does not train weights.</div>
        </div>
      </div>
      <div class="agent-acts" style="margin-top:22px">
        <button type="button" class="to" data-act="edit">Edit tools and instructions<span class="chev" aria-hidden="true"></span></button>
        <button type="button" class="dl-act" data-act="download"><span class="tray" aria-hidden="true"></span>Download tools and instructions</button>
      </div>
      <details class="adv"><summary>Tools and system prompt</summary>
        <div class="blocks" style="margin-top:12px">${tools || `<p class="empty">No tools on this harness.</p>`}</div>
        ${policy
          ? `<textarea readonly style="min-height:140px;margin-top:14px;font-family:var(--sans)">${e(policy)}</textarea>`
          : `<p class="err">No system prompt.</p>`}
      </details>
    </div>`;
    bindList(ctx);
  }

  function toolName(schema) {
    if (!schema || typeof schema !== "object") return "";
    const fn = schema.function && typeof schema.function === "object" ? schema.function : schema;
    return String(fn.name || schema.name || "").trim();
  }

  function toolDesc(schema) {
    if (!schema || typeof schema !== "object") return "";
    const fn = schema.function && typeof schema.function === "object" ? schema.function : schema;
    return String(fn.description || schema.description || "").trim();
  }

  const STARTERS = [
    { id: "read", label: "read", schema: { type: "function", function: { name: "read", description: "Read a file at a path.", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } } } },
    { id: "write", label: "write", schema: { type: "function", function: { name: "write", description: "Write text to a file path.", parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] } } } },
    { id: "summarize", label: "summarize", schema: { type: "function", function: { name: "summarize", description: "Summarize the given text.", parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"] } } } },
    { id: "web_search", label: "web search", schema: { type: "function", function: { name: "web_search", description: "Search the web for a query.", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } } } },
    { id: "list_dir", label: "list_dir", schema: { type: "function", function: { name: "list_dir", description: "List files in a folder.", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } } } },
    { id: "fetch_url", label: "fetch_url", schema: { type: "function", function: { name: "fetch_url", description: "Fetch a URL and return the text.", parameters: { type: "object", properties: { url: { type: "string" } }, required: ["url"] } } } },
  ];
  const AUTO_POLICY = "Use the tools. Do not invent file paths, ids, or search results. If a tool misses, tell the user.";

  function starterSchema(id) {
    const fromApi = ((state.starters && state.starters.basic) || []).find((b) => b.id === id);
    if (fromApi && fromApi.schema) return fromApi.schema;
    const local = STARTERS.find((s) => s.id === id);
    return local ? local.schema : null;
  }

  function parseToolsField() {
    const el = document.getElementById("atools");
    const raw = (el && el.value || "").trim();
    if (!raw) return [];
    try {
      const j = JSON.parse(raw);
      if (Array.isArray(j)) return j;
      if (j && Array.isArray(j.tools)) return j.tools;
    } catch (_) {}
    return null;
  }

  function setToolsField(tools) {
    const el = document.getElementById("atools");
    const text = JSON.stringify(tools || [], null, 2);
    if (el) el.value = text;
    state.formTools = text;
  }

  function mergeChipIntoJson(id, on) {
    const err = document.getElementById("aerr");
    const schema = starterSchema(id);
    if (!schema) return;
    const tools = parseToolsField();
    if (tools === null) {
      if (err) err.textContent = "Tools JSON is not valid JSON. Fix it, or clear the box and tick again.";
      return;
    }
    const name = toolName(schema);
    const next = tools.filter((t) => toolName(t) !== name);
    if (on) next.push(schema);
    setToolsField(next);
    const policyEl = document.getElementById("apolicy");
    if (policyEl && !String(policyEl.value || "").trim()) policyEl.value = AUTO_POLICY;
    if (err) err.textContent = "";
  }

  function applySpec(id) {
    const spec = ((state.starters && state.starters.specs) || []).find((s) => s.id === id);
    const err = document.getElementById("aerr");
    if (!spec) {
      if (err) err.textContent = "Could not load that spec.";
      return;
    }
    setToolsField(spec.tools || []);
    const policyEl = document.getElementById("apolicy");
    if (policyEl) policyEl.value = spec.policy || "";
    state.formPolicy = spec.policy || "";
    state.formSpec = id;
    state.formChips = [];
    document.querySelectorAll("#aspecs .chip").forEach((el) => {
      el.classList.toggle("on", el.dataset.spec === id);
    });
    document.querySelectorAll("#achips .chip").forEach((el) => {
      el.classList.remove("on");
      const inp = el.querySelector("input");
      if (inp) inp.checked = false;
    });
    if (err) err.textContent = "";
  }

  function paintForm(ctx) {
    const e = (ctx && ctx.esc) || esc;
    const editing = state.editing;
    const title = editing ? `Edit ${editing}` : "Add agent";
    const sub = editing
      ? "Replace tools JSON and the system prompt. Existing batches stay."
      : "Name the agent, then fill tools JSON and the system prompt three ways. Save writes harness.json and opens the hub.";
    const specs = (state.starters && state.starters.specs) || [];
    const specBtns = specs.map((s) =>
      `<button type="button" class="chip ${state.formSpec === s.id ? "on" : ""}" data-spec="${e(s.id)}">${e(s.name)}</button>`
    ).join("");
    const chips = STARTERS.map((s) => {
      const on = (state.formChips || []).includes(s.id);
      return `<label class="chip ${on ? "on" : ""}">
        <input type="checkbox" data-chip="${e(s.id)}" ${on ? "checked" : ""}/>
        ${e(s.label)}
      </label>`;
    }).join("");
    const fills = editing ? "" : `<div class="lab">Choose a spec</div>
        <div class="chips" id="aspecs">${specBtns || `<p class="empty">Specs did not load.</p>`}</div>
        <p class="sub">GitHub, Coding, Intercom, Linear, or Amazon. Picking one fills tools JSON and the system prompt. You can edit after.</p>
        <div class="lab">Or tick basic tools</div>
        <div class="chips" id="achips">${chips}</div>
        <p class="sub">read, write, summarize, web search. Each tick is merged into the tools JSON below.</p>
        <div class="lab">Or paste</div>`;
    ctx.main.innerHTML = `<div class="page">
      <div class="row-head">
        <h1>${e(title)}</h1>
        <button type="button" class="btn" data-act="cancel">Back</button>
      </div>
      <p class="sub">${e(sub)}</p>
      <form class="form" id="agent-form">
        <label class="block"><span>Name</span>
          <input id="aname" type="text" maxlength="64" ${editing ? "readonly" : ""}
            placeholder="desk-notes" value="${e(state.formName || "")}"/></label>
        ${fills}
        <label class="block"><span>Tools JSON</span>
          <textarea id="atools" placeholder='[{"type":"function","function":{"name":"read"}}]'>${e(state.formTools || "")}</textarea></label>
        <label class="block"><span>System prompt</span>
          <textarea id="apolicy" style="font-family:var(--sans);min-height:120px">${e(state.formPolicy || "")}</textarea></label>
        <label class="block"><span>Upload tools or spec.json</span>
          <input id="afile" type="file" accept=".json,application/json"/></label>
        <p class="err" id="aerr">${e(state.formError || "")}</p>
        <button class="go" type="submit" id="asave" ${state.saving ? "disabled" : ""}>${editing ? "Save" : "Save agent"}</button>
      </form>
    </div>`;
    bindForm(ctx);
  }

  async function openAdd(ctx) {
    state.view = "form";
    state.editing = null;
    state.formName = "";
    state.formTools = "";
    state.formPolicy = "";
    state.formChips = [];
    state.formSpec = "";
    state.formError = "";
    if (!state.starters || !(state.starters.specs || []).length) {
      const d = await api("/api/starters");
      state.starters = d && !d.error ? d : { specs: [], basic: STARTERS };
    }
    paintForm(ctx);
  }

  function bindList(ctx) {
    const main = ctx.main;
    main.onclick = (ev) => {
      const act = ev.target.closest("[data-act]");
      if (act) {
        const kind = act.dataset.act;
        if (kind === "add") {
          openAdd(ctx);
          return;
        }
        if (kind === "reload") {
          load(ctx);
          return;
        }
        if (kind === "edit") {
          const a = findAgent(state.selected);
          if (!a) return;
          openEdit(ctx, a);
          return;
        }
        if (kind === "download") {
          downloadHarness(findAgent(state.selected || hubIdFromHash()));
          return;
        }
        if (kind === "grid") {
          location.hash = "#agents";
          return;
        }
      }
      const card = ev.target.closest(".block[data-id]");
      if (!card) return;
      const id = card.dataset.id;
      state.selected = id;
      remember(id, ctx);
      location.hash = "#agents/" + encodeURIComponent(id);
    };
    const q = document.getElementById("agent-q");
    if (q) {
      q.oninput = () => {
        state.q = q.value;
        paintList(ctx);
        const again = document.getElementById("agent-q");
        if (again) {
          again.focus();
          const n = again.value.length;
          again.setSelectionRange(n, n);
        }
      };
    }
  }

  function bindForm(ctx) {
    const form = document.getElementById("agent-form");
    const err = document.getElementById("aerr");
    ctx.main.onclick = (ev) => {
      const act = ev.target.closest("[data-act]");
      if (act && act.dataset.act === "cancel") {
        state.view = "list";
        paintList(ctx);
        return;
      }
      const specBtn = ev.target.closest("#aspecs [data-spec]");
      if (specBtn) {
        ev.preventDefault();
        applySpec(specBtn.dataset.spec);
      }
    };
    const file = document.getElementById("afile");
    if (file) {
      file.onchange = async (ev) => {
        const f = ev.target.files && ev.target.files[0];
        if (!f) return;
        const text = await f.text();
        const toolsEl = document.getElementById("atools");
        const policyEl = document.getElementById("apolicy");
        const nameEl = document.getElementById("aname");
        try {
          const j = JSON.parse(text);
          if (j.policy && policyEl) policyEl.value = j.policy;
          if (j.system_prompt && policyEl && !policyEl.value.trim()) policyEl.value = j.system_prompt;
          if (j.name && nameEl && !state.editing && !nameEl.value.trim()) nameEl.value = j.name;
          if (Array.isArray(j.tools)) setToolsField(j.tools);
          else if (toolsEl) toolsEl.value = text;
        } catch (_) {
          if (toolsEl) toolsEl.value = text;
        }
      };
    }
    const chips = document.getElementById("achips");
    if (chips) {
      chips.onchange = (ev) => {
        const inp = ev.target.closest("[data-chip]");
        if (!inp) return;
        const id = inp.dataset.chip;
        const set = new Set(state.formChips || []);
        if (inp.checked) set.add(id);
        else set.delete(id);
        state.formChips = [...set];
        const lab = inp.closest(".chip");
        if (lab) lab.classList.toggle("on", inp.checked);
        mergeChipIntoJson(id, inp.checked);
      };
    }
    if (form) {
      form.onsubmit = (ev) => {
        ev.preventDefault();
        submitForm(ctx);
      };
    }
  }

  function openEdit(ctx, a) {
    state.view = "form";
    state.editing = a.id || a.name;
    state.formName = a.name || a.id;
    state.formTools = JSON.stringify(a.tools || [], null, 2);
    state.formPolicy = a.policy || "";
    state.formError = "";
    paintForm(ctx);
  }

  function downloadHarness(a) {
    if (!a) return;
    const blob = new Blob([JSON.stringify({
      name: a.name || a.id,
      tools: a.tools || [],
      policy: a.policy || "",
    }, null, 2) + "\n"], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = (a.id || a.name || "agent") + "-harness.json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 500);
  }

  async function submitForm(ctx) {
    const err = document.getElementById("aerr");
    const nameEl = document.getElementById("aname");
    const toolsEl = document.getElementById("atools");
    const policyEl = document.getElementById("apolicy");
    const name = (nameEl && nameEl.value || "").trim();
    const toolsRaw = (toolsEl && toolsEl.value || "").trim();
    const policy = (policyEl && policyEl.value || "").trim();
    if (!name) {
      if (err) err.textContent = "Name is required.";
      return;
    }
    if (!toolsRaw) {
      if (err) err.textContent = "Add tools: choose a spec, tick basic tools, or paste JSON.";
      return;
    }
    try { JSON.parse(toolsRaw); } catch (ex) {
      if (err) err.textContent = "tools JSON: " + ex.message;
      return;
    }
    if (!policy) {
      if (err) err.textContent = "System prompt is required.";
      return;
    }
    const payload = {
      name,
      tools: toolsRaw,
      policy,
      system_prompt: policy,
      replace: !!state.editing,
    };
    state.saving = true;
    const saveBtn = document.getElementById("asave");
    if (saveBtn) saveBtn.disabled = true;
    const out = await api("/api/agents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.saving = false;
    if (out.error) {
      if (err) err.textContent = out.error;
      if (saveBtn) saveBtn.disabled = false;
      return;
    }
    const id = out.id || out.name || name;
    state.view = "list";
    state.editing = null;
    state.selected = id;
    remember(id, ctx);
    location.hash = "#agents/" + encodeURIComponent(id);
    await load(ctx);
  }

  async function load(ctx) {
    state.loading = true;
    state.error = "";
    paintList(ctx);
    const data = await api("/api/agents");
    state.loading = false;
    if (data.error && !(data.agents || []).length) {
      state.error = data.error;
      state.agents = [];
      paintList(ctx);
      return;
    }
    state.agents = data.agents || (Array.isArray(data) ? data : []);
    if (data.error) state.error = data.error;
    if (!state.selected && ctx.agent && ctx.agent !== "new") state.selected = ctx.agent;
    if (!state.selected) state.selected = readStored();
    if (state.selected && !findAgent(state.selected)) {
      state.selected = "";
      remember("", ctx);
    }
    if (state.selected) remember(state.selected, ctx);
    paintList(ctx);
  }

  async function render(ctx) {
    const main = (ctx && ctx.main) || document.getElementById("main");
    if (!main) return;
    const localCtx = {
      main,
      esc: (ctx && ctx.esc) || esc,
      href: (ctx && ctx.href) || ((page, agent) => href({ href: null }, page, agent)),
      setAgent: (ctx && ctx.setAgent) || ((name) => { ZP.agent = String(name || ""); }),
      agent: (ctx && ctx.agent) || ZP.agent || readStored(),
      rest: (ctx && ctx.rest) || [],
    };
    const origSet = localCtx.setAgent;
    localCtx.setAgent = function (name) {
      origSet(name);
      remember(name, { setAgent: origSet });
    };

    if (localCtx.agent === "new") {
      if (ctx && ctx.setAgent) ctx.setAgent("");
      await openAdd(localCtx);
      return;
    }

    state.view = "list";

    if (localCtx.agent && localCtx.agent !== "new") {
      state.selected = localCtx.agent;
    } else if (!state.selected) {
      state.selected = readStored();
    }

    await load(localCtx);
  }

  ZP.pages.agents = render;
  ZP.agents = {
    render,
    list: () => state.agents.slice(),
    selected: () => state.selected,
  };
})();
