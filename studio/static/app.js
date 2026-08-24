const PAGES = ["agents", "data", "simulate", "grade", "audit", "train"];

window.ZP = window.ZP || { pages: {}, agent: "" };

ZP.agent = sessionStorage.getItem("zp_agent") || localStorage.getItem("zp-agent") || "";
ZP.agentsList = ZP.agentsList || [];

ZP.canonicalKind = function (mode) {
  const m = String(mode || "").toLowerCase();
  if (m === "explore" || m === "unique") return "explore";
  if (m === "grpo") return "rl";
  return m;
};

ZP.KIND_FILTERS = [
  { v: "", l: "All" },
  { v: "explore", l: "Explore" },
  { v: "rl", l: "Repeats" },
  { v: "sft", l: "Imitation" },
  { v: "adaptive", l: "Adaptive" },
];

ZP.batchKind = function (r) {
  if (!r || typeof r !== "object") {
    return ZP.canonicalKind(r);
  }
  const stem = String(r.stem || r.name || "").toLowerCase();
  const source = String(r.source || "").toLowerCase();
  const tags = (r.tags || []).map((t) => String(t).toLowerCase());
  if (stem.includes("unique") || source.includes("unique") || tags.includes("unique")) {
    return "explore";
  }
  return ZP.canonicalKind(r.mode);
};

ZP.kindLabel = function (modeOrRun) {
  const m = ZP.batchKind(modeOrRun);
  if (m === "rl") return "Repeats";
  if (m === "sft") return "Imitation";
  if (m === "explore") return "Unique situations";
  if (m === "adaptive") return "Adaptive";
  return m || "Batch";
};

ZP.kindNote = function (modeOrRun) {
  const m = ZP.batchKind(modeOrRun);
  if (m === "explore") return "New situations, one try each.";
  if (m === "sft") return "Several wordings of the same ask.";
  if (m === "rl") return "Same wording, several independent tries.";
  if (m === "adaptive") return "Steer toward gaps.";
  return "";
};

ZP.uniqueOrigin = function (r) {
  if (!r || typeof r !== "object") return "";
  const stem = String(r.stem || r.name || "").toLowerCase();
  const source = String(r.source || "").toLowerCase();
  const tags = (r.tags || []).map((t) => String(t).toLowerCase());
  if (stem.includes("unique") || source.includes("unique") || tags.includes("unique")) {
    return "deduped";
  }
  return "";
};

ZP.matchesKindFilter = function (r, mode) {
  const want = ZP.canonicalKind(mode);
  if (!want) return true;
  return ZP.batchKind(r) === want;
};

ZP.kindFilterHtml = function (selected) {
  const esc = (s) => String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
  const cur = ZP.canonicalKind(selected);
  return ZP.KIND_FILTERS.map((m) => {
    const on = (m.v || "") === (cur || "");
    const note = m.v ? ZP.kindNote(m.v) : "";
    return `<button type="button" data-mode="${esc(m.v)}" class="${on ? "on" : ""}"${
      note ? ` title="${esc(note)}"` : ""}>${esc(m.l)}</button>`;
  }).join("");
};

ZP.gradeWord = function (bin) {
  if (bin === 1 || bin === "1") return "Pass";
  if (bin === 0 || bin === "0") return "Fail";
  if (bin === 0.5 || bin === "0.5") return "Half";
  return "Ungraded";
};

ZP.passRate = function (n0, n1) {
  const a = Number(n0) || 0;
  const b = Number(n1) || 0;
  if (a + b <= 0) return "—";
  return Math.round((100 * b) / (a + b)) + "%";
};

ZP.whenStamp = function (ts) {
  if (!ts) return "n/a";
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
};

ZP.batchMeta = function (r) {
  const bits = [
    ZP.whenStamp(r.mtime),
    `${r.n1 || 0} Pass`,
    `${r.n0 || 0} Fail`,
    `Pass rate ${ZP.passRate(r.n0, r.n1)}`,
  ];
  if (r.n_split) bits.push(`${r.n_split} with both pass and fail`);
  return bits.join(" · ");
};

ZP.runTitle = function (r, agent) {
  const who = agent || (r && r.agent) || ZP.agent || "agent";
  const n = (r && (r.n ?? r.n_rows)) || 0;
  return `${who} · ${ZP.kindLabel(r)} · ${n} conversations`;
};

function parseRoute() {
  const params = new URLSearchParams(location.search);
  const raw = (location.hash || "").replace(/^#\/?/, "");
  const parts = raw.split("/").filter(Boolean);
  let page = (parts[0] || params.get("page") || "agents").toLowerCase();
  if (!PAGES.includes(page)) page = "agents";
  let agent = parts[1] || params.get("agent") || ZP.agent || "";
  if (String(agent).includes("@")) agent = "";
  return { page, agent, rest: parts.slice(2), params };
}

ZP.kindClass = function (modeOrRun) {
  const m = ZP.batchKind ? ZP.batchKind(modeOrRun) : String(modeOrRun || "").toLowerCase();
  if (m === "rl" || m === "grpo") return "kind-rl";
  if (m === "sft") return "kind-sft";
  if (m === "explore" || m === "unique") return "kind-explore";
  if (m === "adaptive") return "kind-adaptive";
  return "";
};

function hashParts() {
  const raw = (location.hash || "").replace(/^#\/?/, "");
  return raw.split("/").filter(Boolean);
}

function href(page, agent) {
  if (page === "agents") return "#agents";
  const a = agent || ZP.agent;
  if (!a || String(a).includes("@")) return "#" + page;
  return "#" + page + "/" + encodeURIComponent(a);
}

function setAgent(name) {
  ZP.agent = String(name || "").includes("@") ? "" : String(name || "");
  if (ZP.agent) {
    sessionStorage.setItem("zp_agent", ZP.agent);
    try { localStorage.setItem("zp-agent", ZP.agent); } catch (_) {}
  } else {
    sessionStorage.removeItem("zp_agent");
    try { localStorage.removeItem("zp-agent"); } catch (_) {}
  }
  paintChrome();
}

function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function paintAgentPick() {
  const box = document.getElementById("agent-pick");
  if (!box) return;
  const agents = ZP.agentsList || [];
  if (!ZP.agent && !agents.length) {
    box.hidden = true;
    box.innerHTML = "";
    return;
  }
  box.hidden = false;
  const current = agents.find((a) => (a.id || a.name) === ZP.agent) || null;
  const opts = agents.length
    ? agents.map((a) => {
        const id = a.id || a.name;
        return `<option value="${esc(id)}" ${id === ZP.agent ? "selected" : ""}>${esc(a.name || id)}</option>`;
      }).join("")
    : `<option value="">No agents</option>`;
  box.innerHTML = `<label for="agent-select">Agent</label>
    <select id="agent-select">${opts}</select>`;
  const sel = document.getElementById("agent-select");
  if (sel) {
    sel.onchange = () => {
      const id = sel.value;
      if (!id) return;
      const { page } = parseRoute();
      setAgent(id);
      if (page === "agents") {
        const parts = hashParts();
        const onHub = parts[0] === "agents" && parts[1] && parts[1] !== "new";
        if (onHub) location.hash = "#agents/" + encodeURIComponent(id);
        else paintAgentPick();
        return;
      }
      const next = href(page, id);
      if (location.hash !== next) location.hash = next;
      else render();
    };
  }
}

function measureChrome() {
  const el = document.querySelector(".chrome");
  if (!el) return;
  const h = Math.ceil(el.getBoundingClientRect().height);
  document.documentElement.style.setProperty("--chrome", h + "px");
}

function paintChrome() {
  const { page } = parseRoute();
  const home = document.querySelector("a.home");
  if (home) {
    home.classList.toggle("on", page === "agents");
    home.href = "#agents";
  }
  document.querySelectorAll("nav a").forEach((a) => {
    if (a.classList.contains("soon")) return;
    a.classList.toggle("on", a.dataset.page === page);
    a.href = href(a.dataset.page);
  });
  paintAgentPick();
  measureChrome();
}

async function loadAgentsChrome() {
  try {
    const path = ZP.apiUrl ? ZP.apiUrl("/api/agents") : "/api/agents";
    const d = await fetch(path, { credentials: "omit" }).then((r) => r.json());
    ZP.agentsList = d.agents || [];
  } catch (_) {
    ZP.agentsList = ZP.agentsList || [];
  }
  paintAgentPick();
}

async function render() {
  const route = parseRoute();
  if (route.agent) setAgent(route.agent);
  paintChrome();
  const main = document.getElementById("main");
  const fn = ZP.pages[route.page];
  const ctx = {
    main,
    page: route.page,
    agent: ZP.agent,
    rest: route.rest,
    params: route.params,
    setAgent,
    href,
    esc,
  };
  if (typeof fn === "function") {
    await fn(ctx);
    paintAgentPick();
    measureChrome();
    return;
  }
  main.innerHTML = `<div class="page"><h1>${esc(route.page)}</h1>
    <p class="empty">This tab is not loaded.</p></div>`;
}

window.addEventListener("hashchange", render);
window.addEventListener("popstate", render);
window.addEventListener("resize", measureChrome);
if (window.ResizeObserver) {
  const chromeEl = document.querySelector(".chrome");
  if (chromeEl) new ResizeObserver(measureChrome).observe(chromeEl);
}
loadAgentsChrome();
if (!location.hash && !new URLSearchParams(location.search).get("page")) {
  location.hash = "#agents";
} else {
  render();
}
