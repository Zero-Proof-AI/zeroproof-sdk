window.ZP = window.ZP || { pages: {} };

(function () {
  const T = { tab: "packs", shown: 20, open: -1, run: "", seq: 0, mode: "" };

  const COPY =
    "This agent, these conversations. Good behavior pack = Passes. Contrasting repeats = same situation, Pass and Fail. Open See the conversations to see which pack each row is in. Check Audit on a batch first. An LLM judge can be attached later; it is not on by default.";

  function trainUrl(agent, run) {
    let url = "/api/train?agent=" + encodeURIComponent(agent);
    if (run) url += "&run=" + encodeURIComponent(run);
    return url;
  }

  function allLabel(n) {
    const v = Number(n) || 0;
    return "All · " + v + " conversations";
  }

  function esc(s) {
    if (typeof ZP.esc === "function") return ZP.esc(s);
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function download(filename, text) {
    const blob = new Blob([text || ""], { type: "application/x-ndjson" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename || "pack.jsonl";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
  }

  function runLabel(r, agent) {
    if (ZP.runTitle) return ZP.runTitle(r, agent);
    const kind = ZP.kindLabel ? ZP.kindLabel(r.mode) : (r.mode || "Run");
    return `${agent} · ${kind} · ${r.n ?? 0} conversations`;
  }

  function packTagsHtml(packs) {
    return (packs || []).map((p) => {
      const k = p === "Good behavior" ? "good" : p === "Contrasting" ? "contrast" : "neither";
      return `<span class="pack-tag ${k}">${esc(p)}</span>`;
    }).join("");
  }

  function heldWhy(id) {
    if (id === "untestable") return "Only one conversation for this situation.";
    if (id === "saturated") return "Every repeat passed. Nothing to contrast.";
    if (id === "all_zero") return "Every repeat failed. Nothing to copy.";
    if (id === "ungraded") return "Not graded yet. Open Grade first.";
    return "";
  }

  ZP.pages.train = async function (ctx) {
    const main = ctx.main;
    const agent = ctx.agent || ZP.agent || "";

    if (!agent) {
      main.innerHTML = `<div class="page">
        <h1>Train</h1>
        <p class="empty">Pick an agent first.</p>
        <a class="go" href="#agents">All agents</a>
      </div>`;
      return;
    }

    let restRun = (ctx.rest && ctx.rest[0]) || "";
    try { restRun = restRun ? decodeURIComponent(restRun) : ""; } catch (_) {}
    T.run = restRun === "all" ? "" : restRun;
    T.shown = 20;
    T.open = -1;

    main.innerHTML = `<div class="page"><h1>Train</h1><p class="sub">Building packs for ${esc(agent)}.</p></div>`;

    async function load() {
      const seq = ++T.seq;
      let d;
      try {
        const path = trainUrl(agent, T.run);
        d = await fetch(ZP.apiUrl ? ZP.apiUrl(path) : path, { credentials: "omit" }).then((r) => r.json());
      } catch (_) {
        if (seq !== T.seq) return;
        main.innerHTML = `<div class="page"><h1>Train</h1><p class="err">Could not load packs.</p></div>`;
        return;
      }
      if (seq !== T.seq) return;
      if (d.error && d.n == null) {
        main.innerHTML = `<div class="page"><h1>Train</h1>
          <p class="err">${esc(d.error)}</p>
          <a class="go" href="#agents">All agents</a></div>`;
        return;
      }
      paint(d);
    }

    function paint(d) {
      const sft = d.sft || {};
      const grpo = d.grpo || {};
      const held = d.held_out || {};
      const buckets = held.buckets || [];
      const groups = grpo.groups || [];
      const files = d.files || [];
      const sftOn = (sft.n || 0) > 0;
      const grpoOn = (grpo.n || 0) > 0;
      const mixPass = groups.reduce((s, g) => s + (g.n1 || 0), 0);
      const mixFail = groups.reduce((s, g) => s + (g.n0 || 0), 0);
      const repeatsLine = d.repeats_line || "";
      const want = T.run;
      if (T.mode === "unique") T.mode = "explore";
      const kindFiles = files.filter((f) => {
        if (ZP.matchesKindFilter && !ZP.matchesKindFilter(f, T.mode)) {
          const id = f.stem || f.id || "";
          return want && (want === id || want === f.stem || want === f.id);
        }
        return true;
      });
      const nAll = d.n_all != null ? d.n_all : files.reduce((s, f) => s + (f.n || 0), 0);
      const vis = (d.conversations && d.conversations.length) ? d.conversations : groups;
      const slice = vis.slice(0, T.shown);
      const chips = ZP.kindFilterHtml
        ? `<div class="seg" id="mode-seg" style="margin:0 0 12px;width:fit-content">${ZP.kindFilterHtml(T.mode)}</div>`
        : "";
      const runOpts = [`<option value="" ${!want ? "selected" : ""}>${esc(allLabel(nAll))}</option>`]
        .concat(kindFiles.map((f) => {
          const id = f.stem || f.id || "";
          const on = want === id || want === f.stem || want === f.id;
          return `<option value="${esc(id)}" ${on ? "selected" : ""}>${esc(runLabel(f, agent))}</option>`;
        })).join("");
      const selectedFile = files.find((f) => want && (want === (f.stem || "") || want === (f.id || ""))) || null;
      const kindNote = selectedFile && ZP.kindNote ? ZP.kindNote(selectedFile) : "";

      const sitHtml = slice.map((g) => {
        const rows = (g.rows || []).map((r, n) => {
          const word = r.bin === 1 ? "Pass" : r.bin === 0 ? "Fail" : "Ungraded";
          const tone = r.bin === 1 ? "o" : r.bin === 0 ? "z" : "";
          return `<div class="crow has-packs">
            <span class="${tone}">${n + 1}. ${word}</span>
            <span class="pack-tags">${packTagsHtml(r.packs)}</span>
            <span class="why">${esc((r.preview || "").slice(0, 80))}</span>
          </div>`;
        }).join("");
        return `<div class="sit">
          <div class="sit-head">
            <div class="t" style="font-size:16px">This situation · ${g.n} conversations · ${g.n1 || 0} Pass · ${g.n0 || 0} Fail</div>
            <div class="m">${esc((g.prompt || "").slice(0, 220))}</div>
          </div>
          <div class="compact">${rows || `<p class="empty">No conversations in this situation.</p>`}</div>
        </div>`;
      }).join("");

      let extra = "";
      if (T.tab === "conversations") {
        extra = `${sitHtml || `<p class="empty">No conversations in this batch.</p>`}
          ${T.shown < vis.length ? `<button type="button" class="see-more" data-more="1">See more (${vis.length - T.shown} left)</button>` : ""}`;
      } else if (T.tab === "more") {
        const heldBlocks = buckets.filter((b) => (b.n_rows || 0) > 0).map((b) =>
          `<div class="block">
            <div class="t" style="font-size:16px">${esc(heldWhy(b.id) || b.label)}</div>
            <div class="m">${b.n_prompts || 0} situations · ${b.n_rows || 0} conversations. Left out of Contrasting repeats.</div>
          </div>`
        ).join("");
        extra = `<details class="adv" open><summary>How the packs are filtered</summary>
            <p class="sub">A good behavior pack keeps conversations that passed. Contrasting repeats keeps situations where some repeats passed and some failed. ${esc(repeatsLine)}</p>
            <p class="sub">${esc(sft.filename || agent + ".sft.jsonl")} · ${esc(grpo.filename || agent + ".grpo.jsonl")}</p>
          </details>
          <details class="adv"><summary>Left out of Contrasting repeats</summary>
            <div class="blocks" style="margin-top:12px">${heldBlocks || `<p class="empty">Nothing held out.</p>`}</div>
          </details>`;
      }

      main.innerHTML = `<div class="page">
        <div class="row-head">
          <h1>Train</h1>
        </div>
        ${chips}
        <label class="block run-pick"><span>These conversations</span>
          <select id="train-run">${runOpts}</select>
        </label>
        ${kindNote ? `<p class="sub">${esc(kindNote)}</p>` : ""}
        <p class="sub">${esc(COPY)}</p>
        <p class="err">${esc(d.error && d.n != null ? d.error : "")}</p>
        <div class="packs">
          <div class="block">
            <div class="t">Good behavior pack</div>
            <div class="m">${esc(sft.filename || agent + ".sft.jsonl")}</div>
            <div class="n" style="margin-top:14px;color:var(--good)">${sft.n || 0}</div>
            <button type="button" class="go" id="dl-sft" style="margin-top:16px" ${sftOn ? "" : "disabled"}>Download</button>
          </div>
          <div class="block">
            <div class="t">Contrasting repeats</div>
            <div class="m">${esc(grpo.filename || agent + ".grpo.jsonl")}</div>
            <div class="n" style="margin-top:14px">${grpo.n_groups || 0}</div>
            <div class="m">${mixPass} Pass · ${mixFail} Fail</div>
            <button type="button" class="go" id="dl-grpo" style="margin-top:16px" ${grpoOn ? "" : "disabled"}>Download</button>
          </div>
        </div>
        <div class="kpis">
          <div class="block pass"><div class="n">${d.n1 || 0}</div><div class="m">Pass</div></div>
          <div class="block fail"><div class="n">${d.n0 || 0}</div><div class="m">Fail</div></div>
          <div class="block" title="Mean number of conversations per situation. Situations can differ.">
            <div class="n">${esc(d.mean_k_text || String(d.mean_k != null ? d.mean_k : 0))}</div>
            <div class="m">Average repeats / situation</div>
          </div>
        </div>
        ${repeatsLine ? `<p class="sub">${esc(repeatsLine)}</p>` : ""}
        <div class="subtabs">
          <button type="button" class="${T.tab === "packs" ? "on" : ""}" data-tab="packs">Packs</button>
          <button type="button" class="${T.tab === "conversations" ? "on" : ""}" data-tab="conversations">See the conversations</button>
          <button type="button" class="${T.tab === "more" ? "on" : ""}" data-tab="more">More</button>
        </div>
        ${T.tab === "packs"
          ? (sftOn || grpoOn
            ? `<p class="empty">Download a pack above. Open See the conversations to see which pack each row is in.</p>`
            : `<p class="empty">${(d.n_ungraded || 0) > 0 && (d.n1 || 0) === 0
              ? "Conversations are here but none passed. Open Grade."
              : "No packs yet. Simulate, Grade, then come back."}</p>`)
          : extra}
      </div>`;

      const sel = document.getElementById("train-run");
      if (sel) {
        sel.onchange = () => {
          T.run = sel.value;
          T.shown = 20;
          T.open = -1;
          const base = ctx.href ? ctx.href("train", agent) : ("#train/" + encodeURIComponent(agent));
          const dest = T.run ? base + "/" + encodeURIComponent(T.run) : base;
          if (location.hash !== dest) {
            location.hash = dest;
            return;
          }
          load();
        };
      }
      const dlSft = document.getElementById("dl-sft");
      const dlGrpo = document.getElementById("dl-grpo");
      if (dlSft) {
        dlSft.onclick = () => {
          if (!sft.jsonl) return;
          download(sft.filename || `${agent}.sft.jsonl`, sft.jsonl);
        };
      }
      if (dlGrpo) {
        dlGrpo.onclick = () => {
          if (!grpo.jsonl) return;
          download(grpo.filename || `${agent}.grpo.jsonl`, grpo.jsonl);
        };
      }
      main.onclick = (e) => {
        const modeBtn = e.target.closest("#mode-seg button");
        if (modeBtn) {
          T.mode = modeBtn.dataset.mode === "unique" ? "explore" : (modeBtn.dataset.mode || "");
          paint(d);
          return;
        }
        const tab = e.target.closest("[data-tab]");
        if (tab) {
          T.tab = tab.dataset.tab;
          paint(d);
          return;
        }
        if (e.target.closest("[data-more]")) {
          T.shown += 20;
          paint(d);
        }
      };
    }

    await load();
  };
})();
