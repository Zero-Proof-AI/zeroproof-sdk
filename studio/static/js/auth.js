window.ZP = window.ZP || {};

const DEFAULT_EMAIL = "sahana@zeroproofai.com";
const STUB_NOTE = "Stub login. Clerk will replace this.";
const STUB_KEY = "zp_stub_email";
const API_STORE = "STUDIO_API_URL";

function resolveStudioApiUrl() {
  try {
    const params = new URLSearchParams(location.search);
    const q = String(params.get("api") || params.get("studio_api") || "").trim();
    if (q) {
      const url = q.replace(/\/$/, "");
      try { localStorage.setItem(API_STORE, url); } catch (_) {}
      return url;
    }
  } catch (_) {}
  const baked = String(window.STUDIO_API_URL || "").trim().replace(/\/$/, "");
  if (baked) return baked;
  try {
    const stored = String(localStorage.getItem(API_STORE) || "").trim();
    if (stored) return stored.replace(/\/$/, "");
  } catch (_) {}
  return "";
}

function studioApiUrl(path) {
  const base = String(window.STUDIO_API_URL || "").trim().replace(/\/$/, "");
  const p = String(path || "");
  if (base && p.startsWith("/api/")) return base + p;
  return p;
}

window.ZP.apiUrl = studioApiUrl;

(function bindApiBase() {
  const base = resolveStudioApiUrl();
  window.STUDIO_API_URL = base;
  window.ZP.apiUrl = studioApiUrl;
  if (!base) return;
  const orig = window.fetch;
  window.fetch = function (input, init) {
    if (typeof input === "string" && input.startsWith("/api/")) {
      input = base + input;
      init = Object.assign({}, init || {});
      if (!init.credentials || init.credentials === "same-origin") {
        init.credentials = "omit";
      }
      const headers = new Headers(init.headers || {});
      if (base.indexOf("ngrok") !== -1) {
        headers.set("ngrok-skip-browser-warning", "1");
      }
      init.headers = headers;
    }
    return orig.call(this, input, init);
  };
})();

function authEsc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function readStubEmail() {
  try {
    const fromLs = localStorage.getItem(STUB_KEY);
    if (fromLs) return fromLs;
  } catch (_) {}
  const m = document.cookie.match(/(?:^|;\s*)zp_stub_email=([^;]*)/);
  if (m) {
    try {
      return decodeURIComponent(m[1]);
    } catch (_) {
      return m[1];
    }
  }
  return "";
}

function writeStubEmail(email) {
  try {
    localStorage.setItem(STUB_KEY, email);
  } catch (_) {}
  document.cookie =
    STUB_KEY + "=" + encodeURIComponent(email) + "; path=/; SameSite=Lax";
}

function clearStubEmail() {
  try {
    localStorage.removeItem(STUB_KEY);
  } catch (_) {}
  document.cookie = STUB_KEY + "=; path=/; max-age=0";
}

function paintSession(email) {
  const box = document.getElementById("session");
  if (!box) return;
  const e = String(email || "").trim();
  ZP.email = e;
  if (e) {
    writeStubEmail(e);
    box.innerHTML = `<span>${authEsc(e)}</span><span>${authEsc(STUB_NOTE)}</span>
      <button type="button" class="link" id="logout">Sign out</button>`;
  } else {
    box.innerHTML = `<form id="login-form" class="session-form">
        <input type="email" id="login-email" name="email" value="${authEsc(DEFAULT_EMAIL)}" placeholder="you@company.com" autocomplete="username"/>
        <button type="submit" class="link">Continue</button>
      </form>
      <span>${authEsc(STUB_NOTE)}</span>`;
  }
  const login = document.getElementById("login-form");
  if (login) {
    login.onsubmit = (ev) => {
      ev.preventDefault();
      const input = document.getElementById("login-email");
      const next = String((input && input.value) || "").trim() || DEFAULT_EMAIL;
      paintSession(next);
    };
  }
  const outBtn = document.getElementById("logout");
  if (outBtn) {
    outBtn.onclick = () => {
      clearStubEmail();
      paintSession("");
    };
  }
}

paintSession(readStubEmail());
