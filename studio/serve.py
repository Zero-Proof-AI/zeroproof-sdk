#!/usr/bin/env python3
"""ZeroProof studio shell. Dispatches /api/* to studio.api handlers.

    .venv/bin/python studio/serve.py
    open http://127.0.0.1:8765

Binds 0.0.0.0 so Render / Fly / ngrok can reach it. CORS allows the
marketing site (zeroproofai.com), Vercel previews, and localhost.
Hosted Qwen reads VLLM_API_KEY from process env (.env loaded at boot).
The browser never needs that key.
"""
from __future__ import annotations

import json
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STUDIO = Path(__file__).resolve().parent
STATIC = STUDIO / "static"
sys.path.insert(0, str(STUDIO))

from api import agents, audit, data, grade, simulate, train  # noqa: E402
from api import auth  # noqa: E402
from token_gate import check_key, record_usage, InvalidKey, QuotaExceeded  # noqa: E402

HOST = os.environ.get("STUDIO_HOST") or "0.0.0.0"
PORT = int(os.environ.get("PORT") or os.environ.get("STUDIO_PORT") or "8765")

_VERCEL = re.compile(r"^https://([a-z0-9-]+\.)*vercel\.app$", re.I)
_LOCAL = re.compile(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$", re.I)
_SITE = re.compile(r"^https://([a-z0-9-]+\.)?zeroproofai\.com$", re.I)


def cors_origin(origin: str | None) -> str | None:
    """Echo Origin for the marketing site, Vercel previews, or localhost."""
    o = str(origin or "").strip()
    if not o:
        return None
    if _VERCEL.fullmatch(o) or _LOCAL.fullmatch(o) or _SITE.fullmatch(o):
        return o
    extras = str(os.environ.get("STUDIO_CORS_ORIGIN") or "")
    allowed = {x.strip().rstrip("/") for x in extras.split(",") if x.strip()}
    if o.rstrip("/") in allowed:
        return o
    return None


def _json_body(handler) -> dict:
    n = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(n) if n else b"{}"
    try:
        return json.loads(raw.decode() or "{}")
    except json.JSONDecodeError:
        return {}


def _q(parsed, key: str, default: str = "") -> str:
    return (parse_qs(parsed.query).get(key) or [default])[0]


def _check_api_key(handler) -> tuple[int, str] | None:
    """Validate X-Api-Key header.

    Returns:
      - None if authorized
      - (status_code, message) when authorization fails
    """
    api_key = str(handler.headers.get("x-api-key") or handler.headers.get("X-Api-Key") or "").strip()
    try:
        check_key(api_key)
    except InvalidKey:
        return 401, "invalid or missing API key"
    except QuotaExceeded as e:
        return 429, str(e)
    except Exception:
        # Infrastructure/runtime fault (AWS credentials, DynamoDB outage, etc.).
        # Do not leak internals to callers.
        return 503, "authorization backend unavailable"
    return None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write("[studio] " + (fmt % args) + "\n")

    def _apply_cors(self) -> None:
        origin = cors_origin(self.headers.get("Origin"))
        if not origin:
            return
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, ngrok-skip-browser-warning",
        )
        self.send_header("Access-Control-Max-Age", "86400")

    def _send(self, payload, status=200, set_cookie=None):
        if isinstance(payload, dict) and payload.get("error") and status == 200:
            status = 400 if payload["error"] != "not found" else 404
        blob = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(blob)))
        self._apply_cors()
        if set_cookie:
            self.send_header("Set-Cookie", set_cookie)
        self.end_headers()
        self.wfile.write(blob)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._apply_cors()
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        body = _json_body(self)
        if parsed.path == "/api/auth/login":
            payload, cookie = auth.login(body)
            return self._send(payload, set_cookie=cookie)
        if parsed.path == "/api/auth/logout":
            payload, cookie = auth.logout()
            return self._send(payload, set_cookie=cookie)
        if parsed.path == "/api/auth/hosted":
            return self._send(auth.set_hosted(body))
        gate_err = _check_api_key(self)
        if gate_err:
            status, message = gate_err
            return self._send({"error": message}, status)
        api_key = str(self.headers.get("x-api-key") or self.headers.get("X-Api-Key") or "").strip()
        routes = {
            "/api/agents": lambda: agents.create_agent(body),
            "/api/tags": lambda: data.set_tags(body),
            "/api/import": lambda: data.import_data(body),
            "/api/simulate": lambda: simulate.start({**body, "_api_key": api_key}),
            "/api/grade": lambda: grade.grade_now(body),
            "/api/merge": lambda: data.merge_runs(body),
        }
        fn = routes.get(parsed.path)
        if not fn:
            return self._send({"error": "not found"}, 404)
        return self._send(fn())

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            return self._send({"ok": True})
        if parsed.path == "/api/auth/status":
            payload, cookie = auth.status(self.headers.get("Cookie"))
            return self._send(payload, set_cookie=cookie)
        gate_err = _check_api_key(self)
        if gate_err:
            status, message = gate_err
            return self._send({"error": message}, status)
        routes = {
            "/api/agents": lambda: agents.list_agents(),
            "/api/starters": lambda: agents.list_starters(),
            "/api/agent": lambda: agents.get_agent(_q(parsed, "name")),
            "/api/sample": lambda: agents.sample(_q(parsed, "agent", "github")),
            "/api/runs": lambda: data.list_runs(_q(parsed, "agent") or None),
            "/api/run": lambda: data.get_run(_q(parsed, "id")),
            "/api/download/run": lambda: data.download_run(_q(parsed, "id")),
            "/api/download/agent": lambda: data.download_agent(_q(parsed, "agent")),
            "/api/job": lambda: simulate.job_status(_q(parsed, "id")),
            "/api/grade": lambda: grade.features(_q(parsed, "agent")),
            "/api/train": lambda: train.packs(_q(parsed, "agent"), _q(parsed, "run")),
            "/api/audit": lambda: audit.audit(_q(parsed, "agent")),
        }
        fn = routes.get(parsed.path)
        if not fn:
            return super().do_GET()
        return self._send(fn())


def main():
    auth.load_env()
    STATIC.mkdir(parents=True, exist_ok=True)
    (STATIC / "js").mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"http://{HOST}:{PORT}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
