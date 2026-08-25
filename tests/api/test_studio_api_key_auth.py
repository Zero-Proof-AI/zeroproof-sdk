"""Studio API auth contract: X-Api-Key is mandatory on protected routes."""
from __future__ import annotations

from tests.helpers import REPO_ROOT

SRC = (REPO_ROOT / "studio" / "serve.py").read_text()


def test_api_key_header_is_the_auth_credential():
    assert "def _check_api_key(handler)" in SRC
    assert 'handler.headers.get("x-api-key")' in SRC
    assert 'handler.headers.get("X-Api-Key")' in SRC
    assert "check_key(api_key)" in SRC
    assert '"Content-Type, Authorization, X-Api-Key, ngrok-skip-browser-warning"' in SRC
    # Local desk cookie bypass was intentionally removed.
    assert "email_from_cookie" not in SRC


def test_protected_routes_enforce_gate_and_status_codes():
    # Both POST and GET protected paths must enforce check_key() via _check_api_key.
    assert "gate_err = _check_api_key(self)" in SRC
    assert 'status, message = gate_err' in SRC
    assert 'return self._send({"error": message}, status)' in SRC
    assert 'return 503, "authorization backend unavailable"' in SRC


def test_auth_endpoints_are_exempt_but_business_endpoints_are_protected():
    # Explicit auth endpoints remain accessible without X-Api-Key.
    assert 'if parsed.path == "/api/auth/login":' in SRC
    assert 'if parsed.path == "/api/auth/logout":' in SRC
    assert 'if parsed.path == "/api/auth/hosted":' in SRC
    assert 'if parsed.path == "/api/auth/status":' in SRC
    assert 'if not parsed.path.startswith("/api/"):' in SRC
    assert 'return super().do_GET()' in SRC

    # Business endpoints are still routed behind the auth gate.
    assert '"/api/simulate": lambda: simulate.start({**body, "_api_key": api_key})' in SRC
    assert '"/api/agents": lambda: agents.list_agents()' in SRC
