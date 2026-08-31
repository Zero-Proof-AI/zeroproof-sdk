"""Runtime request-gate tests for the studio HTTP handler."""
from __future__ import annotations

import importlib
import sys

import pytest

from tests.helpers import REPO_ROOT


@pytest.fixture
def serve_module():
    studio_path = str(REPO_ROOT / "studio")
    if studio_path not in sys.path:
        sys.path.insert(0, studio_path)
    sys.modules.pop("serve", None)
    return importlib.import_module("serve")


def test_simulate_route_returns_429_when_quota_blocks(serve_module, monkeypatch):
    sent = {}
    started = {"called": False}

    def fake_send(payload, status=200, set_cookie=None):
        sent["payload"] = payload
        sent["status"] = status
        sent["set_cookie"] = set_cookie

    def fake_start(_body):
        started["called"] = True
        return {"ok": True}

    handler = object.__new__(serve_module.Handler)
    handler.path = "/api/simulate"
    handler.headers = {"x-api-key": "zp_blocked"}
    handler._send = fake_send

    monkeypatch.setattr(serve_module, "_json_body", lambda _handler: {"agent": "github"})
    monkeypatch.setattr(
        serve_module,
        "_check_api_key",
        lambda _handler: (429, "daily account quota exceeded"),
    )
    monkeypatch.setattr(serve_module.simulate, "start", fake_start)

    serve_module.Handler.do_POST(handler)

    assert sent == {
        "payload": {"error": "daily account quota exceeded"},
        "status": 429,
        "set_cookie": None,
    }
    assert started["called"] is False


def test_simulate_route_passes_api_key_to_simulate_start(serve_module, monkeypatch):
    sent = {}
    started = {}

    def fake_send(payload, status=200, set_cookie=None):
        sent["payload"] = payload
        sent["status"] = status
        sent["set_cookie"] = set_cookie

    def fake_start(body):
        started["body"] = body
        return {"id": "job-123"}

    handler = object.__new__(serve_module.Handler)
    handler.path = "/api/simulate"
    handler.headers = {"x-api-key": "zp_live"}
    handler._send = fake_send

    monkeypatch.setattr(serve_module, "_json_body", lambda _handler: {"agent": "github"})
    monkeypatch.setattr(serve_module, "_check_api_key", lambda _handler: None)
    monkeypatch.setattr(serve_module.simulate, "start", fake_start)

    serve_module.Handler.do_POST(handler)

    assert started["body"] == {"agent": "github", "_api_key": "zp_live"}
    assert sent == {
        "payload": {"id": "job-123"},
        "status": 200,
        "set_cookie": None,
    }