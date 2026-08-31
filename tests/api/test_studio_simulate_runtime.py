"""Runtime tests for simulate job setup behavior."""
from __future__ import annotations

import importlib
import sys

import pytest

from tests.helpers import REPO_ROOT


@pytest.fixture
def simulate_module(tmp_path, monkeypatch):
    studio_path = str(REPO_ROOT / "studio")
    if studio_path not in sys.path:
        sys.path.insert(0, studio_path)
    sys.modules.pop("api.simulate", None)
    module = importlib.import_module("api.simulate")

    monkeypatch.setattr(module, "_hosted_ready", lambda fresh=False: True)
    monkeypatch.setattr(
        module,
        "_lookup_agent",
        lambda _name: {
            "tools": [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}],
            "policy": "be concise",
        },
    )
    monkeypatch.setattr(module, "_runs_dir", lambda _name: tmp_path / "runs")
    monkeypatch.setattr(module, "_write_meta", lambda *_args, **_kwargs: {})
    return module


def test_start_job_keeps_request_api_key_for_usage_recording(simulate_module, monkeypatch):
    captured = {}

    class DummyThread:
        def __init__(self, target, args=(), kwargs=None, daemon=None):
            captured["target"] = target
            captured["args"] = args
            captured["kwargs"] = kwargs or {}
            captured["daemon"] = daemon

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(simulate_module.threading, "Thread", DummyThread)

    result = simulate_module.start_job({
        "source": "existing",
        "agent": "github",
        "mode": "explore",
        "time_budget": 5,
        "_api_key": "zp_live",
    })

    assert result["status"] == "queued"
    assert captured["started"] is True
    spec = captured["args"][1]
    assert spec["_api_key"] == "zp_live"
