"""A bring-your-own run with no key fails at setup, not after the time budget."""
from __future__ import annotations

import time

import pytest

import zeroproof.simulations as zps
from tests.helpers import POLICY, TOOLS
from zeroproof.simulations.generate.agents import missing_hosted_key


def test_openai_endpoint_without_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    msg = missing_hosted_key("https://api.openai.com/v1")
    assert msg and "OPENAI_API_KEY" in msg and "\n" not in msg


def test_local_endpoints_need_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    assert missing_hosted_key("http://localhost:11434/v1") is None
    assert missing_hosted_key("http://127.0.0.1:8000/v1") is None
    assert missing_hosted_key("http://gpu-box:8000/v1") is None


def test_simulate_with_openai_spec_and_no_key_raises_at_setup(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        zps.simulate(agent="openai:gpt-4.1-mini", tools=TOOLS, system_prompt=POLICY,
                     budget=2, time_budget=30, seed=0)
    assert time.monotonic() - t0 < 5.0
