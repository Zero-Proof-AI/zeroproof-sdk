"""A key the endpoint rejects stops the run on the first 401, not at the clock.

The no-key case already fails at setup (test_missing_key_fails_fast). This is
the present-but-wrong-key case: before this the writer's 401 went into
``search["writer_errors"]`` and the run spent its whole time budget returning
zero rows.
"""

from __future__ import annotations

import time

import pytest

import zeroproof.simulations as zps
from tests.helpers import POLICY, TOOLS, FakeWriter
from zeroproof.simulations.run.engine import _auth_error


def test_auth_error_is_recognised_and_trimmed():
    assert _auth_error("RuntimeError: Hosted Qwen rejected the API key (401).") == (
        "Hosted Qwen rejected the API key (401)."
    )
    assert _auth_error("<agent error: No API key for api.openai.com: set OPENAI_API_KEY>")
    assert _auth_error("RuntimeError: connection reset") is None
    assert _auth_error("") is None


def _rejected_agent(message: str) -> dict:
    raise RuntimeError("Hosted Qwen rejected the API key (401).")


def test_rejected_key_on_the_agent_stops_on_the_first_rollout():
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match=r"rejected the API key \(401\)\.$"):
        zps.simulate(
            _rejected_agent,
            tools=TOOLS,
            system_prompt=POLICY,
            simulator=FakeWriter(),
            budget=40,
            time_budget=60,
            advanced={"concurrency": 1},
        )
    assert time.monotonic() - t0 < 10.0


def test_rejected_key_on_the_writer_stops_the_run(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "zp_wrong")

    def rejecting(*args, **kwargs):
        raise RuntimeError("Hosted Qwen rejected the API key (401).")

    # the hosted writer answers 401 on every wave (conftest blocks the real call)
    monkeypatch.setattr("zeroproof.simulations.generate.generator.complete", rejecting)
    monkeypatch.setattr("zeroproof.simulations.generate.agents.complete", rejecting)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match=r"rejected the API key \(401\)"):
        zps.simulate(
            tools=TOOLS,
            system_prompt=POLICY,
            budget=20,
            time_budget=60,
            advanced={"concurrency": 1},
        )
    assert time.monotonic() - t0 < 15.0
