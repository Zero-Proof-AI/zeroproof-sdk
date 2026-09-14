"""A broken agent callable is reported, not hidden behind the writer."""

from __future__ import annotations

from tests.helpers import simulate_offline

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "restart_service",
            "description": "Restart a service.",
            "parameters": {
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
            },
        },
    }
]


def _raises(message):
    raise RuntimeError("BOOM: my agent is broken")


def _wrong_signature(messages, tools):
    return {"role": "assistant", "content": "ok"}


def _wrong_shape(message):
    return {"role": "assistant", "content": "ok"}


def test_raising_agent_stops_as_agent_failed():
    data = simulate_offline(_raises, tools=TOOLS, budget=8, per_round=8, concurrency=1)
    assert data.trajectories == []
    assert data.stopped_because == "agent_failed"
    assert "agent_errors" in data.degraded
    assert data.search["agent_errors"] >= 8
    assert data.search["first_agent_error"].startswith("RuntimeError: BOOM")


def test_wrong_signature_names_the_type_error():
    data = simulate_offline(_wrong_signature, tools=TOOLS, budget=8, per_round=8, concurrency=1)
    assert data.trajectories == []
    assert data.stopped_because == "agent_failed"
    assert data.search["first_agent_error"].startswith("TypeError:")


def test_wrong_return_shape_is_an_agent_error():
    data = simulate_offline(_wrong_shape, tools=TOOLS, budget=8, per_round=8, concurrency=1)
    assert data.trajectories == []
    assert data.stopped_because == "agent_failed"
    assert "['content', 'role']" in data.search["first_agent_error"]


def test_working_agent_reports_no_agent_errors():
    data = simulate_offline(tools=TOOLS, budget=8, per_round=8, concurrency=1)
    assert data.trajectories
    assert "agent_errors" not in data.search
    assert "agent_errors" not in data.degraded
    assert data.stopped_because != "agent_failed"
