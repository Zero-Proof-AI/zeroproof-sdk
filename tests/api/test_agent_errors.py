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


_CALLS = {"n": 0}


def _flaky_then_fine(message):
    """Fails the first two calls, then behaves."""
    _CALLS["n"] += 1
    if _CALLS["n"] <= 2:
        raise RuntimeError("BOOM: warming up")
    return {
        "steps": [{"tool": "restart_service", "arguments": {"service": "api"}, "result": {}}],
        "final_text": "Restarted api.",
    }


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


def test_dead_agent_is_called_off_early():
    """#88: a dead agent stops after max(16, 2 * budget) lost rollouts, not
    after every re-roll and refill has burned ~15 calls per budgeted row."""
    for budget, concurrency in ((8, 1), (8, 8), (40, 8)):
        data = simulate_offline(
            _raises, tools=TOOLS, budget=budget, per_round=8, concurrency=concurrency
        )
        allowance = max(16, 2 * budget)
        assert data.stopped_because == "agent_failed"
        assert data.trajectories == []
        # the batch in flight when the allowance trips can overshoot by
        # at most one wave of workers
        assert allowance <= data.search["agent_errors"] <= allowance + concurrency, (
            budget,
            concurrency,
            data.search["agent_errors"],
        )


def test_agent_that_recovers_is_not_called_off():
    _CALLS["n"] = 0
    data = simulate_offline(_flaky_then_fine, tools=TOOLS, budget=8, per_round=8, concurrency=1)
    assert len(data.trajectories) == 8
    assert data.stopped_because != "agent_failed"
    assert data.search["agent_errors"] == 2


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


def test_no_rows_and_no_agent_fault_names_the_writer(monkeypatch, caplog):
    import logging

    import zeroproof.simulations as zps
    from tests.helpers import offline, scripted_agent
    from zeroproof.simulations.run import engine as eng

    monkeypatch.setattr(eng.Run, "_refill_pool", lambda self, remaining, take: [])
    monkeypatch.setattr(eng.Run, "_build_batch", lambda self, selected, take: [])
    with caplog.at_level(logging.WARNING, logger="zeroproof.simulations"):
        data = zps.simulate(scripted_agent, budget=4, **offline())
    assert not data.trajectories
    assert data.stopped_because == "writer_failed"
    assert "writer_errors" in data.search
    assert any("no rows" in r.getMessage() for r in caplog.records)
