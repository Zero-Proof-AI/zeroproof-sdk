"""A prompt-only agent whose tool draft fails must say so, not run tool-free in silence."""
from __future__ import annotations

import zeroproof_simulations as zps


def _writer(_dataset=None, index=0):
    return [f"can you check on my refund {index}-{i}" for i in range(6)]


def test_failed_tool_draft_is_a_degraded_note(monkeypatch):
    monkeypatch.setattr("zeroproof_simulations.run.engine.draft_tools",
                        lambda *a, **k: [])
    monkeypatch.setattr(
        "zeroproof_simulations.run.engine.hosted_model",
        lambda tools, system="", **kw: (lambda m: {"steps": [], "final_text": "ok"}))
    data = zps.simulate(
        system_prompt="A refunds assistant that looks up an order first.",
        budget=4, seed=0, concurrency=1, grade=False, time_budget=None,
        simulator=_writer, advanced={"per_round": 8, "mutate_failures": False})
    assert "tool_draft_unavailable" in data.degraded
    assert not data.search.get("drafted_tools")


def test_successful_tool_draft_is_not_flagged(monkeypatch):
    drafted = [{"type": "function", "function": {
        "name": "lookup_order", "drafted": True,
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}}]
    monkeypatch.setattr("zeroproof_simulations.run.engine.draft_tools",
                        lambda *a, **k: drafted)
    monkeypatch.setattr(
        "zeroproof_simulations.run.engine.hosted_model",
        lambda tools, system="", **kw: (lambda m: {"steps": [], "final_text": "ok"}))
    data = zps.simulate(
        system_prompt="A refunds assistant that looks up an order first.",
        budget=4, seed=0, concurrency=1, grade=False, time_budget=None,
        simulator=_writer, advanced={"per_round": 8, "mutate_failures": False})
    assert "tool_draft_unavailable" not in data.degraded
    assert data.search.get("drafted_tools") == ["lookup_order"]
