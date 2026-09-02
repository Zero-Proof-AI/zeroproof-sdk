"""The recommended-dataset path: graded rows in, trainer-ready subset out."""
import json

import pytest

from zeroproof_simulations import SimulationData


def _row(prompt: str, reward, *, tool: str = "add_expense") -> dict:
    return {
        "prompt": prompt,
        "steps": [{"tool": tool, "arguments": {"item": prompt, "amount": 1.0},
                   "result": {"status": "ok"}}],
        "final_text": f"Recorded {prompt}.",
        "reward": reward,
        "arm": "grid",
    }


def _graded_data() -> SimulationData:
    rows = [
        _row("coffee", 1),
        _row("lunch", 1, tool="list_expenses"),
        _row("taxi", 0),
        _row("coffee", 1),  # duplicate prompt never ships twice
    ]
    return SimulationData(trajectories=rows)


def test_select_keeps_only_diverse_passes():
    data = _graded_data()
    selected = data.select(target=10)
    prompts = [r["prompt"] for r in selected]
    assert "taxi" not in prompts
    assert prompts.count("coffee") == 1
    assert "lunch" in prompts
    report = data.search["selection"]
    assert report["n_selected"] == len(selected)
    assert report["n_not_pass"] >= 1


def test_training_set_writes_chat_jsonl_with_policy_and_tools(tmp_path):
    from zeroproof_simulations.adapters import AgentProfile

    data = _graded_data()
    data.profile = AgentProfile(
        policy="You are budget-buddy. Use your tools; refuse anything else.",
        tools=[{"type": "function", "function": {
            "name": "add_expense", "description": "Record an expense",
            "parameters": {"type": "object", "properties": {
                "item": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["item", "amount"]}}}])
    out = tmp_path / "train.jsonl"
    report = data.training_set(str(out))
    assert report["n_written"] == report["selection"]["n_selected"]
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    first = rows[0]
    assert first["messages"][0]["role"] == "system"
    assert "budget-buddy" in first["messages"][0]["content"]
    assert first["tools"][0]["function"]["name"] == "add_expense"
    calls = [m for m in first["messages"]
             if m.get("role") == "assistant" and m.get("tool_calls")]
    assert calls, "tool calls must survive export"
    call = calls[0]["tool_calls"][0]
    assert call["type"] == "function"
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"])["item"]
    assert report["tool_call_roundtrip"]["invalid"] == 0


def test_select_refuses_ungraded_and_nonbinary_rows():
    data = SimulationData(trajectories=[
        {"prompt": "coffee", "steps": [], "final_text": "ok", "arm": "grid"}])
    with pytest.raises(RuntimeError, match="graded"):
        data.select()
    fractional = SimulationData(trajectories=[_row("coffee", 0.5)])
    with pytest.raises(RuntimeError, match="binary"):
        fractional.select()
