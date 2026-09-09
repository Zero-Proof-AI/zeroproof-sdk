"""Conversation topology: who opens is an axis, never a hardcoded frame."""
import zeroproof_simulations as zps
from zeroproof_simulations.generate import agents as zagents
from zeroproof_simulations.ingest.traces import opening_share


def test_agent_opener_rolls_and_exports(monkeypatch):
    replies = iter([
        {"content": "Hi! How can I help you today?", "tool_calls": []},
        {"content": "Your order is on the way.", "tool_calls": []},
    ])
    monkeypatch.setattr(zagents, "complete",
                        lambda *a, **k: next(replies))
    runner = zagents.local_model(
        "http://x/v1", "m", tools=[], system="Help with orders.",
        opening_rate=1.0, min_user_turns=1, max_turns=2)
    row = runner("where is my order 12345")
    assert row["opener"] == "Hi! How can I help you today?"
    assert row["opening"] == "agent"
    row["prompt"] = "where is my order 12345"
    msgs = zps.conversation(row)
    assert msgs[0] == {"role": "assistant",
                       "content": "Hi! How can I help you today?"}
    assert msgs[1]["role"] == "user"


def test_opening_share_reads_trace_evidence():
    rows = [
        {"messages": [{"role": "assistant", "content": "Hi!"},
                      {"role": "user", "content": "hey"}]},
        {"messages": [{"role": "user", "content": "hello"}]},
        {"prompt": "no messages at all"},
    ]
    assert abs(opening_share(rows) - 1 / 3) < 1e-9
    assert opening_share([]) == 0.0


def test_backend_spec_agents_get_the_opening_axis(monkeypatch):
    """The agent="vllm:..." path must honor opening=, same as the others.
    Regression: round 2's first batch generated 0 agent-opened rows."""
    from zeroproof_simulations.generate.adapters import resolve

    replies = iter([
        {"content": "Welcome! What can I do for you?", "tool_calls": []},
        {"content": "Done.", "tool_calls": []},
    ])
    monkeypatch.setattr(zagents, "complete", lambda *a, **k: next(replies))
    runner, kind = resolve("vllm:m@http://x/v1", tools=[], policy="p",
                           opening_rate=1.0, max_turns=2)
    assert kind == "backend_spec"
    row = runner("need help with a thing")
    assert row["opener"] == "Welcome! What can I do for you?"


def test_opening_survives_the_full_simulate_path(monkeypatch):
    """Unit coverage of resolve() was not enough: the rollout row is
    rebuilt from selected runner fields, and opener must survive that."""
    monkeypatch.setattr(
        zagents, "complete",
        lambda *a, **k: {"content": "hello there", "tool_calls": []})
    d = zps.simulate(
        agent="vllm:m@http://x/v1",
        tools=[{"type": "function", "function": {
            "name": "t1", "description": "d",
            "parameters": {"type": "object", "properties": {}}}}],
        system_prompt="Help.", budget=2, seed=1, grade=False,
        simulator=False, time_budget=30, advanced={"opening": "agent"})
    row = d.trajectories[0]
    assert row["opener"] == "hello there"
    assert zps.conversation(row)[0]["role"] == "assistant"
    assert d.search["strategy"]["opening"]["rate"] == 1.0
