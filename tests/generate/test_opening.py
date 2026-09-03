"""Conversation topology: who opens is an axis, never a hardcoded frame."""
import zeroproof_simulations as zps
from zeroproof_simulations import agents as zagents
from zeroproof_simulations.traces import opening_share


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
