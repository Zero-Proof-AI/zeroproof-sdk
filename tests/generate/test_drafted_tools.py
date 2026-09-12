"""A description with no tools gets a drafted tool surface in its own domain."""
import json

from zeroproof.simulations.generate import agents as zagents
from zeroproof.simulations.generate import generator as gen


def test_draft_tools_parses_fenced_json_and_marks_drafted(monkeypatch):
    calls = []

    def fake(base_url, model, messages, **kw):
        calls.append(messages)
        return {"content": "```json\n" + json.dumps([
            {"name": "get_account_balance", "description": "Balance of one account",
             "parameters": {"type": "object", "properties": {
                 "account_id": {"type": "string"}}, "required": ["account_id"]}},
            {"name": "transfer_funds", "description": "Move money between accounts",
             "parameters": {"type": "object", "properties": {
                 "from_account": {"type": "string"}, "to_account": {"type": "string"},
                 "amount": {"type": "number"}}, "required": ["from_account", "to_account", "amount"]}},
            {"name": "Bad Name!", "description": "rejected", "parameters": {}},
        ]) + "\n```", "tool_calls": []}

    monkeypatch.setattr(gen, "complete", fake)
    tools = gen.draft_tools("A personal finance assistant that confirms before moving money.",
                            backend_spec="openai:m")
    names = [t["function"]["name"] for t in tools]
    assert names == ["get_account_balance", "transfer_funds"]
    assert all(t["drafted"] is True for t in tools)
    assert tools[1]["function"]["parameters"]["required"] == ["from_account", "to_account", "amount"]
    assert "finance" in json.dumps(calls[0])


def test_draft_tools_needs_two_valid_tools(monkeypatch):
    monkeypatch.setattr(gen, "complete", lambda *a, **k: {
        "content": json.dumps([{"name": "only_one", "parameters": {}}]), "tool_calls": []})
    assert gen.draft_tools("some agent", backend_spec="openai:m") == []


def test_wire_tools_strips_the_drafted_marker():
    wired = zagents._wire_tools([{"type": "function", "drafted": True, "function": {
        "name": "transfer_funds", "description": "x", "parameters": {"type": "object"}}}])
    assert wired == [{"type": "function", "function": {
        "name": "transfer_funds", "description": "x", "parameters": {"type": "object"}}}]
