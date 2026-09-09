"""The caller's world answers tool calls when execute= is given."""
from zeroproof_simulations.generate.agents import _answer_tool_call, local_model
from zeroproof_simulations.world.sandbox import MockEnvironment
from tests.helpers import TOOLS


def test_execute_answers_and_faults_still_apply():
    seen = []

    def world(tool, args):
        seen.append((tool, args))
        return {"status": "ok", "order_id": args["order_id"], "total": 42}

    env = MockEnvironment(TOOLS)
    out = _answer_tool_call(env, world, "lookup_order", {"order_id": "ORD-9"})
    assert out == {"status": "ok", "order_id": "ORD-9", "total": 42}
    assert seen == [("lookup_order", {"order_id": "ORD-9"})]

    faulted = MockEnvironment(TOOLS, faults={"lookup_order": {"mode": "timeout", "rate": 1.0}})
    out = _answer_tool_call(faulted, world, "lookup_order", {"order_id": "ORD-9"})
    assert out.get("status") != "ok"
    assert len(seen) == 1, "a scheduled fault never reaches the caller's world"


def test_execute_errors_and_bare_values_become_results():
    env = MockEnvironment(TOOLS)

    def boom(tool, args):
        raise RuntimeError("db down")

    out = _answer_tool_call(env, boom, "lookup_order", {"order_id": "x"})
    assert out["status"] == "error"

    out = _answer_tool_call(env, lambda t, a: "plain text", "lookup_order", {})
    assert out == {"status": "ok", "result": "plain text"}


def test_local_model_routes_tool_calls_to_execute(monkeypatch):
    calls = {"n": 0}

    def fake_complete(_url, _model, _messages, **kwargs):
        calls["n"] += 1
        if kwargs.get("tools") and calls["n"] == 1:
            return {"content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "lookup_order",
                             "arguments": '{"order_id":"ORD-7"}'}}]}
        return {"content": "Order ORD-7 ships tomorrow."}

    monkeypatch.setattr("zeroproof_simulations.generate.agents.complete", fake_complete)
    answered = []

    def world(tool, args):
        answered.append(tool)
        return {"status": "ok", "eta": "tomorrow"}

    agent = local_model("http://example", "m", tools=TOOLS, max_turns=6, execute=world)
    out = agent("where is ORD-7")
    tool_steps = [s for s in out["steps"] if s.get("tool")]
    assert answered == ["lookup_order"]
    assert tool_steps[0]["result"] == {"status": "ok", "eta": "tomorrow"}
