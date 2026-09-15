"""Every model call reports what it cost; the row and each agent step keep it.

The server's ``usage`` block (prompt_tokens / completion_tokens) becomes
``_usage`` on the reply, ``input_tokens`` / ``output_tokens`` on the agent
step, and a summed ``usage`` on the row. Rows without it stay as they were.
"""

import http.client
import json

import pytest

import zeroproof.simulations as zps
from tests.helpers import FakeWriter
from zeroproof.simulations import schema
from zeroproof.simulations.generate import agents

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]
REAL_COMPLETE = agents.complete


class _FakeResponse:
    def __init__(self, body: dict, status: int = 200):
        self.status = status
        self._raw = json.dumps(body).encode()

    def read(self):
        return self._raw


class _FakeConn:
    usage: dict | None = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}

    def __init__(self, *_a, **_k):
        pass

    def request(self, _method, _path, body=None, headers=None):
        self.payload = json.loads(body)

    def getresponse(self):
        body = {
            "choices": [
                {"finish_reason": "stop", "message": {"role": "assistant", "content": "Shipped."}}
            ]
        }
        if _FakeConn.usage is not None:
            body["usage"] = _FakeConn.usage
        return _FakeResponse(body)

    def close(self):
        pass


@pytest.fixture
def fake_http(monkeypatch):
    _FakeConn.usage = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
    monkeypatch.setattr(http.client, "HTTPConnection", _FakeConn)
    monkeypatch.setattr(agents, "_tls", type("T", (), {"conn": None, "conn_key": None})())
    yield _FakeConn


def test_complete_keeps_the_servers_usage(fake_http):
    reply = REAL_COMPLETE("http://127.0.0.1:9/v1", "m", [{"role": "user", "content": "hi"}])
    assert reply["_usage"] == {"input_tokens": 120, "output_tokens": 30}
    fake_http.usage = None
    plain = REAL_COMPLETE("http://127.0.0.1:9/v1", "m", [{"role": "user", "content": "hi"}])
    assert "_usage" not in plain


def test_usage_never_goes_back_to_the_server():
    reply = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "get_order", "arguments": "{}"}}],
        "_usage": {"input_tokens": 1, "output_tokens": 1},
    }
    _calls, assistant = agents._calls_from_reply(reply)
    assert "_usage" not in assistant


def test_turn_meta_stamps_tokens():
    assert agents._turn_meta({"_usage": {"input_tokens": 7, "output_tokens": 3}}) == {
        "input_tokens": 7,
        "output_tokens": 3,
    }
    assert agents._turn_meta({"content": "x"}) == {}


def _fake_complete_factory(calls: dict, with_usage: bool):
    def fake_complete(_url, _model, messages, **kwargs):
        calls["n"] = calls.get("n", 0) + 1
        if kwargs.get("tools") and calls["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": '{"order_id": "4412"}'},
                    }
                ],
                **({"_usage": {"input_tokens": 300, "output_tokens": 20}} if with_usage else {}),
                "_finish_reason": "tool_calls",
            }
        if kwargs.get("tools"):
            return {
                "content": "Order 4412 shipped yesterday.",
                **({"_usage": {"input_tokens": 400, "output_tokens": 12}} if with_usage else {}),
                "_finish_reason": "stop",
            }
        return {"content": ""}

    return fake_complete


def _simulate(monkeypatch, with_usage: bool):
    calls: dict = {}
    monkeypatch.setattr(
        "zeroproof.simulations.generate.agents.complete", _fake_complete_factory(calls, with_usage)
    )
    monkeypatch.setattr(
        "zeroproof.simulations.generate.agents.sample_turn_budget", lambda *_a, **_k: 4
    )
    return zps.simulate(
        tools=TOOLS,
        policy="Look up before you answer.",
        extra_situations=["where is order 4412"],
        budget=1,
        grade=False,
        concurrency=1,
        simulator=FakeWriter(),
        backend="vllm:fake@http://127.0.0.1:9",
        seed=0,
        time_budget=None,
        max_turns=4,
        avg_turns=2,
        advanced={"per_round": 2, "mutate_failures": False},
    )


def test_simulate_sums_usage_onto_the_row(monkeypatch):
    data = _simulate(monkeypatch, with_usage=True)
    row = data.trajectories[0]
    tool_steps = [s for s in row["steps"] if isinstance(s, dict) and s.get("tool")]
    assert tool_steps[0]["input_tokens"] == 300 and tool_steps[0]["output_tokens"] == 20
    text_steps = [s for s in row["steps"] if isinstance(s, dict) and s.get("text")]
    assert text_steps[-1]["input_tokens"] == 400 and text_steps[-1]["output_tokens"] == 12
    assert row["usage"] == {"input_tokens": 700, "output_tokens": 32}
    exported = data.rows()[0]
    assert exported["usage"] == {"input_tokens": 700, "output_tokens": 32}
    # The typed objects carry it too, so a roundtrip does not lose the cost.
    task, rollout, _, _ = schema.from_row(exported)
    first_tool = next(s for s in rollout.steps if s.tool)
    assert first_tool.input_tokens == 300 and first_tool.output_tokens == 20
    back = schema.to_row(task, rollout)
    assert back["usage"] == {"input_tokens": 700, "output_tokens": 32}
    assert next(s for s in back["steps"] if s.get("tool"))["input_tokens"] == 300
    assert schema.validate(exported) == []


def test_simulate_without_usage_leaves_rows_alone(monkeypatch):
    data = _simulate(monkeypatch, with_usage=False)
    row = data.trajectories[0]
    assert "usage" not in row
    assert not any("input_tokens" in s for s in row["steps"] if isinstance(s, dict))
