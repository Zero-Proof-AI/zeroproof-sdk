"""simulate(logprobs=True): every agent turn carries its sampling facts."""

import http.client
import json

import pytest

import zeroproof.simulations as zps
from tests.helpers import FakeWriter
from zeroproof.simulations import schema
from zeroproof.simulations.export import training_rows
from zeroproof.simulations.generate import agents
from zeroproof.simulations.score.logprobs import logprob_report, mean_kl
from zeroproof.simulations.score.publish_gate import calibrate

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
POLICY = "Look up before you answer."
# conftest swaps agents.complete for a blocked stub per test; keep the real one
REAL_COMPLETE = agents.complete


class _FakeResponse:
    def __init__(self, body: dict, status: int = 200):
        self.status = status
        self._raw = json.dumps(body).encode()

    def read(self):
        return self._raw


class _FakeConn:
    """Captures the payload and answers with a canned vLLM-shaped body."""

    sent: list[dict] = []
    reject_logprobs = False

    def __init__(self, *_a, **_k):
        pass

    def request(self, _method, _path, body=None, headers=None):
        self.payload = json.loads(body)
        _FakeConn.sent.append(self.payload)

    def getresponse(self):
        if _FakeConn.reject_logprobs and self.payload.get("logprobs"):
            return _FakeResponse({"error": "logprobs not supported"}, status=400)
        return _FakeResponse(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "Order 4412 shipped."},
                        "logprobs": {
                            "content": [
                                {"token": "Order", "logprob": -0.5},
                                {"token": " 4412", "logprob": -1.25},
                                {"token": " shipped.", "logprob": -0.25},
                            ]
                        },
                    }
                ]
            }
        )

    def close(self):
        pass


@pytest.fixture
def fake_http(monkeypatch):
    _FakeConn.sent = []
    _FakeConn.reject_logprobs = False
    monkeypatch.setattr(http.client, "HTTPConnection", _FakeConn)
    monkeypatch.setattr(agents, "_tls", type("T", (), {"conn": None, "conn_key": None})())
    yield _FakeConn


def test_complete_requests_and_summarizes_logprobs(fake_http):
    reply = REAL_COMPLETE(
        "http://127.0.0.1:9/v1", "m", [{"role": "user", "content": "hi"}], logprobs=True
    )
    assert fake_http.sent[-1]["logprobs"] is True
    assert reply["_logprobs"] == {"sum": -2.0, "n": 3}
    assert reply["_finish_reason"] == "length"
    plain = REAL_COMPLETE("http://127.0.0.1:9/v1", "m", [{"role": "user", "content": "hi"}])
    assert "logprobs" not in fake_http.sent[-1]
    assert "_logprobs" not in plain and plain["_finish_reason"] == "length"
    with_tokens = REAL_COMPLETE(
        "http://127.0.0.1:9/v1", "m", [{"role": "user", "content": "hi"}], logprobs="tokens"
    )
    assert with_tokens["_logprobs"]["tokens"] == [-0.5, -1.25, -0.25]


def test_complete_retries_without_logprobs_when_the_server_rejects_them(fake_http):
    fake_http.reject_logprobs = True
    reply = REAL_COMPLETE(
        "http://127.0.0.1:9/v1", "m", [{"role": "user", "content": "hi"}], logprobs=True
    )
    assert reply["content"] == "Order 4412 shipped."
    assert "_logprobs" not in reply
    assert [p.get("logprobs") for p in fake_http.sent] == [True, None]


def test_private_reply_fields_never_go_back_to_the_server():
    reply = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "get_order", "arguments": "{}"}}],
        "_logprobs": {"sum": -1.0, "n": 2},
        "_finish_reason": "stop",
    }
    calls, assistant = agents._calls_from_reply(reply)
    assert calls and not any(k.startswith("_") for k in assistant)


def _fake_complete_factory(calls: dict):
    def fake_complete(_url, _model, messages, **kwargs):
        calls["n"] = calls.get("n", 0) + 1
        calls["logprobs"] = kwargs.get("logprobs")
        want = bool(kwargs.get("logprobs"))  # the real complete only attaches when asked
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
                **({"_logprobs": {"sum": -3.5, "n": 7}} if want else {}),
                "_finish_reason": "tool_calls",
            }
        if kwargs.get("tools"):
            return {
                "content": "Order 4412 shipped yesterday.",
                **({"_logprobs": {"sum": -2.0, "n": 4}} if want else {}),
                "_finish_reason": "length",
            }
        return {"content": ""}

    return fake_complete


def _simulate(monkeypatch, **kw):
    calls: dict = {}
    monkeypatch.setattr(
        "zeroproof.simulations.generate.agents.complete", _fake_complete_factory(calls)
    )
    monkeypatch.setattr(
        "zeroproof.simulations.generate.agents.sample_turn_budget", lambda *_a, **_k: 4
    )
    data = zps.simulate(
        tools=TOOLS,
        policy=POLICY,
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
        **kw,
    )
    return data, calls


def test_simulate_logprobs_stamps_steps_row_and_exports(monkeypatch):
    data, calls = _simulate(monkeypatch, logprobs=True)
    assert calls["logprobs"] is True
    row = data.trajectories[0]
    steps = [s for s in row["steps"] if isinstance(s, dict) and s.get("tool")]
    assert steps[0]["logprob"] == -3.5 and steps[0]["n_tokens"] == 7
    assert "truncated" not in steps[0]
    text_steps = [s for s in row["steps"] if isinstance(s, dict) and s.get("text")]
    assert text_steps and text_steps[-1]["logprob"] == -2.0 and text_steps[-1]["truncated"] is True
    assert row["logprob"] == -5.5 and row["n_tokens"] == 11
    exported = data.rows()[0]
    assert exported["logprob"] == -5.5 and exported["n_tokens"] == 11
    assert exported["steps"][0]["logprob"] == -3.5
    trained = training_rows(data)[0]
    assert trained["logprob"] == -5.5 and trained["n_tokens"] == 11
    task, rollout, _, _ = schema.from_row(exported)
    assert rollout.steps[0].logprob == -3.5 and rollout.steps[0].n_tokens == 7
    assert rollout.extra["logprob"] == -5.5
    back = schema.to_row(task, rollout)
    assert back["logprob"] == -5.5 and back["steps"][0]["n_tokens"] == 7
    assert back["steps"][-1]["truncated"] is True
    assert schema.validate(exported) == []


def test_simulate_without_logprobs_leaves_rows_alone(monkeypatch):
    data, calls = _simulate(monkeypatch)
    assert calls["logprobs"] is False
    row = data.trajectories[0]
    assert "logprob" not in row and "n_tokens" not in row
    # the fake still reports finish_reason length on the text turn
    with pytest.raises(ValueError, match="logprobs"):
        _simulate(monkeypatch, logprobs="all")


def _row(task, lp, n, reward=None, ref=None, **extra):
    row = {"prompt": task, "logprob": lp, "n_tokens": n, "steps": [], **extra}
    if reward is not None:
        row["reward"] = reward
    if ref is not None:
        row["ref_logprob"] = ref
    return row


def test_logprob_report_and_confidence_warning():
    rows = [_row(f"t{i}", -1.0 * (i + 1), 10, reward=1 if i < 4 else 0) for i in range(8)]
    rows.append({"prompt": "none", "steps": []})
    out = logprob_report(rows)
    assert out["n_rows"] == 9 and out["n_with_logprobs"] == 8 and out["n_tokens"] == 80
    assert out["mean_token_logprob"] == round(-36 / 80, 4)
    assert out["row_mean_p50"] is not None
    # confident rows (i small) pass: reward tracks confidence
    assert out["corr_reward_confidence"] > 0.3
    assert any("confidence" in w for w in out["warnings"])
    empty = logprob_report([{"prompt": "x", "steps": []}])
    assert any("simulate(logprobs=True)" in w for w in empty["warnings"])


def test_mean_kl_from_key_and_from_reference_rows_and_calibrate():
    rows = [
        _row("a", -10.0, 10, reward=1, ref=-12.0, scenario_id="a", rollout_index=0),
        _row("a", -20.0, 10, reward=0, ref=-21.0, scenario_id="a", rollout_index=1),
        _row("b", -5.0, 5, reward=1, ref=-5.0, scenario_id="b", rollout_index=0),
        _row("c", -1.0, 1, reward=1, scenario_id="c", rollout_index=0),
    ]
    out = mean_kl(rows)
    assert out["n_rows"] == 3 and out["n_skipped"] == 1 and out["n_tokens"] == 25
    assert out["per_task"] == {"a": 0.15, "b": 0.0}
    assert out["mean_kl"] == round(3.0 / 25, 6)
    ref_rows = [
        {"scenario_id": "a", "rollout_index": 0, "logprob": -12.0},
        {"scenario_id": "a", "rollout_index": 1, "logprob": -21.0},
    ]
    matched = mean_kl(rows, ref_rows)
    assert matched["n_rows"] == 2 and matched["per_task"] == {"a": 0.15}
    report = calibrate(rows, ref="ref_logprob")
    assert report["mean_kl"]["mean_kl"] == out["mean_kl"]
    assert rows[0]["calibration"]["mean_kl"] == 0.15
    assert rows[2]["calibration"]["mean_kl"] == 0.0
    assert rows[3]["calibration"]["mean_kl"] is None
    assert zps.mean_kl is mean_kl and zps.logprob_report is logprob_report
