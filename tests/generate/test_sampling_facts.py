"""Sampling facts on every row: policy_version, sampling, token_logprobs
(rlhf-book ch. 6 async RL, ch. 9), and staleness_report."""

from __future__ import annotations

import hashlib

import zeroproof.simulations as zps
from tests.generate.test_logprobs import POLICY, TOOLS, _simulate
from tests.helpers import simulate_offline
from zeroproof.simulations import schema
from zeroproof.simulations.export import training_rows
from zeroproof.simulations.generate.agents import LOCAL_MODEL_TEMPERATURE
from zeroproof.simulations.score.logprobs import staleness_report


def test_model_backed_rows_carry_policy_version_sampling_and_token_logprobs(monkeypatch):
    data, _calls = _simulate(monkeypatch, logprobs="tokens")
    row = data.trajectories[0]
    assert row["sampling"] == {"temperature": LOCAL_MODEL_TEMPERATURE, "logprobs": "tokens"}
    assert row["policy_version"].startswith(row["model_version"] + "@")
    assert len(row["policy_version"].split("@")[-1]) == 16
    # the fake backend returns summed logprobs only; the tokens list stays absent
    assert "token_logprobs" not in row
    exported = data.rows()[0]
    assert exported["sampling"]["temperature"] == LOCAL_MODEL_TEMPERATURE
    assert exported["policy_version"] == row["policy_version"]
    trained = training_rows(data)[0]
    assert trained["policy_version"] == row["policy_version"] and "sampling" in trained
    assert schema.validate(exported) == []


def test_temperature_knob_is_recorded(monkeypatch):
    data, _ = _simulate(monkeypatch, temperature=0.3)
    row = data.trajectories[0]
    assert row["sampling"] == {"temperature": 0.3, "logprobs": False}


def test_callable_agent_rows_have_a_policy_version_but_no_sampling():
    data = simulate_offline(tools=TOOLS, policy=POLICY, budget=4, per_round=4, concurrency=1)
    row = data.trajectories[0]
    assert "sampling" not in row
    assert row["policy_version"].startswith(row["model_version"] + "@")


def _tokens_complete_factory(calls: dict):
    """A backend that returns per-token logprobs on every turn."""

    def fake_complete(_url, _model, messages, **kwargs):
        calls["n"] = calls.get("n", 0) + 1
        want = kwargs.get("logprobs") == "tokens"
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
                **({"_logprobs": {"sum": -1.0, "n": 2, "tokens": [-0.4, -0.6]}} if want else {}),
                "_finish_reason": "tool_calls",
            }
        if kwargs.get("tools"):
            return {
                "content": "Order 4412 shipped yesterday.",
                **({"_logprobs": {"sum": -0.5, "n": 1, "tokens": [-0.5]}} if want else {}),
                "_finish_reason": "stop",
            }
        return {"content": ""}

    return fake_complete


def test_token_logprobs_roll_up_from_steps_in_order(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(
        "zeroproof.simulations.generate.agents.complete", _tokens_complete_factory(calls)
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
        simulator=False,
        backend="vllm:fake@http://127.0.0.1:9",
        seed=0,
        time_budget=None,
        max_turns=4,
        avg_turns=2,
        logprobs="tokens",
        advanced={"per_round": 2, "mutate_failures": False},
    )
    row = data.trajectories[0]
    assert row["token_logprobs"] == [-0.4, -0.6, -0.5]
    assert row["logprob"] == -1.5 and row["n_tokens"] == 3
    assert row["sampling"]["logprobs"] == "tokens"
    exported = data.rows()[0]
    assert exported["token_logprobs"] == [-0.4, -0.6, -0.5]
    assert training_rows(data)[0]["token_logprobs"] == [-0.4, -0.6, -0.5]
    assert zps.staleness_report([exported])["token_logprob_coverage"] == 1.0


def test_policy_version_round_trips_through_the_schema():
    row = {
        "prompt": "where is order 4412",
        "steps": [],
        "final_text": "shipped",
        "scenario_id": "s1",
        "model_version": "fake",
        "policy_version": "fake@" + hashlib.sha256(b"p").hexdigest()[:16],
        "sampling": {"temperature": 0.8, "logprobs": True},
        "token_logprobs": [-0.1, -0.2],
        "logprob": -0.3,
        "n_tokens": 2,
    }
    task, rollout, _, _ = schema.from_row(row)
    assert rollout.policy.version == row["policy_version"]
    assert rollout.extra["sampling"] == row["sampling"]
    assert rollout.extra["token_logprobs"] == [-0.1, -0.2]
    back = schema.to_row(task, rollout)
    assert back["policy_version"] == row["policy_version"]
    assert back["sampling"] == row["sampling"] and back["token_logprobs"] == [-0.1, -0.2]
    assert schema.validate(row) == []


def test_staleness_report_counts_versions_stale_rows_and_coverage():
    def r(model, version, logprob=None, sampling=None, tokens=None):
        row = {"prompt": "p", "model_version": model, "policy_version": version}
        if logprob is not None:
            row["logprob"] = logprob
        if sampling is not None:
            row["sampling"] = sampling
        if tokens is not None:
            row["token_logprobs"] = tokens
        return row

    rows = [
        r("qwen3", "qwen3@aaa", -1.0, {"temperature": 0.8}, [-0.5, -0.5]),
        r("qwen3", "qwen3@aaa", -2.0, {"temperature": 0.8}),
        r("qwen2.5", "qwen2.5@bbb"),
        {"prompt": "legacy", "model_version": "qwen3"},
    ]
    report = staleness_report(rows, base_model="qwen3")
    assert report["n"] == 4
    assert report["versions"] == {"qwen3@aaa": 2, "qwen2.5@bbb": 1, "qwen3": 1}
    assert report["models"] == {"qwen3": 3, "qwen2.5": 1}
    assert report["stale"] == 1 and report["stale_share"] == 0.25
    assert report["sampling_coverage"] == 0.5 and report["logprob_coverage"] == 0.5
    assert report["token_logprob_coverage"] == 0.25
    assert report["temperatures"] == {"0.8": 2}
    joined = " ".join(report["warnings"])
    assert "3 policy versions" in joined and "off-policy for it" in joined
    assert "carry no logprob" in joined and "do not say how they were sampled" in joined

    clean = staleness_report(rows[:2])
    assert clean["warnings"] == [] and clean["stale"] is None
    assert zps.staleness_report([])["n"] == 0
