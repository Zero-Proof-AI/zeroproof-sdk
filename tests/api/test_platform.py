"""whileai.platform against a fake transport: the calls a coding agent makes."""

from __future__ import annotations

import logging
from datetime import date

import pytest
from pydantic import ValidationError

from whileai.platform import (
    Behavior,
    Dashboard,
    Frontier,
    Harness,
    Judge,
    LiveDay,
    PlatformError,
    Run,
    RunSpec,
    Score,
    Tracked,
    describe,
    track,
)


class Fake:
    """Records every call; answers like the API."""

    def __init__(self, dashboard=None, fail_train_once=False):
        self.calls: list[tuple[str, str, object]] = []
        self.dashboard = dashboard or {"agent": {"id": "a", "name": "a"}}
        self.fail_train_once = fail_train_once

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path == "/agents" and method == "POST":
            return {"id": body["id"], "name": body.get("name"), "serving": None}
        if path == "/runs" and method == "POST":
            return {"id": body.get("id") or "run_abc", "version": body["version"]}
        if path.endswith("/train"):
            if self.fail_train_once:
                self.fail_train_once = False
                raise PlatformError(503, "down")
            return {"points": len(body)}
        if path.endswith("/evals"):
            return {"evals": body}
        if "/dashboard" in path:
            return self.dashboard
        if "/behaviors/" in path:
            return {"name": path.rsplit("/", 1)[1], **body}
        return {"ok": True}

    def paths(self, method=None):
        return [p for m, p, _ in self.calls if method in (None, m)]


# ------------------------------------------------------------------ models


def test_models_go_on_the_wire_in_camel_case():
    f = Frontier(name="Sonnet 5", score=81, cost_per_1k=18.0, p50_s=2.1)
    assert f.wire() == {"name": "Sonnet 5", "score": 81.0, "costPer1k": 18.0, "p50s": 2.1}
    b = Behavior(
        name="refunds",
        test_version="v2",
        n=240,
        judge=Judge(agreement=0.86, human_n=60, length_bias=0.08),
        noise_floor=2.4,
    )
    assert b.wire() == {
        "name": "refunds",
        "testVersion": "v2",
        "n": 240,
        "judge": {"agreement": 0.86, "humanN": 60, "lengthBias": 0.08},
        "noiseFloor": 2.4,
    }
    assert LiveDay(day="2026-09-17", version="v3", replies=10, p50_s=0.6).wire() == {
        "day": "2026-09-17",
        "version": "v3",
        "replies": 10,
        "flagged": 0,
        "p50s": 0.6,
    }


def test_models_validate():
    with pytest.raises(ValidationError):
        Judge(agreement=1.4)
    with pytest.raises(ValidationError):
        Behavior(name="has space")
    with pytest.raises(ValidationError):
        Score(behavior="refunds", score=80, ci=-1)
    with pytest.raises(ValidationError):
        LiveDay(day="yesterday", version="v1", replies=1)


def test_harness_fingerprint_changes_with_the_prompt():
    a = Harness(instructions="Be brief.", tools=["lookup_order", "issue_refund"], model="m")
    b = Harness(instructions="Be brief!", tools=["issue_refund", "lookup_order"], model="m")
    c = Harness(instructions="Be brief.", tools=["issue_refund", "lookup_order"], model="m")
    assert a.fingerprint == c.fingerprint  # tool order does not matter
    assert a.fingerprint != b.fingerprint  # the prompt does
    assert a.version == f"h-{a.fingerprint}"
    assert Harness(label="h2").version == "h2"
    assert a.wire()["hash"] == a.fingerprint and a.wire()["label"] == a.version


# ------------------------------------------------------------------ track


def test_track_by_name_registers():
    fake = Fake()
    t = track(
        "refund-bot",
        model="Qwen/Qwen3-4B",
        harness="h2",
        frontier={"name": "Sonnet 5", "score": 81, "cost_per_1k": 18.0},
        transport=fake,
    )
    assert isinstance(t, Tracked) and t.id == "refund-bot"
    method, path, body = fake.calls[0]
    assert (method, path) == ("POST", "/agents")
    assert body["model"] == "Qwen/Qwen3-4B"
    assert body["harness"]["label"] == "h2"
    assert body["frontier"] == {"name": "Sonnet 5", "score": 81.0, "costPer1k": 18.0}


class OpenAIStyleAgent:
    """Shaped like an OpenAI Agents SDK agent: name, instructions, tools, model."""

    def __init__(self):
        self.name = "Refund bot"
        self.instructions = "Handle refunds politely."
        self.model = "gpt-5"

        def lookup_order():
            pass

        self.tools = [lookup_order, type("Tool", (), {"name": "issue_refund"})()]


class ClaudeStyleOptions:
    system_prompt = "You are a refund agent."
    allowed_tools = ["Read", "Bash"]
    model = "claude-sonnet-5"


def test_describe_reads_frameworks_by_duck_typing():
    d = describe(OpenAIStyleAgent())
    assert d.name == "Refund bot" and d.model == "gpt-5"
    assert d.harness.instructions == "Handle refunds politely."
    assert d.harness.tools == ["lookup_order", "issue_refund"]

    d2 = describe(ClaudeStyleOptions(), name="claude-bot")
    assert d2.harness.tools == ["Read", "Bash"] and d2.harness.model == "claude-sonnet-5"
    assert d2.harness.instructions == "You are a refund agent."

    d3 = describe({"name": "dict-bot", "tools": [{"function": {"name": "search"}}]})
    assert d3.harness.tools == ["search"]


def test_track_takes_the_agent_object():
    fake = Fake()
    t = track(OpenAIStyleAgent(), transport=fake)
    assert t.id == "Refund-bot" and t.model == "gpt-5"
    _, _, body = fake.calls[0]
    assert body["harness"]["tools"] == ["lookup_order", "issue_refund"]
    assert body["harness"]["label"].startswith("h-")


def test_behavior_warns_when_judge_is_reward(caplog):
    fake = Fake()
    t = track("a", transport=fake)
    with caplog.at_level(logging.WARNING, logger="whileai.platform"):
        out = t.behavior("refunds", test_version="v2", n=240, reward_is_judge=True)
    assert isinstance(out, Behavior) and out.name == "refunds"
    _, path, body = fake.calls[-1]
    assert path == "/agents/a/behaviors/refunds"
    assert body == {"testVersion": "v2", "n": 240, "rewardIsJudge": True}
    assert "reward_is_judge=False" in caplog.text


# ------------------------------------------------------------------ runs


def test_run_buffers_logs_and_flushes_in_batches():
    fake = Fake()
    t = track("a", model="Qwen/Qwen3-4B", harness="h2", transport=fake)
    run = t.run("v4", method="GRPO", targets=["refunds"], trained_on=["d1", "d2"], flush_every=3)
    _, _, body = fake.calls[-1]
    assert body["base"] == "Qwen/Qwen3-4B" and body["harness"] == "h2"
    assert body["trainedOn"] == ["d1", "d2"] and body["agent"] == "a"
    assert isinstance(run, Run) and run.id == "run_abc" and isinstance(run.spec, RunSpec)

    run.log(0, reward=0.3, kl=0.0)
    run.log(10, reward=0.4, kl=float("nan"), note="ignored")
    assert fake.paths("POST").count("/runs/run_abc/train") == 0
    run.log(20, reward=0.5)
    assert fake.paths("POST").count("/runs/run_abc/train") == 1
    sent = next(b for m, p, b in fake.calls if p == "/runs/run_abc/train")
    assert sent == [
        {"step": 0, "reward": 0.3, "kl": 0.0},
        {"step": 10, "reward": 0.4},
        {"step": 20, "reward": 0.5},
    ]
    assert run.step == 20


def test_failed_flush_is_retried_not_raised(caplog):
    fake = Fake(fail_train_once=True)
    run = track("a", transport=fake).run("v1", flush_every=1)
    with caplog.at_level(logging.WARNING, logger="whileai.platform"):
        run.log(1, loss=2.0)
    assert run.errors == 1 and "retry" in caplog.text
    run.log(2, loss=1.9)
    sent = [b for m, p, b in fake.calls if p == "/runs/run_abc/train"]
    assert sent[-1] == [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.9}]


def test_score_and_finish(caplog):
    fake = Fake()
    run = track("a", transport=fake).run("v4", flush_every=100)
    run.log(5, reward=0.5)
    with caplog.at_level(logging.WARNING, logger="whileai.platform"):
        run.score("length", 76)
    assert "ci=" in caplog.text
    recorded = run.score(Score(behavior="refunds", score=83, ci=2.7, n=240, test_version="v2"))
    assert isinstance(recorded, Score) and recorded.ci == 2.7
    _, path, body = fake.calls[-1]
    assert path == "/runs/run_abc/evals"
    assert body == [
        {"behavior": "refunds", "score": 83.0, "ci": 2.7, "n": 240, "testVersion": "v2"}
    ]

    run.finish(hours=2.1, gpu="1xH100", cost_usd=31)
    assert run.status == "evaluated"
    assert fake.paths("POST")[-1] == "/runs/run_abc/train"  # flushed before the PATCH
    method, path, body = fake.calls[-1]
    assert (method, path) == ("PATCH", "/runs/run_abc")
    assert body == {
        "status": "evaluated",
        "hours": 2.1,
        "gpu": "1xH100",
        "costUsd": 31.0,
        "steps": 5,
    }


def test_context_manager_fails_the_run_on_exception():
    fake = Fake()
    with pytest.raises(ValueError), track("a", transport=fake).run("v1") as run:
        run.log(1, loss=1.0)
        raise ValueError("boom")
    assert run.status == "failed"
    method, path, body = fake.calls[-1]
    assert (method, path, body["status"]) == ("PATCH", "/runs/run_abc", "failed")
    assert "boom" in body["error"]


def test_trainer_callback_shape_is_honoured():
    """wai.TrainerCallback calls progress, log and finish on the run."""
    fake = Fake()
    run = track("a", transport=fake).run("v2", flush_every=100)
    run.progress(0, 300)
    run.log(10, loss=1.2, reward_mean=0.4)
    run.finish("done", summary={"train_loss": 0.9})
    assert run.total_steps == 300
    _, _, body = fake.calls[-1]
    assert body["steps"] == 300 and body["summary"] == {"train_loss": 0.9}


# ------------------------------------------------------------------ live, verdict


def test_live_and_promote():
    fake = Fake()
    t = track("a", transport=fake)
    t.live(date(2026, 9, 17), version="v3", replies=2400, flagged=98, p50_s=0.6, cost_usd=1.7)
    _, path, body = fake.calls[-1]
    assert path == "/live"
    assert body == [
        {
            "agent": "a",
            "day": "2026-09-17",
            "version": "v3",
            "replies": 2400,
            "flagged": 98,
            "p50s": 0.6,
            "costUsd": 1.7,
        }
    ]
    t.promote("v4")
    assert fake.calls[-1] == ("POST", "/agents/a/promote", {"version": "v4"})


def test_dashboard_and_verdict_are_typed():
    dash = {
        "agent": {"id": "a", "name": "a", "serving": "v3", "candidate": "v4"},
        "behavior": {"name": "refunds", "testVersion": "v2", "n": 240},
        "behaviors": ["refunds", "length"],
        "versions": [{"v": "base", "score": 62, "ci": 3.1}, {"v": "v4", "score": 83, "ci": 2.7}],
        "deltas": [
            {"name": "refunds", "delta": 5, "target": True},
            {"name": "length", "delta": -4},
        ],
        "live": {"days": ["d-1"], "flaggedPct": [4.1], "replies7d": 18400, "newFailures": 412},
        "verdict": {
            "candidate": "v4",
            "serving": "v3",
            "delta": 5,
            "excludesZero": True,
            "regressions": 1,
        },
        "someNewField": 1,
    }
    t = track("a", transport=Fake(dashboard=dash))
    d = t.dashboard()
    assert isinstance(d, Dashboard) and d.versions[1].score == 83 and d.live.replies_7d == 18400
    assert d.deltas[0].target and d.deltas[1].delta == -4
    assert str(t.verdict()) == "refunds: v4 beats v3 by 5 (interval excludes zero); 1 regression"

    empty = track(
        "a",
        transport=Fake(dashboard={"agent": {"id": "a", "name": "a"}, "verdict": {"serving": "v3"}}),
    )
    assert str(empty.verdict()) == "?: no candidate scored against v3 yet"


def test_missing_key_names_the_fix(monkeypatch, tmp_path):
    monkeypatch.delenv("WHILEAI_API_KEY", raising=False)
    monkeypatch.delenv("ZEROPROOF_API_KEY", raising=False)
    monkeypatch.setenv("WHILEAI_HOME", str(tmp_path))
    monkeypatch.setenv("ZEROPROOF_HOME", str(tmp_path))
    with pytest.raises(PlatformError) as e:
        track("a")
    assert "whileai login" in str(e.value)
