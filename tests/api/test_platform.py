"""whileai.platform against a fake transport: the calls a coding agent makes."""

from __future__ import annotations

import logging

import pytest

from whileai.platform import Agent, PlatformError, Run


class Fake:
    """Records every call; answers like the API."""

    def __init__(self, dashboard=None, fail_train_once=False):
        self.calls: list[tuple[str, str, object]] = []
        self.dashboard = dashboard or {}
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
        if path.endswith("/dashboard") or "/dashboard?" in path:
            return self.dashboard
        return {"ok": True}

    def paths(self, method=None):
        return [p for m, p, _ in self.calls if method in (None, m)]


def test_agent_registers_with_camel_case_fields():
    fake = Fake()
    Agent(
        "refund-bot",
        model="Qwen/Qwen3-4B",
        harness="h2",
        frontier={"name": "Sonnet 5", "score": 81, "cost_per_1k": 18.0, "p50_s": 2.1},
        transport=fake,
    )
    method, path, body = fake.calls[0]
    assert (method, path) == ("POST", "/agents")
    assert body["harness"] == {"label": "h2"}
    assert body["frontier"] == {"name": "Sonnet 5", "score": 81, "costPer1k": 18.0, "p50s": 2.1}


def test_behavior_maps_names_and_warns_when_judge_is_reward(caplog):
    fake = Fake()
    agent = Agent("a", transport=fake)
    with caplog.at_level(logging.WARNING, logger="whileai.platform"):
        agent.behavior(
            "refunds",
            test_version="v2",
            n=240,
            judge={"agreement": 0.86, "human_n": 60, "length_bias": 0.08},
            noise_floor=2.4,
            reward_is_judge=True,
        )
    _, path, body = fake.calls[-1]
    assert path == "/agents/a/behaviors/refunds"
    assert body == {
        "testVersion": "v2",
        "n": 240,
        "judge": {"agreement": 0.86, "humanN": 60, "lengthBias": 0.08},
        "noiseFloor": 2.4,
        "rewardIsJudge": True,
    }
    assert "reward_is_judge=False" in caplog.text


def test_run_buffers_logs_and_flushes_in_batches():
    fake = Fake()
    agent = Agent("a", model="Qwen/Qwen3-4B", transport=fake)
    run = agent.run(
        "v4", method="GRPO", targets=["refunds"], trained_on=["d1", "d2"], flush_every=3
    )
    _, _, body = fake.calls[-1]
    assert body["base"] == "Qwen/Qwen3-4B"
    assert body["trainedOn"] == ["d1", "d2"]
    assert isinstance(run, Run) and run.id == "run_abc"

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
    agent = Agent("a", transport=fake)
    run = agent.run("v1", flush_every=1)
    with caplog.at_level(logging.WARNING, logger="whileai.platform"):
        run.log(1, loss=2.0)
    assert run.errors == 1
    assert "retry" in caplog.text
    run.log(2, loss=1.9)
    sent = [b for m, p, b in fake.calls if p == "/runs/run_abc/train"]
    assert sent[-1] == [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.9}]


def test_score_and_finish(caplog):
    fake = Fake()
    agent = Agent("a", transport=fake)
    run = agent.run("v4", flush_every=100)
    run.log(5, reward=0.5)
    with caplog.at_level(logging.WARNING, logger="whileai.platform"):
        run.score("length", 76)
    assert "ci=" in caplog.text
    run.score("refunds", 83, ci=2.7, n=240, test_version="v2")
    _, path, body = fake.calls[-1]
    assert path == "/runs/run_abc/evals"
    assert body == [{"behavior": "refunds", "score": 83, "ci": 2.7, "n": 240, "testVersion": "v2"}]

    run.finish(hours=2.1, gpu="1xH100", cost_usd=31)
    assert run.status == "evaluated"
    # the buffered point went out before the PATCH
    assert fake.paths("POST")[-1] == "/runs/run_abc/train"
    method, path, body = fake.calls[-1]
    assert (method, path) == ("PATCH", "/runs/run_abc")
    assert body == {"status": "evaluated", "hours": 2.1, "gpu": "1xH100", "costUsd": 31, "steps": 5}


def test_context_manager_fails_the_run_on_exception():
    fake = Fake()
    agent = Agent("a", transport=fake)
    with pytest.raises(ValueError), agent.run("v1") as run:
        run.log(1, loss=1.0)
        raise ValueError("boom")
    assert run.status == "failed"
    method, path, body = fake.calls[-1]
    assert (method, path, body["status"]) == ("PATCH", "/runs/run_abc", "failed")
    assert "boom" in body["error"]


def test_trainer_callback_shape_is_honoured():
    """wai.TrainerCallback calls progress, log and finish on the run."""
    fake = Fake()
    run = Agent("a", transport=fake).run("v2", flush_every=100)
    run.progress(0, 300)
    run.log(10, loss=1.2, reward_mean=0.4)
    run.finish("done", summary={"train_loss": 0.9})
    assert run.total_steps == 300
    _, _, body = fake.calls[-1]
    assert body["steps"] == 300 and body["summary"] == {"train_loss": 0.9}


def test_live_and_promote():
    fake = Fake()
    agent = Agent("a", transport=fake)
    agent.live("2026-09-17", version="v3", replies=2400, flagged=98, p50_s=0.6, cost_usd=1.7)
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
    agent.promote("v4")
    assert fake.calls[-1] == ("POST", "/agents/a/promote", {"version": "v4"})


def test_verdict_reads_the_dashboard():
    dash = {
        "behavior": {"name": "refunds"},
        "verdict": {
            "candidate": "v4",
            "serving": "v3",
            "delta": 5,
            "excludesZero": True,
            "regressions": 1,
        },
    }
    agent = Agent("a", transport=Fake(dashboard=dash))
    assert agent.verdict() == "refunds: v4 beats v3 by 5 (interval excludes zero); 1 regression"
    agent2 = Agent(
        "a",
        transport=Fake(dashboard={"behavior": {"name": "refunds"}, "verdict": {"serving": "v3"}}),
    )
    assert agent2.verdict() == "refunds: no candidate scored against v3 yet"


def test_missing_key_names_the_fix(monkeypatch, tmp_path):
    monkeypatch.delenv("WHILEAI_API_KEY", raising=False)
    monkeypatch.delenv("ZEROPROOF_API_KEY", raising=False)
    monkeypatch.setenv("WHILEAI_HOME", str(tmp_path))
    monkeypatch.setenv("ZEROPROOF_HOME", str(tmp_path))
    with pytest.raises(PlatformError) as e:
        Agent("a")
    assert "whileai login" in str(e.value)
