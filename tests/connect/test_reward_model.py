"""A finished ``method="rm"`` run as a judge: ``reward_model``."""

from __future__ import annotations

import pytest

import whileai.simulations as wai
from whileai.simulations.training import METHODS, RewardModel, TrainingRun, reward_model, train

ROW_PASS = {
    "prompt": "refund order 812",
    "final_text": "<tool_call>lookup</tool_call>",
    "scenario_id": "s1",
}
ROW_FAIL = {"prompt": "refund order 812", "final_text": "Refunded!", "scenario_id": "s1"}


class Host:
    """The gate's score route, scripted: scores by row text, one threshold."""

    def __init__(self, threshold=0.25):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.threshold = threshold

    def __call__(self, method, path, api_key=None, body=None, **kw):
        if method == "GET" and path.endswith("/profile"):
            # every task has a pass and a fail: clean for a paired method
            return {
                "profile": {
                    "rows": 160,
                    "split": {"pass": 80, "fail": 80, "ungraded": 0},
                    "tasks": 40,
                    "tasks_with_repeats": 40,
                    "mixed_tasks": 40,
                }
            }
        self.calls.append((method, path, body))
        if method == "POST" and path.endswith("/score"):
            scores = [
                1.5 if "tool_call" in str(r.get("final_text")) else -0.5 for r in body["rows"]
            ]
            return {
                "runId": "run_rm1",
                "scores": scores,
                "threshold": self.threshold,
                "pass": [s >= self.threshold for s in scores],
                "base": "Qwen/Qwen2.5-1.5B-Instruct",
            }
        if method == "POST" and path.endswith("/train"):
            return {
                "training": {
                    "runId": "run_rm1",
                    "method": "rm",
                    "status": "running",
                    "callId": "fc",
                }
            }
        raise AssertionError(f"unexpected {method} {path}")


def test_rm_is_a_train_method():
    assert "rm" in METHODS
    host = Host()
    run = train("ds_1", method="rm", steps=30, transport=host)
    assert run.run_id == "run_rm1" and run.method == "rm"
    assert host.calls[0][2] == {"method": "rm", "steps": 30}


def test_reward_model_honors_the_judge_contract():
    host = Host()
    judge = reward_model("run_rm1", transport=host)
    assert judge.__name__ == "reward_model:run_rm1"
    out = judge(ROW_PASS)
    assert out["reward"] == 1 and out["rm_score"] == 1.5 and out["threshold"] == 0.25
    assert "run_rm1" in out["reason"]
    assert judge(ROW_FAIL)["reward"] == 0
    assert host.calls[0][1] == "/runs/run_rm1/score" and host.calls[0][2] == {"rows": [ROW_PASS]}


def test_reward_model_runs_through_run_judge():
    scored = wai.run_judge(
        [ROW_PASS, ROW_FAIL], reward_model("run_rm1", transport=Host()), concurrency=1
    )
    rewards = [r["reward"] for r in scored.rows]
    assert rewards == [1, 0]
    assert all(r["judge_status"] == "ok" for r in scored.rows)
    assert scored.rows[0]["judge_name"] == "reward_model:run_rm1"
    assert scored.rows[0]["judge_meta"]["rm_score"] == 1.5


def test_threshold_override_and_batching():
    host = Host()
    judge = reward_model("run_rm1", threshold=2.0, transport=host)
    rows = [ROW_PASS] * 300
    out = judge.score(rows)
    assert len(out) == 300 and all(o["reward"] == 0 for o in out)  # 1.5 < 2.0
    assert [len(c[2]["rows"]) for c in host.calls] == [256, 44]
    assert judge.base == "Qwen/Qwen2.5-1.5B-Instruct"


def test_reward_model_from_a_run_handle():
    run = TrainingRun("run_rm1", name="rm", transport=Host())
    assert reward_model(run, transport=Host()).run_id == "run_rm1"
    with pytest.raises(ValueError):
        RewardModel("")


def test_short_answer_is_an_error_not_a_zero():
    def bad(method, path, api_key=None, body=None, **kw):
        return {"scores": [], "threshold": 0.0}

    with pytest.raises(RuntimeError):
        reward_model("run_rm1", transport=bad).score([ROW_PASS])
