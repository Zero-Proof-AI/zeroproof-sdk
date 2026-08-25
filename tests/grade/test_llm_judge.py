"""Optional LLM judge pass (offline mocked)."""
from __future__ import annotations

import json

import pytest

import zeroproof_simulations as zps
from tests.helpers import simulate_offline
from zeroproof_simulations.llm_judge import DEFAULT_JUDGE_SPEC, MISSING_JUDGE_KEY


def _fake_judge_complete(_url, _model, messages, **_kwargs):
    return {"content": json.dumps({"score": 0.5, "reason": "partial compliance"})}


def _run(**kwargs):
    kwargs.setdefault("grade", True)
    kwargs.setdefault("budget", 6)
    kwargs.setdefault("per_round", 6)
    return simulate_offline(**kwargs)


def test_simulate_default_does_not_llm_grade(monkeypatch):
    calls: list[str] = []

    def tracked(*_args, **_kwargs):
        calls.append("llm")
        return {"content": '{"score": 1, "reason": "ok"}'}

    monkeypatch.setattr("zeroproof_simulations.llm_judge.complete", tracked)
    data = _run(budget=8)
    assert len(data.trajectories) == 8
    assert all(t.get("llm_reward") is None for t in data.trajectories)
    assert calls == []


def test_grade_llm_true_writes_fields(monkeypatch, tmp_path):
    monkeypatch.setattr("zeroproof_simulations.llm_judge.complete", _fake_judge_complete)
    data = _run()
    before = [(t["reward"], t.get("grader_reason")) for t in data.trajectories]
    data.grade(llm=True, api_key="sk-test")
    assert all(t["llm_reward"] == 0.5 for t in data.trajectories)
    assert all(t["llm_reason"] == "partial compliance" for t in data.trajectories)
    after = [(t["reward"], t.get("grader_reason")) for t in data.trajectories]
    assert before == after

    path = str(tmp_path / "scored.jsonl")
    data.save(path)
    row = json.loads(open(path).readline())
    assert "llm_reward" in row and "llm_reason" in row
    assert "reward" not in row or row.get("reward") is None
    assert row["scenario_id"]
    assert "selection_reason" not in row
    assert "grader_reason" not in row


def test_llm_grade_needs_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    data = _run(budget=4)
    with pytest.raises(RuntimeError, match="API key") as exc:
        data.grade(llm=True)
    assert str(exc.value) == MISSING_JUDGE_KEY
    assert "\n" not in str(exc.value)
    assert str(exc.value).count(".") == 1


def test_llm_grade_helper_and_unreachable(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise OSError("judge offline")

    monkeypatch.setattr("zeroproof_simulations.llm_judge.complete", blocked)
    data = _run(budget=4)
    zps.llm_grade(data, api_key="sk-test")
    assert all(t["llm_reward"] is None for t in data.trajectories)
    assert "llm_judge_unreachable" in data.degraded


def test_simulate_llm_grade_flag_runs_after_rollout(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("zeroproof_simulations.llm_judge.complete", _fake_judge_complete)
    data = _run(budget=4, llm_grade=True)
    assert all(t["llm_reward"] == 0.5 for t in data.trajectories)
    assert all(t["reward"] is None for t in data.trajectories)


def test_llm_judge_uses_openai_not_hosted(monkeypatch):
    seen: list[tuple] = []

    def fake(url, model, messages, **kwargs):
        seen.append((url, model, kwargs.get("api_key"), messages[0].get("role")))
        return {"content": '{"score": 1, "reason": "ok"}'}

    monkeypatch.setattr("zeroproof_simulations.llm_judge.complete", fake)
    data = _run(budget=4)
    data.grade(llm=True, api_key="sk-test")
    assert seen
    assert "modal.run" not in seen[0][0]
    assert seen[0][1] == "gpt-4o-mini"
    assert seen[0][2] == "sk-test"
    assert seen[0][3] == "system"
    assert DEFAULT_JUDGE_SPEC.startswith("openai:")


def test_llm_grade_rewrites_same_jsonl(monkeypatch, tmp_path):
    monkeypatch.setattr("zeroproof_simulations.llm_judge.complete", _fake_judge_complete)
    dest = tmp_path / "out.jsonl"
    data = _run(output=str(dest), budget=4)
    first = json.loads(dest.read_text().splitlines()[0])
    assert "llm_reward" not in first
    assert first["scenario_id"]
    data.grade(llm=True, api_key="sk-test")
    row = json.loads(dest.read_text().splitlines()[0])
    assert row["llm_reward"] == 0.5
    assert row["llm_reason"] == "partial compliance"
    assert "reward" not in row
