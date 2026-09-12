"""The default path must execute every claimed stage, end to end."""
from __future__ import annotations

import json
import re

from tests.helpers import simulate_offline
import zeroproof.simulations as zps

INTERNAL_FIELDS = (
    "scenario_id", "scenario_dimensions", "arm", "prompt", "world_state",
    "faults", "steps", "final_text", "behavior_signature", "reward",
    "grader_reason", "rollout_index", "seed", "semantic_cluster",
    "semantic_novelty", "parent_failure_id",
)

CHAIN = [
    "agent ingestion",
    "generated candidate with arm provenance",
    "semantic embedding produced",
    "selection reason / novelty",
    "world/fault instantiated",
    "model rollout",
    "full tool trajectory",
    "behavior signature",
    "row stored",
]


class _SemanticStub:
    name = "stub-semantic"
    semantic = True

    def embed(self, texts):
        vectors = []
        for i, text in enumerate(texts):
            vec = [0.0] * 8
            vec[i % 8] = 1.0
            vec[len(text) % 8] += 0.25
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            vectors.append([x / norm for x in vec])
        return vectors


def _fake_complete(_base_url, _model, messages, **_kwargs):
    prompt = messages[-1]["content"]
    ids = re.findall(r'"region_id":\s*"([^"]+)"', prompt)
    payload = []
    for i, region_id in enumerate(ids[:12]):
        payload.append({
            "region_id": region_id,
            "message": (f"Hi, I need a refund on ORD-{1000 + i}. "
                        f"The last attempt failed and your system timed out."),
        })
    payload.append({"region_id": None,
                    "message": "What is the capital of Mongolia?"})
    return {"content": json.dumps(payload)}


def test_default_path_executes_every_stage(monkeypatch, capsys):
    monkeypatch.setattr("zeroproof.simulations.generate.generator.complete", _fake_complete)

    data = simulate_offline(
        budget=12, embedder=_SemanticStub(),
        simulator="vllm:fake@http://example",
        concurrency=6, mode="adaptive", per_round=6)

    for stage in CHAIN:
        assert stage in data.stages, f"missing stage: {stage}\n{data.stages}"
    assert data.semantic is True
    assert "generator_fallback" not in data.degraded
    assert "semantic_embedding_unavailable" not in data.degraded
    assert data.stages == CHAIN
    assert data.profile and data.profile.tools
    assert data.profile.capabilities

    row = data.trajectories[0]
    for field in INTERNAL_FIELDS:
        assert field in row, f"row missing {field}"
    assert row["arm"]
    assert row["seed"] == 0
    assert row["behavior_signature"]
    assert row["steps"]
    assert row["selection_reason"]
    assert row["semantic_cluster"] is not None
    assert row["semantic_novelty"] is not None
    exported = data.rows()[0]
    assert {"prompt", "messages", "steps", "final_text"} <= set(exported)
    assert exported["messages"][0]["role"] == "user"
    assert "reward" not in exported
    for dropped in ("selection_reason", "parent_failure_id", "arm",
                    "semantic_cluster", "semantic_novelty", "behavior_signature",
                    "grader_reason", "seed", "scenario_dimensions"):
        assert dropped not in exported
    assert any(t.get("faults") or t.get("world_state") for t in data.trajectories)
    assert any((t.get("scenario_dimensions") or {}).get("tool")
               or (t.get("scenario_dimensions") or {}).get("stance")
               or t.get("arm") == "open_ended"
               for t in data.trajectories)

    frozen = [dict(t) for t in data.trajectories]
    assert all(t["reward"] is None for t in frozen)
    data.grade(grader=lambda t: 0.25)
    assert all(t["reward"] == 0.25 for t in data.trajectories)
    assert [t["behavior_signature"] for t in frozen] == [
        t["behavior_signature"] for t in data.trajectories]

    chain = " → ".join(CHAIN + ["grade() works afterward on frozen rows"])
    print(chain)
    captured = capsys.readouterr()
    assert "agent ingestion" in captured.out
    assert "grade() works afterward on frozen rows" in captured.out



def test_one_failed_writer_wave_is_not_a_fallback(monkeypatch):
    import threading
    state = {"writer_calls": 0, "lock": threading.Lock()}

    def flaky_complete(base_url, model, messages, **kwargs):
        if '"region_id"' in messages[-1]["content"]:
            with state["lock"]:
                state["writer_calls"] += 1
                first = state["writer_calls"] == 1
            if first:
                raise RuntimeError("HTTP 502 from the writer")
        return _fake_complete(base_url, model, messages, **kwargs)

    monkeypatch.setattr("zeroproof.simulations.generate.generator.complete", flaky_complete)
    data = simulate_offline(budget=8, simulator="vllm:fake@http://example",
                            concurrency=4, per_round=6)
    assert len(data.trajectories) == 8
    assert "generator_fallback" not in data.degraded
    assert "502" in data.search["writer_errors"]["llm_guided"]
