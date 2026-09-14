"""Every label says which judge made it; the judge is measured against gold."""

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations import schema
from zeroproof.simulations.score import grade_llm
from zeroproof.simulations.score.agreement import judge_agreement
from zeroproof.simulations.score.judging import run_judge


def _rows(n, start=0):
    return [
        {
            "prompt": f"ask {i}",
            "scenario_id": f"sc-{i}",
            "rollout_index": 0,
            "final_text": "Order shipped." if i % 2 else "I invented it.",
            "steps": [],
        }
        for i in range(start, start + n)
    ]


def test_grade_stamps_judge_provenance_through_attach(monkeypatch):
    monkeypatch.setattr(grade_llm, "require_judge_key", lambda *a, **k: "vllm:judge-model@http://x")
    monkeypatch.setattr(
        grade_llm,
        "grade_one",
        lambda row, **k: {"reward": 1 if "shipped" in row["final_text"] else 0, "reason": "r"},
    )
    rows = _rows(4)
    rows[0]["reason"] = "stale reason from an earlier pass"
    rows[0]["failure_class"] = "fabrication"
    report = grade_llm.apply_grade_llm(rows)
    assert report["graded"] == 4
    version = report["judge_version"]
    assert version.startswith("judge-model@") and len(version.split("@")[1]) == 12
    for row in rows:
        assert row["judge_name"] == "grade_llm"
        assert row["judge_status"] == "ok"
        meta = row["judge_meta"]
        assert meta["model"] == "judge-model"
        assert meta["version"] == version
        assert meta["temperature"] == 0.0
        assert meta["prompt_sha"] == version.split("@")[1]
    assert rows[1]["reward"] == 1 and "failure_class" not in rows[1]
    assert rows[0]["reward"] == 0 and rows[0]["failure_class"] == "fabrication"
    # a custom rubric is a different judge
    assert grade_llm.judge_version("vllm:m@http://x", "grade harder") != grade_llm.judge_version(
        "vllm:m@http://x"
    )
    _, _, judgments, _ = schema.from_row(rows[1])
    assert judgments[0].scorer.version == version
    assert judgments[0].scorer.kind == "judge"


def test_run_judge_version_lands_in_lineage_and_scorer():
    scored = run_judge(_rows(2), lambda t: 1, judge_name="mine", version="mine-v2")
    assert scored[0]["lineage"]["judge_version"] == "mine-v2"
    _, _, judgments, _ = schema.from_row(scored[0])
    assert judgments[0].scorer.version == "mine-v2"
    plain = run_judge(_rows(1), lambda t: 1)
    assert "judge_version" not in plain[0]["lineage"]


def test_attach_replaces_stale_verdict_fields():
    row = {"prompt": "p", "reward": 0, "reason": "old", "failure_class": "arithmetic"}
    judgment = schema.Judgment(
        rollout_id="r", scorer=schema.ScorerRef(name="j", kind="judge"), reward=1.0
    )
    schema.attach(row, judgment)
    assert row["reward"] == 1 and isinstance(row["reward"], int)
    assert "reason" not in row and "failure_class" not in row
    assert row["judge_name"] == "j" and row["judge_status"] == "ok"
    assert "judge_meta" not in row


def test_agreement_against_a_gold_key():
    rows = []
    for i in range(60):
        gold = 1 if i % 2 else 0
        judge = gold
        if i in (0, 2, 4):  # judge passes a gold failure: the leak
            judge = 1
        if i == 1:  # judge fails a gold pass
            judge = 0
        rows.append({"prompt": f"a{i}", "reward": judge, "gold_reward": gold})
    rows.append({"prompt": "partial", "reward": 0.5, "gold_reward": 1})
    rows.append({"prompt": "unlabeled", "reward": 1})
    out = judge_agreement(rows)
    assert out["n"] == 60 and out["n_skipped"] == 2
    assert out["confusion"] == {"tp": 29, "fp": 3, "fn": 1, "tn": 27}
    assert out["agreement"] == round(56 / 60, 4)
    assert out["pass_when_gold_fail"] == 0.1
    assert out["fail_when_gold_pass"] == round(1 / 30, 4)
    assert 0.8 < out["kappa"] < 0.9
    assert any("passed 3 of 30 gold failures" in w for w in out["warnings"])
    assert not any("coarse" in w for w in out["warnings"])


def test_agreement_against_a_second_pass_and_small_sample_warning():
    first = run_judge(_rows(10), lambda t: 1 if "shipped" in t["final_text"] else 0)
    second = run_judge(_rows(10), lambda t: 1)
    out = first.agreement(second.rows)
    assert out["n"] == 10 and out["n_unmatched"] == 0
    # judge = first pass, gold = second pass (all 1): the five 0s read as misses
    assert out["confusion"] == {"tp": 5, "fp": 0, "fn": 5, "tn": 0}
    assert out["agreement"] == 0.5
    assert any("coarse" in w for w in out["warnings"])
    unmatched = judge_agreement(first.rows, _rows(3, start=100))
    assert unmatched["n"] == 0 and unmatched["n_unmatched"] == 10
    assert any("no judged row matched" in w for w in unmatched["warnings"])


def test_agreement_is_exported():
    assert zps.judge_agreement is judge_agreement
    with pytest.raises(TypeError):
        judge_agreement()  # type: ignore[call-arg]


def _marker_judge(row):
    """The documented judge contract, returning a behavior measurement."""
    return {
        "reward": 1 if "shipped" in row["final_text"] else 0,
        "reason": "ok",
        "markers": {"phantom_number": 0.0 if "shipped" in row["final_text"] else 1.0},
    }


def test_markers_a_judge_returns_reach_marker_summary():
    """A marker is only a measurement if something can read it back.

    ``normalize_judge_result`` sweeps every non-reward key into
    ``judge_meta``; ``marker_summary`` reads ``row["markers"]``. Without
    the lift the two never meet and every marker silently reports nothing.
    """
    scored = run_judge(_rows(10), _marker_judge, judge_name="m")
    rows = list(scored.rows)
    assert all(
        row["markers"] == {"phantom_number": pytest.approx(0.0 if i % 2 else 1.0)}
        for i, row in enumerate(rows)
    )
    # judge_meta keeps its copy: nothing that read it before breaks.
    assert rows[0]["judge_meta"]["markers"] == {"phantom_number": 1.0}
    summary = zps.marker_summary(rows)
    assert summary["phantom_number"]["n_rows"] == 10
    assert summary["phantom_number"]["mean"] == pytest.approx(0.5)


def test_judge_markers_merge_with_markers_already_on_the_row():
    rows = [dict(row, markers={"kept": 1.0}) for row in _rows(2)]
    scored = run_judge(rows, _marker_judge, judge_name="m")
    assert sorted(zps.marker_summary(scored.rows)) == ["kept", "phantom_number"]


def test_a_judge_without_markers_adds_no_markers_key():
    scored = run_judge(_rows(2), lambda t: 1)
    assert all("markers" not in row for row in scored.rows)


def test_simulate_grader_markers_survive_onto_the_trajectories():
    """``grader=`` is the only in-simulate score hook; it copied six keys
    and dropped markers on the floor."""
    from tests.helpers import simulate_offline

    data = simulate_offline(budget=8, per_round=8, concurrency=1, grader=_marker_judge)
    assert data.trajectories
    assert all(isinstance(row.get("markers"), dict) for row in data.trajectories)
    assert zps.marker_summary(data.trajectories)["phantom_number"]["n_rows"] == len(
        data.trajectories
    )
    # The lifted markers are wire-legal and round-trip to Marker objects.
    assert all(not schema.validate(row) for row in data.trajectories)
    assert schema.from_row(data.trajectories[0])[3][0].name == "phantom_number"
