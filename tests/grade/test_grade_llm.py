"""Binary judge contract. Offline: no hosted calls."""

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.score.grade_llm import (
    AUDIT_SYSTEM,
    JUDGE_MAX_TOKENS,
    JUDGE_SYSTEM,
    _injected_fault_lead,
    _parse_verdict,
    _render_payload,
    _user_message,
)


def test_data_grade_is_the_optional_qwen_followup(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    seen = {}

    def fake_apply(rows, **kwargs):
        seen["rows"] = rows
        seen["key"] = kwargs.get("api_key")
        rows[0]["reward"] = 1
        rows[0]["reason"] = "ok"
        return {"status": "judged", "graded": 1}

    monkeypatch.setattr("zeroproof.simulations.data.apply_grade_llm", fake_apply)
    data = zps.SimulationData(
        trajectories=[
            {
                "prompt": "help",
                "steps": [],
                "final_text": "done",
            }
        ]
    )
    report = data.grade()
    assert report == {
        "status": "judged",
        "graded": 1,
        "rubric": "conduct_floor",
        "note": "conduct floor only: pass rubric= or add rubric.md to the spec so the task itself is scored",
    }
    assert seen["rows"] is data.trajectories
    assert seen["key"] is None
    assert data.trajectories[0]["reward"] == 1


def test_judge_system_grades_agent_not_sandbox():
    text = JUDGE_SYSTEM.lower()
    assert "grade the agent" in text
    assert "injected" in text
    assert "world_state" in text
    assert "honest" in text or "told the truth" in text
    assert "exists" in text
    assert '"score": 0' in JUDGE_SYSTEM
    assert '"score": 1' in JUDGE_SYSTEM
    assert '"reason"' in JUDGE_SYSTEM
    assert "(a)" in JUDGE_SYSTEM and "(b)" in JUDGE_SYSTEM
    assert "0.5" not in JUDGE_SYSTEM
    assert JUDGE_MAX_TOKENS >= 120


def test_audit_system_checks_existing_labels():
    text = AUDIT_SYSTEM.lower()
    assert "fair" in text
    assert "grade the agent" in text
    assert "existing" in text
    assert '"score": 0' in AUDIT_SYSTEM
    assert '"score": 1' in AUDIT_SYSTEM
    assert "0.5" not in AUDIT_SYSTEM


def test_judge_payload_includes_world_and_faults():
    blob = _render_payload(
        {
            "prompt": "find towels",
            "world_state": "entity exists",
            "faults": {"*": {"mode": "timeout", "rate": 1.0}},
            "steps": [
                {
                    "tool": "search_products",
                    "arguments": {"query": "towels"},
                    "result": {"status": "timeout"},
                }
            ],
            "final_text": "The search timed out.",
        }
    )
    assert "entity exists" in blob
    assert "timeout" in blob
    assert "injected_faults" in blob
    assert "world_state" in blob


def test_injected_fault_leads_the_prompt():
    row = {
        "prompt": "what time do you close",
        "world_state": "entity exists",
        "faults": {"*": {"mode": "timeout", "rate": 1.0}},
        "steps": [
            {"tool": "get_hours", "arguments": {"id": "r1"}, "result": {"status": "timeout"}}
        ],
        "final_text": "The hours lookup timed out.",
    }
    lead = _injected_fault_lead(row)
    assert lead.startswith("A timeout was injected on get_hours.")
    assert "honestly" in lead.lower()
    user = _user_message(row)
    assert user.startswith("A timeout was injected on get_hours.")
    assert user.find("A timeout was injected") < user.find("injected_faults")


def test_parse_verdict_reason_before_score():
    score, reason = _parse_verdict('{"reason": "Honest timeout report.", "score": 1}')
    assert score == 1
    assert "timeout" in reason.lower()


def test_long_trajectory_payload_keeps_faults_and_ending():
    import json

    from zeroproof.simulations.score.grade_llm import _render_payload

    steps = [
        {"tool": f"step_{i}", "arguments": {"n": i}, "result": {"status": "ok", "data": "x" * 300}}
        for i in range(40)
    ]
    steps[17]["result"] = {"status": "timeout"}
    row = {
        "prompt": "do the long thing",
        "steps": steps,
        "final_text": "All done, but step 17 timed out.",
    }
    text = _render_payload(row)
    assert len(text) <= 8000
    payload = json.loads(text)
    rendered = [s for s in payload["steps"] if "tool" in s]
    names = [s["tool"] for s in rendered]
    assert "step_0" in names  # opening kept
    assert "step_17" in names  # the fault kept
    assert "step_39" in names  # the ending kept
    assert payload["final_text"].startswith("All done")
    assert any("skipped_steps" in s for s in payload["steps"])


def test_parse_verdict_rejects_non_binary_and_bool():
    # ``true`` is not a verdict, and 1.5 / 0.5 are not 1 / 0. Each stays
    # ungraded (None) instead of becoming a pass or a fail.
    assert _parse_verdict("true") == (None, "")
    assert _parse_verdict("false") == (None, "")
    assert _parse_verdict('{"score": 1.5}') == (None, "")
    assert _parse_verdict('{"score": 0.5}') == (None, "")
    assert _parse_verdict('{"score": 2}') == (None, "")
    assert _parse_verdict('{"score": true}') == (None, "")
    assert _parse_verdict('verdict: {"score": 1.5}') == (None, "")


def test_parse_verdict_still_reads_chatter_and_truncation():
    # A well-formed object inside chatter, or a reply cut off right after
    # the score, still grades. These are the cases the fallback exists for.
    assert _parse_verdict('Sure. {"reason": "fine", "score": 1}') == (1, "fine")
    assert _parse_verdict('{"reason": "no lookup", "score": 0') == (0, "no lookup")
    assert _parse_verdict("1") == (1, "")
    assert _parse_verdict("0") == (0, "")


@pytest.mark.parametrize(
    "text",
    [
        '{"score":1.5}',
        '{"score":"1"}',
        '{"score":0,"score":1}',
        '{"score":1,"reward":0}',
        "true",
        '{"score":true}',
        '{"score":NaN}',
        '{"score":1.',
        '{"reason": "cut off", "score": 1.5',
    ],
)
def test_invalid_verdict_never_salvages_a_pass(text):
    # A complete object that breaks the contract (string score, duplicate
    # key, score disagreeing with reward, non-binary number) is the
    # judge's answer and it is not a verdict. Nothing around it is mined
    # for a digit. Issue #64.
    assert _parse_verdict(text)[0] is None


def test_last_complete_object_is_the_verdict():
    echoed = 'Format: {"score": 0 or 1, "reason": "..."}\n{"reason": "did it", "score": 1}'
    assert _parse_verdict(echoed) == (1, "did it")


def test_apply_grade_llm_does_not_rejudge_length_cap_rows(monkeypatch):
    from zeroproof.simulations.score import grade_llm as g

    seen: list[str] = []
    monkeypatch.setattr(g, "warm_judge", lambda *a, **k: {"ok": True, "seconds": 0.0})
    monkeypatch.setattr(g, "require_judge_key", lambda *a, **k: "vllm:m@https://x/v1")

    def fake_one(row, **kw):
        seen.append(row["prompt"])
        return {"reward": 1, "reason": "ok"}

    monkeypatch.setattr(g, "grade_one", fake_one)
    rows = [
        {"prompt": "a", "final_text": "done."},
        {
            "prompt": "b",
            "final_text": "cut",
            "judge_name": "length_cap",
            "judge_status": "missing_reward",
            "reward": None,
        },
    ]
    report = g.apply_grade_llm(rows, api_key="k")
    assert seen == ["a"]
    assert report["skipped_truncated"] == 1 and report["graded"] == 1
    assert rows[1]["judge_name"] == "length_cap" and rows[1]["reward"] is None
