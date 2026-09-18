"""The package's judges on a ``typesafe:`` spec: typed questions out, a
verdict with its probability back. Offline: requests.post is faked."""

from __future__ import annotations

import json

import pytest

from whileai.simulations.defaults import DECISION_UNSURE_BAND, PASS_THRESHOLD
from whileai.simulations.generate import typesafe_backend as tb
from whileai.simulations.score import decision_judge as dj
from whileai.simulations.score import grade_llm, llm_judge, pairwise
from whileai.simulations.score.preflight import FAILURE_CLASSES
from whileai.simulations.score.rubric import (
    RUBRIC_JUDGE_SYSTEM,
    Criterion,
    Rubric,
    rubric_judge,
)

SPEC = "typesafe:jev-latest"
ROW = {
    "rollout_id": "r1",
    "prompt": "refund order A-104",
    "steps": [
        {
            "tool": "lookup_order",
            "arguments": {"id": "A-104"},
            "result": {"status": "ok", "total": 40},
        }
    ],
    "final_text": "Refunded 40.",
}


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    monkeypatch.delenv("WHILEAI_TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    monkeypatch.setattr(tb.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        tb.requests,
        "get",
        lambda *a, **k: _Response({"models": [{"name": "jev-latest"}]}),
    )


class _Response:
    def __init__(self, payload, status=200, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload

    @property
    def text(self):
        return json.dumps(self._payload)


def _answers(**answers):
    return _Response(
        {
            "model": "jev-latest",
            "answers": answers,
            "usage": {"input_tokens": 9, "output_tokens": 0},
        }
    )


def _serve(monkeypatch, *responses):
    sent = []
    queue = list(responses)

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.append({"url": url, "body": json, "timeout": timeout})
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(tb.requests, "post", fake_post)
    return sent


def _verdict(p, failure_class=None, share=0.8):
    answers = {"verdict": {"type": "noul", "noul": p}}
    if failure_class is not None:
        rest = round((1 - share) / 2, 3)
        answers["failure_class"] = {
            "type": "choice",
            "choice": failure_class,
            "confidence": share,
            "probabilities": {failure_class: share, "none": rest, "incompleteness": rest},
        }
    return _answers(**answers)


# --- the questions ----------------------------------------------------------


def test_the_reply_contract_comes_off_every_judge_prompt():
    for prompt in (
        grade_llm.JUDGE_SYSTEM,
        grade_llm.AUDIT_SYSTEM,
        grade_llm.rubric_prompt("Refund only after the order is confirmed."),
        pairwise.PAIRWISE_SYSTEM,
        RUBRIC_JUDGE_SYSTEM,
        llm_judge.JUDGE_SYSTEM,
    ):
        stripped = dj.strip_output_contract(prompt)
        # the prose may still say "a JSON record"; the reply contract is gone
        assert "Reply with" not in stripped and "Answer with" not in stripped, stripped
        assert '{"' not in stripped, stripped
        assert '"score"' not in stripped and '"winner"' not in stripped
        assert '"criteria"' not in stripped and '"reason"' not in stripped
        assert "Write the reason first" not in stripped and "Write reason first" not in stripped
    assert "Grade the agent, not the sandbox" in dj.strip_output_contract(grade_llm.JUDGE_SYSTEM)
    assert "(a)" in dj.strip_output_contract(grade_llm.JUDGE_SYSTEM)
    assert "You are given the agent's tools" in dj.strip_output_contract(llm_judge.JUDGE_SYSTEM)
    assert "Refund only after" in dj.strip_output_contract(
        grade_llm.rubric_prompt("Refund only after the order is confirmed.")
    )


def test_the_failure_class_choice_is_the_preflight_vocabulary_and_every_class_is_described():
    questions = dj.verdict_questions()
    assert list(questions) == ["verdict", "failure_class"]
    assert questions["verdict"]["type"] == "noul"
    assert set(questions["verdict"]["criteria"]) == {"true", "false"}
    criteria = questions["failure_class"]["criteria"]
    assert tuple(criteria) == ("none", *FAILURE_CLASSES)
    undescribed = [name for name, text in criteria.items() if not text]
    assert not undescribed, (
        f"describe these in decision_judge.FAILURE_CLASS_CRITERIA: {undescribed}"
    )
    assert list(dj.verdict_questions(classify=False)) == ["verdict"]


def test_the_version_suffix_is_stable_and_names_the_questions():
    assert dj.version_suffix() == dj.version_suffix()
    assert dj.version_suffix().startswith("|decision:")


def test_split_payload_keeps_the_fault_lead_and_decodes_the_record():
    note, record = dj.split_payload('Injected fault: timeout.\n{"a": 1}')
    assert note == "Injected fault: timeout." and record == {"a": 1}
    # a record cut mid-JSON by payload_chars stays a string the API accepts
    note, record = dj.split_payload('{"a": 1, "b": [1, 2')
    assert note == "" and record == '{"a": 1, "b": [1, 2'


# --- grade ------------------------------------------------------------------


def test_grade_one_fails_a_row_from_the_verdict_and_names_the_class(monkeypatch):
    sent = _serve(monkeypatch, _verdict(0.07, "fabrication", share=0.81))
    out = grade_llm.grade_one(ROW, backend_spec=SPEC, policy="Confirm before refunding.")
    assert out["reward"] == 0
    assert out["failure_class"] == "fabrication"
    assert out["confidence"] == pytest.approx(0.93)
    assert out["unsure"] is False
    assert out["reason"] == "jev: p(did what it should)=0.07, fabrication (0.81)"
    assert out["probabilities"]["verdict"] == 0.07
    body = sent[0]["body"]
    assert sent[0]["url"].endswith("/v1/systemone")
    assert body["model"] == "jev-latest"
    assert list(body["questions"]) == ["verdict", "failure_class"]
    state = body["state"]
    assert "Reply with" not in state["instructions"]
    assert "Grade the agent, not the sandbox" in state["instructions"]
    record = json.dumps(state["record"])
    assert "refund order A-104" in record
    assert "Confirm before refunding." in record
    assert "lookup_order" in record


def test_grade_one_passes_a_near_even_row_and_marks_it_unsure(monkeypatch):
    p = PASS_THRESHOLD + DECISION_UNSURE_BAND / 2
    _serve(monkeypatch, _verdict(p, "fabrication"))
    out = grade_llm.grade_one(ROW, backend_spec=SPEC)
    assert out["reward"] == 1
    assert out["unsure"] is True
    # a passing row carries no failure class, whatever the choice said
    assert out["failure_class"] is None
    assert out["confidence"] == pytest.approx(p)


def test_grade_one_with_no_verdict_answer_stays_ungraded(monkeypatch):
    _serve(monkeypatch, _answers(other={"type": "noul", "noul": 0.9}))
    assert grade_llm.grade_one(ROW, backend_spec=SPEC) == {
        "reward": None,
        "reason": "",
        "confidence": None,
        "unsure": False,
        "failure_class": None,
        "probabilities": {},
    }


def test_grade_one_on_a_dead_endpoint_stays_ungraded(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr(tb.requests, "post", boom)
    assert grade_llm.grade_one(ROW, backend_spec=SPEC) == {"reward": None, "reason": ""}


def test_grade_one_sends_the_fault_lead_as_a_note(monkeypatch):
    sent = _serve(monkeypatch, _verdict(0.9))
    row = {**ROW, "faults": {"lookup_order": {"mode": "timeout", "rate": 1.0}}}
    grade_llm.grade_one(row, backend_spec=SPEC)
    state = sent[0]["body"]["state"]
    assert state["note"]
    assert isinstance(state["record"], dict)


def test_a_custom_prompt_is_the_instructions_and_the_privileged_block_rides_along(monkeypatch):
    sent = _serve(monkeypatch, _verdict(0.9))
    row = {**ROW, "privileged": {"reference": "Refund 40 after confirming."}}
    grade_llm.grade_one(
        row, backend_spec=SPEC, prompt="Grade against the reference.", use_privileged=True
    )
    state = sent[0]["body"]["state"]
    assert state["instructions"].startswith("Grade against the reference.")
    assert "judge_only" in state["instructions"]
    assert state["record"]["judge_only"]["reference"] == "Refund 40 after confirming."


def test_apply_grade_llm_writes_confidence_and_counts_unsure_rows(monkeypatch):
    unsure_p = PASS_THRESHOLD + DECISION_UNSURE_BAND / 2
    _serve(monkeypatch, _verdict(0.07, "fabrication", share=0.81), _verdict(unsure_p))
    rows = [dict(ROW), {**ROW, "rollout_id": "r2", "final_text": "Refunded 40, confirmed."}]
    report = grade_llm.apply_grade_llm(rows, backend_spec=SPEC, concurrency=1, trust="off")
    assert report["status"] == "judged"
    assert report["graded"] == 2 and report["n0"] == 1 and report["n1"] == 1
    assert report["unsure"] == 1
    assert report["backend"] == SPEC
    assert report["warmup"] == {"ok": True, "seconds": 0.0, "models": ["jev-latest"]}
    assert report["judge_version"].startswith("jev-latest@")
    # the questions are part of the version: same prompt, different hash
    assert report["judge_version"] == grade_llm.judge_version(
        SPEC, grade_llm.JUDGE_SYSTEM + dj.version_suffix()
    )
    assert report["judge_version"] != grade_llm.judge_version(SPEC, grade_llm.JUDGE_SYSTEM)
    assert report["self_judged"] is False
    first, second = rows
    assert first["reward"] == 0
    assert first["failure_class"] == "fabrication"
    assert first["judge_meta"]["confidence"] == pytest.approx(0.93)
    assert first["judge_meta"]["decision"] is True
    assert "unsure" not in first["judge_meta"]
    assert second["reward"] == 1
    assert second["judge_meta"]["unsure"] is True
    assert second["judge_meta"]["confidence"] == pytest.approx(unsure_p)
    assert "failure_class" not in second


def test_apply_grade_llm_empty_report_has_the_unsure_key():
    assert grade_llm.apply_grade_llm([], backend_spec=SPEC)["unsure"] == 0


def test_warm_judge_is_the_key_check(monkeypatch):
    assert grade_llm.warm_judge(SPEC) == {"ok": True, "seconds": 0.0, "models": ["jev-latest"]}
    monkeypatch.setattr(tb.requests, "get", lambda *a, **k: _Response({"error": "no"}, status=401))
    out = grade_llm.warm_judge(SPEC)
    assert out["ok"] is False and "TYPESAFE_API_KEY" in out["error"]


def test_require_judge_key_names_the_typesafe_fix(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        grade_llm.require_judge_key(spec=SPEC)
    assert grade_llm.judge_spec(spec=SPEC) == SPEC


# --- audit ------------------------------------------------------------------


def test_audit_one_asks_blind_and_reports_disagreement(monkeypatch):
    sent = _serve(monkeypatch, _verdict(0.2))
    out = grade_llm.audit_one({**ROW, "reward": 1, "reason": "fine"}, backend_spec=SPEC)
    assert out["audit_reward"] == 0
    assert out["agreed"] is False
    assert out["audit_reason"] == "jev: p(did what it should)=0.20"
    assert out["existing"] == 1 and out["judge_reason"] == "fine"
    body = sent[0]["body"]
    assert list(body["questions"]) == ["verdict"]
    assert "existing_label" not in json.dumps(body)


# --- pairwise ---------------------------------------------------------------


def test_pairwise_judge_picks_the_choice_and_keeps_the_distribution(monkeypatch):
    sent = _serve(
        monkeypatch,
        _answers(
            winner={
                "type": "choice",
                "choice": "B",
                "confidence": 0.85,
                "probabilities": {"A": 0.1, "B": 0.85, "tie": 0.05},
            }
        ),
    )
    judge = pairwise.pairwise_judge(SPEC)
    assert judge.__name__.startswith("jev-latest@")
    b = {**ROW, "final_text": "Refunded 40 after you confirmed."}
    out = judge(ROW, b)
    assert out == {"winner": "B", "reason": "jev: A=0.10, B=0.85, tie=0.05"}
    body = sent[0]["body"]
    assert list(body["questions"]["winner"]["criteria"]) == ["A", "B", "tie"]
    assert body["state"]["request"] == "refund order A-104"
    assert body["state"]["record"]["A"]["final_text"] == "Refunded 40."
    assert body["state"]["record"]["B"]["final_text"] == "Refunded 40 after you confirmed."
    assert "Answer with" not in body["state"]["instructions"]


def test_pairwise_judge_with_an_unknown_label_has_no_winner(monkeypatch):
    _serve(
        monkeypatch, _answers(winner={"type": "choice", "choice": "C", "probabilities": {"C": 1.0}})
    )
    assert pairwise.pairwise_judge(SPEC)(ROW, ROW)["winner"] is None


# --- rubric -----------------------------------------------------------------


def test_rubric_judge_asks_one_question_per_item_and_scores_with_the_rubric(monkeypatch):
    sent = _serve(
        monkeypatch,
        _answers(**{"1": {"type": "noul", "noul": 0.9}, "2": {"type": "noul", "noul": 0.2}}),
    )
    rubric = Rubric(
        criteria=(
            Criterion(title="Confirms before refunding", kind="hard"),
            Criterion(
                title="Invents an amount",
                description="States a total the tools did not return.",
                kind="pitfall",
            ),
        )
    )
    out = rubric_judge(rubric, spec=SPEC)(ROW)
    assert out["reward"] == 1.0
    assert out["reason"] == "jev: 1=0.90, 2=0.20"
    assert out["markers"]["rubric:confirms_before_refunding"] == 1.0
    assert out["markers"]["rubric:invents_an_amount"] == 1.0
    questions = sent[0]["body"]["questions"]
    assert list(questions) == ["1", "2"]
    assert "hard rule" in questions["1"]["instructions"]
    assert "Confirms before refunding" in questions["1"]["instructions"]
    assert (
        "pitfall" in questions["2"]["instructions"] and "exhibits" in questions["2"]["instructions"]
    )
    assert "States a total" in questions["2"]["instructions"]
    assert sent[0]["body"]["state"]["rubric"] == rubric.checklist()
    assert sent[0]["body"]["state"]["record"]["final_text"] == "Refunded 40."


def test_rubric_judge_with_no_answered_item_stays_ungraded(monkeypatch):
    _serve(monkeypatch, _answers(other={"type": "noul", "noul": 0.9}))
    rubric = Rubric(criteria=(Criterion(title="Confirms first", kind="hard"),))
    out = rubric_judge(rubric, spec=SPEC)(ROW)
    assert out["reward"] is None and "no criteria" in out["reason"]


# --- advisory ---------------------------------------------------------------


def test_advisory_judge_scales_the_expected_level(monkeypatch):
    sent = _serve(
        monkeypatch,
        _answers(
            score={
                "type": "score",
                "score": 1.6,
                "confidence": 0.7,
                "legend": {"0": "a", "1": "b", "2": "c"},
                "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
            }
        ),
    )
    out = llm_judge.judge_one(ROW, backend_spec=SPEC)
    assert out == {
        "llm_reward": 0.8,
        "llm_reason": "jev: E[level]=1.60 of 2 (0=0.10, 1=0.20, 2=0.70)",
    }
    question = sent[0]["body"]["questions"]["score"]
    assert question["type"] == "score"
    assert question["criteria"] == list(dj.ADVISORY_LEVELS)
    assert "Reply with only JSON" not in sent[0]["body"]["state"]["instructions"]


def test_advisory_judge_key_resolves_from_the_typesafe_env():
    assert llm_judge.resolve_judge_key(None, SPEC) == "ts-test"
