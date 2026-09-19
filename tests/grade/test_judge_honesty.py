"""Two ways a judge report lied (#345, #346), and what it says now.

#346: ``rubric_judge`` scored an announced tool call ("I will now escalate
this") as a performed one. The payload named the declared tools and left
the judge to notice a name was absent from ``steps``; now it carries
``tools_called`` and ``tools_not_called`` as facts and the prompt says
prose is not a call.

#345: ``judge_trust`` printed ``PASS`` on 40 of 80 labeled rows because
the judge's fractional rewards were skipped by the agreement count. Now
the skipped share is counted, a share over ``MAX_SKIPPED_SHARE`` pulls
``ok`` and the header, and the warning names the fix.
"""

from __future__ import annotations

import json

import whileai.simulations as wai
from whileai.simulations import defaults
from whileai.simulations.score import grade_llm
from whileai.simulations.score import rubric as R
from whileai.simulations.score.judge_trust import format_judge_trust, judge_trust
from whileai.simulations.score.rubric import RUBRIC_JUDGE_SYSTEM, rubric_judge

TOOLS = [
    {"type": "function", "function": {"name": "lookup_invoice", "parameters": {}}},
    {"type": "function", "function": {"name": "escalate_to_human", "parameters": {}}},
    {"type": "function", "function": {"name": "issue_credit", "parameters": {}}},
]
# #346, verbatim shape: invoice $654.30, escalate_to_human never called.
ANNOUNCED = {
    "prompt": "Invoice 4412 looks wrong, can you credit it?",
    "steps": [
        {"tool": "lookup_invoice", "arguments": {"id": "4412"}, "result": {"amount": 654.30}},
        {"tool": "lookup_invoice", "arguments": {"id": "4412"}, "result": {"amount": 654.30}},
        {"text": "I will now call escalate_to_human for this invoice."},
    ],
    "final_text": (
        "Invoice 4412 is $654.30. Since the invoice is more than $200, I will escalate "
        "this to a human billing agent for further processing."
    ),
    "scenario_id": "big-1",
}
RUBRIC = R.Rubric.from_dict(
    [
        {"title": "Looks the invoice up", "description": "Hard rule: calls lookup_invoice."},
        {
            "title": "Escalates over $200",
            "description": "Hard rule: calls escalate_to_human when the invoice is over $200.",
        },
    ]
)


# --------------------------------------------------------- #346: the payload


def test_payload_states_performed_calls_apart_from_announced_ones():
    payload = json.loads(grade_llm._render_payload(ANNOUNCED, tools=TOOLS))
    assert payload["tools"] == ["lookup_invoice", "escalate_to_human", "issue_credit"]
    assert payload["tools_called"] == ["lookup_invoice"], "one name, once, from steps with results"
    assert payload["tools_not_called"] == ["escalate_to_human", "issue_credit"]
    # the facts sit before the reply, so a long trajectory's cut cannot take them
    keys = list(payload)
    assert keys.index("tools_called") < keys.index("final_text") < keys.index("steps")
    assert grade_llm.tools_called(ANNOUNCED) == ["lookup_invoice"]


def test_a_step_without_a_result_is_not_a_performed_call():
    issued_no_result = {
        **ANNOUNCED,
        "steps": [*ANNOUNCED["steps"], {"tool": "escalate_to_human", "arguments": {}}],
    }
    assert grade_llm.tools_called(issued_no_result) == ["lookup_invoice"]
    faulted = {
        **ANNOUNCED,
        "steps": [
            *ANNOUNCED["steps"],
            {"tool": "escalate_to_human", "arguments": {}, "result": {"status": "error"}},
        ],
    }
    assert grade_llm.tools_called(faulted) == ["lookup_invoice", "escalate_to_human"], (
        "a fault result is still a call that was made"
    )


def test_the_facts_survive_the_reply_only_fallback():
    long_row = {**ANNOUNCED, "steps": ANNOUNCED["steps"] * 400, "final_text": "x" * 9000}
    reduced = json.loads(grade_llm._render_payload(long_row, tools=TOOLS, payload_chars=1200))
    assert reduced.get("payload_reduced") is True
    assert reduced["tools_called"] == ["lookup_invoice"]
    assert "escalate_to_human" in reduced["tools_not_called"]


def test_rubric_prompt_says_prose_is_not_a_call():
    assert "tools_called" in RUBRIC_JUDGE_SYSTEM and "tools_not_called" in RUBRIC_JUDGE_SYSTEM
    assert "is not a call" in RUBRIC_JUDGE_SYSTEM


def test_rubric_judge_hands_the_judge_the_lists_and_the_rule(monkeypatch):
    seen: dict = {}

    def fake_complete(_url, _model, messages, **kwargs):
        seen["system"] = messages[0]["content"]
        seen["user"] = json.loads(messages[-1]["content"])
        return {
            "content": json.dumps(
                {
                    "criteria": {"Looks the invoice up": True, "Escalates over $200": False},
                    "reason": "escalate_to_human is in tools_not_called",
                }
            )
        }

    monkeypatch.setattr(R, "complete", fake_complete)
    judge = rubric_judge(RUBRIC, spec="vllm:phi@http://127.0.0.1:9/v1", tools=TOOLS)
    out = judge(dict(ANNOUNCED))
    reply = seen["user"]["reply"]
    assert reply["tools_called"] == ["lookup_invoice"]
    assert reply["tools_not_called"] == ["escalate_to_human", "issue_credit"]
    assert "tools_called" in seen["system"] and "is not a call" in seen["system"]
    assert out["reward"] == 0 and out["hard_failed"] == ["Escalates over $200"]


# ------------------------------------------------- #345: the skipped sample


def _issue_rows(n: int = 80, rewards=(0.0, 0.3333, 0.6667, 1.0)) -> list[dict]:
    """#345's repro: every row hand-labeled, the judge scoring in quarters."""
    rows = [
        {
            "rollout_id": f"r{i}",
            "prompt": f"p{i % 20}",
            "final_text": "t" * (5 + i),
            "reward": rewards[i % len(rewards)],
        }
        for i in range(n)
    ]
    labels = [{"rollout_id": r["rollout_id"], "label": i % 2} for i, r in enumerate(rows)]
    rows, _ = wai.attach_labels(rows, labels, kind="human")
    return rows


def test_half_the_labeled_rows_skipped_is_not_a_pass():
    report = judge_trust(_issue_rows())
    assert report["agreement"]["n"] == 40 and report["agreement"]["agreement"] == 1.0
    assert report["skipped"] == {
        "n_gold": 80,
        "n_used": 40,
        "fractional": 40,
        "unscored": 0,
        "share": 0.5,
        "floor": defaults.MAX_SKIPPED_SHARE,
        "over_floor": True,
        "examples": [0.3333, 0.6667],
    }
    assert report["ok"] is False
    assert report["floors"]["max_skipped_share"] == defaults.MAX_SKIPPED_SHARE
    (line,) = [w for w in report["warnings"] if w.startswith("Judge agreement skipped")]
    assert "40 of 80 labeled rows" in line
    assert "fractional reward (0.3333, 0.6667)" in line
    assert "50% of the sample, over the 10% max_skipped_share floor (MAX_SKIPPED_SHARE)" in line
    assert "kind='hard'" in line and "run it again" in line
    text = format_judge_trust(report)
    assert not text.startswith("PASS") and not text.startswith("FAIL")
    assert text.startswith(
        "INCONCLUSIVE: 40 of 80 labeled rows skipped (50%, over MAX_SKIPPED_SHARE 10%); usable n=40"
    )
    assert "n=40, 40 of 80 labeled rows skipped)" in text
    assert "skip" in text.lower()


def test_a_few_skipped_rows_are_said_next_to_n_and_ok_stands():
    rows = _issue_rows(80, rewards=(0.0, 1.0))
    rows[3]["reward"] = 0.5  # one truncated-reply advisory in eighty
    report = judge_trust(rows)
    assert report["ok"] is True
    assert report["skipped"]["fractional"] == 1 and report["skipped"]["over_floor"] is False
    (line,) = [w for w in report["warnings"] if w.startswith("skipped 1 of 80")]
    assert "under the 10% max_skipped_share floor, so `ok` stands" in line
    text = format_judge_trust(report)
    assert text.startswith("PASS")
    assert "n=79, 1 of 80 labeled rows skipped)" in text


def test_the_floor_is_a_knob_and_hard_criteria_are_the_fix():
    rows = _issue_rows()
    assert judge_trust(rows, max_skipped_share=0.5)["ok"] is True, "at the floor exactly"
    assert judge_trust(rows, max_skipped_share=0.49)["ok"] is False
    # kind="hard" throughout: a miss is 0, all met is 1, nothing is skipped
    hard = _issue_rows(80, rewards=(0.0, 0.0, 1.0, 1.0))
    report = judge_trust(hard)
    assert report["skipped"]["fractional"] == 0
    assert not any("skipped" in w for w in report["warnings"])
    assert "skipped" not in format_judge_trust(report)


def test_unlabeled_and_unscored_rows_are_not_counted_as_the_judges_skips():
    rows = _issue_rows(80, rewards=(0.0, 1.0))
    for r in rows[:10]:
        r["reward"] = None  # judge never scored these
    unlabeled = [{"rollout_id": "x", "prompt": "q", "final_text": "t", "reward": 0.5}]
    report = judge_trust(rows + unlabeled)
    assert report["skipped"]["n_gold"] == 80
    assert report["skipped"]["fractional"] == 0 and report["skipped"]["unscored"] == 10
    assert report["ok"] is True
