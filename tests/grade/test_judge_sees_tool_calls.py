"""The judge is given counted tool calls, not left to infer them.

#346: asked to read a JSON blob of interleaved text and tool steps, the judge
credited an action the agent ANNOUNCED as one it performed. Its own reason
asserted "escalated to a human" on rows where escalate_to_human was never
called: 18 of 18 rows the rule failed in one regime, kappa 0.04 over 222
rollouts. The system prompt already forbade it, and rewriting the criterion
to spell the rule out did not move the leak rate, so the instruction was not
the missing piece; the inference was. Every test here fails on a tree whose
payload has no ``tool_calls`` ledger.
"""

from __future__ import annotations

import json

from whileai.simulations import defaults
from whileai.simulations.score.grade_llm import _render_payload
from whileai.simulations.score.grounding import tool_calls_of
from whileai.simulations.score.hygiene import performed_tool_calls, tool_call_counts, tool_calls
from whileai.simulations.score.rubric import RUBRIC_JUDGE_SYSTEM

TOOLS = [
    {
        "type": "function",
        "function": {"name": "lookup_invoice", "description": "d", "parameters": {}},
    },
    {
        "type": "function",
        "function": {"name": "issue_credit", "description": "d", "parameters": {}},
    },
    {
        "type": "function",
        "function": {"name": "escalate_to_human", "description": "d", "parameters": {}},
    },
]
DECLARED = ["lookup_invoice", "issue_credit", "escalate_to_human"]


def _announced_but_not_called() -> dict:
    return {
        "prompt": "invoice 4471?",
        "final_text": "Since the invoice is more than $200, I will escalate this to a human.",
        "steps": [
            {"tool": "lookup_invoice", "arguments": {"id": "4471"}, "result": {"amount": 654.30}},
            {"tool": "lookup_invoice", "arguments": {"id": "4471"}, "result": {"amount": 654.30}},
            {"text": "I will escalate this to a human billing agent."},
        ],
    }


def test_an_uncalled_tool_is_reported_as_zero_not_omitted():
    """Absence from a step list is easy to miss; a 0 is not."""
    blob = json.loads(_render_payload(_announced_but_not_called(), tools=TOOLS))
    assert blob["tool_calls"] == {"escalate_to_human": 0, "issue_credit": 0, "lookup_invoice": 2}


def test_the_ledger_precedes_steps_so_a_long_payload_keeps_it():
    """An oversized payload loses its tail; the counts must survive that."""
    blob = json.loads(_render_payload(_announced_but_not_called(), tools=TOOLS))
    keys = list(blob)
    assert keys.index("tool_calls") < keys.index("steps")


def test_the_ledger_survives_structure_reduction_on_a_long_trajectory():
    """The regime the issue did not hit but the code has: past the payload
    cap the steps are shrunk and finally dropped, and without the ledger the
    judge keeps the declared tool names and the agent's claim while losing
    the evidence against them."""
    row = _announced_but_not_called()
    row["steps"] = [
        {"tool": "lookup_invoice", "arguments": {"id": str(i)}, "result": {"note": "x" * 900}}
        for i in range(40)
    ] + [{"text": "I will escalate this to a human."}]
    blob = json.loads(_render_payload(row, tools=TOOLS, payload_chars=3000))
    assert len(blob["steps"]) < 41 and any("skipped_steps" in s for s in blob["steps"])
    assert blob["tool_calls"] == {"escalate_to_human": 0, "issue_credit": 0, "lookup_invoice": 40}


def test_the_ledger_survives_the_reply_only_fallback():
    """When nothing else fits the judge sees the reply alone. It must still
    see that the escalation never ran."""
    row = _announced_but_not_called()
    row["prompt"] = "p" * 5000
    row["final_text"] = "f" * 5000
    blob = json.loads(_render_payload(row, tools=TOOLS, payload_chars=400))
    assert "steps" not in blob and blob["payload_reduced"] is True
    assert blob["tool_calls"]["escalate_to_human"] == 0
    assert blob["tool_calls"]["lookup_invoice"] == 2


def test_a_tool_called_but_never_declared_still_counts():
    row = {"prompt": "p", "final_text": "f", "steps": [{"tool": "undeclared_tool", "result": {}}]}
    blob = json.loads(_render_payload(row, tools=TOOLS))
    assert blob["tool_calls"]["undeclared_tool"] == 1
    assert blob["tool_calls"]["escalate_to_human"] == 0


def test_no_steps_means_every_declared_tool_reads_zero():
    blob = json.loads(_render_payload({"prompt": "p", "final_text": "f"}, tools=TOOLS))
    assert blob["tool_calls"] == dict.fromkeys(DECLARED, 0)


def test_the_system_prompt_tells_the_judge_what_zero_means():
    assert "tool_calls" in RUBRIC_JUDGE_SYSTEM
    assert "was NOT called" in RUBRIC_JUDGE_SYSTEM


# ---------------------------------------------------------- one definition


def test_one_definition_of_a_performed_call():
    """The ledger, ``tool_calls`` and ``tool_calls_of`` count the same
    calls, off the keys ``PERFORMED_CALL_SOURCES`` names."""
    row = _announced_but_not_called()
    counts = tool_call_counts(row, DECLARED)
    assert sum(counts.values()) == tool_calls(row) == len(performed_tool_calls(row)) == 2
    assert [c["tool"] for c in tool_calls_of(row)] == ["lookup_invoice", "lookup_invoice"]
    assert defaults.PERFORMED_CALL_SOURCES == ("steps", "tool_trace")


def test_a_platform_tool_trace_counts_the_same_way():
    """A platform pull carries ``tool_trace`` (tool/input/output), not
    ``steps``; the same tool at the same count."""
    row = {
        "prompt": "p",
        "final_text": "done",
        "tool_trace": [
            {"tool": "issue_credit", "input": {"amount": 20}, "output": {"status": "ok"}},
            {"tool": "lookup_invoice", "input": {"id": "1"}, "output": {"amount": 20}},
        ],
    }
    blob = json.loads(_render_payload(row, tools=TOOLS))
    assert blob["tool_calls"] == {"escalate_to_human": 0, "issue_credit": 1, "lookup_invoice": 1}
    assert tool_calls(row) == 2


def test_a_call_written_in_text_is_not_a_performed_call():
    """The exact leak: a ``<tool_call>`` block or a sentence in the reply is
    a claim, and the ledger reads 0 for it."""
    row = {
        "prompt": "p",
        "final_text": (
            '<tool_call>{"name": "escalate_to_human", "arguments": {}}</tool_call> '
            "I have escalated this to a human."
        ),
        "steps": [{"text": "Escalating now via escalate_to_human."}],
    }
    assert tool_call_counts(row, DECLARED)["escalate_to_human"] == 0
    assert tool_calls(row) == 0
