"""The judge is given counted tool calls, not left to infer them.

#346: asked to read a JSON blob of interleaved text and tool steps, the judge
credited an action the agent ANNOUNCED as one it performed. Its own reason
asserted "escalated to a human" on rows where escalate_to_human was never
called -- 18 of 18 rows the rule failed in one regime, kappa 0.04 over 222
rollouts. The system prompt already forbade it, so the instruction was not the
missing piece; the inference was.
"""

import json

from whileai.simulations.score.grade_llm import _render_payload
from whileai.simulations.score.rubric import RUBRIC_JUDGE_SYSTEM

TOOLS = [
    {"type": "function", "function": {"name": "lookup_invoice", "description": "d", "parameters": {}}},
    {"type": "function", "function": {"name": "issue_credit", "description": "d", "parameters": {}}},
    {"type": "function", "function": {"name": "escalate_to_human", "description": "d", "parameters": {}}},
]


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


def test_a_tool_called_but_never_declared_still_counts():
    row = {"prompt": "p", "final_text": "f", "steps": [{"tool": "undeclared_tool", "result": {}}]}
    blob = json.loads(_render_payload(row, tools=TOOLS))
    assert blob["tool_calls"]["undeclared_tool"] == 1
    assert blob["tool_calls"]["escalate_to_human"] == 0


def test_no_steps_means_every_declared_tool_reads_zero():
    blob = json.loads(_render_payload({"prompt": "p", "final_text": "f"}, tools=TOOLS))
    assert set(blob["tool_calls"].values()) == {0}


def test_the_system_prompt_tells_the_judge_what_zero_means():
    assert "tool_calls" in RUBRIC_JUDGE_SYSTEM
    assert "was NOT called" in RUBRIC_JUDGE_SYSTEM
