"""Argument grounding: an invented tool argument is caught on any row shape."""

from __future__ import annotations

from zeroproof.simulations.score.delta import delta_report
from zeroproof.simulations.score.grounding import (
    argument_grounding,
    grounding_report,
    mark_grounding,
    tool_calls_of,
    ungrounded_arguments,
)

PROMPT = "Hi, can you check ORD-4017 for me? The jacket arrived torn."


def test_steps_grounded_in_prompt_and_earlier_results():
    row = {
        "prompt": PROMPT,
        "steps": [
            {
                "tool": "lookup_order",
                "arguments": {"order_id": "ORD-4017"},
                "result": {"customer_id": "C-88", "total": 40},
            },
            {
                "tool": "create_refund",
                "arguments": {"order_id": "ORD-4017", "customer_id": "C-88", "amount": 40},
            },
        ],
    }
    assert ungrounded_arguments(row) == []
    assert argument_grounding(row) == 1.0
    assert [c["tool"] for c in tool_calls_of(row)] == ["lookup_order", "create_refund"]


def test_invented_id_is_named():
    row = {
        "prompt": "I need a refund on my last order, the mug is chipped.",
        "steps": [{"tool": "lookup_order", "arguments": {"order_id": "ORD-1234"}}],
    }
    assert ungrounded_arguments(row) == [
        {"tool": "lookup_order", "key": "order_id", "value": "ORD-1234"}
    ]
    assert argument_grounding(row) == 0.0


def test_separators_and_case_do_not_count_as_invented():
    row = {
        "prompt": "order ord 4017 please",
        "steps": [{"tool": "lookup_order", "arguments": {"order_id": "ORD-4017"}}],
    }
    assert argument_grounding(row) == 1.0


def test_messages_tool_calls_and_tool_call_blocks_are_read():
    wire = {
        "prompt": PROMPT,
        "messages": [
            {"role": "user", "content": PROMPT},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "lookup_order", "arguments": '{"order_id": "ORD-9999"}'}}
                ],
            },
        ],
    }
    assert [b["value"] for b in ungrounded_arguments(wire)] == ["ORD-9999"]
    block = {
        "prompt": PROMPT,
        "final_text": '<tool_call>\n{"name": "lookup_order", "arguments": {"order_id": "ORD-4017"}}\n</tool_call>',
    }
    assert argument_grounding(block) == 1.0
    block["final_text"] = block["final_text"].replace("4017", "4018")
    assert argument_grounding(block) == 0.0


def test_no_calls_free_text_keys_allow_list_and_short_values():
    assert argument_grounding({"prompt": "hello", "final_text": "Which order?"}) == 1.0
    row = {
        "prompt": "refund ORD-4017",
        "steps": [
            {
                "tool": "create_refund",
                "arguments": {
                    "order_id": "ORD-4017",
                    "reason": "Customer reported damage",
                    "currency": "USD",
                    "ok": "y",
                },
            }
        ],
    }
    assert {b["key"] for b in ungrounded_arguments(row)} == {"reason", "currency"}
    assert ungrounded_arguments(row, ignore_keys=["reason"], allow=["USD"]) == []


def test_mark_and_report_and_delta_guard():
    good = {
        "prompt": PROMPT,
        "task_id": "t",
        "reward": 1,
        "steps": [{"tool": "lookup_order", "arguments": {"order_id": "ORD-4017"}}],
    }
    bad = {
        "prompt": "refund my last order",
        "task_id": "u",
        "reward": 1,
        "steps": [{"tool": "lookup_order", "arguments": {"order_id": "ORD-5555"}}],
    }
    rows = mark_grounding(
        [good, bad, {"prompt": "hi", "task_id": "v", "reward": 1, "final_text": "Hello."}]
    )
    assert [r["markers"]["argument_grounding"] for r in rows] == [1.0, 0.0, 1.0]
    rep = grounding_report([good, bad, bad])
    assert (
        rep["rows"] == 3 and rep["rows_with_calls"] == 3 and rep["grounded_rate"] == round(1 / 3, 4)
    )
    assert rep["invented"][0] == {
        "argument": "lookup_order.order_id",
        "rows": 2,
        "example": "ORD-5555",
    }

    # Before: eight tasks, all grounded. After: the same eight, five invented.
    before = mark_grounding(
        [
            {
                **good,
                "task_id": f"t{i}",
                "prompt": f"check ORD-{4000 + i}",
                "steps": [{"tool": "lookup_order", "arguments": {"order_id": f"ORD-{4000 + i}"}}],
            }
            for i in range(8)
        ]
        * 3
    )
    after = mark_grounding(
        [
            {
                **good,
                "task_id": f"t{i}",
                "prompt": f"check ORD-{4000 + i}",
                "steps": [
                    {
                        "tool": "lookup_order",
                        "arguments": {
                            "order_id": f"ORD-{9000 + i}" if i < 5 else f"ORD-{4000 + i}"
                        },
                    }
                ],
            }
            for i in range(8)
        ]
        * 3
    )
    report = delta_report(
        before, after, target="pass_at_1", must_not_regress=["argument_grounding"], n_boot=200
    )
    assert report["ok"] is False and report["regressions"] == ["marker:argument_grounding"]
