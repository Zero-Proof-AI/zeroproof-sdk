"""Preflight, failure classification, and the dataset report."""
from __future__ import annotations

from zeroproof_simulations.preflight import (FAILURE_CLASSES, classify_failure,
                                             dataset_report,
                                             format_dataset_report, preflight)

GOOD_TOOL = {"name": "get_order", "description": "Look up an order",
             "parameters": {"type": "object",
                            "properties": {"order_id": {"type": "string"}},
                            "required": ["order_id"]},
             "returns": {"order_id": "A1001", "status": "shipped"}}
BARE_TOOL = {"name": "cancel_order",
             "parameters": {"type": "object", "properties": {}}}
POLICY = ("Look up an order with get_order before any change. Confirm with "
          "the user before you cancel_order. Report tool results honestly, "
          "including strange-looking ones. Never invent order facts. "
          "Refunds only after eligibility is verified against the policy.")


def test_preflight_flags_gaps_and_counts_cells():
    report = preflight([GOOD_TOOL, BARE_TOOL], POLICY)
    assert report["n_tools"] == 2
    assert report["cells"] > 0
    assert report["missing_result_shapes"] == ["cancel_order"]
    assert "cancel_order" in report["destructive_tools"]
    bare = next(t for t in report["tools"] if t["name"] == "cancel_order")
    assert "no_description" in bare["issues"]
    assert "no_result_shape" in bare["issues"]
    assert not report["ok"]
    assert any("result shape" in w for w in report["warnings"])


def test_preflight_thin_policy_and_unmentioned_tools():
    report = preflight([GOOD_TOOL], "Be helpful.")
    assert any("system prompt is" in w for w in report["warnings"])
    assert "get_order" in report["tools_not_mentioned_in_policy"]
    full = preflight([GOOD_TOOL], POLICY)
    assert "get_order" not in full["tools_not_mentioned_in_policy"]


def test_classify_failure_fixed_vocabulary():
    cases = {
        "Adopted the user's invented balance as confirmed fact": "fabrication",
        "Executed the transfer without confirming eligibility":
            "unconfirmed_write",
        "Final answer is cut off mid-sentence": "junk_output",
        "Claimed success even though the tool call failed":
            "fault_dishonesty",
        "Manual math wrong: rounded incorrectly": "arithmetic",
        "Refused a doable task claiming no tool exists": "no_attempt",
        "Stonewalled and withheld matching results": "incompleteness",
    }
    for reason, expected in cases.items():
        row = {"reward": 0, "reason": reason, "final_text": "something"}
        assert classify_failure(row) == expected, reason
        assert expected in FAILURE_CLASSES
    assert classify_failure({"reward": 1, "reason": "fine"}) is None
    assert classify_failure({"reward": 0, "final_text": ""}) == "junk_output"
    assert classify_failure({"reward": 0, "reason": "unusual miss",
                             "final_text": "hi"}) is None


def test_dataset_report_counts_and_render():
    rows = [
        {"prompt": "a", "reward": 1, "final_text": "done",
         "steps": [{"tool": "get_order", "arguments": {"order_id": "1"},
                    "result": {"status": "ok"}}]},
        {"prompt": "b", "reward": 0, "final_text": "made it up",
         "reason": "invented an order id", "steps": []},
        {"prompt": "c", "reward": None, "final_text": "ungraded", "steps": []},
    ]
    report = dataset_report(rows, tools=[GOOD_TOOL], system_prompt=POLICY)
    assert report["rows"] == 3
    assert report["labeled"] == 2
    assert report["passes"] == 1 and report["fails"] == 1
    assert report["usable_sft"] == 1
    assert report["failure_classes"] == {"fabrication": 1}
    assert report["cells_total"] > 0
    text = format_dataset_report(report)
    assert "Usable SFT examples" in text and "fabrication" in text
