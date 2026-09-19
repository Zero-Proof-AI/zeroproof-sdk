"""The trace-mining numbers are named defaults with a keyword each."""

from __future__ import annotations

import pytest

from tests.helpers import TOOLS
from whileai.simulations import defaults
from whileai.simulations.ingest.otel import rows_from_otel
from whileai.simulations.ingest.traces import (
    FAULT_TO_AXIS,
    RESULT_BINDING,
    behavior_state,
    dimensions_from_traces,
    leakage_report,
    mine_result_exemplars,
    split_pseudo_production,
)


def _row(prompt, result, reward=None, **extra):
    row = {
        "prompt": prompt,
        "steps": [{"tool": "get_order", "arguments": {"order_id": "1"}, "result": result}],
        "final_text": "done",
        **extra,
    }
    if reward is not None:
        row["reward"] = reward
    return row


def test_exemplar_size_cap_is_a_keyword_with_the_named_default():
    big = {"id": 1, "note": "x" * 120, "items": [{"a": 1}, {"b": 2}]}
    rows = [_row("p", big)]
    assert mine_result_exemplars(rows)["get_order"], "the default cap keeps a 150-char payload"
    assert mine_result_exemplars(rows, max_chars=50) == {}
    assert defaults.TRACE_EXEMPLAR_MAX_CHARS == 500 and defaults.TRACE_EXEMPLARS_PER_TOOL == 3


def test_leakage_examples_are_capped_by_a_keyword_and_the_count_stays_full():
    sources = [{"prompt": "refund order 81"}, {"prompt": "refund order 82"}]
    report = leakage_report(sources, sources, examples=1)
    assert report["n_leaky"] == 2 and len(report["leaky"]) == 1
    assert len(leakage_report(sources, sources)["leaky"]) == 2
    assert report["threshold"] == defaults.TRACE_LEAK_THRESHOLD == 0.9


def test_fault_to_axis_is_a_public_table_the_caller_can_remap():
    rows = [_row("p", {"status": "error"}, reward=0)]
    default_axes = dimensions_from_traces(rows, TOOLS)
    remapped = dimensions_from_traces(
        rows,
        TOOLS,
        fault_to_axis={**FAULT_TO_AXIS, "error": ("tool_condition", "permission_denied")},
    )
    conditions = set(default_axes["tool_condition"])
    assert "timeout" in default_axes["tool_condition"][:2]
    assert "permission_denied" in remapped["tool_condition"][:2]
    assert set(remapped["tool_condition"]) <= conditions | {"permission_denied"}


def test_behavior_state_support_priority_and_exploration_are_keywords():
    rows = [
        _row("p", {"status": "timeout"}, reward=0, ts=1),
        _row("q", {"status": "timeout"}, reward=0, ts=2),
    ]
    by_default = behavior_state(rows)["regions"][0]
    assert by_default["low_support"] is True and defaults.TRACE_MIN_SUPPORT == 3
    assert behavior_state(rows, min_support=1)["regions"][0]["low_support"] is False
    weight = defaults.TRACE_STATE_PRIORITY[by_default["status"]]
    halved = behavior_state(
        rows, priority={**defaults.TRACE_STATE_PRIORITY, by_default["status"]: weight / 2}
    )["regions"][0]
    assert halved["priority"] == pytest.approx(by_default["priority"] / 2, abs=1e-3)
    with pytest.raises(ValueError, match="every status"):
        behavior_state(rows, priority={"new": 1.0})
    assert behavior_state(rows, exploration=0.9)["exploration_share"] == 0.6, "clamped"
    assert behavior_state(rows, exploration=0.9, max_exploration=0.9)["exploration_share"] == 0.9


def test_pseudo_production_fraction_default_is_the_named_number():
    rows = [_row(f"ask {i}", {"ok": True}, reward=1) for i in range(10)]
    held, rest = split_pseudo_production(rows)
    assert len(held) == round(defaults.TRACE_PSEUDO_PRODUCTION_FRACTION * 10) == 2
    assert len(rest) == 8


def test_result_binding_order_is_documented_and_fixed():
    assert RESULT_BINDING == ("id", "name", "fifo")


def test_otel_reward_attribute_is_a_keyword():
    spans = [
        {
            "traceId": "t1",
            "startTimeUnixNano": 1,
            "attributes": {
                "gen_ai.input.messages": '[{"role": "user", "content": "hi"}]',
                "my.reward": 1,
            },
        }
    ]
    assert "reward" not in rows_from_otel(spans)[0]
    assert rows_from_otel(spans, reward_keys=("my.reward",))[0]["reward"] == 1
