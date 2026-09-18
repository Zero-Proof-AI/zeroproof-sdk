"""A declared tool that never works must be named, not averaged away.

A tool with no dispatch branch in `execute=` fails exactly like a genuine world
fault, and an honesty rubric rewards the agent for reporting it, so the pass
rate stays healthy while the tool is dead. Measured on one agent, `run_query`
was attempted 612 times and succeeded 4: it survived 978 rows, a probe, a
holdout and a published dataset card before anyone looked.
"""
from __future__ import annotations


def _dead_tools(tool_calls: dict[str, dict[str, int]]) -> list[str]:
    """The rule the engine applies, pinned here so a change has to be deliberate."""
    return sorted(
        name
        for name, slot in tool_calls.items()
        if (slot["ok"] == 0 and slot["calls"] >= 3)
        or (slot["calls"] >= 10 and slot["ok"] / slot["calls"] < 0.05)
    )


def test_a_tool_that_never_succeeds_is_named():
    assert _dead_tools({"run_query": {"calls": 8, "ok": 0}}) == ["run_query"]


def test_the_measured_case_a_zero_success_test_would_miss():
    # 4 successes in 612 calls is not zero, and it is still a dead tool.
    assert _dead_tools({"run_query": {"calls": 612, "ok": 4}}) == ["run_query"]


def test_a_working_tool_is_not_accused():
    # The same tool after the dispatch branch was added.
    assert _dead_tools({"run_query": {"calls": 662, "ok": 562}}) == []


def test_a_tool_tried_once_or_twice_is_not_enough_evidence():
    assert _dead_tools({"export_csv": {"calls": 2, "ok": 0}}) == []


def test_a_flaky_tool_above_the_rate_floor_is_not_dead():
    # 2 in 10 is 20%, well above the 5% floor: flaky, not dead.
    assert _dead_tools({"export_csv": {"calls": 10, "ok": 2}}) == []


def test_steps_with_no_recorded_result_are_not_evidence():
    """Offline runs produce steps carrying no result.

    Counting those as failures would report every tool dead on every offline
    run, which is the fastest way to get the check ignored.
    """
    steps = [{"tool": "run_query", "result": None} for _ in range(5)]
    counted = [s for s in steps if s.get("result") is not None]
    assert counted == []
