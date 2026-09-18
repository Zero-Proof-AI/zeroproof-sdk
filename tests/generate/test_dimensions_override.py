"""``simulate(dimensions=)`` overrides an axis of the coverage grid. It used
to replace the grid: ``{"stance": [...]}`` left two regions with no tool or
rule axis, and ``{"tier": [...]}`` steered nothing because no cell reads a
``tier`` key. A caller asked for a harder set and got a smaller one."""

from __future__ import annotations

import pytest

from whileai.simulations.generate.diversity import behavior_tier
from whileai.simulations.generate.scenarios import (
    COVERAGE_AXES,
    build_dimensions,
    check_dimensions,
    merge_dimensions,
    scenario_regions,
)

TOOLS = [
    {
        "name": "lookup_order",
        "description": "look up an order",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
    },
    {
        "name": "refund",
        "description": "refund an order",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
    },
]
POLICY = "1. Never refund over $100 without approval.\n2. Verify identity first."


def _axes(regions: list[dict]) -> set[str]:
    return {k for r in regions for k in (r.get("assignment") or {})}


def _tiers(regions: list[dict]) -> set[str]:
    return {behavior_tier(r.get("assignment") or {}) for r in regions}


def test_stance_override_keeps_the_rest_of_the_grid() -> None:
    default = scenario_regions(TOOLS, POLICY)
    hard = scenario_regions(TOOLS, POLICY, dimensions={"stance": ["adversarial", "boundary"]})
    assert _axes(hard) == _axes(default) == set(COVERAGE_AXES)
    assert _tiers(hard) == {"adversarial", "boundary"}
    assert "ordinary" in _tiers(default)
    # Still a covering grid, not a two-cell stub.
    assert len(hard) >= 10
    tools_seen = {r["assignment"]["tool"] for r in hard}
    assert {"lookup_order", "refund"} <= tools_seen


def test_merge_dimensions_overrides_only_the_named_axis() -> None:
    base = build_dimensions(TOOLS, POLICY)
    merged = merge_dimensions(base, {"stance": ["boundary"]})
    assert merged["stance"] == ["boundary"]
    for axis in COVERAGE_AXES:
        if axis != "stance":
            assert merged[axis] == base[axis]
    # An empty override list leaves the axis alone.
    assert merge_dimensions(base, {"stance": []})["stance"] == base["stance"]
    assert merge_dimensions(base, None) == base


def test_tier_is_refused_and_the_message_names_stance() -> None:
    with pytest.raises(ValueError, match="stance"):
        check_dimensions({"tier": ["adversarial", "boundary"]})


def test_unknown_axis_is_refused_and_lists_the_axes() -> None:
    with pytest.raises(ValueError, match="tool, rule, stance"):
        check_dimensions({"mood": ["grumpy"]})


def test_unknown_stance_value_is_refused_and_lists_known_ones() -> None:
    with pytest.raises(ValueError, match="adversarial"):
        check_dimensions({"stance": ["sceptical"]})


@pytest.mark.parametrize("bad", [{}, "stance", {"stance": []}, {"stance": "boundary"}])
def test_shapes_that_would_steer_nothing_are_refused(bad: object) -> None:
    with pytest.raises(ValueError, match="dimensions="):
        check_dimensions(bad)


def test_none_and_a_full_grid_pass() -> None:
    check_dimensions(None)
    check_dimensions(build_dimensions(TOOLS, POLICY))


def test_simulate_refuses_a_tier_axis_before_the_run_starts() -> None:
    from tests.helpers import simulate_offline

    with pytest.raises(ValueError, match="stance"):
        simulate_offline(budget=4, dimensions={"tier": ["adversarial"]})


def test_offline_run_with_a_stance_override_pins_every_structured_card() -> None:
    from tests.helpers import simulate_offline

    data = simulate_offline(
        budget=24, concurrency=1, dimensions={"stance": ["adversarial", "boundary"]}
    )
    with_stance = [
        row for row in data.trajectories if (row.get("scenario_dimensions") or {}).get("stance")
    ]
    assert len(with_stance) >= 10, "the grid never reached the rows"
    assert {row["tier"] for row in with_stance} <= {"adversarial", "boundary"}
    # Open-ended and behavior probes carry no stance. They are not pinned,
    # and they are not ordinary either; the report counts them unlabelled.
    without = [row for row in data.trajectories if row not in with_stance]
    assert {row.get("arm") for row in without} <= {"open_ended", "behavior_targeted"}, without
