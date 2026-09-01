"""steering_weight= must move the draw, not just the paperwork: weight w
sends structured card draws to the front (trace-mined) half of the steered
axes with probability w, marks those rows, and records the applied weight
only when it applied. w=0 or no traces stays byte-identical to today."""
from __future__ import annotations

from collections import Counter

from tests.helpers import POLICY, TOOLS, scripted_agent
from zeroproof_simulations.scenarios import (scenario_regions,
                                             steer_region_picks,
                                             steering_front_values)
from zeroproof_simulations.traces import (dimensions_from_traces,
                                          simulate_from_traces)

TRACES = [
    {"prompt": "where is order 4412",
     "steps": [{"tool": "lookup_order", "arguments": {"order_id": "4412"},
                "result": {"status": "not_found"}}],
     "final_text": "I could not find that order."},
    {"prompt": "refund order 9911 now",
     "steps": [{"tool": "create_refund", "arguments": {"order_id": "9911"},
                "result": {"status": "timeout"}}],
     "final_text": "The refund request timed out."},
]

_OFFLINE = dict(mode="explore", budget=16, seed=0, grade=False,
                concurrency=4, simulator=False, time_budget=None,
                advanced={"per_round": 16, "mutate_failures": False})


def _aimed(region: dict, front: dict) -> bool:
    assignment = region.get("assignment") or {}
    return all(str(assignment.get(axis)) in values
               for axis, values in front.items())


def _run(weight: float):
    return simulate_from_traces(TRACES, scripted_agent, tools=TOOLS,
                                policy=POLICY, steering_weight=weight,
                                **_OFFLINE)


def test_front_values_are_the_trace_mined_half():
    dims = dimensions_from_traces(TRACES, TOOLS, POLICY)
    front = steering_front_values(dims)
    # dimensions_from_traces sorts observed tools first; the front half
    # of the tool axis is exactly the trace-mined tools here.
    assert front["tool"] == {"lookup_order", "create_refund"}
    for axis, values in front.items():
        assert values <= set(dims[axis])
        assert len(values) <= max(1, len(dims[axis]) // 2)


def test_draw_helper_biases_toward_the_aimed_pool():
    dims = dimensions_from_traces(TRACES, TOOLS, POLICY)
    front = steering_front_values(dims)
    regions = scenario_regions(TOOLS, POLICY, dimensions=dims)
    ranked = sorted(regions, key=lambda region: region["id"])
    assert any(_aimed(region, front) for region in ranked), \
        "the covering grid must contain aimed cells for this test"
    background = [region for region in ranked
                  if not _aimed(region, front)][:6]

    plain, steered = steer_region_picks(
        background, ranked, seed=0, round_index=0, weight=0.0, front=front)
    assert plain == background
    assert steered == set()

    biased, steered = steer_region_picks(
        background, ranked, seed=0, round_index=0, weight=1.0, front=front)
    assert steered, "w=1 must actually draw from the aimed pool"
    for region in biased:
        if region["id"] in steered:
            assert _aimed(region, front)
    n_aimed = sum(1 for region in biased if _aimed(region, front))
    assert n_aimed > sum(1 for region in background
                         if _aimed(region, front))
    # Same seed, same call: the biased draw is deterministic.
    again, steered_again = steer_region_picks(
        background, ranked, seed=0, round_index=0, weight=1.0, front=front)
    assert again == biased and steered_again == steered


def test_w0_and_w1_produce_different_axis_distributions():
    front = steering_front_values(dimensions_from_traces(TRACES, TOOLS, POLICY))
    mined_tools = front["tool"]

    def tool_counts(data) -> Counter:
        return Counter(str((t.get("scenario_dimensions") or {}).get("tool"))
                       for t in data.trajectories
                       if t.get("scenario_dimensions"))

    d0, d1 = _run(0.0), _run(1.0)
    c0, c1 = tool_counts(d0), tool_counts(d1)
    assert c0 != c1, "same seed, w=0 vs w=1 must draw different cells"

    def mined_share(counts: Counter) -> float:
        total = sum(counts.values()) or 1
        return sum(n for tool, n in counts.items()
                   if tool in mined_tools) / total

    assert mined_share(c1) > mined_share(c0), \
        "w=1 must overrepresent the trace-mined axis values"

    targeted = [t for t in d1.trajectories
                if (t.get("steering") or {}).get("origin") == "targeted"]
    assert targeted, "w=1 must mark the rows it drew the biased way"
    for row in targeted:
        assert row["steering"]["weight"] == 1.0
        assert str(row["scenario_dimensions"]["tool"]) in mined_tools


def test_w0_marks_zero_targeted_rows():
    data = _run(0.0)
    assert all((t.get("steering") or {}).get("origin") != "targeted"
               for t in data.trajectories)
    assert data.metadata["targeted_rows"] == 0
    assert data.metadata["background_rows"] == len(data.trajectories)


def test_metadata_reports_only_what_applied():
    d0, d1 = _run(0.0), _run(1.0)
    # w=0 never applied: no applied weight, no targeted rows.
    assert d0.metadata["applied_steering_weight"] is None
    assert d0.search["strategy"]["steering_weight"] == {
        "requested": 0.0, "applied": None, "source": "override"}
    # w=1 applied: the recorded weight and counter match the rows.
    assert d1.metadata["applied_steering_weight"] == 1.0
    assert d1.search["strategy"]["steering_weight"] == {
        "requested": 1.0, "applied": 1.0, "source": "override"}
    marked = sum(1 for t in d1.trajectories
                 if (t.get("steering") or {}).get("origin") == "targeted")
    assert d1.metadata["targeted_rows"] == marked > 0
    assert d1.metadata["targeted_rows"] + d1.metadata["background_rows"] == \
        len(d1.trajectories)
    # No traces: nothing to aim at, nothing applied.
    from zeroproof_simulations import simulate
    plain = simulate(scripted_agent, tools=TOOLS, system_prompt=POLICY,
                     **{k: v for k, v in _OFFLINE.items() if k != "mode"},
                     mode="explore")
    assert plain.metadata["applied_steering_weight"] is None
    assert plain.metadata["targeted_rows"] == 0
