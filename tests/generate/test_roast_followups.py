"""Trace allocation reads the fail rate; coverage reports its pairs."""
from __future__ import annotations

from zeroproof.simulations.generate.coverage import pairwise_coverage
from zeroproof.simulations.generate.diversity import sample_turn_budget
from zeroproof.simulations.ingest.traces import behavior_state


def _row(marker: str, ok: bool, version: str) -> dict:
    return {"prompt": f"p-{marker}-{version}-{ok}", "model_version": version,
            "scores": {marker: 1 if ok else 0},
            "scenario_dimensions": {"tool": "edit_file", "stance": "hurried"},
            "steps": [{"tool": "edit_file", "arguments": {}, "result": {"status": "ok"}}],
            "final_text": "done"}


def test_rare_failure_draws_less_budget_than_frequent_failure():
    rows = []
    rows += [_row("rare", False, "v0")] + [_row("rare", True, "v0")] * 49
    rows += [_row("rare", False, "v1")] + [_row("rare", True, "v1")] * 49
    rows += [_row("often", False, "v0")] * 25 + [_row("often", True, "v0")] * 25
    rows += [_row("often", False, "v1")] * 25 + [_row("often", True, "v1")] * 25
    by = {r["region"]: r for r in behavior_state(rows)["regions"]}
    assert by["rare"]["status"] == by["often"]["status"] == "persistent"
    assert by["rare"]["fail_rate"] < 0.05 and by["often"]["fail_rate"] > 0.45
    assert by["often"]["budget_share"] > 2 * by["rare"]["budget_share"]
    assert by["rare"]["n_graded"] == 100 and not by["rare"]["low_support"]


def test_one_row_region_is_flagged_low_support():
    state = behavior_state([_row("blip", False, "v0")] + [_row("steady", False, "v0")] * 10)
    by = {r["region"]: r for r in state["regions"]}
    assert by["blip"]["low_support"] is True
    assert by["steady"]["low_support"] is False
    assert abs(sum(r["budget_share"] for r in state["regions"])
               + state["exploration_share"] - 1.0) < 0.01


def test_pairwise_coverage_counts_planned_pairs_only():
    planned = [{"tool": "a", "stance": "x", "history": "fresh"},
               {"tool": "b", "stance": "y", "history": "fresh"}]
    full = pairwise_coverage(planned, planned)
    assert full["fraction"] == 1.0 and full["pairs_planned"] == 6
    half = pairwise_coverage(planned, [planned[0]])
    assert half["pairs_covered"] == 3 and half["fraction"] == 0.5
    extra = pairwise_coverage(planned, [{"tool": "a", "stance": "x",
                                         "history": "fresh", "origin": "realized",
                                         "tool_condition": "timeout"}])
    assert extra["pairs_covered"] == 3
    assert pairwise_coverage([], [])["fraction"] is None


def test_turn_controller_corrects_at_half_gain():
    # far above target: the center moves toward target, not past it
    high = [sample_turn_budget(s, f"k{s}", 12, avg_turns=6, running_mean=10)
            for s in range(400)]
    low = [sample_turn_budget(s, f"k{s}", 12, avg_turns=6, running_mean=2)
           for s in range(400)]
    assert sum(high) / len(high) < sum(low) / len(low)
    assert 2 <= min(high) and max(low) <= 12
