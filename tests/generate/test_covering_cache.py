"""The covering array is memoized per grid, and callers get their own rows."""

from __future__ import annotations

from zeroproof.simulations.generate import scenarios
from zeroproof.simulations.generate.scenarios import (
    _covering_assignments,
    _covering_assignments_uncached,
)

DIMS = {
    "tool": ["get_order", "refund"],
    "tool_condition": ["success", "timeout", "not_found"],
    "stance": ["calm", "angry"],
}


def test_cached_result_equals_the_greedy_result():
    scenarios._COVERING_CACHE.clear()
    assert _covering_assignments(DIMS, 2) == _covering_assignments_uncached(DIMS, 2)
    assert len(scenarios._COVERING_CACHE) == 1


def test_second_call_does_not_recompute(monkeypatch):
    scenarios._COVERING_CACHE.clear()
    _covering_assignments(DIMS, 2)
    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("recomputed")

    monkeypatch.setattr(scenarios, "_covering_assignments_uncached", boom)
    assert _covering_assignments(DIMS, 2)
    assert calls["n"] == 0
    # A different strength or grid is a different key.
    monkeypatch.setattr(scenarios, "_covering_assignments_uncached", lambda d, t: [{"x": "y"}])
    assert _covering_assignments(DIMS, 1) == [{"x": "y"}]


def test_callers_may_mutate_their_rows():
    scenarios._COVERING_CACHE.clear()
    first = _covering_assignments(DIMS, 2)
    first[0]["tool_condition"] = "mutated"
    first.clear()
    second = _covering_assignments(DIMS, 2)
    assert second and second[0]["tool_condition"] != "mutated"
