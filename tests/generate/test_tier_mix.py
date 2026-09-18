"""``ordinary_share=`` is a dial, not a suggestion: the mixer honours it,
the hard tiers share the remainder, and every run says what it drew."""

from __future__ import annotations

import threading

import pytest

from tests.helpers import simulate_offline
from whileai.simulations.generate.diversity import (
    behavior_tier,
    current_ordinary_share,
    mix_items_by_tier,
    set_ordinary_share,
)


def _items(**per_tier: int) -> list[dict]:
    return [{"assignment": {"stance": stance}} for stance, n in per_tier.items() for _ in range(n)]


def _tier(row: dict) -> str:
    return behavior_tier(row.get("assignment") or {})


def _counts(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[_tier(row)] = out.get(_tier(row), 0) + 1
    return out


@pytest.mark.parametrize("n", [20, 100])
@pytest.mark.parametrize("share", [0.2, 0.3, 0.5])
def test_mixer_draws_the_share_asked_for(n: int, share: float) -> None:
    # The old mixer floored ordinary at (n + 1) // 2, so 0.2, 0.3 and 0.5
    # all came back as exactly half ordinary.
    items = _items(ordinary=n, ambiguous=n, boundary=n, adversarial=n)
    picked = mix_items_by_tier(items, n, _tier, ordinary_share=share)
    assert len(picked) == n
    assert _counts(picked)["ordinary"] == round(n * share)


def test_hard_tiers_share_the_remainder_evenly() -> None:
    # Draining "ambiguous" first turned 25% ordinary into 25/75/0/0.
    items = _items(ordinary=100, ambiguous=100, boundary=100, adversarial=100)
    picked = mix_items_by_tier(items, 100, _tier, ordinary_share=0.25)
    assert _counts(picked) == {
        "ordinary": 25,
        "ambiguous": 25,
        "boundary": 25,
        "adversarial": 25,
    }


def test_round_robin_skips_an_empty_hard_tier() -> None:
    # The breadth-first pass still seeds one row per tier present, so a
    # short run touches every tier; the remainder follows the share.
    items = _items(ordinary=10, ambiguous=10, adversarial=10)
    picked = mix_items_by_tier(items, 20, _tier, ordinary_share=0.0)
    assert _counts(picked) == {"ordinary": 1, "ambiguous": 10, "adversarial": 9}


def test_share_falls_back_to_ordinary_when_hard_tiers_run_dry() -> None:
    items = _items(ordinary=30, boundary=2)
    picked = mix_items_by_tier(items, 20, _tier, ordinary_share=0.1)
    assert len(picked) == 20
    assert _counts(picked) == {"ordinary": 18, "boundary": 2}


def test_mixer_reads_the_run_share_when_no_share_is_passed() -> None:
    items = _items(ordinary=50, ambiguous=50, boundary=50, adversarial=50)
    set_ordinary_share(0.2)
    try:
        assert current_ordinary_share() == 0.2
        assert _counts(mix_items_by_tier(items, 50, _tier))["ordinary"] == 10
    finally:
        set_ordinary_share(None)
    assert current_ordinary_share() == 0.6
    assert _counts(mix_items_by_tier(items, 50, _tier))["ordinary"] == 30


@pytest.mark.parametrize("bad", [1.5, -0.1, "hard"])
def test_ordinary_share_is_validated_before_the_run_starts(bad: object) -> None:
    with pytest.raises(ValueError, match="ordinary_share"):
        simulate_offline(budget=4, ordinary_share=bad)


def test_plain_offline_run_records_the_mix_it_drew() -> None:
    data = simulate_offline(budget=40, concurrency=1, ordinary_share=0.2)
    mix = data.search["tier_mix"]
    counts = {}
    for row in data.trajectories:
        counts[row["tier"]] = counts.get(row["tier"], 0) + 1
    assert mix["requested_ordinary_share"] == 0.2
    assert mix["rows"] == len(data.trajectories) >= 20
    assert mix["counts"] == counts
    assert mix["realized_ordinary_share"] == round(counts.get("ordinary", 0) / mix["rows"], 4)
    # The dial moved the set well under the default 0.60, though rows the
    # mixer never sees (open asks, cells with no stance) keep it above 0.2.
    assert mix["realized_ordinary_share"] < 0.5
    if mix["realized_ordinary_share"] > 0.3:
        assert "ordinary_share=0.2 asked" in mix["note"]
        assert "dimensions={'stance'" in mix["note"]
        assert mix["note"] in data.warnings


def test_default_run_records_the_mix_without_a_note() -> None:
    data = simulate_offline(budget=40, concurrency=1)
    mix = data.search["tier_mix"]
    assert mix["requested_ordinary_share"] == 0.6
    assert "note" not in mix
    assert not any("ordinary_share" in w for w in data.warnings)


def test_share_reaches_the_writer_threads() -> None:
    # The share lives in a context variable. A worker thread starts with
    # an empty context on 3.10 to 3.13, so the pools have to set it.
    seen: list[tuple[bool, float]] = []
    lock = threading.Lock()

    def writer(_dataset=None, index=0):
        with lock:
            seen.append(
                (threading.current_thread() is threading.main_thread(), current_ordinary_share())
            )
        return [f"refund order {index}-{i}" for i in range(12)]

    simulate_offline(
        budget=24,
        concurrency=4,
        simulator=writer,
        ordinary_share=0.25,
        advanced={"scenario_concurrency": 2, "mutate_failures": False},
    )
    assert seen, "the writer never ran"
    assert any(not on_main for on_main, _ in seen), "the writer never ran on a worker"
    assert {share for _, share in seen} == {0.25}
