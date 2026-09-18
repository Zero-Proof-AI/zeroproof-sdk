"""``hard_share=`` is a dial, not a suggestion: the mixer honours it, the
hard tiers share it evenly, and every run says what it drew."""

from __future__ import annotations

import json
import re
import threading

import pytest

from tests.helpers import simulate_offline
from whileai.simulations.generate import generator
from whileai.simulations.generate.diversity import (
    HARD_SHARE,
    behavior_tier,
    mix_items_by_tier,
)
from whileai.simulations.generate.generator import ModelSimulator, make_default_generator


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
@pytest.mark.parametrize("share", [0.5, 0.7, 0.8])
def test_mixer_draws_the_share_asked_for(n: int, share: float) -> None:
    # The old mixer floored ordinary at (n + 1) // 2, so any hard share over
    # a half came back as exactly half hard.
    items = _items(ordinary=n, ambiguous=n, boundary=n, adversarial=n)
    picked = mix_items_by_tier(items, n, _tier, hard_share=share)
    assert len(picked) == n
    assert n - _counts(picked)["ordinary"] == round(n * share)


def test_hard_tiers_share_the_remainder_evenly() -> None:
    # Draining "ambiguous" first turned 75% hard into 25/75/0/0.
    items = _items(ordinary=100, ambiguous=100, boundary=100, adversarial=100)
    picked = mix_items_by_tier(items, 100, _tier, hard_share=0.75)
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
    picked = mix_items_by_tier(items, 20, _tier, hard_share=1.0)
    assert _counts(picked) == {"ordinary": 1, "ambiguous": 10, "adversarial": 9}


def test_share_falls_back_to_ordinary_when_hard_tiers_run_dry() -> None:
    items = _items(ordinary=30, boundary=2)
    picked = mix_items_by_tier(items, 20, _tier, hard_share=0.9)
    assert len(picked) == 20
    assert _counts(picked) == {"ordinary": 18, "boundary": 2}


def test_mixer_default_is_the_package_share() -> None:
    items = _items(ordinary=50, ambiguous=50, boundary=50, adversarial=50)
    assert HARD_SHARE == 0.4
    assert _counts(mix_items_by_tier(items, 50, _tier))["ordinary"] == 30
    assert _counts(mix_items_by_tier(items, 50, _tier, hard_share=None))["ordinary"] == 30


@pytest.mark.parametrize("bad", [1.5, -0.1, "hard"])
def test_hard_share_is_validated_before_the_run_starts(bad: object) -> None:
    with pytest.raises(ValueError, match="hard_share"):
        simulate_offline(budget=4, hard_share=bad)


def test_plain_offline_run_records_the_mix_it_drew() -> None:
    data = simulate_offline(budget=40, concurrency=1, hard_share=0.8)
    mix = data.search["tier_mix"]
    counts = {}
    for row in data.trajectories:
        counts[row["tier"]] = counts.get(row["tier"], 0) + 1
    hard = sum(counts.get(t, 0) for t in ("ambiguous", "boundary", "adversarial"))
    assert mix["hard_share_requested"] == 0.8
    assert mix["rows"] == len(data.trajectories) >= 20
    assert mix["counts"] == counts
    assert mix["hard_share_realized"] == round(hard / mix["rows"], 4)
    # The dial moved the set well over the default 0.40, though rows the
    # mixer never sees (open asks, cells with no stance) keep it under 0.8.
    assert mix["hard_share_realized"] > 0.5
    if mix["hard_share_realized"] < 0.7:
        assert "hard_share=0.8 asked" in mix["note"]
        assert "dimensions={'stance'" in mix["note"]
        assert mix["note"] in data.warnings


def test_default_run_records_the_mix_without_a_note() -> None:
    data = simulate_offline(budget=40, concurrency=1)
    mix = data.search["tier_mix"]
    assert mix["hard_share_requested"] == 0.4
    assert "note" not in mix
    assert not any("hard_share" in w for w in data.warnings)


def test_share_reaches_the_writers_as_an_argument() -> None:
    # No hidden channel: the share is a constructor argument on the model
    # writer and a keyword on the structured writer's mixer calls.
    gen = make_default_generator([], simulator=False, hard_share=0.75)
    assert gen.model is None
    model = ModelSimulator("openai:x", tools=[], hard_share=0.75)
    assert model.hard_share == 0.75
    assert ModelSimulator("openai:x", tools=[]).hard_share == HARD_SHARE


def test_share_reaches_the_writer_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    # The model writer's rounds run on the scenario pool's workers, and it
    # mixes the cards for every request there. Each mixer call must carry
    # the run's share as an explicit keyword; a thread reading it from
    # ambient state would see the default instead.
    seen: list[tuple[bool, float | None]] = []
    lock = threading.Lock()
    real = generator.mix_items_by_tier

    def spy(items, n, tier_of, *, hard_share=None):
        with lock:
            seen.append((threading.current_thread() is threading.main_thread(), hard_share))
        return real(items, n, tier_of, hard_share=hard_share)

    def fake_writer(_url, _model, messages, **_kwargs):
        ids = re.findall(r'"region_id":\s*"(sc-[^"]+)"', messages[-1]["content"])
        return {
            "content": json.dumps(
                [{"region_id": rid, "message": f"refund order {rid}"} for rid in ids[:6]]
            )
        }

    monkeypatch.setattr(generator, "mix_items_by_tier", spy)
    monkeypatch.setattr(generator, "complete", fake_writer)
    simulate_offline(
        budget=24,
        concurrency=4,
        simulator="vllm:writer@http://127.0.0.1:9",
        hard_share=0.75,
        advanced={"scenario_concurrency": 2, "mutate_failures": False},
    )
    assert seen, "the writer never mixed"
    assert any(not on_main for on_main, _ in seen), "the writer never ran on a worker"
    assert {share for _, share in seen} == {0.75}
