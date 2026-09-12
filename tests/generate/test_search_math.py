"""Three search formulas that were measured wrong and are now pinned."""
from __future__ import annotations

import random

import pytest

from tests.helpers import POLICY, TOOLS
from zeroproof.simulations.generate.diversity import (accept_anneal_candidate,
                                                      apply_annealing_explore)
from zeroproof.simulations.generate.scenarios import (SEARCH_ARMS, complete_yields,
                                                      reallocate_search_arms)
from zeroproof.simulations.score.optimize import recommend


def _rate(novelty: float, temperature: float = 1.0, n: int = 2000) -> float:
    rng = random.Random(0)
    return sum(accept_anneal_candidate(novelty, temperature=temperature, rng=rng)
               for _ in range(n)) / n


def test_explore_slots_prefer_novel_candidates():
    assert _rate(0.9) > 0.9
    assert _rate(0.0) < 0.25
    assert _rate(0.9) > _rate(0.5) > _rate(0.0)
    assert _rate(0.9, temperature=0.05) < _rate(0.9, temperature=1.0)


def test_explore_swaps_in_the_most_novel_pool_members():
    # batch holds the two least novel of five; the pool's most novel index wins the slot
    novelty_of = {0: 0.05, 1: 0.10, 2: 0.95, 3: 0.60, 4: 0.90}
    out = apply_annealing_explore([0, 1], texts_len=5, novelty_of=novelty_of,
                                  need=2, round_index=0, seed=0)
    assert 2 in out or 4 in out
    assert 0 not in out or 1 not in out


def test_idle_arm_carries_no_vote():
    filled = complete_yields({"structured": 0.4, "llm_guided": 0.2})
    assert filled["open_ended"] == filled["behavior_targeted"] == pytest.approx(0.3)

    weights = dict(SEARCH_ARMS)
    for _ in range(10):
        weights = reallocate_search_arms(
            weights, {"structured": 0.5, "llm_guided": 0.5})
    # rare arms never ran; they must not have gained against the field
    assert weights["failure_mutation"] <= SEARCH_ARMS["failure_mutation"] + 1e-9
    assert weights["behavior_targeted"] <= SEARCH_ARMS["behavior_targeted"] + 1e-9


def test_arm_that_found_more_gains_share():
    weights = dict(SEARCH_ARMS)
    for _ in range(5):
        weights = reallocate_search_arms(
            weights, {"structured": 1.5, "llm_guided": 0.1})
    assert weights["structured"] > weights["llm_guided"]


def test_recommend_expected_mixed_rows_meet_the_target():
    for rate in (0.5, 0.2, 0.1, 0.04):
        rec = recommend(TOOLS, POLICY, mode="rl", target=800, mixed_rate=rate)
        expected_mixed = rec["budget"] * rate
        assert expected_mixed >= 800, (rate, rec["budget"], expected_mixed)
        assert rec["situations"] * rec["rollouts_per_request"] == rec["budget"]
