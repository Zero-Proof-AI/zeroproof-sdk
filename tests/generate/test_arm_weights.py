"""A caller can weight the situation search toward harder arms.

Measured on five eval sets (2026-09-17): base models score 18-32 points lower
on adversarial situations than on ordinary ones across three unrelated domains,
and structured generation is 9-16 points harder than open-ended on the airline
lanes. An eval set is only as hard as the arms it drew from, so the mix has to
be something a caller can set rather than a constant they inherit.
"""

import pytest

from whileai.simulations.generate.generator import make_default_generator
from whileai.simulations.generate.scenarios import SEARCH_ARMS, cap_open_ended_weight


def test_default_is_unchanged_when_nothing_is_asked_for():
    gen = make_default_generator([], simulator=False)
    assert gen.arm_weights == dict(SEARCH_ARMS)
    assert gen.arm_weights_pinned is False


def test_a_callers_weights_win():
    want = {"structured": 0.8, "llm_guided": 0.1, "open_ended": 0.1}
    gen = make_default_generator([], simulator=False, arm_weights=want)
    assert gen.arm_weights == want
    assert gen.arm_weights_pinned is True


def test_pinned_weights_survive_reallocation():
    """The search must not learn its way back to the default mid-run.

    A caller who raised the structured share to make a harder eval set has
    stated an intent about the eval, not a hypothesis for the search to test.
    """
    want = {"structured": 0.9, "open_ended": 0.1}
    gen = make_default_generator([], simulator=False, arm_weights=want)
    after = gen.reallocate({"structured": 0.0, "open_ended": 5.0})
    assert after == want
    assert gen.arm_weights == want


def test_unpinned_weights_still_reallocate():
    gen = make_default_generator([], simulator=False)
    before = dict(gen.arm_weights)
    gen.reallocate({"structured": 5.0, "open_ended": 0.0, "llm_guided": 0.0})
    assert gen.arm_weights != before


def test_open_ended_stays_capped_however_much_is_asked_for():
    """The 5-10% band holds against a caller as it holds against the search."""
    capped = cap_open_ended_weight({"structured": 0.1, "open_ended": 0.9})
    assert capped["open_ended"] <= 0.10


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"structured": -1.0},
        {"structured": "lots"},
        {"no_such_arm": 1.0},
        {"structured": 0.0, "open_ended": 0.0},
    ],
)
def test_bad_weights_are_refused_with_a_reason(bad):
    import whileai.simulations as wai

    with pytest.raises(ValueError) as err:
        wai.simulate(tools=[{"type": "function", "function": {"name": "t"}}],
                     system_prompt="p", arm_weights=bad, budget=0, simulator=False)
    assert "arm_weights" in str(err.value)
