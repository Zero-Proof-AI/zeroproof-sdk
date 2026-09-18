"""The search must aim at the criterion that failed, not just at a low mean.

#285 measured that `traces=` reproduced situations but not failure modes: the
rules that got aimed at all had a world condition behind them, and four rules
about how the agent phrased its reply were missed. Part of the mechanism is
here -- a rubric scored as the mean of its criteria keeps a single broken rule
above a 0.5 threshold, so the row never became a mutation parent.
"""

from whileai.simulations.run.engine import _graded_failure
from whileai.simulations.run.rows import failed_criteria


def test_one_broken_rule_of_three_is_a_graded_failure():
    """0.667 passes a 0.5 threshold; the broken rule is still a failure.

    This is the row #285 is about: `one_question` fails, the other two pass,
    and reading only the scalar leaves the situation un-aimed-at.
    """
    row = {"reward": 2 / 3, "markers": {"grounded": 1.0, "one_question": 0.0, "stale": 1.0}}
    assert failed_criteria(row) == ["one_question"]
    assert _graded_failure(row) is True


def test_a_clean_row_is_not_a_failure():
    assert _graded_failure({"reward": 1.0, "markers": {"a": 1.0, "b": 1.0}}) is False
    assert failed_criteria({"reward": 1.0, "markers": {"a": 1.0}}) == []


def test_ungraded_rows_are_not_failures():
    """A judge error is not evidence the agent did anything wrong."""
    assert _graded_failure({"reward": None}) is False
    assert _graded_failure({}) is False
    assert failed_criteria({}) == []


def test_scalar_judges_still_work():
    """A 0/1 judge with no markers keeps the old behaviour."""
    assert _graded_failure({"reward": 0.0}) is True
    assert _graded_failure({"reward": 1.0}) is False


def test_markers_may_be_bools_or_unscorable():
    assert failed_criteria({"markers": {"a": True, "b": False}}) == ["b"]
    # a marker that is not a number is not a verdict, and is not a failure
    assert failed_criteria({"markers": {"note": "n/a"}}) == []
    assert failed_criteria({"markers": "not a dict"}) == []


def test_every_failed_criterion_is_named_not_just_the_first():
    row = {"reward": 0.25, "markers": {"a": 0.0, "b": 1.0, "c": 0.0}}
    assert failed_criteria(row) == ["a", "c"]
