"""A held-out set that cannot prove a gain must say so before the GPU spend.

``score.eval_power`` reads ``holdout_size`` and ``detectable_effect`` off the base
rows, so its numbers are theirs by construction, and adds where the tasks
sit against the difficulty band. Every test here fails on a tree without
it; the agreement tests fail on a second formula.
"""

from __future__ import annotations

import pytest

import whileai.simulations as wai
from whileai.simulations import defaults
from whileai.simulations.score.stats import detectable_effect, holdout_size


def _rows(per_task, k=4):
    """per_task: list of pass counts out of k, one per task."""
    out = []
    for i, passes in enumerate(per_task):
        for j in range(k):
            out.append({"prompt": f"task {i}", "reward": 1.0 if j < passes else 0.0})
    return out


MIXED = _rows([0, 1, 2, 3, 4] * 12)  # 60 tasks at k=4, base 0.5, 36 in the 20-80 band


def test_the_sizing_is_holdout_size_and_detectable_effect_on_the_same_rows():
    got = wai.score.eval_power(MIXED)
    assert got["n_tasks"] == 60 and got["n_rollouts"] == 240 and got["k"] == 4
    assert got["base"] == 0.5
    sizing = holdout_size(defaults.PROVE_EFFECT, before=MIXED)
    assert got["n_needed"] == sizing["n_tasks"]
    assert got["task_std"] == sizing["task_std"]
    assert got["task_spread"] == sizing["base_spread"]
    assert got["resolvable"] == detectable_effect(60, base=0.5, k=4)
    assert got["effect"] == defaults.PROVE_EFFECT == 0.05


def test_sixty_tasks_at_k4_cannot_prove_five_points_and_say_how_many_could():
    """The package's own sizing: at spread 0.35 a 5-point gain wants a few
    hundred tasks, so 60 is underpowered, and the warning says the count."""
    got = wai.score.eval_power(MIXED)
    assert got["verdict"] == "underpowered"
    assert got["resolvable"] > 0.05 and got["n_needed"] > 60
    assert got["in_band"] == 36
    assert len(got["warnings"]) == 1
    assert f"about {got['n_needed']} tasks" in got["warnings"][0]
    assert "effect=" in got["warnings"][0]


def test_the_same_set_is_usable_for_a_gain_it_can_resolve():
    got = wai.score.eval_power(MIXED, effect=0.2)
    assert got["verdict"] == "usable"
    assert got["resolvable"] <= 0.2
    assert got["warnings"] == []
    assert got["n_needed"] == holdout_size(0.2, before=MIXED)["n_tasks"] <= 60


def test_power_and_alpha_move_eval_power_and_holdout_size_together():
    strict = wai.score.eval_power(MIXED, power=0.9, alpha=0.01)
    assert strict["n_needed"] == holdout_size(0.05, before=MIXED, power=0.9, alpha=0.01)["n_tasks"]
    assert strict["resolvable"] == detectable_effect(60, base=0.5, k=4, power=0.9, alpha=0.01)
    assert strict["n_needed"] > wai.score.eval_power(MIXED)["n_needed"]


def test_the_band_is_a_knob():
    assert wai.score.eval_power(MIXED)["band"] == defaults.DIFFICULTY_BAND == (0.2, 0.8)
    assert wai.score.eval_power(MIXED, band=(0.4, 0.6))["in_band"] == 12  # only the 2-of-4 tasks
    with pytest.raises(ValueError, match="band"):
        wai.score.eval_power(MIXED, band=(0.8, 0.2))


def test_a_floored_set_is_named_instead_of_looking_hard():
    """Measured: an eval built to be "harder" came back base 0.000 with 0 of
    60 tasks in band. Its base score was the only number that looked like
    progress; the pass rate cannot tell hard tasks from a broken harness."""
    got = wai.score.eval_power(_rows([0] * 60))
    assert got["verdict"] == "floored"
    assert got["in_band"] == 0 and got["tied_fail"] == 60 and got["base"] == 0.0
    assert got["task_spread"] == 0.0
    assert "dead_tools" in got["warnings"][0]
    assert "20% to 80%" in got["warnings"][0]


def test_a_saturated_set_has_nothing_left_to_gain():
    got = wai.score.eval_power(_rows([4] * 40))
    assert got["verdict"] == "saturated"
    assert got["tied_pass"] == 40 and got["in_band"] == 0
    assert "nothing left to measure" in got["warnings"][0]
    # not an all-pass rule: base 0.97 cannot show a 5-point gain either
    near = wai.score.eval_power(_rows([4] * 39 + [3]))
    assert near["verdict"] == "saturated" and near["tied_pass"] == 39


def test_one_rollout_per_task_reports_the_band_as_unreadable_not_dead():
    """At k=1 a task's rate is one draw, so in_band is 0 by construction; the
    sizing still stands (the model at k=1 is the widest per-task spread)."""
    rows = _rows([0, 1] * 20, k=1)
    got = wai.score.eval_power(rows)
    assert got["k"] == 1 and got["single_rollout"] == 40 and got["in_band"] == 0
    assert got["verdict"] == "underpowered"
    assert any("k=1" in note for note in got["notes"])
    assert got["n_needed"] == holdout_size(0.05, before=rows)["n_tasks"]


def test_the_two_measured_arms_are_told_apart():
    """Same world, same rubric, same base model. The default set sits in
    band and is short on tasks; the one built to be harder is floored. The
    pass rate alone does not distinguish them."""
    default = _rows([2] * 23 + [4] * 22 + [0] * 14)  # 23 in band, base ~0.59
    harder = _rows([0] * 60)
    d, h = wai.score.eval_power(default), wai.score.eval_power(harder)
    assert d["verdict"] == "underpowered" and d["in_band"] == 23 and 0.55 < d["base"] < 0.65
    assert h["verdict"] == "floored" and h["in_band"] == 0
    assert d["resolvable"] is not None and d["resolvable"] < 0.5


def test_an_empty_or_ungraded_set_answers_with_the_fix():
    got = wai.score.eval_power([])
    assert got["verdict"] == "empty" and got["n_tasks"] == 0 and got["resolvable"] is None
    assert "grade" in got["warnings"][0]
    ungraded = wai.score.eval_power([{"prompt": "a"}, {"prompt": "b"}])
    assert ungraded["verdict"] == "empty"


def test_the_platform_holdout_effect_is_the_same_constant():
    assert defaults.PLATFORM_HOLDOUT_PROVE_EFFECT is defaults.PROVE_EFFECT
