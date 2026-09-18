"""A held-out set that cannot prove a gain must say so before the GPU spend."""

from __future__ import annotations

import warnings

import whileai.simulations as wai


def _rows(per_task, k=4):
    """per_task: list of pass counts out of k, one per task."""
    out = []
    for i, passes in enumerate(per_task):
        for j in range(k):
            out.append({"prompt": f"task {i}", "reward": 1.0 if j < passes else 0.0})
    return out


def test_a_set_whose_tasks_disagree_is_usable_and_sized():
    rows = _rows([0, 1, 2, 3, 4] * 12)  # 60 tasks, most of them mixed
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a usable set must not warn
        got = wai.eval_power(rows)
    assert got["verdict"] == "usable"
    assert got["informative"] == 36, got
    assert got["n_tasks"] == 60 and got["n_rollouts"] == 240
    assert got["resolvable"] and 0.0 < got["resolvable"] < 0.5
    assert got["n_needed"] and got["n_needed"] > 0


def test_a_floored_set_warns_instead_of_looking_hard():
    """Measured: an eval built to be 'harder' came back base 0.000 with 0 of
    60 tasks disagreeing. Its interval is tight around zero and means
    nothing, and its base score is the only number that looks like progress."""
    rows = _rows([0] * 60)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        got = wai.eval_power(rows)
    assert got["verdict"] == "floored"
    assert got["informative"] == 0 and got["tied_fail"] == 60
    assert got["base_pass"] == 0.0 and got["task_sd"] == 0.0
    assert got["resolvable"] is None
    assert any("cannot prove a gain" in str(w.message) for w in seen)


def test_a_saturated_set_warns_at_the_other_end():
    rows = _rows([4] * 40)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        got = wai.eval_power(rows)
    assert got["verdict"] == "saturated"
    assert got["tied_pass"] == 40 and got["informative"] == 0
    assert any("nothing left to gain" in str(w.message) for w in seen)


def test_one_rollout_per_task_cannot_disagree_with_itself():
    rows = _rows([0, 1] * 20, k=1)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        got = wai.eval_power(rows)
    assert got["verdict"] == "dead"
    assert got["single_rollout"] == 40 and got["informative"] == 0
    assert any("disagree" in str(w.message) for w in seen)


def test_the_two_measured_arms_are_told_apart():
    """The real pair, same world, same rubric, same base model. The default
    set resolves; the one built to be harder does not, and the pass rate
    alone does not distinguish them."""
    default = _rows([2] * 23 + [4] * 22 + [0] * 14)  # 23 disagree, base ~0.59
    harder = _rows([0] * 60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d, h = wai.eval_power(default), wai.eval_power(harder)
    assert d["verdict"] == "usable" and d["informative"] == 23
    assert h["verdict"] == "floored" and h["informative"] == 0
    assert d["resolvable"] is not None and h["resolvable"] is None


def test_an_empty_set_answers_rather_than_dividing_by_zero():
    got = wai.eval_power([])
    assert got["verdict"] == "empty" and got["n_tasks"] == 0
