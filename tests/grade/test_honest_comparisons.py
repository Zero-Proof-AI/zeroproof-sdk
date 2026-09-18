"""#375 and #392: two honesty gaps in the comparison and sizing code.

#375: ``delta_report`` failed two offline arms as ``NOT COMPARABLE:
writer_model`` when one was drawn with ``seeds=`` (stamp ``seed``) and the
other replayed the same situations under another stamp, although both arms
covered the same tasks. Comparability now keys on the situation identity
(``task_key``): a writer difference over the same task set is a note, not a
failure; over different task sets it still fails. ``simulate(seeds=...,
runs=N)`` also raised, because the replays carried the seeds again.

#392: ``holdout_size(effect, before=rows)`` on a baseline every task passes
returned ``n_tasks=2, task_std=0.0`` with no warning: the binomial model at
``p = 1`` and the sizing formula's floor, not evidence. A saturated
baseline now answers with the model at the default base, says ``saturated``,
and warns with the ceiling and the fix.
"""

from __future__ import annotations

import pytest

import whileai.simulations as wai
from tests.helpers import simulate_offline
from whileai.simulations import defaults
from whileai.simulations.score import delta
from whileai.simulations.score.delta import delta_report, format_delta_report
from whileai.simulations.score.stats import MIN_HOLDOUT_TASKS, holdout_size

# ------------------------------------------------------------------ #375


def _arm(pass_by_task: dict[str, float], k: int = 4, **stamp) -> list[dict]:
    rows = []
    for task, p in pass_by_task.items():
        passes = round(p * k)
        for i in range(k):
            rows.append(
                {
                    "scenario_id": task,
                    "prompt": f"ask {task}",
                    "rollout_index": i,
                    "reward": 1.0 if i < passes else 0.0,
                    **stamp,
                }
            )
    return rows


SAME_TASKS = {f"t{i}": 0.5 for i in range(12)}


def test_writer_stamps_that_differ_over_the_same_situations_are_a_note_not_a_failure():
    """The issue's shape: a seeds= arm against a replay of its own task set
    under another writer stamp. Every task_key is on both sides."""
    before = _arm(SAME_TASKS, writer_model="seed")
    after = _arm({t: 0.75 for t in SAME_TASKS}, writer_model="Qwen/Qwen3.8-27B")
    report = delta_report(before, after, n_boot=100)
    assert report["not_comparable"] == []
    assert report["ok"] is True
    assert report["headline_verdict"] == "moved_unreplicated"
    [note] = [w for w in report["warnings"] if "writer_model was 'seed'" in w]
    assert "every one of the 12 tasks is on both sides (task_key)" in note
    assert "the delta stands" in note
    assert not any(w.startswith("NOT COMPARABLE: writer_model") for w in report["warnings"])
    assert format_delta_report(report).splitlines()[0].startswith("PASS")


def test_writer_stamps_that_differ_over_different_situations_still_fail():
    before = _arm(SAME_TASKS | {f"a{i}": 0.5 for i in range(3)}, writer_model="seed")
    after = _arm(SAME_TASKS | {f"b{i}": 0.5 for i in range(3)}, writer_model="template")
    report = delta_report(before, after, n_boot=100)
    assert "writer_model" in report["not_comparable"] and report["ok"] is False
    assert any(w.startswith("NOT COMPARABLE: writer_model") for w in report["warnings"])


def test_a_user_model_that_moved_fails_even_over_the_same_situations():
    """The user model plays turns inside every rollout, so the task set does
    not clear it."""
    before = _arm(SAME_TASKS, user_model="qwen-a")
    after = _arm(SAME_TASKS, user_model="qwen-b")
    report = delta_report(before, after, n_boot=100)
    assert report["not_comparable"] == ["user_model"]


def test_seeds_arm_and_its_tasks_replay_compare_end_to_end():
    seeds = [f"refund order 8{i}" for i in range(6)] + [f"where is order 7{i}" for i in range(6)]
    before = simulate_offline(seeds=seeds, repeats=2, budget=24, grade=True, concurrency=1)
    after = simulate_offline(tasks=before, budget=24, grade=True, concurrency=1)
    report = delta_report(before.trajectories, after.trajectories, n_boot=100)
    assert report["not_comparable"] == []
    assert report["n_paired_tasks"] == len({r["scenario_id"] for r in before.trajectories})


def test_seeds_with_runs_draws_once_and_replays():
    seeds = [f"refund order 8{i}" for i in range(6)]
    data = simulate_offline(seeds=seeds, repeats=1, budget=6, runs=2, concurrency=1, seed=3)
    runs = sorted({r["lineage"]["eval_run"] for r in data.trajectories})
    assert runs == [0, 1]
    first = {r["scenario_id"] for r in data.trajectories if r["lineage"]["eval_run"] == 0}
    second = {r["scenario_id"] for r in data.trajectories if r["lineage"]["eval_run"] == 1}
    assert first == second
    assert {r["writer_model"] for r in data.trajectories} == {"seed"}


def test_ceiling_share_has_one_home():
    assert delta.CEILING_PASS_RATE is defaults.CEILING_PASS_RATE


# ------------------------------------------------------------------ #392


def _graded(rate: float, n_tasks: int = 10, k: int = 6) -> list[dict]:
    rows = []
    for t in range(n_tasks):
        passes = round(rate * k)
        for i in range(k):
            rows.append(
                {
                    "scenario_id": f"q{t}",
                    "prompt": f"q {t}",
                    "reward": 1.0 if i < passes else 0.0,
                    "rollout_index": i,
                }
            )
    return rows


def test_the_issue_repro_no_longer_answers_two():
    rows = [
        {"scenario_id": f"q{t}", "prompt": f"q {t}", "reward": 1.0, "rollout_index": k}
        for t in range(10)
        for k in range(6)
    ]
    hs = wai.holdout_size(0.05, before=rows)
    assert hs["saturated"] is True
    assert hs["base"] == 1.0 and hs["k"] == 6
    assert hs["sd_source"] == "model"
    # the model's answer at the default base, the same as with no rows
    assert hs["n_tasks"] == wai.holdout_size(0.05, base=defaults.BASE_PASS_RATE, k=6)["n_tasks"]
    assert hs["n_tasks"] > MIN_HOLDOUT_TASKS and hs["task_std"] > 0 and hs["half_width"] > 0
    [line] = hs["warnings"]
    assert line.startswith("CEILING:")
    assert "pass 1.00 of tasks, at or above the ceiling 0.90" in line
    assert f"({MIN_HOLDOUT_TASKS} tasks), which is the model collapsing, not evidence" in line
    assert f"default base {defaults.BASE_PASS_RATE:.2f} with these rows' k=6" in line
    assert "harder situations" in line and "20%-80% difficulty band" in line
    assert "hard_share" in line and "rlhfbook.com/c/14-reasoning.html" in line


def test_both_arms_saturated_measure_a_zero_sd_and_get_the_same_answer():
    before = _graded(1.0)
    after = _graded(1.0)
    hs = holdout_size(0.05, before=before, after=after)
    assert hs["saturated"] is True and hs["sd_source"] == "model"
    assert hs["n_tasks"] == wai.holdout_size(0.05, base=defaults.BASE_PASS_RATE, k=6)["n_tasks"]
    assert hs["n_paired"] is None
    assert "task_std 0.000 measured" in " ".join(hs["notes"])
    assert hs["warnings"] and "at or above the ceiling" in hs["warnings"][0]


def test_a_floor_baseline_with_zero_measured_sd_is_also_saturated():
    """Every task fails on both arms: the base is 0, the measured paired sd
    is 0, and the answer would be the floor again."""
    hs = holdout_size(0.05, before=_graded(0.0), after=_graded(0.0))
    assert hs["saturated"] is True
    [line] = hs["warnings"]
    assert line.startswith("DEGENERATE: the paired difference is the same on every one of the 10")
    assert "(task_std 0.000 measured is 0) at a base of 0.00" in line
    assert hs["n_tasks"] > MIN_HOLDOUT_TASKS


def test_a_baseline_inside_the_band_is_unchanged():
    before = _graded(0.5)
    hs = holdout_size(0.05, before=before)
    assert hs["saturated"] is False and hs["warnings"] == []
    assert hs["base"] == 0.5 and hs["sd_source"] == "model"
    assert hs["n_tasks"] == holdout_size(0.05, base=0.5, k=6)["n_tasks"]
    after = _graded(0.5)
    for row in after:  # a gain that varies by task, so the paired sd is real
        if row["scenario_id"] in {"q0", "q1", "q2"} and row["reward"] == 0.0:
            row["reward"] = 1.0
    measured = holdout_size(0.05, before=before, after=after)
    assert measured["saturated"] is False and measured["sd_source"] == "rows"
    assert measured["task_std"] > 0 and measured["warnings"] == []


def test_the_ceiling_is_a_knob():
    rows = _graded(0.9, k=10)
    assert holdout_size(0.05, before=rows)["saturated"] is True
    relaxed = holdout_size(0.05, before=rows, ceiling_pass_rate=0.95)
    assert relaxed["saturated"] is False and relaxed["base"] == pytest.approx(0.9)
    # a given task_std is the caller's measurement; the rows do not override it
    given = holdout_size(0.05, before=_graded(1.0), task_std=0.38)
    assert given["saturated"] is False and given["sd_source"] == "given"
