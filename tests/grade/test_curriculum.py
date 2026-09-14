"""curriculum() splits graded tasks by difficulty; retire_solved drops them."""

from __future__ import annotations

from zeroproof.simulations.score.curriculum import (
    curriculum,
    format_curriculum,
    retire_solved,
)


def rows_for(prompt, labels):
    return [{"prompt": prompt, "reward": v, "scenario_id": prompt} for v in labels]


def make():
    rows = []
    rows += rows_for("solved", [1, 1, 1, 1])  # pass 1.0 -> retire
    rows += rows_for("hard", [0, 0, 0, 0])  # pass 0.0 -> not ready
    rows += rows_for("mid", [1, 0, 1, 0])  # pass 0.5 -> trainable, in band
    rows += rows_for("easyish", [1, 1, 1, 0])  # pass 0.75 -> trainable, in band
    rows += rows_for("thin", [1])  # one rollout -> thin
    return rows


def test_buckets_by_difficulty():
    c = curriculum(make())
    assert c["n_tasks"] == 5
    assert [s["task_id"] for s in c["retired"]] == ["solved"]
    assert [s["task_id"] for s in c["not_ready"]] == ["hard"]
    assert {s["task_id"] for s in c["trainable"]} == {"mid", "easyish"}
    assert [s["task_id"] for s in c["thin"]] == ["thin"]


def test_schedule_is_easy_to_hard():
    c = curriculum(make())
    # easyish (0.75) before mid (0.5)
    assert c["schedule"] == ["easyish", "mid"]


def test_in_band_count():
    c = curriculum(make(), band=(0.2, 0.8))
    assert c["n_in_band"] == 2  # 0.5 and 0.75 both in [0.2, 0.8]


def test_thresholds_are_configurable():
    c = curriculum(make(), solved=0.7)
    # now easyish (0.75) is retired too
    assert {s["task_id"] for s in c["retired"]} == {"solved", "easyish"}
    assert [s["task_id"] for s in c["trainable"]] == ["mid"]


def test_tiers_split_trainable():
    rows = []
    for i in range(9):
        rows += rows_for(f"t{i}", [1] * (i % 3) + [0] * (3 - i % 3))  # varied pass rates
    c = curriculum(rows, tiers=3)
    # every trainable task lands in exactly one tier
    tiered = sum(len(b) for b in c["tiers"])
    assert tiered == c["n_trainable"]


def test_retire_solved_drops_rows():
    rows = make()
    out = retire_solved(rows)
    assert not any(r["prompt"] == "solved" for r in out)
    assert any(r["prompt"] == "mid" for r in out)
    # thin task (one rollout) is kept: not enough evidence to retire
    assert any(r["prompt"] == "thin" for r in out)


def test_format_runs():
    assert "trainable" in format_curriculum(curriculum(make()))


def test_empty_and_ungraded():
    assert curriculum([])["n_tasks"] == 0
    ungraded = [{"prompt": "x"}, {"prompt": "x"}]  # no reward
    assert curriculum(ungraded)["n_tasks"] == 0
