"""stage lineage: stamp, resolve, and catch eval/train leaks."""

from __future__ import annotations

import pytest

from zeroproof.simulations.score.stage import (
    STAGES,
    StageError,
    format_stages,
    stage_of,
    stage_report,
    stamp_stage,
)


def test_stamp_and_validate():
    rows = stamp_stage([{"prompt": "a"}], "sft")
    assert rows[0]["stage"] == "sft"
    with pytest.raises(StageError):
        stamp_stage([{"prompt": "a"}], "nonsense")


def test_stamp_is_non_mutating():
    orig = [{"prompt": "a"}]
    stamp_stage(orig, "rl")
    assert "stage" not in orig[0]


def test_stage_of_infers_eval():
    assert stage_of({"stage": "rl"}) == "rl"
    assert stage_of({"lineage": {"source": "eval"}}) == "eval"
    assert stage_of({"purpose": "eval"}) == "eval"
    assert stage_of({"prompt": "x"}) is None


def test_report_counts_and_tasks():
    rows = (
        stamp_stage([{"scenario_id": "t1"}, {"scenario_id": "t2"}], "sft")
        + stamp_stage([{"scenario_id": "t3"}], "rl")
        + stamp_stage([{"scenario_id": "t9"}], "eval")
    )
    rep = stage_report(rows)
    assert rep["counts"] == {"sft": 2, "rl": 1, "eval": 1}
    assert rep["tasks_per_stage"]["sft"] == 2
    assert rep["n_leaks"] == 0


def test_cross_stage_leak_detected():
    rows = stamp_stage([{"scenario_id": "shared"}], "rl") + stamp_stage(
        [{"scenario_id": "shared"}, {"scenario_id": "clean"}], "eval"
    )
    rep = stage_report(rows)
    assert rep["eval_train_leaks"] == ["shared"]
    assert any("both eval and training" in w for w in rep["warnings"])


def test_unstamped_counted_and_warned():
    rep = stage_report([{"prompt": "x"}, {"prompt": "y"}])
    assert rep["n_unstamped"] == 2
    assert any("no stage" in w for w in rep["warnings"])


def test_format_and_stages_constant():
    assert set(STAGES) == {"sft", "rm", "rl", "eval", "mid"}
    rep = stage_report(stamp_stage([{"scenario_id": "t"}], "sft"))
    assert "sft" in format_stages(rep)
