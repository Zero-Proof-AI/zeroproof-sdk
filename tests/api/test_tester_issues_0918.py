"""Tester run, 2026-09-18: `.rows` had two shapes across one flow (#344).

`simulate()` hands back a `SimulationData` whose `.rows` is callable;
`run_judge()` / `grade()` / `evaluate()` hand back a `ScoredData` whose
`.rows` was a plain list. `hasattr(x, "rows")` is true for both, so a
guard written to be defensive picked the wrong branch on the judging
path and died with `TypeError: 'list' object is not callable` a line
later than the mistake.
"""

from __future__ import annotations

import whileai.simulations as wai
from whileai.simulations.data import RowList
from whileai.simulations.score.judging import run_judge

ROWS = [
    {"rollout_id": "r1", "prompt": "refund order 4412", "final_text": "Refunded.", "steps": []},
    {"rollout_id": "r2", "prompt": "refund order 4413", "final_text": "No.", "steps": []},
]


def _judge(row: dict) -> dict:
    return {"reward": 1.0 if "Refunded" in row["final_text"] else 0.0, "reason": "ok"}


def test_scored_rows_answers_to_both_spellings():
    scored = run_judge([dict(r) for r in ROWS], _judge)
    # The list spelling is what every existing caller uses; it must not move.
    assert isinstance(scored.rows, list) and len(scored.rows) == 2
    # The call spelling is the one that used to raise TypeError.
    assert scored.rows() is scored.rows
    assert [r["rollout_id"] for r in scored.rows()] == ["r1", "r2"]
    assert [r["reward"] for r in scored.rows] == [1.0, 0.0]


def test_the_reported_hasattr_guard_now_picks_a_working_branch():
    # Verbatim shape of the guard in the report: hasattr is true for both
    # types, so the `.rows()` branch is taken for a ScoredData as well.
    scored = run_judge([dict(r) for r in ROWS], _judge)
    picked = scored.rows() if hasattr(scored, "rows") else list(scored)
    assert [r["rollout_id"] for r in picked] == ["r1", "r2"]
    # list(scored) is the workaround the report used; it still agrees.
    assert list(scored) == list(picked)


def test_simulate_and_run_judge_hand_back_the_same_rows_shape():
    data = wai.simulate(
        lambda messages: {"steps": [], "final_text": "done"},
        tools=[],
        system_prompt="be brief",
        budget=4,
        simulator=False,
        seed=1,
    )
    scored = run_judge([dict(r) for r in ROWS], _judge)
    for holder in (data.rows, scored.rows):
        assert isinstance(holder, RowList)
        assert holder() is holder
        assert isinstance(holder(), list)


def test_rows_are_the_scored_trajectories_not_exported_rows():
    # The shapes match; the contents must not. ScoredData.rows are the
    # judged trajectories, so `reward` is on them -- swapping in exported
    # rows here would silently change what grade()/evaluate() return.
    scored = run_judge([dict(r) for r in ROWS], _judge)
    assert scored.rows[0]["reward"] == 1.0
    assert scored.rows[0]["judge_status"] == "ok"
    assert scored.rows[0]["final_text"] == "Refunded."
    # export_row scrubs and renames; these rows have not been through it.
    assert "schema_version" not in scored.rows[0]
