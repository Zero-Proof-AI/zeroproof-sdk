"""optimize() reads mode and tools off a SimulationData; ScoredData feeds it as rows."""

from __future__ import annotations

import zeroproof.simulations as zps
from zeroproof.simulations import SimulationData, optimize
from zeroproof.simulations.score.judging import ScoredData


def _row(prompt, reward, final, idx):
    return {
        "prompt": prompt,
        "scenario_id": prompt,
        "rollout_index": idx,
        "reward": reward,
        "judge_status": "ok",
        "final_text": final,
        "steps": [
            {"tool": "lookup_order", "arguments": {"order_id": "o1"}, "result": {"status": "ok"}}
        ],
        "messages": [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "lookup_order", "arguments": {"order_id": "o1"}}],
            },
            {"role": "tool", "name": "lookup_order", "content": '{"status": "ok"}'},
            {"role": "assistant", "content": final},
        ],
    }


def _rows():
    rows = []
    for prompt in ("refund order 41", "refund order 42"):
        for i, reward in enumerate((1, 0, 1, 0)):  # mixed: in the RL band
            rows.append(_row(prompt, reward, f"Reply {i} about {prompt}.", i))
    for i in range(4):  # unanimous: dead gradient for RL, fine for SFT
        rows.append(_row("status of order 43", 1, f"Order 43 is fine, reply {i}.", i))
    return rows


def test_rl_mode_comes_from_the_data_and_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = SimulationData(trajectories=_rows(), mode="rl")
    picked, report = optimize(data)
    assert report["mode"] == "rl"
    assert "path" not in report and not list(tmp_path.iterdir())
    prompts = {r["prompt"] for r in picked}
    assert prompts == {"refund order 41", "refund order 42"}
    assert len(picked) == 8, "whole mixed groups, never split"


def test_sft_mode_comes_from_the_data_and_selects_one_demonstration_per_prompt():
    data = SimulationData(trajectories=_rows(), mode="sft")
    picked, report = optimize(data)
    assert report["mode"] == "sft"
    assert all(r["reward"] == 1 for r in picked)
    assert sorted(r["prompt"] for r in picked) == [
        "refund order 41",
        "refund order 42",
        "status of order 43",
    ]


def test_an_explicit_mode_overrides_the_data():
    data = SimulationData(trajectories=_rows(), mode="sft")
    _, report = optimize(data, mode="rl")
    assert report["mode"] == "rl"


def test_scored_data_feeds_optimize_as_its_rows():
    rows = _rows()
    scored = ScoredData(rows, run_id="r1", source="grade", judge_name="j")
    picked_scored, report_scored = optimize(scored, mode="rl")
    picked_rows, report_rows = zps.optimize(rows, mode="rl")
    assert [r["prompt"] for r in picked_scored] == [r["prompt"] for r in picked_rows]
    assert report_scored["n_selected"] == report_rows["n_selected"]
