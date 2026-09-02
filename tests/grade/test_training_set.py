"""The recommended-dataset path: graded rows in, trainer-ready subset out."""
import json

import pytest

from zeroproof_simulations import SimulationData


def _row(prompt: str, reward, *, tool: str = "add_expense") -> dict:
    return {
        "prompt": prompt,
        "steps": [{"tool": tool, "arguments": {"item": prompt, "amount": 1.0},
                   "result": {"status": "ok"}}],
        "final_text": f"Recorded {prompt}.",
        "reward": reward,
        "arm": "grid",
    }


def _graded_data() -> SimulationData:
    rows = [
        _row("coffee", 1),
        _row("lunch", 1, tool="list_expenses"),
        _row("taxi", 0),
        _row("coffee", 1),  # duplicate prompt never ships twice
    ]
    return SimulationData(trajectories=rows)


def test_select_keeps_only_diverse_passes():
    data = _graded_data()
    selected = data.select(target=10)
    prompts = [r["prompt"] for r in selected]
    assert "taxi" not in prompts
    assert prompts.count("coffee") == 1
    assert "lunch" in prompts
    report = data.search["selection"]
    assert report["n_selected"] == len(selected)
    assert report["n_not_pass"] >= 1


def test_select_refuses_ungraded_rows():
    data = SimulationData(trajectories=[
        {"prompt": "coffee", "steps": [], "final_text": "ok", "arm": "grid"}])
    with pytest.raises(RuntimeError, match="graded"):
        data.select()
