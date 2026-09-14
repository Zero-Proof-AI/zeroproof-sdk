"""The bring-your-own-agent example runs offline end to end.

The contract part produces the repeats it asks for (so the scripted
agent's careless third repeat actually happens), the broken part blames
the agent and names the first error, and the eval part shows the
provenance guard with the numbers the README quotes.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.score.judging import evaluate, run_judge

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "bring-your-own-agent"


def _offline_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("OPENAI_API_KEY", "ZEROPROOF_API_KEY", "VLLM_API_KEY"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(REPO)
    return env


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("byoa_example_run", EXAMPLE / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("byoa_example_run", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def contract_data(example):
    return example.part_contract()


def test_contract_run_carries_every_repeat(contract_data):
    rows = contract_data.trajectories
    assert len(rows) == 16
    assert contract_data.stopped_because == "budget"
    by_ask: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        by_ask[row["scenario_id"]].add(row["rollout_index"])
    # 4 asks x 4 repeats, all up front: repeat_policy="fixed" is doing its job.
    assert len(by_ask) == 4
    assert all(indices == {0, 1, 2, 3} for indices in by_ask.values()), dict(by_ask)
    careless = [row for row in rows if len(row["steps"]) == 1]
    assert len(careless) == 4
    assert all(row["rollout_index"] == 2 for row in careless)
    assert all(row["steps"][0]["tool"] == "restart_service" for row in careless)


def test_broken_agent_is_reported_as_the_agent(example, capsys):
    data = zps.simulate(
        example.agent_that_raises,
        tools=example.TOOLS,
        system_prompt=example.POLICY,
        simulator=False,
        budget=8,
        seed=0,
    )
    assert data.trajectories == []
    assert data.stopped_because == "agent_failed"
    assert data.search["agent_errors"] >= 1
    assert data.search["first_agent_error"] == "RuntimeError: model endpoint returned 502"
    assert "agent_errors" in data.degraded

    example.part_broken()
    out = capsys.readouterr().out
    assert out.count("stopped_because='agent_failed'") == 2
    assert "RuntimeError: model endpoint returned 502" in out
    assert "TypeError: agent returned keys ['content', 'role']" in out


def test_eval_rows_are_counted_not_dropped(example, contract_data, capsys):
    rows = contract_data.trajectories
    train = run_judge(rows, example.runbook_judge)
    held_out = evaluate(rows, example.runbook_judge)
    # Same judge, same rows, same number: 12 of 16 read the runbook.
    assert train.pass_at.pass_at_1 == held_out.pass_at.pass_at_1 == 0.75
    assert all(row["lineage"]["source"] == "eval" for row in held_out)
    assert all(row["lineage"]["source"] != "eval" for row in train)

    kept_train, report_train = train.select_for_rl()
    kept_eval, report_eval = held_out.select_for_rl()
    # Nothing is dropped on either side; the count is the guard.
    assert len(kept_train) == len(kept_eval) == 8
    assert report_train["eval_sourced"] == 0
    assert report_eval["eval_sourced"] == 8
    assert any("evaluate()" in w for w in report_eval["hygiene_warnings"])

    example.part_eval(contract_data)
    out = capsys.readouterr().out
    assert "pass@1 on both: 0.75" in out
    assert "run_judge: selected=8 eval_sourced=0  clean" in out
    assert "evaluate : selected=8 eval_sourced=8" in out


def test_cli_runs_all_three_parts_offline(tmp_path):
    out = subprocess.run(
        [sys.executable, str(EXAMPLE / "run.py")],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=_offline_env(),
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    for header in ("== contract:", "== broken:", "== eval:"):
        assert header in out.stdout
    assert "4 of 16 rows skipped the runbook" in out.stdout
    assert "eval_sourced=8" in out.stdout
