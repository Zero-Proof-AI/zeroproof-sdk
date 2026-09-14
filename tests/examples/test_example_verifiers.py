"""The verifiers example runs with no key and shows the numbers it claims,
and the answer key it scores against never reaches a training row.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.score.judging import run_judge
from zeroproof.simulations.verify import All, MathEqual, Regex

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "examples" / "verifiers" / "run.py"


def _env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("OPENAI_API_KEY", "ZEROPROOF_API_KEY", "VLLM_API_KEY"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(cwd),
        env=_env(),
        timeout=300,
    )


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("verifiers_example", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_prints_every_section_with_the_documented_counts(tmp_path):
    out = _run(cwd=tmp_path)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    text = out.stdout
    for section, tally in (
        ("== math: MathEqual", "2/3 passed"),
        ("== math + format: answer+format", "1/3 passed"),
        ("== code execution: CodeExec", "1/2 passed"),
        ("== json schema: JSONSchema", "1/2 passed"),
    ):
        head = text.index(section)
        assert tally in text[head : head + 400], f"{section} did not report {tally}"
    assert "== optimize(mode='rl'): 3 rows from 2 prompt groups" in text
    assert "0 carry the answer key" in text


def test_run_takes_no_arguments(tmp_path):
    out = _run("--rows", "3", cwd=tmp_path)
    assert out.returncode == 2
    assert "usage:" in out.stderr


def test_rows_carry_the_gold_in_privileged_only(example):
    for row in example.MATH_ROWS:
        assert set(row) == {"prompt", "final_text", "privileged", "scenario_id", "rollout_index"}
        assert "reference" in row["privileged"]
    for row in example.CODE_ROWS:
        assert "tests" in row["privileged"]


def test_verifier_rewards_match_the_script(example):
    scored = run_judge(example.MATH_ROWS, MathEqual(), source="grade")
    assert [r["reward"] for r in scored.rows] == [1, 0, 1]
    gate = All([MathEqual(), Regex(r"</think>")], name="answer+format")
    assert [r["reward"] for r in run_judge(example.MATH_ROWS, gate).rows] == [1, 0, 0]


def test_answer_key_stops_at_the_training_export(example):
    scored = run_judge(example.MATH_ROWS, MathEqual(), source="grade")
    rows, report = zps.optimize(scored, mode="rl")
    assert report["groups_selected"] == 2
    assert all("privileged" in r for r in rows), "optimize keeps SDK rows whole"
    trainer_rows = zps.training_rows(rows)
    assert len(trainer_rows) == len(rows) == 3
    assert not any("privileged" in r for r in trainer_rows)
    assert all("messages" in r and "loss_mask" in r for r in trainer_rows)
