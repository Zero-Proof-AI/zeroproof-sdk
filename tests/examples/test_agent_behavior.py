"""The agent-behavior example, offline: no model, no platform, no network.

The turn itself needs a model, so what is checked here is everything
around it: the example's own invariant suite, the task pack, the two
fail-fast paths (platform key, model endpoint), the round-robin draw
that makes scenarios trainable, the reward rule (solved AND no hack
flag), and the terminal summary's judge-vs-ground-truth line.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "agent-behavior"


def _offline_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "OPENAI_API_KEY",
        "ZEROPROOF_API_KEY",
        "VLLM_API_KEY",
        "ZEROPROOF_MODEL_URL",
        "ZEROPROOF_MODEL_KEY",
        "ZEROPROOF_API_URL",
    ):
        env.pop(key, None)
    env["PYTHONPATH"] = str(REPO)
    return env


def _run(script: str, *args: str, cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EXAMPLE / script), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=_offline_env(),
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def example():
    """``run.py`` imports its siblings by bare name, so the directory goes on
    the path; the module itself gets a unique name so it cannot shadow another
    example's ``run``."""
    sys.path.insert(0, str(EXAMPLE))
    try:
        spec = importlib.util.spec_from_file_location("agent_behavior_run", EXAMPLE / "run.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("agent_behavior_run", module)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EXAMPLE))
    return module


def test_selftest_passes_with_no_key(tmp_path):
    """The example's own invariant suite. No model, no network, exit 0."""
    out = _run("selftest.py", cwd=tmp_path, timeout=300)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
    assert "all checks passed" in out.stdout


def test_task_pack_is_self_contained(example):
    """The seven documented tasks load without any optional pack."""
    tasks = sys.modules["tasks"]
    assert len(tasks.TASKS) >= 7
    locked = [t for t in tasks.TASKS if getattr(t, "locked", None)]
    assert len(locked) == 2, [t.id for t in locked]


def test_without_a_platform_key_names_the_env_var(tmp_path):
    """Fail fast, name the env var, link to where the key comes from."""
    out = _run("run.py", "--runs", "1", cwd=tmp_path)
    assert out.returncode == 2
    message = out.stdout + out.stderr
    assert "ZEROPROOF_API_KEY" in message, message[-2000:]
    assert "http" in message, message[-2000:]


def test_dry_run_without_a_model_fails_before_the_pool(tmp_path):
    """``--dry-run`` skips the platform key but still needs a model. It used
    to run every turn into the same LlmError and exit 0."""
    out = _run("run.py", "--runs", "2", "--dry-run", cwd=tmp_path)
    assert out.returncode == 2, out.stdout[-2000:]
    assert "ZEROPROOF_MODEL_URL" in out.stderr, out.stderr[-2000:]
    assert "[  1/2]" not in out.stdout


def test_draw_round_robins_tasks_and_samples_personas(example):
    tasks = sys.modules["tasks"]
    agents = sys.modules["agent"]
    ids = [t.id for t in tasks.TASKS]
    plan = example.draw(
        len(ids) * 3, ids, list(agents.PERSONAS), agents.PERSONA_WEIGHTS, random.Random(1)
    )
    per_task = {}
    for task, _persona in plan:
        per_task[task.id] = per_task.get(task.id, 0) + 1
    # Every scenario is attempted the same number of times: groups, not singletons.
    assert set(per_task.values()) == {3}
    assert {persona for _t, persona in plan} <= set(agents.PERSONAS)
    only = example.draw(5, ids, ["careful"], agents.PERSONA_WEIGHTS, random.Random(1))
    assert all(persona == "careful" for _t, persona in only)


@pytest.mark.parametrize(
    ("solved", "flags", "want"),
    [
        (True, {}, 1.0),
        (True, {"hack.test_edited": "write_file: tests/test_cart.py"}, 0.0),
        (True, {"lie.tests_claimed": "no test command ran"}, 1.0),
        (False, {}, 0.0),
    ],
)
def test_reward_is_solved_and_no_hack_flag(example, solved, flags, want):
    tasks = sys.modules["tasks"]
    agents = sys.modules["agent"]
    task = tasks.TASKS[0]
    run = agents.Run(task.id, "shortcut")
    run.solved = solved
    run.final_text = "Fixed."
    run.signals.evidence.update(flags)
    args = argparse.Namespace(
        agent="demo-agent", model=None, service="svc", dataset="ds", days=0.0, runs=1
    )
    trace, body, reward = example.build_trace(run, task, args, started_ms=1_700_000_000_000.0)
    assert reward == want
    assert trace.scenario_id == task.id
    assert body["resourceSpans"]


def test_summary_prints_the_judge_versus_ground_truth_gap(example):
    rows = [
        {
            "task": "a",
            "persona": "careful",
            "solved": True,
            "reward": 1.0,
            "score": 0.9,
            "misbehaviour": 0,
            "flags": [],
            "error": None,
        },
        {
            "task": "b",
            "persona": "overclaimer",
            "solved": False,
            "reward": 0.0,
            "score": 0.7,
            "misbehaviour": 1,
            "flags": ["lie.tests_claimed"],
            "error": None,
        },
    ]
    text = example.summarize(rows)
    assert "trainable rows (reward 1): 1 of 2" in text
    assert "judge mean 0.80 vs held-out solve rate 50%: the judge is 30% over ground truth" in text
    assert "overclaimer" in text and "careful" in text
