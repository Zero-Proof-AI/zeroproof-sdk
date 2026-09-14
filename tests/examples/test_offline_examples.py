"""Every checked-in example still imports and runs with no key on the machine.

An example is the first thing a coding agent copies, so a broken one is
worse than a missing one. These checks are cheap: a ``--help`` that exits 0
proves every import in the module resolved, which is where examples rot (a
helper file that was never committed, a renamed parameter, a moved import).
"""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"

# Scripts that parse arguments. ``--help`` runs the whole module body up to
# the parser, so it catches import-time breakage without a real run.
CLI_EXAMPLES = [
    "agent-behavior/run.py",
    "bring-your-own-agent/run.py",
    "hosted-loop/run.py",
    "character/from_model_spec.py",
    "character/measure.py",
    "character/run.py",
    "hugging-face/roundtrip.py",
    "identity/generate.py",
    "pass-at-k/measure.py",
    "prime-intellect-rl/diagnose.py",
    "prime-intellect-rl/export_prompts.py",
    "prime-intellect-rl/generate.py",
    "reward-hacking/run.py",
    "schema/migrate.py",
    "schema/project.py",
    "verifiers/run.py",
]

# Need the ``modal`` client, which is not a dev dependency. Compiled, not run.
NEEDS_MODAL = {
    "dpo/train_modal.py",
    "grpo/train_modal.py",
    "identity/eval_modal.py",
    "identity/train_modal.py",
}


def _offline_env() -> dict[str, str]:
    """The environment of an agent that has not configured anything yet."""
    env = dict(os.environ)
    for key in (
        "OPENAI_API_KEY",
        "ZEROPROOF_API_KEY",
        "VLLM_API_KEY",
        "ZEROPROOF_MODEL_URL",
        "ZEROPROOF_API_URL",
        "HF_TOKEN",
    ):
        env.pop(key, None)
    env["PYTHONPATH"] = str(REPO)
    # A saved `zeroproof login` credential would count as a key too.
    env["ZEROPROOF_HOME"] = str(REPO / "tests" / "fixtures" / "no-such-home")
    return env


def _run(script: Path, *args: str, cwd: Path, timeout: int = 120):
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=_offline_env(),
        timeout=timeout,
    )


def test_every_example_directory_is_tracked_by_git():
    """``.gitignore`` has ``examples/*``; a new example is invisible until unignored."""
    tracked = subprocess.run(
        ["git", "ls-files", "examples"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=True,
    ).stdout.split()
    tracked_dirs = {Path(p).parts[1] for p in tracked if len(Path(p).parts) > 1}
    on_disk = {
        d.name for d in EXAMPLES.iterdir() if d.is_dir() and not d.name.startswith((".", "__"))
    }
    assert on_disk - tracked_dirs == set(), (
        "example directory on disk but not in git; add a `!examples/<name>/` line to .gitignore"
    )


def test_every_example_module_compiles():
    scripts = sorted(EXAMPLES.glob("*/*.py"))
    assert scripts, "no example scripts found"
    for script in scripts:
        py_compile.compile(str(script), doraise=True)
    listed = set(CLI_EXAMPLES) | NEEDS_MODAL
    on_disk = {p.relative_to(EXAMPLES).as_posix() for p in scripts}
    unlisted = sorted(on_disk - listed)
    # Library modules imported by a CLI script are covered through it.
    for rel in unlisted:
        assert rel.split("/")[0] in {p.split("/")[0] for p in listed}, (
            f"{rel}: new example directory with no CLI entry point in this test"
        )


@pytest.mark.parametrize("rel", CLI_EXAMPLES)
def test_cli_example_imports_and_answers_help(rel, tmp_path):
    out = _run(EXAMPLES / rel, "--help", cwd=tmp_path)
    assert out.returncode == 0, f"{rel} --help failed:\n{out.stderr[-2000:]}"
    assert "usage:" in out.stdout


def test_modal_examples_are_listed_not_forgotten():
    for rel in NEEDS_MODAL:
        assert (EXAMPLES / rel).exists(), f"{rel} is listed here but gone from disk"


# The agent-behavior example's own checks live in test_agent_behavior.py.


def test_hosted_loop_without_a_key_names_the_env_var(tmp_path):
    out = _run(EXAMPLES / "hosted-loop/run.py", cwd=tmp_path)
    assert out.returncode != 0
    message = out.stdout + out.stderr
    assert "ZEROPROOF_API_KEY" in message, message[-2000:]
    assert "http" in message, message[-2000:]
