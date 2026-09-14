"""The pass-at-k example's CLI paths, offline.

tests/api/test_pass_at_example.py covers ``simulate_rows`` -> ``report``
in process. This file covers what that leaves out: the seed actually
deciding the draw at the default concurrency, the ``graded.jsonl`` path,
the headline line carrying the interval, and the below-min-k note.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import zeroproof.simulations as zps

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "pass-at-k"


def _offline_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("OPENAI_API_KEY", "ZEROPROOF_API_KEY", "VLLM_API_KEY"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(REPO)
    return env


def _cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EXAMPLE / "measure.py"), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=_offline_env(),
        timeout=300,
    )


@pytest.fixture(scope="module")
def measure():
    spec = importlib.util.spec_from_file_location("pass_at_k_cli_measure", EXAMPLE / "measure.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("pass_at_k_cli_measure", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rows(measure):
    return measure.simulate_rows(asks=6, k=4, seed=0, concurrency=4)


def test_seed_decides_the_draw_at_default_concurrency(measure, rows):
    """``reproducible=True``: same seed, same asks, same numbers, threads or not."""
    again = measure.simulate_rows(asks=6, k=4, seed=0, concurrency=4)
    assert measure.report(again)["pass_at"] == measure.report(rows)["pass_at"]
    assert sorted(r["prompt"] for r in again) == sorted(r["prompt"] for r in rows)


def test_headline_carries_the_interval(measure, rows):
    out = measure.report(rows)
    assert out["summary"].startswith("pass@1 ")
    assert out["pass_at"]["ci95"] is not None
    lo, hi = out["pass_at"]["ci95"]
    assert f"[{lo:.2f}..{hi:.2f}]" in out["summary"]
    assert out["summary"] == str(zps.pass_at(rows))


def test_below_min_k_withholds_the_k_way_numbers(measure):
    thin = measure.simulate_rows(asks=3, k=2, seed=0, concurrency=4)
    out = measure.report(thin)
    assert out["pass_at"]["pass_at_1"] is not None
    assert out["pass_at"]["pass_at_k"] is None
    assert "n/a" in out["summary"]
    assert out["verdict"][-1] == out["pass_at"]["note"]
    assert "repeats" in out["verdict"][-1]


def test_cli_simulates_and_prints(tmp_path):
    out = _cli("--asks", "6", "--k", "4", cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    lines = out.stdout.splitlines()
    assert lines[0].startswith("pass@1 ") and "headroom" in lines[0]
    assert [line.split()[0] for line in lines[1:4]] == ["never", "sometimes", "always"]
    assert lines[4].startswith("- production sees pass@1 = ")


def test_cli_reads_a_graded_jsonl(measure, rows, tmp_path):
    path = tmp_path / "graded.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    out = _cli(str(path), cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.splitlines()[0] == measure.report(rows)["summary"]
