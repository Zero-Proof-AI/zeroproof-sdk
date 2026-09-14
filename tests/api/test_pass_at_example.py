"""The pass-at-k example runs offline end to end and its numbers agree."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "pass-at-k"


def _load():
    path = EXAMPLE / "measure.py"
    spec = importlib.util.spec_from_file_location("pass_at_k_example_measure", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("pass_at_k_example_measure", module)
    spec.loader.exec_module(module)
    return module


def test_simulate_then_report(capsys):
    measure = _load()
    # concurrency=1 so seed 0 draws the same asks every run; the mixed
    # ask below is a property of that draw, not a guarantee of the agent.
    rows = measure.simulate_rows(asks=8, k=8, seed=0, concurrency=1)
    assert len(rows) == 64
    out = measure.report(rows)
    rates = out["pass_at"]
    assert rates["k"] == 8
    assert rates["n_groups"] == 8
    assert 0.0 < rates["pass_at_1"] < 1.0
    # The careless-repeat agent gives this draw a mixed ask, so the k-way
    # numbers exist and bracket pass@1.
    assert rates["pass_pow_k"] <= rates["pass_at_1"] <= rates["pass_at_k"]
    assert rates["headroom"] == pytest.approx(rates["pass_at_k"] - rates["pass_at_1"])
    assert out["n_mixed"] >= 1
    assert sum(out["histogram"].values()) == 8
    assert any("pass@1" in line for line in out["verdict"])

    assert measure.main([]) == 0
    printed = capsys.readouterr().out
    assert "pass@1" in printed and "headroom" in printed
