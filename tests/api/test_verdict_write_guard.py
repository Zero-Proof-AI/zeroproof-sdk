"""Direct verdict-key writes are frozen at today's count per module.

``Judgment`` owns the verdict keys. Until the grading paths route through
``attach``, the existing direct writes stay where they are, and no module
may gain a new one. ``schema.py`` is exempt: ``to_row`` is the sanctioned
projection. Lower a number here when you remove a write; never raise one.
"""
from __future__ import annotations

import re
from pathlib import Path

import zeroproof.simulations as zps
from zeroproof.simulations.schema import VERDICT_KEYS

PKG = Path(zps.__file__).resolve().parent
PATTERN = re.compile(r'\w+\["(%s)"\]\s*=' % "|".join(VERDICT_KEYS))
EXEMPT = {"schema.py"}

BASELINE = {
    "data.py": 10,
    "ingest/otel.py": 1,
    "ingest/traces.py": 1,
    "run/engine.py": 3,
    "score/grade_llm.py": 4,
    "score/judging.py": 7,
    "score/llm_judge.py": 2,
    "score/preflight.py": 2,
}


def _counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG).as_posix()
        if rel in EXEMPT:
            continue
        n = len(PATTERN.findall(path.read_text(encoding="utf-8")))
        if n:
            counts[rel] = n
    return counts


def test_no_module_gains_a_direct_verdict_write():
    counts = _counts()
    grown = {rel: (n, BASELINE.get(rel, 0)) for rel, n in counts.items()
             if n > BASELINE.get(rel, 0)}
    assert not grown, (
        "new direct writes to a verdict key (route them through attach): "
        f"{grown}")


def test_baseline_is_not_stale():
    counts = _counts()
    shrunk = {rel: (counts.get(rel, 0), n) for rel, n in BASELINE.items()
              if counts.get(rel, 0) < n}
    assert not shrunk, f"writes were removed; lower BASELINE: {shrunk}"
