"""The README's parameter tables state the defaults the code resolves.

Every row of the form ``| `name` | `literal` | ...`` whose name maps to a
RunConfig field is compared against a default resolve_run_config(). A
wrong number in the table is a test failure, not a support ticket.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from zeroproof_simulations.run.config import resolve_run_config

README = Path(__file__).resolve().parents[2] / "README.md"
ROW = re.compile(r"^\| `([a-z_]+)` \| `([^`]*)` \|", re.M)

# README knob -> how to read it off a RunConfig
FIELDS = {
    "budget": lambda c: c.budget,
    "time_budget": lambda c: c.time_budget,
    "fault_rate": lambda c: c.fault_rate,
    "concurrency": lambda c: c.concurrency,
    "embedder": lambda c: c.embedder,
    "seed": lambda c: c.seed,
    "avg_turns": lambda c: c.avg_turns,
    "mode": lambda c: c.topo["mode"],
    "until": lambda c: c.until_key,
    "grade": lambda c: c.grade,
    "llm_grade": lambda c: c.llm_grade,
    "stop_grace": lambda c: c.stop_grace_s,
    "reproducible": lambda c: c.reproducible,
}


def _documented_defaults() -> dict[str, object]:
    out: dict[str, object] = {}
    for name, raw in ROW.findall(README.read_text(encoding="utf-8")):
        if name not in FIELDS or name in out:
            continue
        try:
            out[name] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            continue
    return out


def test_readme_parameter_tables_match_resolved_defaults():
    documented = _documented_defaults()
    assert len(documented) >= 8, documented
    cfg = resolve_run_config(None, tools=[{"type": "function", "function": {
        "name": "noop", "parameters": {"type": "object", "properties": {}}}}],
        system_prompt="An agent.")
    mismatched = {name: (value, FIELDS[name](cfg))
                  for name, value in documented.items()
                  if FIELDS[name](cfg) != value}
    assert not mismatched, f"README says / code does: {mismatched}"
