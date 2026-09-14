"""A policy edit keeps the task grid (#98).

The ``rule`` axis is the policy's own clauses. The grid is built in
layers, a rule-free block over the other axes plus one block per rule, so
adding, removing or rewording a clause changes that clause's rows only.
"""

from __future__ import annotations

import itertools

from zeroproof.simulations.generate import scenarios
from zeroproof.simulations.generate.scenarios import (
    RULE_FREE,
    _covering_assignments_uncached,
    _prefer_success,
    build_dimensions,
    scenario_regions,
)

TOOLS = [
    {
        "name": "run_sql",
        "description": "Run SQL.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "returns": {"type": "object", "properties": {"rows": {"type": "array"}}},
    }
]
POLICY = (
    "You are a data agent.\n1. Never guess column names.\n2. Only report a number run_sql returned."
)


def _ids(regions):
    return {r["id"] for r in regions}


def _rows_with(regions, rule):
    return {r["id"] for r in regions if r["assignment"].get("rule") == rule}


def test_appending_a_rule_keeps_every_existing_task():
    for mode in ("sft", "rl"):
        base = scenario_regions(TOOLS, POLICY, mode=mode)
        edited = scenario_regions(TOOLS, POLICY + "\n3. Be concise.", mode=mode)
        assert _ids(base) <= _ids(edited)
        assert len(edited) > len(base)
        # The new rows are the new rule's block and nothing else.
        assert _ids(edited) - _ids(base) == _rows_with(edited, "Be concise.")


def test_rewording_one_rule_changes_only_its_own_rows():
    base = scenario_regions(TOOLS, POLICY)
    edited = scenario_regions(TOOLS, POLICY.replace("Never guess", "Do not guess"))
    changed = _rows_with(base, "Never guess column names.")
    assert changed
    assert _ids(base) - _ids(edited) == changed
    assert _ids(edited) - _ids(base) == _rows_with(edited, "Do not guess column names.")


def test_same_policy_and_different_agent_still_pair():
    assert _ids(scenario_regions(TOOLS, POLICY)) == _ids(scenario_regions(TOOLS, POLICY))
    assert _ids(scenario_regions(TOOLS, POLICY)) == _ids(
        scenario_regions(TOOLS, POLICY, mode="rl", prefer_success=True)
    )


def test_layers_cover_every_pair_and_every_rule_meets_every_value():
    dims = build_dimensions(TOOLS, POLICY)
    rows = _covering_assignments_uncached(dims, 2)
    names = list(dims)
    seen = {tuple(sorted(row.items())) for row in rows}
    assert len(seen) == len(rows)  # no duplicate cells
    for left, right in itertools.combinations(names, 2):
        for a, b in itertools.product(dims[left], dims[right]):
            assert any(row[left] == a and row[right] == b for row in rows), (left, a, right, b)
    others = [name for name in names if name != "rule"]
    for rule in dims["rule"]:
        block = [row for row in rows if row["rule"] == rule]
        for axis in others:
            assert {row[axis] for row in block} == set(dims[axis]), (rule, axis)
    assert any(row["rule"] == RULE_FREE for row in rows)


def test_no_policy_is_the_rule_free_block_only():
    rows = _covering_assignments_uncached(build_dimensions(TOOLS, ""), 2)
    assert rows and all(row["rule"] == RULE_FREE for row in rows)


def test_grids_without_a_rule_axis_use_the_greedy_array():
    dims = {"tool": ["a", "b"], "stance": ["x", "y", "z"]}
    rows = _covering_assignments_uncached(dims, 2)
    assert rows == scenarios._greedy_covering(dims, 2)
    assert len(rows) == 6


def test_prefer_success_keeps_every_fault_kind_and_is_per_row():
    dims = build_dimensions(TOOLS, POLICY)
    rows = _covering_assignments_uncached(dims, 2)
    kept = _prefer_success(rows)
    kinds = {row["tool_condition"] for row in rows}
    assert {row["tool_condition"] for row in kept} == kinds
    faults = [row for row in kept if row["tool_condition"] != "success"]
    assert len(kinds) - 1 <= len(faults) <= len(kept) // 4
    # Growing the list does not change the verdict on rows already in it.
    more = rows + [dict(row, rule="Be concise.") for row in rows if row["rule"] != RULE_FREE]
    again = {tuple(sorted(row.items())) for row in _prefer_success(more)}
    assert {tuple(sorted(row.items())) for row in kept} <= again


def test_rule_free_cells_carry_no_rule_hint():
    from zeroproof.simulations.generate.diversity import sample_cell_tags

    free = sample_cell_tags(7, 0, "k", {"tool": "run_sql", "rule": RULE_FREE})
    ruled = sample_cell_tags(7, 0, "k", {"tool": "run_sql", "rule": "Never guess column names."})
    assert "rule" not in free
    assert ruled.get("rule")
