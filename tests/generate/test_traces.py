"""Trace mining, focused grids, pseudo-production splits, leakage gate.

Offline: hash embedder, template generator, scripted agent. No GPU.
"""
from __future__ import annotations

import zeroproof_simulations as zps
from tests.helpers import POLICY, TOOLS, scripted_agent
from zeroproof_simulations.traces import (dimensions_from_traces,
                                          drop_leaky_rows, leakage_report,
                                          mine_traces, simulate_from_traces,
                                          split_pseudo_production)


def _trace(prompt, tool, status, reward=None, **result_extra):
    result = {"status": status, **result_extra}
    return {
        "prompt": prompt,
        "steps": [{"tool": tool, "arguments": {"order_id": "ord_11"},
                   "result": result}],
        "final_text": "I could not find that order." if status != "ok"
        else "Order ord_11 is confirmed.",
        "reward": reward,
    }


TRACES = [
    _trace("where is order 4412", "lookup_order", "not_found", reward=0),
    _trace("check on my order pls", "lookup_order", "not_found", reward=0),
    _trace("refund order 9911 now", "create_refund", "timeout", reward=1),
    _trace("status of order 130", "lookup_order", "ok", reward=1),
    _trace("refund my double charge", "create_refund", "ok", reward=1),
]


def test_mine_traces_counts_tools_faults_and_flaws():
    mined = mine_traces(TRACES)
    assert mined["n"] == 5
    assert mined["tools"]["lookup_order"]["n"] == 3
    assert mined["tools"]["lookup_order"]["fault_n"] == 2
    assert mined["faults"] == {"not_found": 2, "timeout": 1}
    assert len(mined["flaw_rows"]) == 3  # two misses plus the timeout row
    assert len(mined["asks"]) == 5


def test_dimensions_focus_on_observed_behaviors():
    dims = dimensions_from_traces(TRACES, TOOLS, POLICY, broaden=False)
    base = zps.build_dimensions(TOOLS, POLICY)
    # Failing tool leads; the never-observed tool is dropped when narrow.
    assert dims["tool"][0] == "lookup_order"
    assert "get_refund_status" not in dims["tool"]
    # Observed faults, plus the clean value for contrast.
    assert dims["tool_condition"][0] == "success"
    assert "timeout" in dims["tool_condition"]
    assert "not_found" not in dims["tool_condition"]  # world axis, not a condition
    assert dims["world_state"][0] == "entity exists"
    assert "entity missing" in dims["world_state"]
    # Axes the traces cannot see stay whole.
    assert dims["stance"] == base["stance"]
    assert dims["rule"] == base["rule"]


def test_dimensions_broaden_keeps_unseen_tools_behind_observed():
    dims = dimensions_from_traces(TRACES, TOOLS, POLICY, broaden=True)
    assert dims["tool"][0] == "lookup_order"
    assert "get_refund_status" in dims["tool"]


def test_split_pseudo_production_holds_out_each_unique_flaw():
    production, remainder = split_pseudo_production(TRACES, fraction=0.4, seed=3)
    assert len(production) + len(remainder) == len(TRACES)
    prod_prompts = {row["prompt"] for row in production}
    rem_prompts = {row["prompt"] for row in remainder}
    assert not (prod_prompts & rem_prompts)
    # Both distinct flaw kinds are represented on the production side.
    from zeroproof_simulations.grading import trace_fault
    faults = {trace_fault(row) for row in production}
    assert "not_found" in faults
    assert "timeout" in faults
    again, _ = split_pseudo_production(TRACES, fraction=0.4, seed=3)
    assert [row["prompt"] for row in again] == [row["prompt"] for row in production]


def test_leakage_report_flags_copies_not_fresh_asks():
    generated = [
        {"prompt": "where is order 4412"},              # exact copy
        {"prompt": "Where is  ORDER 4412"},             # case/space copy
        {"prompt": "my package never arrived and support is not answering"},
    ]
    report = leakage_report(generated, TRACES, threshold=0.9)
    assert report["n_leaky"] == 2
    assert report["max_similarity"] == 1.0
    kept, drop = drop_leaky_rows(generated, TRACES, threshold=0.9)
    assert len(kept) == 1
    assert kept[0]["prompt"].startswith("my package")
    assert drop["n_dropped"] == 2


def test_simulate_from_traces_offline_end_to_end():
    data = simulate_from_traces(
        TRACES, scripted_agent, tools=TOOLS, policy=POLICY,
        mode="explore", budget=6, seed=0, grade=False, concurrency=6,
        simulator=False, time_budget=20,
        advanced={"per_round": 4, "mutate_failures": False})
    assert data.trajectories
    mining = data.search["trace_mining"]
    assert mining["n_traces"] == 5
    assert mining["faults"]["not_found"] == 2
    assert "focused_dimensions" in mining
    leak = data.search["trace_leakage"]
    assert leak["n_sources"] == 5
    trace_prompts = {" ".join(t["prompt"].lower().split()) for t in TRACES}
    for row in data.trajectories:
        assert " ".join(str(row["prompt"]).lower().split()) not in trace_prompts


def test_flaw_rows_returns_next_round_seeds():
    from zeroproof_simulations.traces import flaw_rows
    flawed = flaw_rows(TRACES)
    prompts = {row["prompt"] for row in flawed}
    assert len(flawed) == 3
    assert "status of order 130" not in prompts
    assert "where is order 4412" in prompts


def test_rl_retarget_lets_behavior_gap_lead():
    from zeroproof_simulations.scenarios import retarget_regions, scenario_regions
    regions = scenario_regions(TOOLS, POLICY, mode="rl")[:6]
    gappy = {regions[0]["id"]}

    def behavior_value(assignment):
        import json as _json
        key = _json.dumps(assignment, sort_keys=True, default=str)
        first = _json.dumps(regions[0]["assignment"], sort_keys=True, default=str)
        return 1.0 if key == first else 0.0

    default = [dict(r) for r in regions]
    retarget_regions(default, TOOLS, behavior_value=behavior_value)
    rl = [dict(r) for r in regions]
    retarget_regions(rl, TOOLS, behavior_value=behavior_value, mode="rl")
    spread_default = default[0]["weight"] - default[1]["weight"]
    spread_rl = rl[0]["weight"] - rl[1]["weight"]
    # Same behavior gap moves RL weights harder than explore weights.
    assert spread_rl > spread_default
    assert gappy  # regions resolved
