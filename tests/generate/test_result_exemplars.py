"""Trace-grounded result exemplars: real tool results mined from traces
become shape templates for invented results. Agent specs declare tool
inputs but almost never result shapes; production traces carry the real
payloads, so a run with traces grounds its sandbox on them. A run
without traces is unchanged."""
from __future__ import annotations

import json

import zeroproof_simulations as zps
from tests.helpers import POLICY, TOOLS
from zeroproof_simulations.sandbox import MockEnvironment
from zeroproof_simulations.traces import (exemplar_result_shapes,
                                          mine_result_exemplars)

TRACES = [
    {"prompt": "where is order 4412",
     "steps": [{"tool": "lookup_order", "arguments": {"order_id": "4412"},
                "result": {"status": "ok", "order_id": "4412",
                           "eta": "2026-09-02", "carrier": "UPS"}}],
     "final_text": "Arrives September 2 via UPS."},
    {"prompt": "refund order 9911 now",
     "steps": [{"tool": "create_refund", "arguments": {"order_id": "9911"},
                "result": {"status": "timeout"}}],
     "final_text": "The refund request timed out."},
    {"prompt": "check order 130",
     # Platform ingest spelling: result arrives under `output`, as a string.
     "steps": [{"tool": "lookup_order", "arguments": {"order_id": "130"},
                "output": json.dumps({"orders": [{"order_id": "130",
                                                  "eta": "2026-09-04"}]})}],
     "final_text": "Order 130 arrives September 4."},
]


def test_exemplars_are_mined_shape_diverse_and_filtered():
    exemplars = mine_result_exemplars(TRACES + [
        # Same shape as the first lookup result: adds nothing, skipped.
        {"steps": [{"tool": "lookup_order", "arguments": {},
                    "result": {"status": "ok", "order_id": "88",
                               "eta": "2026-09-09", "carrier": "DHL"}}]},
        # Empty, error-only, and status-only payloads never qualify.
        {"steps": [{"tool": "lookup_order", "arguments": {}, "result": {}},
                   {"tool": "lookup_order", "arguments": {},
                    "result": {"status": "ok"}},
                   {"tool": "get_refund_status", "arguments": {},
                    "result": {"status": "permission_denied"}}]},
    ])
    # create_refund only ever timed out; get_refund_status only errored.
    assert set(exemplars) == {"lookup_order"}
    assert len(exemplars["lookup_order"]) == 2
    assert exemplars["lookup_order"][0]["carrier"] == "UPS"
    assert "orders" in exemplars["lookup_order"][1]


def test_per_tool_cap_and_serialized_size_cap():
    rows = [{"steps": [{"tool": "t", "arguments": {}, "result": {f"k{i}": i}}]}
            for i in range(5)]
    assert len(mine_result_exemplars(rows)["t"]) == 3
    big = mine_result_exemplars([{"steps": [
        {"tool": "t", "arguments": {}, "result": {"text": "x" * 5000}}]}])
    assert len(json.dumps(big["t"][0])) <= 500


def test_exemplar_templates_pattern_mock_results():
    shapes = exemplar_result_shapes(mine_result_exemplars(TRACES))
    env = MockEnvironment(TOOLS, world_state="entity exists",
                          result_shapes=shapes)
    out = env.call("lookup_order", {"order_id": "ord_777"})
    data = out.get("data") or {}
    # The invented result carries the observed fields, not a generic stub.
    assert {"eta", "carrier"} <= set(data)


def _capture_hosted(seen):
    def fake_hosted(tools, system="", *, result_shapes=None, **kwargs):
        seen["shapes"] = result_shapes
        return lambda m: {"steps": [], "final_text": "ok"}
    return fake_hosted


def test_simulate_seeds_runner_shapes_from_traces(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr("zeroproof_simulations.hosted_model",
                        _capture_hosted(seen))
    data = zps.simulate(
        None, tools=TOOLS, policy=POLICY, traces=TRACES, budget=4, seed=0,
        grade=False, simulator=False, concurrency=2,
        advanced={"per_round": 4, "mutate_failures": False})
    # The runner's sandbox got the trace-grounded template.
    assert seen["shapes"]["lookup_order"]["carrier"] == "UPS"
    # Disclosed: exemplar counts per tool in the trace mining report.
    mining = data.search["trace_mining"]
    assert mining["result_exemplars"] == {"lookup_order": 2}


def test_no_trace_run_gets_no_shapes_and_no_report(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr("zeroproof_simulations.hosted_model",
                        _capture_hosted(seen))
    data = zps.simulate(
        None, tools=TOOLS, policy=POLICY, budget=4, seed=0,
        grade=False, simulator=False, concurrency=2,
        advanced={"per_round": 4, "mutate_failures": False})
    assert seen["shapes"] == {}
    assert "trace_mining" not in data.search
