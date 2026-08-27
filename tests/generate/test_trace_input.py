"""Canonical trace input: load_traces normalization, the pre-simulation
trace report with explicit aiming, and the product-critical integration
test — raw UNGRADED traces to simulated data with no judge involved."""
from __future__ import annotations

import json

from tests.helpers import POLICY, TOOLS, scripted_agent
from zeroproof_simulations.traces import (format_trace_report, load_traces,
                                          simulate_from_traces, trace_report)


def test_load_traces_normalizes_alternate_keys():
    rows = load_traces([
        {"question": "where is order 4412",
         "tool_trace": [{"tool": "lookup_order",
                         "arguments": {"order_id": "4412"},
                         "result": {"status": "not_found"}}],
         "final": "I could not find it.", "reward": "0", "extra": "kept"},
    ])
    row = rows[0]
    assert row["prompt"] == "where is order 4412"
    assert row["steps"][0]["tool"] == "lookup_order"
    assert row["final_text"] == "I could not find it."
    assert row["reward"] == 0
    assert row["extra"] == "kept"


def test_load_traces_converts_message_only_rollouts():
    rows = load_traces([
        {"messages": [
            {"role": "user", "content": "refund order 9911"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {
                 "name": "create_refund",
                 "arguments": json.dumps({"order_id": "9911"})}}]},
            {"role": "tool",
             "content": json.dumps({"status": "timeout"})},
            {"role": "assistant", "content": "That timed out."},
        ]},
    ])
    row = rows[0]
    assert row["prompt"] == "refund order 9911"
    tool_step = next(s for s in row["steps"] if "tool" in s)
    assert tool_step["arguments"] == {"order_id": "9911"}
    assert tool_step["result"] == {"status": "timeout"}
    assert row["final_text"] == "That timed out."
    assert "reward" not in row


def test_reward_stays_optional_and_coerces_cleanly():
    rows = load_traces([
        {"prompt": "a", "steps": [], "final_text": "x"},
        {"prompt": "b", "steps": [], "final_text": "x", "reward": True},
        {"prompt": "c", "steps": [], "final_text": "x", "reward": 1.0},
        {"prompt": "d", "steps": [], "final_text": "x", "reward": "maybe"},
        {"prompt": "e", "steps": [], "final_text": "x", "reward": 0.7},
    ])
    assert "reward" not in rows[0]
    assert rows[1]["reward"] == 1
    assert rows[2]["reward"] == 1
    assert "reward" not in rows[3]
    assert "reward" not in rows[4]


def test_junk_rows_are_dropped():
    rows = load_traces(["not a dict", {}, {"prompt": "real ask"}])
    assert len(rows) == 1
    assert rows[0]["prompt"] == "real ask"


UNGRADED = [
    {"prompt": "where is order 4412",
     "steps": [{"tool": "lookup_order", "arguments": {"order_id": "4412"},
                "result": {"status": "not_found"}}],
     "final_text": "I could not find that order."},
    {"prompt": "refund order 9911 now",
     "steps": [{"tool": "create_refund", "arguments": {"order_id": "9911"},
                "result": {"status": "timeout"}}],
     "final_text": "The refund request timed out."},
    {"prompt": "status of order 130",
     "steps": [{"tool": "lookup_order", "arguments": {"order_id": "130"},
                "result": {"status": "ok"}}],
     "final_text": "Order 130 is confirmed."},
]


def test_trace_report_counts_and_explicit_aiming():
    report = trace_report(UNGRADED, tools=TOOLS, policy=POLICY)
    assert report["traces"] == 3
    assert report["unique_prompts"] == 3
    assert report["graded"] == 0
    assert report["ungraded"] == 3
    assert report["tools_observed"]["lookup_order"]["n"] == 2
    assert "timeout" in report["faults_observed"]
    assert "emphasis" in report
    text = format_trace_report(report)
    assert "extra generation weight goes to" in text
    assert "ungraded 3" in text


def test_trace_report_without_tools_defers_emphasis():
    report = trace_report(UNGRADED)
    assert "emphasis" not in report
    assert "simulate time" in format_trace_report(report)


def test_raw_ungraded_traces_drive_simulation_without_a_judge():
    # The product claim: traces in, simulated data out, no reward needed.
    assert all("reward" not in r for r in load_traces(UNGRADED))
    data = simulate_from_traces(
        load_traces(UNGRADED), scripted_agent, tools=TOOLS, policy=POLICY,
        mode="explore", budget=6, seed=0, grade=False, concurrency=6,
        simulator=False, time_budget=20,
        advanced={"per_round": 4, "mutate_failures": False})
    assert data.trajectories
    mining = data.search["trace_mining"]
    assert mining["n_traces"] == 3
    assert mining["faults"]["timeout"] == 1
    leak = data.search["trace_leakage"]
    assert leak["n_sources"] == 3
    source_prompts = {" ".join(t["prompt"].lower().split())
                      for t in UNGRADED}
    for row in data.trajectories:
        normalized = " ".join(str(row["prompt"]).lower().split())
        assert normalized not in source_prompts


# --- regression pins from the retired adversarial battery (real bugs) ----

def test_parallel_tool_results_attach_by_name_then_fifo():
    rows = load_traces([{"messages": [
        {"role": "user", "content": "check a and b"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_a", "arguments": "{}"}},
            {"function": {"name": "get_b", "arguments": "{}"}}]},
        {"role": "tool", "name": "get_b", "content": '{"status": "error"}'},
        {"role": "tool", "content": '{"status": "ok"}'},
    ]}])
    steps = {s["tool"]: s for s in rows[0]["steps"] if "tool" in s}
    assert steps["get_b"]["result"] == {"status": "error"}
    assert steps["get_a"]["result"] == {"status": "ok"}


def test_empty_steps_list_never_masks_other_sources():
    rows = load_traces([{"prompt": "p", "steps": [], "tool_trace": [
        {"tool": "t", "arguments": {}, "result": {"status": "ok"}}]}])
    assert rows[0]["steps"][0]["tool"] == "t"


def test_orphan_tool_result_becomes_its_own_step():
    rows = load_traces([{"messages": [
        {"role": "user", "content": "hi"},
        {"role": "tool", "name": "late_tool",
         "content": '{"status": "timeout"}'}]}])
    assert any(s.get("tool") == "late_tool" for s in rows[0]["steps"])


def test_qwen_labeled_rows_are_advisory_not_ungraded():
    rep = trace_report([{"prompt": "p", "qwen_reward": 0, "steps": [
        {"tool": "t", "arguments": {}, "result": {"status": "ok"}}]}])
    assert rep["ungraded"] == 0
    assert rep["advisory_labels"] == 1
