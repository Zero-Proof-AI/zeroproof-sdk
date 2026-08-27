"""End-to-end ingestion of a daisy-shaped OTLP bundle.

The fixture reproduces daisy's emitter byte-conventions exactly
(daisy@src/otel.ts): root invoke_agent span with raw-text input and JSON
output messages, chat spans with wire-format tool_calls, execute_tool
spans flagging failures via gen_ai.tool.status plus an exception event,
string epoch-nano timestamps, doubleValue numerics. Swap DAISY_BUNDLE for
a recorded real bundle when one is available; assertions hold either way.
"""
from __future__ import annotations

import json

from tests.helpers import scripted_agent
from zeroproof_simulations import rows_from_otel
from zeroproof_simulations.traces import (load_traces, simulate_from_traces,
                                          trace_report)

DAISY_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
]
DAISY_POLICY = ("You are daisy, a coding agent. Only act on files the user "
                "names. Report command failures honestly. Never invent "
                "file contents.")


def _attr(key, value):
    if isinstance(value, bool):
        v = {"boolValue": value}
    elif isinstance(value, (int, float)):
        v = {"doubleValue": value}
    else:
        v = {"stringValue": str(value)}
    return {"key": key, "value": v}


def _span(trace, sid, parent, name, start_ms, attrs, events=(),
          status_code=1):
    return {"traceId": trace, "spanId": sid, "parentSpanId": parent,
            "name": name, "kind": 1,
            "startTimeUnixNano": str(int(start_ms * 1e6)),
            "endTimeUnixNano": str(int((start_ms + 5) * 1e6)),
            "attributes": attrs, "events": list(events),
            "status": {"code": status_code}}


def _daisy_bundle():
    trace = "ab" * 8
    spans = [
        _span(trace, "s-root", "", "invoke_agent daisy", 1000, [
            _attr("gen_ai.operation.name", "invoke_agent"),
            _attr("gen_ai.agent.name", "daisy"),
            _attr("daisy.turn_id", "turn-7"),
            _attr("gen_ai.conversation.id", "daisy-sess-1"),
            _attr("gen_ai.input.messages",
                  "fix the failing test in utils.py"),
            _attr("gen_ai.output.messages", json.dumps(
                [{"role": "assistant",
                  "content": "The command failed, so I stopped."}])),
        ]),
        _span(trace, "s-llm", "s-root", "chat qwen", 1010, [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.request.model", "qwen"),
            _attr("gen_ai.provider.name", "openai-compatible"),
            _attr("gen_ai.input.messages", json.dumps(
                [{"role": "user",
                  "content": "fix the failing test in utils.py"}])),
            _attr("gen_ai.output.messages", json.dumps(
                [{"role": "assistant", "content": "",
                  "tool_calls": [{"id": "c1", "function": {
                      "name": "read_file",
                      "arguments": json.dumps({"path": "utils.py"})}}]}])),
            _attr("gen_ai.usage.input_tokens", 120.0),
        ]),
        _span(trace, "s-t1", "s-root", "execute_tool read_file", 1020, [
            _attr("gen_ai.operation.name", "execute_tool"),
            _attr("gen_ai.tool.name", "read_file"),
            _attr("gen_ai.tool.call.arguments",
                  json.dumps({"path": "utils.py"})),
            _attr("gen_ai.tool.call.result", "def add(a, b): return a - b"),
            _attr("gen_ai.tool.status", "success"),
        ]),
        _span(trace, "s-t2", "s-root", "execute_tool run_command", 1030, [
            _attr("gen_ai.operation.name", "execute_tool"),
            _attr("gen_ai.tool.name", "run_command"),
            _attr("gen_ai.tool.call.arguments",
                  json.dumps({"command": "pytest -q"})),
            _attr("gen_ai.tool.call.result", ""),
            _attr("gen_ai.tool.status", "error"),
        ], events=[{"name": "exception", "time": "0", "attributes": [
            _attr("exception.message", "pytest exited 1: 3 failed")]}],
            status_code=2),
    ]
    return {"resourceSpans": [{
        "resource": {"attributes": [
            _attr("service.name", "daisy"),
            _attr("deployment.environment", "demo")]},
        "scopeSpans": [{"scope": {"name": "daisy"}, "spans": spans}]}]}


def test_daisy_bundle_converts_with_error_status_surfaced():
    rows = rows_from_otel(_daisy_bundle())
    assert len(rows) == 1
    row = rows[0]
    assert row["prompt"] == "fix the failing test in utils.py"
    assert row["final_text"] == "The command failed, so I stopped."
    tools = {s["tool"]: s for s in row["steps"] if "tool" in s}
    ok = tools["read_file"]["result"]
    assert ok == "def add(a, b): return a - b", \
        "success results must pass through untouched"
    failed = tools["run_command"]["result"]
    assert failed["status"] == "error"
    assert "pytest exited 1" in failed["error"]
    assert "reward" not in row


def test_daisy_failure_reaches_mining_and_report():
    rows = load_traces(rows_from_otel(_daisy_bundle()))
    report = trace_report(rows, tools=DAISY_TOOLS, policy=DAISY_POLICY)
    assert report["tools_observed"]["run_command"]["fault_n"] >= 1, \
        "the flagged failure must count as a fault, not a plain result"
    assert any("error" in k for k in report["faults_observed"]), \
        report["faults_observed"]


def test_daisy_traces_drive_simulation_end_to_end():
    rows = load_traces(rows_from_otel(_daisy_bundle()))
    data = simulate_from_traces(
        rows, scripted_agent, tools=DAISY_TOOLS, policy=DAISY_POLICY,
        mode="explore", budget=6, seed=0, grade=False, concurrency=6,
        simulator=False, time_budget=20,
        advanced={"per_round": 4, "mutate_failures": False})
    assert data.trajectories
    assert data.search["trace_mining"]["n_traces"] == len(rows)
    source = {" ".join(r["prompt"].lower().split()) for r in rows}
    for generated in data.trajectories:
        assert " ".join(str(generated["prompt"]).lower().split()) not in source


def test_platform_gate_spans_order_by_started_ms():
    """The platform gate dumps spans with flat attributes and startedMs
    only (no nano timestamps) — the shape of real patchsmith/daisy
    traces pulled 2026-08-27. Step order must come from time, not span
    arrival order, because OTLP batch exporters reorder spans."""
    base = 1787841777000
    tool_seq = [("list_files", "{}", '{"status": "ok", "files": ["a.py"]}'),
                ("read_file", '{"path": "a.py"}', '{"status": "ok"}'),
                ("write_file", '{"path": "a.py"}', '{"status": "ok"}'),
                ("run_tests", "{}", '{"status": "ok", "exit": 0}')]
    spans = [{
        "traceId": "t1", "spanId": "root", "name": "agent patchsmith",
        "startedMs": base,
        "attributes": {
            "gen_ai.agent.name": "patchsmith",
            "gen_ai.input.messages": json.dumps(
                [{"role": "user", "content": "fix the failing tests"}]),
            "gen_ai.output.messages": json.dumps(
                [{"role": "assistant", "content": "done, tests pass"}]),
            "zeroproof.reward": 1,
        }}]
    for i, (tool, args, result) in enumerate(tool_seq):
        spans.append({
            "traceId": "t1", "spanId": f"s{i}", "parentSpanId": "root",
            "name": f"tool {tool}", "startedMs": base + 10 * (i + 1),
            "attributes": {"gen_ai.tool.name": tool,
                           "gen_ai.tool.call.arguments": args,
                           "gen_ai.tool.call.result": result}})
    scrambled = [spans[3], spans[1], spans[4], spans[0], spans[2]]
    rows = rows_from_otel({"spans": scrambled})
    assert len(rows) == 1
    row = rows[0]
    assert row["prompt"] == "fix the failing tests"
    assert [s["tool"] for s in row["steps"] if "tool" in s] == \
        [t for t, _, _ in tool_seq]
    assert row["reward"] == 1
    assert row["final_text"] == "done, tests pass"
