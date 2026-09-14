"""OTLP GenAI spans assemble into trajectory rows the SDK can consume."""

from __future__ import annotations

import json

import zeroproof.simulations as zps
from zeroproof.simulations.ingest.otel import rows_from_otel


def _attr(key, value):
    return {"key": key, "value": {"stringValue": value}}


def _span(name, start, attrs, trace="t1"):
    return {"name": name, "traceId": trace, "startTimeUnixNano": start, "attributes": attrs}


BATCH = {
    "resourceSpans": [
        {
            "scopeSpans": [
                {
                    "spans": [
                        _span(
                            "chat gpt-4",
                            100,
                            [
                                _attr("gen_ai.conversation.id", "conv-9"),
                                _attr(
                                    "gen_ai.input.messages",
                                    json.dumps(
                                        [{"role": "user", "content": "where is order 4412"}]
                                    ),
                                ),
                                _attr(
                                    "gen_ai.output.messages",
                                    json.dumps([{"role": "assistant", "content": "Let me check."}]),
                                ),
                            ],
                        ),
                        _span(
                            "execute_tool lookup_order",
                            200,
                            [
                                _attr("gen_ai.conversation.id", "conv-9"),
                                _attr("gen_ai.tool.name", "lookup_order"),
                                _attr("gen_ai.tool.call.arguments", '{"order_id": "4412"}'),
                                _attr("gen_ai.tool.call.result", '{"status": "not_found"}'),
                            ],
                        ),
                        _span(
                            "chat gpt-4",
                            300,
                            [
                                _attr("gen_ai.conversation.id", "conv-9"),
                                _attr(
                                    "gen_ai.input.messages",
                                    json.dumps(
                                        [
                                            {"role": "user", "content": "where is order 4412"},
                                            {"role": "user", "content": "try 4413 then"},
                                        ]
                                    ),
                                ),
                                _attr(
                                    "gen_ai.output.messages",
                                    json.dumps(
                                        [
                                            {
                                                "role": "assistant",
                                                "content": "I could not find order 4412.",
                                            }
                                        ]
                                    ),
                                ),
                            ],
                        ),
                        # A second conversation using the OpenInference-style dialect.
                        _span(
                            "llm",
                            150,
                            [
                                _attr("session.id", "conv-10"),
                                _attr(
                                    "llm.input_messages",
                                    json.dumps(
                                        [{"role": "user", "content": "cancel my subscription"}]
                                    ),
                                ),
                                _attr(
                                    "llm.output_messages",
                                    json.dumps(
                                        [{"role": "assistant", "content": "Done, cancelled."}]
                                    ),
                                ),
                            ],
                            trace="t2",
                        ),
                    ]
                }
            ]
        }
    ]
}


def test_rows_from_otel_assembles_conversations():
    rows = rows_from_otel(BATCH)
    assert len(rows) == 2
    by_id = {r["conversation_id"]: r for r in rows}
    row = by_id["conv-9"]
    assert row["prompt"] == "where is order 4412"
    assert row["final_text"] == "I could not find order 4412."
    tools = [s for s in row["steps"] if s.get("tool")]
    assert tools[0]["tool"] == "lookup_order"
    assert tools[0]["arguments"] == {"order_id": "4412"}
    assert tools[0]["result"] == {"status": "not_found"}
    users = [s for s in row["steps"] if s.get("user")]
    assert users and users[0]["user"] == "try 4413 then"
    other = by_id["conv-10"]
    assert other["prompt"] == "cancel my subscription"
    assert other["final_text"] == "Done, cancelled."


def test_otel_rows_feed_trace_mining_directly():
    rows = rows_from_otel(BATCH)
    mined = zps.mine_traces(rows)
    assert mined["n"] == 2
    assert mined["faults"] == {"not_found": 1}
    assert mined["tools"]["lookup_order"]["fault_n"] == 1


def _int_attr(key, value):
    # OTLP JSON encodes int64 attribute values as strings.
    return {"key": key, "value": {"intValue": str(value)}}


def test_rows_from_otel_sums_gen_ai_usage_into_the_row():
    spans = [
        _span(
            "chat",
            100,
            [
                _attr("gen_ai.conversation.id", "c-usage"),
                _attr("gen_ai.input.messages", json.dumps([{"role": "user", "content": "hi"}])),
                _int_attr("gen_ai.usage.input_tokens", 120),
                _int_attr("gen_ai.usage.output_tokens", 8),
            ],
        ),
        _span(
            "execute_tool lookup",
            200,
            [
                _attr("gen_ai.conversation.id", "c-usage"),
                _attr("gen_ai.tool.name", "lookup"),
                _attr("gen_ai.tool.call.arguments", "{}"),
                _attr("gen_ai.tool.call.result", "{}"),
            ],
        ),
        _span(
            "chat",
            300,
            [
                _attr("gen_ai.conversation.id", "c-usage"),
                _attr(
                    "gen_ai.output.messages",
                    json.dumps([{"role": "assistant", "content": "Found it."}]),
                ),
                _int_attr("gen_ai.usage.input_tokens", 200),
                _int_attr("gen_ai.usage.output_tokens", 12),
            ],
        ),
        # OpenInference dialect, separate conversation.
        _span(
            "llm",
            100,
            [
                _attr("session.id", "c-oi"),
                _attr("llm.input_messages", json.dumps([{"role": "user", "content": "yo"}])),
                _attr("llm.output_messages", json.dumps([{"role": "assistant", "content": "hey"}])),
                _int_attr("llm.token_count.prompt", 5),
                _int_attr("llm.token_count.completion", 2),
            ],
            trace="t-oi",
        ),
        # No usage emitted: no usage key on the row, not zeros.
        _span(
            "llm",
            100,
            [
                _attr("session.id", "c-none"),
                _attr("llm.input_messages", json.dumps([{"role": "user", "content": "q"}])),
                _attr("llm.output_messages", json.dumps([{"role": "assistant", "content": "a"}])),
            ],
            trace="t-none",
        ),
    ]
    by_id = {r["conversation_id"]: r for r in rows_from_otel(spans)}
    assert by_id["c-usage"]["usage"] == {"input_tokens": 320, "output_tokens": 20}
    assert by_id["c-oi"]["usage"] == {"input_tokens": 5, "output_tokens": 2}
    assert "usage" not in by_id["c-none"]
