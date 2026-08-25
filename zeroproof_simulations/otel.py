"""OTel GenAI spans to trajectory rows: ``simulate(traces=...)`` input.

The platform's trace gate stores raw OTLP batches; agents instrument with
the GenAI semantic conventions (or one of their common dialects). This
reads either an OTLP/HTTP JSON batch or a flat list of span dicts, groups
spans into conversations, and emits the row shape the whole SDK speaks:
``{prompt, steps, final_text}``.

Attribute fallbacks mirror the platform's trace inspector exactly, so
both ends of the wire read the same fields.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

_MESSAGE_KEYS = ("gen_ai.input.messages", "gen_ai.prompt",
                 "llm.input_messages")
_OUTPUT_KEYS = ("gen_ai.output.messages", "gen_ai.completion",
                "llm.output_messages")
_TOOL_IN_KEYS = ("gen_ai.tool.call.arguments", "gen_ai.tool.input",
                 "tool.input")
_TOOL_OUT_KEYS = ("gen_ai.tool.call.result", "gen_ai.tool.output",
                  "tool.output")
_TOOL_NAME_KEYS = ("gen_ai.tool.name", "tool.name")
_CONVERSATION_KEYS = ("gen_ai.conversation.id", "conversation.id",
                      "session.id", "gen_ai.session.id", "thread.id")


def _otlp_value(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
            if key in value:
                return value[key]
        if "arrayValue" in value:
            return [_otlp_value(v) for v in
                    (value["arrayValue"].get("values") or [])]
    return value


def _attributes(span: dict) -> dict:
    raw = span.get("attributes")
    if isinstance(raw, dict):
        return raw
    out = {}
    for item in raw or []:
        if isinstance(item, dict) and "key" in item:
            out[item["key"]] = _otlp_value(item.get("value"))
    return out


def _first(attrs: dict, keys: Sequence[str]) -> Any:
    for key in keys:
        if attrs.get(key) not in (None, ""):
            return attrs[key]
    return None


def _parse(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "[{":
            try:
                return json.loads(text)
            except ValueError:
                return value
    return value


def _iter_spans(source: Any):
    if isinstance(source, dict) and "resourceSpans" in source:
        for resource in source.get("resourceSpans") or []:
            for scope in (resource.get("scopeSpans")
                          or resource.get("instrumentationLibrarySpans")
                          or []):
                yield from scope.get("spans") or []
        return
    if isinstance(source, dict) and "spans" in source:
        yield from source["spans"] or []
        return
    yield from source or []


def _message_texts(value: Any, role: str) -> list[str]:
    parsed = _parse(value)
    out: list[str] = []
    if isinstance(parsed, list):
        for message in parsed:
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "").lower() != role:
                continue
            content = message.get("content")
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text") or part.get("content") or "")
                    if isinstance(part, dict) else str(part)
                    for part in content)
            if content:
                out.append(str(content))
    elif isinstance(parsed, str) and parsed and role == "user":
        out.append(parsed)
    return out


def rows_from_otel(source: Any) -> list[dict]:
    """Conversations from an OTLP JSON batch or an iterable of spans.

    One row per conversation id (trace id when no conversation id is
    emitted): the first user message becomes ``prompt``, tool spans in
    time order become ``steps``, later user turns become user steps, and
    the last assistant output becomes ``final_text``.
    """
    groups: dict[str, list[tuple[int, dict, dict]]] = {}
    for span in _iter_spans(source):
        if not isinstance(span, dict):
            continue
        attrs = _attributes(span)
        conv = str(_first(attrs, _CONVERSATION_KEYS)
                   or span.get("traceId") or span.get("trace_id") or "")
        start = int(span.get("startTimeUnixNano")
                    or span.get("start_time_unix_nano") or 0)
        groups.setdefault(conv, []).append((start, span, attrs))

    rows: list[dict] = []
    for conv, entries in groups.items():
        entries.sort(key=lambda item: item[0])
        prompt = ""
        final_text = ""
        steps: list[dict] = []
        seen_users: set[str] = set()
        for _, span, attrs in entries:
            tool_args = _first(attrs, _TOOL_IN_KEYS)
            if tool_args is not None:
                name = str(_first(attrs, _TOOL_NAME_KEYS)
                           or span.get("name") or "tool")
                steps.append({
                    "tool": name,
                    "arguments": _parse(tool_args),
                    "result": _parse(_first(attrs, _TOOL_OUT_KEYS)),
                })
                continue
            for text in _message_texts(_first(attrs, _MESSAGE_KEYS), "user"):
                key = " ".join(text.lower().split())
                if key in seen_users:
                    continue
                seen_users.add(key)
                if not prompt:
                    prompt = text
                else:
                    steps.append({"user": text})
            outputs = _message_texts(_first(attrs, _OUTPUT_KEYS), "assistant")
            if outputs:
                final_text = outputs[-1]
        if prompt or steps or final_text:
            rows.append({"prompt": prompt, "steps": steps,
                         "final_text": final_text,
                         "conversation_id": conv})
    return rows


__all__ = ["rows_from_otel"]
