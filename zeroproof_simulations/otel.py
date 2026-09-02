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
    reward_keys = ("zeroproof.reward",)
    # Emitters like daisy set the conversation id on the root span only;
    # child spans carry just the trace id. First let any span's
    # conversation id claim its whole trace, then group.
    spans: list[tuple[str, str, int, dict, dict]] = []
    trace_conv: dict[str, str] = {}
    for span in _iter_spans(source):
        if not isinstance(span, dict):
            continue
        attrs = _attributes(span)
        trace = str(span.get("traceId") or span.get("trace_id") or "")
        conv = str(_first(attrs, _CONVERSATION_KEYS) or "")
        if conv and trace and trace not in trace_conv:
            trace_conv[trace] = conv
        start = int(span.get("startTimeUnixNano")
                    or span.get("start_time_unix_nano") or 0)
        if not start:
            # Platform-gate span dumps stamp milliseconds, not nanos.
            # Without a timestamp, step order silently becomes span
            # arrival order, which OTLP batch exporters do not preserve.
            ms = (span.get("startedMs") or span.get("started_ms")
                  or span.get("startTimeMs") or 0)
            start = int(ms) * 1_000_000
        spans.append((trace, conv, start, span, attrs))
    groups: dict[str, list[tuple[int, dict, dict]]] = {}
    for trace, conv, start, span, attrs in spans:
        key = conv or trace_conv.get(trace) or trace
        groups.setdefault(key, []).append((start, span, attrs))

    rows: list[dict] = []
    for conv, entries in groups.items():
        entries.sort(key=lambda item: item[0])
        prompt = ""
        final_text = ""
        steps: list[dict] = []
        reward = None
        model_version = None
        seen_users: set[str] = set()
        model_generic = None
        for _, span, attrs in entries:
            # The explicit zeroproof key wins across ALL spans; a generic
            # model name on an earlier span must not freeze the choice.
            mv = _first(attrs, ("zeroproof.model_version",))
            if mv and model_version is None:
                model_version = str(mv)
            if model_generic is None:
                generic = _first(attrs, ("gen_ai.request.model",))
                if generic:
                    model_generic = str(generic)
            raw_reward = _first(attrs, reward_keys)
            if raw_reward is not None and reward is None:
                try:
                    value = float(raw_reward)
                except (TypeError, ValueError):
                    value = None
                if value is not None and 0.0 <= value <= 1.0:
                    reward = int(value) if value in (0.0, 1.0) else value
            tool_args = _first(attrs, _TOOL_IN_KEYS)
            if tool_args is not None:
                name = str(_first(attrs, _TOOL_NAME_KEYS)
                           or span.get("name") or "tool")
                result = _parse(_first(attrs, _TOOL_OUT_KEYS))
                # Emitters like daisy flag failures via gen_ai.tool.status
                # rather than inside the result payload. Surface that as a
                # canonical failure result so fault mining and trace aiming
                # see the error instead of a plain-looking output.
                tool_status = str(_first(attrs, ("gen_ai.tool.status",))
                                  or "").lower()
                if tool_status in {"error", "failed", "failure"} and not (
                        isinstance(result, dict) and result.get("status")):
                    error = ""
                    for event in span.get("events") or []:
                        for attr in (event or {}).get("attributes") or []:
                            if attr.get("key") == "exception.message":
                                error = str(_otlp_value(attr.get("value")))
                                break
                    wrapped: dict = {"status": "error", "output": result}
                    if error:
                        wrapped["error"] = error
                    result = wrapped
                steps.append({
                    "tool": name,
                    "arguments": _parse(tool_args),
                    "result": result,
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
            row = {"prompt": prompt, "steps": steps,
                   "final_text": final_text,
                   "conversation_id": conv}
            # zeroproof.reward span attributes are preserved when present
            # and in [0, 1]; rows without them stay ungraded, first-class.
            if reward is not None:
                row["reward"] = reward
            # Which weights produced this trace: zeroproof.model_version
            # wins (adapters share the base model name), the request
            # model is the fallback. Rounds of the continual loop are
            # indistinguishable without it.
            if model_version or model_generic:
                row["model_version"] = model_version or model_generic
            rows.append(row)
    return rows


__all__ = ["rows_from_otel"]
