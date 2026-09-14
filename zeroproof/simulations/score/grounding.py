"""Argument grounding: did every tool argument come from the conversation?

A policy trained to call a tool learns to call it before it learns when
not to. On the refund environment both GRPO and DPO learned to invent an
order id on a quarter of the prompts that gave none; the headline pass@1
rose while that happened, and only a per-category split caught it. This
marker catches the same thing on any agent, with no categories: for each
tool call in a rollout, every string argument must appear in what the
agent had to work with, the prompt, the user and system turns, and the
results of earlier tool calls. An argument that appears nowhere was made
up.

``argument_grounding(row)`` is 1.0 when every string argument of every
call is grounded (rows with no calls count as grounded), else 0.0, so the
marker reads higher-is-better like the rest and ``delta_report(...,
must_not_regress=["argument_grounding"])`` fails a run that learned to
invent. ``ungrounded_arguments(row)`` names the offenders for a reviewer.
``mark_grounding(rows)`` stamps the marker. Structured, not a regex over
prose: tool calls are read from ``steps``, from assistant ``tool_calls``
in ``messages``, and from ``<tool_call>`` blocks in the reply text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

MARKER = "argument_grounding"

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_WS = re.compile(r"\s+")


def _norm(text: Any) -> str:
    return _WS.sub(" ", str(text or "")).strip().lower()


def _parse_json_obj(raw: Any) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _calls_from_steps(row: dict) -> list[tuple[str, dict, int]]:
    """(tool, arguments, index) for each tool step, in order."""
    out: list[tuple[str, dict, int]] = []
    steps = row.get("steps") or row.get("tool_trace") or []
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or not step.get("tool"):
            continue
        args = step.get("arguments")
        if args is None:
            args = step.get("input")
        args = _parse_json_obj(args) or {}
        out.append((str(step["tool"]), args, i))
    return out


def _calls_from_messages(row: dict) -> list[tuple[str, dict, int]]:
    out: list[tuple[str, dict, int]] = []
    for i, message in enumerate(row.get("messages") or []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            call = call if isinstance(call, dict) else {}
            fn = call["function"] if isinstance(call.get("function"), dict) else call
            out.append((str(fn.get("name") or ""), _parse_json_obj(fn.get("arguments")) or {}, i))
        for m in _TOOL_CALL_BLOCK.finditer(str(message.get("content") or "")):
            obj = _parse_json_obj(m.group(1))
            if obj and obj.get("name"):
                out.append((str(obj["name"]), _parse_json_obj(obj.get("arguments")) or {}, i))
    return out


def _calls_from_text(row: dict) -> list[tuple[str, dict, int]]:
    out: list[tuple[str, dict, int]] = []
    for m in _TOOL_CALL_BLOCK.finditer(str(row.get("final_text") or "")):
        obj = _parse_json_obj(m.group(1))
        if obj and obj.get("name"):
            out.append((str(obj["name"]), _parse_json_obj(obj.get("arguments")) or {}, 0))
    return out


def tool_calls_of(row: dict) -> list[dict[str, Any]]:
    """Every tool call the rollout made, as ``{"tool", "arguments", "context"}``
    where ``context`` is the text the call could have drawn on: the prompt,
    the user and system turns before it, and earlier tool results."""
    prompt = _norm(row.get("prompt"))
    system = _norm(row.get("system_prompt"))
    calls: list[dict[str, Any]] = []
    step_calls = _calls_from_steps(row)
    if step_calls:
        steps = row.get("steps") or row.get("tool_trace") or []
        for tool, args, i in step_calls:
            earlier = [
                _norm(s.get("result") if s.get("result") is not None else s.get("output"))
                for s in steps[:i]
                if isinstance(s, dict)
            ]
            calls.append(
                {"tool": tool, "arguments": args, "context": " ".join([prompt, system, *earlier])}
            )
        return calls
    message_calls = _calls_from_messages(row)
    if message_calls:
        messages = row.get("messages") or []
        for tool, args, i in message_calls:
            earlier = [
                _norm(m.get("content"))
                for m in messages[:i]
                if isinstance(m, dict) and m.get("role") in ("user", "system", "tool")
            ]
            calls.append(
                {"tool": tool, "arguments": args, "context": " ".join([prompt, system, *earlier])}
            )
        return calls
    for tool, args, _ in _calls_from_text(row):
        user_turns = [
            _norm(m.get("content"))
            for m in row.get("messages") or []
            if isinstance(m, dict) and m.get("role") in ("user", "system")
        ]
        calls.append(
            {"tool": tool, "arguments": args, "context": " ".join([prompt, system, *user_turns])}
        )
    return calls


def _string_values(args: Any, *, prefix: str = "") -> list[tuple[str, str]]:
    """(key path, value) for every string inside ``args``, nested."""
    out: list[tuple[str, str]] = []
    if isinstance(args, dict):
        for k, v in args.items():
            out.extend(_string_values(v, prefix=f"{prefix}{k}."))
    elif isinstance(args, list):
        for i, v in enumerate(args):
            out.extend(_string_values(v, prefix=f"{prefix}{i}."))
    elif isinstance(args, str):
        out.append((prefix.rstrip("."), args))
    return out


def ungrounded_arguments(
    row: dict,
    *,
    ignore_keys: Sequence[str] = (),
    allow: Sequence[str] = (),
    min_len: int = 3,
) -> list[dict[str, str]]:
    """The string arguments of the rollout's tool calls that appear nowhere
    in the context the call could draw on. Each entry is ``{"tool", "key",
    "value"}``. ``ignore_keys`` skips argument names that are free text by
    design (a note, a message body); ``allow`` lists values that are legal
    without appearing in the conversation (an enum, a default); strings
    shorter than ``min_len`` are skipped."""
    skip = {k.lower() for k in ignore_keys}
    allowed = {_norm(v) for v in allow}
    out: list[dict[str, str]] = []
    for call in tool_calls_of(row):
        for key, value in _string_values(call["arguments"]):
            leaf = key.split(".")[-1].lower()
            v = _norm(value)
            if leaf in skip or len(v) < min_len or v in allowed:
                continue
            if v in call["context"]:
                continue
            # A value the prompt spells with different separators (ORD 4017 vs
            # ORD-4017) is still the prompt's value.
            loose = re.sub(r"[\s_\-]+", "", v)
            if loose and loose in re.sub(r"[\s_\-]+", "", call["context"]):
                continue
            out.append({"tool": call["tool"], "key": key, "value": value})
    return out


def argument_grounding(row: dict, **kwargs: Any) -> float:
    """1.0 when every string argument of every tool call is grounded in the
    conversation (a rollout with no calls is grounded), else 0.0. Keyword
    arguments are those of ``ungrounded_arguments``."""
    return 0.0 if ungrounded_arguments(row, **kwargs) else 1.0


def mark_grounding(rows: Sequence[dict], **kwargs: Any) -> list[dict]:
    """Copies of ``rows`` with ``markers["argument_grounding"]`` stamped, so
    ``marker_summary``, ``delta_report`` and the run page read it."""
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            out.append(row)
            continue
        marks = dict(row.get("markers") or {})
        marks[MARKER] = argument_grounding(row, **kwargs)
        out.append({**row, "markers": marks})
    return out


def grounding_report(rows: Sequence[dict], **kwargs: Any) -> dict[str, Any]:
    """Over a row set: the share of rollouts with every argument grounded,
    the share with any call at all, and the most common invented values
    by tool and key, for a reviewer to look at."""
    n = 0
    grounded = 0
    with_calls = 0
    offenders: dict[str, int] = {}
    examples: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        n += 1
        calls = tool_calls_of(row)
        if calls:
            with_calls += 1
        bad = ungrounded_arguments(row, **kwargs)
        if not bad:
            grounded += 1
        for b in bad:
            k = f"{b['tool']}.{b['key']}"
            offenders[k] = offenders.get(k, 0) + 1
            examples.setdefault(k, b["value"])
    top = sorted(offenders.items(), key=lambda kv: -kv[1])[:10]
    return {
        "rows": n,
        "grounded_rate": round(grounded / n, 4) if n else None,
        "rows_with_calls": with_calls,
        "invented": [{"argument": k, "rows": c, "example": examples[k]} for k, c in top],
    }


__all__ = [
    "MARKER",
    "argument_grounding",
    "grounding_report",
    "mark_grounding",
    "tool_calls_of",
    "ungrounded_arguments",
]
