"""Training-ready rows: system prompt, tool schemas, standard wire format.

A simulated row stores the conversation without the agent's own system
prompt or tool schemas; the run knows them, the row does not. A trainer
needs all three, in the shape chat templates expect, whether or not the
target model emits thinking tokens. ``training_rows`` closes that gap:

* prepends the ``system`` message,
* attaches the tool schemas on each row,
* converts tool calls to the OpenAI wire format (``id``, ``type``,
  ``function.name``, ``function.arguments`` as a JSON string) and links
  each tool result by ``tool_call_id``,
* strips ``<think>`` blocks from assistant turns, so a thinking rollout
  model never teaches a non-thinking student to emit them.

The result trains any chat-template model. No field is model-specific.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from .quality import _load_jsonl, _write_jsonl

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
_CARRY_KEYS = ("reward", "qwen_reward", "reason", "prompt", "scenario_id",
               "world_state", "faults", "label_source")


def _strip_think(text: str) -> str:
    return _THINK_BLOCK.sub("", str(text or "")).strip()


def _decoded_arguments(arguments: Any) -> Any:
    """Unwrap repeated JSON string encoding down to the structured value."""
    value = arguments
    for _ in range(3):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except ValueError:
            break
    return value


def _wire_arguments(arguments: Any) -> str:
    # Chat templates render arguments with |tojson, so a pre-encoded string
    # would be quoted twice and the model learns string-wrapped arguments.
    value = _decoded_arguments(arguments)
    if isinstance(value, str):
        return value
    return json.dumps(value or {}, separators=(",", ": "), default=str)


def tool_call_roundtrip(rows: Sequence[dict]) -> dict[str, Any]:
    """Check every tool call in exported rows parses back to a dict.

    Guards the training run, not the export: arguments that survive as
    strings get re-quoted by chat templates and teach the model to emit
    string-wrapped arguments, which then spiral on tool rejections.
    """
    checked = invalid = 0
    bad_rows: list[int] = []
    for i, row in enumerate(rows):
        row_bad = False
        for message in (row.get("messages") or []):
            for call in (message.get("tool_calls") or []):
                fn = call.get("function") if isinstance(
                    call.get("function"), dict) else {}
                checked += 1
                raw = fn.get("arguments")
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                except ValueError:
                    parsed = None
                if not isinstance(parsed, dict):
                    invalid += 1
                    row_bad = True
        if row_bad:
            bad_rows.append(i)
    return {"checked": checked, "invalid": invalid, "rows": bad_rows[:20]}


def _convert_messages(messages: Sequence[dict], *, system: str,
                      strip_think: bool) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    pending_ids: list[str] = []
    call_i = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "assistant":
            if strip_think:
                content = _strip_think(content)
            entry: dict[str, Any] = {"role": "assistant", "content": content}
            calls = message.get("tool_calls") or []
            if calls:
                wire = []
                for call in calls:
                    call = call if isinstance(call, dict) else {}
                    fn = call.get("function") if isinstance(
                        call.get("function"), dict) else call
                    call_id = str(call.get("id") or f"call_{call_i:04d}")
                    call_i += 1
                    pending_ids.append(call_id)
                    wire.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": str(fn.get("name") or ""),
                            "arguments": _wire_arguments(fn.get("arguments")),
                        },
                    })
                entry["tool_calls"] = wire
            out.append(entry)
        elif role == "tool":
            entry = {"role": "tool", "content": content}
            if message.get("name"):
                entry["name"] = str(message["name"])
            call_id = (str(message.get("tool_call_id"))
                       if message.get("tool_call_id")
                       else (pending_ids.pop(0) if pending_ids else ""))
            if call_id:
                entry["tool_call_id"] = call_id
            out.append(entry)
        elif role in {"user", "system"}:
            out.append({"role": role, "content": content})
    return out


def _resolve(source) -> tuple[list[dict], str, list, str]:
    """rows, system prompt, tools, source path (best effort)."""
    if hasattr(source, "trajectories"):
        profile = getattr(source, "profile", None)
        system = str(getattr(profile, "policy", "") or "")
        tools = list(getattr(profile, "tools", None) or [])
        return list(source.trajectories), system, tools, ""
    if isinstance(source, (str, Path)):
        return _load_jsonl(source), "", [], str(source)
    return list(source), "", [], ""


def training_rows(source, *, system_prompt: str | None = None,
                  tools: Sequence[dict] | None = None,
                  strip_think: bool = True) -> list[dict]:
    """Rows a trainer can consume directly. See the module docstring.

    ``source`` is a ``SimulationData`` (system prompt and tools come from
    its profile), a row list, or a JSONL path. For lists and paths, pass
    ``system_prompt=`` and ``tools=`` explicitly; a row exported without
    its policy trains an agent that never saw its rules.
    """
    from zeroproof_simulations import conversation
    rows, system, resolved_tools, _ = _resolve(source)
    if system_prompt is not None:
        system = str(system_prompt)
    if tools is not None:
        resolved_tools = list(tools)
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        messages = row.get("messages") or conversation(row)
        entry: dict[str, Any] = {
            "messages": _convert_messages(messages, system=system,
                                          strip_think=strip_think),
        }
        if resolved_tools:
            entry["tools"] = list(resolved_tools)
        for key in _CARRY_KEYS:
            if row.get(key) is not None:
                entry[key] = row[key]
        out.append(entry)
    return out


def export_training(source, output: str | None = None, *,
                    system_prompt: str | None = None,
                    tools: Sequence[dict] | None = None,
                    strip_think: bool = True,
                    validate: bool = True) -> dict[str, Any]:
    """Write ``training_rows`` as JSONL. Never overwrites the source.

    With a path source and no ``output``, writes ``<name>.train.jsonl``
    next to it. ``validate=True`` refuses to write a dataset whose tool
    calls do not round-trip to structured arguments; pass ``validate=False``
    to export anyway and read the report instead.
    """
    rows = training_rows(source, system_prompt=system_prompt, tools=tools,
                         strip_think=strip_think)
    roundtrip = tool_call_roundtrip(rows)
    if validate and roundtrip["invalid"]:
        raise ValueError(
            f"tool_call_roundtrip_invalid: {roundtrip['invalid']} of "
            f"{roundtrip['checked']} tool calls do not parse back to "
            f"structured arguments (rows {roundtrip['rows']}); training on "
            "them teaches string-wrapped arguments. Fix the rows or pass "
            "validate=False.")
    _, _, _, src = _resolve(source)
    dest = output
    if not dest and src:
        path = Path(src)
        dest = str(path.with_name(path.stem + ".train"
                                  + (path.suffix or ".jsonl")))
    report: dict[str, Any] = {
        "n": len(rows),
        "with_system": sum(1 for r in rows
                           if r["messages"] and
                           r["messages"][0]["role"] == "system"),
        "with_tools": sum(1 for r in rows if r.get("tools")),
        "tool_call_roundtrip": roundtrip,
    }
    if dest:
        report["path"] = _write_jsonl(dest, rows)
        report["n_written"] = len(rows)
    return report


__all__ = ["training_rows", "export_training", "tool_call_roundtrip"]
