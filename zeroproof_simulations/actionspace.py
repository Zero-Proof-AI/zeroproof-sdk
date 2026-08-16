"""Tool-sequence shapes over this agent's schemas."""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

PROVENANCE = ("user", "derived", "invented", "absent")
ENUM_CAP = 400


def _tool_name(schema: dict) -> str:
    fn = schema.get("function", schema) if isinstance(schema, dict) else {}
    return str(fn.get("name") or "")


def _tool_args(schema: dict) -> list[str]:
    fn = schema.get("function", schema) if isinstance(schema, dict) else {}
    props = ((fn.get("parameters") or fn.get("input_schema") or {}).get(
        "properties") or {})
    return sorted(props)


def _tool_desc(schema: dict) -> str:
    fn = schema.get("function", schema) if isinstance(schema, dict) else {}
    return str(fn.get("description") or _tool_name(schema).replace("_", " "))


@dataclass(frozen=True)
class Call:
    tool: str
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class Shape:
    calls: tuple[Call, ...]
    confirmed: bool = False

    def key(self) -> str:
        body = ";".join(f"{c.tool}({','.join(c.provenance)})" for c in self.calls)
        return f"{'C' if self.confirmed else 'U'}|{body}"

    def tools(self) -> list[str]:
        return [c.tool for c in self.calls]


def shape_as_tags(shape: Shape, tool_schemas: Sequence[dict]) -> dict[str, Any]:
    """Tiny meta tags for the generator. No wording, no domain story."""
    arg_index = {_tool_name(s): _tool_args(s) for s in tool_schemas}
    return {
        "key": shape.key(),
        "confirmed": shape.confirmed,
        "calls": [
            {
                "tool": call.tool,
                "args": {
                    name: source for name, source in
                    zip(arg_index.get(call.tool, ()), call.provenance)
                },
            }
            for call in shape.calls
        ],
    }


def render_target_situation(shape: Shape, tool_schemas: Sequence[dict]) -> str:
    """A short user request from this agent's tool descriptions."""
    by_name = {_tool_name(s): _tool_desc(s) for s in tool_schemas}
    jobs = [by_name.get(c.tool, c.tool.replace("_", " ")) for c in shape.calls]
    jobs = [j.rstrip(".") for j in jobs if j]
    if not jobs:
        return "Can you help me with this?"
    if len(jobs) == 1:
        text = f"Can you {jobs[0][0].lower() + jobs[0][1:]} for me?"
    else:
        text = f"I need {jobs[0][0].lower() + jobs[0][1:]}, then {jobs[1][0].lower() + jobs[1][1:]}."
    if any(src == "absent" for call in shape.calls for src in call.provenance):
        text += " I do not have every detail."
    if shape.confirmed:
        text += " Yes, go ahead."
    return text


def enumerate_shapes(tool_schemas: list[dict], max_len: int = 2,
                     cap: int = ENUM_CAP) -> list[Shape]:
    tools = [(_tool_name(s), _tool_args(s)) for s in tool_schemas if _tool_name(s)]
    per_tool: list[Call] = []
    for name, args in tools:
        if not args:
            per_tool.append(Call(name, ()))
            continue
        per_tool.append(Call(name, tuple("user" for _ in args)))
        if args:
            mixed = tuple("user" if i == 0 else "absent" for i in range(len(args)))
            per_tool.append(Call(name, mixed))
    shapes: list[Shape] = []
    for length in range(1, max_len + 1):
        for seq in itertools.product(per_tool, repeat=length):
            for confirmed in (False, True):
                shapes.append(Shape(tuple(seq), confirmed))
                if len(shapes) >= cap:
                    return shapes
    return shapes


def action_space_targets(tool_schemas: list[dict], *, max_len: int = 2,
                         cap: int = ENUM_CAP) -> tuple[list[Shape], dict]:
    shapes = enumerate_shapes(tool_schemas, max_len=max_len, cap=cap)
    return shapes, {"targets": len(shapes), "max_len": max_len}


def uncovered_action_shapes(targets: Iterable[Shape], induced_keys: set[str],
                            *, limit: int = 12) -> list[Shape]:
    missing = [s for s in targets if s.key() not in induced_keys]
    return missing[:max(0, int(limit))]


def _arg_source(value: Any, prompt: str, prior: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "absent"
    if text in prompt:
        return "user"
    if text in prior:
        return "derived"
    return "invented"


def shape_from_trajectory(trajectory: dict,
                          tool_schemas: Sequence[dict]) -> Shape | None:
    """Best-effort action shape observed in a rollout."""
    steps = trajectory.get("steps") or []
    prompt = str(trajectory.get("prompt", "")).lower()
    prior = ""
    calls: list[Call] = []
    arg_index = {_tool_name(s): _tool_args(s) for s in tool_schemas}
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool = str(step.get("tool") or "")
        if not tool:
            continue
        args = step.get("arguments") if isinstance(step.get("arguments"), dict) else {}
        names = arg_index.get(tool) or sorted(args)
        prov = tuple(_arg_source(args.get(name), prompt, prior) for name in names)
        calls.append(Call(tool, prov))
        prior += json.dumps(step.get("result"), default=str).lower()
    if not calls:
        return None
    return Shape(tuple(calls), confirmed=False)


def induced_keys_from_trajectory(trajectory: dict, targets: Iterable[Shape],
                                 tool_schemas: Sequence[dict]) -> set[str]:
    """Target shape keys the agent naturally exhibited."""
    observed = shape_from_trajectory(trajectory, tool_schemas)
    if observed is None:
        return set()
    target_list = list(targets)
    keys: set[str] = set()
    for target in target_list:
        if target.calls == observed.calls:
            keys.add(target.key())
    if observed.key() in {t.key() for t in target_list}:
        keys.add(observed.key())
    return keys
