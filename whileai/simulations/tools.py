"""``@tool``: a typed Python function is the tool.

The function's signature is the schema and its docstring is the
description, so the OpenAI function-calling JSON the engine needs is
never written by hand::

    import whileai as wai

    @wai.tool
    def get_order(order_id: str) -> dict:
        \"\"\"Look up an order by id.\"\"\"
        ...

    data = wai.simulate(agent, tools=[get_order], system_prompt=POLICY)

Every call that takes ``tools=`` accepts a mix of decorated functions,
plain functions and the raw schema dicts; ``schemas()`` is the one place
they are normalised. The mock world still answers the calls, faults
first, unless you hand the engine ``execute=wai.Tool.dispatch([...])`` to
run the bodies for real.

Type hints map to JSON Schema the way pydantic maps them: ``str``,
``int``, ``float``, ``bool``, ``list[T]``, ``dict``, ``Optional[T]``,
``Literal[...]``, ``Enum`` subclasses, pydantic models, and
``Annotated[T, "description"]`` for a per-parameter description. A
parameter with a default is optional. A Google-style ``Args:`` block in
the docstring also fills in descriptions.
"""

from __future__ import annotations

import enum
import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

try:  # 3.10+
    from types import UnionType
except ImportError:  # pragma: no cover
    UnionType = None  # type: ignore[assignment,misc]

from typing import Annotated

__all__ = ["Tool", "schemas", "tool"]

_SIMPLE: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_type(annotation: Any) -> dict[str, Any]:
    """JSON Schema for one Python annotation. Unknown types become an open value."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    if annotation in _SIMPLE:
        return {"type": _SIMPLE[annotation]}
    if annotation is type(None):
        return {"type": "null"}
    out: dict[str, Any]
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        base, *meta = args
        out = _json_type(base)
        note = next((m for m in meta if isinstance(m, str)), None)
        if note:
            out["description"] = note
        return out
    if origin is Literal:
        values = list(args)
        out = {"enum": values}
        kinds = {_SIMPLE.get(type(v)) for v in values}
        if len(kinds) == 1 and None not in kinds:
            out["type"] = kinds.pop()
        return out
    if origin is Union or (UnionType is not None and origin is UnionType):
        members = [a for a in args if a is not type(None)]
        if len(members) == 1:
            return _json_type(members[0])
        return {"anyOf": [_json_type(a) for a in members]}
    if annotation is list or origin in (list, Sequence, tuple, set, frozenset):
        out = {"type": "array"}
        if args:
            out["items"] = _json_type(args[0])
        return out
    if annotation is dict or origin in (dict, Mapping):
        return {"type": "object"}
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        values = [m.value for m in annotation]
        out = {"enum": values}
        kinds = {_SIMPLE.get(type(v)) for v in values}
        if len(kinds) == 1 and None not in kinds:
            out["type"] = kinds.pop()
        return out
    model_schema = getattr(annotation, "model_json_schema", None)
    if callable(model_schema):
        try:
            return dict(model_schema())
        except Exception:  # a model that cannot build its schema is an open value
            return {"type": "object"}
    return {}


_ARGS_HEAD = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$", re.IGNORECASE)
_ARG_LINE = re.compile(r"^\s*(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.+?)\s*$")


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """First paragraph as the description; ``Args:`` lines as parameter notes."""
    text = inspect.cleandoc(doc or "")
    if not text:
        return "", {}
    lines = text.splitlines()
    head: list[str] = []
    for line in lines:
        if not line.strip() or _ARGS_HEAD.match(line):
            break
        head.append(line.strip())
    notes: dict[str, str] = {}
    in_args = False
    current: str | None = None
    for line in lines:
        if _ARGS_HEAD.match(line):
            in_args = True
            continue
        if not in_args:
            continue
        if line.strip() and not line.startswith((" ", "\t")):
            break  # a new section at column zero
        m = _ARG_LINE.match(line)
        if m and (current is None or not line.startswith("        ")):
            current = m.group(1).lstrip("*")
            notes[current] = m.group(2)
        elif current and line.strip():
            notes[current] = f"{notes[current]} {line.strip()}"
    return " ".join(head), notes


@dataclass(frozen=True)
class Tool:
    """A tool the agent can call: its schema, and the function behind it.

    Built by ``@tool``. ``schema`` is the OpenAI function-calling dict the
    engine and every model backend read. Calling the object calls the
    function. ``Tool.dispatch(tools)`` turns a list of tools into the
    ``execute=(name, arguments) -> result`` callable ``simulate`` takes,
    for when the bodies should answer instead of the mock world.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any] | None = field(default=None, compare=False, repr=False)

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    # duck-typed for adapters._schema_from_object and third-party readers
    @property
    def params_json_schema(self) -> dict[str, Any]:
        return self.parameters

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.fn is None:
            raise TypeError(f"tool {self.name!r} has no function body to call")
        return self.fn(*args, **kwargs)

    @staticmethod
    def dispatch(tools: Sequence[Any]) -> Callable[[str, Mapping[str, Any]], Any]:
        """An ``execute`` callable that routes each call to the tool's body.

        Unknown names and raised exceptions come back as a result dict with
        ``status`` set, the shape the mock world uses, so one bad call does
        not end the rollout.
        """
        by_name: dict[str, Tool] = {}
        for t in tools:
            t = t if isinstance(t, Tool) else tool(t) if callable(t) else None
            if t is not None and t.fn is not None:
                by_name[t.name] = t

        def execute(name: str, arguments: Mapping[str, Any] | None) -> Any:
            t = by_name.get(str(name))
            if t is None:
                return {"status": "not_found", "error": f"no tool named {name!r}"}
            try:
                return t(**dict(arguments or {}))
            except Exception as exc:  # the world answered with an error, the run goes on
                return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

        return execute


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Turn a typed function into a ``Tool``: signature to schema, docstring to text.

    Use bare (``@wai.tool``) or with arguments (``@wai.tool(name="lookup")``).
    ``self`` and ``cls`` are skipped, ``*args``/``**kwargs`` are ignored, a
    parameter with a default is optional, and ``Annotated[T, "note"]`` or a
    docstring ``Args:`` block gives a parameter its description.
    """

    def build(f: Callable[..., Any]) -> Tool:
        if isinstance(f, Tool):
            return f
        try:
            hints = get_type_hints(f, include_extras=True)
        except Exception:
            hints = getattr(f, "__annotations__", {}) or {}
        sig = inspect.signature(f)
        desc, notes = _parse_docstring(getattr(f, "__doc__", None))
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in sig.parameters.values():
            if p.name in ("self", "cls") or p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            spec = _json_type(hints.get(p.name, p.annotation))
            if p.name in notes and "description" not in spec:
                spec["description"] = notes[p.name]
            if p.default is not inspect.Parameter.empty:
                if p.default is not None:
                    spec["default"] = p.default
            else:
                required.append(p.name)
            properties[p.name] = spec
        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        return Tool(
            name=name or str(getattr(f, "__name__", "tool")),
            description=(description if description is not None else desc)[:1000],
            parameters=parameters,
            fn=f,
        )

    return build(fn) if fn is not None else build


def schemas(tools: Sequence[Any] | None) -> list[dict[str, Any]] | None:
    """Normalise ``tools=``: decorated functions, plain functions and raw dicts to dicts.

    ``None`` stays ``None`` so callers keep their "no tools given" branch.
    Anything else raises a ``TypeError`` that says what is accepted.
    """
    if tools is None:
        return None
    out: list[dict[str, Any]] = []
    for t in tools:
        if isinstance(t, Tool):
            out.append(t.schema)
        elif isinstance(t, dict):
            out.append(t)
        elif callable(t):
            out.append(tool(t).schema)
        else:
            raise TypeError(
                f"tools= takes @wai.tool functions, plain functions or OpenAI function "
                f"schemas (dicts); got {type(t).__name__}"
            )
    return out
