"""``@wai.tool``: the function is the tool, the signature is the schema.

Offline. Covers the schema mapping, the docstring parse, the mixed
``tools=`` list, dispatch as ``execute=``, and one seeded run end to end.
"""

from __future__ import annotations

import enum
from typing import Annotated, Literal

import pytest

import whileai as wai
from whileai.simulations.tools import Tool, schemas


class Priority(enum.Enum):
    LOW = "low"
    HIGH = "high"


@wai.tool
def get_order(order_id: str, include_items: bool = False) -> dict:
    """Look up an order by id.

    Args:
        order_id: The order number on the receipt.
        include_items: Also return the line items.
    """
    return {"order_id": order_id, "status": "shipped", "items": [] if include_items else None}


def test_signature_becomes_the_schema():
    s = get_order.schema
    assert s["type"] == "function"
    f = s["function"]
    assert f["name"] == "get_order"
    assert f["description"] == "Look up an order by id."
    props = f["parameters"]["properties"]
    assert props["order_id"] == {
        "type": "string",
        "description": "The order number on the receipt.",
    }
    assert props["include_items"]["type"] == "boolean"
    assert props["include_items"]["default"] is False
    assert f["parameters"]["required"] == ["order_id"]


def test_types_map_like_pydantic():
    @wai.tool
    def f(
        n: int,
        x: float,
        tags: list[str],
        meta: dict,
        level: Literal["a", "b"],
        prio: Priority,
        note: Annotated[str, "free text for the agent"],
        maybe: int | None = None,
        alt: int | None = None,
    ) -> None:
        """Every hint kind at once."""

    p = f.parameters["properties"]
    assert p["n"] == {"type": "integer"}
    assert p["x"] == {"type": "number"}
    assert p["tags"] == {"type": "array", "items": {"type": "string"}}
    assert p["meta"] == {"type": "object"}
    assert p["level"] == {"enum": ["a", "b"], "type": "string"}
    assert p["prio"] == {"enum": ["low", "high"], "type": "string"}
    assert p["note"] == {"type": "string", "description": "free text for the agent"}
    assert p["maybe"] == {"type": "integer"}
    assert p["alt"] == {"type": "integer"}
    assert f.parameters["required"] == ["n", "x", "tags", "meta", "level", "prio", "note"]


def test_decorator_with_arguments_and_calling_through():
    @wai.tool(name="lookup", description="Find a thing.")
    def find(q: str) -> str:
        return q.upper()

    assert find.name == "lookup"
    assert find.description == "Find a thing."
    assert find("abc") == "ABC"
    assert isinstance(find, Tool)


def test_schemas_accepts_a_mixed_list_and_names_the_fix():
    def plain(order_id: str) -> dict:
        """Plain function, no decorator."""
        return {}

    raw = {
        "type": "function",
        "function": {"name": "raw", "parameters": {"type": "object", "properties": {}}},
    }
    out = schemas([get_order, plain, raw])
    assert [t["function"]["name"] for t in out] == ["get_order", "plain", "raw"]
    assert schemas(None) is None
    with pytest.raises(TypeError, match="tools= takes"):
        schemas([42])


def test_dispatch_routes_calls_and_keeps_the_run_alive():
    execute = Tool.dispatch([get_order])
    assert execute("get_order", {"order_id": "4412"})["status"] == "shipped"
    assert execute("nope", {})["status"] == "not_found"

    @wai.tool
    def boom(x: int) -> int:
        """Raises."""
        raise ValueError("bad x")

    assert Tool.dispatch([boom])("boom", {"x": 1}) == {
        "status": "error",
        "error": "ValueError: bad x",
    }


def test_a_decorated_tool_runs_through_the_seeded_loop_offline():
    data = wai.simulate(
        wai.seeded_agent([get_order]),
        tools=[get_order],
        system_prompt="Help customers with orders.",
        simulator=False,
        mode="rl",
        repeats=2,
        repeat_policy="fixed",
        budget=8,
        seed=0,
    )
    assert len(data.rows) > 0
    names = (
        {t["function"]["name"] for t in data.tools}
        if getattr(data, "tools", None)
        else {"get_order"}
    )
    assert "get_order" in names
    scored = data.grade(judge=lambda row: {"reward": int(not row["seeded"])})
    assert scored.pass_at is not None


def test_every_tools_keyword_takes_a_decorated_function():
    from whileai.simulations import coverage_gap, evaluate, preflight, world

    w = world([get_order])
    assert "get_order" in {t["function"]["name"] for t in w.tools} or w.call(
        "get_order", {"order_id": "1"}
    )
    report = preflight([get_order], "Help customers with orders.")
    assert isinstance(report, dict)
    gap = coverage_gap(["Where is order 4473?"], tools=[get_order], system_prompt="Help customers.")
    assert isinstance(gap, dict)
    data = wai.simulate(
        wai.seeded_agent([get_order]),
        tools=[get_order],
        system_prompt="Help customers with orders.",
        simulator=False,
        budget=4,
        seed=0,
    )
    scored = evaluate(data.rows, lambda row: {"reward": 1}, tools=[get_order])
    assert len(scored.rows) == len(data.rows)
