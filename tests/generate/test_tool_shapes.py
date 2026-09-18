"""Three tool shapes go in, one shape comes out.

Tools arrive in the OpenAI envelope, bare, or in Anthropic's
``input_schema`` spelling. Everything downstream reads ``parameters``, so
an Anthropic tool used to reach the wire with its arguments missing and
the agent saw a tool it could not call with anything.
"""

from __future__ import annotations

from whileai.simulations.generate.adapters import inspect
from whileai.simulations.generate.agents import _wire_tools

SCHEMA = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}},
    "required": ["order_id"],
}

ENVELOPE = {
    "type": "function",
    "function": {"name": "lookup_order", "description": "Find an order.", "parameters": SCHEMA},
}
BARE = {"name": "get_status", "description": "Order status.", "parameters": SCHEMA}
ANTHROPIC = {"name": "create_refund", "description": "Refund it.", "input_schema": SCHEMA}


def _params(tool: dict) -> dict:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return fn.get("parameters")


def test_all_three_shapes_land_with_parameters():
    profile = inspect(None, tools=[ENVELOPE, BARE, ANTHROPIC], system_prompt="Be careful.")
    by_name = {(t.get("function") or t).get("name"): t for t in profile.tools}
    assert set(by_name) == {"lookup_order", "get_status", "create_refund"}
    for tool in profile.tools:
        assert _params(tool) == SCHEMA


def test_the_anthropic_tool_reaches_the_wire_with_its_arguments():
    profile = inspect(None, tools=[ANTHROPIC], system_prompt="Be careful.")
    wire = _wire_tools(profile.tools)
    assert wire == [
        {
            "type": "function",
            "function": {
                "name": "create_refund",
                "description": "Refund it.",
                "parameters": SCHEMA,
            },
        }
    ]


def test_the_callers_dict_is_not_edited():
    caller = dict(ANTHROPIC)
    inspect(None, tools=[caller], system_prompt="Be careful.")
    assert caller == ANTHROPIC and "parameters" not in caller


def test_an_explicit_parameters_key_wins():
    mixed = {"name": "odd", "parameters": SCHEMA, "input_schema": {"type": "object"}}
    profile = inspect(None, tools=[mixed], system_prompt="")
    assert _params(profile.tools[0]) == SCHEMA
