"""Shared offline fixtures. Not part of the SDK."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
GITHUB_SPEC = FIXTURES / "github"
LINEAR_SPEC = FIXTURES / "linear"

TOOLS = [
    {"type": "function", "function": {"name": "lookup_order", "parameters": {
        "type": "object", "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "create_refund", "parameters": {
        "type": "object", "properties": {"order_id": {"type": "string"},
                                          "amount": {"type": "number"}},
        "required": ["order_id", "amount"]}}},
    {"type": "function", "function": {"name": "get_refund_status", "parameters": {
        "type": "object", "properties": {"refund_id": {"type": "string"}},
        "required": ["refund_id"]}}},
]
POLICY = "Look up an order before refunding it. Report failures honestly."


def scripted_agent(message: str) -> dict:
    raw = int((re.search(r"(\d+)", message) or [0, "40"])[1])
    amount = raw % 200
    token = re.search(r"[A-Za-z]+[-_]?\d+", message)
    order = token.group(0) if token else f"ord_{amount}"
    if amount > 100:
        steps = [{"tool": "create_refund",
                  "arguments": {"order_id": f"acct_{raw * 7}", "amount": amount},
                  "result": {"status": "created", "id": f"re_{amount}"}}]
        return {"steps": steps, "final_text": f"Refunded ${amount}."}
    if amount % 7 == 0:
        steps = [{"tool": "lookup_order", "arguments": {"order_id": order},
                  "result": {"status": "not_found"}}]
        return {"steps": steps, "final_text": f"Refunded ${amount} successfully."}
    steps = [{"tool": "lookup_order", "arguments": {"order_id": order},
              "result": {"status": "ok", "order_id": order}},
             {"tool": "create_refund",
              "arguments": {"order_id": order, "amount": amount},
              "result": {"status": "created", "id": f"re_{amount}"}}]
    return {"steps": steps, "final_text": f"Refunded ${amount}."}


def offline(**kwargs: Any) -> dict[str, Any]:
    """Kwargs for an offline ``simulate()`` call. Never hits a GPU."""
    advanced = {"per_round": 8, "mutate_failures": False}
    extra = dict(kwargs.pop("advanced", None) or {})
    if "per_round" in kwargs:
        extra.setdefault("per_round", kwargs.pop("per_round"))
    if "mutate_failures" in kwargs:
        extra.setdefault("mutate_failures", kwargs.pop("mutate_failures"))
    advanced.update(extra)
    kw: dict[str, Any] = {
        "seed": 0,
        "grade": False,
        "concurrency": 4,
        "simulator": False,
        "time_budget": None,
        "advanced": advanced,
    }
    if "spec" not in kwargs:
        kw["tools"] = TOOLS
        kw["policy"] = POLICY
    kw.update(kwargs)
    return kw


def simulate_offline(agent: Any = None, **kwargs: Any):
    import zeroproof_simulations as zps
    return zps.simulate(
        scripted_agent if agent is None else agent, **offline(**kwargs))
