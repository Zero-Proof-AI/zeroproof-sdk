"""The offline writer speaks as a customer, never as the tool schema.

A tester on ``simulator=False`` got asks that were the tool description
wrapped in a template ("Can you look up an order by id. Returns item,
total, order date and status for me?"), which tests the agent on text it
already has in its own tools.
"""

from __future__ import annotations

import whileai.simulations as wai
from tests.helpers import POLICY, scripted_agent
from whileai.simulations.generate.actionspace import enumerate_shapes, render_target_situation

SENTENCES = [
    "Look up an order by id",
    "Returns item, total, order date and status",
    "Issue a refund for an order",
    "Only after checking policy",
]

SPEC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": f"{SENTENCES[0]}. {SENTENCES[1]}.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_refund",
            "description": f"{SENTENCES[2]}. {SENTENCES[3]}.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["order_id", "amount"],
            },
        },
    },
]


def _quotes_a_description(text: str) -> list[str]:
    low = text.lower()
    return [s for s in SENTENCES if s.lower() in low]


def test_action_shape_asks_never_quote_a_description():
    asks = [
        render_target_situation(shape, SPEC_TOOLS)
        for shape in enumerate_shapes(SPEC_TOOLS, max_len=2)
    ]
    assert asks
    assert not [ask for ask in asks if _quotes_a_description(ask)]
    assert "Can you check an order for me?" in asks


def test_an_offline_run_never_quotes_a_description():
    data = wai.simulate(
        scripted_agent,
        tools=SPEC_TOOLS,
        system_prompt=POLICY,
        budget=40,
        seed=3,
        simulator=False,
        grade=False,
        time_budget=None,
        advanced={"per_round": 6},
    )
    prompts = [row["prompt"] for row in data.trajectories]
    assert prompts
    # the action-shape arm ran, so this run covers the writer that quoted
    # descriptions rather than only the coverage-grid one
    assert "Can you check an order for me?" in prompts
    assert not [p for p in prompts if _quotes_a_description(p)]


def test_the_offline_writer_stays_deterministic():
    def run() -> list[str]:
        data = wai.simulate(
            scripted_agent,
            tools=SPEC_TOOLS,
            system_prompt=POLICY,
            budget=8,
            seed=7,
            simulator=False,
            grade=False,
            reproducible=True,
            time_budget=None,
            advanced={"per_round": 8, "mutate_failures": False},
        )
        return [row["prompt"] for row in data.trajectories]

    assert run() == run()
