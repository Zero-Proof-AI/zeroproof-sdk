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
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
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
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["order_id", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_refund_status",
            "parameters": {
                "type": "object",
                "properties": {"refund_id": {"type": "string"}},
                "required": ["refund_id"],
            },
        },
    },
]
POLICY = "Look up an order before refunding it. Report failures honestly."


def scripted_agent(message: str) -> dict:
    raw = int((re.search(r"(\d+)", message) or [0, "40"])[1])
    amount = raw % 200
    token = re.search(r"[A-Za-z]+[-_]?\d+", message)
    order = token.group(0) if token else f"ord_{amount}"
    if amount > 100:
        steps = [
            {
                "tool": "create_refund",
                "arguments": {"order_id": f"acct_{raw * 7}", "amount": amount},
                "result": {"status": "created", "id": f"re_{amount}"},
            }
        ]
        return {"steps": steps, "final_text": f"Refunded ${amount}."}
    if amount % 7 == 0:
        steps = [
            {
                "tool": "lookup_order",
                "arguments": {"order_id": order},
                "result": {"status": "not_found"},
            }
        ]
        return {"steps": steps, "final_text": f"Refunded ${amount} successfully."}
    steps = [
        {
            "tool": "lookup_order",
            "arguments": {"order_id": order},
            "result": {"status": "ok", "order_id": order},
        },
        {
            "tool": "create_refund",
            "arguments": {"order_id": order, "amount": amount},
            "result": {"status": "created", "id": f"re_{amount}"},
        },
    ]
    return {"steps": steps, "final_text": f"Refunded ${amount}."}


class FakeWriter:
    """Stands in for the model writer so tests need no GPU.

    Every situation is model-written in the product; there is no template
    arm. A test still needs situations, so it supplies this. It reads the
    same scenario cards the real writer reads and emits one ask per card,
    labelled with the card's coordinates rather than written in English,
    so every row carries its ``scenario_dimensions`` and fault plan exactly
    as a model-written one would. Defined here, never shipped.
    """

    def __init__(self, tools: Any = None, policy: str = "", per_round: int = 8) -> None:
        self.tools = list(tools or [])
        self.policy = str(policy or "")
        self.per_round = int(per_round)
        self.seed = 0
        self.last_errors: dict[str, str] = {}
        self.fault_plans: dict[str, dict] = {}
        self.last_candidate_provenance: dict[str, dict] = {}
        self._regions: list[dict] | None = None

    def bind(self, tools: Any, policy: str) -> FakeWriter:
        self.tools, self.policy = list(tools or []), str(policy or "")
        self._regions = None
        return self

    def set_search_context(self, **_kw: Any) -> None:
        return None

    def _cards(self) -> list[dict]:
        if self._regions is None:
            from zeroproof.simulations.generate.scenarios import scenario_regions

            self._regions = list(scenario_regions(self.tools, self.policy) or [])
        return self._regions

    def __call__(self, _dataset: Any = None, index: int = 0) -> list[str]:
        """A few cards per call, three phrasings each, like the real writer.

        The old template arm emitted per_round // 8 asks a round. A fake that
        answered instantly with per_round asks per wave flooded the pool and
        made batch selection quadratic; and sft mode needs several phrasings
        of one situation before it can select any, so each card gets three.
        """
        from zeroproof.simulations.generate.scenarios import fault_plan_for_region

        cards = self._cards()
        per_call = max(2, min(8, self.per_round // 8))
        texts: list[str] = []
        self.last_candidate_provenance = {}
        for i in range(per_call):
            if not cards:
                text = f"situation r{int(index)}s{i}: please handle this"
                texts.append(text)
                self.last_candidate_provenance[text] = {
                    "arm": "llm_guided",
                    "generator": "model",
                    "round": int(index),
                    "seed": self.seed,
                }
                continue
            region = cards[(int(index) * per_call + i) % len(cards)]
            a = dict(region.get("assignment") or {})
            plan = fault_plan_for_region(region) or {}
            for j in range(3):
                text = (
                    f"[{region['id']}] {a.get('tool', 'any')} / {a.get('stance', 'ordinary')} / "
                    f"{a.get('world_state', 'unspecified')} (round {int(index)}, phrasing {j})"
                )
                texts.append(text)
                self.last_candidate_provenance[text] = {
                    "arm": "llm_guided",
                    "parent": None,
                    "scenario_dimensions": a,
                    "seed": self.seed,
                    "region_id": region["id"],
                    "assignment": a,
                    "generator": "model",
                    "round": int(index),
                }
                if plan:
                    self.fault_plans[text] = dict(plan)
        return texts


def offline(**kwargs: Any) -> dict[str, Any]:
    """Kwargs for a ``simulate()`` call that needs no GPU: a fake writer."""
    # stop_grace 0: the model path waits for in-flight writer waves on
    # stop; a fake writer has none, and five seconds per run is the suite.
    advanced = {"per_round": 8, "mutate_failures": False, "stop_grace": 0}
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
        "simulator": FakeWriter(per_round=advanced.get("per_round", 8)),  # bound below
        "time_budget": None,
        "advanced": advanced,
    }
    if "spec" not in kwargs:
        kw["tools"] = TOOLS
        kw["policy"] = POLICY
    kw.update(kwargs)
    sim = kw.get("simulator")
    if isinstance(sim, FakeWriter) and not sim.tools:
        sim.bind(kw.get("tools") or [], kw.get("policy") or kw.get("system_prompt") or "")
    return kw


def simulate_offline(agent: Any = None, **kwargs: Any):
    import zeroproof.simulations as zps

    return zps.simulate(scripted_agent if agent is None else agent, **offline(**kwargs))
