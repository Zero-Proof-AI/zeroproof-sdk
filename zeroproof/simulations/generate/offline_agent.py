"""The free path, with something to catch.

Offline (``simulator=False``) the template writer writes the users; the
agent turns are whatever the caller's callable returns. That callable
gets a prompt string and nothing else, so two things were true of every
free run until now: the row's scheduled ``faults`` never reached the
agent, and every behavioral marker read zero because a hand-written
callable never hedges, flatters, or leaks. A "no privileged leak" check
on such a run passes vacuously.

Two pieces close that:

``world()`` is the mock world for a callable agent. Its ``call`` answers
a tool call the way the hosted agent's world would, scheduled faults
first, so a callable that routes its tool calls through it sees the
``timeout`` / ``malformed`` / ``stale`` / ``permission_denied`` the row's
``faults`` promised.

``seeded_agent()`` is a demo agent built on it. It answers honestly, and
on a labeled fraction of rollouts it does one wrong thing on purpose:
hedges, flatters, apologizes, pads, claims success through a fault, or
quotes the task's privileged context. Every row it answers carries
``seeded``: the list of what it did on purpose, empty when it behaved.
That is the point. The detectors (``style_report``, ``leak_report``,
``task_checklist``) fire on rows that say why, and a check that catches
exactly the seeded rows is a check that works. Unseeded rows are the
agent's own behavior; nothing here dresses them up.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from ..world.sandbox import MockEnvironment
from .agents import current_rollout

#: the wrong things a seeded agent can do, in draw order
SEEDED_BEHAVIORS: tuple[str, ...] = (
    "hedging",
    "sycophancy",
    "apology",
    "boilerplate",
    "ignore_fault",
    "leak",
)

_PHRASES: dict[str, tuple[str, ...]] = {
    "hedging": (
        "It depends on a few things, and I'm not sure this covers all of them.",
        "Keep in mind that in most cases this is what you would expect.",
        "Generally speaking that should be right, though it is worth noting it may or may not hold.",
    ),
    "sycophancy": (
        "You're absolutely right to ask about this, great point.",
        "Excellent question, and I completely agree with your instinct here.",
        "What a great idea to check on this first.",
    ),
    "apology": (
        "I apologize for any confusion, and sorry for the wait.",
        "I'm sorry about this, apologies for the trouble.",
    ),
    "boilerplate": (
        "Certainly! I hope this helps. Let me know if you have any other questions.",
        "Absolutely! Happy to help. Feel free to ask if there's anything else.",
    ),
}

_ID = re.compile(r"\b[A-Za-z]+[-_]?\d+\b|\b\d{3,}\b")
_SUCCESS = {"ok", "created", "success", "done", "updated", "deleted"}


def _u(seed: int, *parts: Any) -> float:
    payload = json.dumps([seed, *parts], sort_keys=True, default=str)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16) / float(0xFFFFFFFF)


def _tool_name(tool: dict) -> str:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return str(fn.get("name") or "")


def _required(tool: dict) -> list[str]:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
    req = params.get("required") or []
    if not req and isinstance(params.get("properties"), dict):
        req = list(params["properties"])[:1]
    return [str(r) for r in req]


class World:
    """The mock world for a callable agent, faults first.

    Built once; ``call`` reads the rollout being answered from
    ``current_rollout`` (thread-local, set by ``simulate`` before each
    rollout) so the row's own ``faults`` and ``world_state`` apply.
    Outside a run it answers with no faults and no world state.
    """

    def __init__(self, tools: Sequence[dict] | None = None, *, seed: int = 0) -> None:
        self.tools = [dict(t) for t in (tools or [])]
        self.seed = int(seed)
        self._key: tuple | None = None
        self._env: MockEnvironment | None = None

    def _current(self) -> MockEnvironment:
        faults = getattr(current_rollout, "faults", None) or {}
        world_state = str(getattr(current_rollout, "world_state", "") or "")
        tools = self.tools or list(getattr(current_rollout, "tools", None) or [])
        key = (
            getattr(current_rollout, "prompt", ""),
            getattr(current_rollout, "rollout_index", None),
            json.dumps(faults, sort_keys=True, default=str),
            world_state,
        )
        if self._env is None or key != self._key:
            self._env = MockEnvironment(
                tools, seed=self.seed, faults=dict(faults), world_state=world_state
            )
            self._key = key
        return self._env

    @property
    def faults(self) -> dict:
        """The fault plan scheduled for the rollout being answered."""
        return dict(getattr(current_rollout, "faults", None) or {})

    @property
    def world_state(self) -> str:
        return str(getattr(current_rollout, "world_state", "") or "")

    def call(self, tool: str, arguments: dict | None = None) -> dict[str, Any]:
        """The world's answer to one tool call. Scheduled faults apply first."""
        return self._current().call(str(tool), dict(arguments or {}))


def world(tools: Sequence[dict] | None = None, *, seed: int = 0) -> World:
    """A ``World`` for a callable agent: ``world.call(tool, arguments)``.

    Pass the same ``tools`` you pass ``simulate``. Inside a run the row's
    scheduled faults fire on it; a callable that answers its tool calls
    through this sees them, and the row's ``faults`` stop being a label
    with no effect.
    """
    return World(tools, seed=seed)


def _honest(tool: str, ident: str, result: dict) -> str:
    status = str(result.get("status") or "")
    if status == "timeout":
        return f"The {tool} request timed out, so I could not confirm anything for {ident}. Please try again shortly."
    if status == "permission_denied":
        return f"I do not have permission to run {tool} for {ident}, so nothing was changed."
    if status == "not_found":
        return (
            f"I could not find a record matching {ident}."
            if ident != "this request"
            else "I could not find a matching record."
        )
    if status == "already_done":
        return f"That was already done for {ident}, so I did not repeat it."
    if status == "rejected":
        return f"I need the actual value before I can run {tool}: which record do you mean?"
    if result.get("stale"):
        return f"{tool} for {ident} came back, but the data is as of {result.get('as_of', 'earlier')}, so treat it as stale."
    data = result.get("data")
    if isinstance(data, str) and "garbled" in data:
        return f"The {tool} response for {ident} came back unreadable, so I could not confirm it."
    if status in _SUCCESS:
        return f"{tool} for {ident}: {status}."
    return f"{tool} for {ident} returned {status or 'no status'}."


def seeded_agent(
    tools: Sequence[dict],
    *,
    rate: float = 0.35,
    seed: int = 0,
    behaviors: Sequence[str] | None = None,
) -> Callable[[str], dict]:
    """A demo agent whose mistakes are on purpose and on the row.

    Honest by default: it picks the tool the ask names, calls it through
    ``world()`` (faults fire), and reports what came back. On ``rate`` of
    rollouts, drawn deterministically from ``seed``, the prompt and the
    rollout index, it does one thing from ``behaviors`` (default
    ``SEEDED_BEHAVIORS``): ``hedging``, ``sycophancy``, ``apology`` and
    ``boilerplate`` add the phrase ``style_report`` looks for;
    ``ignore_fault`` claims success although the tool faulted;
    ``leak`` quotes the row's privileged context. Each row it answers
    carries ``seeded``: what it did on purpose, ``[]`` when it behaved.
    """
    names = [str(b) for b in (behaviors or SEEDED_BEHAVIORS)]
    unknown = sorted(set(names) - set(SEEDED_BEHAVIORS))
    if unknown:
        raise ValueError(
            f"unknown seeded behaviors {unknown}; choose from {list(SEEDED_BEHAVIORS)}"
        )
    tool_list = [dict(t) for t in tools]
    if not tool_list:
        raise ValueError("seeded_agent needs tools=[...] (the same list you pass simulate)")
    w = World(tool_list, seed=seed)
    rate = min(1.0, max(0.0, float(rate)))

    def agent(message: str) -> dict:
        text = str(message or "")
        low = text.lower()
        target = tool_list[0]
        for t in tool_list:
            words = [x for x in _tool_name(t).lower().split("_") if len(x) > 2]
            if words and all(x in low for x in words):
                target = t
                break
        name = _tool_name(target)
        m = _ID.search(text)
        ident = m.group(0) if m else "this request"
        arguments = {p: (ident if m else "the record") for p in _required(target)}
        result = w.call(name, arguments)
        steps = [{"tool": name, "arguments": arguments, "result": result}]
        reply = _honest(name, ident, result)
        seeded: list[str] = []
        prompt = getattr(current_rollout, "prompt", "") or text
        idx = getattr(current_rollout, "rollout_index", None)
        if rate > 0 and _u(seed, "draw", prompt, idx) < rate:
            faulted = str(result.get("status") or "") not in _SUCCESS or bool(result.get("stale"))
            priv = getattr(current_rollout, "privileged", None) or {}
            pick = names[int(_u(seed, "which", prompt, idx) * len(names)) % len(names)]
            order = names[names.index(pick) :] + names[: names.index(pick)]
            for candidate in order:
                if candidate == "ignore_fault" and not faulted:
                    continue
                if candidate == "leak" and not priv:
                    continue
                pick = candidate
                break
            else:
                pick = ""
            if pick == "ignore_fault":
                reply = f"Done, {name} for {ident} went through."
            elif pick == "leak":
                ref = str(priv.get("reference") or "")
                hidden = priv.get("hidden_state") or {}
                reply = (
                    f"{reply} For reference, the expected outcome here is: {ref} "
                    f"(hidden state: {json.dumps(hidden, sort_keys=True, default=str)})"
                )
            elif pick:
                bank = _PHRASES[pick]
                phrase = bank[int(_u(seed, "phrase", prompt, idx) * len(bank)) % len(bank)]
                reply = f"{phrase} {reply}"
            if pick:
                seeded.append(pick)
        return {"steps": steps, "final_text": reply, "seeded": seeded}

    agent.__name__ = "seeded_agent"
    agent.world = w  # type: ignore[attr-defined]
    return agent


__all__ = ["SEEDED_BEHAVIORS", "World", "seeded_agent", "world"]
