"""The generate-side defaults are named, grounded, and the ones a caller
plausibly needs to move have a knob: a patience hazard table and a user
temperature on ``local_model``, a writer temperature on ``advanced=``.

Each test here fails on the code before the refactor: the knob was an
unknown keyword, or the table was read as a level name and silently fell
back to ``normal``.
"""

from __future__ import annotations

import pytest

from tests.helpers import POLICY, TOOLS, simulate_offline
from whileai.simulations.generate import diversity, generator
from whileai.simulations.generate.agents import (
    HUMAN_TOOL_TEMPERATURE,
    LOCAL_MODEL_TEMPERATURE,
    PATIENCE_HAZARDS,
    USER_TURN_TEMPERATURE,
    _user_walks_away,
    local_model,
    patience_hazards,
    patience_may_leave,
)
from whileai.simulations.generate.diversity import (
    WRITER_TEMP_HI,
    WRITER_TEMP_LO,
    sample_writer_temperature,
    writer_temperature_band,
)
from whileai.simulations.generate.generator import ModelSimulator, make_default_generator

MSGS = [f"refund order ORD-{i}, it arrived broken" for i in range(40)]
ASK = "Which order is it, and what was the amount?"
# A new question every turn: the loop ends a thread whose agent repeats
# itself, so only patience can end these early.
QUESTIONS = [
    ASK,
    "Was it the blue lamp or the desk?",
    "What date did it arrive?",
    "Do you have the receipt number?",
    "Which card did you pay with?",
    "Should I refund to the same card?",
    "Anything else I should note on the order?",
]
ANSWERS = [
    "it was order ORD-7, twenty dollars",
    "the receipt says ORD-7 and forty two dollars",
    "order ORD-7, I paid nineteen fifty for it",
    "ORD-7 for the blue lamp, thirty dollars",
    "the number is ORD-7, total came to sixty",
    "ORD-7 again, the amount was eighty five",
]


def _asked_so_far(messages: list[dict]) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant" or m.get("tool_calls"))


def test_patience_takes_a_hazard_table_not_only_a_level_name():
    always = {"second": 1.0, "later": 1.0}
    never = (0.0, 0.0)
    # A table is read as (second, later), not fallen back to "normal".
    assert all(_user_walks_away(m, 3, questions=1, patience=always) for m in MSGS)
    assert all(_user_walks_away(m, 5, questions=4, patience=always) for m in MSGS)
    assert not any(_user_walks_away(m, 3, questions=1, patience=never) for m in MSGS)
    # The first question is always tried, whatever the table says.
    assert not any(_user_walks_away(m, 1, questions=0, patience=always) for m in MSGS)
    # Names still resolve to the package table; None is normal.
    assert patience_hazards("normal") == PATIENCE_HAZARDS["normal"]
    assert patience_hazards(None) == PATIENCE_HAZARDS["normal"]
    assert patience_hazards({"second": 0.2, "later": 0.5}) == (0.2, 0.5)
    assert patience_hazards([0.1, 0.4]) == (0.1, 0.4)
    assert patience_may_leave(never) is False
    assert patience_may_leave("short") is True
    assert patience_may_leave("endless") is False
    # Refusals name the fix.
    with pytest.raises(ValueError, match="outside 0"):
        patience_hazards({"second": 1.5, "later": 0.1})
    with pytest.raises(ValueError, match="level"):
        patience_hazards("forever")
    with pytest.raises(ValueError, match="second"):
        patience_hazards({"later": 0.5})


def _asking_agent():
    def fake_complete(_url, _model, messages, **kwargs):
        if kwargs.get("tools"):
            return {"content": QUESTIONS[_asked_so_far(messages) % len(QUESTIONS)]}
        body = str(messages[-1].get("content", ""))
        answered = sum(1 for q in QUESTIONS if q in body)
        return {"content": ANSWERS[answered % len(ANSWERS)]}

    return fake_complete


def _threads(monkeypatch, fake_complete, **kw) -> list[dict]:
    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    monkeypatch.setattr(
        "whileai.simulations.generate.agents.sample_turn_budget", lambda *_a, **_k: 12
    )
    agent = local_model("http://example", "m", tools=TOOLS, max_turns=12, **kw)
    return [agent(m) for m in MSGS]


def test_local_model_patience_table_reaches_the_loop(monkeypatch):
    # (1, 1): every thread ends at the agent's second question.
    rows = _threads(monkeypatch, _asking_agent(), patience=(1.0, 1.0))
    assert all(r.get("ended_by") == "user_left" for r in rows)
    assert all(sum(1 for s in r["steps"] if s.get("user")) == 1 for r in rows)
    # (0, 0): the person never leaves; the depth cap ends every thread.
    # Under the old code the table read as "normal" and about half left.
    rows = _threads(monkeypatch, _asking_agent(), patience={"second": 0.0, "later": 0.0})
    assert not any(r.get("ended_by") for r in rows)


def test_user_temperature_moves_every_simulated_user_line(monkeypatch):
    seen: dict[str, set[float]] = {"agent": set(), "user": set(), "human": set()}
    human_tool = {
        "type": "function",
        "kind": "human",
        "function": {
            "name": "ask_user",
            "parameters": {"type": "object", "properties": {"question": {"type": "string"}}},
        },
    }

    def fake_complete(_url, _model, messages, **kwargs):
        system = str(messages[0].get("content") or "")
        if system.startswith("You write only the human's next spoken line"):
            seen["user"].add(kwargs["temperature"])
            return {"content": "it was order ORD-7, twenty dollars"}
        if system.startswith("You are the person the assistant is helping"):
            seen["human"].add(kwargs["temperature"])
            return {"content": "the amount was forty two dollars, ORD-7"}
        seen["agent"].add(kwargs["temperature"])
        if _asked_so_far(messages) == 0:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "ask_user", "arguments": '{"question": "amount?"}'},
                    }
                ],
            }
        return {"content": ASK}

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    monkeypatch.setattr(
        "whileai.simulations.generate.agents.sample_turn_budget", lambda *_a, **_k: 12
    )
    tools = [*TOOLS, human_tool]
    local_model("http://example", "m", tools=tools, max_turns=12, user_temperature=0.05)(MSGS[0])
    assert seen["user"] == {0.05} and seen["human"] == {0.05}
    assert seen["agent"] == {LOCAL_MODEL_TEMPERATURE}
    # Without the knob the two user-side defaults stand, named.
    for key in seen:
        seen[key].clear()
    local_model("http://example", "m", tools=tools, max_turns=12)(MSGS[1])
    assert seen["user"] == {USER_TURN_TEMPERATURE}
    assert seen["human"] == {HUMAN_TOOL_TEMPERATURE}


def test_writer_temperature_pins_or_narrows_the_band(monkeypatch):
    assert writer_temperature_band(None) == (WRITER_TEMP_LO, WRITER_TEMP_HI)
    assert writer_temperature_band(0.2) == (0.2, 0.2)
    assert sample_writer_temperature(0, 3, band=(0.2, 0.2)) == 0.2
    assert all(0.5 <= sample_writer_temperature(7, r, band=(0.5, 0.6)) <= 0.6 for r in range(20))
    for bad in ((0.9, 0.1), "hot", -0.1, 3.0, (0.1, 0.2, 0.3)):
        with pytest.raises(ValueError, match="writer_temperature"):
            writer_temperature_band(bad)
    # The knob reaches the writer's request.
    temps: list[float] = []

    def fake_complete(_url, _model, _messages, **kwargs):
        temps.append(kwargs["temperature"])
        return {"content": "[]"}

    monkeypatch.setattr("whileai.simulations.generate.generator.complete", fake_complete)
    writer = ModelSimulator("openai:x", tools=TOOLS, policy=POLICY, writer_temperature=0.2)
    writer(0)
    writer(1)
    assert temps == [0.2, 0.2]
    gen = make_default_generator(
        TOOLS, policy=POLICY, simulator="openai:x", writer_temperature=(0.5, 0.7)
    )
    assert gen.model.writer_temperature == (0.5, 0.7)


def test_writer_temperature_is_an_advanced_key_and_is_validated_offline():
    # Consumed by the writer, not forwarded as an unknown keyword.
    data = simulate_offline(budget=4, per_round=4, advanced={"writer_temperature": (0.5, 0.7)})
    assert len(data.trajectories) == 4
    # A bad value is refused before any row is written, model or not.
    with pytest.raises(ValueError, match="writer_temperature"):
        simulate_offline(budget=4, per_round=4, advanced={"writer_temperature": 3.0})


def test_dead_and_duplicated_constants_are_gone():
    # A ten-point tier bag that contradicted HARD_SHARE, never read.
    assert not hasattr(diversity, "_TIER_BAG")
    assert not hasattr(generator, "_OUT_TOKENS_SMALL")
    # One home for the values two modules read.
    from whileai.simulations import defaults
    from whileai.simulations.generate import agents, anthropic_backend, embeddings

    assert not hasattr(anthropic_backend, "_TRANSIENT_TRIES")
    assert not hasattr(agents, "_TRANSIENT_TRIES")
    # the wire, not the value: both backends read the one constant
    assert agents.TRANSIENT_TRIES is defaults.TRANSIENT_TRIES
    assert anthropic_backend.TRANSIENT_TRIES is defaults.TRANSIENT_TRIES
    assert isinstance(embeddings.HASH_DIM, int) and embeddings.HASH_DIM > 0
    assert round(1.0 - diversity.ORDINARY_SHARE, 2) == diversity.HARD_SHARE
