"""The last mile of "no hardcoding": every knob the four lanes named
reaches the code from one ``simulate()`` call, and the values two
modules read are one object. Each test fails on a tree where the wire
is missing (the keyword is unknown or the constant has two homes)."""

from __future__ import annotations

import inspect

import pytest

import whileai.simulations as wai
from tests.helpers import POLICY, TOOLS, offline, scripted_agent
from whileai.simulations import data as data_module
from whileai.simulations import defaults, export
from whileai.simulations.generate import adapters, agents, diversity, scenarios
from whileai.simulations.run.config import resolve_run_config
from whileai.simulations.score import privileged
from whileai.simulations.world.sandbox import FAULT_MODES, WorldOptions

BACKEND = "vllm:agent-model@http://127.0.0.1:9"


def _cfg(mode: str = "explore", **advanced):
    return resolve_run_config(None, tools=TOOLS, system_prompt=POLICY, mode=mode, advanced=advanced)


def _capture_runner(monkeypatch, where: str) -> list[dict]:
    """Replace ``local_model`` at ``where`` with a stub that records its
    keyword arguments and answers every rollout with one line."""
    seen: list[dict] = []

    def fake_local(url, model, **kwargs):
        seen.append(kwargs)

        def agent(message):
            return {"steps": [], "final_text": "ok"}

        return agent

    monkeypatch.setattr(where, fake_local)
    return seen


# ------------------------------------------------------------ (a) world


def test_world_options_reach_the_mock_world_from_advanced(monkeypatch):
    """A fault mode added through advanced={"world": ...} answers the
    agent's tool call inside a run."""

    def rate_limited(env, tool, arguments):
        return {"status": "rate_limited", "retry_after_s": 30}

    calls = {"n": 0}

    def fake_complete(_url, _model, messages, **kwargs):
        calls["n"] += 1
        if kwargs.get("tools") and messages[-1]["role"] == "user":
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{calls['n']}",
                        "type": "function",
                        "function": {"name": "lookup_order", "arguments": '{"order_id":"ORD-7"}'},
                    }
                ],
            }
        return {"content": "Done."}

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    world = {"fault_modes": {name: rate_limited for name in FAULT_MODES}}
    data = wai.simulate(
        backend=BACKEND,
        budget=16,
        risk=1.0,
        avg_turns=2,
        min_user_turns=1,
        patience="endless",
        **offline(advanced={"world": world, "prefer_success": False}),
    )
    statuses = {
        str(step.get("result", {}).get("status"))
        for row in data.trajectories
        for step in row.get("steps", [])
        if step.get("tool")
    }
    assert "rate_limited" in statuses, statuses
    # and none of the shipped modes fired in its place
    assert not statuses & {"timeout", "permission_denied"}, statuses


def test_world_options_land_on_the_runner_and_on_local_model(monkeypatch):
    seen = _capture_runner(monkeypatch, "whileai.simulations.run.engine.local_model")
    wai.simulate(backend=BACKEND, budget=2, **offline(advanced={"world": {"search_hits": (2, 2)}}))
    assert seen and isinstance(seen[0]["world_options"], WorldOptions)
    assert seen[0]["world_options"].search_hits == (2, 2)
    # a run with no world key sends nothing, so local_model keeps its defaults
    seen.clear()
    wai.simulate(backend=BACKEND, budget=2, **offline())
    assert seen and "world_options" not in seen[0]
    assert "world_options" in inspect.signature(agents.local_model).parameters


def test_resolve_passes_world_and_user_temperature_to_local_model(monkeypatch):
    seen = _capture_runner(monkeypatch, "whileai.simulations.generate.adapters.local_model")
    adapters.resolve(
        BACKEND, tools=TOOLS, user_temperature=0.3, world_options={"exists_share": 1.0}
    )
    assert seen[0]["user_temperature"] == 0.3
    assert seen[0]["world_options"] == {"exists_share": 1.0}


# --------------------------------------------- (b) patience / user_temperature


def test_patience_table_and_user_temperature_reach_the_loop_from_simulate(monkeypatch):
    seen = _capture_runner(monkeypatch, "whileai.simulations.run.engine.local_model")
    wai.simulate(
        backend=BACKEND,
        budget=2,
        **offline(advanced={"patience": {"second": 0.1, "later": 0.2}, "user_temperature": 0.3}),
    )
    assert seen[0]["patience"] == (0.1, 0.2)
    assert seen[0]["user_temperature"] == 0.3
    # the pair form and a level name both still work
    assert _cfg(patience=(0.4, 0.5)).patience == (0.4, 0.5)
    assert _cfg(patience="Short").patience == "short"
    assert _cfg().patience == "normal" and _cfg().user_temperature is None


def test_the_run_records_its_knobs_and_the_report_shows_them():
    """The report says what the run ran under: every RunKnobs field plus
    patience, user_temperature and the world options. Fails on a tree
    where coverage carries none of them."""
    from dataclasses import fields

    from whileai.simulations.defaults import RunKnobs

    data = wai.simulate(
        scripted_agent,
        budget=2,
        **offline(
            advanced={
                "patience": (0.6, 0.9),
                "user_temperature": 0.3,
                "closing_margin": 3.5,
                "world": {"search_hits": (2, 2)},
            }
        ),
    )
    report = data.report()
    assert report["patience"] == (0.6, 0.9)
    assert report["user_temperature"] == 0.3
    knobs = report["knobs"]
    assert set(knobs) == {f.name for f in fields(RunKnobs)}
    assert knobs["closing_margin"] == 3.5
    assert knobs["pass_threshold"] == defaults.PASS_THRESHOLD
    world = report["world"]
    assert world["search_hits"] == (2, 2)
    assert world["fault_modes"] == sorted(FAULT_MODES)
    assert "timeout" in world["condition_modes"]
    assert not any(callable(v) for v in world.values())
    # the defaults are recorded too, so a run with no advanced= says so
    plain = wai.simulate(scripted_agent, budget=2, **offline()).report()
    assert plain["patience"] == "normal" and plain["user_temperature"] is None
    assert plain["knobs"]["closing_margin"] == defaults.knob_default("closing_margin")


def test_the_run_records_hard_share_and_fault_rate():
    """The two ``simulate()`` keywords a run used to keep only in
    ``search["tier_mix"]`` (hard_share) or nowhere (fault_rate) are on the
    coverage record: hard_share as asked, None when not set; fault_rate as
    resolved. Fails with KeyError on a tree that records neither."""
    report = wai.simulate(
        scripted_agent, mode="rl", repeats=4, budget=32, hard_share=0.8, fault_rate=1.0, **offline()
    ).report()
    assert report["hard_share"] == 0.8
    assert report["fault_rate"] == 1.0
    # left to the run: hard_share is None (the tier mix says what was drawn),
    # fault_rate is the rate the mode resolved, not the world's default
    plain = wai.simulate(scripted_agent, mode="rl", repeats=4, budget=32, **offline()).report()
    assert plain["hard_share"] is None
    assert plain["fault_rate"] == defaults.RL_FAULT_RATE
    assert plain["world"]["default_fault_rate"] != plain["fault_rate"]


def test_a_fault_mode_added_through_advanced_world_fires_in_a_run():
    """A fault mode the caller adds reaches the coverage grid: name it as a
    tool_condition and rows carry it. Fails on a tree where scenarios.py
    hardcodes the condition-to-mode table."""

    def rate_limited(env, tool, arguments):
        return {"status": "error", "error": "429 too many requests", "retry_after_s": 30}

    data = wai.simulate(
        scripted_agent,
        budget=12,
        fault_rate=1.0,
        dimensions={"tool_condition": ["rate_limited"]},
        **offline(
            per_round=12,
            advanced={
                "prefer_success": False,
                "world": {"fault_modes": {**FAULT_MODES, "rate_limited": rate_limited}},
            },
        ),
    )
    modes = {(row.get("faults") or {}).get("*", {}).get("mode") for row in data.trajectories}
    assert "rate_limited" in modes
    # the same table steers the shipped conditions: a condition pointed at
    # a different shipped mode carries that mode
    data = wai.simulate(
        scripted_agent,
        budget=12,
        fault_rate=1.0,
        dimensions={"tool_condition": ["timeout"]},
        **offline(
            per_round=12,
            advanced={"prefer_success": False, "world": {"condition_modes": {"timeout": "stale"}}},
        ),
    )
    modes = {(row.get("faults") or {}).get("*", {}).get("mode") for row in data.trajectories}
    assert modes <= {"stale", None} and "stale" in modes
    # a condition the world does not know is refused before any rollout
    with pytest.raises(ValueError, match="tool_condition"):
        wai.simulate(
            scripted_agent,
            budget=2,
            dimensions={"tool_condition": ["rate_limited"]},
            **offline(),
        )


def test_patience_and_user_temperature_are_validated_before_any_model_call():
    with pytest.raises(ValueError, match="second"):
        _cfg(patience={"second": 1.5, "later": 0.2})
    with pytest.raises(ValueError, match="not a level"):
        _cfg(patience="forever")
    with pytest.raises(ValueError, match="user_temperature"):
        _cfg(user_temperature=3.0)


# ------------------------------------------------------------ (c) avg_turns


def test_avg_turns_has_one_default_for_simulate_local_model_and_the_sampler():
    assert diversity.DEFAULT_AVG_TURNS is defaults.DEFAULT_AVG_TURNS
    assert inspect.signature(agents.local_model).parameters["avg_turns"].default == (
        defaults.DEFAULT_AVG_TURNS
    )
    assert inspect.signature(adapters.resolve).parameters["avg_turns"].default == (
        defaults.DEFAULT_AVG_TURNS
    )
    assert _cfg().avg_turns == defaults.DEFAULT_AVG_TURNS == 12.0


# ------------------------------------------------- (d) one value, one home


def test_shared_constants_have_one_home():
    assert defaults.MAX_SAMPLES_PER_CALL is defaults.MAX_COMPLETIONS_PER_REQUEST
    assert defaults.RL_ROLLOUTS_PER_ASK is defaults.RL_ROLLOUTS_PER_PROMPT
    assert not hasattr(scenarios, "DEFAULT_FAULT_RATE") or (
        scenarios.DEFAULT_FAULT_RATE is defaults.DEFAULT_FAULT_RATE
    )
    assert _cfg().fault_rate == defaults.DEFAULT_FAULT_RATE
    assert _cfg(mode="rl").fault_rate == defaults.RL_FAULT_RATE
    assert inspect.signature(privileged.leak_report).parameters["min_len"].default == (
        defaults.LEAK_MIN_QUOTE_CHARS
    )


def test_export_counts_pass_and_fail_at_the_shared_threshold(monkeypatch):
    rows = [
        {"prompt": "p", "reward": 0.7, "final_text": "a", "steps": []},
        {"prompt": "p", "reward": 0.2, "final_text": "b", "steps": []},
    ]
    n0 = sum(1 for r in rows if r["reward"] < defaults.PASS_THRESHOLD)
    assert n0 == 1
    # the exporter reads the one constant: move it and the count moves
    monkeypatch.setattr(export, "PASS_THRESHOLD", 0.9)
    groups = [dict(r) for r in rows]
    export._stamp_groups(groups)
    assert (groups[0]["n0"], groups[0]["n1"]) == (2, 0)
    monkeypatch.setattr(export, "PASS_THRESHOLD", defaults.PASS_THRESHOLD)
    groups = [dict(r) for r in rows]
    export._stamp_groups(groups)
    assert (groups[0]["n0"], groups[0]["n1"]) == (1, 1)


# ------------------------------------------------------- (e) judge knobs


def test_grade_forwards_payload_chars_and_max_tokens_to_the_judge(monkeypatch):
    seen: list[dict] = []

    def fake_apply(rows, **kwargs):
        seen.append(kwargs)
        return {"graded": 0, "n0": 0, "n1": 0}

    monkeypatch.setattr(data_module, "apply_grade_llm", fake_apply)
    monkeypatch.setattr(data_module, "require_judge_key", lambda *a, **k: None)
    data = wai.simulate(scripted_agent, budget=4, **offline())
    data.grade(rubric="be right", payload_chars=1234, max_tokens=77)
    assert seen[-1]["payload_chars"] == 1234 and seen[-1]["max_tokens"] == 77
    data.grade_llm(rubric="be right")
    assert seen[-1]["payload_chars"] == defaults.JUDGE_PAYLOAD_CHARS
    assert seen[-1]["max_tokens"] == defaults.JUDGE_MAX_TOKENS
    data_module.grade_llm(list(data.trajectories), payload_chars=99, max_tokens=5)
    assert seen[-1]["payload_chars"] == 99 and seen[-1]["max_tokens"] == 5
