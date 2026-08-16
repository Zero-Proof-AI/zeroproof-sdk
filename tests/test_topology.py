"""Public simulate() knobs: aliases, N/n/k, and live-wired behavior."""
from __future__ import annotations

import json
import threading
import time

import pytest

from tests.helpers import TOOLS, POLICY, scripted_agent
import zeroproof_simulations as zps
from zeroproof_simulations.scenarios import SEARCH_ARMS, build_dimensions, reallocate_search_arms


def _offline(**kwargs):
    kw = dict(
        tools=TOOLS, policy=POLICY, seed=0, grade=False, concurrency=4,
        simulator=False, time_budget=None,
        advanced={"per_round": 8, "mutate_failures": False})
    kw.update(kwargs)
    return kw


def test_resolve_topology_defaults_and_aliases():
    adaptive = zps.resolve_topology()
    assert adaptive["mode"] == "adaptive"
    assert adaptive["repeat_policy"] == "adaptive"
    assert adaptive["n_req"] == 1
    assert adaptive["k"] == 1
    assert adaptive["k_explicit"] is False

    by_n = zps.resolve_topology(n=4)
    by_req = zps.resolve_topology(requests_per_situation=4)
    assert by_n["n_req"] == by_req["n_req"] == 4
    assert by_n["k"] == 1

    by_k = zps.resolve_topology(repeats=5)
    by_roll = zps.resolve_topology(rollouts_per_request=5)
    by_prompt = zps.resolve_topology(rollouts_per_prompt=5)
    assert by_k["k"] == by_roll["k"] == by_prompt["k"] == 5
    assert by_k["k_explicit"] is True

    unique = zps.resolve_topology(unique=True)
    flagged = zps.resolve_topology(unique_situations=True)
    none = zps.resolve_topology(repeat_policy="none")
    explore = zps.resolve_topology(mode="explore")
    assert unique["unique_situations"] is True
    assert flagged["unique_situations"] is True
    assert unique["n_req"] == unique["k"] == 1
    assert flagged["n_req"] == flagged["k"] == 1
    assert none["mode"] == explore["mode"] == "explore"
    assert none["unique_situations"] is True
    assert explore["n_req"] == explore["k"] == 1

    sft = zps.resolve_topology(mode="sft")
    assert sft["n_req"] == 3 and sft["k"] == 1
    rl = zps.resolve_topology(mode="rl")
    assert rl["n_req"] == 1 and rl["k"] == 3 and rl["k_explicit"] is False


def test_unique_situations_defaults_n_k_unless_set():
    base = zps.resolve_topology(unique_situations=True)
    assert base["n_req"] == 1 and base["k"] == 1
    with_k = zps.resolve_topology(unique_situations=True, rollouts_per_request=5)
    assert with_k["n_req"] == 1 and with_k["k"] == 5
    with_n = zps.resolve_topology(unique_situations=True, requests_per_situation=3)
    assert with_n["n_req"] == 3 and with_n["k"] == 1
    sft = zps.resolve_topology(mode="sft", unique_situations=True)
    assert sft["mode"] == "sft" and sft["n_req"] == 1 and sft["k"] == 1
    rl = zps.resolve_topology(mode="rl", unique_situations=True, rollouts_per_request=5)
    assert rl["mode"] == "rl" and rl["k"] == 5 and rl["n_req"] == 1
    alias = zps.resolve_topology(unique=True, repeats=2)
    assert alias["unique_situations"] is True
    assert alias["k"] == 2


def test_situations_int_is_n_list_is_seed():
    n_cards, seeds = zps._parse_situations_arg(3, ["extra opener"])
    assert n_cards == 3
    assert seeds == ["extra opener"]
    with pytest.raises(ValueError, match="seed_prompts"):
        zps.simulate(
            scripted_agent, situations=["not an N"], budget=2, **_offline())
    none, listed = zps._parse_situations_arg(None, ["where is order ORD-1"])
    assert none is None
    assert listed == ["where is order ORD-1"]

    capped = zps.simulate(
        scripted_agent, situations=2, requests_per_situation=1, repeats=1,
        budget=20, **_offline())
    assert capped.n_situations == 2
    assert capped.requests_per_situation == 1
    assert capped.rollouts_per_request == 1
    keys = {json.dumps(t["scenario_dimensions"], sort_keys=True, default=str)
            for t in capped.trajectories if t.get("scenario_dimensions")}
    assert len(keys) <= 2

    seeded = zps.simulate(
        scripted_agent, repeats=1, budget=6,
        extra_situations=["please refund ORD-9 now", "also look up ORD-1"],
        **_offline())
    prompts = {t["prompt"] for t in seeded.trajectories}
    assert any("ORD-9" in p for p in prompts)
    assert any("ORD-1" in p for p in prompts)


def test_public_n_is_requests_per_situation_not_completions(monkeypatch):
    seen: list[int] = []

    def fake_complete(_url, _model, _messages, **kwargs):
        seen.append(int(kwargs.get("n") or 1))
        idx = len(seen)
        return {"content": json.dumps([
            {"region_id": None, "message": f"where's my order ORD-{idx}"}])}

    monkeypatch.setattr("zeroproof_simulations.generator.complete", fake_complete)
    data = zps.simulate(
        scripted_agent, n=5, repeats=1, budget=6, seed=0, grade=False,
        concurrency=4, simulator="vllm:fake@http://127.0.0.1:9",
        time_budget=60,
        advanced={"per_round": 6, "mutate_failures": False})
    assert data.requests_per_situation == 5
    assert data.rollouts_per_request == 1
    assert seen
    assert all(n <= 3 for n in seen)

    seen.clear()
    data2 = zps.simulate(
        scripted_agent, n=1, repeats=1, budget=4, seed=0, grade=False,
        concurrency=4, simulator="vllm:fake@http://127.0.0.1:9",
        time_budget=60,
        advanced={"per_round": 6, "mutate_failures": False,
                  "completions_per_request": 6})
    assert data2.requests_per_situation == 1
    assert seen
    assert max(seen) <= 6
    assert max(seen) >= 3


def test_mode_sft_rl_explore_change_n_and_k():
    sft = zps.simulate(
        scripted_agent, mode="sft", budget=12, **_offline())
    assert sft.mode == "sft"
    assert sft.rollouts_per_request == 1
    assert sft.requests_per_situation == 3
    assert len({t["prompt"] for t in sft.trajectories}) == len(sft.trajectories)

    rl = zps.simulate(
        scripted_agent, mode="rl", budget=6, **_offline())
    assert rl.mode == "rl"
    assert rl.rollouts_per_request == 3
    prompts = [t["prompt"] for t in rl.trajectories]
    assert len(set(prompts)) == 2
    assert rl.allocator.get("explore", 0) + rl.allocator.get("expand", 0) >= 1

    explore = zps.simulate(
        scripted_agent, mode="explore", budget=10, **_offline())
    assert explore.repeat_policy == "none"
    assert explore.requests_per_situation == 1
    assert len({t["prompt"] for t in explore.trajectories}) == len(explore.trajectories)


def test_n_openers_are_not_k_rollouts():
    """n = different openers of one card. k = same opener, k trajectories."""
    from collections import Counter

    n_run = zps.simulate(
        scripted_agent, requests_per_situation=3, rollouts_per_request=1,
        budget=12, **_offline())
    k_run = zps.simulate(
        scripted_agent, requests_per_situation=1, rollouts_per_request=3,
        budget=12, **_offline())
    assert n_run.requests_per_situation == 3
    assert n_run.rollouts_per_request == 1
    assert k_run.requests_per_situation == 1
    assert k_run.rollouts_per_request == 3
    n_prompts = [t["prompt"] for t in n_run.trajectories]
    k_prompts = [t["prompt"] for t in k_run.trajectories]
    assert len(n_run.trajectories) == 12
    assert len(k_run.trajectories) == 12
    assert len(set(n_prompts)) == 12
    assert max(Counter(n_prompts).values()) == 1
    assert len(set(k_prompts)) == 4
    assert set(Counter(k_prompts).values()) == {3}


def test_unique_situations_keeps_new_cards_unless_n_k_set():
    plain = zps.simulate(
        scripted_agent, mode="sft", unique_situations=True, budget=8, **_offline())
    assert plain.unique_situations is True
    assert plain.requests_per_situation == 1
    assert plain.rollouts_per_request == 1
    assert len({t["prompt"] for t in plain.trajectories}) == len(plain.trajectories)

    rl = zps.simulate(
        scripted_agent, mode="rl", rollouts_per_request=5, budget=10, **_offline())
    assert rl.rollouts_per_request == 5
    assert rl.requests_per_situation == 1
    prompts = [t["prompt"] for t in rl.trajectories]
    assert len(set(prompts)) == 2
    from collections import Counter
    assert set(Counter(prompts).values()) == {5}

    alias = zps.simulate(
        scripted_agent, unique=True, budget=8, **_offline())
    assert alias.unique_situations is True
    assert alias.requests_per_situation == 1
    assert alias.rollouts_per_request == 1


def test_unique_is_topology_not_writer_flight():
    lock = threading.Lock()
    peaks = {"unique": 0, "default": 0}

    def make_writer(label):
        active = 0

        def writer(_dataset=None, index=0):
            nonlocal active
            with lock:
                active += 1
                peaks[label] = max(peaks[label], active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return [f"{label} request {index}-{i}" for i in range(12)]

        return writer

    u = zps.simulate(
        scripted_agent, unique=True, budget=24, seed=0, grade=False,
        concurrency=8, simulator=make_writer("unique"), until="compute",
        time_budget=None, advanced={"mutate_failures": False})
    d = zps.simulate(
        scripted_agent, unique=False, repeats=1, budget=24, seed=0, grade=False,
        concurrency=8, simulator=make_writer("default"), until="compute",
        time_budget=None, advanced={"mutate_failures": False})
    assert 1 <= peaks["unique"] <= 4
    assert 1 <= peaks["default"] <= 4
    assert peaks["unique"] == peaks["default"] or peaks["unique"] >= 2
    assert u.unique_situations is True
    assert d.unique_situations is False
    assert len({t["prompt"] for t in u.trajectories}) == len(u.trajectories)


def test_until_compute_vs_saturation_and_aliases():
    dims = {
        "tool": ["lookup_order"], "rule": ["unspecified"], "stance": ["ordinary"],
        "world_state": ["unspecified"], "tool_condition": ["success"],
        "history": ["fresh"],
    }
    compute = zps.simulate(
        lambda m: {"steps": [], "final_text": "ok"},
        budget=40, until="budget_only", dimensions=dims, repeats=1,
        **_offline())
    assert compute.stopped_because == "budget"
    assert compute.coverage["until"] == "compute"
    assert len(compute.trajectories) == 40

    halt = zps.simulate(
        lambda m: {"steps": [], "final_text": "ok"},
        budget=40, until="first", dimensions=dims, rollouts_per_request=5,
        **_offline())
    assert halt.stopped_because == "saturation"
    assert halt.coverage["until"] == "saturation"
    assert len(halt.trajectories) < 40


def test_budget_and_time_budget_are_compute_caps():
    rows = zps.simulate(
        scripted_agent, budget=7, repeats=1, **_offline())
    assert len(rows.trajectories) == 7
    assert rows.stopped_because == "budget"
    assert rows.budget == 7

    clock = zps.simulate(
        scripted_agent, budget=200, time_budget=0.15, repeats=1,
        concurrency=2, simulator=False, seed=0, grade=False,
        advanced={"per_round": 4, "mutate_failures": False})
    assert clock.stopped_because == "time_budget"
    assert len(clock.trajectories) < 200


def test_risk_aliases_fault_rate_and_stays_off_fail_arms():
    assert SEARCH_ARMS["failure_mutation"] <= 0.03 + 1e-9
    assert SEARCH_ARMS["behavior_targeted"] <= 0.03 + 1e-9
    stances = build_dimensions(TOOLS, POLICY)["stance"]
    assert "adversarial" in stances
    assert "boundary" in stances

    hot = {arm: 1.0 if arm == "failure_mutation" else 0.0 for arm in SEARCH_ARMS}
    weights = dict(SEARCH_ARMS)
    for _ in range(20):
        weights = reallocate_search_arms(weights, hot)
    assert weights["failure_mutation"] <= 0.08 + 1e-9
    assert weights["failure_mutation"] < 0.15

    off = zps.simulate(
        scripted_agent, risk=0, repeats=1, budget=16, dimensions={
            "tool": ["lookup_order"], "rule": ["unspecified"],
            "stance": ["ordinary"], "world_state": ["unspecified"],
            "tool_condition": ["timeout"], "history": ["fresh"],
        }, **_offline())
    on = zps.simulate(
        scripted_agent, risk=1, repeats=1, budget=16, dimensions={
            "tool": ["lookup_order"], "rule": ["unspecified"],
            "stance": ["ordinary"], "world_state": ["unspecified"],
            "tool_condition": ["timeout"], "history": ["fresh"],
        }, **_offline())
    assert sum(1 for t in off.trajectories if t.get("faults")) == 0
    assert sum(1 for t in on.trajectories if t.get("faults")) > 0
    assert off.arm_weights["failure_mutation"] <= 0.08 + 1e-9


def test_seed_grade_grader_dimensions_texture_output(tmp_path):
    a = zps.simulate(scripted_agent, repeats=1, budget=8, **_offline())
    b = zps.simulate(scripted_agent, repeats=1, budget=8, **_offline(seed=1))
    assert [t["prompt"] for t in a.trajectories] != [t["prompt"] for t in b.trajectories]

    graded = zps.simulate(
        scripted_agent, grade=True, repeats=1, budget=4,
        tools=TOOLS, policy=POLICY, seed=0, concurrency=4, simulator=False,
        time_budget=None, advanced={"per_round": 6, "mutate_failures": False})
    raw = zps.simulate(
        scripted_agent, grade=False, repeats=1, budget=4,
        tools=TOOLS, policy=POLICY, seed=0, concurrency=4, simulator=False,
        time_budget=None, advanced={"per_round": 6, "mutate_failures": False})
    assert all(t.get("reward") is not None for t in graded.trajectories)
    assert all(t.get("reward") is None for t in raw.trajectories)

    scored = zps.simulate(
        scripted_agent, grade=True, grader=lambda _t: 0.25, repeats=1, budget=4,
        tools=TOOLS, policy=POLICY, seed=0, concurrency=4, simulator=False,
        time_budget=None, advanced={"per_round": 6, "mutate_failures": False})
    assert all(t["reward"] == 0.25 for t in scored.trajectories)

    tiny = {
        "tool": ["lookup_order"], "rule": ["unspecified"], "stance": ["ordinary"],
        "world_state": ["unspecified"], "tool_condition": ["success"],
        "history": ["fresh"],
    }
    dimmed = zps.simulate(
        scripted_agent, dimensions=tiny, repeats=1, budget=10, **_offline())
    cells = {json.dumps(t.get("scenario_dimensions"), sort_keys=True, default=str)
             for t in dimmed.trajectories if t.get("scenario_dimensions")}
    assert cells
    assert all("lookup_order" in c for c in cells)

    dest = tmp_path / "out.jsonl"
    written = zps.simulate(
        scripted_agent, output=str(dest), repeats=1, budget=3, **_offline())
    assert dest.exists()
    assert len(dest.read_text().splitlines()) == len(written.trajectories)


def test_texture_reaches_writer_tag_draw(monkeypatch):
    from zeroproof_simulations.diversity import sample_cell_tags as orig

    seen: list[float] = []

    def tracked(seed, round_index, key, assignment=None, texture_rate=0.08, **kw):
        seen.append(float(texture_rate))
        return orig(seed, round_index, key, assignment,
                    texture_rate=texture_rate, **kw)

    monkeypatch.setattr("zeroproof_simulations.generator.sample_cell_tags", tracked)

    def fake_complete(_url, _model, _messages, **_kwargs):
        return {"content": json.dumps([
            {"region_id": None, "message": "where's my order ORD-1"}])}

    monkeypatch.setattr("zeroproof_simulations.generator.complete", fake_complete)
    zps.simulate(
        scripted_agent, texture=0.0, repeats=1, budget=3, seed=0, grade=False,
        concurrency=2, simulator="vllm:fake@http://127.0.0.1:9",
        time_budget=None, advanced={"per_round": 4, "mutate_failures": False})
    assert seen
    assert all(rate == 0.0 for rate in seen)
    seen.clear()
    zps.simulate(
        scripted_agent, texture=1.0, repeats=1, budget=3, seed=0, grade=False,
        concurrency=2, simulator="vllm:fake@http://127.0.0.1:9",
        time_budget=None, advanced={"per_round": 4, "mutate_failures": False})
    assert seen
    assert all(rate == 1.0 for rate in seen)


def test_avg_turns_max_turns_concurrency_temperature_backend(monkeypatch):
    seen_temp = []
    seen_backend = []

    def fake_complete(_url, _model, messages, **kwargs):
        seen_temp.append(kwargs.get("temperature"))
        last = messages[-1] if messages else {}
        if last.get("role") == "user":
            return {"content": "Order ORD-1 is packed."}
        return {"content": "done"}

    def fake_local(url, model, **kwargs):
        seen_backend.append((url, model, kwargs.get("max_turns"),
                             kwargs.get("avg_turns"), kwargs.get("temperature")))

        def agent(message):
            return {"steps": [], "final_text": "ok"}

        return agent

    monkeypatch.setattr("zeroproof_simulations.local_model", fake_local)
    zps.simulate(
        tools=TOOLS, policy=POLICY, backend="vllm:fake@http://127.0.0.1:9",
        max_turns=6, avg_turns=2, temperature=0.2, budget=3, repeats=1,
        grade=False, concurrency=2, simulator=False, seed=0, time_budget=None,
        advanced={"per_round": 4, "mutate_failures": False})
    assert seen_backend
    assert seen_backend[0][2] == 6
    assert seen_backend[0][3] == 2.0
    assert seen_backend[0][4] == 0.2

    lock = threading.Lock()
    peak = 0
    active = 0

    def slow(_message):
        nonlocal peak, active
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return {"steps": [], "final_text": "ok"}

    zps.simulate(
        slow, tools=TOOLS, policy=POLICY, budget=8, repeats=1, grade=False,
        concurrency=3, simulator=False, seed=0, time_budget=None,
        advanced={"per_round": 8, "mutate_failures": False})
    assert 2 <= peak <= 3


def test_embedder_is_used_for_selection():
    called = {"n": 0}

    class Spy:
        name = "spy"
        semantic = False

        def embed(self, texts):
            called["n"] += len(texts)
            return [[float(i), 0.0, 1.0] for i, _ in enumerate(texts)]

    data = zps.simulate(
        scripted_agent, embedder=Spy(), repeats=1, budget=8, **_offline())
    assert called["n"] > 0
    assert data.embedder_name == "spy"


def test_spec_tools_policy_agent_simulator_change_rows():
    github = zps.simulate(
        scripted_agent, spec="specs/github", budget=4, repeats=1,
        grade=False, concurrency=4, simulator=False, seed=0, time_budget=None,
        advanced={"per_round": 6, "mutate_failures": False})
    names = {(t.get("function") or t).get("name") for t in github.profile.tools}
    assert "search_issues" in names

    def writer(_dataset=None, index=0):
        return [f"custom writer line {index}-{i}" for i in range(6)]

    custom = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=4, repeats=1,
        grade=False, concurrency=4, simulator=writer, seed=0, time_budget=None,
        advanced={"per_round": 6, "mutate_failures": False})
    assert any("custom writer line" in t["prompt"] for t in custom.trajectories)


def test_k_does_not_clone_followups(monkeypatch):
    follow_n = {"n": 0}

    def fake_complete(_url, _model, messages, **kwargs):
        if not kwargs.get("tools"):
            follow_n["n"] += 1
            return {"content": f"also check refund {follow_n['n']}"}
        last = messages[-1] if messages else {}
        if last.get("role") == "user" and "refund" in str(last.get("content", "")):
            return {"content": f"Refund note {last.get('content')}"}
        return {"content": "Order ORD-1 is packed. Want me to check the refund too?"}

    monkeypatch.setattr("zeroproof_simulations.agents.complete", fake_complete)
    monkeypatch.setattr("zeroproof_simulations.agents.sample_turn_budget",
                        lambda *_a, **_k: 8)
    data = zps.simulate(
        tools=TOOLS, policy=POLICY,
        extra_situations=["where is my order ORD-1"],
        budget=2, repeats=2, grade=False, concurrency=2,
        simulator=False, backend="vllm:fake@http://127.0.0.1:9",
        seed=0, time_budget=None, max_turns=8, avg_turns=4,
        advanced={"per_round": 4, "mutate_failures": False})
    assert len(data.trajectories) == 2
    assert len({t["prompt"] for t in data.trajectories}) == 1
    follows = []
    for t in data.trajectories:
        follows.extend(s.get("user") for s in (t.get("steps") or [])
                       if isinstance(s, dict) and s.get("user"))
    assert len(set(follows)) == 2
    assert {"also check refund 1", "also check refund 2"} == set(follows)


def test_adaptive_allocator_records_explore_expand_verify():
    data = zps.simulate(
        scripted_agent, mode="adaptive", budget=16, **_offline())
    assert data.mode == "adaptive"
    assert data.allocator
    assert data.allocator.get("explore", 0) >= 1
    assert data.coverage.get("mode") == "adaptive"
    assert data.coverage.get("requests_per_situation") == data.requests_per_situation
    assert data.coverage.get("rollouts_per_request") == data.rollouts_per_request
