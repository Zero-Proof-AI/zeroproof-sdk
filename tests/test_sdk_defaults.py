import inspect
import threading
import time

from tests.helpers import TOOLS, POLICY, scripted_agent
import zeroproof_simulations as zps


def test_default_budget_is_500():
    sig = inspect.signature(zps.simulate)
    params = sig.parameters
    public = [
        "agent", "spec", "tools", "system_prompt", "budget", "time_budget", "until",
        "mode", "situations", "requests_per_situation", "rollouts_per_request",
        "unique_situations", "grade", "llm_grade", "output", "advanced",
    ]
    named = [name for name, p in params.items()
             if p.kind is not inspect.Parameter.VAR_KEYWORD]
    assert named == public
    assert params["budget"].default == 1000
    assert params["until"].default == "compute"
    assert params["mode"].default == "explore"
    assert params["requests_per_situation"].default is None
    assert params["rollouts_per_request"].default is None
    assert params["unique_situations"].default is False
    assert params["situations"].default is None
    assert params["grade"].default is True
    assert params["llm_grade"].default is False
    assert params["time_budget"].default == 60
    assert params["spec"].default is None
    assert params["output"].default is None
    assert params["advanced"].default is None
    for moved in ("concurrency", "dimensions", "simulator", "backend",
                  "fault_rate", "risk", "texture", "max_turns", "avg_turns",
                  "temperature", "seed", "grader", "llm_spec",
                  "embedder", "unique", "repeats", "n", "phrasings", "repeat_policy",
                  "extra_situations", "rollouts_per_prompt", "policy"):
        assert moved not in params
    assert "length" not in params
    assert "turns" not in params
    assert "seconds" not in params


def test_system_prompt_alias_policy():
    data = zps.simulate(
        scripted_agent, tools=TOOLS, system_prompt=POLICY, budget=4, seed=0,
        grade=False, time_budget=None,
        advanced={"simulator": False, "concurrency": 4, "per_round": 6,
                  "mutate_failures": False})
    assert data.profile.system_prompt == POLICY
    alias = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=4, seed=0,
        grade=False, time_budget=None,
        advanced={"simulator": False, "concurrency": 4, "per_round": 6,
                  "mutate_failures": False})
    assert alias.profile.policy == POLICY
    try:
        zps.simulate(
            scripted_agent, tools=TOOLS, system_prompt="a", policy="b",
            budget=1, grade=False, time_budget=None,
            advanced={"simulator": False, "concurrency": 4})
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "system_prompt" in str(exc)


def test_adaptive_defaults_follow_allocator_not_n1_k1():
    data = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=4, grade=False,
        time_budget=None, mode="adaptive",
        advanced={"simulator": False, "concurrency": 4, "seed": 0,
                  "per_round": 6, "mutate_failures": False})
    plan = zps.adaptive_allocator(None, "compute")
    assert data.mode == "adaptive"
    assert data.requests_per_situation == plan["n_req"]
    assert data.rollouts_per_request == plan["k"]
    assert data.requests_per_situation > 1
    assert data.rollouts_per_request > 1


def test_time_budget_none_or_zero_is_unlimited():
    for knob in (0, None):
        data = zps.simulate(
            scripted_agent, tools=TOOLS, policy=POLICY, budget=8, seed=0,
            time_budget=knob, grade=False, concurrency=4, simulator=False,
            advanced={"per_round": 6, "mutate_failures": False})
        assert len(data.trajectories) == 8
        assert data.stopped_because != "time_budget"


def test_unique_no_duplicate_prompt():
    data = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=40, seed=0,
        unique=True, grade=False, concurrency=8, simulator=False,
        advanced={"per_round": 20, "mutate_failures": False})
    prompts = [t["prompt"] for t in data.trajectories]
    assert prompts
    assert len(prompts) == len(set(prompts))


def test_repeats_two_same_prompt():
    data = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=4, seed=0,
        repeats=2, grade=False, concurrency=4, simulator=False,
        advanced={"per_round": 12, "mutate_failures": False})
    assert len(data.trajectories) == 4
    prompts = [t["prompt"] for t in data.trajectories]
    assert len(set(prompts)) == 2
    for prompt in set(prompts):
        rows = [t for t in data.trajectories if t["prompt"] == prompt]
        assert len(rows) == 2
        assert {t["rollout_index"] for t in rows} == {0, 1}


def test_missing_spec_path_is_clear():
    try:
        zps.simulate(spec="specs/definitely-missing-xyz")
    except FileNotFoundError as exc:
        assert "definitely-missing-xyz" in str(exc)
        assert "spec.json" in str(exc)
        assert "\n" not in str(exc)
        assert str(exc).rstrip(".").count(".") == 1
    else:
        raise AssertionError("expected FileNotFoundError")


def test_empty_simulate_is_one_sentence():
    try:
        zps.simulate()
    except ValueError as exc:
        assert "agent" in str(exc) and "tools=" in str(exc)
        assert "\n" not in str(exc)
        assert str(exc).count(".") <= 1
    else:
        raise AssertionError("expected ValueError")


def test_github_example_spec_works():
    data = zps.simulate(
        scripted_agent, spec="specs/github", budget=6, seed=0,
        grade=True, simulator=False, concurrency=4,
        advanced={"per_round": 6, "mutate_failures": False})
    names = {(t.get("function") or t).get("name") for t in data.profile.tools}
    assert {"search_issues", "get_pr"} <= names
    row = data.rows()[0]
    assert {"prompt", "messages", "steps", "final_text", "scenario_id",
            "reward", "reason"} <= set(row)
    assert row["messages"][0] == {"role": "user", "content": row["prompt"]}
    assert "selection_reason" not in row
    assert "arm" not in row


def test_hosted_key_message_is_one_sentence(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from zeroproof_simulations.agents import missing_hosted_key
    msg = missing_hosted_key(
        "https://zeroproofai--stressd-vllm-serve.modal.run/v1")
    assert msg is not None
    assert "VLLM_API_KEY" in msg
    assert "\n" not in msg


def test_touch_hosted_gets_models(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    seen = {}

    class FakeResp:
        def read(self):
            return b'{"data":[]}'

    class FakeConn:
        def __init__(self, host, port=None, timeout=None):
            seen["host"] = host

        def request(self, method, path, headers=None):
            seen["method"] = method
            seen["path"] = path
            seen["auth"] = (headers or {}).get("Authorization")

        def getresponse(self):
            return FakeResp()

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(
        "zeroproof_simulations.agents.http.client.HTTPConnection", FakeConn)
    from zeroproof_simulations.agents import touch_hosted
    touch_hosted("http://127.0.0.1:9/v1")
    assert seen["method"] == "GET"
    assert seen["path"] == "/v1/models"
    assert seen["auth"] == "Bearer test-key"


def test_touch_hosted_skips_without_key(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    called = []
    monkeypatch.setattr(
        "zeroproof_simulations.agents.http.client.HTTPSConnection",
        lambda *a, **k: called.append(True))
    from zeroproof_simulations.agents import touch_hosted
    touch_hosted("https://zeroproofai--stressd-vllm-serve.modal.run/v1")
    assert called == []


def test_output_does_not_wipe_when_no_rows(tmp_path):
    dest = tmp_path / "out.jsonl"
    dest.write_text('{"keep": true}\n')
    try:
        zps.simulate(spec="specs/definitely-missing-xyz", output=str(dest))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
    assert dest.read_text().startswith('{"keep"')


def test_rollouts_per_prompt_alias():
    data = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=4, seed=0,
        rollouts_per_prompt=2, grade=False, concurrency=4, simulator=False,
        advanced={"per_round": 12, "mutate_failures": False})
    assert len(data.trajectories) == 4
    assert len({t["prompt"] for t in data.trajectories}) == 2


def test_scenario_producers_run_concurrently():
    lock = threading.Lock()
    active = 0
    peak = 0

    def writer(_dataset=None, index=0):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return [f"human request {index}-{i}" for i in range(20)]

    data = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=120, seed=0,
        unique=True, grade=False, concurrency=32,
        simulator=writer, until="budget_only",
        advanced={"scenario_concurrency": 4, "scenarios_per_request": 12,
                  "mutate_failures": False})

    assert peak >= 2
    assert len(data.trajectories) == 120


def test_unique_writer_flight_stays_small():
    lock = threading.Lock()
    peak = 0
    active = 0

    def writer(_dataset=None, index=0):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return [f"human request {index}-{i}" for i in range(16)]

    data = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=40, seed=0,
        unique=True, grade=False, concurrency=8, simulator=writer,
        until="budget_only",
        advanced={"mutate_failures": False})
    assert 1 <= peak <= 4
    assert len(data.trajectories) == 40


def test_refill_does_not_stall_inflight_rollouts():
    def writer(_dataset=None, index=0):
        if int(index or 0) > 0:
            time.sleep(1.4)
        return [f"human request {index}-{i}" for i in range(8)]

    t0 = time.monotonic()
    data = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=16, seed=0,
        unique=True, grade=False, concurrency=8, simulator=writer,
        until="budget_only", time_budget=3.0,
        advanced={"mutate_failures": False, "scenario_concurrency": 2})
    elapsed = time.monotonic() - t0
    assert len(data.trajectories) == 16
    assert elapsed < 2.8


def test_unique_dupes_do_not_stop_as_generator_exhausted():
    calls = {"n": 0}

    def writer(_dataset=None, index=0):
        calls["n"] += 1
        n = calls["n"]
        if n <= 3:
            return [f"human request {index}-{i}-{n}" for i in range(6)]
        return ["please refund this order now"] * 6

    data = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=200, seed=0,
        unique=True, grade=False, concurrency=8, simulator=writer,
        until="budget_only", time_budget=1.2,
        advanced={"mutate_failures": False, "scenario_concurrency": 4})
    assert data.stopped_because == "time_budget"
    assert data.stopped_because != "generator_exhausted"
    assert calls["n"] >= 8
    prompts = [t["prompt"] for t in data.trajectories]
    assert prompts
    assert len(prompts) == len(set(prompts))


def test_generator_exhausted_is_not_emitted():
    from pathlib import Path
    src = Path(zps.__file__).read_text()
    assert "generator_exhausted" not in src


def test_open_ended_weight_cannot_exceed_cap():
    from zeroproof_simulations import _reallocate, _SEARCH_ARMS
    from zeroproof_simulations.scenarios import cap_open_ended_weight

    clipped = cap_open_ended_weight({
        "structured": 0.2, "open_ended": 0.4, "llm_guided": 0.2,
        "behavior_targeted": 0.1, "failure_mutation": 0.1})
    assert clipped["open_ended"] <= 0.10 + 1e-9
    assert abs(sum(clipped.values()) - 1.0) < 1e-9
    weights = dict(_SEARCH_ARMS)
    hot = {arm: 1.0 if arm == "open_ended" else 0.0 for arm in weights}
    for _ in range(20):
        weights = _reallocate(weights, hot)
    assert weights["open_ended"] <= 0.10 + 1e-9
    assert weights["open_ended"] >= 0.05 - 1e-9


def test_open_ended_probe_families_stay_intact():
    from zeroproof_simulations.scenarios import _PROBE_FAMILIES, open_ended_probes

    names = [name for name, _ in _PROBE_FAMILIES]
    assert {"out_of_domain_factual", "creative", "garbage_input",
            "prompt_injection"} <= set(names)
    probes = open_ended_probes(TOOLS, POLICY, per_round=16, seed=0)
    blob = " ".join(probes).lower()
    assert "mongolia" in blob or "haiku" in blob or "asdf" in blob
    assert "file number" in blob
    assert "ignore" in blob


def test_writer_and_agent_default_to_hosted_qwen(monkeypatch):
    from zeroproof_simulations.agents import (
        DEFAULT_AGENT, default_agent_spec, default_simulator_spec)

    monkeypatch.delenv("ZEROPROOF_SURROGATE", raising=False)
    monkeypatch.delenv("ZEROPROOF_AGENT", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert default_simulator_spec() == DEFAULT_AGENT
    assert default_agent_spec() == DEFAULT_AGENT
