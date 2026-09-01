"""behavior_state v2: evidence in, allocation out. Regions are named
behavioral predicates (markers, fault-response, capability fallback),
never coordinate tuples; history buckets by model version; targeting
memory distinguishes solved from lucky and flags coordinate rotation."""
from zeroproof_simulations.traces import behavior_state


def _mrow(marker, ok, version, tool="edit_file", dims=None):
    return {"prompt": f"p-{marker}-{version}-{ok}",
            "model_version": version,
            "scores": {marker: 1 if ok else 0},
            "scenario_dimensions": dims or {"tool": tool,
                                            "stance": "hurried"},
            "steps": [{"tool": tool, "arguments": {},
                       "result": {"status": "ok"}}],
            "final_text": "done"}


def test_three_round_lifecycle_with_model_versions():
    rows = []
    # round v0: edits_before_reading failing hard; claims_fix failing some
    rows += [_mrow("edits_before_reading", False, "v0")] * 4
    rows += [_mrow("edits_before_reading", True, "v0")] * 4
    rows += [_mrow("claims_fix_without_test", False, "v0")] * 2
    rows += [_mrow("claims_fix_without_test", True, "v0")] * 8
    # round v1: ebr targeted and improved; cfwt targeted, did not move
    rows += [_mrow("edits_before_reading", True, "v1")] * 7
    rows += [_mrow("claims_fix_without_test", False, "v1")] * 2
    rows += [_mrow("claims_fix_without_test", True, "v1")] * 4
    # brand new failure appears in the latest round
    rows += [_mrow("timeout_no_recovery", False, "v1")] * 2

    state = behavior_state(rows, targeted=["edits_before_reading",
                                           "claims_fix_without_test"])
    assert state["buckets"] == ["v0", "v1"]
    by = {r["region"]: r for r in state["regions"]}
    assert by["edits_before_reading"]["status"] == "solved"
    assert by["claims_fix_without_test"]["status"] == "persistent"
    assert by["claims_fix_without_test"]["rotate_coordinates"] is True
    assert by["timeout_no_recovery"]["status"] == "new"
    shares = {k: v["budget_share"] for k, v in by.items()}
    assert shares["timeout_no_recovery"] > shares["edits_before_reading"]
    assert shares["claims_fix_without_test"] > shares["edits_before_reading"]
    hist = by["edits_before_reading"]["history"]
    assert hist[0]["fail_rate"] == 0.5 and hist[1]["fail_rate"] == 0.0


def test_region_is_predicate_not_tuple_and_recipe_collects_coords():
    rows = [_mrow("edits_before_reading", False, "v0",
                  dims={"tool": "edit_file", "stance": "hurried"}),
            _mrow("edits_before_reading", False, "v0",
                  dims={"tool": "edit_file", "stance": "ordinary",
                        "world_state": "entity exists"})]
    state = behavior_state(rows)
    region = state["regions"][0]
    assert region["region"] == "edits_before_reading"
    assert region["kind"] == "marker"
    assert sorted(region["recipe"]["stance"]) == ["hurried", "ordinary"]
    assert region["recipe"]["tool"] == ["edit_file"]


def test_fault_response_and_capability_fallback():
    fault_fail = {"prompt": "a", "reward": 0, "model_version": "v0",
                  "steps": [{"tool": "search", "arguments": {},
                             "result": {"status": "timeout"}}],
                  "final_text": "gave up"}
    fault_ok = {"prompt": "b", "reward": 1, "model_version": "v0",
                "steps": [{"tool": "search", "arguments": {},
                           "result": {"status": "timeout"}}],
                "final_text": "retried and answered"}
    plain_fail = {"prompt": "c", "reward": 0, "model_version": "v0",
                  "steps": [], "final_text": "wrong"}
    state = behavior_state([fault_fail, fault_ok, plain_fail])
    by = {r["region"]: r for r in state["regions"]}
    # single model version -> time-half buckets: fail lands old, pass recent
    rates = [h["fail_rate"] for h in by["recover_after_timeout"]["history"]]
    assert rates == [1.0, 0.0]
    assert by["recover_after_timeout"]["status"] == "improving"
    assert by["task:no-tool"]["kind"] == "capability"


def test_exploration_reserved_and_saturation():
    loud = [_mrow("loud_marker", False, "v0")] * 40
    quiet = [_mrow("quiet_marker", False, "v0")]
    state = behavior_state(loud + quiet)
    assert state["exploration_share"] >= 0.2
    shares = {r["region"]: r["budget_share"] for r in state["regions"]}
    assert shares["loud_marker"] < 3 * shares["quiet_marker"]
    total = sum(shares.values())
    assert abs(total + state["exploration_share"] - 1.0) < 0.01


def test_empty_history():
    state = behavior_state([])
    assert state["regions"] == [] and state["exploration_share"] == 1.0


def test_behavior_state_rides_trace_fed_simulate():
    from tests.helpers import POLICY, TOOLS, scripted_agent
    from zeroproof_simulations.traces import simulate_from_traces
    traces = [
        {"prompt": "where is order 4412",
         "steps": [{"tool": "lookup_order", "arguments": {"order_id": "4412"},
                    "result": {"status": "not_found"}}],
         "final_text": "I could not find that order.", "reward": 0},
        {"prompt": "refund order 9911 now",
         "steps": [{"tool": "create_refund", "arguments": {"order_id": "9911"},
                    "result": {"status": "timeout"}}],
         "final_text": "The refund request timed out.", "reward": 0},
    ]
    data = simulate_from_traces(
        traces, scripted_agent, tools=TOOLS, policy=POLICY, mode="explore",
        budget=4, seed=0, grade=False, concurrency=4, simulator=False,
        time_budget=20, advanced={"per_round": 4, "mutate_failures": False})
    state = data.search["behavior_state"]
    assert state["traces"] == 2
    names = {r["region"] for r in state["regions"]}
    assert "recover_after_not_found" in names
    assert "recover_after_timeout" in names
    assert state["exploration_share"] >= 0.2


def test_rows_carry_dims_and_model_version_everywhere():
    from tests.helpers import POLICY, TOOLS, scripted_agent
    from zeroproof_simulations import simulate
    data = simulate(agent=scripted_agent, tools=TOOLS, system_prompt=POLICY,
                    mode="explore", budget=16, seed=0, grade=False,
                    concurrency=4, simulator=False, time_budget=20,
                    advanced={"per_round": 8, "mutate_failures": False,
                              "seed_prompts": ["where is order 12"],
                              "model_version": "v3-adapter"})
    assert all(t.get("scenario_dimensions") for t in data.trajectories)
    assert all(t.get("model_version") == "v3-adapter"
               for t in data.trajectories)


def test_otel_reads_model_version():
    from zeroproof_simulations import rows_from_otel
    spans = [{"traceId": "t1", "spanId": "r", "name": "agent x",
              "startedMs": 1000,
              "attributes": {
                  "zeroproof.model_version": "identity-v5-adapter",
                  "gen_ai.input.messages":
                      '[{"role": "user", "content": "hi"}]',
                  "gen_ai.output.messages":
                      '[{"role": "assistant", "content": "hello"}]'}}]
    rows = rows_from_otel({"spans": spans})
    assert rows[0]["model_version"] == "identity-v5-adapter"


def test_allocation_actually_shifts_generation():
    """The missing link, closed: region budget shares must change what
    the simulator generates, not just be reported."""
    from tests.helpers import POLICY, TOOLS, scripted_agent
    from zeroproof_simulations import simulate
    from zeroproof_simulations.traces import simulate_from_traces
    hot = [{"prompt": f"refund order {i} failed again",
            "steps": [{"tool": "create_refund", "arguments": {"order_id": str(i)},
                       "result": {"status": "timeout"}}],
            "final_text": "The refund timed out.", "reward": 0,
            "model_version": "v0"} for i in range(4)]
    common = dict(tools=TOOLS, mode="explore", budget=32, seed=3,
                  grade=False, concurrency=4, simulator=False,
                  time_budget=30,
                  advanced={"per_round": 8, "mutate_failures": False})

    def refund_share(data):
        n = sum(1 for t in data.trajectories
                if (t.get("scenario_dimensions") or {}).get("tool")
                == "create_refund")
        return n / max(1, len(data.trajectories))

    cold = simulate(agent=scripted_agent, system_prompt=POLICY, **common)
    aimed = simulate_from_traces(hot, scripted_agent, policy=POLICY, **common)
    state = aimed.search["behavior_state"]
    assert state["applied"] is True
    assert any(r["region"] == "recover_after_timeout"
               for r in state["regions"])
    assert refund_share(aimed) > refund_share(cold), (
        f"allocation had no effect: aimed {refund_share(aimed):.2f} "
        f"vs cold {refund_share(cold):.2f}")


def test_region_progress_same_rules_both_sides():
    """Trace regions re-measured on generated graded rows: the
    hill-climb readout speaks one vocabulary."""
    from tests.helpers import POLICY, TOOLS, scripted_agent
    from zeroproof_simulations.traces import simulate_from_traces
    traces = [{"prompt": "refund order 9911 now",
               "steps": [{"tool": "create_refund", "arguments": {"order_id": "9911"},
                          "result": {"status": "timeout"}}],
               "final_text": "The refund request timed out.", "reward": 0,
               "model_version": "v0"}]

    def grader(row):
        return {"reward": 1, "reason": "ok"}

    data = simulate_from_traces(
        traces, scripted_agent, tools=TOOLS, policy=POLICY, mode="explore",
        budget=12, seed=1, grade=False, grader=grader, concurrency=4,
        simulator=False, time_budget=20,
        advanced={"per_round": 6, "mutate_failures": False})
    prog = data.search["behavior_state"]["region_progress"]
    rec = next(p for p in prog if p["region"] == "recover_after_timeout")
    assert rec["trace_fail_rate"] == 1.0
    if rec["generated_n"]:
        assert rec["generated_fail_rate"] is not None
