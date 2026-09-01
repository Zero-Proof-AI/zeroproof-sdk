"""behavior_state: trace history in, budget allocation out. Each of the
lifecycle rules gets a synthetic timeline; exploration is always
reserved; loud repeated failures saturate instead of monopolizing."""
from zeroproof_simulations.traces import behavior_state


def _row(tool, ok, ts):
    result = {"status": "ok"} if ok else {"status": "timeout"}
    return {"prompt": f"p{ts}", "ts": ts, "reward": 1 if ok else 0,
            "steps": [{"tool": tool, "arguments": {}, "result": result}],
            "final_text": "done"}


def test_lifecycle_classification():
    rows = []
    rows += [_row("refund", False, t) for t in (1, 2)]      # old fail
    rows += [_row("refund", False, t) for t in (11, 12)]    # recent fail -> persistent
    rows += [_row("lookup", False, 3)] + [_row("lookup", True, 13)]   # improving
    rows += [_row("cancel", False, 4)]                      # old fail, no recent -> uncertain
    rows += [_row("book", False, 14)]                       # new
    rows += [_row("status", True, 5), _row("status", True, 15)]  # passing
    state = behavior_state(rows, recent_fraction=0.5)
    by = {r["region"]: r["status"] for r in state["regions"]}
    assert by["refund"] == "persistent"
    assert by["lookup"] == "improving"
    assert by["cancel"] == "uncertain"
    assert by["book"] == "new"
    assert by["status"] == "passing"
    shares = {r["region"]: r["budget_share"]
              for r in state["regions"]}
    assert shares["book"] > shares["cancel"] > shares["lookup"]
    assert shares["refund"] > shares["improving"] if "improving" in shares \
        else shares["refund"] > shares["lookup"]


def test_exploration_always_reserved_and_shares_sum():
    rows = [_row("a", False, t) for t in range(10)]
    state = behavior_state(rows)
    assert state["exploration_share"] >= 0.2
    total = sum(r["budget_share"] for r in state["regions"])
    assert abs(total + state["exploration_share"] - 1.0) < 0.01


def test_repetition_saturates():
    loud = [_row("loud", False, t) for t in range(1, 40)]
    quiet = [_row("quiet", False, 39)]
    state = behavior_state(loud + quiet)
    shares = {r["region"]: r["budget_share"]
              for r in state["regions"]}
    assert shares["loud"] < 3 * shares["quiet"]


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
    regions = {r["region"]: r for r in state["regions"]}
    assert "lookup_order" in regions and "create_refund" in regions
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
    realized = [t for t in data.trajectories
                if t["scenario_dimensions"].get("origin") == "realized"]
    for t in realized:
        assert "tool" in t["scenario_dimensions"]
        assert "tool_condition" in t["scenario_dimensions"]


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
