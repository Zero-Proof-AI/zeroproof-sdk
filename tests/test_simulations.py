import json

from tests.helpers import TOOLS, POLICY, scripted_agent
import zeroproof_simulations as zps

_DROPPED = {
    "selection_reason", "parent_failure_id", "arm", "scenario_dimensions",
    "behavior_signature", "grader_reason", "rollout_index", "seed",
    "semantic_cluster", "semantic_novelty", "llm_reward", "llm_reason",
}
_PUBLIC = {"prompt", "messages", "scenario_id", "steps", "final_text",
           "world_state", "faults", "fault_detected", "reward", "reason",
           "tier", "ask_family", "intent_known", "tool_known",
           "stance", "tone", "length", "ask", "vagueness", "phrasing",
           "pressure", "user", "texture", "history"}


def test_conversation_drops_stale_final_text():
    clarify = "Could you please provide the repository name?"
    row = {
        "prompt": "why is this branch still open",
        "steps": [
            {"text": clarify},
            {"user": "auth-service pr 421"},
            {"tool": "get_pr", "arguments": {"repo": "auth-service", "number": 421},
             "result": {"status": "ok"}},
        ],
        "final_text": clarify,
    }
    msgs = zps.conversation(row)
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "user", "assistant", "tool"]
    spoken = [m["content"] for m in msgs if m["role"] == "assistant"]
    assert spoken.count(clarify) == 1


def test_finish_on_agent_drops_pre_tool_clarify():
    from zeroproof_simulations.agents import _finish_on_agent
    clarify = "Which repo and PR number?"
    steps = [
        {"text": clarify},
        {"user": "acme/app 42"},
        {"tool": "get_pr", "arguments": {"number": 42}, "result": {"status": "ok"}},
    ]
    out = _finish_on_agent(steps, clarify)
    assert out["final_text"] != clarify
    assert out["final_text"] == ""


def test_finish_on_agent_keeps_post_tool_speech():
    from zeroproof_simulations.agents import _finish_on_agent
    steps = [
        {"text": "Which repo?"},
        {"user": "acme/app 42"},
        {"tool": "get_pr", "arguments": {}, "result": {"status": "ok"}},
        {"text": "PR 42 is open."},
    ]
    out = _finish_on_agent(steps, "Which repo?")
    assert out["final_text"] == "PR 42 is open."


def test_conversation_is_user_agent_turns():
    row = {
        "prompt": "where's order ORD-1",
        "steps": [
            {"tool": "lookup_order", "arguments": {"order_id": "ORD-1"},
             "result": {"status": "ok"}, "text": "let me look"},
            {"user": "and the refund?"},
            {"text": "still pending"},
        ],
        "final_text": "still pending",
    }
    msgs = zps.conversation(row)
    assert [m["role"] for m in msgs] == [
        "user", "assistant", "tool", "user", "assistant"]
    assert msgs[0]["content"] == "where's order ORD-1"
    assert msgs[1]["tool_calls"][0]["name"] == "lookup_order"
    assert msgs[3]["content"] == "and the refund?"
    exported = zps._export_row(row)
    assert exported["messages"] == msgs


def test_simulate_offline_end_to_end(tmp_path):
    data = zps.simulate(scripted_agent, tools=TOOLS, policy=POLICY, budget=80, seed=0,
                        grade=True, simulator=False)
    assert len(data.trajectories) == 80
    assert {"structured", "open_ended"} <= set(data.arm_yield)
    assert any(t["reward"] < 1.0 for t in data.trajectories), "planted bug found"
    assert any(t["faults"] for t in data.trajectories), "fault worlds instantiated"
    assert all(t.get("fault_detected") for t in data.trajectories if t.get("faults"))
    sft = data.sft_rows()
    assert sft and all(r["chosen_response"] is None for r in sft)
    path = data.save(str(tmp_path / "r.jsonl"))
    row = json.loads(open(path).readline())
    assert {"prompt", "messages", "steps", "final_text", "reward", "reason"} <= set(row)
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][0]["content"] == row["prompt"]
    assert any(m.get("role") == "assistant" for m in row["messages"])
    assert set(row) <= _PUBLIC
    assert not (_DROPPED & set(row))
    assert row.get("tier") in {"ordinary", "ambiguous", "boundary", "adversarial"}
    assert "selection_reason" not in row
    assert "arm" not in row


def test_save_omits_grade_when_ungraded(tmp_path):
    data = zps.simulate(scripted_agent, tools=TOOLS, policy=POLICY, budget=8, seed=0,
                        grade=False, concurrency=4, simulator=False,
                        advanced={"per_round": 6, "mutate_failures": False})
    row = json.loads(open(data.save(str(tmp_path / "u.jsonl"))).readline())
    assert {"prompt", "steps", "final_text"} <= set(row)
    assert "reward" not in row and "reason" not in row
    assert "fault_detected" not in row
    assert "llm_reward" not in row
    assert not (_DROPPED & set(row))
    exported = data.rows()[0]
    assert "selection_reason" not in exported
    assert "reward" not in exported


def test_custom_grader_and_dimensions_knobs():
    dims = zps.build_dimensions(TOOLS, POLICY)
    dims["stance"] = ["adversarial"]
    data = zps.simulate(scripted_agent, tools=TOOLS, policy=POLICY, budget=30,
                        seed=1, dimensions=dims, grader=lambda t: 0.5,
                        simulator=False)
    assert all(t["reward"] == 0.5 for t in data.trajectories)


def test_generate_then_grade_separately():
    data = zps.simulate(scripted_agent, tools=TOOLS, policy=POLICY, budget=60,
                        seed=0, grade=False, simulator=False)
    assert all(t["reward"] is None for t in data.trajectories)
    assert all(t["steps"] is not None for t in data.trajectories)
    data.grade()
    assert any(t["reward"] == 0.0 for t in data.trajectories)
    data.grade(grader=lambda t: 0.5)
    assert all(t["reward"] == 0.5 for t in data.trajectories)


def test_policy_optional_and_arms_stay_alive():
    data = zps.simulate(scripted_agent, tools=TOOLS, budget=20, seed=0,
                        grade=False, concurrency=8, simulator=False,
                        repeats=1,
                        advanced={"per_round": 16, "mutate_failures": False})
    assert len(data.trajectories) == 20
    assert len({t["prompt"] for t in data.trajectories}) == 20
    assert data.arm_weights
    assert data.arm_weights["structured"] >= 0.15 - 1e-9
    assert data.arm_weights["llm_guided"] >= 0.15 - 1e-9
    assert data.arm_weights["open_ended"] <= 0.10 + 1e-9
    assert data.arm_weights["failure_mutation"] <= 0.08 + 1e-9
    assert data.arm_weights["behavior_targeted"] <= 0.08 + 1e-9


_TINY_DIMS = {
    "tool": ["lookup_order"], "rule": ["unspecified"], "stance": ["ordinary"],
    "world_state": ["unspecified"], "tool_condition": ["success"],
    "history": ["fresh"],
}


def test_tiny_grid_compute_does_not_stop_on_saturation():
    data = zps.simulate(
        lambda m: {"steps": [], "final_text": "ok"},
        tools=TOOLS, policy=POLICY, budget=80, seed=0, grade=False,
        concurrency=8, until="compute", dimensions=_TINY_DIMS,
        simulator=False, time_budget=None,
        advanced={"per_round": 4, "mutate_failures": False})
    assert data.stopped_because == "budget"
    assert data.coverage.get("saturation") is False
    assert len(data.trajectories) == 80
    assert data.search.get("cell_counts")


def test_short_run_does_not_saturate_on_signature_blip():
    data = zps.simulate(
        lambda m: {"steps": [], "final_text": "ok"},
        tools=TOOLS, policy=POLICY, budget=80, seed=0, grade=False,
        concurrency=8, until="compute", simulator=False,
        advanced={"per_round": 4, "mutate_failures": False})
    assert data.stopped_because == "budget"
    assert data.coverage.get("saturation") is False
    assert len(data.trajectories) == 80
