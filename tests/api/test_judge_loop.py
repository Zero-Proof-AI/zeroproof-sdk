"""The judge contract and the full loop it closes.

define fake agent -> trajectories -> custom judge -> scored -> select
passes -> export with roundtrip validation -> mock held-out eval ->
failed traces -> aim the next simulation from them.
"""
from __future__ import annotations

import json

from zeroproof_simulations.export import export_training
from zeroproof_simulations.judging import (ScoredData, evaluate,
                                           normalize_judge_result, run_judge)
from zeroproof_simulations.traces import dimensions_from_traces, mine_traces

TOOLS = [{"name": "get_order", "description": "Look up an order",
          "parameters": {"type": "object",
                         "properties": {"order_id": {"type": "string"}},
                         "required": ["order_id"]},
          "returns": {"order_id": "A1", "status": "shipped"}},
         {"name": "refund_order", "description": "Refund an order",
          "parameters": {"type": "object",
                         "properties": {"order_id": {"type": "string"}},
                         "required": ["order_id"]},
          "returns": {"ok": True}}]
POLICY = ("Look up an order with get_order before any refund_order call. "
          "Confirm with the user before refunding. Report tool results "
          "honestly and never invent order facts.")


def _traj(i, *, final, tool="get_order", ok=True):
    return {
        "prompt": f"where is order A{i}",
        "scenario_id": f"sc-{i:04d}",
        "steps": [{"tool": tool, "arguments": {"order_id": f"A{i}"},
                   "result": {"status": "ok" if ok else "not_found"}}],
        "final_text": final,
        "messages": [
            {"role": "user", "content": f"where is order A{i}"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": tool,
                             "arguments": {"order_id": f"A{i}"}}]},
            {"role": "tool", "name": tool, "content": '{"status": "ok"}'},
            {"role": "assistant", "content": final},
        ],
    }


def honest_judge(t):
    return {"reward": int("invented" not in t["final_text"]),
            "reason": "grounded" if "invented" not in t["final_text"]
            else "made up a fact"}


def test_full_loop_scored_to_export_to_eval_to_traces(tmp_path):
    trajectories = [_traj(i, final="Order shipped.") for i in range(4)]
    trajectories += [_traj(9, final="I invented a tracking number.")]

    scored = run_judge(trajectories, honest_judge, judge_name="honest")
    assert isinstance(scored, ScoredData)
    assert len(scored) == 5
    assert len(scored.passes()) == 4 and len(scored.failures()) == 1
    assert trajectories[0].get("reward") is None  # originals untouched
    assert all(r["judge_status"] == "ok" for r in scored)
    assert all(r["lineage"]["scoring_run_id"] == scored.run_id
               for r in scored)

    out = str(tmp_path / "train.jsonl")
    report = export_training(scored.passes(), out, system_prompt=POLICY,
                             tools=TOOLS)
    assert report["tool_call_roundtrip"]["invalid"] == 0
    exported = [json.loads(l) for l in open(out)]
    assert exported[0]["lineage"]["scoring_run_id"] == scored.run_id
    assert exported[0]["judge_name"] == "honest"

    rollouts = [_traj(i, final="Order shipped.") for i in (20, 21)]
    rollouts += [_traj(i, final="I invented a refund confirmation.")
                 for i in (22, 23)]
    evald = evaluate(rollouts, honest_judge, model="tuned-v1")
    assert evald.source == "eval"
    fails = evald.failed_traces()
    assert len(fails) == 2
    assert all(f["lineage"]["model"] == "tuned-v1" for f in fails)

    mined = mine_traces(fails)
    assert mined["flaw_rows"] == [0, 1]
    dims = dimensions_from_traces(evald, TOOLS, POLICY)  # container direct
    assert dims["tool"][0] in {"get_order", "refund_order"}



def test_judge_contract_edges():
    assert normalize_judge_result(1)["reward"] == 1
    assert normalize_judge_result(0.7)["reward"] == 0.7
    assert normalize_judge_result(True)["reward"] == 1
    missing = normalize_judge_result({"reason": "forgot the reward"})
    assert missing["reward"] is None
    assert missing["judge_status"] == "missing_reward"
    bad_type = normalize_judge_result({"reward": "great"})
    assert bad_type["reward"] is None
    assert bad_type["judge_status"] == "invalid_result"
    garbage = normalize_judge_result("looks good to me")
    assert garbage["judge_status"] == "invalid_result"
    extras = normalize_judge_result({"reward": 0, "failure_class":
                                     "fabrication", "confidence": 0.9})
    assert extras["judge_meta"] == {"failure_class": "fabrication",
                                    "confidence": 0.9}


def test_judge_exception_marks_rows_not_zero():
    rows = [_traj(1, final="fine"), _traj(2, final="fine")]

    def crashing(t):
        raise RuntimeError("judge bug")

    scored = run_judge(rows, crashing, concurrency=1)
    assert [r["reward"] for r in scored] == [None, None]
    assert all(r["judge_status"] == "error" for r in scored)
    assert "judge bug" in scored[0]["judge_meta"]["error"]
    assert scored.failures() == []          # not silently failures
    assert len(scored.unjudged()) == 2


def test_all_pass_all_fail_mixed_and_float_rewards():
    rows = [_traj(i, final="fine") for i in range(3)]
    all_pass = run_judge(rows, lambda t: 1)
    assert len(all_pass.passes()) == 3 and not all_pass.failures()
    all_fail = run_judge(rows, lambda t: {"reward": 0})
    assert len(all_fail.failures()) == 3
    mixed = run_judge(rows, lambda t: {"reward": 0.5, "reason": "partial"})
    assert mixed.select_by_reward(0.5) and not mixed.passes()
    dist = mixed.report()["reward_distribution"]
    assert dist == {"0.5": 3}


def test_failure_class_from_judge_and_lineage_chain():
    rows = [_traj(1, final="I invented it.")]
    judge = lambda t: {"reward": 0, "failure_class": "fabrication"}
    scored = run_judge(rows, judge)
    assert scored[0]["failure_class"] == "fabrication"
    rescored = run_judge(scored, lambda t: {"reward": 0}, source="eval")
    assert rescored[0]["lineage"]["prior_scoring_run_id"] == scored.run_id


def test_duplicates_and_missing_metadata_flow_through():
    dup = _traj(5, final="fine")
    bare = {"final_text": "no prompt or steps here"}
    scored = run_judge([dup, dict(dup), bare], lambda t: 1)
    assert len(scored) == 3
    assert scored[2]["lineage"]["parent"] == "row_2"
    mined = mine_traces(scored.rows)
    assert mined["n"] == 3


def test_preference_pairs_and_export(tmp_path):
    rows = [_traj(1, final="Order shipped."), _traj(1, final="I invented it."),
            _traj(2, final="Order shipped.")]
    scored = run_judge(rows, honest_judge)
    pairs, rep = scored.select_for_preference()
    assert rep["pairs"] == 1 and rep["prompts_with_contrast"] == 1
    pair = pairs[0]
    assert pair["chosen"]["reward"] == 1 and pair["rejected"]["reward"] == 0
    assert pair["lineage"]["chosen"]["scoring_run_id"] == scored.run_id

    from zeroproof_simulations.export import export_preference
    out = str(tmp_path / "prefs.jsonl")
    report = export_preference(pairs, out, system_prompt=POLICY, tools=TOOLS)
    assert report["pairs"] == 1
    assert report["tool_call_roundtrip"]["invalid"] == 0
    line = json.loads(open(out).readline())
    assert line["chosen"][0]["role"] == "system"
    assert line["rejected"][-1]["content"] == "I invented it."

    no_contrast = run_judge([_traj(3, final="fine")], lambda t: 1)
    empty, rep2 = no_contrast.select_for_preference()
    assert empty == [] and "contrast" in rep2["note"]


def test_selection_lanes_and_dataset_alias(tmp_path):
    rows = [_traj(i, final="Order shipped.") for i in range(6)]
    rows += [_traj(i, final="I invented it.") for i in (6, 7)]
    scored = run_judge(rows, honest_judge)
    sft, rep = scored.select_for_sft(target=4)
    assert len(sft) <= 4
    assert all(r["reward"] == 1 for r in sft)
    from zeroproof_simulations.export import export_dataset, export_training
    assert export_dataset is export_training
    out = export_dataset(sft, str(tmp_path / "d.jsonl"),
                         system_prompt=POLICY, tools=TOOLS)
    assert out["tool_call_roundtrip"]["invalid"] == 0


def test_scaffold_is_generation_only():
    import zeroproof_simulations as zps
    from tests.helpers import TOOLS as HT, POLICY as HP, scripted_agent
    data = zps.simulate(scripted_agent, tools=HT, policy=HP, budget=4,
                        seed=0, simulator=False,
                        scaffold="Ground every claim in tool results.")
    assert data.scaffold_chars == len("Ground every claim in tool results.")
    assert "Ground every claim" not in str(data.profile.policy)
    exported = zps.training_rows(data.trajectories[:1], system_prompt=HP,
                                 tools=HT)
    assert "Ground every claim" not in exported[0]["messages"][0]["content"]


def test_bool_rewards_count_in_trace_mining():
    rows = [dict(_traj(1, final="fine"), reward=False),
            dict(_traj(2, final="fine"), reward=True)]
    mined = mine_traces(rows)
    assert mined["flaw_rows"] == [0]
