"""V1 product-path readiness battery (Agent 2 audit, 2026-08-26).

Every test here verifies a claim destined for SDK_READINESS.md. The path
under audit: agent/spec OR raw traces -> inspect -> simulate -> external
judge -> ScoredData -> selection -> safe export -> [external training] ->
evaluate -> failed traces back into simulate. All offline.
"""
from __future__ import annotations

import json

from tests.connect.test_otel import BATCH
from tests.helpers import POLICY, TOOLS, scripted_agent
from zeroproof_simulations import rows_from_otel
from zeroproof_simulations.export import (tool_call_roundtrip, training_rows)
from zeroproof_simulations.judging import (ScoredData, build_preference_pairs,
                                           evaluate, run_judge)
from zeroproof_simulations.optimize import select_for_sft
from zeroproof_simulations.traces import (dimensions_from_traces, load_traces,
                                          simulate_from_traces, trace_report)

RAW_UNGRADED = [
    {"prompt": "where is order 4412",
     "steps": [{"tool": "lookup_order", "arguments": {"order_id": "4412"},
                "result": {"status": "not_found"}}],
     "final_text": "I could not find that order."},
    {"prompt": "refund order 9911 now",
     "steps": [{"tool": "create_refund", "arguments": {"order_id": "9911"},
                "result": {"status": "timeout"}}],
     "final_text": "The refund request timed out."},
    {"prompt": "status of order 130",
     "steps": [{"tool": "lookup_order", "arguments": {"order_id": "130"},
                "result": {"status": "ok"}}],
     "final_text": "Order 130 is confirmed."},
]


def _sim(traces, **kw):
    return simulate_from_traces(
        traces, scripted_agent, tools=TOOLS, policy=POLICY,
        mode="explore", budget=6, seed=0, grade=False, concurrency=6,
        simulator=False, time_budget=20,
        advanced={"per_round": 4, "mutate_failures": False}, **kw)


# --- item 1/3: OTel-converted traces are first-class trace input ---------

def test_otel_rows_reach_simulation_without_grades():
    rows = rows_from_otel(BATCH)
    assert rows, "rows_from_otel produced nothing from the fixture batch"
    normalized = load_traces(rows)
    assert normalized
    for row in normalized:
        assert row["prompt"], "OTel rows must carry a first user ask"
        assert isinstance(row["steps"], list)
    report = trace_report(normalized, tools=TOOLS, policy=POLICY)
    assert report["traces"] == len(normalized)
    data = _sim(normalized)
    assert data.trajectories
    assert data.search["trace_mining"]["n_traces"] == len(normalized)


def test_otel_rows_do_not_pretend_to_carry_the_agent_schema():
    # OTel gives observed calls/results/messages, not the agent definition.
    rows = rows_from_otel(BATCH)
    for row in rows:
        assert "tools" not in row or not row.get("tools"), \
            "OTel conversion must not invent tool schemas"
        assert not row.get("policy"), \
            "OTel conversion must not invent a policy"


# --- item 2: declared agent stays authoritative; traces only aim ---------

def test_traces_aim_but_never_replace_the_declared_agent():
    alien = [{"prompt": "do the alien thing",
              "steps": [{"tool": "alien_tool", "arguments": {},
                         "result": {"status": "timeout"}}],
              "final_text": "alien failed"}]
    dims = dimensions_from_traces(alien + RAW_UNGRADED, TOOLS, POLICY)
    declared = {str((t.get("function") or t).get("name") or "")
                for t in TOOLS} | {"unrelated", "multi_tool"}
    assert set(dims["tool"]) <= declared, \
        "trace-observed alien tools must not enter the generation grid"
    report = trace_report(alien + RAW_UNGRADED, tools=TOOLS, policy=POLICY)
    assert "alien_tool" in report.get("foreign_tools", []), \
        "foreign tools must be disclosed, not silently dropped"


# --- item 5: judge contract, no silent zeros -----------------------------

def test_judge_exceptions_never_become_reward_zero():
    def broken(row):
        raise RuntimeError("judge crashed")
    scored = run_judge(RAW_UNGRADED, broken, concurrency=1)
    assert all(r["reward"] is None for r in scored)
    assert all(r["judge_status"] == "error" for r in scored)
    assert len(scored.unjudged()) == len(RAW_UNGRADED)
    assert scored.select_for_sft()[0] == [], \
        "unjudged rows must never be selected as passes"


def test_bad_judge_returns_are_marked_not_coerced():
    for raw in ("garbage", None, (), {"reward": "yes"}):
        scored = run_judge([RAW_UNGRADED[0]], lambda row, r=raw: r,
                           concurrency=1)
        assert scored[0]["reward"] is None, (raw, scored[0])
        assert scored[0]["judge_status"] == "invalid_result"
    ok = run_judge([RAW_UNGRADED[0]], lambda row: {"reward": 0,
                                                   "reason": "bad"},
                   concurrency=1)
    assert ok[0]["reward"] == 0 and ok[0]["judge_status"] == "ok"


def test_scalar_reward_lane_every_legal_reward_is_visible():
    # DECIDED 2026-08-26: 1 pass, 0 fail, (0,1) partial, None unjudged.
    scored = run_judge([RAW_UNGRADED[0]], lambda row: 0.7, concurrency=1)
    assert scored[0]["reward"] == 0.7
    assert scored[0]["judge_status"] == "ok"
    assert scored.passes() == [] and scored.failures() == []
    assert len(scored.partials()) == 1
    assert len(scored.select_by_reward_range(0.5, 0.9)) == 1
    assert scored.unjudged() == []


def test_out_of_range_rewards_are_contract_breaks():
    for raw in (5, -0.3, 1.5, {"reward": 2}):
        scored = run_judge([RAW_UNGRADED[0]], lambda row, r=raw: r,
                           concurrency=1)
        assert scored[0]["reward"] is None, raw
        assert scored[0]["judge_status"] == "invalid_result"
        assert len(scored.unjudged()) == 1


def test_otel_reward_attribute_is_preserved():
    from tests.connect.test_otel import _attr, _span
    batch = {"resourceSpans": [{"scopeSpans": [{"spans": [
        _span("agent airline", 100, [
            _attr("gen_ai.conversation.id", "c-graded"),
            _attr("gen_ai.input.messages",
                  json.dumps([{"role": "user",
                              "content": "cancel res 45678"}])),
            _attr("zeroproof.reward", 0.0)]),
        _span("agent airline", 200, [
            _attr("gen_ai.conversation.id", "c-partial"),
            _attr("gen_ai.input.messages",
                  json.dumps([{"role": "user",
                              "content": "check my bag count"}])),
            _attr("zeroproof.reward", 0.7)]),
        _span("agent airline", 300, [
            _attr("gen_ai.conversation.id", "c-bad"),
            _attr("gen_ai.input.messages",
                  json.dumps([{"role": "user",
                              "content": "book a flight"}])),
            _attr("zeroproof.reward", 7)]),
    ]}]}]}
    rows = {r["conversation_id"]: r for r in rows_from_otel(batch)}
    assert rows["c-graded"]["reward"] == 0
    assert rows["c-partial"]["reward"] == 0.7
    assert "reward" not in rows["c-bad"], "out-of-range attr must be ignored"
    ungraded = {r["conversation_id"]: r for r in rows_from_otel(BATCH)}
    assert all("reward" not in r for r in ungraded.values())


def test_grade_and_eval_share_one_contract():
    judge = lambda row: 1
    graded = run_judge(RAW_UNGRADED, judge, concurrency=1)
    evald = evaluate(RAW_UNGRADED, judge, model="agent-v2", concurrency=1)
    for g, e in zip(graded, evald):
        assert set(k for k in g if k.startswith("judge")) == \
            set(k for k in e if k.startswith("judge"))
    assert graded[0]["lineage"]["source"] == "grade"
    assert evald[0]["lineage"]["source"] == "eval"
    assert evald[0]["lineage"]["model"] == "agent-v2"


# --- item 6: selection inventory; failures preserved ---------------------

def test_selection_takes_passes_but_never_destroys_failures():
    rows = [dict(r, reward=i % 2, prompt=f"ask {i}", scenario_id=f"s{i}")
            for i, r in enumerate(RAW_UNGRADED * 4)]
    scored = ScoredData([dict(r, judge_status="ok") for r in rows],
                        run_id="t", source="grade", judge_name="j")
    selected, report = scored.select_for_sft()
    assert all(r["reward"] == 1 for r in selected)
    assert len(scored.failures()) == 6, \
        "selection must not remove failures from the scored set"
    assert len(scored.failed_traces()) == 6


def test_preference_pairs_require_same_prompt_pass_and_fail():
    rows = [
        {"prompt": "same ask", "reward": 1, "judge_status": "ok",
         "final_text": "good", "steps": []},
        {"prompt": "same ask", "reward": 0, "judge_status": "ok",
         "final_text": "bad", "steps": []},
        {"prompt": "lonely ask", "reward": 1, "judge_status": "ok",
         "final_text": "good", "steps": []},
    ]
    pairs, _rep = build_preference_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]["chosen"]["final_text"] == "good"
    assert pairs[0]["rejected"]["final_text"] == "bad"


# --- items 7/9: export safety and lineage provenance ---------------------

def test_lineage_survives_scoring_selection_and_export():
    sim_rows = [{"prompt": f"ask {i}", "scenario_id": f"sc-{i}",
                 "final_text": "Done.", "steps": [
                     {"user": f"ask {i}"},
                     {"tool": "lookup_order",
                      "arguments": {"order_id": str(i)},
                      "result": {"status": "ok", "data": {"id": i}}}],
                 "messages": None} for i in range(4)]
    for r in sim_rows:
        r.pop("messages")
    scored = run_judge(sim_rows, lambda row: 1, judge_name="my_judge",
                       concurrency=1)
    selected, _ = scored.select_for_sft()
    assert selected
    exported = training_rows(selected, system_prompt=POLICY, tools=TOOLS)
    gate = tool_call_roundtrip(exported)
    assert gate["invalid"] == 0
    for row in exported:
        lineage = row.get("lineage")
        assert lineage and lineage["judge"] == "my_judge"
        assert lineage["scoring_run_id"].startswith("score_")
        assert lineage["parent"].startswith("sc-")
        assert row.get("scenario_id", "").startswith("sc-")


# --- item 8: the closed loop, offline, with a mock training boundary -----

def test_full_feedback_loop_offline():
    # 1) raw ungraded traces -> simulate
    data = _sim(RAW_UNGRADED)
    assert data.trajectories
    # 2) external judge -> ScoredData
    judge = lambda row: {"reward": int("confirmed" in
                                       str(row.get("final_text", "")).lower()
                                       or "$" in
                                       str(row.get("final_text", ""))),
                         "reason": "deterministic"}
    scored = run_judge(data.trajectories, judge, concurrency=1)
    assert len(scored) == len(data.trajectories)
    # 3) select + safe export
    selected, _ = scored.select_for_sft()
    exported = training_rows(selected or list(scored)[:1],
                             system_prompt=POLICY, tools=TOOLS)
    assert tool_call_roundtrip(exported)["invalid"] == 0
    # 4) [external training happens elsewhere] -> evaluate new rollouts
    rollouts = [{"prompt": r["prompt"], "steps": r.get("steps") or [],
                 "final_text": "I could not find that order."}
                for r in list(scored)[:4]]
    evald = evaluate(rollouts, judge, model="tuned-v2", concurrency=1)
    fails = evald.failed_traces()
    assert fails, "the deterministic judge must fail these rollouts"
    assert all(f["lineage"]["source"] == "eval" for f in fails)
    # 5) failed eval traces -> simulate again; lineage rides along
    second = _sim(fails)
    assert second.trajectories
    assert second.search["trace_mining"]["n_traces"] == len(fails)


def test_label_free_path_stands_alone():
    assert all("reward" not in r for r in load_traces(RAW_UNGRADED))
    data = _sim(load_traces(RAW_UNGRADED))
    assert data.trajectories
