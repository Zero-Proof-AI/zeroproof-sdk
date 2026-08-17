"""Adapter detection, inspect, and Claude stream normalisation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import zeroproof_simulations as zps
from zeroproof_simulations.adapters import detect, inspect, parse_claude_stream


TOOLS = [
    {"type": "function", "function": {"name": "lookup_item", "parameters": {
        "type": "object", "properties": {"item_id": {"type": "string"}},
        "required": ["item_id"]}}},
    {"type": "function", "function": {"name": "create_order", "parameters": {
        "type": "object", "properties": {"item_id": {"type": "string"}},
        "required": ["item_id"]}}},
]


def test_detect_names_supported_transports():
    assert detect(lambda m: {"steps": [], "final_text": m}) == "callable"
    assert detect("https://example.com/v1") == "http"
    assert detect("vllm:llama@https://example.com/v1") == "backend_spec"
    assert detect(["python", "agent.py"]) == "subprocess"
    with pytest.raises(ValueError) as exc:
        detect(12345)
    assert "callable" in str(exc.value)
    assert str(exc.value).count(".") <= 1


def test_inspect_merges_user_extras_onto_the_agent():
    class Holder:
        instructions = "Look up an item first."
        tools = TOOLS

    extra = [{"type": "function", "function": {
        "name": "cancel_order",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"}}}}}]
    profile = inspect(Holder(), tools=extra, policy="Never invent ids.")
    names = {t["function"]["name"] for t in profile.tools}
    assert {"lookup_item", "create_order", "cancel_order"} <= names
    assert "Look up an item first." in profile.policy
    assert "Never invent ids." in profile.policy


def test_user_situations_and_spec_feed_simulate():
    from tests.helpers import TOOLS as REFUND_TOOLS, POLICY, scripted_agent
    data = zps.simulate(
        scripted_agent, tools=REFUND_TOOLS, policy=POLICY, budget=6, seed=0,
        grade=False, concurrency=6, simulator=False,
        spec={"instructions": "Do not refund twice."},
        extra_situations=["pls refund ORD-9 i already paid twice??"],
        advanced={"per_round": 4, "mutate_failures": False})
    assert any("ORD-9" in t["prompt"] for t in data.trajectories)
    assert "Do not refund twice." in data.profile.policy


def test_grades_after_rollout_by_default():
    from tests.helpers import TOOLS as REFUND_TOOLS, POLICY, scripted_agent
    data = zps.simulate(
        scripted_agent, tools=REFUND_TOOLS, policy=POLICY, budget=8, seed=0,
        concurrency=8, simulator=False,
        advanced={"per_round": 8, "mutate_failures": False})
    assert all(t["reward"] is not None for t in data.trajectories)
    assert all(t["grader_reason"] for t in data.trajectories)


def test_inspect_reads_tools_policy_capabilities():
    class Fake:
        name = "shopper"
        instructions = "Look up an item before ordering it."
        tools = TOOLS

    profile = inspect(Fake())
    assert profile.policy.startswith("Look up")
    assert {t["function"]["name"] for t in profile.tools} == {
        "lookup_item", "create_order"}
    assert "read" in profile.capabilities["lookup_item"]
    assert "mutate" in profile.capabilities["create_order"]
    assert "item_id" in profile.constraints["required_user_fields"]
    connected = zps.connect(lambda m: {"steps": [], "final_text": "ok"},
                            tools=TOOLS, policy="be careful")
    assert connected.transport == "callable"
    assert connected.profile.policy == "be careful"


def test_inspect_fills_simulate_when_tools_omitted():
    class Holder:
        instructions = "Be honest."
        tools = TOOLS

        def __call__(self, message):
            return {"steps": [{"tool": "lookup_item", "arguments": {"item_id": "A"},
                               "result": {"status": "ok"}}],
                    "final_text": "found it"}

    data = zps.simulate(Holder(), budget=8, seed=0, grade=False,
                        simulator=False, advanced={"per_round": 4})
    assert data.profile.tools
    assert data.stages[0] == "agent ingestion"


def test_parse_claude_stream_pairs_any_tool_names():
    events = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "a", "name": "mcp__jira__create_ticket",
             "input": {"title": "bug"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "TICKET-9"}]}},
        {"type": "result", "result": "done, ticket filed"},
    ]
    traj = parse_claude_stream("\n".join(json.dumps(e) for e in events))
    assert [s["tool"] for s in traj["steps"]] == ["mcp__jira__create_ticket"]
    assert traj["final_text"] == "done, ticket filed"


def test_parse_text_tool_calls():
    from zeroproof_simulations.agents import parse_text_tool_calls
    raw = (
        '<tool_call>\n{"name": "list_events", "arguments": {"date": "2023-10-05"}}\n'
        '</tool_call>\n<tool_call>\n{"name": "create_event", "arguments": '
        '{"title": "Team Meeting", "start": "2023-10-05T09:00:00"}}\n</tool_call>'
    )
    calls = parse_text_tool_calls(raw)
    assert [c["function"]["name"] for c in calls] == ["list_events", "create_event"]


def test_subprocess_adapter(tmp_path):
    script = tmp_path / "agent.py"
    script.write_text(
        "import sys, json\n"
        "print(json.dumps({'steps': [{'tool': 'lookup_item', 'arguments': "
        "{'item_id': '1'}, 'result': {'status': 'ok'}}], 'final_text': 'ok'}))\n")
    connected = zps.connect([sys.executable, str(script)], tools=TOOLS,
                            policy="ok")
    assert connected.transport == "subprocess"
    out = connected("hello")
    assert out["steps"][0]["tool"] == "lookup_item"


def test_hash_embedder_does_not_claim_semantic_diversity():
    data = zps.simulate(
        lambda m: {"steps": [], "final_text": "ok"},
        tools=TOOLS, policy="ok", budget=6, seed=0, grade=False,
        embedder="hash", simulator=False, advanced={"per_round": 3})
    assert data.semantic is False
    assert "semantic_embedding_unavailable" in data.degraded or not data.semantic
    assert all(t["semantic_cluster"] is None for t in data.trajectories)
    assert "semantic embedding produced" not in data.stages


def test_refuses_to_mix_lexical_and_semantic_vectors():
    from zeroproof_simulations.embeddings import EmbeddingArchive, select_execution_batch

    class Sem:
        name = "stub-semantic"
        semantic = True

        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    archive = EmbeddingArchive("hash-word-bigram", False)
    archive.add([[0.0, 1.0]])
    _, info = select_execution_batch(
        ["hello there friend"], embedder=Sem(), archive=archive, batch_size=1, seed=0)
    assert info["mixed_spaces_refused"] is True
    assert "embedding_space_mismatch" in info["degraded"]


def test_model_prompt_asks_for_gaps():
    from zeroproof_simulations.generator import ModelSimulator
    sim = ModelSimulator("vllm:fake@http://example", tools=TOOLS,
                         policy="Look up an item first. Do not invent ids.",
                         timeout=1)
    sim.set_search_context(
        avoid=["basic refund please"],
        underexplored=["tool timeout on a half-finished order"],
        behavior_gaps=["same lookup then refund every time"])
    prompt = sim._prompt(1, sim.regions[:2] if sim.regions else [])
    assert "UNDEREXPLORED" in prompt
    assert "AVOID" in prompt
    assert "basic refund please" in prompt
    assert "tool timeout" in prompt
    assert "you are the user" in prompt.lower()
    assert "you are talking to" in prompt.lower()
    assert "Look up an item first" not in prompt
    assert "lookup_order" not in prompt


def test_coverage_axes_come_from_this_agent():
    policy = ("Look up an item before ordering it.\n"
              "2. Never invent an item_id.")
    dims = zps.build_dimensions(TOOLS, policy)
    assert "lookup_item" in dims["tool"]
    assert "create_order" in dims["tool"]
    assert "unrelated" in dims["tool"]
    assert {"ordinary", "ambiguous", "boundary", "adversarial"} <= set(dims["stance"])
    assert len(dims["stance"]) > 4
    assert "length" not in dims
    assert "vagueness" not in dims
    assert any("Look up an item" in clause for clause in dims["rule"])
    assert any("invent" in clause.lower() for clause in dims["rule"])
    regions = zps.scenario_regions(TOOLS, policy)
    tools_hit = {r["assignment"]["tool"] for r in regions}
    stances_hit = {r["assignment"]["stance"] for r in regions}
    assert {"lookup_item", "create_order"} <= tools_hit
    assert {"ordinary", "adversarial", "boundary", "ambiguous"} <= stances_hit


def test_selection_keeps_typical_and_fills_gaps():
    from zeroproof_simulations.embeddings import EmbeddingArchive, select_execution_batch

    class Stub:
        name = "stub-semantic"
        semantic = True

        def embed(self, texts):
            vectors = []
            for text in texts:
                if "common" in text:
                    vectors.append([1.0, 0.0, 0.0])
                elif "other" in text:
                    vectors.append([0.0, 1.0, 0.0])
                else:
                    vectors.append([0.0, 0.0, 1.0])
            return vectors

    texts = [f"common request {i}" for i in range(8)] + [
        "rare outlier alpha", "other side beta"]
    archive = EmbeddingArchive("stub-semantic", True)
    batch, info = select_execution_batch(
        texts, embedder=Stub(), archive=archive, batch_size=6, seed=0)
    reasons = [row["reason"] for row in batch]
    assert any(r.startswith("typical") for r in reasons)
    assert any(r.startswith("fill") for r in reasons)
    assert info["batch_design"] == "typical_kcenter_fill_plus_annealing_explore"
    assert info["concentrated"]
    assert info["sparse"]
