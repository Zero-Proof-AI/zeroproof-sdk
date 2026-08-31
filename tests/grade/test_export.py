"""Training export: system + tools attached, wire format, think stripped."""
from __future__ import annotations

import json

from zeroproof_simulations.export import export_training, training_rows
from zeroproof_simulations.optimize import recommend

TOOLS = [{"type": "function", "function": {
    "name": "get_issue",
    "parameters": {"type": "object",
                   "properties": {"number": {"type": "number"}},
                   "required": ["number"]}}}]
POLICY = "Look before you act. Report misses honestly."

ROW = {
    "prompt": "look up issue 4412",
    "reward": 1,
    "steps": [{"tool": "get_issue", "arguments": {"number": 4412},
               "result": {"status": "ok", "number": 4412}}],
    "final_text": "Issue 4412 is open.",
    "messages": [
        {"role": "user", "content": "look up issue 4412"},
        {"role": "assistant",
         "content": "<think>user wants the issue, call the tool</think>",
         "tool_calls": [{"name": "get_issue",
                         "arguments": {"number": 4412}}]},
        {"role": "tool", "name": "get_issue",
         "content": '{"status": "ok", "number": 4412}'},
        {"role": "assistant", "content": "Issue 4412 is open."},
    ],
}


def test_training_rows_attach_system_tools_and_wire_format():
    rows = training_rows([ROW], system_prompt=POLICY, tools=TOOLS)
    assert len(rows) == 1
    out = rows[0]
    msgs = out["messages"]
    assert msgs[0] == {"role": "system", "content": POLICY}
    assert out["tools"] == TOOLS
    assert out["reward"] == 1
    call_msg = msgs[2]
    call = call_msg["tool_calls"][0]
    assert call["type"] == "function"
    assert call["id"].startswith("call_")
    assert call["function"]["name"] == "get_issue"
    args = json.loads(call["function"]["arguments"])
    assert args == {"number": 4412}
    tool_msg = msgs[3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == call["id"]


def test_think_blocks_never_reach_a_student_model():
    rows = training_rows([ROW], system_prompt=POLICY, tools=TOOLS)
    blob = json.dumps(rows)
    assert "<think>" not in blob
    kept = training_rows([ROW], system_prompt=POLICY, tools=TOOLS,
                         strip_think=False)
    assert "<think>" in json.dumps(kept)


def test_export_training_writes_sibling_never_source(tmp_path):
    src = tmp_path / "run.jsonl"
    src.write_text(json.dumps(ROW) + "\n")
    report = export_training(str(src), system_prompt=POLICY, tools=TOOLS)
    assert report["n"] == 1
    assert report["with_system"] == 1
    assert report["with_tools"] == 1
    assert report["path"].endswith("run.train.jsonl")
    assert src.read_text().count("\n") == 1


def test_recommend_sizes_from_the_grid():
    sft = recommend(TOOLS, POLICY, mode="sft")
    assert sft["grid_cells"] > 0
    assert sft["budget"] >= 3 * sft["optimize_target"]
    assert sft["budget"] >= sft["grid_cells"] * 5
    assert sft["simulate_kwargs"]["mode"] == "sft"
    rl = recommend(TOOLS, POLICY, mode="rl", target=400)
    assert rl["rollouts_per_request"] == 8
    assert rl["situations"] * 8 == rl["budget"]
    assert rl["situations"] >= 2 * 400 // 8
    assert rl["simulate_kwargs"]["situations"] == rl["situations"]
    assert any("covering grid" in line for line in rl["reasoning"])


def _row_with_arguments(arguments):
    return {
        "prompt": "look up issue 4412",
        "reward": 1,
        "messages": [
            {"role": "user", "content": "look up issue 4412"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "get_issue", "arguments": arguments}]},
            {"role": "tool", "name": "get_issue", "content": '{"status": "ok"}'},
            {"role": "assistant", "content": "Issue 4412 is open."},
        ],
    }


def test_double_encoded_arguments_normalize_to_structured():
    double = json.dumps(json.dumps({"number": 4412}))
    rows = training_rows([_row_with_arguments(double)],
                         system_prompt=POLICY, tools=TOOLS)
    wire = rows[0]["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(wire) == {"number": 4412}


def test_export_refuses_unparseable_tool_arguments():
    import pytest
    from zeroproof_simulations.export import tool_call_roundtrip
    bad = _row_with_arguments("number equals 4412")
    with pytest.raises(ValueError, match="tool_call_roundtrip_invalid"):
        export_training([bad], system_prompt=POLICY, tools=TOOLS)
    report = export_training([bad], system_prompt=POLICY, tools=TOOLS,
                             validate=False)
    assert report["tool_call_roundtrip"]["invalid"] == 1
    clean = training_rows([ROW], system_prompt=POLICY, tools=TOOLS)
    assert tool_call_roundtrip(clean) == {"checked": 1, "invalid": 0,
                                          "rows": []}


def _graded(prompt: str, reward: int, tier: str = "ordinary") -> dict:
    return {
        "prompt": prompt, "reward": reward, "tier": tier, "stance": "curt",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "done"},
        ],
    }


def test_repeated_prompts_get_group_identity():
    rows = training_rows(
        [_graded("refund A", 1), _graded("refund A", 0),
         _graded("refund A", 0), _graded("lookup B", 1),
         _graded("lookup B", 1)],
        system_prompt=POLICY, tools=TOOLS)
    a = [r for r in rows if r["prompt"] == "refund A"]
    b = [r for r in rows if r["prompt"] == "lookup B"]
    assert {r["group_id"] for r in a} != {r["group_id"] for r in b}
    assert all(r["k"] == 3 and r["n0"] == 2 and r["n1"] == 1 for r in a)
    assert all(r["k"] == 2 and r["n0"] == 0 and r["n1"] == 2 for r in b)


def test_unique_prompts_get_no_group_fields():
    rows = training_rows([_graded("one", 1), _graded("two", 0)],
                         system_prompt=POLICY, tools=TOOLS)
    assert all("group_id" not in r and "k" not in r for r in rows)


def test_diversity_axes_survive_export():
    rows = training_rows([_graded("refund A", 1, tier="adversarial")],
                         system_prompt=POLICY, tools=TOOLS)
    assert rows[0]["tier"] == "adversarial"
    assert rows[0]["stance"] == "curt"


def test_pulled_tool_trace_rows_export_with_tool_calls():
    row = {
        "prompt": "fix the paging bug",
        "final_text": "Done, tests pass.",
        "reward": 1,
        "tool_trace": [
            {"tool": "read_file", "input": '{"path": "paging.py"}',
             "output": "def page_count(): ..."},
        ],
    }
    rows = training_rows([row], system_prompt=POLICY, tools=TOOLS)
    calls = [c for m in rows[0]["messages"] for c in (m.get("tool_calls") or [])]
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "paging.py"}
    from zeroproof_simulations.export import tool_call_roundtrip as rt
    assert rt(rows) == {"checked": 1, "invalid": 0, "rows": []}
