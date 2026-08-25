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
