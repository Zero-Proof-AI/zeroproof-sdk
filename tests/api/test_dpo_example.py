"""The DPO example's pair builder, offline: rows in, TRL rows out."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import zeroproof.simulations as zps

REPO_ROOT = Path(__file__).resolve().parents[2]
GRPO = REPO_ROOT / "examples" / "grpo"
DPO = REPO_ROOT / "examples" / "dpo"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _modules():
    # pairs.py imports ``reward`` by its bare name, as it does on Modal.
    reward = _load("reward", GRPO / "reward.py")
    pairs = _load("dpo_example_pairs", DPO / "pairs.py")
    return reward, pairs


CALL = '<tool_call>\n{"name": "lookup_order", "arguments": {"order_id": "ORD-4017"}}\n</tool_call>'
REFUND = '<tool_call>\n{"name": "create_refund", "arguments": {"order_id": "ORD-4017", "amount": 20}}\n</tool_call>'


def test_first_turn_prefers_the_tool_step_then_the_assistant_message():
    _, p = _modules()
    row = {
        "steps": [{"tool": "lookup_order", "arguments": {"order_id": "ORD-1"}}],
        "final_text": "done",
    }
    assert json.loads(p.first_turn(row).split("\n")[1]) == {
        "name": "lookup_order",
        "arguments": {"order_id": "ORD-1"},
    }
    row = {
        "steps": [],
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Which order?"},
        ],
    }
    assert p.first_turn(row) == "Which order?"
    wire = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "lookup_order", "arguments": '{"order_id": "ORD-2"}'}}
        ],
    }
    assert '"order_id": "ORD-2"' in p.first_turn({"messages": [wire]})
    assert p.first_turn({"final_text": " plain "}) == "plain"


def test_dpo_rows_keep_contrast_and_drop_identical_first_turns():
    _, p = _modules()
    pairs = [
        {
            "prompt": "check ORD-4017",
            "chosen": {"final_text": CALL},
            "rejected": {"final_text": REFUND},
            "margin": 1.0,
        },
        {
            "prompt": "check ORD-4017",
            "chosen": {"final_text": CALL},
            "rejected": {"final_text": CALL},
            "margin": 1.0,
        },
        {"prompt": "", "chosen": {"final_text": CALL}, "rejected": {"final_text": REFUND}},
    ]
    rows = p.dpo_rows(pairs, "SYS")
    assert len(rows) == 1
    row = rows[0]
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]
    assert row["prompt"][0]["content"] == "SYS"
    assert row["chosen"] == [{"role": "assistant", "content": CALL}]
    assert row["rejected"][0]["content"] == REFUND
    assert row["margin"] == 1.0


def test_sampled_pairs_pair_a_pass_with_a_fail_per_prompt():
    r, p = _modules()
    prompts = [
        {
            "prompt": "please check ORD-4017 for me",
            "case": r.case_for("please check ORD-4017 for me"),
            "scenario_id": "a",
        },
        {
            "prompt": "How tall is Kilimanjaro?",
            "case": r.case_for("How tall is Kilimanjaro?"),
            "scenario_id": "b",
        },
    ]
    replies = [
        [CALL, REFUND, CALL, "Sure, refunding now."],
        ["About 5,895 m.", "About 5,895 m.", "5,895 metres.", "Nearly six kilometres."],
    ]
    rows, report = p.sampled_pairs(prompts, replies, system="SYS")
    # Prompt a has passes and fails; prompt b is all passes, so no contrast.
    assert report["trl_rows"] == len(rows) >= 1
    assert all(row["prompt"][1]["content"] == prompts[0]["prompt"] for row in rows)
    assert all(row["chosen"][0]["content"] == CALL for row in rows)
    assert all(row["chosen"][0]["content"] != row["rejected"][0]["content"] for row in rows)


def test_load_export_reads_export_preference_output(tmp_path):
    r, p = _modules()
    rows = r.reward_rows(
        [
            {
                "prompt": "please check ORD-4017 for me",
                "case": r.case_for("please check ORD-4017 for me"),
                "scenario_id": "a",
            }
        ],
        [[CALL, REFUND]],
    )
    pairs, _ = zps.build_preference_pairs(rows)
    assert len(pairs) == 1
    out = tmp_path / "pairs.jsonl"
    zps.export_preference(pairs, str(out), validate=False)
    loaded = p.load_export(str(out), system="SYS")
    assert len(loaded) == 1
    assert loaded[0]["prompt"][0] == {"role": "system", "content": "SYS"}
    assert loaded[0]["prompt"][-1]["role"] == "user"
    assert loaded[0]["chosen"][0]["content"] == CALL
    assert loaded[0]["rejected"][0]["content"] == REFUND


def test_constructed_negatives_pair_the_ask_against_an_invented_call():
    r, p = _modules()
    prompts = [
        {
            "prompt": "I want a refund but lost the order number",
            "case": r.case_for("I want a refund but lost the order number"),
            "scenario_id": "a",
        },
        {
            "prompt": "Do you sell gift cards?",
            "case": r.case_for("Do you sell gift cards?"),
            "scenario_id": "b",
        },
        {
            "prompt": "please check ORD-4017 for me",
            "case": r.case_for("please check ORD-4017 for me"),
            "scenario_id": "c",
        },
        {
            "prompt": "My order never came, no idea of the id",
            "case": r.case_for("My order never came, no idea of the id"),
            "scenario_id": "d",
        },
    ]
    replies = [
        [
            "Sure, what is the order number?",
            CALL,
            "Could you share the order id so I can look it up?",
        ],
        ["No, we do not sell gift cards.", "Not at the moment."],
        [CALL, REFUND],
        [CALL, CALL],  # always invents: no chosen side to build from
    ]
    out = p.constructed_negatives(prompts, replies, system="SYS")
    assert [row["prompt"][1]["content"] for row in out] == [
        prompts[0]["prompt"],
        prompts[1]["prompt"],
    ]
    assert (
        out[0]["chosen"][0]["content"] == "Sure, what is the order number?"
    )  # the shortest passing reply
    rejected = out[0]["rejected"][0]["content"]
    assert "<tool_call>" in rejected and "lookup_order" in rejected
    fake = json.loads(rejected.split("<tool_call>")[1].split("</tool_call>")[0])["arguments"][
        "order_id"
    ]
    assert fake.startswith("ORD-") and fake.lower() not in prompts[0]["prompt"].lower()
    assert p.invented_call(prompts[0]["prompt"]) == p.invented_call(prompts[0]["prompt"])
    assert all(row["constructed"] for row in out)
    rows, report = p.sampled_pairs(
        prompts, replies, system="SYS", constructed=True, constructed_share=1.0
    )
    assert report["constructed_pairs"] == 2 and report["trl_rows"] == len(rows)
    assert sum(1 for row in rows if row.get("constructed")) == 2
    # Repeated prompts (--balance) give one constructed pair, not one per copy.
    doubled = p.constructed_negatives(prompts + prompts, replies + replies, system="SYS")
    assert len(doubled) == 2
    assert len(p.constructed_negatives(prompts, replies, system="SYS", max_pairs=1)) == 1
    _, rep = p.sampled_pairs(
        prompts, replies, system="SYS", constructed=True, constructed_share=0.3
    )
    on_policy = rep["trl_rows"] - rep["constructed_pairs"]
    assert rep["constructed_pairs"] <= max(1, int(0.3 * on_policy))
