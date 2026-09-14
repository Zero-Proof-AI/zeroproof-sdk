"""The DPO example, offline: the pair builder (rows in, TRL rows out), the
Modal script's imports and flags against its README."""

from __future__ import annotations

import inspect
import json

from example_helpers import (
    EXAMPLES,
    assert_readme_matches_entrypoints,
    load_modal_script,
    load_script,
)

import zeroproof.simulations as zps

GRPO = EXAMPLES / "grpo"
DPO = EXAMPLES / "dpo"
README = DPO / "README.md"


def _modules():
    # pairs.py imports ``reward`` by its bare name, as it does on Modal.
    reward = load_script("reward", GRPO / "reward.py")
    pairs = load_script("dpo_example_pairs", DPO / "pairs.py")
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
    # The README's ``pair_report``: how many prompts had contrast, and how
    # often the chosen side is the longer one (the length exploit check).
    assert report["prompts_seen"] == 2 and report["prompts_with_contrast"] == 1
    assert 0.0 <= report["length"]["chosen_longer_frac"] <= 1.0


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


def test_invented_call_never_names_an_id_from_the_prompt():
    _, p = _modules()
    for prompt in ("refund ORD-1234 now", "where is ord 5555", "no id here"):
        call = json.loads(p.invented_call(prompt).split("\n")[1])
        assert call["name"] == "lookup_order"
        assert call["arguments"]["order_id"].lower() not in prompt.lower()


# --------------------------------------------------------- the Modal script


def test_train_modal_imports_and_its_flags_match_the_readme():
    mod = load_modal_script("dpo_train_modal", DPO / "train_modal.py")
    assert_readme_matches_entrypoints(README, {"examples/dpo/train_modal.py": mod.main})
    # The remote function takes what the entrypoint forwards.
    remote = inspect.signature(mod.train).parameters
    for name in ("loss_type", "beta", "pair_samples", "from_run", "constructed_negatives", "gpu"):
        assert name in remote, name
    assert remote["loss_type"].default == "sigmoid" and remote["beta"].default == 0.1
    assert mod.BASE_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"


def test_readme_has_no_placeholders_and_names_only_real_things():
    text = README.read_text(encoding="utf-8")
    assert "PLACEHOLDER" not in text
    for rel in ("pairs.py", "../grpo/reward.py"):
        assert rel in text and (DPO / rel).exists(), rel
    assert "examples/grpo/prompts.jsonl" in text and (GRPO / "prompts.jsonl").exists()
    _, p = _modules()
    assert "pairs.constructed_negatives" in text and hasattr(p, "constructed_negatives")
    for api in ("build_preference_pairs", "export_preference", "TrainerCallback"):
        assert api in text and hasattr(zps, api)
