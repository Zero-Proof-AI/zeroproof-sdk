"""The GRPO example's model-written prompt set: the pure parts, offline."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "grpo"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _modules():
    reward = _load("reward", EXAMPLE / "reward.py")
    prompts = _load("grpo_example_prompts", EXAMPLE / "prompts.py")
    return reward, prompts


def _seed(reward, text):
    return {"prompt": text, "case": reward.case_for(text), "scenario_id": "s-" + text[:8]}


def test_writer_messages_carry_the_category_rule():
    r, p = _modules()
    with_id = p.writer_messages(_seed(r, "please check ORD-4017 for me"), 0, 6)
    assert "ORD-4017" in with_id[1]["content"] and "exactly 6 strings" in with_id[1]["content"]
    no_id = p.writer_messages(_seed(r, "I need help with a refund"), 1, 4)
    assert (
        "Never include an id" in no_id[1]["content"]
        and "different kind of customer" in no_id[1]["content"]
    )
    off = p.writer_messages(_seed(r, "How tall is Kilimanjaro?"))
    assert "not about an order" in off[1]["content"]


def test_parse_messages_reads_an_array_or_quoted_lines():
    _, p = _modules()
    assert p.parse_messages('Sure:\n["a message", "another one"]\n') == ["a message", "another one"]
    assert p.parse_messages('1. "first line"\n- "second line",\nnot a message') == [
        "first line",
        "second line",
    ]
    assert p.parse_messages("") == []


def test_keep_enforces_category_and_drops_near_duplicates():
    r, p = _modules()
    s_id = _seed(r, "please check ORD-4017 for me")
    s_no = _seed(r, "I need help with a refund")
    s_off = _seed(r, "How tall is Kilimanjaro?")
    cands = [
        ("Hi, can you look into order ORD-4017? It arrived broken.", s_id),
        ("Hi can you look into order ORD-4017, it arrived broken", s_id),  # near duplicate
        ("Can you look into ORD-9999 for me?", s_id),  # wrong id
        ("Where is my refund? I ordered a jacket last week and nothing came.", s_no),
        ("Refund please, order ORD-1234", s_no),  # an id where none is allowed
        ("Do you sell gift cards?", s_off),
        ("Is my refund on order coming?", s_off),  # in domain where off topic is required
        ("short", s_id),
    ]
    kept = p.keep(cands)
    assert [k["prompt"] for k in kept] == [
        "Hi, can you look into order ORD-4017? It arrived broken.",
        "Where is my refund? I ordered a jacket last week and nothing came.",
        "Do you sell gift cards?",
    ]
    assert (
        kept[0]["scenario_id"] == s_id["scenario_id"] and kept[0]["case"]["order_id"] == "ORD-4017"
    )
    assert p.summary(kept) == {
        "prompts": 3,
        "scenarios": 3,
        "with_id": 1,
        "no_id": 1,
        "off_topic": 1,
    }


def test_load_prompts_rebuilds_cases_and_split_by_scenario(tmp_path):
    r, p = _modules()
    path = tmp_path / "prompts.jsonl"
    rows = [
        {"prompt": "Can you check ORD-4017?", "scenario_id": "a"},
        {"prompt": "Order ORD-4017 never arrived, help", "scenario_id": "a"},
        {"prompt": "I want a refund but lost the number", "scenario_id": "b"},
    ]
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    items = p.load_prompts(str(path))
    assert [i["case"]["order_id"] for i in items] == ["ORD-4017", "ORD-4017", None]
    train, held = r.split_holdout(items, 0.5)
    sides = {}
    for it in train:
        sides.setdefault(it["scenario_id"], set()).add("train")
    for it in held:
        sides.setdefault(it["scenario_id"], set()).add("held")
    assert all(len(v) == 1 for v in sides.values()), "a scenario landed on both sides"


def test_checked_in_prompt_set_is_well_formed():
    _, p = _modules()
    path = EXAMPLE / "prompts.jsonl"
    assert path.exists(), "examples/grpo/prompts.jsonl is checked in; *.jsonl is ignored so it needs its own unignore line"
    items = p.load_prompts(str(path))
    assert len(items) >= 200
    assert len({i["prompt"] for i in items}) == len(items)
    s = p.summary(items)
    assert min(s["with_id"], s["no_id"], s["off_topic"]) >= 20
