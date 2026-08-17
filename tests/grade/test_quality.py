"""Deterministic quality ranker. No GPU."""
from __future__ import annotations

import json

from tests.helpers import REPO_ROOT, simulate_offline
import zeroproof_simulations as zps
from zeroproof_simulations.quality import DIMENSIONS, FAIL, score_row


def _good(**extra):
    final = "Refund of $40 is in. You should see it in 3 days."
    row = {
        "prompt": "can you look up order ORD-4412, the tracking looks stuck",
        "messages": [
            {"role": "user",
             "content": "can you look up order ORD-4412, the tracking looks stuck"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "lookup_order",
                             "arguments": {"order_id": "ORD-4412"}}]},
            {"role": "tool", "name": "lookup_order",
             "content": '{"status": "ok"}'},
            {"role": "assistant",
             "content": "ORD-4412 is packed and sitting in Chicago."},
            {"role": "user",
             "content": "ok and can you start the refund for the late delivery"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "create_refund",
                             "arguments": {"order_id": "ORD-4412",
                                           "amount": 40}}]},
            {"role": "tool", "name": "create_refund",
             "content": '{"status": "created"}'},
            {"role": "assistant", "content": final},
        ],
        "steps": [
            {"tool": "lookup_order", "arguments": {"order_id": "ORD-4412"},
             "result": {"status": "ok"}},
            {"text": "ORD-4412 is packed and sitting in Chicago."},
            {"user": "ok and can you start the refund for the late delivery"},
            {"tool": "create_refund",
             "arguments": {"order_id": "ORD-4412", "amount": 40},
             "result": {"status": "created"}},
            {"text": final},
        ],
        "final_text": final,
        "ask_family": "tool",
        "tool_known": True,
    }
    row.update(extra)
    return row


def test_known_good_row_passes():
    scored = score_row(_good())
    assert scored["quality"] >= 0.85
    assert scored["quality_reason"] == "conforms"
    assert set(scored["quality_scores"]) == set(DIMENSIONS)
    assert all(v >= FAIL for v in scored["quality_scores"].values())


def test_agent_voice_opener_fails():
    row = _good()
    row["prompt"] = "Could you please provide the order number so I can look it up?"
    row["messages"][0]["content"] = row["prompt"]
    scored = score_row(row)
    assert scored["quality_scores"]["opener"] < FAIL
    assert "agent-voice" in scored["quality_reason"]


def test_desk_voice_opener_fails():
    row = _good()
    row["prompt"] = "Thank you for contacting support. The tracking on ORD-4412 looks stuck."
    row["messages"][0]["content"] = row["prompt"]
    scored = score_row(row)
    assert scored["quality_scores"]["opener"] < FAIL
    assert "desk-voice" in scored["quality_reason"]


def test_stacked_assistant_variants_fail_ping_pong():
    row = {
        "prompt": "where's order ORD-1",
        "messages": [
            {"role": "user", "content": "where's order ORD-1"},
            {"role": "assistant", "content": "I found ORD-1. It shipped yesterday."},
            {"role": "assistant",
             "content": "Looking again, ORD-1 appears to have shipped yesterday."},
            {"role": "assistant",
             "content": "To rephrase, your order ORD-1 shipped yesterday."},
            {"role": "assistant",
             "content": "Just to confirm, ORD-1 left the warehouse yesterday."},
        ],
        "steps": [
            {"text": "I found ORD-1. It shipped yesterday."},
            {"text": "Looking again, ORD-1 appears to have shipped yesterday."},
            {"text": "To rephrase, your order ORD-1 shipped yesterday."},
            {"text": "Just to confirm, ORD-1 left the warehouse yesterday."},
        ],
        "final_text": "Just to confirm, ORD-1 left the warehouse yesterday.",
    }
    scored = score_row(row)
    assert scored["quality_scores"]["ping_pong"] < FAIL
    assert "stacked" in scored["quality_reason"]


def test_leak_fails():
    row = _good()
    row["prompt"] = "short prompt: look up ORD-1 with ordinary behavior"
    row["messages"][0]["content"] = row["prompt"]
    scored = score_row(row)
    assert scored["quality_scores"]["leak"] < FAIL
    assert "leak" in scored["quality_reason"]


def test_assistant_want_to_check_is_not_a_leak():
    row = _good()
    row["messages"][-1]["content"] = (
        "You may want to check your settings. I looked up ORD-4412.")
    row["final_text"] = row["messages"][-1]["content"]
    scored = score_row(row)
    assert scored["quality_scores"]["leak"] == 1.0
    assert scored["quality_reason"] == "conforms"


def test_dead_end_fails_complexity():
    row = {
        "prompt": "refund",
        "messages": [
            {"role": "user", "content": "refund"},
            {"role": "assistant", "content": "ok"},
        ],
        "steps": [{"text": "ok"}],
        "final_text": "ok",
    }
    scored = score_row(row)
    assert scored["quality_scores"]["complexity"] < FAIL
    assert "dead end" in scored["quality_reason"]


def test_clarify_and_stop_fails_complexity():
    row = {
        "prompt": "I need that PR merged, it has been stuck for days",
        "messages": [
            {"role": "user",
             "content": "I need that PR merged, it has been stuck for days"},
            {"role": "assistant",
             "content": "Could you please provide the repository name and PR number?"},
        ],
        "steps": [{"text": "Could you please provide the repository name and PR number?"}],
        "final_text": "Could you please provide the repository name and PR number?",
        "ask_family": "tool",
        "tool_known": True,
    }
    scored = score_row(row)
    assert scored["quality_scores"]["complexity"] < FAIL
    assert "clarify" in scored["quality_reason"] or "unused tools" in scored["quality_reason"]


def test_stale_final_text_fails_structure():
    first = "I've successfully moved the file."
    last = "The directory listing still shows the old name."
    row = {
        "prompt": "move src/old.js to src/new.js",
        "messages": [
            {"role": "user", "content": "move src/old.js to src/new.js"},
            {"role": "assistant", "content": first},
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "list_dir", "arguments": {"path": "src"}}]},
            {"role": "tool", "name": "list_dir", "content": '{"status": "ok"}'},
            {"role": "assistant", "content": last},
        ],
        "steps": [
            {"text": first},
            {"tool": "list_dir", "arguments": {"path": "src"},
             "result": {"status": "ok"}},
            {"text": last},
        ],
        "final_text": first,
    }
    scored = score_row(row)
    assert scored["quality_scores"]["structure"] < FAIL
    assert "stale" in scored["quality_reason"]


def test_ends_on_user_fails_structure():
    row = _good()
    row["messages"].append({"role": "user", "content": "thanks"})
    scored = score_row(row)
    assert scored["quality_scores"]["structure"] < FAIL
    assert "ends on user" in scored["quality_reason"]


def test_rank_rows_writes_fields():
    rows = [_good(), {
        "prompt": "How can I help you today?",
        "messages": [
            {"role": "user", "content": "How can I help you today?"},
            {"role": "assistant", "content": "I am the assistant."},
        ],
        "steps": [{"text": "I am the assistant."}],
        "final_text": "I am the assistant.",
    }]
    out = zps.rank_rows(rows)
    assert out is rows
    assert rows[0]["quality"] >= 0.85
    assert rows[1]["quality"] < rows[0]["quality"]
    assert "quality_reason" in rows[0] and "quality_scores" in rows[0]


def test_rank_path_rewrites_jsonl(tmp_path):
    src = tmp_path / "rows.jsonl"
    with open(src, "w") as fh:
        fh.write(json.dumps(_good()) + "\n")
        fh.write(json.dumps({
            "prompt": "asdf qwerty zxcv",
            "messages": [
                {"role": "user", "content": "asdf qwerty zxcv"},
                {"role": "assistant", "content": "ok"},
            ],
            "steps": [{"text": "ok"}],
            "final_text": "ok",
        }) + "\n")
    report = zps.rank(str(src))
    assert report["n"] == 2
    assert report["worst"]
    lines = src.read_text().strip().splitlines()
    scored = [json.loads(line) for line in lines]
    assert all("quality" in r and "quality_reason" in r for r in scored)
    assert scored[0]["prompt"].startswith("can you look up")
    assert scored[0]["quality"] > scored[1]["quality"]


def test_rank_filter_writes_kept_only(tmp_path):
    src = tmp_path / "all.jsonl"
    dest = tmp_path / "kept.jsonl"
    with open(src, "w") as fh:
        fh.write(json.dumps(_good()) + "\n")
        fh.write(json.dumps({
            "prompt": "!!!!",
            "messages": [
                {"role": "user", "content": "!!!!"},
                {"role": "assistant", "content": "ok"},
            ],
            "steps": [{"text": "ok"}],
            "final_text": "ok",
        }) + "\n")
    report = zps.rank(str(src), output=str(dest), min_quality=0.7)
    kept = [json.loads(line) for line in dest.read_text().splitlines() if line]
    assert report["n"] == 2
    assert report["n_kept"] == 1
    assert len(kept) == 1
    assert kept[0]["quality"] >= 0.7
    original = [json.loads(line) for line in src.read_text().splitlines() if line]
    assert len(original) == 2
    assert "quality" not in original[0]


def test_score_is_deterministic():
    row = _good()
    assert score_row(row) == score_row(row)


def test_steps_only_row_uses_conversation():
    row = {
        "prompt": "where's order ORD-12",
        "steps": [
            {"tool": "lookup_order", "arguments": {"order_id": "ORD-12"},
             "result": {"status": "ok"}, "text": "let me look"},
            {"user": "and the refund?"},
            {"text": "still pending on ORD-12"},
        ],
        "final_text": "still pending on ORD-12",
    }
    scored = score_row(row)
    assert scored["quality_scores"]["ping_pong"] >= FAIL
    assert scored["quality_scores"]["opener"] >= FAIL


def test_simulate_does_not_write_quality_until_rank(tmp_path):
    data = simulate_offline(budget=4, per_round=6)
    assert all(t.get("quality") is None for t in data.trajectories)
    path = data.save(str(tmp_path / "u.jsonl"))
    raw = json.loads(open(path).readline())
    assert "quality" not in raw
    report = data.rank()
    assert report["n"] == 4
    assert all(t.get("quality") is not None for t in data.trajectories)
    exported = zps._export_row(data.trajectories[0])
    assert "quality" in exported and "quality_scores" in exported
    assert exported["quality_reason"]


def test_live_example_sample_scores():
    path = REPO_ROOT / "examples" / "sft" / "coding.jsonl"
    if not path.is_file():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines() if line][:8]
    zps.rank_rows(rows)
    assert all("quality" in r for r in rows)
    assert any(r["quality_scores"]["opener"] >= FAIL for r in rows)
