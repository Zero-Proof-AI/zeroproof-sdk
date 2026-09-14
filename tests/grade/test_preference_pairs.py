"""Preference pairs carry margin, policy provenance, and the length check."""

import json

import pytest

from zeroproof.simulations.export import export_preference
from zeroproof.simulations.score.judging import build_preference_pairs


def _row(prompt, reward, final, model="qwen-a", **extra):
    return {
        "prompt": prompt,
        "reward": reward,
        "judge_status": "ok",
        "final_text": final,
        "steps": [],
        "model_version": model,
        **extra,
    }


def test_default_margin_pairs_only_full_pass_against_full_fail():
    rows = [
        _row("ask", 1, "Order shipped."),
        _row("ask", 0.5, "Order maybe shipped, I think."),
        _row("ask", 0, "I invented it."),
    ]
    pairs, report = build_preference_pairs(rows)
    assert len(pairs) == 1
    pair = pairs[0]
    assert (pair["chosen_score"], pair["rejected_score"], pair["margin"]) == (1.0, 0.0, 1.0)
    assert report["partial_score_pairs"] == 0
    assert report["min_margin"] == 1.0


def test_lower_margin_admits_partial_credit_rows():
    rows = [
        _row("ask", 1, "Order shipped."),
        _row("ask", 0.5, "Order maybe shipped, I think."),
    ]
    none, _ = build_preference_pairs(rows)
    assert none == []
    pairs, report = build_preference_pairs(rows, min_margin=0.5)
    assert len(pairs) == 1
    assert pairs[0]["margin"] == 0.5
    assert report["partial_score_pairs"] == 1
    assert report["mean_margin"] == 0.5


def test_length_match_takes_the_closest_rejected_reply():
    short_fail = _row("ask", 0, "No.")
    long_fail = _row("ask", 0, "I invented a tracking number and told the user it shipped.")
    chosen = _row("ask", 1, "Order 4412 shipped yesterday; tracking is 1Z999.")
    pairs, _ = build_preference_pairs([short_fail, chosen, long_fail])
    assert pairs[0]["rejected"] is long_fail
    assert abs(pairs[0]["length_delta"]) < len(chosen["final_text"]) - len("No.")
    unmatched, _ = build_preference_pairs([short_fail, chosen, long_fail], length_match=False)
    assert unmatched[0]["rejected"] is short_fail


def test_policy_provenance_and_mixed_policy_warning():
    rows = [
        _row("ask", 1, "Order shipped.", model="student"),
        _row("ask", 0, "I invented it.", model="teacher"),
        _row("ask2", 1, "Refund issued.", model="student"),
        _row("ask2", 0, "Refund invented.", model="student"),
        _row("ask3", 1, "Done.", model=None),
        _row("ask3", 0, "Not done.", model=None),
    ]
    pairs, report = build_preference_pairs(rows)
    by_prompt = {p["prompt"]: p for p in pairs}
    assert by_prompt["ask"]["same_policy"] is False
    assert by_prompt["ask"]["chosen_model"] == "student"
    assert by_prompt["ask"]["rejected_model"] == "teacher"
    assert by_prompt["ask2"]["same_policy"] is True
    assert by_prompt["ask3"]["same_policy"] is None
    assert report["same_policy_pairs"] == 1
    assert report["mixed_policy_pairs"] == 1
    assert any("mix policies" in w for w in report["warnings"])


def test_length_exploit_warning_when_chosen_is_always_longer():
    rows = []
    for i in range(8):
        rows.append(_row(f"ask{i}", 1, "A long, careful, complete answer with every detail. " * 3))
        rows.append(_row(f"ask{i}", 0, "no"))
    pairs, report = build_preference_pairs(rows)
    assert len(pairs) == 8
    assert report["length"]["chosen_longer_frac"] == 1.0
    assert any("longer reply" in w for w in report["warnings"])


def test_each_row_pairs_at_most_once_per_prompt():
    rows = [
        _row("ask", 1, "good one"),
        _row("ask", 1, "good two"),
        _row("ask", 0, "bad one"),
    ]
    pairs, report = build_preference_pairs(rows, max_pairs_per_prompt=3)
    assert len(pairs) == 1
    assert report["prompts_with_contrast"] == 1


def test_export_carries_pair_metadata(tmp_path):
    rows = [
        _row("ask", 1, "Order 4412 shipped yesterday.", model="student"),
        _row("ask", 0, "I invented it.", model="student"),
    ]
    pairs, _ = build_preference_pairs(rows)
    out = str(tmp_path / "prefs.jsonl")
    report = export_preference(pairs, out)
    assert report["mean_margin"] == 1.0
    assert report["chosen_longer_frac"] == 1.0
    line = json.loads(open(out).readline())
    for key in (
        "chosen_score",
        "rejected_score",
        "margin",
        "chosen_model",
        "rejected_model",
        "same_policy",
        "length_delta",
    ):
        assert key in line, key
    assert line["same_policy"] is True
    assert line["chosen"][-1]["content"] == "Order 4412 shipped yesterday."


def test_export_refuses_to_write_an_empty_pair_file(tmp_path):
    rows = [_row("ask", 1, "Shipped."), _row("ask", 1, "Shipped yesterday.")]  # all pass
    pairs, _ = build_preference_pairs(rows)
    assert pairs == []
    out = tmp_path / "prefs.jsonl"
    with pytest.raises(ValueError, match="no_preference_pairs"):
        export_preference(pairs, str(out))
    assert not out.exists(), "a 0-byte JSONL crashes datasets downstream; no file is written"
    report = export_preference(pairs, str(out), validate=False)
    assert report["pairs"] == 0 and report["path"] is None
    assert not out.exists()
