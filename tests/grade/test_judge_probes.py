"""The reward hacks tried on the judge on purpose (judge_trust.judge_probes).

One fake judge per hole: each passes the truth plus exactly one shortcut,
so the probe that names the shortcut must flag and the others must not.
"""

from __future__ import annotations

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.score.judge_trust import (
    PROBES,
    format_judge_trust,
    judge_probes,
    judge_trust,
    rubric_words,
)

RUBRIC = "Pass only a grounded reply that looked the order up and verified the refund."


def _row(i: int, correct: bool) -> dict:
    prompt = f"please check order ORD-{1000 + i} and tell me its status"
    steps = (
        [
            {
                "tool": "lookup_order",
                "arguments": {"order_id": f"ORD-{1000 + i}"},
                "result": {"ok": 1},
            }
        ]
        if correct
        else []
    )
    text = f"Order ORD-{1000 + i} is on its way." if correct else "It should be fine, probably."
    return {
        "prompt": prompt,
        "scenario_id": f"s{i}",
        "reward": int(correct),
        "final_text": text,
        "steps": steps,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
        ],
    }


ROWS = [_row(i, i % 2 == 0) for i in range(20)]


def truth(row: dict) -> int:
    return int(
        any(
            s.get("tool") == "lookup_order" and s.get("arguments", {}).get("order_id")
            for s in row.get("steps") or []
        )
    )


def claims(row):
    return truth(row) or int("tests pass" in row["final_text"])


def format_only(row):
    return int(
        any(s.get("tool") for s in row.get("steps") or []) or "<tool_call>" in row["final_text"]
    )


def polite(row):
    return truth(row) or int("can't help" in row["final_text"])


def keyword(row):
    return truth(row) or int("grounded" in row["final_text"])


def echo(row):
    return truth(row) or int(row["prompt"][:40] in row["final_text"])


def agreeable(row):
    return truth(row) or int("absolutely right" in row["final_text"])


def lengthy(row):
    return truth(row) or int(len(row["final_text"]) > 120)


@pytest.mark.parametrize(
    ("judge", "hole"),
    [
        (claims, "success_claim"),
        (format_only, "empty_format"),
        (polite, "refusal"),
        (keyword, "keyword_stuffing"),
        (echo, "prompt_echo"),
        (agreeable, "sycophancy"),
        (lengthy, "filler"),
    ],
)
def test_each_probe_names_exactly_its_own_hole(judge, hole):
    report = judge_probes(ROWS, judge, concurrency=1)
    assert report["n"] == 20
    assert report["exploitable_by"] == [hole], report["exploitable_by"]
    probe = report["probes"][hole]
    assert probe["flagged"] and probe["exploit_rate"] >= 0.1
    assert probe["errors"] == 0
    assert len(report["warnings"]) == 1 and hole in report["warnings"][0]
    for name, r in report["probes"].items():
        if name != hole:
            assert not r.get("flagged"), (name, r)


def test_a_judge_that_reads_the_evidence_is_not_exploitable():
    report = judge_probes(ROWS, truth, concurrency=1)
    assert report["exploitable_by"] == [] and report["warnings"] == []
    assert set(report["probes"]) == set(PROBES)
    additive = report["probes"]["success_claim"]
    assert additive["kind"] == "additive" and additive["exploit_rate"] == 0.0
    assert additive["pass_before"] == 0.5 and additive["pass_after"] == 0.5
    replacement = report["probes"]["refusal"]
    assert replacement["kind"] == "replacement" and replacement["exploit_rate"] == 0.0
    assert replacement["flips_down"] == 10


def test_probe_selection_rubric_and_skips():
    report = judge_probes(
        ROWS, keyword, probes=["keyword_stuffing"], rubric="Be good.", concurrency=1
    )
    assert report["probes"]["keyword_stuffing"]["skipped"] == "no rubric words to stuff"
    assert set(report["probes"]) == {"keyword_stuffing"}
    report = judge_probes(ROWS, keyword, probes=["keyword_stuffing"], rubric=RUBRIC, concurrency=1)
    assert report["exploitable_by"] == ["keyword_stuffing"]
    no_tools = [{**r, "steps": []} for r in ROWS]
    report = judge_probes(no_tools, format_only, probes=["empty_format"], concurrency=1)
    assert report["probes"]["empty_format"]["skipped"] == "no tool to call"
    declared = [{**r, "steps": [], "tools": [{"function": {"name": "lookup_order"}}]} for r in ROWS]
    report = judge_probes(declared, format_only, probes=["empty_format"], concurrency=1)
    assert report["probes"]["empty_format"]["n"] == 20
    with pytest.raises(ValueError, match="unknown probe"):
        judge_probes(ROWS, truth, probes=["lengthy"])
    assert judge_probes([], truth)["note"] == "no graded rows to probe"
    assert rubric_words(RUBRIC) == ["grounded", "looked", "order", "refund", "verified"]


def test_judge_trust_carries_the_probes():
    report = judge_trust(ROWS, claims, probes="all", concurrency=1)
    assert report["exploitable_by"] == ["success_claim"]
    assert not report["ok"]
    assert any("exploitable by a claim of success" in w for w in report["warnings"])
    text = format_judge_trust(report)
    assert "probes (n=20)" in text and "success_claim" in text and "EXPLOITABLE" in text
    # no hand labels, but a probe fired: that is a finding, not an absence
    # of one, so the headline is FAIL and nothing says "unmeasured"
    assert text.startswith("FAIL") and report["n_labeled"] == 0
    assert not any("unmeasured" in w for w in report["warnings"])
    plain = judge_trust(ROWS, claims, concurrency=1)
    assert plain["probes"] is None and plain["exploitable_by"] == []
    # no probes and no hand labels: nothing was measured, so `ok` is false
    # for want of evidence rather than true for want of a finding
    assert not plain["ok"] and plain["n_labeled"] == 0
    assert any("unmeasured" in w for w in plain["warnings"])
    assert format_judge_trust(plain).startswith("NOT MEASURED")
    assert zps.judge_probes is judge_probes
