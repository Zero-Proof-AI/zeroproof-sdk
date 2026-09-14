"""Intervals, paired comparison, decontamination, delta report, judge trust."""

from __future__ import annotations

import json

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.score.agreement import judge_agreement
from zeroproof.simulations.score.delta import delta_report, format_delta_report
from zeroproof.simulations.score.judge_trust import (
    FILLER,
    format_judge_trust,
    judge_trust,
    length_sensitivity,
)
from zeroproof.simulations.score.stats import (
    bootstrap_ci,
    compare_runs,
    decontaminate,
    marker_summary,
    metric_summary,
    task_means,
    wilson_interval,
)


def _row(prompt, reward=1, final="Issue 1 is open.", *, markers=None, gold=None):
    row = {
        "prompt": prompt,
        "reward": reward,
        "final_text": final,
        "steps": [{"tool": "get_issue", "arguments": {}, "result": {"ok": 1}}],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final},
        ],
    }
    if markers is not None:
        row["markers"] = markers
    if gold is not None:
        row["gold_reward"] = gold
    return row


def _run(pass_rates: dict[str, float], k: int = 4, marker_shift: float = 0.0) -> list[dict]:
    rows = []
    for task, p in pass_rates.items():
        passes = round(p * k)
        for i in range(k):
            r = 1 if i < passes else 0
            rows.append(_row(task, r, markers={"honest": min(1.0, r * 0.8 + 0.1 + marker_shift)}))
    return rows


# ---------------------------------------------------------------- intervals


def test_wilson_and_bootstrap_basics():
    lo, hi = wilson_interval(8, 10)
    assert 0.49 < lo < 0.6 and 0.9 < hi < 0.97
    assert wilson_interval(0, 0) is None
    assert bootstrap_ci([1.0, 1.0]) is None
    ci = bootstrap_ci([0.0, 0.5, 1.0, 0.5, 0.5, 0.25, 0.75], seed=1)
    assert ci is not None and ci[0] <= 0.5 <= ci[1]
    assert bootstrap_ci([0.0, 0.5, 1.0, 0.5], seed=3) == bootstrap_ci([0.0, 0.5, 1.0, 0.5], seed=3)


def test_task_means_and_metric_summary_cluster_by_task():
    rows = _run({"a": 1.0, "b": 0.0, "c": 0.5, "d": 0.25})
    assert task_means(rows) == {"a": 1.0, "b": 0.0, "c": 0.5, "d": 0.25}
    summary = metric_summary(rows)
    assert summary["n_tasks"] == 4 and summary["n_rows"] == 16
    assert summary["mean"] == pytest.approx(0.4375)
    assert summary["ci95"] is not None and summary["ci95"][0] <= 0.4375 <= summary["ci95"][1]
    markers = marker_summary(rows)
    assert set(markers) == {"honest"} and markers["honest"]["n_tasks"] == 4


def test_pass_at_carries_a_ci():
    rows = _run({"a": 1.0, "b": 0.0, "c": 0.5, "d": 0.25, "e": 0.75})
    rates = zps.pass_at(rows)
    assert rates.ci95 is not None and rates.ci95[0] <= rates.pass_at_1 <= rates.ci95[1]
    assert rates.to_dict()["ci95"] == list(rates.ci95)
    assert zps.pass_at(_run({"a": 0.5, "b": 0.5})).ci95 is None


# ---------------------------------------------------------------- comparison


def test_compare_runs_detects_a_real_paired_improvement():
    before = _run({f"t{i}": 0.25 for i in range(12)})
    after = _run({f"t{i}": 0.75 for i in range(12)})
    out = compare_runs(before, after)
    assert out["paired"] and out["n_paired"] == 12
    assert out["delta"] == pytest.approx(0.5)
    assert out["ci95"][0] > 0 and out["verdict"] == "b_better"
    assert out["p_value"] is not None and out["p_value"] < 0.05


def test_compare_runs_no_difference_and_unpaired_fallback():
    a = _run({f"t{i}": 0.5 for i in range(10)})
    same = compare_runs(a, a)
    assert same["delta"] == 0 and same["verdict"] == "no_difference_detected"
    b = _run({f"u{i}": 0.5 for i in range(10)})
    unpaired = compare_runs(a, b)
    assert not unpaired["paired"] and unpaired["n_paired"] == 0 and "unpaired" in unpaired["note"]
    assert unpaired["n_only_a"] == 10 and unpaired["n_only_b"] == 10
    assert compare_runs(a, b, metric="marker:honest")["metric"] == "marker:honest"
    assert compare_runs([], [])["verdict"] == "insufficient_data"


# ---------------------------------------------------------------- decontamination


def test_decontaminate_by_ngram_and_exact_match(tmp_path):
    evals = [
        {
            "prompt": "please look up order ORD-4017 and tell me whether the refund has been issued yet"
        },
        {"prompt": "short eval ask"},
    ]
    rows = [
        _row("please look up order ORD-4017 and tell me whether the refund has been issued yet"),
        _row("could you look up order ORD-4017 and tell me whether the refund has been issued"),
        _row("Short eval ask"),
        _row("a different question about a different order entirely, thanks"),
        _row(
            "unrelated",
            final="please look up order ORD-4017 and tell me whether the refund has been issued yet",
        ),
    ]
    kept, report = decontaminate(rows, evals)
    assert report["n_contaminated"] == 4 and len(kept) == 1
    assert report["examples"][0]["field"] == "prompt"
    assert report["examples"][-1]["field"] == "final_text"
    assert report["n_eval_rows"] == 2 and report["ngram"] == 8

    # The eval set's own replies are not a contamination source: shared tool
    # boilerplate between two replies says nothing about the eval question.
    evals_with_replies = [
        {"prompt": "an eval question nobody trained on", "final_text": "Issue 1 is open."}
    ]
    _kept3, report3 = decontaminate(rows, evals_with_replies, n=3)
    assert report3["n_contaminated"] == 0
    with_answer = [{"prompt": "another eval", "answer": "Issue 1 is open."}]
    # four rows carry the default reply; the fifth reply is the eval prompt text
    assert decontaminate(rows, with_answer, n=3)[1]["n_contaminated"] == len(rows) - 1

    path = tmp_path / "eval.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in evals))
    _kept2, report2 = decontaminate(rows, [str(path)], n=15)
    assert report2["n_contaminated"] == 3  # the paraphrase shares 14 words, not 15
    assert zps.decontaminate(rows, evals)[1]["n"] == 5


# ---------------------------------------------------------------- delta report


def test_delta_report_headline_and_regressions():
    before = _run({f"t{i}": 0.25 for i in range(12)}, marker_shift=0.0)
    after = _run({f"t{i}": 0.75 for i in range(12)}, marker_shift=0.0)
    # Make a second marker regress after training.
    for r in before:
        r["markers"]["polite"] = 1.0
    for i, r in enumerate(after):
        r["markers"]["polite"] = 0.0 if i % 2 else 1.0
    report = delta_report(before, after, target="pass_at_1", must_not_regress=["polite"])
    assert report["target_verdict"] == "moved" and report["target_delta"] == pytest.approx(0.5)
    assert report["regressions"] == ["marker:polite"] and report["ok"] is False
    assert "marker:honest" in report["improved"]
    text = format_delta_report(report)
    assert text.startswith("pass_at_1: moved") and "FAIL" in text and "REGRESSION" in text

    soft = delta_report(before, after, target="pass_at_1")
    assert soft["ok"] and soft["slipped"] == ["marker:polite"]
    missing = delta_report(before, after, target="marker:nope")
    assert missing["target_verdict"] == "target_not_measured"
    assert zps.delta_report(before, after)["n_paired_tasks"] == 12


# ---------------------------------------------------------------- judge trust


def test_agreement_and_length_sensitivity():
    rows = [_row(f"t{i}", 1, gold=1) for i in range(8)]
    rows += [_row(f"u{i}", 0, gold=0) for i in range(8)]
    rows += [_row(f"v{i}", 0, gold=1) for i in range(2)]
    rows += [_row(f"w{i}", 1, gold=0) for i in range(2)]
    a = judge_agreement(rows)
    assert a["n"] == 20 and a["agreement"] == pytest.approx(0.8)
    assert 0.5 < a["kappa"] < 0.7 and a["confusion"] == {"tp": 8, "fp": 2, "fn": 2, "tn": 8}

    # Judge passes long replies, fails short ones, with the same gold label.
    rows = []
    for i in range(12):
        long = i % 2 == 1
        final = ("Detailed answer. " * 12) if long else "Short."
        rows.append(_row(f"t{i}", int(long), final, gold=1))
    ls = length_sensitivity(rows)
    assert ls["gold_1"]["gap_long_minus_short"] == pytest.approx(1.0)
    assert ls["flagged"]


def test_judge_trust_report_with_a_length_reading_judge():
    def judge(row):
        # Pays for length: long replies pass, short fail; deterministic.
        return {"score": 1 if len(str(row.get("final_text") or "")) > 60 else 0, "reason": "len"}

    rows = []
    for i in range(24):
        long = i % 2 == 1
        final = ("Detailed answer. " * 6) if long else "Short reply."
        gold = 1 if i % 3 else 0
        rows.append(_row(f"t{i}", int(long), final, gold=gold))
    report = judge_trust(rows, judge, sample=24, concurrency=1)
    assert report["n_labeled"] == 24 and report["agreement"]["n"] == 24
    assert report["held_out_halves"]["a"]["n"] + report["held_out_halves"]["b"]["n"] == 24
    p = report["perturbation"]
    assert p["consistency_flip_rate"] == 0.0
    assert p["filler_flip_rate"] > 0 and p["filler_flips_up"] > 0 and p["flagged_length"]
    assert any("filler" in w for w in report["warnings"]) and report["ok"] is False
    assert report["disagreements"] and {"task", "gold", "judge", "reason"} <= set(
        report["disagreements"][0]
    )
    text = format_judge_trust(report)
    assert text.startswith("FAIL") and "filler flips" in text and FILLER.strip() not in text

    bare = zps.judge_trust([_row("a", 1)])
    assert bare["n_labeled"] == 0 and any("gold_reward" in w for w in bare["warnings"])
    assert bare["perturbation"] is None


def test_judge_trust_says_when_gold_has_one_class_only():
    # Every gold label is 1 (a self-judge that passed everything): kappa is
    # 0 and the short/long split looks like bias, but neither is a finding.
    rows = []
    for i in range(12):
        long = i % 2 == 1
        rows.append(
            _row(f"t{i}", int(long), ("Detailed answer. " * 12) if long else "Short.", gold=1)
        )
    report = judge_trust(rows)
    assert report["gold_degenerate"] is True
    assert any("gold labels are all 1" in w for w in report["warnings"])
    assert not any(w.startswith("kappa") for w in report["warnings"])
    assert not any("length bias" in w for w in report["warnings"])
    assert report["ok"] is True  # nothing the labels can support was flagged
    mixed = rows + [_row(f"u{i}", 0, "Short.", gold=0) for i in range(6)]
    assert judge_trust(mixed)["gold_degenerate"] is False
