"""Evaluation re-run variance (rlhf-book appendix C) and the noise band on
delta_report."""

from __future__ import annotations

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.score.delta import delta_report, format_delta_report
from zeroproof.simulations.score.stats import eval_variance


def _rows(pass_rates: dict[str, float], k: int = 4, run_id: str | None = None) -> list[dict]:
    rows = []
    for task, p in pass_rates.items():
        passes = round(p * k)
        for i in range(k):
            row = {
                "prompt": task,
                "reward": 1 if i < passes else 0,
                "final_text": "ok",
                "steps": [],
                "messages": [],
                "markers": {"honest": 1.0 if i < passes else 0.0},
            }
            if run_id:
                row["lineage"] = {"scoring_run_id": run_id, "source": "eval"}
            rows.append(row)
    return rows


def test_eval_variance_over_separate_runs():
    base = {f"t{i}": 0.5 for i in range(10)}
    run1 = _rows(base)
    run2 = _rows({**base, "t0": 0.75})
    run3 = _rows({**base, "t1": 0.25})
    report = eval_variance(run1, run2, run3)
    assert report["n_runs"] == 3 and report["metric"] == "pass_at_1"
    assert report["means"] == {"run_1": 0.5, "run_2": 0.525, "run_3": 0.475}
    assert report["mean"] == 0.5 and report["run_std"] == 0.025
    assert report["noise_band"] == 0.05 and report["run_std_points"] == 2.5
    assert report["stability"] == "high_variance"
    assert report["tasks_in_every_run"] == 10 and report["notes"] == []


def test_eval_variance_splits_one_list_by_scoring_run_id_and_by_key():
    base = {f"t{i}": 0.5 for i in range(8)}
    rows = _rows(base, run_id="score_a") + _rows(base, run_id="score_b")
    rows += _rows({"t0": 1.0})  # no run id: left out, noted
    report = eval_variance(rows)
    assert report["n_runs"] == 2 and set(report["means"]) == {"score_a", "score_b"}
    assert report["run_std"] == 0.0 and report["stability"] == "very_stable"
    assert any("no run id" in n for n in report["notes"])
    assert any("two is a difference" in n for n in report["notes"])

    for row in rows:
        row["seed"] = 1 if row.get("lineage", {}).get("scoring_run_id") == "score_a" else 2
    assert eval_variance(rows, by="seed")["n_runs"] == 2
    assert eval_variance(rows, metric="marker:honest")["metric"] == "marker:honest"


def test_eval_variance_with_one_run_has_no_std():
    report = eval_variance(_rows({"t0": 0.5, "t1": 0.5}))
    assert report["n_runs"] == 0 and report["run_std"] is None  # no run ids on the rows
    report = eval_variance(_rows({"t0": 0.5}, run_id="x"))
    assert report["n_runs"] == 1 and report["run_std"] is None and report["stability"] is None


def test_delta_report_refuses_a_verdict_inside_the_noise_band():
    before = _rows({f"t{i}": 0.25 for i in range(12)})
    after = _rows({f"t{i}": 0.5 for i in range(12)})
    loud = delta_report(before, after, target="pass_at_1")
    assert loud["target_verdict"] == "moved" and loud["within_noise"] == []
    assert loud["metrics"]["pass_at_1"]["within_noise"] is False
    # a 0.25 delta inside a 2 x 0.2 band is what re-running the eval does
    quiet = delta_report(before, after, target="pass_at_1", run_std=0.2)
    assert quiet["target_verdict"] == "within_eval_noise" and quiet["ok"] is True
    assert quiet["within_noise"] == ["pass_at_1", "marker:honest"]
    assert quiet["improved"] == [] and quiet["run_std"] == 0.2
    assert any("re-run band" in w for w in quiet["warnings"])
    text = format_delta_report(quiet)
    assert text.startswith("pass_at_1: within_eval_noise") and "noise" in text

    # a regression inside the band is not a regression either
    for r in before:
        r["markers"]["polite"] = 1.0
    for i, r in enumerate(after):
        r["markers"]["polite"] = 0.0 if i % 4 == 0 else 1.0
    strict = delta_report(before, after, target="pass_at_1", must_not_regress=["polite"])
    assert strict["regressions"] == ["marker:polite"] and strict["ok"] is False
    lenient = delta_report(
        before, after, target="pass_at_1", must_not_regress=["polite"], run_std=0.2
    )
    assert lenient["regressions"] == [] and lenient["ok"] is True
    assert "marker:polite" in lenient["within_noise"]


def test_public_surface():
    assert "eval_variance" in zps.__all__ and callable(zps.eval_variance)
    with pytest.raises(ValueError, match="at least one"):
        zps.eval_variance()
