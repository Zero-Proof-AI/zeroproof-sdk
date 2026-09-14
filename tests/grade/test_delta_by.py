"""delta_report(by=...): the target per group, so a moved headline cannot hide a
group that moved the other way."""

from __future__ import annotations

from zeroproof.simulations.score.delta import delta_report, format_delta_report


def _rows(spec: dict[str, tuple], *, n_tasks: int = 12, k: int = 4):
    """Two row sets from per-group (pass rate before, pass rate after[, tasks]);
    deterministic, paired by task id, ``k`` rows per task."""
    before, after = [], []
    for g, cfg in spec.items():
        pa, pb = cfg[0], cfg[1]
        n = cfg[2] if len(cfg) > 2 else n_tasks
        for t in range(n):
            for i in range(k):
                # Spread the rows of this group evenly over 0..100, so a
                # pass rate of p passes the first p of them.
                slot = (t * k + i) * 100 / (n * k)
                before.append(
                    {
                        "task_id": f"{g}-{t}",
                        "prompt": f"{g} prompt {t}",
                        "category": g,
                        "reward": 1 if slot < pa * 100 else 0,
                        "markers": {"well_formed": 1.0},
                    }
                )
                after.append(
                    {
                        "task_id": f"{g}-{t}",
                        "prompt": f"{g} prompt {t}",
                        "category": g,
                        "reward": 1 if slot < pb * 100 else 0,
                        "markers": {"well_formed": 1.0},
                    }
                )
    return before, after


def test_groups_split_the_target_and_flag_the_one_that_dropped():
    # 40 with-id tasks against 6 off-topic ones: the headline is the big group.
    before, after = _rows({"with_id": (0.2, 0.9, 40), "off_topic": (0.9, 0.3, 6)})
    report = delta_report(before, after, target="pass_at_1", by="category", n_boot=300)
    assert report["target_verdict"] == "moved"
    assert report["by"] == "category"
    assert set(report["groups"]) == {"with_id", "off_topic"}
    assert report["groups"]["with_id"]["verdict"] == "b_better"
    assert report["groups"]["off_topic"]["verdict"] == "a_better"
    assert report["groups_down"] == ["off_topic"]
    assert report["groups"]["off_topic"]["rows_a"] == 24
    assert any("moved the wrong way for off_topic" in w for w in report["warnings"])
    assert report["ok"], "groups warn; only must_not_regress fails the report"
    text = format_delta_report(report)
    assert "by category:" in text and "off_topic" in text and "DOWN" in text


def test_by_reads_a_marker_or_a_callable_and_skips_ungrouped_rows():
    before, after = _rows({"a": (0.2, 0.8)})
    for row in before + after:
        row["markers"]["kind"] = row.pop("category")
    report = delta_report(before, after, by="kind", n_boot=200)
    assert list(report["groups"]) == ["a"]
    report = delta_report(before, after, by=lambda r: r["markers"]["kind"].upper(), n_boot=200)
    assert list(report["groups"]) == ["A"] and report["by"] == "<lambda>"
    for row in before + after:
        row["markers"].pop("kind")
    report = delta_report(before, after, by="kind", n_boot=200)
    assert report["groups"] == {} and any(
        "no group is on both row sets" in w for w in report["warnings"]
    )


def test_without_by_the_report_is_unchanged():
    before, after = _rows({"a": (0.2, 0.8)})
    report = delta_report(before, after, n_boot=200)
    assert report["by"] is None and report["groups"] is None and report["groups_down"] == []
    assert "by " not in format_delta_report(report)
