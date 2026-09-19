"""Two arms that drew different situation sets are not a delta, and the
first line of ``format_delta_report`` says what the verdict is.

The 2026-09-18 live test compared a ``hard_share=0.4`` run with a
``hard_share=0.8`` run: 7 of 41 tasks paired, the delta over those seven
was -0.143 [-0.393, +0.107], and the report printed ``PASS``.
"""

from __future__ import annotations

from whileai.simulations.score.delta import delta_report, format_delta_report, headline_word


def _rows(pass_rates: dict[str, float], k: int = 4, **stamp) -> list[dict]:
    rows = []
    for task, p in pass_rates.items():
        passes = round(p * k)
        for i in range(k):
            rows.append({"prompt": task, "reward": 1.0 if i < passes else 0.0, **stamp})
    return rows


SHARED = {f"t{i}": 0.5 for i in range(7)}


def test_arms_that_share_under_half_their_tasks_are_not_comparable():
    before = _rows(SHARED | {f"a{i}": 0.75 for i in range(17)})
    after = _rows({t: 0.25 for t in SHARED} | {f"b{i}": 0.75 for i in range(17)})
    report = delta_report(before, after, n_boot=100)
    assert report["ok"] is False and "situations" in report["not_comparable"]
    assert report["metrics"]["pass_at_1"]["n_paired"] == 7
    line = [w for w in report["warnings"] if w.startswith("NOT COMPARABLE: the two arms drew")]
    assert len(line) == 1
    assert "7 of 41 tasks are on both sides (17 only before, 17 only after)" in line[0]
    assert "tasks=before_rows" in line[0] and "dataset_report" in line[0]
    text = format_delta_report(report)
    first = text.splitlines()[0]
    assert first == "FAIL: NOT COMPARABLE (situations)"
    assert "PASS" not in first


def test_a_few_dropped_tasks_stay_comparable():
    before = _rows({f"t{i}": 0.5 for i in range(20)})
    after = _rows({f"t{i}": 0.5 for i in range(20)} | {"z": 0.5})
    report = delta_report(before, after, n_boot=100)
    assert report["not_comparable"] == [] and report["ok"] is True
    assert not any("drew different situation sets" in w for w in report["warnings"])


def test_headline_words_match_the_verdict():
    same = {f"t{i}": 0.5 for i in range(24)}
    flat = delta_report(_rows(same), _rows(same), n_boot=100)
    assert flat["ok"] is True and flat["headline_verdict"] == "no_change_detected"
    assert format_delta_report(flat).splitlines()[0] == "NO DIFFERENCE"
    # a negative point estimate the interval does not settle is not a pass
    dipped = same | {"t0": 0.25, "t1": 0.25, "t2": 0.75}
    down = delta_report(_rows(same), _rows(dipped), n_boot=100)
    assert down["metrics"]["pass_at_1"]["delta"] < 0
    assert down["headline_verdict"] == "no_change_detected"
    assert format_delta_report(down).splitlines()[0] == "NO DIFFERENCE"
    # inside the re-run band the line says so
    noisy = delta_report(_rows(same), _rows({t: 0.5 for t in same}), run_std=0.05, n_boot=100)
    assert noisy["headline_verdict"] == "no_change_detected"
    assert headline_word({"ok": True, "headline_verdict": "within_eval_noise"}) == (
        "NO DIFFERENCE (within eval noise)"
    )
    # a supported gain is the one PASS
    up = delta_report(
        _rows({t: 0.25 for t in same}),
        _rows({t: 0.75 for t in same}),
        run_std=0.02,
        n_boot=100,
    )
    assert up["headline_verdict"] == "moved"
    assert format_delta_report(up).splitlines()[0] == "PASS"
    # a regression keeps FAIL
    regressed = delta_report(
        _rows({t: 0.75 for t in same}),
        _rows({t: 0.25 for t in same}),
        run_std=0.02,
        n_boot=100,
    )
    assert regressed["headline_verdict"] == "moved_the_wrong_way"
    assert format_delta_report(regressed).splitlines()[0] == "FAIL"
