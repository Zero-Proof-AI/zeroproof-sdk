"""An eval-scored row is reported, by count, wherever it could become a reward."""

from __future__ import annotations

from zeroproof.simulations.score.judging import build_preference_pairs, evaluate, run_judge
from zeroproof.simulations.score.optimize import eval_sourced, select_for_rl, select_for_sft


def _row(prompt: str, reward, final: str, **extra) -> dict:
    return {
        "prompt": prompt,
        "reward": reward,
        "judge_status": "ok",
        "final_text": final,
        "steps": [{"tool": "get_issue", "arguments": {"number": 1}, "result": {"status": "ok"}}],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final},
        ],
        "model_version": "qwen-a",
        **extra,
    }


def _mixed_groups(**extra) -> list[dict]:
    rows = []
    for i in range(4):
        rows.append(_row(f"ask {i}", 1, f"Issue {i} is open.", **extra))
        rows.append(_row(f"ask {i}", 0, "I invented it.", **extra))
    return rows


def test_grade_sourced_rows_report_zero():
    rows = _mixed_groups(lineage={"source": "grade", "scoring_run_id": "score_a"})
    assert eval_sourced(rows) == 0
    _, rl = select_for_rl(rows, target=100)
    assert rl["eval_sourced"] == 0
    assert not any("evaluate()" in w for w in rl["hygiene_warnings"])
    _, sft = select_for_sft(rows, target=100)
    assert sft["eval_sourced"] == 0
    assert "warning" not in sft
    _, pairs = build_preference_pairs(rows)
    assert pairs["eval_sourced"] == 0


def test_eval_sourced_rows_are_counted_and_warned():
    rows = _mixed_groups(lineage={"source": "eval", "scoring_run_id": "score_e"})
    assert eval_sourced(rows) == 8
    picked, rl = select_for_rl(rows, target=100)
    assert rl["eval_sourced"] == len(picked) == 8
    assert any("evaluate()" in w for w in rl["hygiene_warnings"])
    picked, sft = select_for_sft(rows, target=100)
    assert sft["eval_sourced"] == len(picked) == 4
    assert "evaluate()" in sft["warning"]
    made, pairs = build_preference_pairs(rows)
    assert pairs["eval_sourced"] == len(made) == 4
    assert any("evaluate()" in w for w in pairs["warnings"])


def test_evaluate_then_select_for_rl_reports_the_leak():
    def judge(row):
        return 1 if "open" in str(row.get("final_text")) else 0

    rows = [dict(r, reward=None) for r in _mixed_groups()]
    for r in rows:
        r.pop("judge_status")
    scored = evaluate(rows, judge)
    _, report = scored.select_for_rl()
    assert report["eval_sourced"] == 8
    graded = run_judge(rows, judge)
    _, report = graded.select_for_rl()
    assert report["eval_sourced"] == 0


def test_select_for_rl_names_the_reason_when_nothing_survives():
    # every group unanimous: graded, but no contrast for an RL update
    rows = [_row(f"ask {i}", 1, f"Issue {i} is open.") for i in range(3) for _ in range(2)]
    selected, report = select_for_rl(rows)
    assert selected == []
    assert report["eval_sourced_input"] == 0
    assert not any("grade first" in w for w in report["hygiene_warnings"])
    assert any(
        w.startswith("nothing selected: 6 graded row(s) in") for w in report["hygiene_warnings"]
    )


def test_select_for_rl_reports_eval_rows_on_the_input_even_when_none_selected():
    rows = evaluate(
        [_row(f"ask {i}", None, f"Issue {i} is open.") for i in range(3) for _ in range(2)],
        lambda r: 1,
    ).rows
    selected, report = select_for_rl(rows)
    assert selected == []
    assert report["eval_sourced"] == 0
    assert report["eval_sourced_input"] == 6
    assert any(
        "6 input row(s) carry rewards from evaluate()" in w for w in report["hygiene_warnings"]
    )
