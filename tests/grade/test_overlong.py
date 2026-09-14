"""select_for_rl(truncated=): drop, keep, or penalize rollouts cut at the
token cap (rlhf-book ch. 6 DAPO overlong handling, ch. 7 overlong filtering)."""

from __future__ import annotations

import pytest

from zeroproof.simulations.score.optimize import optimize, select_for_rl

LONG_CUT = "The order shipped on Tuesday and the carrier picked it up from the warehouse " * 4
LONG_DONE = LONG_CUT.strip() + "."


def _row(prompt, reward, final, reason="graded"):
    return {
        "prompt": prompt,
        "reward": reward,
        "reason": reason,
        "final_text": final,
        "steps": [{"tool": "get_order", "arguments": {"id": "1"}, "result": {"ok": 1}}],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final},
        ],
    }


def _rows():
    return [
        _row("a", 1, LONG_DONE),
        _row("a", 1, LONG_CUT),  # judged a pass, but cut at the cap
        _row("a", 0, "No."),
        _row("b", 1, LONG_DONE),
        _row("b", 0, LONG_CUT),
        _row("b", 0.5, LONG_CUT + " and then", reason="reply truncated at token cap"),  # advisory
    ]


def test_drop_is_the_default_and_unchanged():
    rows = _rows()
    picked, report = select_for_rl(rows, target=100)
    assert report["truncated_policy"] == "drop" and report["truncated_dropped"] == 2
    assert report["truncated_kept"] == 0 and report["truncated_penalized"] == 0
    assert all("overlong" not in r for r in picked)
    assert all("overlong" not in r for r in rows)  # inputs untouched


def test_keep_marks_overlong_and_leaves_the_reward():
    picked, report = select_for_rl(_rows(), target=100, truncated="keep")
    assert report["truncated_policy"] == "keep" and report["truncated_kept"] == 3
    assert report["truncated_dropped"] == 0
    cut = [r for r in picked if r.get("overlong")]
    assert {r["reward"] for r in cut} == {1, 0} and all(
        r["markers"]["finished"] == 0.0 for r in cut
    )
    assert not any(r.get("reward") == 0.5 for r in picked)  # advisory 0.5 is still unusable
    legacy, legacy_report = select_for_rl(_rows(), target=100, drop_truncated=False)
    assert legacy_report["truncated_policy"] == "keep" and len(legacy) == len(picked)


def test_penalize_turns_a_cut_rollout_into_a_failure():
    picked, report = select_for_rl(_rows(), target=100, truncated="penalize")
    assert report["truncated_penalized"] == 3 and report["truncated_dropped"] == 0
    cut = [r for r in picked if r.get("overlong")]
    assert cut and all(r["reward"] == 0 for r in cut)
    assert sorted(r["reward_before_penalty"] for r in cut) == [0, 0.5, 1]
    # group a now has two fails and a pass: still mixed, still selected
    assert {r["prompt"] for r in picked} == {"a", "b"}
    with pytest.raises(ValueError, match="truncated must be one of"):
        select_for_rl(_rows(), truncated="ignore")


def test_optimize_threads_the_policy():
    _, report = optimize(_rows(), mode="rl", truncated="penalize")
    assert report["truncated_policy"] == "penalize" and report["truncated_penalized"] == 3
