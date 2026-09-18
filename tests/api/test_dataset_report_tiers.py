"""``dataset_report`` says how hard a set is, and warns when it is not."""

from __future__ import annotations

from whileai.simulations.score.preflight import dataset_report, format_dataset_report


def _row(stance: str | None, reward: int | None = 1, prompt: str = "") -> dict:
    dims = {"tool": "get_order"}
    if stance is not None:
        dims["stance"] = stance
    return {
        "prompt": prompt or f"{stance or 'plain'} ask {reward}",
        "reward": reward,
        "final_text": "done",
        "reason": "invented an order id" if reward == 0 else "",
        "steps": [],
        "scenario_dimensions": dims,
    }


def test_tier_counts_every_tier_and_keeps_unlabelled_apart() -> None:
    rows = (
        [_row("ordinary") for _ in range(5)]
        + [_row("boundary") for _ in range(3)]
        + [_row("adversarial") for _ in range(2)]
        + [_row("vague")]  # an alias of the ambiguous tier
        + [_row(None) for _ in range(2)]  # no stance: not evidence of an easy set
    )
    report = dataset_report(rows)
    assert report["tier_counts"] == {
        "ordinary": 5,
        "boundary": 3,
        "adversarial": 2,
        "unlabelled": 2,
        "ambiguous": 1,
    }
    assert report["hard_share"] == round(6 / 13, 3)


def test_tier_fail_rate_is_per_tier_and_skips_ungraded_rows() -> None:
    rows = [
        _row("ordinary", 1),
        _row("ordinary", 1),
        _row("ordinary", 0),
        _row("boundary", 0),
        _row("boundary", 0),
        _row("adversarial", None),
    ]
    report = dataset_report(rows)
    assert report["tier_fail_rate"] == {"boundary": 1.0, "ordinary": 0.333}
    assert report["hard_share"] == 0.5


def test_easy_set_warning_names_the_share_dial_and_the_pin() -> None:
    rows = [_row("ordinary", prompt=f"o{i}") for i in range(16)] + [
        _row("boundary", prompt=f"b{i}") for i in range(4)
    ]
    report = dataset_report(rows)
    (warning,) = report["warnings"]
    assert warning.startswith("easy set: 20% of rows")
    assert "hard_share=0.7" in warning and "asks for 70%" in warning
    assert "draws 70%" not in warning
    assert "dimensions={'stance'" in warning
    text = format_dataset_report(report)
    assert "Hard tiers:" in text and "20%" in text
    assert "! easy set" in text


def test_no_warning_at_the_floor_or_on_a_small_set() -> None:
    at_floor = [_row("ordinary", prompt=f"o{i}") for i in range(14)] + [
        _row("adversarial", prompt=f"a{i}") for i in range(6)
    ]
    assert dataset_report(at_floor)["hard_share"] == 0.3
    assert dataset_report(at_floor)["warnings"] == []
    small = [_row("ordinary", prompt=f"o{i}") for i in range(9)] + [_row("boundary")]
    report = dataset_report(small)
    assert report["hard_share"] == 0.1
    assert report["warnings"] == []
    assert "warnings" in dataset_report([])


def test_unlabelled_rows_get_their_own_warning_past_ten_percent() -> None:
    mostly_hard = [_row("boundary", prompt=f"b{i}") for i in range(17)]
    three_blank = mostly_hard + [_row(None, prompt=f"n{i}") for i in range(3)]
    (warning,) = dataset_report(three_blank)["warnings"]
    assert warning.startswith("3 rows carry no stance")
    assert "dimensions={'stance'" in warning
    one_blank = [*mostly_hard, _row(None, prompt="n0")]
    assert dataset_report(one_blank)["warnings"] == []
