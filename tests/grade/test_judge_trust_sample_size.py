"""A judge that agrees with every human label but has too few of them is
told to label more, with the count, not to change the judge."""

import whileai.simulations as wai


def _labeled(n: int, *, wrong: int = 0) -> list[dict]:
    rows = [
        {"prompt": f"p{i}", "final_text": "x", "reward": 1 if i % 2 else 0, "steps": []}
        for i in range(n)
    ]
    labels = [
        {"prompt": f"p{i}", "final_text": "x", "label": (0 if i < wrong else (1 if i % 2 else 0))}
        for i in range(n)
    ]
    wai.attach_labels(rows, labels, kind="human")
    return rows


def test_perfect_judge_on_few_labels_is_told_to_label_more():
    report = wai.judge_trust(_labeled(14))
    assert report["ok"] is False
    assert report["agreement"]["agreement"] == 1.0
    notes = [w for w in report["warnings"] if "floor" in w]
    assert len(notes) == 1
    assert "The judge is not the problem" in notes[0]
    assert "Label about 16 rows" in notes[0]
    assert "Change the judge" not in notes[0]


def test_enough_perfect_labels_pass():
    report = wai.judge_trust(_labeled(28))
    assert report["ok"] is True
    assert not [w for w in report["warnings"] if "floor" in w]


def test_judge_under_the_floor_is_still_told_to_change():
    # 6 of 14 wrong: agreement 0.57, under the floor on its own merits.
    report = wai.judge_trust(_labeled(14, wrong=6))
    notes = [
        w for w in report["warnings"] if "floor" in w and "agreement with human labels is" in w
    ]
    assert notes and "Change the judge prompt or the judge model" in notes[0]
