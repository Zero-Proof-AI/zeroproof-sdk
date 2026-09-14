"""Stock over-optimization markers detect, stamp, and feed delta_report."""

from __future__ import annotations

from zeroproof.simulations.score.delta import delta_report
from zeroproof.simulations.score.markers import (
    STOCK_MARKERS,
    behavioral_markers,
    detect,
    format_markers,
    mark_rows,
    row_markers,
)


def row(text, **kw):
    r = {"final_text": text, "prompt": kw.get("prompt", "p")}
    r.update(kw)
    return r


def test_names_and_detect():
    assert set(STOCK_MARKERS) == {
        "boilerplate",
        "self_reference",
        "hedging",
        "refusal",
        "sycophancy",
    }
    assert detect("boilerplate", "Certainly! Here is the answer.") == 1
    assert detect("self_reference", "As an AI language model, I ...") == 1
    assert detect("refusal", "I can't help with that.") == 1
    assert detect("sycophancy", "Great question! You're absolutely right.") == 1
    assert detect("hedging", "It depends, generally speaking.") == 1
    assert detect("boilerplate", "The capital is Paris.") == 0


def test_row_markers_reads_last_assistant_turn():
    r = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "As an AI, I can't help with that."},
        ]
    }
    m = row_markers(r)
    assert m["self_reference"] == 1 and m["refusal"] == 1


def test_behavioral_markers_rates():
    rows = [
        row("Certainly! Here's how."),
        row("The answer is 4."),
        row("I cannot assist with that."),
        row("The answer is 5."),
    ]
    rates = behavioral_markers(rows)
    assert rates["boilerplate"] == 0.25
    assert rates["refusal"] == 0.25
    assert rates["sycophancy"] == 0.0


def test_mark_rows_stamps_and_preserves():
    rows = [row("Great question!", markers={"custom": 1.0})]
    marked = mark_rows(rows)
    assert marked[0]["markers"]["sycophancy"] == 1
    assert marked[0]["markers"]["custom"] == 1.0  # existing kept
    assert rows[0].get("markers") == {"custom": 1.0}  # original untouched


def test_extra_custom_detector():
    rows = [row("hello world")]
    marked = mark_rows(rows, extra={"words": lambda r: len(r["final_text"].split())})
    assert marked[0]["markers"]["words"] == 2.0


def test_feeds_delta_report():
    # Same tasks before and after; refusal rises after training (over-optimization).
    before = mark_rows(
        [row("The answer is 4.", prompt="t1"), row("The answer is 5.", prompt="t2")]
    )
    after = mark_rows(
        [
            row("I can't help with that.", prompt="t1"),
            row("I won't do that.", prompt="t2"),
        ]
    )
    rep = delta_report(before=before, after=after, target="refusal", must_not_regress=["refusal"])
    assert "marker:refusal" in rep["metrics"]  # delta keys markers as marker:<name>


def test_format_and_empty():
    assert behavioral_markers([]) == {name: 0.0 for name in STOCK_MARKERS}
    assert "refusal" in format_markers(behavioral_markers([row("I cannot help.")]))
