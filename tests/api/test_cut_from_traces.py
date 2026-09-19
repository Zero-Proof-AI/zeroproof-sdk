"""wai.cut / wai.cuts: the platform's "Make training data" button as one line."""

from __future__ import annotations

import pytest

import whileai.simulations as wai


@pytest.fixture
def calls(monkeypatch):
    """Record what the client sends, and answer as the gate does."""
    seen: list[dict] = []

    def fake_call(method, path, api_key=None, body=None, **kw):
        seen.append({"method": method, "path": path, "body": body})
        if path.startswith("/traces/cuts"):
            return {"prompts": 41, "rl": {"prompts": 30, "train": 24, "holdout": 6}}
        return {
            "made": [
                {"role": "train", "datasetId": "ds_train", "rows": 180, "prompts": 24},
                {"role": "holdout", "datasetId": "ds_hold", "rows": 44, "prompts": 6},
            ],
            "kind": "rl",
            "band": [0.2, 0.8],
        }

    from whileai.simulations.ingest import platform

    monkeypatch.setattr(platform, "_call", fake_call)
    return seen


def test_cut_posts_the_filter_and_names_both_sets(calls):
    made = wai.cut(agent="My-Agent", kind="rl")

    assert calls[0]["method"] == "POST"
    assert calls[0]["path"] == "/datasets/cut"
    assert calls[0]["body"] == {"filter": {"agent": "my-agent", "from": "all"}, "kind": "rl"}
    # The two ids are what the next line needs, without digging through `made`.
    assert made["train"]["datasetId"] == "ds_train"
    assert made["holdout"]["datasetId"] == "ds_hold"


def test_cut_passes_window_band_and_trace_filters(calls):
    wai.cut(agent="a", kind="sft", since="7d", band=(0.1, 0.9), holdout=0.25, model="gpt-4.1")

    body = calls[0]["body"]
    assert body["kind"] == "sft"
    assert body["filter"] == {"model": "gpt-4.1", "agent": "a", "from": "7d"}
    assert (body["lo"], body["hi"]) == (0.1, 0.9)
    assert body["holdout"] == 0.25


def test_cut_without_an_agent_covers_every_trace(calls):
    wai.cut()

    assert calls[0]["body"] == {"filter": {"from": "all"}, "kind": "rl"}


def test_a_cut_with_no_holdout_still_answers(calls, monkeypatch):
    from whileai.simulations.ingest import platform

    monkeypatch.setattr(
        platform,
        "_call",
        lambda *a, **k: {"made": [{"role": "train", "datasetId": "ds_only", "rows": 9}]},
    )
    made = wai.cut(agent="a")

    assert made["train"]["datasetId"] == "ds_only"
    assert made["holdout"] is None


def test_cuts_reads_the_summary_without_making_anything(calls):
    summary = wai.cuts(agent="a", since="24h")

    assert calls[0]["method"] == "GET"
    assert calls[0]["path"] == "/traces/cuts?agent=a&from=24h"
    assert summary["rl"]["prompts"] == 30
    # The report says what it was read for, so the next line cuts the same traces.
    assert summary["filter"] == {"agent": "a", "from": "24h"}


def test_the_next_line_cuts_the_agent_the_report_was_read_for(calls):
    """Run the printed line as printed and it must not cut every agent."""
    out = wai.format_cuts(wai.cuts(agent="tool-choice"))

    assert 'next: wai.cut(agent="tool-choice", kind="rl")' in out


# ---- format_cuts: the summary as the sentence the traces page leads with ----

# The real payload of the walk that asked for this: 20 prompts, 3 in the band,
# too few to hold any back.
LIVE = {
    "runs": 60,
    "prompts": 20,
    "noReward": 0,
    "kinds": {"fail": {"prompts": 2}, "band": {"prompts": 3}, "solved": {"prompts": 15}},
    "rl": {"prompts": 3, "rows": 9, "support": 0.889, "passRate": 0.667, "train": 3, "holdout": 0},
    "sft": {"prompts": 18, "rows": 18, "train": 14, "holdout": 4},
    "more": {"prompts": 0},
}


def test_format_cuts_answers_in_three_lines_and_names_the_next_call():
    out = wai.format_cuts(LIVE, agent="getting-started")

    assert out.splitlines() == [
        "  3 prompts are worth training on",
        "  9 runs · 3 train · none held out",
        '  next: wai.cut(agent="getting-started", kind="rl")',
    ]
    # No RL jargon in what a researcher prints: `support` stays in the payload.
    assert "support" not in out


def test_format_cuts_falls_back_to_sft_and_counts_the_holdout():
    out = wai.format_cuts({**LIVE, "rl": {"prompts": 0}})

    assert "18 prompts have a run worth copying" in out
    assert "best run each · 14 train · 4 held out" in out
    assert 'next: wai.cut(kind="sft")' in out


def test_format_cuts_sends_ungraded_runs_to_the_call_that_grades_them():
    ungraded = {**LIVE, "prompts": 0, "noReward": 12, "rl": {}, "sft": {}}

    out = wai.format_cuts(ungraded)

    assert "12 runs have a prompt, none has a pass or fail" in out
    assert 'next: wai.send_score("<trace id>", 1.0)' in out


def test_format_cuts_never_ends_on_a_dead_stop():
    for report in [
        {},
        {**LIVE, "prompts": 0, "rl": {}, "sft": {}},
        {**LIVE, "rl": {}, "sft": {}, "more": {"prompts": 4}},
    ]:
        assert "next: " in wai.format_cuts(report)
