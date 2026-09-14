"""zps.cut / zps.cuts: the platform's "Make training data" button as one line."""

from __future__ import annotations

import pytest

import zeroproof.simulations as zps


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

    from zeroproof.simulations.ingest import platform

    monkeypatch.setattr(platform, "_call", fake_call)
    return seen


def test_cut_posts_the_filter_and_names_both_sets(calls):
    made = zps.cut(agent="My-Agent", kind="rl")

    assert calls[0]["method"] == "POST"
    assert calls[0]["path"] == "/datasets/cut"
    assert calls[0]["body"] == {"filter": {"agent": "my-agent", "from": "all"}, "kind": "rl"}
    # The two ids are what the next line needs, without digging through `made`.
    assert made["train"]["datasetId"] == "ds_train"
    assert made["holdout"]["datasetId"] == "ds_hold"


def test_cut_passes_window_band_and_trace_filters(calls):
    zps.cut(agent="a", kind="sft", since="7d", band=(0.1, 0.9), holdout=0.25, model="gpt-4.1")

    body = calls[0]["body"]
    assert body["kind"] == "sft"
    assert body["filter"] == {"model": "gpt-4.1", "agent": "a", "from": "7d"}
    assert (body["lo"], body["hi"]) == (0.1, 0.9)
    assert body["holdout"] == 0.25


def test_cut_without_an_agent_covers_every_trace(calls):
    zps.cut()

    assert calls[0]["body"] == {"filter": {"from": "all"}, "kind": "rl"}


def test_a_cut_with_no_holdout_still_answers(calls, monkeypatch):
    from zeroproof.simulations.ingest import platform

    monkeypatch.setattr(
        platform,
        "_call",
        lambda *a, **k: {"made": [{"role": "train", "datasetId": "ds_only", "rows": 9}]},
    )
    made = zps.cut(agent="a")

    assert made["train"]["datasetId"] == "ds_only"
    assert made["holdout"] is None


def test_cuts_reads_the_summary_without_making_anything(calls):
    summary = zps.cuts(agent="a", since="24h")

    assert calls[0]["method"] == "GET"
    assert calls[0]["path"] == "/traces/cuts?agent=a&from=24h"
    assert summary["rl"]["prompts"] == 30
