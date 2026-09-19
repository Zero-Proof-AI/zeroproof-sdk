"""Purging an agent and empty datasets: what gets deleted, and dry runs that delete nothing."""

from __future__ import annotations

import pytest

import whileai.simulations as wai
from whileai.simulations.ingest import platform


class Fake:
    def __init__(self):
        self.deleted = []
        self.traces = {"demo-agent": ["t1", "t2", "t3"]}
        self.datasets = [
            {"datasetId": "ds_a", "agent": "demo-agent", "sizeBytes": 10, "rows": 5},
            {"datasetId": "ds_b", "agent": "keep", "sizeBytes": 10, "rows": 5},
            {"datasetId": "ds_empty", "agent": None, "sizeBytes": 0, "rows": None},
            {"datasetId": "ds_smoke", "agent": "keep", "sizeBytes": 30, "rows": 2},
        ]

    def __call__(self, method, path, api_key, body=None, **kw):
        if method == "DELETE":
            self.deleted.append(path)
            return {"deleted": True}
        if path.startswith("/traces?agent="):
            slug = path.split("agent=")[1].split("&")[0]
            return {"traces": [{"traceId": t} for t in self.traces.get(slug, [])], "pages": 1}
        if path == "/datasets":
            return {"datasets": self.datasets}
        raise AssertionError(path)


def test_purge_agent_removes_traces_datasets_and_record(monkeypatch):
    fake = Fake()
    monkeypatch.setattr(platform, "_call", fake)
    assert wai.purge_agent("demo-agent", dry_run=True) == {
        "agent": "demo-agent",
        "traces": 3,
        "datasets": 1,
        "deleted": False,
    }
    assert fake.deleted == []
    out = wai.purge_agent("Demo-Agent")
    assert out["deleted"] is True
    assert fake.deleted == [
        "/traces/t1",
        "/traces/t2",
        "/traces/t3",
        "/datasets/ds_a",
        "/agents/demo-agent",
    ]
    with pytest.raises(ValueError):
        wai.purge_agent("")


def test_delete_empty_datasets_and_smoke_sets(monkeypatch):
    fake = Fake()
    monkeypatch.setattr(platform, "_call", fake)
    assert wai.delete_empty_datasets(dry_run=True) == {"datasets": ["ds_empty"], "deleted": False}
    assert wai.delete_empty_datasets(max_rows=2, dry_run=True)["datasets"] == [
        "ds_empty",
        "ds_smoke",
    ]
    wai.delete_empty_datasets()
    assert fake.deleted == ["/datasets/ds_empty"]
