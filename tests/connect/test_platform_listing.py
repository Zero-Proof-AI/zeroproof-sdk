"""Listing and deleting what the account holds: datasets, runs, hosted models."""

from __future__ import annotations

import zeroproof.simulations as zps
from zeroproof.simulations import training
from zeroproof.simulations.ingest import platform


class Recorder:
    """Scripted gate: answers by (method, path), records every call."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def __call__(self, method, path, api_key=None, body=None, **kw):
        self.calls.append((method, path, body))
        return self.replies.get((method, path), {})


def test_datasets_lists_with_one_get_and_returns_the_reply_untouched(monkeypatch):
    rec = Recorder(
        {("GET", "/datasets"): {"datasets": [{"id": "ds_2"}, {"id": "ds_1"}], "storageBytes": 10}}
    )
    monkeypatch.setattr(platform, "_call", rec)
    out = zps.datasets(api_key="zp_x")
    assert [d["id"] for d in out["datasets"]] == ["ds_2", "ds_1"]
    assert out["storageBytes"] == 10
    assert rec.calls == [("GET", "/datasets", None)]


def test_delete_dataset_is_a_delete_on_the_dataset_route(monkeypatch):
    rec = Recorder({("DELETE", "/datasets/ds_1"): {"deleted": True}})
    monkeypatch.setattr(platform, "_call", rec)
    assert zps.delete_dataset("ds_1", api_key="zp_x") == {"deleted": True}
    assert rec.calls == [("DELETE", "/datasets/ds_1", None)]
    # the package name is an alias for platform.delete, not a second route
    assert zps.delete_dataset is platform.delete


def test_list_runs_unwraps_the_runs_key(monkeypatch):
    rec = Recorder({("GET", "/runs"): {"runs": [{"runId": "run_2"}, {"runId": "run_1"}]}})
    monkeypatch.setattr(training, "_call", rec)
    assert [r["runId"] for r in zps.list_runs(api_key="k")] == ["run_2", "run_1"]
    assert rec.calls == [("GET", "/runs", None)]


def test_get_run_returns_the_record_with_its_series(monkeypatch):
    rec = Recorder(
        {
            ("GET", "/runs/run_1"): {
                "runId": "run_1",
                "status": "done",
                "series": [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.5}],
            }
        }
    )
    monkeypatch.setattr(training, "_call", rec)
    run = zps.get_run("run_1", api_key="k")
    assert run["status"] == "done"
    assert [p["step"] for p in run["series"]] == [1, 2]
    assert rec.calls == [("GET", "/runs/run_1", None)]


def test_delete_run_is_a_delete_on_the_run_route(monkeypatch):
    rec = Recorder({("DELETE", "/runs/run_1"): {"runId": "run_1", "deleted": True}})
    monkeypatch.setattr(training, "_call", rec)
    assert zps.delete_run("run_1", api_key="k")["deleted"] is True
    assert rec.calls == [("DELETE", "/runs/run_1", None)]


def test_models_unwraps_the_models_key(monkeypatch):
    rec = Recorder(
        {
            ("GET", "/models"): {
                "models": [
                    {"name": "refund-v2", "endpoint": "https://serve.example/v1", "version": 2}
                ]
            }
        }
    )
    monkeypatch.setattr(training, "_call", rec)
    out = zps.models(api_key="k")
    assert out[0]["name"] == "refund-v2" and out[0]["version"] == 2
    assert out[0]["endpoint"] == "https://serve.example/v1"
    assert rec.calls == [("GET", "/models", None)]


def test_list_runs_and_models_are_empty_lists_when_the_gate_answers_oddly(monkeypatch):
    rec = Recorder({("GET", "/runs"): {}, ("GET", "/models"): ["not", "a", "dict"]})
    monkeypatch.setattr(training, "_call", rec)
    assert zps.list_runs(api_key="k") == []
    assert zps.models(api_key="k") == []
