"""whileai.ingest: the exporter env, the nanosecond guard, gzip, the dataset override."""

from __future__ import annotations

import gzip
import json

import pytest

from whileai import ingest


class Reply:
    def __init__(self, status=202, body=None, text=""):
        self.status_code = status
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


def _batch(start="1700000000000000000"):
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": "app"}}]
                },
                "scopeSpans": [{"spans": [{"name": "llm.call", "startTimeUnixNano": start}]}],
            }
        ]
    }


def _capture_post(monkeypatch, reply):
    seen = {}

    def post(url, data=None, headers=None, timeout=None):
        seen.update(url=url, data=data, headers=headers, timeout=timeout)
        return reply

    monkeypatch.setattr(ingest.requests, "post", post)
    return seen


def test_otel_env_points_the_exporter_at_the_gate_with_the_key_as_a_header(monkeypatch):
    monkeypatch.delenv("WHILEAI_TRACE_URL", raising=False)
    env = ingest.otel_env("zp_abc", dataset="prod-refunds")
    assert env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == "https://api.zeroproofai.com/v1/traces"
    assert env["OTEL_EXPORTER_OTLP_HEADERS"] == "x-api-key=zp_abc"
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/json"
    # The gate names the dataset from zeroproof.dataset only: without it the
    # batch lands in a dataset called `traces` and the 202 says so.
    assert "zeroproof.dataset=prod-refunds" in env["OTEL_RESOURCE_ATTRIBUTES"].split(",")


def test_otel_env_honors_the_base_url_override_and_the_env_var(monkeypatch):
    env = ingest.otel_env("zp_abc", base_url="http://localhost:8080/")
    assert env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == "http://localhost:8080/v1/traces"
    monkeypatch.setenv("WHILEAI_TRACE_URL", "http://gate.test")
    assert ingest.otel_env("zp_abc")["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == (
        "http://gate.test/v1/traces"
    )


def test_send_traces_posts_the_key_and_returns_the_gate_body(monkeypatch):
    seen = _capture_post(monkeypatch, Reply(202, {"datasetId": "ds_1", "rows": 1}))
    body = json.dumps(_batch()).encode()
    out = ingest.send_traces("zp_abc", body, base_url="http://gate.test")
    assert out == {"datasetId": "ds_1", "rows": 1}
    assert seen["url"] == "http://gate.test/v1/traces"
    assert seen["headers"]["X-Api-Key"] == "zp_abc"
    assert seen["headers"]["Content-Type"] == "application/json"
    assert "Content-Encoding" not in seen["headers"]
    assert seen["data"] is body


def test_gzipped_batches_go_up_as_is_with_the_encoding_header(monkeypatch):
    seen = _capture_post(monkeypatch, Reply(202, {"datasetId": "ds_1"}))
    body = gzip.compress(json.dumps(_batch()).encode())
    ingest.send_traces("zp_abc", body)
    assert seen["headers"]["Content-Encoding"] == "gzip"
    assert seen["data"] is body


def test_millisecond_timestamps_are_refused_before_anything_is_uploaded(monkeypatch):
    def post(*a, **k):
        raise AssertionError("a batch with bad units must never reach the gate")

    monkeypatch.setattr(ingest.requests, "post", post)
    body = json.dumps(_batch(start="1700000000000")).encode()
    with pytest.raises(ingest.WhileIngestError, match="not nanoseconds"):
        ingest.send_traces("zp_abc", body)


def test_gate_rejection_becomes_an_ingest_error_with_the_status(monkeypatch):
    monkeypatch.setattr(
        ingest.requests, "post", lambda *a, **k: Reply(415, text="protobuf not accepted")
    )
    with pytest.raises(ingest.WhileIngestError, match="HTTP 415 protobuf not accepted"):
        ingest.send_traces("zp_abc", json.dumps(_batch()).encode())


def test_ingest_traces_renames_the_dataset_on_every_resource(tmp_path, monkeypatch):
    seen = _capture_post(monkeypatch, Reply(202, {"datasetId": "ds_9"}))
    batch = _batch()
    batch["resourceSpans"][0]["resource"]["attributes"].append(
        {"key": "whileai.dataset", "value": {"stringValue": "old-name"}}
    )
    path = tmp_path / "traces.json.gz"
    path.write_bytes(gzip.compress(json.dumps(batch).encode()))
    out = ingest.ingest_traces("zp_abc", str(path), dataset="new-name")
    assert out["datasetId"] == "ds_9"
    sent = json.loads(seen["data"])
    attrs = sent["resourceSpans"][0]["resource"]["attributes"]
    for key in ("zeroproof.dataset", "whileai.dataset"):
        names = [a["value"]["stringValue"] for a in attrs if a["key"] == key]
        assert names == ["new-name"], f"{key}: the old name is replaced, not doubled"
    assert any(a["key"] == "service.name" for a in attrs), "other attributes survive"


def test_ingest_traces_sends_bytes_untouched_without_a_dataset(tmp_path, monkeypatch):
    seen = _capture_post(monkeypatch, Reply(202, {"datasetId": "ds_9"}))
    raw = json.dumps(_batch()).encode()
    path = tmp_path / "traces.json"
    path.write_bytes(raw)
    ingest.ingest_traces("zp_abc", str(path))
    assert seen["data"] == raw


def test_ingest_traces_refuses_an_empty_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.requests, "post", lambda *a, **k: Reply(202, {}))
    path = tmp_path / "empty.json"
    path.write_bytes(b"")
    with pytest.raises(ingest.WhileIngestError, match="empty file"):
        ingest.ingest_traces("zp_abc", str(path))


def test_list_traces_reads_with_the_key_and_surfaces_a_rejection(monkeypatch):
    seen = {}

    def get(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers)
        return Reply(200, {"traces": [{"name": "prod", "rows": 3}]})

    monkeypatch.setattr(ingest.requests, "get", get)
    out = ingest.list_traces("zp_abc", base_url="http://gate.test")
    assert out["traces"][0]["rows"] == 3
    assert seen["url"] == "http://gate.test/traces"
    assert seen["headers"] == {"X-Api-Key": "zp_abc"}
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: Reply(401, text="bad key"))
    with pytest.raises(ingest.WhileIngestError, match="HTTP 401"):
        ingest.list_traces("zp_abc")
