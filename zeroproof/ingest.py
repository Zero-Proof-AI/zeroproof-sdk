"""
OTLP trace ingest for the Zero Proof token gate.

Push agent traces so they land as a dataset on your account. There is one
call and one response: POST the OTLP/HTTP JSON batch with your zp_ key, get
back a datasetId. No presigned URL, no finalize step, nothing to poll.

Batches append. Everything an exporter sends under one dataset name on one
UTC day belongs to one dataset, so a five-second flush interval does not turn
into thousands of datasets.

Two ways in, both dependency-light:

1. Live exporter. Point any OpenTelemetry SDK at the gate. This is the least
   invasive option: no Zero Proof code in your app at all, just environment.

       import os
       from zeroproof.ingest import otel_env
       os.environ.update(otel_env("zp_...", dataset="prod-refunds"))

2. Local file. Replay an OTLP batch you already have on disk (JSON, gzip or
   raw).

       from zeroproof.ingest import ingest_traces
       print(ingest_traces("zp_...", "traces.json")["datasetId"])

Auth is the X-Api-Key header, because exporters cannot carry a Clerk JWT.
"""

import gzip
import json
import os
from typing import Dict, Optional

import requests

# Where the gate lives. Override with ZEROPROOF_TRACE_URL if you point at a
# different deployment (the endpoint is account-specific).
_DEFAULT_TRACE_URL = "https://wch04mgo2k.execute-api.us-east-1.amazonaws.com"
_GZIP_MAGIC = b"\x1f\x8b"


class ZeroProofIngestError(Exception):
    """Raised when the gate rejects a trace batch."""


def _base(base_url: Optional[str] = None) -> str:
    url = base_url or os.environ.get("ZEROPROOF_TRACE_URL") or _DEFAULT_TRACE_URL
    return url.rstrip("/")


def _traces_endpoint(base_url: Optional[str] = None) -> str:
    return _base(base_url) + "/v1/traces"


def otel_env(api_key: str, dataset: str = "traces", base_url: Optional[str] = None) -> Dict[str, str]:
    """
    Environment for an OpenTelemetry OTLP/HTTP exporter.

    The exporter sends the batch body itself and forwards
    ``OTEL_EXPORTER_OTLP_HEADERS`` as request headers, so the key reaches the
    gate. ``http/json`` is required: the gate parses the OTLP JSON wire format
    and answers protobuf batches with a 415.
    """
    return {
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": _traces_endpoint(base_url),
        "OTEL_EXPORTER_OTLP_HEADERS": "x-api-key=" + api_key,
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
        # Resource attribute the gate reads to name the dataset.
        "OTEL_RESOURCE_ATTRIBUTES": "zeroproof.dataset=" + dataset,
    }


def send_traces(
    api_key: str,
    body: bytes,
    base_url: Optional[str] = None,
    timeout: int = 60,
) -> Dict:
    """POST one OTLP/HTTP JSON batch (raw or gzipped) and return the 202 body."""
    headers = {"X-Api-Key": api_key, "Content-Type": "application/json"}
    if body[:2] == _GZIP_MAGIC:
        headers["Content-Encoding"] = "gzip"
    res = requests.post(_traces_endpoint(base_url), data=body, headers=headers, timeout=timeout)
    if res.status_code >= 300:
        raise ZeroProofIngestError("ingest failed: HTTP %s %s" % (res.status_code, res.text[:400]))
    return res.json()


def ingest_traces(
    api_key: str,
    file: str,
    dataset: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict:
    """
    Push a local OTLP batch file end to end and return ``{datasetId, dataset,
    rows}``.

    ``dataset`` overrides the dataset name by setting the
    ``zeroproof.dataset`` resource attribute on every resourceSpan, which
    requires reading the batch; leave it unset to send the bytes untouched.
    """
    with open(file, "rb") as fh:
        body = fh.read()
    if not body:
        raise ZeroProofIngestError("empty file: " + file)

    if dataset is not None:
        raw = gzip.decompress(body) if body[:2] == _GZIP_MAGIC else body
        batch = json.loads(raw.decode("utf-8"))
        for resource_span in batch.get("resourceSpans", []):
            resource = resource_span.setdefault("resource", {})
            attributes = [
                a for a in resource.get("attributes", []) if a.get("key") != "zeroproof.dataset"
            ]
            attributes.append({"key": "zeroproof.dataset", "value": {"stringValue": dataset}})
            resource["attributes"] = attributes
        body = json.dumps(batch).encode("utf-8")

    return send_traces(api_key, body, base_url=base_url)


def list_traces(api_key: str, base_url: Optional[str] = None, timeout: int = 30) -> Dict:
    """
    What this key's account has ingested: one entry per dataset name per day,
    with row counts and sizes, plus the account totals.

        for t in list_traces("zp_...")["traces"]:
            print(t["name"], t["rows"], t["sizeBytes"])
    """
    res = requests.get(
        _base(base_url) + "/traces",
        headers={"X-Api-Key": api_key},
        timeout=timeout,
    )
    if res.status_code >= 300:
        raise ZeroProofIngestError("list failed: HTTP %s %s" % (res.status_code, res.text[:400]))
    return res.json()
