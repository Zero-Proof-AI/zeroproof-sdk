"""
OTLP trace ingest for the While token gate.

Push agent traces so they land as a dataset on your account. There is one
call and one response: POST the OTLP/HTTP JSON batch with your zp_ key, get
back a datasetId. No presigned URL, no finalize step, nothing to poll.

Batches append. Everything an exporter sends under one dataset name on one
UTC day belongs to one dataset, so a five-second flush interval does not turn
into thousands of datasets.

Two ways in, both dependency-light:

1. Live exporter. Point any OpenTelemetry SDK at the gate. This is the least
   invasive option: no While code in your app at all, just environment.

       import os
       from whileai.ingest import otel_env
       os.environ.update(otel_env("zp_...", dataset="prod-refunds"))

2. Local file. Replay an OTLP batch you already have on disk (JSON, gzip or
   raw).

       from whileai.ingest import ingest_traces
       print(ingest_traces("zp_...", "traces.json")["datasetId"])

Auth is the X-Api-Key header, because exporters cannot carry a Clerk JWT.
"""

import gzip
import json

import requests

from whileai._env import getenv

# Where the gate lives. Override with WHILEAI_TRACE_URL to point at a
# different deployment.
_DEFAULT_TRACE_URL = "https://api.zeroproofai.com"
_GZIP_MAGIC = b"\x1f\x8b"

# The resource attribute that names the dataset. The gate reads
# `zeroproof.dataset` and nothing else, so sending only the whileai spelling
# lands every batch in a dataset called `traces` whatever you asked for, with
# a 202 that says so too late to notice. Both are written: the second costs
# one attribute and means the rename needs no release.
_DATASET_KEYS = ("zeroproof.dataset", "whileai.dataset")


class WhileIngestError(Exception):
    """Raised when the gate rejects a trace batch."""


def _base(base_url: str | None = None) -> str:
    url = base_url or getenv("TRACE_URL") or _DEFAULT_TRACE_URL
    return url.rstrip("/")


def _traces_endpoint(base_url: str | None = None) -> str:
    return _base(base_url) + "/v1/traces"


def otel_env(api_key: str, dataset: str = "traces", base_url: str | None = None) -> dict[str, str]:
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
        "OTEL_RESOURCE_ATTRIBUTES": ",".join(k + "=" + dataset for k in _DATASET_KEYS),
    }


def _check_nanos(body: bytes) -> None:
    """Reject span times that are not nanoseconds.

    The store divides by 1e6 without a unit guard, so a batch sent in
    milliseconds or seconds lands near 1970 and disappears from every bounded
    time window: the upload succeeds and the traces are simply never seen.
    Better to fail here, where the sender can still fix it.
    """
    if body[:2] == _GZIP_MAGIC:
        return
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return
    for resource in payload.get("resourceSpans") or []:
        for scope in resource.get("scopeSpans") or []:
            for span in scope.get("spans") or []:
                raw = span.get("startTimeUnixNano")
                if raw in (None, "", 0, "0"):
                    continue
                try:
                    value = int(raw)
                except (TypeError, ValueError):
                    continue
                # 1e15 ns is 1970-01-12; any real timestamp is far above it
                if 0 < value < 10**15:
                    raise WhileIngestError(
                        f"span {span.get('name') or 'unnamed'!r} has "
                        f"startTimeUnixNano={value}, which is not nanoseconds. "
                        "Multiply by 1e6 for milliseconds or 1e9 for seconds; "
                        "as sent, these traces would be stored near 1970 and "
                        "hidden from every time window."
                    )


def send_traces(
    api_key: str,
    body: bytes,
    base_url: str | None = None,
    timeout: int = 60,
) -> dict:
    """POST one OTLP/HTTP JSON batch (raw or gzipped) and return the 202 body."""
    _check_nanos(body)
    headers = {"X-Api-Key": api_key, "Content-Type": "application/json"}
    if body[:2] == _GZIP_MAGIC:
        headers["Content-Encoding"] = "gzip"
    res = requests.post(_traces_endpoint(base_url), data=body, headers=headers, timeout=timeout)
    if res.status_code >= 300:
        raise WhileIngestError(f"ingest failed: HTTP {res.status_code} {res.text[:400]}")
    return res.json()


def ingest_traces(
    api_key: str,
    file: str,
    dataset: str | None = None,
    base_url: str | None = None,
) -> dict:
    """
    Push a local OTLP batch file end to end and return ``{datasetId, dataset,
    rows}``.

    ``dataset`` overrides the dataset name by setting the dataset resource
    attribute on every resourceSpan, which requires reading the batch; leave
    it unset to send the bytes untouched.
    """
    with open(file, "rb") as fh:
        body = fh.read()
    if not body:
        raise WhileIngestError("empty file: " + file)

    if dataset is not None:
        raw = gzip.decompress(body) if body[:2] == _GZIP_MAGIC else body
        batch = json.loads(raw.decode("utf-8"))
        for resource_span in batch.get("resourceSpans", []):
            resource = resource_span.setdefault("resource", {})
            attributes = [
                a for a in resource.get("attributes", []) if a.get("key") not in _DATASET_KEYS
            ]
            attributes += [{"key": k, "value": {"stringValue": dataset}} for k in _DATASET_KEYS]
            resource["attributes"] = attributes
        body = json.dumps(batch).encode("utf-8")

    return send_traces(api_key, body, base_url=base_url)


def list_traces(api_key: str | None = None, base_url: str | None = None, timeout: int = 30) -> dict:
    """
    What this key's account has ingested: one entry per dataset name per day,
    with row counts and sizes, plus the account totals.

        for t in list_traces()["traces"]:
            print(t["name"], t["rows"], t["sizeBytes"])

    ``api_key`` resolves like every other platform call: the argument, then
    ``WHILEAI_API_KEY``, then the key ``whileai login`` saved.
    """
    from .auth import resolve_api_key

    key = resolve_api_key(api_key)
    if not key:
        raise WhileIngestError(
            "no API key: pass api_key=, set WHILEAI_API_KEY, or run `whileai login`"
        )
    res = requests.get(
        _base(base_url) + "/traces",
        headers={"X-Api-Key": key},
        timeout=timeout,
    )
    if res.status_code >= 300:
        raise WhileIngestError(f"list failed: HTTP {res.status_code} {res.text[:400]}")
    return res.json()


# The name this exception had before the package was renamed.
ZeroProofIngestError = WhileIngestError
