"""Zero Proof Labs platform client: dataset upload, listing, download.

Datasets generated locally with ``simulate()`` push to your Zero Proof Labs
account, where the optimization framework iterates on them. Auth is the
``zp_`` API key from https://www.zeroproofai.com/platform, read from the
``ZEROPROOF_API_KEY`` env var unless passed explicitly.

Stdlib only, matching the package's no-dependencies rule.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_API_URL = "https://wch04mgo2k.execute-api.us-east-1.amazonaws.com"


class PlatformError(RuntimeError):
    pass


def _api_url() -> str:
    return os.environ.get("ZEROPROOF_API_URL", DEFAULT_API_URL).rstrip("/")


def _key(api_key: str | None) -> str:
    key = api_key or os.environ.get("ZEROPROOF_API_KEY", "")
    if not key:
        raise PlatformError(
            "No API key. Pass api_key=... or set ZEROPROOF_API_KEY. "
            "Get one at https://www.zeroproofai.com/platform")
    return key


def _call(method: str, path: str, api_key: str | None, body: dict | None = None,
          *, raw_url: str | None = None, data: bytes | None = None,
          content_type: str | None = None, timeout: int = 120) -> dict | bytes:
    url = raw_url or (_api_url() + path)
    headers: dict[str, str] = {}
    if not raw_url:
        headers["X-Api-Key"] = _key(api_key)
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as err:
        detail = err.read().decode(errors="replace")[:400]
        try:
            detail = json.loads(detail).get("error", detail)
        except (ValueError, AttributeError):
            pass
        raise PlatformError(f"{method} {url.split('?')[0]} -> {err.code}: {detail}") from None
    except urllib.error.URLError as err:
        raise PlatformError(f"{method} {url.split('?')[0]} failed: {err.reason}") from None
    if raw_url:
        return payload
    return json.loads(payload) if payload else {}


def push_rows(rows: list[dict], name: str, *, api_key: str | None = None,
              parent: str | None = None) -> dict:
    """Upload rows as JSONL to your Zero Proof Labs account.

    Returns the registry entry, including ``datasetId``. Pass ``parent`` (a
    ``ds_...`` id) when this dataset is an iteration of an existing one, so
    lineage shows on the platform.
    """
    body: dict = {"name": name}
    if parent:
        body["parentDatasetId"] = parent
    created = _call("POST", "/datasets", api_key, body)
    payload = "".join(json.dumps(r, default=str) + "\n" for r in rows).encode()
    _call("PUT", "", api_key, raw_url=created["uploadUrl"], data=payload,
          content_type="application/jsonl")
    final = _call("POST", f"/datasets/{created['datasetId']}/finalize", api_key)
    return final


def push_file(path: str, name: str | None = None, *, api_key: str | None = None,
              parent: str | None = None) -> dict:
    """Upload an existing JSONL file. ``name`` defaults to the file name."""
    with open(path, "rb") as fh:
        payload = fh.read()
    stem = os.path.basename(path)
    if stem.endswith(".jsonl"):
        stem = stem[:-6]
    body: dict = {"name": name or stem}
    if parent:
        body["parentDatasetId"] = parent
    created = _call("POST", "/datasets", api_key, body)
    _call("PUT", "", api_key, raw_url=created["uploadUrl"], data=payload,
          content_type="application/jsonl")
    return _call("POST", f"/datasets/{created['datasetId']}/finalize", api_key)


def datasets(*, api_key: str | None = None) -> dict:
    """List your datasets plus storage used, newest first."""
    return _call("GET", "/datasets", api_key)


def pull(dataset_id: str, path: str | None = None, *,
         api_key: str | None = None) -> str | list[dict]:
    """Download a dataset. Writes JSONL to ``path`` and returns the path,
    or returns the parsed rows when ``path`` is omitted.

    A dataset is stored as one or more parts, and the grant lists every one
    of them. Datasets pushed with ``push_rows`` are a single part, which is
    why reading only ``downloadUrl`` looked correct for so long; a dataset
    filled by trace ingest is one part per trace, and that path returned the
    first row of a 60-row dataset without saying so.
    """
    grant = _call("GET", f"/datasets/{dataset_id}/download", api_key)
    # `downloadUrl` is parts[0], kept for older grants that predate the list.
    urls = [u for u in (grant.get("parts") or []) if isinstance(u, str)]
    if not urls:
        urls = [grant["downloadUrl"]]

    payloads = [_call("GET", "", api_key, raw_url=url) for url in urls]
    # Parts are whole JSONL objects but need not end in a newline, so joining
    # blind would weld the last row of one part onto the first of the next.
    payload = b"\n".join(p.strip() for p in payloads if p.strip())

    if path:
        with open(path, "wb") as fh:
            fh.write(payload + b"\n" if payload else payload)
        return path
    return [json.loads(line) for line in payload.decode().splitlines() if line.strip()]


def delete(dataset_id: str, *, api_key: str | None = None) -> dict:
    """Permanently delete a dataset from your account."""
    return _call("DELETE", f"/datasets/{dataset_id}", api_key)
