"""Zero Proof Labs platform client: dataset upload, listing, download.

Datasets generated locally with ``simulate()`` push to your Zero Proof Labs
account, where the optimization framework iterates on them. Runtime access
uses a short-lived delegated credential (``zp_dc_...``), which is issued by a
valid Clerk session token and then passed as the X-Api-Key on protected routes.

The legacy ``ZEROPROOF_API_KEY`` env var still works for compatibility, but the
preferred runtime credential is ``ZEROPROOF_DELEGATED_CREDENTIAL``.

Stdlib only, matching the package's no-dependencies rule.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

#: Overridable with ``ZEROPROOF_API_URL``, which is what a self-hosted gate or
#: a staging one uses. The default is the production token gate behind the
#: ZeroProof AWS account, and the SDK prefers delegated credentials over static
#: keys at runtime.
DEFAULT_API_URL = "https://api.zeroproofai.com"


class PlatformError(RuntimeError):
    pass


def _api_url() -> str:
    return os.environ.get("ZEROPROOF_API_URL", DEFAULT_API_URL).rstrip("/")


def _key(api_key: str | None) -> str:
    key = api_key or os.environ.get("ZEROPROOF_DELEGATED_CREDENTIAL") or os.environ.get("ZEROPROOF_API_KEY", "")
    if not key:
        raise PlatformError(
            "No delegated credential. Pass api_key=... or set "
            "ZEROPROOF_DELEGATED_CREDENTIAL (preferred) / ZEROPROOF_API_KEY "
            "for compatibility.")
    return key


def _call(method: str, path: str, api_key: str | None, body: dict | None = None,
          *, raw_url: str | None = None, data: bytes | None = None,
          content_type: str | None = None, timeout: int = 120,
          auth_token: str | None = None, require_api_key: bool = False) -> dict | bytes:
    url = raw_url or (_api_url() + path)
    headers: dict[str, str] = {}
    if auth_token is not None:
        headers["Authorization"] = f"Bearer {auth_token}"
    elif require_api_key or not raw_url:
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


def issue_delegated_credential(clerk_token: str | None, *, ttl_seconds: int = 3600,
                              name: str = "sdk-default",
                              timeout: int = 120) -> dict:
    """Create a short-lived delegated credential for SDK or backend use.

    ``clerk_token`` must be a valid Clerk session token or other authenticated
    backend token. This helper sends that token as a bearer token to the auth
    endpoint to mint the delegated credential.
    """
    if not clerk_token:
        raise PlatformError("A valid Clerk session token is required to mint a delegated credential.")
    body = {"name": name, "ttlSeconds": int(ttl_seconds)}
    return _call("POST", "/auth/issue-credential", None, body, timeout=timeout,
                 auth_token=clerk_token)


def refresh_delegated_credential(clerk_token: str | None, credential: str, *,
                               ttl_seconds: int = 3600, timeout: int = 120) -> dict:
    """Refresh a delegated credential before it expires."""
    if not clerk_token:
        raise PlatformError("A valid Clerk session token is required to refresh a delegated credential.")
    if not credential:
        raise PlatformError("Pass the current delegated credential to refresh it.")
    body = {"credential": credential, "ttlSeconds": int(ttl_seconds)}
    return _call("POST", "/auth/refresh-credential", None, body, timeout=timeout,
                 auth_token=clerk_token)


def revoke_delegated_credential(clerk_token: str | None, credential: str,
                               *, timeout: int = 120) -> dict:
    """Revoke a delegated credential for the authenticated user."""
    if not clerk_token:
        raise PlatformError("A valid Clerk session token is required to revoke a delegated credential.")
    if not credential:
        raise PlatformError("Pass the delegated credential to revoke it.")
    return _call("POST", "/auth/revoke-credential", None, {"credential": credential},
                 timeout=timeout, auth_token=clerk_token)


DEFAULT_STUDIO_URL = (
    "https://zeroproofai--zeroproof-studio-api-serve.modal.run")
_STUDIO_MODES = ("explore", "sft", "rl", "adaptive")
_STUDIO_MAX_ROWS = 20_000


def _studio_url() -> str:
    return os.environ.get("ZEROPROOF_STUDIO_URL",
                          DEFAULT_STUDIO_URL).rstrip("/")


def push_to_studio(rows: list[dict], agent: str, mode: str, *,
                   tags: list[str] | None = None,
                   filename: str | None = None,
                   api_key: str | None = None) -> dict:
    """Import rows into the studio runs store the platform UI reads.

    ``push_rows`` lands in the datasets registry; the platform's
    datasets page reads the studio's runs store instead, so rows pushed
    there never appear in the UI. This posts to the studio's import
    endpoint, which grades rows against the agent's declared tools and
    writes into the same store the page lists.

    ``agent`` must exist in the STUDIO agent registry (separate from
    trace agents; an unregistered name is rejected by the studio).
    ``mode`` labels the batch (one of explore/sft/rl/adaptive) and is
    required: the store would otherwise silently label everything "rl".
    """
    if mode not in _STUDIO_MODES:
        raise PlatformError(
            f"mode= must be one of {'/'.join(_STUDIO_MODES)}")
    if len(rows) > _STUDIO_MAX_ROWS:
        raise PlatformError(
            f"studio import caps at {_STUDIO_MAX_ROWS} rows; "
            f"got {len(rows)} - split the push")
    body: dict = {"agent": agent, "mode": mode, "rows": list(rows)}
    if tags:
        body["tags"] = list(tags)
    if filename:
        body["filename"] = filename
    url = _studio_url() + "/api/import"
    data = json.dumps(body, default=str).encode()
    headers = {"X-Api-Key": _key(api_key),
               "Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, method="POST",
                                     headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = response.read()
    except urllib.error.HTTPError as err:
        detail = err.read().decode(errors="replace")[:400]
        try:
            detail = json.loads(detail).get("error", detail)
        except (ValueError, AttributeError):
            pass
        hint = (" (is the agent registered in the studio? the studio "
                "registry is separate from trace agents)"
                if err.code in (400, 404) else "")
        raise PlatformError(
            f"POST {url} -> {err.code}: {detail}{hint}") from None
    except urllib.error.URLError as err:
        raise PlatformError(f"POST {url} failed: {err.reason}") from None
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
