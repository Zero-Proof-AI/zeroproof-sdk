"""Live studio integration test for daily quota enforcement on /api/simulate."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import pytest


_studio_url = os.environ.get("TEST_STUDIO_URL") or os.environ.get("ZEROPROOF_STUDIO_URL")
if not _studio_url:
    pytest.skip(
        "set TEST_STUDIO_URL or ZEROPROOF_STUDIO_URL to run studio integration tests",
        allow_module_level=True,
    )
STUDIO_URL = _studio_url.rstrip("/")
OVER_QUOTA_KEY = (
    os.environ.get("TEST_STUDIO_OVER_QUOTA_KEY")
    or os.environ.get("TEST_OVER_QUOTA_DELEGATED_CREDENTIAL")
)
UNDER_QUOTA_KEY = (
    os.environ.get("TEST_STUDIO_UNDER_QUOTA_KEY")
    or os.environ.get("TEST_UNDER_QUOTA_DELEGATED_CREDENTIAL")
)

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
KEYS_TABLE = os.environ.get("TABLE_API_KEYS", "zeroproof-api-keys")
USAGE_TABLE = os.environ.get("TABLE_DAILY_USAGE", "zeroproof-daily-usage")


def _api_call(method: str, path: str, api_key: str, body: dict | None = None) -> tuple[int, dict]:
    payload = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        STUDIO_URL + path,
        data=payload,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            return int(response.status), json.loads(raw or b"{}")
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            data = {"error": raw.decode(errors="replace")}
        return int(err.code), data


def _simulate_request(api_key: str) -> tuple[int, dict]:
    body = {
        "agent": "github",
        "mode": "explore",
        "stop": "rows",
        "budget": 1,
    }
    return _api_call("POST", "/api/simulate", api_key, body)


def _wait_for_job(api_key: str, job_id: str, *, timeout_s: float = 120.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status, payload = _api_call("GET", f"/api/job?id={job_id}", api_key)
        if status != 200:
            raise AssertionError(f"/api/job returned status={status} payload={payload}")
        state = str(payload.get("status") or "").lower()
        if state in {"done", "error"}:
            return payload
        time.sleep(1.0)
    raise AssertionError(f"Timed out waiting for job {job_id}")


def _usage_for_key(api_key: str) -> tuple[int, int]:
    try:
        import boto3
    except ModuleNotFoundError:
        pytest.skip("boto3 is required to verify live usage counters")

    db = boto3.client("dynamodb", region_name=AWS_REGION)
    item = db.get_item(TableName=KEYS_TABLE, Key={"apiKey": {"S": api_key}}).get("Item")
    if not item or not item.get("active", {}).get("BOOL"):
        pytest.skip(f"api key {api_key[:12]}... is not active in {KEYS_TABLE}")
    user_id = item.get("userId", {}).get("S")
    if not user_id:
        pytest.skip(f"api key {api_key[:12]}... has no userId in {KEYS_TABLE}")

    today = time.strftime("%Y-%m-%d", time.gmtime())
    usage = db.get_item(
        TableName=USAGE_TABLE,
        Key={"apiKey": {"S": f"user#{user_id}"}, "date": {"S": today}},
    ).get("Item") or {}
    return (
        int(usage.get("inputTokens", {}).get("N", "0")),
        int(usage.get("outputTokens", {}).get("N", "0")),
    )


def test_live_simulate_rejects_over_quota_credential():
    if not OVER_QUOTA_KEY:
        pytest.skip(
            "set TEST_STUDIO_OVER_QUOTA_KEY or "
            "TEST_OVER_QUOTA_DELEGATED_CREDENTIAL to run over-quota check"
        )
    status, payload = _simulate_request(str(OVER_QUOTA_KEY))

    assert status == 429
    assert "daily account quota exceeded" in str(payload.get("error") or "").lower()


def test_live_simulate_increases_daily_usage_when_under_quota():
    if not UNDER_QUOTA_KEY:
        pytest.skip(
            "set TEST_STUDIO_UNDER_QUOTA_KEY or "
            "TEST_UNDER_QUOTA_DELEGATED_CREDENTIAL to run usage-growth check"
        )

    before_in, before_out = _usage_for_key(str(UNDER_QUOTA_KEY))

    status, payload = _simulate_request(str(UNDER_QUOTA_KEY))
    assert status == 200, payload
    job_id = str(payload.get("id") or "")
    assert job_id, payload

    job = _wait_for_job(str(UNDER_QUOTA_KEY), job_id)
    assert str(job.get("status") or "").lower() == "done", job

    after_in, after_out = _usage_for_key(str(UNDER_QUOTA_KEY))
    increased = after_in > before_in or after_out > before_out
    assert increased, {
        "before_input": before_in,
        "before_output": before_out,
        "after_input": after_in,
        "after_output": after_out,
        "job_id": job_id,
    }