"""push_to_studio: the bridge from SDK output to the runs store the
platform datasets page actually reads. Wire-level: capture the request,
assert the contract (URL, auth header, body shape, explicit mode)."""
import io
import json
import urllib.request

import pytest

from zeroproof_simulations.platform import PlatformError, push_to_studio
from studio.api import data as studio_data

ROWS = [{"prompt": "log a coffee expense", "final_text": "Logged."}]


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_posts_import_contract(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data)
        captured["method"] = request.get_method()
        return _Resp(b'{"run_id": "run_1", "rows": 1}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("ZEROPROOF_API_KEY", "zp_test_key")
    out = push_to_studio(ROWS, "budget-buddy", "explore",
                         tags=["from-sdk"], filename="run.jsonl")
    assert out == {"run_id": "run_1", "rows": 1}
    assert captured["url"].endswith("/api/import")
    assert captured["method"] == "POST"
    assert captured["headers"].get("X-api-key") == "zp_test_key"
    body = captured["body"]
    assert body["agent"] == "budget-buddy"
    assert body["mode"] == "explore"
    assert body["rows"] == ROWS
    assert body["tags"] == ["from-sdk"]
    assert body["filename"] == "run.jsonl"


def test_mode_is_required_and_validated(monkeypatch):
    monkeypatch.setenv("ZEROPROOF_API_KEY", "zp_test_key")
    with pytest.raises(TypeError):
        push_to_studio(ROWS, "budget-buddy")
    with pytest.raises(PlatformError, match="mode="):
        push_to_studio(ROWS, "budget-buddy", "training")


def test_row_cap_and_unregistered_agent_hint(monkeypatch):
    monkeypatch.setenv("ZEROPROOF_API_KEY", "zp_test_key")
    with pytest.raises(PlatformError, match="caps at 20000"):
        push_to_studio([{}] * 20001, "budget-buddy", "explore")

    def not_found(request, timeout=0):
        raise urllib.error.HTTPError(
            request.full_url, 404, "nf", {},
            io.BytesIO(b'{"error": "unknown agent"}'))

    monkeypatch.setattr(urllib.request, "urlopen", not_found)
    with pytest.raises(PlatformError, match="studio registry is separate"):
        push_to_studio(ROWS, "not-a-studio-agent", "explore")


def test_imported_sdk_run_lists_opens_and_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_data, "AGENTS", tmp_path / "agents")
    rows = [{
        "prompt": "check pull request 42",
        "steps": [{
            "tool": "get_pr",
            "arguments": {"number": 42},
            "result": {"status": "ok", "state": "open"},
        }],
        "final_text": "Pull request 42 is open.",
        "reward": 1,
        "reason": "The response matches the tool result.",
        "judge_status": "ok",
        "judge_name": "customer-policy",
        "lineage": {"scoring_run_id": "score_1"},
    }]

    imported = studio_data.import_data({
        "agent": "github",
        "mode": "explore",
        "rows": rows,
        "filename": "sdk-run.jsonl",
    })
    listed = studio_data.list_runs("github")
    opened = studio_data.get_run(imported["id"], "github")
    downloaded = studio_data.download_run(imported["id"])
    raw = [json.loads(line) for line in downloaded["jsonl"].splitlines()]

    assert listed["runs"][0]["id"] == imported["id"]
    assert listed["runs"][0]["mode"] == "explore"
    assert opened["rows"][0]["reward"] == 1
    assert opened["rows"][0]["reason"] == rows[0]["reason"]
    assert opened["rows"][0]["n_tools"] == 1
    assert raw[0]["judge_name"] == "customer-policy"
    assert raw[0]["judge_status"] == "ok"
    assert raw[0]["lineage"] == {"scoring_run_id": "score_1"}
