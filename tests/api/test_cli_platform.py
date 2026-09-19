"""The platform verbs of the `whileai` command, against a fake API."""

from __future__ import annotations

import json

import pytest

from whileai import cli, platform

DASH = {
    "agent": {
        "id": "refund-bot",
        "name": "refund-bot",
        "model": "Qwen/Qwen3-4B",
        "serving": "v3",
        "candidate": "v4",
    },
    "behavior": {"name": "refunds", "testVersion": "v2", "n": 240},
    "behaviors": ["refunds", "length"],
    "versions": [{"v": "v3", "score": 78, "ci": 2.8}, {"v": "v4", "score": 83, "ci": 2.7}],
    "deltas": [{"name": "refunds", "delta": 5, "target": True}],
    "verdict": {
        "candidate": "v4",
        "serving": "v3",
        "delta": 5,
        "excludesZero": True,
        "regressions": 0,
    },
}


@pytest.fixture
def fake(monkeypatch):
    calls: list[tuple[str, str, object]] = []

    def request(method, path, *, api_key, body=None, timeout=30.0):
        calls.append((method, path, body))
        assert api_key == "zp_test"
        if path == "/agents":
            return {
                "agents": [
                    {
                        "id": "refund-bot",
                        "name": "refund-bot",
                        "model": "Qwen/Qwen3-4B",
                        "serving": "v3",
                    }
                ]
            }
        if path.startswith("/agents/refund-bot/dashboard"):
            return DASH
        if path == "/agents/refund-bot/behaviors":
            return {
                "behaviors": [
                    {"name": "refunds", "testVersion": "v2", "n": 240},
                    {"name": "length"},
                ]
            }
        if path == "/runs?agent=refund-bot":
            return {
                "runs": [
                    {
                        "id": "r4",
                        "version": "v4",
                        "method": "GRPO",
                        "status": "evaluated",
                        "targets": ["refunds"],
                        "createdAt": "2026-09-18T10:00:00Z",
                    }
                ]
            }
        if path == "/agents/refund-bot/promote":
            return {"id": "refund-bot", "serving": body["version"]}
        if path == "/keys":
            return {
                "keys": [
                    {
                        "name": "laptop",
                        "key": "zp_1234…89ab",
                        "tier": "full",
                        "createdAt": "2026-09-18T00:00:00Z",
                    }
                ],
                "limit": 5,
            }
        if path == "/live":
            return {"rows": 1}
        raise AssertionError(f"unexpected {method} {path}")

    monkeypatch.setattr(platform, "_request", request)
    monkeypatch.setenv("WHILEAI_API_KEY", "zp_test")
    return calls


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_agents_table_and_json(fake, capsys):
    code, out, _ = run(capsys, "agents")
    assert code == 0 and "refund-bot" in out and "serving v3" in out
    code, out, _ = run(capsys, "agents", "--json")
    assert json.loads(out)[0]["id"] == "refund-bot"


def test_agent_shows_behaviors_and_verdict(fake, capsys):
    code, out, _ = run(capsys, "agent", "refund-bot")
    assert code == 0
    assert "refunds" in out and "test v2" in out
    assert "v4 beats v3 by 5" in out


def test_runs_and_verdict(fake, capsys):
    code, out, _ = run(capsys, "runs", "refund-bot")
    assert code == 0 and "v4" in out and "GRPO" in out and "refunds" in out
    code, out, _ = run(capsys, "verdict", "refund-bot", "--behavior", "refunds")
    assert code == 0 and "refunds: v4 beats v3 by 5" in out
    assert any(p.endswith("dashboard?behavior=refunds") for _, p, _ in fake)


def test_promote_live_and_keys(fake, capsys):
    code, out, _ = run(capsys, "promote", "refund-bot", "v4")
    assert code == 0 and "v4 is now serving" in out
    assert ("POST", "/agents/refund-bot/promote", {"version": "v4"}) in fake

    code, out, _ = run(
        capsys,
        "live",
        "refund-bot",
        "--day",
        "2026-09-17",
        "--version",
        "v3",
        "--replies",
        "2400",
        "--flagged",
        "98",
        "--p50",
        "0.6",
    )
    assert code == 0 and "recorded for v3" in out
    body = next(b for m, p, b in fake if p == "/live")
    assert body == [
        {
            "agent": "refund-bot",
            "day": "2026-09-17",
            "version": "v3",
            "replies": 2400,
            "flagged": 98,
            "p50s": 0.6,
        }
    ]

    code, out, _ = run(capsys, "keys")
    assert code == 0 and "laptop" in out and "zp_1234" in out and "Account" in out


def test_errors_go_to_stderr(fake, capsys, monkeypatch):
    code, _, err = run(
        capsys, "live", "refund-bot", "--day", "yesterday", "--version", "v3", "--replies", "1"
    )
    assert code == 1 and "day" in err

    def down(method, path, *, api_key, body=None, timeout=30.0):
        raise platform.PlatformError(503, f"{method} {path}: down")

    monkeypatch.setattr(platform, "_request", down)
    code, _, err = run(capsys, "agents")
    assert code == 1 and "down" in err


def test_purge_is_gone(capsys):
    with pytest.raises(SystemExit):
        cli.main(["purge", "--agent", "x"])
