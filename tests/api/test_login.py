"""`zeroproof login`: the device flow from the CLI's side, with the gate faked."""
from __future__ import annotations

import json
import time

import pytest

from zeroproof import auth, cli
from zeroproof.simulations.ingest import platform


class FakeGate:
    """Records the calls the CLI makes and answers like the token gate."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.approved = False
        self.started = 0

    def post(self, path, body, timeout=30):
        self.calls.append((path, body))
        if path == "/device/code":
            self.started += 1
            return 200, {
                "device_code": "d" * 64, "user_code": "ABCD-EFGH",
                "verification_uri": "https://www.zeroproofai.com/device",
                "verification_uri_complete": "https://www.zeroproofai.com/device?code=ABCD-EFGH",
                "expires_in": 900, "interval": 0,
            }
        if path == "/device/token":
            assert body == {"device_code": "d" * 64, "user_code": "ABCD-EFGH"}
            if self.approved:
                return 200, {"api_key": "zp_" + "a" * 48, "name": "cli box", "user_id": "user_1"}
            return 400, {"error": "authorization_pending"}
        raise AssertionError(path)


@pytest.fixture
def gate(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROPROOF_HOME", str(tmp_path))
    monkeypatch.delenv("ZEROPROOF_API_KEY", raising=False)
    monkeypatch.delenv("ZEROPROOF_DELEGATED_CREDENTIAL", raising=False)
    monkeypatch.delenv("ZEROPROOF_API_URL", raising=False)
    fake = FakeGate()
    monkeypatch.setattr(auth, "_post", fake.post)
    monkeypatch.setattr(auth.time, "sleep", lambda s: None)
    return fake


def test_login_prints_link_and_saves_key_after_approval(gate, tmp_path):
    lines: list[str] = []
    polls = {"n": 0}

    def approve_on_second_poll(path, body, timeout=30):
        if path == "/device/token":
            polls["n"] += 1
            gate.approved = polls["n"] >= 2
        return FakeGate.post(gate, path, body, timeout)

    auth._post = approve_on_second_poll
    key = auth.login(open_browser=False, out=lines.append)

    assert key == "zp_" + "a" * 48
    text = "\n".join(lines)
    assert "https://www.zeroproofai.com/device?code=ABCD-EFGH" in text
    assert "ABCD-EFGH" in text
    saved = json.loads((tmp_path / "credentials.json").read_text())
    assert saved["api_key"] == key
    assert saved["name"] == "cli box"
    assert saved["api_url"] == "https://api.zeroproofai.com"
    assert not (tmp_path / "pending-login.json").exists()
    assert auth.stored_api_key() == key
    assert auth.resolve_api_key() == key


def test_no_wait_then_resume_uses_the_same_code(gate, tmp_path):
    assert auth.login(wait=False, open_browser=False, out=lambda s: None) is None
    assert (tmp_path / "pending-login.json").exists()
    assert gate.started == 1

    gate.approved = True
    lines: list[str] = []
    key = auth.login(open_browser=False, out=lines.append)
    assert key
    assert gate.started == 1, "resumed instead of minting a new code"
    assert "Resuming" in lines[0]


def test_timeout_leaves_the_pending_login_for_next_time(gate, tmp_path):
    assert auth.login(open_browser=False, timeout=0, out=lambda s: None) is None
    assert (tmp_path / "pending-login.json").exists()
    assert auth.status()["pending"] is True


def test_expired_pending_login_starts_over(gate, tmp_path):
    auth.login(wait=False, open_browser=False, out=lambda s: None)
    pending = json.loads((tmp_path / "pending-login.json").read_text())
    pending["expires_at"] = time.time() - 1
    (tmp_path / "pending-login.json").write_text(json.dumps(pending))
    gate.approved = True
    assert auth.login(open_browser=False, out=lambda s: None)
    assert gate.started == 2


def test_expired_code_at_the_gate_is_a_clear_error(gate, tmp_path):
    def expired(path, body, timeout=30):
        if path == "/device/token":
            return 400, {"error": "expired_token"}
        return FakeGate.post(gate, path, body, timeout)

    auth._post = expired
    with pytest.raises(auth.LoginError, match="expired"):
        auth.login(open_browser=False, out=lambda s: None)
    assert not (tmp_path / "pending-login.json").exists()


def test_key_name_defaults_to_the_host(gate):
    auth.login(wait=False, open_browser=False, out=lambda s: None)
    name = gate.calls[0][1]["name"]
    assert name.startswith("cli ") and len(name) <= 50
    auth.logout()
    auth.login(wait=False, name="claude-code", open_browser=False, out=lambda s: None)
    assert gate.calls[-1][1]["name"] == "claude-code"


def test_platform_calls_fall_back_to_the_saved_key(gate, monkeypatch):
    with pytest.raises(platform.PlatformError, match="zeroproof login"):
        platform._key(None)
    gate.approved = True
    auth.login(open_browser=False, out=lambda s: None)
    assert platform._key(None) == "zp_" + "a" * 48
    monkeypatch.setenv("ZEROPROOF_API_KEY", "zp_env")
    assert platform._key(None) == "zp_env", "the env var still wins"
    assert platform._key("zp_explicit") == "zp_explicit"


def test_logout_and_status(gate, capsys):
    assert auth.logout() is False
    gate.approved = True
    assert cli.main(["login", "--no-browser"]) == 0
    assert cli.main(["status"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["source"] == "file"
    assert shown["key"] == "zp_aaaa...aaaa"
    assert shown["name"] == "cli box"
    assert cli.main(["logout"]) == 0
    assert auth.stored_api_key() is None
    assert cli.main(["status"]) == 0
    assert json.loads(capsys.readouterr().out.split("Logged out.\n")[-1])["source"] is None


def test_cli_exit_codes(gate):
    assert cli.main(["login", "--no-browser", "--no-wait"]) == 0
    assert cli.main(["login", "--no-browser", "--timeout", "0"]) == 2

    def down(path, body, timeout=30):
        raise auth.LoginError("Could not reach the gate")

    auth._post = down
    auth.logout()
    assert cli.main(["login", "--no-browser"]) == 1
