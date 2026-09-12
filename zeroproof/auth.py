"""Sign in from a terminal or a coding agent: ``zeroproof login``.

Device authorization flow (RFC 8628 shape) against the Zero Proof Labs token
gate. The CLI asks the gate for a code pair, prints a link and a short code,
and polls until the human has signed in and pressed Approve in the browser.
The API key that comes back is written to ``~/.zeroproof/credentials.json``
and every SDK call reads it from there when ``ZEROPROOF_API_KEY`` is unset.

A pending login survives the process: if the harness running the command
stops it before approval, the next ``zeroproof login`` resumes the same code
instead of printing a new one.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Callable

DEFAULT_API_URL = "https://api.zeroproofai.com"


class LoginError(RuntimeError):
    pass


def _api_url() -> str:
    return os.environ.get("ZEROPROOF_API_URL", DEFAULT_API_URL).rstrip("/")


def config_dir() -> Path:
    """``$ZEROPROOF_HOME`` or ``~/.zeroproof``."""
    return Path(os.environ.get("ZEROPROOF_HOME") or Path.home() / ".zeroproof")


def credentials_path() -> Path:
    return config_dir() / "credentials.json"


def _pending_path() -> Path:
    return config_dir() / "pending-login.json"


def _write_private(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:  # Windows
        pass
    os.replace(tmp, path)


def _read(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def stored_api_key() -> str | None:
    """The key saved by ``zeroproof login``, or ``None``."""
    data = _read(credentials_path())
    key = (data or {}).get("api_key")
    return str(key) if key else None


def resolve_api_key(explicit: str | None = None) -> str | None:
    """``explicit`` > ``ZEROPROOF_API_KEY`` > the saved credentials file."""
    return explicit or os.environ.get("ZEROPROOF_API_KEY") or stored_api_key()


def _post(path: str, body: dict, timeout: int = 30) -> tuple[int, dict]:
    request = urllib.request.Request(
        _api_url() + path, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as err:
        raw = err.read().decode(errors="replace")
        try:
            return err.code, json.loads(raw)
        except ValueError:
            return err.code, {"error": raw[:200]}
    except urllib.error.URLError as err:
        raise LoginError(f"Could not reach {_api_url()}: {err.reason}") from None


def _default_name() -> str:
    host = socket.gethostname().split(".")[0] or "cli"
    return f"cli {host}"[:50]


def _start(name: str) -> dict:
    status, flow = _post("/device/code", {"name": name})
    if status != 200 or "device_code" not in flow:
        raise LoginError(f"Login could not start ({status}): {flow.get('error', flow)}")
    flow["api_url"] = _api_url()
    flow["expires_at"] = time.time() + int(flow.get("expires_in", 900))
    return flow


def _resume() -> dict | None:
    flow = _read(_pending_path())
    if not flow or flow.get("api_url") != _api_url():
        return None
    if float(flow.get("expires_at", 0)) - 30 <= time.time():
        return None
    return flow


def login(*, name: str | None = None, wait: bool = True, timeout: float | None = None,
          open_browser: bool = True, out: Callable[[str], None] | None = None) -> str | None:
    """Run the device login. Returns the API key, or ``None`` if still pending.

    ``wait=False`` prints the link and returns at once; run again to finish.
    ``timeout`` caps the wait in seconds (default: until the code expires).
    """
    say = out or (lambda s: print(s, file=sys.stderr, flush=True))
    flow = _resume()
    if flow is None:
        flow = _start(name or _default_name())
        _write_private(_pending_path(), flow)
        say("Open this link and press Approve:")
    else:
        say("Resuming the login you started. Open this link and press Approve:")
    say("")
    say(f"    {flow['verification_uri_complete']}")
    say("")
    say(f"    code: {flow['user_code']}")
    say("")
    if open_browser:
        try:
            webbrowser.open(flow["verification_uri_complete"])
        except Exception:  # noqa: BLE001  headless box, no browser
            pass
    if not wait:
        say("Run `zeroproof login` again once you have approved.")
        return None

    deadline = float(flow["expires_at"])
    if timeout is not None:
        deadline = min(deadline, time.time() + timeout)
    interval = max(1, int(flow.get("interval", 5)))
    body = {"device_code": flow["device_code"], "user_code": flow["user_code"]}
    say("Waiting for approval...")
    while True:
        status, data = _post("/device/token", body)
        if status == 200 and data.get("api_key"):
            _write_private(credentials_path(), {
                "api_key": data["api_key"],
                "api_url": flow["api_url"],
                "name": data.get("name"),
                "user_id": data.get("user_id"),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            try:
                _pending_path().unlink()
            except OSError:
                pass
            say(f"Logged in. Key saved to {credentials_path()}")
            return data["api_key"]
        error = data.get("error", "")
        if error == "authorization_pending":
            if time.time() >= deadline:
                say("Still waiting. Run `zeroproof login` again to keep waiting.")
                return None
            time.sleep(interval)
            continue
        if error == "slow_down":
            interval += 5
            time.sleep(interval)
            continue
        try:
            _pending_path().unlink()
        except OSError:
            pass
        if error == "expired_token":
            raise LoginError("That code expired. Run `zeroproof login` again.")
        raise LoginError(f"Login failed ({status}): {error or data}")


def logout() -> bool:
    """Delete the saved credentials. Returns whether anything was removed."""
    removed = False
    for path in (credentials_path(), _pending_path()):
        try:
            path.unlink()
            removed = removed or path.name == "credentials.json"
        except OSError:
            pass
    return removed


def status() -> dict:
    """What the SDK would use right now, with the key masked."""
    env = os.environ.get("ZEROPROOF_API_KEY")
    saved = _read(credentials_path()) or {}
    key = env or saved.get("api_key")
    return {
        "api_url": _api_url(),
        "source": "ZEROPROOF_API_KEY" if env else ("file" if saved.get("api_key") else None),
        "path": str(credentials_path()),
        "key": (key[:7] + "..." + key[-4:]) if key and len(key) > 12 else (key or None),
        "name": None if env else saved.get("name"),
        "pending": _resume() is not None,
    }
