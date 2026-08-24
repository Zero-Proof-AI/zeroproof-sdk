"""Local desk session and hosted-key status.

Hosted Qwen reads VLLM_API_KEY from the process environment only
(.env loaded at boot, or POST /api/auth/hosted into memory).
Never return, log, or persist the key.
"""
from __future__ import annotations

import hmac
import hashlib
import os
import re
import secrets
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote

STUDIO = Path(__file__).resolve().parent.parent
ROOT = STUDIO.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COOKIE = "zp_session"
_SECRET = secrets.token_bytes(32)
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_HANDLE = re.compile(r"^[a-z0-9._-]{2,40}$")
DESK_USER = "sahana@zeroproofai.com"
DESK_PASSWORD = "123abc"
ENDPOINT_DOWN = "The endpoint isn't up"
_PROBE_TTL = 15.0
_PROBE = {"at": 0.0, "ok": False}
_PROBE_LOCK = threading.Lock()


def _normalize_identity(raw: str) -> str | None:
    s = str(raw or "").strip().lower()
    if not s or len(s) > 120:
        return None
    if s in {
        "sahana", "sahnaa", "sahana@zeroproof", "sahnaa@zeroproof",
        "sahana@zeroproofai.com", "sahana@zeroproof.local",
        "sahnaa@zeroproofai.com", "sahnaa@zeroproof.local",
    }:
        return DESK_USER
    if _EMAIL.match(s) or _HANDLE.match(s):
        return s
    return None


def load_env() -> None:
    if str(os.environ.get("VLLM_API_KEY") or "").strip():
        return
    for path in (ROOT / ".env", STUDIO / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() != "VLLM_API_KEY":
                continue
            key = value.strip().strip("'").strip('"')
            if key:
                os.environ["VLLM_API_KEY"] = key
            return


def hosted_key() -> str:
    return str(os.environ.get("VLLM_API_KEY") or "").strip()


def _invalidate_probe() -> None:
    with _PROBE_LOCK:
        _PROBE["at"] = 0.0
        _PROBE["ok"] = False


def hosted_ready(*, fresh: bool = False) -> bool:
    """True only if a key is present and hosted Qwen / stressd-vllm is answering."""
    if not hosted_key():
        _invalidate_probe()
        return False
    now = time.monotonic()
    with _PROBE_LOCK:
        if not fresh and now - _PROBE["at"] < _PROBE_TTL:
            return bool(_PROBE["ok"])
    from zeroproof_simulations.agents import ping_hosted
    ok = bool(ping_hosted(timeout=2.5))
    with _PROBE_LOCK:
        _PROBE["at"] = time.monotonic()
        _PROBE["ok"] = ok
    return ok


def has_hosted_key() -> bool:
    return hosted_ready()


def _sign(email: str) -> str:
    sig = hmac.new(_SECRET, email.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{email}|{sig}"


def _verify(raw: str) -> str | None:
    try:
        value = unquote(raw)
    except Exception:
        return None
    if "|" not in value:
        return None
    email, sig = value.rsplit("|", 1)
    expect = hmac.new(_SECRET, email.encode(), hashlib.sha256).hexdigest()[:32]
    ident = _normalize_identity(email)
    if not ident:
        return None
    if not hmac.compare_digest(sig, expect):
        return None
    return ident


def email_from_cookie(header: str | None) -> str | None:
    if not header:
        return None
    for part in header.split(";"):
        part = part.strip()
        if part.startswith(COOKIE + "="):
            return _verify(part.split("=", 1)[1])
    return None


def cookie_header(email: str, *, clear: bool = False) -> str:
    if clear:
        return f"{COOKIE}=; HttpOnly; Path=/; SameSite=Lax; Max-Age=0"
    token = _sign(email)
    return f"{COOKIE}={token}; HttpOnly; Path=/; SameSite=Lax; Max-Age=2592000"


def status(raw: str | None = None) -> tuple[dict, str | None]:
    return {
        "email": email_from_cookie(raw),
        "has_hosted_key": has_hosted_key(),
    }, None


def _password_ok(password: str) -> bool:
    got = str(password or "").encode("utf-8")
    want = DESK_PASSWORD.encode("utf-8")
    if len(got) != len(want):
        hmac.compare_digest(want, want)
        return False
    return hmac.compare_digest(got, want)


def login(body: dict | None = None) -> tuple[dict, str | None]:
    body = body or {}
    email = _normalize_identity(str(body.get("email") or ""))
    if not email or not _password_ok(str(body.get("password") or "")):
        return {"error": "invalid login"}, None
    return {"email": email, "has_hosted_key": has_hosted_key()}, cookie_header(email)


def logout() -> tuple[dict, str]:
    return {"email": None, "has_hosted_key": has_hosted_key()}, cookie_header("", clear=True)


def set_hosted(body: dict | None = None) -> dict:
    key = str((body or {}).get("key") or "").strip()
    if not key:
        return {"error": "key required"}
    os.environ["VLLM_API_KEY"] = key
    _invalidate_probe()
    return {"has_hosted_key": has_hosted_key()}
