"""One OpenAI-compatible chat call, and nothing else.

The recipe writes its own records and questions, plays the teacher and runs
the judge through this, so every model call in the lane goes through one
place. Keys are read from the environment on every call and never written to
disk or logged: a recipe that prints a key once ends up with that key in a
notebook, a CI log and a screenshot.

Point ``VLLM_BASE_URL`` (or ``--base-url``) at any OpenAI-compatible server.
Nothing here is specific to a provider.

``dry_run(True)`` swaps the network call for canned replies keyed on the
system prompt, so ``smoke.sh`` can walk the whole pipeline with no key.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

#: Process-wide defaults, set once by ``configure`` from the CLI flags.
MODEL = DEFAULT_MODEL
BASE_URL = ""
DRY_RUN = False

# ``--dry-run`` writes records whose id looks like this; the canned writer,
# teacher and judge all key on it.
_DRY_ID = re.compile(r"\bREC\d{5}\b")
# The canned judge calls a reply concise under this many characters of judge
# input (the reply plus the fixed wrapper). The canned teacher writes about
# 60 characters, the canned base about 300.
_DRY_CONCISE_CHARS = 200


def configure(*, model: str = "", base_url: str = "") -> None:
    """Set the default writer/teacher model and endpoint for this process."""
    global MODEL, BASE_URL
    if model:
        MODEL = model
    if base_url:
        BASE_URL = base_url


def dry_run(on: bool = True) -> None:
    """Route every ``chat`` call to canned replies. No key, no network."""
    global DRY_RUN
    DRY_RUN = on


def require_env(base_url: str = "") -> None:
    """Say what is missing in one sentence, before any work is done."""
    if DRY_RUN:
        return
    if not str(os.environ.get("VLLM_API_KEY") or "").strip():
        raise SystemExit(
            "VLLM_API_KEY is not set. Export the key for your OpenAI-compatible endpoint "
            "in your shell (never in a file git can see), or pass --dry-run."
        )
    if not (base_url or BASE_URL or os.environ.get("VLLM_BASE_URL") or "").strip():
        raise SystemExit(
            "No endpoint. Pass --base-url or set VLLM_BASE_URL to an OpenAI-compatible "
            "/v1 URL (https://.../v1)."
        )


def _canned(messages: list[dict]) -> str:
    """Deterministic stand-ins for the four roles the lane asks a model to play."""
    system = next((str(m.get("content") or "") for m in messages if m.get("role") == "system"), "")
    user = next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "")
    if system.startswith("You invent ONE realistic record"):
        m = re.search(r"record number (\d+)", user)
        i = int(m.group(1)) if m else 0
        return json.dumps(
            {
                "record_id": f"REC{i:05d}",
                "status": ("open", "merged", "closed")[i % 3],
                "owner": f"team-{i % 4}",
                "opened": f"2026-09-{1 + i % 28:02d}",
                "comments": 3 * i,
                "labels": ["bug"],
                "priority": "p2",
            }
        )
    if system.startswith("You write the opening message"):
        ids = _DRY_ID.findall(user) or ["my record"]
        return (
            f"Can you tell me the status and owner of {' and '.join(ids)}? I need it for standup."
        )
    if system.startswith("You judge whether a reply"):
        return "YES" if len(user) < _DRY_CONCISE_CHARS else "NO"
    ids = ", ".join(_DRY_ID.findall(system)) or "the record"
    if "Never sacrifice a fact" in system:
        return f"{ids} is open, owned by team-1, no blockers."
    return (
        "Thank you so much for reaching out, and I completely understand why you would "
        f"want an update on {ids}. Let me walk you through everything I was able to find. "
        f"First, I checked {ids} in our system, and I can confirm that it is currently open. "
        "It is owned by team-1. There are no blockers at this time. Please do not hesitate "
        "to let me know if there is anything else at all I can help you with today!"
    )


def chat(
    messages: list[dict],
    *,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    retries: int = 4,
) -> str:
    """One completion. The key comes from ``api_key`` or ``VLLM_API_KEY``, never a file."""
    if DRY_RUN:
        return _canned(messages)
    key = str(api_key or os.environ.get("VLLM_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "VLLM_API_KEY is not set. Export it in your shell; do not put it in a file "
            "that git can see."
        )
    url = (base_url or BASE_URL or os.environ.get("VLLM_BASE_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "No endpoint. Pass --base-url or set VLLM_BASE_URL to an "
            "OpenAI-compatible /v1 endpoint."
        )
    body = json.dumps(
        {
            "model": model or MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode()
    last: Exception | None = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(
                url.rstrip("/") + "/chat/completions",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read())
            return str(payload["choices"][0]["message"].get("content") or "")
        except Exception as exc:
            last = exc
    raise RuntimeError(f"chat failed after {retries} attempts: {type(last).__name__}: {last}")
