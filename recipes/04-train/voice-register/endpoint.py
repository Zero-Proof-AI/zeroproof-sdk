"""One OpenAI-compatible chat call, and nothing else.

The recipe writes its own situations and plays its own judge through this,
so every model call in the lane goes through one place. The key is read from
the environment on every call and is never written to disk or logged: a
recipe that prints a key once ends up with that key in a notebook, a CI log
and a screenshot.

Point ``--base-url`` at any OpenAI-compatible server. Nothing here is
specific to a provider.
"""
from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def chat(messages: list[dict], *, model: str = DEFAULT_MODEL, base_url: str = "",
         temperature: float = 0.7, max_tokens: int = 2048, retries: int = 4) -> str:
    """One completion. ``VLLM_API_KEY`` comes from the environment only."""
    key = str(os.environ.get("VLLM_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "VLLM_API_KEY is not set. Export it in your shell; do not put it in a file "
            "that git can see."
        )
    url = (base_url or os.environ.get("VLLM_BASE_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "No endpoint. Pass --base-url or set VLLM_BASE_URL to an "
            "OpenAI-compatible /v1 endpoint."
        )
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}).encode()
    last: Exception | None = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(
                url.rstrip("/") + "/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read())
            return str(payload["choices"][0]["message"].get("content") or "")
        except Exception as exc:  # noqa: BLE001 - retried, then surfaced
            last = exc
    raise RuntimeError(f"chat failed after {retries} attempts: {type(last).__name__}: {last}")
