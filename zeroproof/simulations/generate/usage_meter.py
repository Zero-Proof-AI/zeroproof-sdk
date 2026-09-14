"""Report hosted-model tokens to the platform so the Usage page counts them.

Every call to the hosted policy or judge answers with a ``usage`` block. The
serving endpoints authenticate with a shared key and cannot tell accounts
apart, so the SDK, which holds the account's own key, reports what it used:
``POST /usage`` on the platform API with the input and output token counts,
batched, from a background thread, and once more at exit.

Only calls to the hosted endpoints are reported; a run against a bring-your-own
model is the customer's own bill. Set ``ZEROPROOF_NO_USAGE_REPORT=1`` to turn
the meter off. Reporting never raises and never blocks a model call.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

FLUSH_EVERY_S = 15.0
FLUSH_EVERY_CALLS = 50


def _api_url() -> str:
    from ...auth import _api_url as auth_url

    return auth_url()


def _api_key() -> str | None:
    from ...auth import resolve_api_key

    return resolve_api_key()


class UsageMeter:
    """Thread-safe accumulator with a background flusher."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in = 0
        self._out = 0
        self._calls = 0
        self._last_flush = time.monotonic()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.sent: list[dict[str, int]] = []  # for tests and `zeroproof status`
        self.dropped = 0

    def enabled(self) -> bool:
        return os.environ.get("ZEROPROOF_NO_USAGE_REPORT", "").strip() not in {"1", "true", "yes"}

    def add(self, input_tokens: int, output_tokens: int) -> None:
        if not self.enabled():
            return
        if input_tokens <= 0 and output_tokens <= 0:
            return
        with self._lock:
            self._in += max(0, int(input_tokens))
            self._out += max(0, int(output_tokens))
            self._calls += 1
            due = self._calls >= FLUSH_EVERY_CALLS
        self._ensure_thread()
        if due:
            self.flush()

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop, name="zeroproof-usage-meter", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(FLUSH_EVERY_S):
            self.flush()

    def _take(self) -> tuple[int, int]:
        with self._lock:
            pending = (self._in, self._out)
            self._in = self._out = self._calls = 0
            self._last_flush = time.monotonic()
        return pending

    def flush(self) -> bool:
        """Send whatever is pending. True when nothing is left owed."""
        tokens_in, tokens_out = self._take()
        if not tokens_in and not tokens_out:
            return True
        if self._post(tokens_in, tokens_out):
            self.sent.append({"input_tokens": tokens_in, "output_tokens": tokens_out})
            self.dropped = 0
            return True
        # Put it back so the next flush retries, unless the key is missing, in
        # which case nobody is there to credit and holding it only leaks memory.
        with self._lock:
            self._in += tokens_in
            self._out += tokens_out
        self.dropped += 1
        if self.dropped >= 3:
            self._take()
        return False

    def _post(self, tokens_in: int, tokens_out: int) -> bool:
        key = _api_key()
        if not key:
            return False
        body = json.dumps({"input_tokens": tokens_in, "output_tokens": tokens_out}).encode()
        req = urllib.request.Request(
            _api_url().rstrip("/") + "/usage",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "X-Api-Key": key},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as res:
                return 200 <= res.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False


METER = UsageMeter()


def report_usage(reply: Any, *, hosted: bool) -> None:
    """Called after every completion; a no-op unless the call was hosted."""
    if not hosted or not isinstance(reply, dict):
        return
    usage = reply.get("_usage")
    if not isinstance(usage, dict):
        return
    METER.add(int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0))


def flush_usage() -> bool:
    return METER.flush()


atexit.register(flush_usage)
