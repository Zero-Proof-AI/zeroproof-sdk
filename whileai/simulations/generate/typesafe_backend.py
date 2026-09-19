"""TypeSafe's Jev as a judge: typed decisions with calibrated probabilities.

``typesafe:<model>`` is a backend spec like ``anthropic:<model>``, with one
difference: it is a judge-only spec. Jev is a System One model (TypeSafe
AI, released 2026-09-15): it takes a piece of state plus typed questions
and returns, for every question, a probability over the answers you named,
generating no output tokens. That is the grader's shape, not the writer's.
The pointwise judge (``grade``), the audit, the pairwise judge, the rubric
judge and the advisory judge each become one request of questions, built in
``score.decision_judge``; ``agent=``, ``simulator=`` and ``user_model=``
need a chat model, and ``complete()`` says so when handed this spec.

Wire: ``POST {base}/v1/systemone`` with ``{"state", "model", "questions"}``.
Three question types: ``noul`` (a yes/no statement, answered as the
probability it is true), ``choice`` (one label from a set, answered as the
label, its confidence and the whole distribution) and ``score`` (an ordered
rubric, answered as the expected level, its confidence and the
distribution). The shape is what ``typesafe-sdk`` 0.7 sends (its generated
OpenAPI models); the calls go over ``requests``, which the package already
depends on. ``TYPESAFE_BASE_URL`` (the SDK's own variable) points the spec
at a gateway or a test server.

Access is early access behind a waitlist as of 2026-09-18; the key error
below says where. Nothing here was run against the live API yet: the
request and response shapes are the SDK's, checked offline.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any
from urllib.parse import urlparse

import requests

from whileai._env import getenv

from ..defaults import DECISION_TIMEOUT_S, TRANSIENT_BACKOFF_S, TRANSIENT_TRIES

TYPESAFE_BASE_URL = "https://api.typesafe.ai"
TYPESAFE_HOST = "api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
KEY_ENV = "TYPESAFE_API_KEY"
OVERRIDE_ENV = "WHILEAI_TYPESAFE_API_KEY"
BASE_URL_ENV = "TYPESAFE_BASE_URL"
SYSTEM_ONE_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"
SPEC_PREFIX = "typesafe:"

MISSING_TYPESAFE_KEY = (
    f"No TypeSafe API key: set {KEY_ENV} (or {OVERRIDE_ENV} to override it) to a key "
    "from console.typesafe.ai/settings/keys. Jev is in early access; the waitlist "
    "is at typesafe.ai."
)

NO_CHAT = (
    "{spec} answers typed questions, not chat. Use it as the judge: "
    'data.grade(spec="{spec}"), pairwise_judge(spec="{spec}"), '
    'rubric_judge(spec="{spec}"). Give agent=, simulator= and user_model= a '
    "chat model such as openai:<model>, anthropic:<model> or vllm:<model>@<url>."
)

# ERROR_BODY_CHARS = 300: how much of an error body a raised message keeps
# (convention; the SDK's own cap is 200).
ERROR_BODY_CHARS = 300
# RETRY_AFTER_CAP_S = 30: the longest a 429's retry-after is honored before
# the backoff schedule takes over; Jev's stated limit is 1,200 requests a
# minute, so a longer wait is a stuck server, not a queue (convention,
# untested).
RETRY_AFTER_CAP_S = 30.0

_TRANSIENT_STATUSES = {
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
}
ANSWER_TYPES = ("noul", "choice", "score")


def base_url() -> str:
    """The API root: ``TYPESAFE_BASE_URL`` when set, else the public one."""
    override = str(os.environ.get(BASE_URL_ENV) or "").strip().rstrip("/")
    return override or TYPESAFE_BASE_URL


def is_typesafe_url(url: str | None) -> bool:
    """True when this base URL is the TypeSafe API (or its override)."""
    if not url:
        return False
    raw = str(url)
    if "://" not in raw:
        raw = "https://" + raw
    if (urlparse(raw).hostname or "").lower() == TYPESAFE_HOST:
        return True
    return raw.rstrip("/").lower() == base_url().lower()


def is_decision_spec(spec: Any) -> bool:
    """True for a ``typesafe:`` spec string."""
    return isinstance(spec, str) and spec.strip().lower().startswith(SPEC_PREFIX)


def no_chat_error(spec: Any) -> str | None:
    """The one-sentence refusal for a decision spec in a chat role, or None."""
    if not is_decision_spec(spec):
        return None
    return NO_CHAT.format(spec=str(spec).strip())


def resolve_key(api_key: str | None = None) -> str:
    """The key for this endpoint: an explicit one, else the override, else
    ``TYPESAFE_API_KEY``. Never logged or returned in an error message."""
    if api_key:
        return str(api_key).strip()
    override = str(getenv("TYPESAFE_API_KEY") or "").strip()
    if override:
        return override
    return str(os.environ.get(KEY_ENV) or "").strip()


def missing_key(api_key: str | None = None) -> str | None:
    """The one-sentence auth error, or None when a key is present."""
    return None if resolve_key(api_key) else MISSING_TYPESAFE_KEY


def _headers(key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _error_message(body: Any) -> str:
    """The server's message from an error body, in the SDK's own order:
    ``error`` (string or ``{message}``), ``message``, ``detail`` (string,
    ``{message}`` or a validation list)."""
    if isinstance(body, str):
        return body[:ERROR_BODY_CHARS]
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    if isinstance(error, str):
        return error[:ERROR_BODY_CHARS]
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return str(error["message"])[:ERROR_BODY_CHARS]
    if isinstance(body.get("message"), str):
        return str(body["message"])[:ERROR_BODY_CHARS]
    detail = body.get("detail")
    if isinstance(detail, str):
        return detail[:ERROR_BODY_CHARS]
    if isinstance(detail, dict) and isinstance(detail.get("message"), str):
        return str(detail["message"])[:ERROR_BODY_CHARS]
    if isinstance(detail, list):
        parts = []
        for entry in detail:
            if isinstance(entry, dict) and isinstance(entry.get("msg"), str):
                loc = entry.get("loc")
                path = ".".join(str(x) for x in loc if x != "body") if isinstance(loc, list) else ""
                parts.append(f"{path}: {entry['msg']}" if path else entry["msg"])
        return "; ".join(parts)[:ERROR_BODY_CHARS]
    return ""


def _retry_after(headers: Mapping[str, str] | None) -> float | None:
    """Seconds from a 429's ``retry-after-ms`` or ``retry-after``, capped."""
    if not headers:
        return None
    lower = {str(k).lower(): v for k, v in headers.items()}
    raw_ms = lower.get("retry-after-ms")
    if raw_ms is not None:
        try:
            return min(RETRY_AFTER_CAP_S, max(0.0, float(raw_ms) / 1000.0))
        except (TypeError, ValueError):
            pass
    raw = lower.get("retry-after")
    if raw is not None:
        try:
            return min(RETRY_AFTER_CAP_S, max(0.0, float(raw)))
        except (TypeError, ValueError):
            return None
    return None


def _body_of(resp: Any) -> Any:
    try:
        return resp.json()
    except Exception:
        return str(getattr(resp, "text", "") or "")


def _post(url: str, model: str, body: dict, key: str, timeout: float) -> dict:
    """POST with the transient-retry policy both HTTP backends share: a 5xx
    or a 429 is retried ``TRANSIENT_TRIES`` times with a doubling backoff
    (a 429's retry-after wins when the server sends one); any other 4xx
    raises with the server's message and the model."""
    delay = TRANSIENT_BACKOFF_S
    tries = 0
    while True:
        resp = requests.post(url, headers=_headers(key), json=body, timeout=timeout)
        status = int(getattr(resp, "status_code", 0) or 0)
        if HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES:
            payload = _body_of(resp)
            if not isinstance(payload, dict):
                raise RuntimeError(f"TypeSafe returned a non-object body for {model}")
            return payload
        transient = status in _TRANSIENT_STATUSES or status == HTTPStatus.TOO_MANY_REQUESTS
        if transient and tries < TRANSIENT_TRIES:
            tries += 1
            wait = delay
            if status == HTTPStatus.TOO_MANY_REQUESTS:
                hinted = _retry_after(getattr(resp, "headers", None))
                wait = hinted if hinted is not None else delay
            time.sleep(wait)
            delay *= 2
            continue
        message = _error_message(_body_of(resp))
        if status == HTTPStatus.UNAUTHORIZED:
            raise RuntimeError(
                f"TypeSafe rejected the key for {model} (401): {MISSING_TYPESAFE_KEY}"
            )
        raise RuntimeError(f"TypeSafe {status} for {model}: {message or 'no message'}")


def decide(
    base: str,
    model: str,
    state: Any,
    questions: Mapping[str, Mapping[str, Any]],
    *,
    api_key: str | None = None,
    timeout: float = DECISION_TIMEOUT_S,
) -> dict[str, Any]:
    """One System One request: ``state`` plus named ``questions``, answered
    together. Returns ``{"model", "answers": {name: answer}, "usage"}``
    where each answer carries its ``type`` (``noul``: ``noul`` probability;
    ``choice``: ``choice``, ``confidence``, ``probabilities``; ``score``:
    ``score``, ``confidence``, ``legend``, ``probabilities``). An answer of a
    type this module does not know is dropped, as the SDK drops it."""
    key = resolve_key(api_key)
    if not key:
        raise RuntimeError(MISSING_TYPESAFE_KEY)
    if not questions:
        raise ValueError("decide() needs at least one question")
    root = str(base or base_url()).rstrip("/")
    if "://" not in root:
        root = "https://" + root
    body = {
        "state": state,
        "model": model or DEFAULT_MODEL,
        "questions": {str(k): dict(v) for k, v in questions.items()},
    }
    payload = _post(root + SYSTEM_ONE_PATH, model, body, key, timeout)
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, dict):
        raise RuntimeError(f"TypeSafe reply for {model} carried no answers object")
    answers: dict[str, dict] = {}
    for name, answer in raw_answers.items():
        if isinstance(answer, dict) and answer.get("type") in ANSWER_TYPES:
            answers[str(name)] = answer
    raw_usage = payload.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    return {
        "model": str(payload.get("model") or model),
        "answers": answers,
        "usage": {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
        },
    }


def list_models(
    base: str | None = None, *, api_key: str | None = None, timeout: float = DECISION_TIMEOUT_S
) -> list[str]:
    """``GET /v1/models``: the model names the key can use. The judge
    warm-up calls this, so a bad key fails once, before the fan-out."""
    key = resolve_key(api_key)
    if not key:
        raise RuntimeError(MISSING_TYPESAFE_KEY)
    root = str(base or base_url()).rstrip("/")
    if "://" not in root:
        root = "https://" + root
    resp = requests.get(root + MODELS_PATH, headers=_headers(key), timeout=timeout)
    status = int(getattr(resp, "status_code", 0) or 0)
    if status == HTTPStatus.UNAUTHORIZED:
        raise RuntimeError(f"TypeSafe rejected the key (401): {MISSING_TYPESAFE_KEY}")
    if not HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES:
        raise RuntimeError(f"TypeSafe {status} on {MODELS_PATH}: {_error_message(_body_of(resp))}")
    payload = _body_of(resp)
    models = payload.get("models") if isinstance(payload, dict) else None
    names: list[str] = []
    for entry in models or []:
        if isinstance(entry, dict) and entry.get("name"):
            names.append(str(entry["name"]))
    return names


# --- answer readers ---------------------------------------------------------


def noul_probability(answer: Mapping[str, Any] | None) -> float | None:
    """The probability a ``noul`` statement is true, or None."""
    if not isinstance(answer, Mapping) or answer.get("type") != "noul":
        return None
    raw = answer.get("noul")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    return value if 0.0 <= value <= 1.0 else None


def choice_of(answer: Mapping[str, Any] | None) -> tuple[str | None, dict[str, float]]:
    """``(label, probabilities)`` from a ``choice`` answer; the label is the
    most probable one (the server's ``choice``, else the argmax)."""
    if not isinstance(answer, Mapping) or answer.get("type") != "choice":
        return None, {}
    probabilities: dict[str, float] = {}
    raw = answer.get("probabilities")
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                probabilities[str(k)] = float(v)
    label = answer.get("choice")
    if not isinstance(label, str) or not label:
        label = max(probabilities, key=probabilities.__getitem__) if probabilities else None
    return label, probabilities


def score_of(answer: Mapping[str, Any] | None) -> tuple[float | None, dict[int, float]]:
    """``(expected level, probabilities by level)`` from a ``score`` answer."""
    if not isinstance(answer, Mapping) or answer.get("type") != "score":
        return None, {}
    probabilities: dict[int, float] = {}
    raw = answer.get("probabilities")
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            try:
                level = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                probabilities[level] = float(v)
    value = answer.get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if not probabilities:
            return None, {}
        value = sum(level * p for level, p in probabilities.items())
    return float(value), probabilities
