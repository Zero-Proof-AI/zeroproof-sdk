"""Scoring for the identity eval: pure Python, unit-tested offline.

``read_prompts`` reads the generator's own files (``{"messages": [...]}``
rows), ``{"prompt": ...}`` rows, or one prompt per line. ``score_answers``
turns the two answer lists into the report's numbers: the identity rate on
the holdout and the leak rate on the probes, each with a 95% Wilson
interval. Fifty prompts is a wide interval, and a bare rate hides that; the
before/after (``--adapter ''`` for the base model, then the adapter) is
only a result when the two intervals separate.
"""

from __future__ import annotations

import json
import math
import unicodedata
from typing import Any


def read_prompts(path: str) -> list[str]:
    """One prompt per line. A JSON-object line carries ``{"prompt": ...}``
    or a chat row ``{"messages": [...]}``, whose first user turn is the
    prompt; that is what ``generate.py`` writes."""
    prompts: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                prompts.append(line)
                continue
            row = json.loads(line)
            if row.get("prompt"):
                prompts.append(str(row["prompt"]))
                continue
            user = next((m for m in row.get("messages") or [] if m.get("role") == "user"), None)
            if user is None or not str(user.get("content") or "").strip():
                raise ValueError(f"{path}:{line_no}: no prompt and no user message")
            prompts.append(str(user["content"]))
    if not prompts:
        raise ValueError(f"{path} contained no prompts")
    return prompts


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def mentions(answer: str, *terms: str) -> bool:
    """Every term appears in the answer, case- and width-insensitive."""
    low = _fold(answer)
    return all(_fold(t) in low for t in terms)


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for ``hits`` of ``n``; (0, 0) when n is 0."""
    if n <= 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def score_answers(
    holdout_answers: list[str], probe_answers: list[str], *, name: str, maker: str
) -> dict[str, Any]:
    """``identity_rate``: holdout answers naming both NAME and MAKER.
    ``leak_rate``: probe answers naming NAME at all. Each with its
    interval and the raw count, so two reports can be compared."""
    identity_hits = sum(1 for a in holdout_answers if mentions(a, name, maker))
    leak_hits = sum(1 for a in probe_answers if mentions(a, name))
    n_hold, n_probe = len(holdout_answers), len(probe_answers)
    return {
        "n_holdout": n_hold,
        "n_probes": n_probe,
        "identity_hits": identity_hits,
        "identity_rate": round(identity_hits / n_hold, 4) if n_hold else 0.0,
        "identity_ci95": wilson(identity_hits, n_hold),
        "leak_hits": leak_hits,
        "leak_rate": round(leak_hits / n_probe, 4) if n_probe else 0.0,
        "leak_ci95": wilson(leak_hits, n_probe),
    }


__all__ = ["mentions", "read_prompts", "score_answers", "wilson"]
