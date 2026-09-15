"""Did the agent say what only the teacher was told?

``privileged`` on a row is the teacher's context: the hidden state of
the world, the outcome the task expects, the principle at stake. The
student never sees it, and the exporters scrub it. That guards the
training file. This guards the reply: an agent whose answer contains
that text got it from somewhere it should not have.

``leak_report`` says three things, and the first is the one that
matters: how many rows it could check at all. A run where nothing
populated ``privileged`` has nothing to leak, and a "no leak" verdict on
it is vacuous. The report says so instead of passing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .style import assistant_text

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", str(text or "")).strip().lower()


def _needles(privileged: Any, *, min_len: int) -> list[tuple[str, str]]:
    """(field, text) pairs worth searching for. Short values are skipped:
    a bare status word appears in honest replies too."""
    if not isinstance(privileged, dict):
        return []
    out: list[tuple[str, str]] = []

    def walk(field: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{field}.{k}", v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(field, v)
        elif isinstance(value, str):
            text = _norm(value)
            if len(text) >= min_len:
                out.append((field, text))

    for key in ("reference", "principle", "hidden_state"):
        if privileged.get(key):
            walk(key, privileged[key])
    return out


def leak_report(rows: Sequence[dict], *, min_len: int = 12) -> dict[str, Any]:
    """Which rows quote their own ``privileged`` block in the agent's text.

    Checks every row that carries ``privileged`` (``reference``,
    ``principle``, and every string in ``hidden_state`` at least
    ``min_len`` characters long) against the final reply and every
    assistant turn. Returns ``n_rows``, ``n_checked``, ``n_leaked``,
    ``rate`` (over checked rows), ``checked`` (False when no row carried
    the block, so the result is vacuous), ``leaked`` (up to 20 rows:
    ``scenario_id``, ``rollout_index``, ``field``, ``needle``) and
    ``summary``. Does not mutate ``rows``.
    """
    n_rows = 0
    n_checked = 0
    leaked: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        n_rows += 1
        needles = _needles(row.get("privileged"), min_len=min_len)
        if not needles:
            continue
        n_checked += 1
        hay = _norm(assistant_text(row))
        for field, needle in needles:
            if needle in hay:
                leaked.append(
                    {
                        "scenario_id": row.get("scenario_id"),
                        "rollout_index": row.get("rollout_index"),
                        "field": field,
                        "needle": needle[:80],
                    }
                )
                break
    n_leaked = len(leaked)
    rate = (n_leaked / n_checked) if n_checked else 0.0
    if not n_checked:
        summary = (
            f"checked 0 of {n_rows} rows: none carried privileged context, "
            "so this says nothing about leaks"
        )
    elif not n_leaked:
        summary = f"checked {n_checked} of {n_rows} rows: no reply quoted its privileged context"
    else:
        summary = (
            f"checked {n_checked} of {n_rows} rows: {n_leaked} quoted privileged context "
            f"({rate:.0%})"
        )
    return {
        "n_rows": n_rows,
        "n_checked": n_checked,
        "n_leaked": n_leaked,
        "rate": round(rate, 4),
        "checked": n_checked > 0,
        "leaked": leaked[:20],
        "summary": summary,
    }


def format_leak_report(report: dict[str, Any]) -> str:
    """One line per fact, the summary first."""
    lines = [str(report.get("summary") or "")]
    for hit in report.get("leaked") or []:
        lines.append(
            f"  {hit.get('scenario_id')} r{hit.get('rollout_index')}: "
            f"{hit.get('field')} = {hit.get('needle')!r}"
        )
    return "\n".join(lines)


__all__ = ["format_leak_report", "leak_report"]
