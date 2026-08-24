"""Shared Audit + Train copy. One source for repeats and pack tags."""
from __future__ import annotations


def repeats_mean_text(mean: float) -> str:
    n = float(mean or 0)
    if n <= 0:
        return "0"
    if abs(n - round(n)) < 0.05:
        return f"{int(round(n))}.0"
    return f"{n:.1f}"


def repeats_counts(
    mean: float,
    n_prompts: int,
    n_rows: int,
    min_k: int | None = None,
    max_k: int | None = None,
) -> str:
    """Number half of the repeats line. Range only when n is not uniform."""
    shown = repeats_mean_text(mean)
    lo = int(min_k) if min_k is not None else 0
    hi = int(max_k) if max_k is not None else 0
    if lo > 0 and hi > lo:
        shown = f"{shown} (range {lo}–{hi})"
    return (
        f"{shown} · {int(n_prompts or 0)} situations · "
        f"{int(n_rows or 0)} conversations"
    )


def repeats_line(
    mean: float,
    n_prompts: int,
    n_rows: int,
    min_k: int | None = None,
    max_k: int | None = None,
) -> str:
    return (
        "Average repeats per situation: "
        + repeats_counts(mean, n_prompts, n_rows, min_k, max_k)
    )


def pack_tags(bin: int | None, contrasting: bool) -> list[str]:
    """Which download pack this conversation belongs to.

    Good behavior: this row passed.
    Contrasting: this situation has both Pass and Fail; the row is one repeat.
    Neither: Fail with no contrast, or ungraded.
    """
    tags: list[str] = []
    if bin == 1:
        tags.append("Good behavior")
    if contrasting:
        tags.append("Contrasting")
    if not tags:
        tags.append("Neither")
    return tags
