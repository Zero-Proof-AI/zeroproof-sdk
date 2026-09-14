"""pass@1, pass^k and pass@k from the same graded groups.

Three numbers, one primitive: the per-task pass rate ``c / n`` over the
``k`` repeats of one phrasing. Each has one job.

* ``pass@1`` is the mean per-task pass rate. The agent runs once in
  production, so this is the measurement headline.
* ``pass^k`` is the chance all ``k`` repeats pass (tau-bench's
  consistency metric). A behavior that lands 60% of the time is not a
  contract; this is the reliability line.
* ``pass@k`` is the chance at least one of ``k`` repeats passes.
  ``pass@k - pass@1`` is the headroom a grouped RL update can learn
  from: the same asks ``group_signal`` counts as mixed.

Both ``pass^k`` and ``pass@k`` use the unbiased combinatorial estimators
(Chen et al. 2021 for pass@k; Yao et al. 2024 for pass^k), so a group
with ``n`` graded repeats contributes ``1 - C(n-c, k) / C(n, k)`` and
``C(c, k) / C(n, k)`` rather than a plug-in power of ``c / n``.

Judge noise: with an LLM judge, pass@k inflates on false positives and
pass^k inflates on false negatives. pass@1 is the least sensitive of the
three, which is why it carries the headline.

Intervals: only ``pass@1`` carries one (``.ci95``, a task bootstrap).
``pass^k`` and ``pass@k`` are point estimates — there is no ``ci95`` for
them and none is computed. To report the reliability line with an
interval, do it over tasks yourself: the per-group values are
``.per_task``, and ``score.stats.bootstrap_ci`` / ``wilson_interval``
take a vector of them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

JUDGE_NOISE_NOTE = (
    "pass@k inflates on judge false positives, pass^k on false negatives; "
    "pass@1 is the least judge-sensitive"
)


def _comb(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def _pass_at_k_group(n: int, c: int, k: int) -> float:
    """Unbiased P(at least one of k draws passes) from n samples, c passing."""
    total = _comb(n, k)
    if total == 0:
        return float("nan")
    return 1.0 - _comb(n - c, k) / total


def _pass_pow_k_group(n: int, c: int, k: int) -> float:
    """Unbiased P(all k draws pass) from n samples, c passing."""
    total = _comb(n, k)
    if total == 0:
        return float("nan")
    return _comb(c, k) / total


@dataclass(frozen=True)
class PassAt:
    """pass@1 / pass^k / pass@k over graded groups. See module docstring."""

    k: int
    pass_at_1: float | None
    pass_pow_k: float | None
    pass_at_k: float | None
    n_groups: int
    n_rows: int
    n_groups_at_k: int = 0
    #: unanimous groups shorter than k counted as if they stayed unanimous
    n_groups_imputed: int = 0
    #: ``{prompt: c / n}`` — a **dict keyed by the group's prompt string**,
    #: not a list, so ``per_task[0]`` is a ``KeyError``, not the first task.
    #: Iterate it as ``.per_task.items()``; ``.per_task.values()`` is the
    #: pass-rate vector pass@1 averages and ``ci95`` bootstraps.
    per_task: dict[str, float] = field(default_factory=dict)
    note: str = ""
    #: task-bootstrap 95% interval on **pass@1 only**. ``pass_pow_k`` and
    #: ``pass_at_k`` carry no interval; there is no field for one.
    ci95: tuple[float, float] | None = None

    @property
    def headroom(self) -> float | None:
        """pass@k minus pass@1: what a grouped RL update has to learn from."""
        if self.pass_at_k is None or self.pass_at_1 is None:
            return None
        return self.pass_at_k - self.pass_at_1

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "pass_at_1": self.pass_at_1,
            "pass_pow_k": self.pass_pow_k,
            "pass_at_k": self.pass_at_k,
            "headroom": self.headroom,
            "n_groups": self.n_groups,
            "n_groups_at_k": self.n_groups_at_k,
            "n_groups_imputed": self.n_groups_imputed,
            "n_rows": self.n_rows,
            "note": self.note,
            "ci95": list(self.ci95) if self.ci95 else None,
        }

    def __str__(self) -> str:
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.2f}"

        ci = f" [{self.ci95[0]:.2f}..{self.ci95[1]:.2f}]" if self.ci95 else ""
        head = (
            f"pass@1 {fmt(self.pass_at_1)}{ci} | pass^{self.k} {fmt(self.pass_pow_k)} | "
            f"pass@{self.k} {fmt(self.pass_at_k)} | headroom {fmt(self.headroom)}"
        )
        tail = f"({self.n_groups} groups, k={self.k}"
        if self.note:
            tail += f"; {self.note}"
        return f"{head} {tail})"


def pass_at(
    rows: Sequence[dict] | Any,
    *,
    k: int | None = None,
    min_k: int = 4,
    unanimous_short: bool = False,
) -> PassAt:
    """pass@1, pass^k and pass@k from graded rows, grouped by prompt.

    Only binary ``reward`` (or ``qwen_reward``) rows count; partial and
    unjudged rows are skipped, the same rule ``group_signal`` uses.
    ``k`` defaults to the smallest group of two or more repeats, so every
    such group contributes to the ``k``-way estimators; groups with fewer
    than ``k`` graded repeats are left out of pass^k and pass@k (counted
    in ``n_groups_at_k``). pass@1 always averages every group.

    Below ``min_k`` repeats the ``k``-way numbers are ``None`` with a
    ``note`` instead of a number too noisy to act on. Pass ``k=`` to
    choose the draw size yourself.

    ``unanimous_short=True`` counts a unanimous group shorter than ``k``
    as if it stayed unanimous (pass^k and pass@k equal to its pass rate,
    1 or 0). That is the assumption a successive-allocation run stopped
    on, and leaving those groups out would score only the prompts that
    split and inflate the headroom. Mixed short groups still stay out.
    """
    from .optimize import _group_label_lists

    groups = _group_label_lists(list(rows) if not isinstance(rows, list) else rows)
    n_rows = sum(len(labels) for labels in groups.values())
    if not groups:
        return PassAt(
            k=int(k or 1),
            pass_at_1=None,
            pass_pow_k=None,
            pass_at_k=None,
            n_groups=0,
            n_rows=0,
            note="no binary rewards; grade first",
        )

    per_task = {prompt: sum(labels) / len(labels) for prompt, labels in groups.items()}
    pass_at_1 = sum(per_task.values()) / len(per_task)

    multi = [len(labels) for labels in groups.values() if len(labels) >= 2]
    resolved_k = int(k) if k is not None else (min(multi) if multi else 1)
    if resolved_k < 1:
        raise ValueError("k must be at least 1")

    eligible = [labels for labels in groups.values() if len(labels) >= resolved_k]
    imputed = (
        [labels for labels in groups.values() if len(labels) < resolved_k and len(set(labels)) == 1]
        if unanimous_short
        else []
    )
    note = ""
    pass_pow_k: float | None = None
    pass_at_k: float | None = None
    sizes = sorted(set(multi))
    uneven = k is None and len(sizes) > 1
    if resolved_k < max(1, int(min_k)):
        note = f"set repeats>={int(min_k)} for pass^k and pass@k"
        if uneven:
            # A time or row budget cut mid-group leaves ragged groups; k
            # defaults to the smallest, so the k-way numbers vanish even
            # though most groups reached the requested repeats.
            reached = sum(1 for n in multi if n >= sizes[-1])
            note = (
                f"groups are uneven ({sizes[0]} to {sizes[-1]} repeats); k defaults "
                f"to the smallest, so pass k={sizes[-1]} to score the {reached} "
                f"group(s) that reached it, or finish the cut groups"
            )
    elif not eligible and not imputed:
        note = f"no group has {resolved_k} graded repeats"
    else:
        pow_vals = [_pass_pow_k_group(len(g), sum(g), resolved_k) for g in eligible]
        at_vals = [_pass_at_k_group(len(g), sum(g), resolved_k) for g in eligible]
        for g in imputed:
            unanimous_value = float(g[0])
            pow_vals.append(unanimous_value)
            at_vals.append(unanimous_value)
        pass_pow_k = sum(pow_vals) / len(pow_vals)
        pass_at_k = sum(at_vals) / len(at_vals)
        if uneven:
            note = f"groups are uneven ({sizes[0]} to {sizes[-1]} repeats); k is the smallest"
    from .stats import bootstrap_ci

    return PassAt(
        k=resolved_k,
        pass_at_1=pass_at_1,
        pass_pow_k=pass_pow_k,
        pass_at_k=pass_at_k,
        n_groups=len(groups),
        n_rows=n_rows,
        n_groups_at_k=(len(eligible) + len(imputed)) if pass_at_k is not None else 0,
        n_groups_imputed=len(imputed) if pass_at_k is not None else 0,
        per_task=per_task,
        note=note,
        ci95=bootstrap_ci(list(per_task.values())),
    )


__all__ = ["JUDGE_NOISE_NOTE", "PassAt", "pass_at"]
