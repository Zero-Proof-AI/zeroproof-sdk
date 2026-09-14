"""Confidence intervals, paired run comparison, and decontamination.

The numbers an engineer needs before believing a result (rlhf-book ch. 16:
labs win by raising the statistical power of the few evaluations they
track; contamination is found by n-gram overlap between training prompts
and evaluation prompts, 8-gram in the Tulu 3 decontamination).

* Every score gets an interval. Per-task pass rates and per-rollout marker
  values are clustered by task, so the bootstrap resamples tasks, not
  rows: rollouts of one ask are not independent draws.
* Two runs are compared on the tasks they share, as paired differences,
  with a bootstrap interval and a sign-flip permutation p-value. Unpaired
  comparison is the fallback and is labeled as such.
* Decontamination is word n-gram overlap (default 8) between a dataset's
  prompts and replies and the evaluation prompts it must not have seen.
  Short prompts fall back to exact normalized match.

Everything here is stdlib and deterministic under ``seed``.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_BOOT = 2000
_WORD = re.compile(r"[a-z0-9]+")


def _norm(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _words(text: Any) -> list[str]:
    return _WORD.findall(str(text or "").lower())


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


# ------------------------------------------------------------------ intervals


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a proportion. ``None`` when n is 0."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(
    values: Sequence[float],
    *,
    stat: Callable[[Sequence[float]], float] = _mean,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval of ``stat`` over ``values``. ``None``
    below three values, where the interval would be the data itself."""
    vals = [float(v) for v in values]
    if len(vals) < 3:
        return None
    rng = random.Random(seed)
    n = len(vals)
    stats = sorted(stat([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo_i = int((1 - level) / 2 * n_boot)
    hi_i = int((1 + level) / 2 * n_boot) - 1
    return (stats[max(0, lo_i)], stats[min(n_boot - 1, hi_i)])


def _task_of(row: dict) -> str:
    return str(row.get("task_id") or row.get("scenario_id") or row.get("prompt") or "")


def _by_task(rows: Sequence[dict], value: Callable[[dict], float | None]) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        v = value(row)
        if v is None:
            continue
        groups.setdefault(_task_of(row), []).append(float(v))
    return groups


def _binary(row: dict, key: str = "reward") -> float | None:
    v = row.get(key)
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f in (0.0, 1.0) else None


def _marker(name: str) -> Callable[[dict], float | None]:
    def get(row: dict) -> float | None:
        markers = row.get("markers")
        if not isinstance(markers, dict) or name not in markers:
            return None
        try:
            return float(markers[name])
        except (TypeError, ValueError):
            return None

    return get


def task_means(rows: Sequence[dict], metric: str = "pass_at_1") -> dict[str, float]:
    """Per-task mean of a metric: ``"pass_at_1"`` (binary reward) or
    ``"marker:<name>"``. The unit every interval and comparison rests on."""
    getter = _binary if metric == "pass_at_1" else _marker(metric.split(":", 1)[1])
    groups = _by_task(rows, getter)
    return {task: _mean(vals) for task, vals in groups.items()}


def metric_summary(
    rows: Sequence[dict], metric: str = "pass_at_1", *, n_boot: int = DEFAULT_BOOT, seed: int = 0
) -> dict[str, Any]:
    """Mean over tasks with a task-bootstrap 95% interval."""
    means = task_means(rows, metric)
    values = list(means.values())
    return {
        "metric": metric,
        "n_tasks": len(values),
        "n_rows": sum(
            len(v)
            for v in _by_task(
                rows, _binary if metric == "pass_at_1" else _marker(metric.split(":", 1)[1])
            ).values()
        ),
        "mean": _mean(values) if values else None,
        "ci95": bootstrap_ci(values, n_boot=n_boot, seed=seed),
    }


def marker_names(rows: Sequence[dict]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        markers = row.get("markers") if isinstance(row, dict) else None
        if isinstance(markers, dict):
            names.update(str(k) for k in markers)
    return sorted(names)


def marker_summary(
    rows: Sequence[dict],
    *,
    names: Sequence[str] | None = None,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """``metric_summary`` for every marker on the rows (or ``names``)."""
    return {
        name: metric_summary(rows, f"marker:{name}", n_boot=n_boot, seed=seed)
        for name in (names or marker_names(rows))
    }


# ------------------------------------------------------------------ comparison


def compare_runs(
    a: Sequence[dict],
    b: Sequence[dict],
    *,
    metric: str = "pass_at_1",
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
    min_paired: int = 5,
) -> dict[str, Any]:
    """Is run ``b`` different from run ``a`` on ``metric``?

    Tasks the two runs share are compared as paired differences (b minus
    a, per task); the interval is a bootstrap over those pairs and the
    p-value is a sign-flip permutation test. With fewer than ``min_paired``
    shared tasks the comparison falls back to unpaired task means and says
    so. ``verdict`` is one of ``"b_better"``, ``"a_better"``,
    ``"no_difference_detected"``: the last means the interval covers zero,
    not that the runs are equal.
    """
    ma = task_means(a, metric)
    mb = task_means(b, metric)
    shared = sorted(set(ma) & set(mb))
    paired = len(shared) >= min_paired
    rng = random.Random(seed)
    if paired:
        diffs = [mb[t] - ma[t] for t in shared]
        delta = _mean(diffs)
        ci = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
        # Sign-flip permutation: under H0 each paired difference is
        # equally likely to have either sign.
        observed = abs(delta)
        n = len(diffs)
        extreme = 0
        for _ in range(n_boot):
            flipped = _mean([d if rng.random() < 0.5 else -d for d in diffs])
            if abs(flipped) >= observed - 1e-12:
                extreme += 1
        p_value = (extreme + 1) / (n_boot + 1)
        n_used = n
    else:
        va, vb = list(ma.values()), list(mb.values())
        delta = (_mean(vb) - _mean(va)) if va and vb else float("nan")
        ci = None
        if len(va) >= 3 and len(vb) >= 3:
            boots = []
            for _ in range(n_boot):
                sa = _mean([va[rng.randrange(len(va))] for _ in va])
                sb = _mean([vb[rng.randrange(len(vb))] for _ in vb])
                boots.append(sb - sa)
            boots.sort()
            ci = (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1])
        p_value = None
        n_used = min(len(va), len(vb))
    if ci is None or math.isnan(delta):
        verdict = "insufficient_data"
    elif ci[0] > 0:
        verdict = "b_better"
    elif ci[1] < 0:
        verdict = "a_better"
    else:
        verdict = "no_difference_detected"
    return {
        "metric": metric,
        "paired": paired,
        "n_paired": len(shared),
        "n_only_a": len(set(ma) - set(mb)),
        "n_only_b": len(set(mb) - set(ma)),
        "n_used": n_used,
        "mean_a": _mean(list(ma.values())) if ma else None,
        "mean_b": _mean(list(mb.values())) if mb else None,
        "delta": delta if not math.isnan(delta) else None,
        "ci95": ci,
        "p_value": p_value,
        "verdict": verdict,
        "note": (
            ""
            if paired
            else f"fewer than {min_paired} shared tasks; unpaired task means, weaker test"
        ),
    }


# ------------------------------------------------------------------ decontamination


def _ngrams(words: Sequence[str], n: int) -> set[tuple[str, ...]]:
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _load_rows(source: Any) -> list[dict]:
    """Rows from a list, a JSONL path, or a platform dataset id."""
    if isinstance(source, (str, Path)):
        text = str(source)
        if text.startswith("ds_"):
            from ..ingest.platform import pull

            rows = pull(text)
            return rows if isinstance(rows, list) else []
        from .quality import load_jsonl

        return load_jsonl(text)
    return [r for r in source if isinstance(r, dict)]


def _eval_texts(row: dict) -> list[str]:
    """What an evaluation row contributes: its prompt and its gold answer
    or reference. Not its ``final_text``: on a rollout-shaped eval set
    that is a policy's reply, and tool boilerplate shared between any two
    replies would flag training rows that never saw the eval question
    (rlhf-book ch. 16 decontaminates on prompt overlap)."""
    out = [str(row.get("prompt") or "")]
    for key in ("answer", "reference"):
        if row.get(key):
            out.append(str(row[key]))
    return out


def decontaminate(
    rows: Sequence[dict],
    against: Sequence[Any] | Any,
    *,
    n: int = 8,
    fields: Sequence[str] = ("prompt", "final_text"),
) -> tuple[list[dict], dict[str, Any]]:
    """Drop rows that share any word ``n``-gram with an evaluation set.

    ``against`` is one or more evaluation sources: row lists, JSONL paths,
    or platform dataset ids (``ds_...``). Evaluation prompts, answers and
    references contribute n-grams (not the eval set's own replies); a
    training row is contaminated when any of
    its ``fields`` shares an n-gram, or, for text shorter than ``n`` words,
    matches an evaluation prompt exactly after normalization. Returns the
    clean rows and a report with the first offenders.
    """
    sources = (
        against
        if isinstance(against, (list, tuple)) and not (against and isinstance(against[0], dict))
        else [against]
    )
    eval_ngrams: set[tuple[str, ...]] = set()
    eval_exact: set[str] = set()
    n_eval = 0
    for source in sources:
        for row in _load_rows(source):
            n_eval += 1
            for text in _eval_texts(row):
                words = _words(text)
                eval_ngrams |= _ngrams(words, n)
                if words:
                    eval_exact.add(_norm(text))
    kept: list[dict] = []
    flagged: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        hit: tuple[str, str] | None = None
        for field in fields:
            text = str(row.get(field) or "")
            words = _words(text)
            if not words:
                continue
            if len(words) < n:
                if _norm(text) in eval_exact:
                    hit = (field, "exact")
            else:
                grams = _ngrams(words, n)
                shared = grams & eval_ngrams
                if shared:
                    hit = (field, " ".join(next(iter(shared))))
            if hit:
                break
        if hit:
            flagged.append({"index": i, "field": hit[0], "match": hit[1][:120]})
        else:
            kept.append(row)
    total = sum(1 for r in rows if isinstance(r, dict))
    return kept, {
        "n": total,
        "n_kept": len(kept),
        "n_contaminated": len(flagged),
        "contamination_rate": (len(flagged) / total) if total else 0.0,
        "n_eval_rows": n_eval,
        "ngram": n,
        "fields": list(fields),
        "examples": flagged[:20],
    }


__all__ = [
    "DEFAULT_BOOT",
    "bootstrap_ci",
    "compare_runs",
    "decontaminate",
    "marker_names",
    "marker_summary",
    "metric_summary",
    "task_means",
    "wilson_interval",
]
