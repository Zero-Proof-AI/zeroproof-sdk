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


# ------------------------------------------------------------------ re-run variance


#: Olmo 3's bands for the standard deviation of a benchmark across re-runs
#: of one model, in points on a 0-100 scale: MMLU/MATH/PopQA sit near 0.2,
#: GPQA/AlpacaEval above 1.2. rlhf-book ch. 16, "Why Many External
#: Evaluation Comparisons Are Unreliable", puts most post-training
#: evaluations between 0.25 and 1.5 points with the setup held constant.
VARIANCE_BANDS = (("very_stable", 0.35), ("stable", 0.7), ("high_variance", float("inf")))


def _run_key(row: dict, by: str | None) -> str | None:
    if by:
        value = row.get(by)
        if value is None and isinstance(row.get("lineage"), dict):
            value = row["lineage"].get(by)
        return str(value) if value not in (None, "") else None
    lineage = row.get("lineage")
    if isinstance(lineage, dict) and lineage.get("scoring_run_id"):
        return str(lineage["scoring_run_id"])
    return None


def eval_variance(
    *runs: Sequence[dict],
    metric: str = "pass_at_1",
    by: str | None = None,
) -> dict[str, Any]:
    """How much an evaluation moves when the same model is evaluated
    again (rlhf-book ch. 16, Evaluation).

    Pass each re-run's rows as its own argument, or one row list whose
    rows say which run they belong to: ``lineage.scoring_run_id`` (what
    ``evaluate(run_id=)`` stamps) or a top-level or lineage key named by
    ``by``. Each run's ``metric`` is a mean over tasks; the report is
    those means, their mean, the sample standard deviation ``run_std``,
    and ``noise_band`` = 2 x ``run_std``: a before/after delta inside it
    is what re-running the eval does on its own. Hand ``run_std`` to
    ``delta_report(run_std=)`` and it refuses to call such a delta a
    change. ``stability`` places ``run_std`` on Olmo 3's bands in points.
    Fewer than three runs is a difference, not a distribution; the report
    says so and ``run_std`` is ``None`` below two.
    """
    if not runs:
        raise ValueError("eval_variance needs at least one row list")
    groups: list[list[dict]]
    if len(runs) == 1:
        split: dict[str, list[dict]] = {}
        unkeyed = 0
        for row in runs[0]:
            if not isinstance(row, dict):
                continue
            key = _run_key(row, by)
            if key is None:
                unkeyed += 1
                continue
            split.setdefault(key, []).append(row)
        groups = list(split.values())
        labels = list(split)
    else:
        groups = [list(r) for r in runs]
        labels = [f"run_{i + 1}" for i in range(len(groups))]
        unkeyed = 0
    means: dict[str, float | None] = {}
    task_sets: list[set[str]] = []
    for label, rows in zip(labels, groups):
        per_task = task_means(rows, metric)
        means[label] = round(_mean(list(per_task.values())), 4) if per_task else None
        task_sets.append(set(per_task))
    values = [v for v in means.values() if v is not None]
    n = len(values)
    mean = _mean(values) if values else None
    std = (
        (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
        if n >= 2 and mean is not None
        else None
    )
    common = set.intersection(*task_sets) if task_sets else set()
    stability = None
    if std is not None:
        points = std * 100
        stability = next(name for name, cap in VARIANCE_BANDS if points < cap)
    out: dict[str, Any] = {
        "metric": metric,
        "n_runs": n,
        "means": means,
        "mean": round(mean, 4) if mean is not None else None,
        "run_std": round(std, 4) if std is not None else None,
        "run_std_points": round(std * 100, 2) if std is not None else None,
        "noise_band": round(2 * std, 4) if std is not None else None,
        "stability": stability,
        "tasks_in_every_run": len(common),
        "notes": [],
    }
    if unkeyed:
        out["notes"].append(f"{unkeyed} row(s) carried no run id and were left out")
    if n < 3:
        out["notes"].append(
            f"{n} run(s): two is a difference, not a distribution; three or more re-runs "
            "give a standard deviation worth reading"
        )
    if task_sets and any(s != common for s in task_sets):
        out["notes"].append("runs do not cover the same tasks; means are not strictly comparable")
    return out


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

    Tasks on one side only are dropped from a paired comparison, and
    ``note`` says how many, since a verdict over a quarter of the tasks is
    not a verdict over the eval. ``paired_share`` is the shared fraction
    of every task either run saw.
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
    n_only_a = len(set(ma) - set(mb))
    n_only_b = len(set(mb) - set(ma))
    n_all = len(set(ma) | set(mb))
    paired_share = (len(shared) / n_all) if n_all else None
    if not paired:
        note = f"fewer than {min_paired} shared tasks; unpaired task means, weaker test"
    elif n_only_a or n_only_b:
        note = (
            f"{n_only_a} tasks only in a and {n_only_b} only in b were dropped; "
            f"the verdict rests on the {len(shared)} shared"
        )
        if paired_share is not None and paired_share < 0.5:
            note = "most tasks unpaired: " + note
    else:
        note = ""
    return {
        "metric": metric,
        "paired": paired,
        "n_paired": len(shared),
        "n_only_a": n_only_a,
        "n_only_b": n_only_b,
        "paired_share": paired_share,
        "n_used": n_used,
        "mean_a": _mean(list(ma.values())) if ma else None,
        "mean_b": _mean(list(mb.values())) if mb else None,
        "delta": delta if not math.isnan(delta) else None,
        "ci95": ci,
        "p_value": p_value,
        "verdict": verdict,
        "note": note,
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
    fields: Sequence[str] = ("prompt",),
    overlap: float = 0.8,
) -> tuple[list[dict], dict[str, Any]]:
    """Drop rows whose prompt overlaps an evaluation set (rlhf-book ch. 16).

    ``against`` is one or more evaluation sources: row lists, JSONL paths,
    or platform dataset ids (``ds_...``). Evaluation prompts, answers and
    references are the texts (not the eval set's own replies). A training
    row is contaminated when one of its ``fields`` is an evaluation text
    verbatim (after normalization), or when one evaluation text covers at
    least ``overlap`` of its words with shared word ``n``-grams (the
    Llama 2 rule: 80% of tokens). Texts shorter than ``n`` words match
    verbatim only. The default field is the prompt, the book's method;
    add ``"final_text"`` to ask the stricter question of whether replies
    reproduce eval answers or references.

    One shared n-gram is the book's test for free-form sets. Situations
    written from templates share whole sentences that say nothing about
    which question was asked, so any-n-gram flags every row of a
    template-written set; the coverage rule counts a row when one eval
    text accounts for most of it. ``overlap=0`` restores any-n-gram.
    Returns the clean rows and a report: verbatim hits (``n_exact``) apart
    from near copies (``n_near``), hits per field, the eval text count,
    and the first offenders with their coverage.
    """
    sources = (
        against
        if isinstance(against, (list, tuple)) and not (against and isinstance(against[0], dict))
        else [against]
    )
    texts: dict[str, int] = {}
    n_eval = 0
    index: dict[tuple[str, ...], set[int]] = {}
    for source in sources:
        for row in _load_rows(source):
            n_eval += 1
            for text in _eval_texts(row):
                words = _words(text)
                if not words or _norm(text) in texts:
                    continue
                tid = texts[_norm(text)] = len(texts)
                for gram in _ngrams(words, n):
                    index.setdefault(gram, set()).add(tid)
    threshold = max(0.0, min(1.0, float(overlap)))
    kept: list[dict] = []
    flagged: list[dict[str, Any]] = []
    by_field: dict[str, int] = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        hit: dict[str, Any] | None = None
        for field in fields:
            text = str(row.get(field) or "")
            words = _words(text)
            if not words:
                continue
            if _norm(text) in texts:
                hit = {"field": field, "match": "exact", "coverage": 1.0}
                break
            if len(words) < n:
                continue
            covered: dict[int, set[int]] = {}
            first: dict[int, int] = {}
            for start_i in range(len(words) - n + 1):
                for tid in index.get(tuple(words[start_i : start_i + n]), ()):
                    covered.setdefault(tid, set()).update(range(start_i, start_i + n))
                    first.setdefault(tid, start_i)
            if not covered:
                continue
            best = max(covered, key=lambda t: (len(covered[t]), -t))
            coverage = len(covered[best]) / len(words)
            if threshold <= 0 or coverage >= threshold:
                hit = {
                    "field": field,
                    "match": " ".join(words[first[best] : first[best] + n])[:120],
                    "coverage": round(coverage, 3),
                }
                break
        if hit:
            flagged.append({"index": i, **hit})
            by_field[hit["field"]] = by_field.get(hit["field"], 0) + 1
        else:
            kept.append(row)
    total = sum(1 for r in rows if isinstance(r, dict))
    n_exact = sum(1 for f in flagged if f["match"] == "exact")
    return kept, {
        "n": total,
        "n_kept": len(kept),
        "n_contaminated": len(flagged),
        "contamination_rate": (len(flagged) / total) if total else 0.0,
        # verbatim reuse of an eval text, and near copies under the coverage rule
        "n_exact": n_exact,
        "n_near": len(flagged) - n_exact,
        "n_eval_rows": n_eval,
        "n_eval_texts": len(texts),
        "ngram": n,
        "overlap": threshold,
        "fields": list(fields),
        "by_field": by_field,
        "examples": flagged[:20],
    }


__all__ = [
    "DEFAULT_BOOT",
    "bootstrap_ci",
    "compare_runs",
    "decontaminate",
    "eval_variance",
    "marker_names",
    "marker_summary",
    "metric_summary",
    "task_means",
    "wilson_interval",
]
