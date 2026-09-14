"""Benchmark a model on a frozen eval across seeds (RLHF book ch. 16).

pass@1 with a task-bootstrap interval (``pass_at(rows).ci95``) captures one
source of noise: which tasks you happened to pick. It does not capture the
other: the model is stochastic, so the same eval re-run with a different
sampling seed gives a different number. A single-run score hides that, and a
before/after delta built on one run each can be sampling noise.

``benchmark_report`` takes several graded runs of the *same* eval, one per
seed, and reports pass@1 mean and standard deviation across the seeds, the
spread, and which tasks are unstable (they pass on some seeds and fail on
others). Run a held-out eval three to five times and read the SD before you
believe a delta.

    runs = [evaluate(roll(seed=s), judge=v).rows for s in range(5)]
    rep = benchmark_report(runs, name="telecom-holdout")
    print(format_benchmark(rep))

Report-only; nothing here calls a model or mutates a row. Pair with
``run_benchmark`` to produce the runs, or bring your own.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from typing import Any

from .passat import pass_at
from .stats import decontaminate


def _task_key(row: dict) -> str:
    return str((row or {}).get("scenario_id") or (row or {}).get("prompt") or "")


def benchmark_report(
    seed_runs: Sequence[Sequence[dict]],
    *,
    name: str = "eval",
    decontaminate_against: Sequence[Any] | None = None,
    unstable_threshold: float = 0.0,
) -> dict[str, Any]:
    """Aggregate N graded runs of one eval into a variance-aware scorecard.

    ``seed_runs`` is a list of row lists, one per seed/re-run of the same
    eval set. Each run is scored with ``pass_at`` for its own pass@1. The
    report carries the mean and population SD across seeds, the min/max, the
    per-task pass fraction over seeds, and the tasks whose pass fraction sits
    strictly between 0 and 1 (they flip seed to seed) — the ones a single run
    would misreport. ``decontaminate_against`` runs an 8-gram overlap of the
    eval prompts against training rows/paths/ids and reports offenders.
    """
    runs = [list(r) for r in seed_runs if r is not None]
    per_seed = []
    for rows in runs:
        pa = pass_at(rows)
        if pa.pass_at_1 is not None:
            per_seed.append(round(pa.pass_at_1, 4))
    n_seeds = len(per_seed)

    # Per-task pass fraction across seeds.
    task_hits: dict[str, list[int]] = {}
    for rows in runs:
        for row in rows:
            r = (row or {}).get("reward")
            if r is None or isinstance(r, bool) or float(r) not in (0.0, 1.0):
                continue
            task_hits.setdefault(_task_key(row), []).append(int(float(r)))
    task_pass = {t: round(sum(v) / len(v), 4) for t, v in task_hits.items() if v}
    unstable = sorted(
        (t for t, p in task_pass.items() if unstable_threshold < p < 1.0 - unstable_threshold),
        key=lambda t: abs(task_pass[t] - 0.5),
    )

    mean = round(statistics.fmean(per_seed), 4) if per_seed else None
    sd = round(statistics.pstdev(per_seed), 4) if n_seeds > 1 else 0.0

    report: dict[str, Any] = {
        "name": name,
        "n_seeds": n_seeds,
        "n_tasks": len(task_pass),
        "pass_at_1_mean": mean,
        "pass_at_1_sd": sd,
        "pass_at_1_min": min(per_seed) if per_seed else None,
        "pass_at_1_max": max(per_seed) if per_seed else None,
        "per_seed": per_seed,
        "n_unstable": len(unstable),
        "unstable_rate": round(len(unstable) / len(task_pass), 4) if task_pass else 0.0,
        "unstable_tasks": unstable[:50],
        "task_pass": task_pass,
        "warnings": [],
    }
    if n_seeds < 3:
        report["warnings"].append(f"only {n_seeds} seed(s); run 3-5 to read run-to-run variance")
    if mean is not None and sd is not None and sd >= 0.05:
        report["warnings"].append(
            f"high run-to-run SD {sd}; a delta below ~{round(2 * sd, 3)} is noise"
        )
    if decontaminate_against is not None:
        flat = [row for rows in runs for row in rows]
        _, decon = decontaminate(flat, against=list(decontaminate_against))
        report["decontamination"] = {
            "checked": decon.get("n"),
            "contaminated": decon.get("n_contaminated"),
            "rate": decon.get("contamination_rate"),
            "examples": [e.get("match") for e in (decon.get("examples") or [])[:5]],
        }
        if decon.get("n_contaminated"):
            report["warnings"].append(
                f"{decon['n_contaminated']} eval rows overlap the training set (8-gram)"
            )
    return report


def run_benchmark(
    eval_set: Sequence[Any],
    *,
    judge: Callable[[dict], Any],
    rollout: Callable[[int], Sequence[dict]],
    seeds: int = 5,
    name: str = "eval",
    decontaminate_against: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Drive ``rollout(seed)`` for each seed, grade with ``judge``, aggregate.

    ``rollout`` is anything that returns the eval's rollouts for a seed (a
    ``simulate``/``evaluate`` closure over your frozen ``eval_set``); this
    function only owns the seed loop, the grading, and the aggregation, so it
    stays model-agnostic. Returns the same report as ``benchmark_report``.
    """
    from .judging import run_judge

    runs = []
    for seed in range(seeds):
        rows = list(rollout(seed))
        runs.append(run_judge(rows, judge, source="eval").rows)
    return benchmark_report(runs, name=name, decontaminate_against=decontaminate_against)


def format_benchmark(report: dict[str, Any]) -> str:
    r = report
    mean = r["pass_at_1_mean"]
    head = (
        f"{r['name']}: pass@1 {mean:.3f} +/- {r['pass_at_1_sd']:.3f} "
        f"over {r['n_seeds']} seeds "
        f"[{r['pass_at_1_min']}, {r['pass_at_1_max']}]"
        if mean is not None
        else f"{r['name']}: no graded seeds"
    )
    lines = [head, f"  tasks {r['n_tasks']}, unstable {r['n_unstable']} ({r['unstable_rate']:.0%})"]
    for w in r.get("warnings", []):
        lines.append(f"  ! {w}")
    return "\n".join(lines)
