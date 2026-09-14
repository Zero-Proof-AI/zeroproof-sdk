"""Did training move the behavior, and did anything else slip?

One call over two graded, marker-scored row sets: the rollouts before a
training run and the rollouts after it, on the same tasks. Every metric
the two sets share (pass@1 and each marker) is compared as paired task
differences with a bootstrap interval (``stats.compare_runs``), so the
answer is "moved by X, interval Y" and not a pair of means.

Markers are read as higher-is-better. A metric named in
``must_not_regress`` whose interval sits entirely below zero is a
regression and fails the report; any other metric that drops
significantly is a warning (rlhf-book ch. 15: post-training on one thing
forgets others, and on-policy data forgets less, which is only visible
if you measure the others).

``proxy`` names the metric the run was trained on (the training reward,
kept on the rows as a marker) when it is not the target. Over-
optimization is the two curves parting (rlhf-book ch. 14): the proxy
moved up while the target did not follow, or the proxy's interval sits
entirely above the target's. The report says so and fails.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .stats import DEFAULT_BOOT, compare_runs, marker_names

GROUP_KEYS = ("delta", "ci95", "verdict", "mean_a", "mean_b", "n_used", "n_paired", "paired")


def _group_of(row: dict, by: str | Callable[[dict], Any]) -> str | None:
    if callable(by):
        value = by(row)
    else:
        value = row.get(by)
        if value is None and isinstance(row.get("markers"), dict):
            value = row["markers"].get(by)
    if value is None or value == "":
        return None
    return str(value)


def _by_group(
    before: Sequence[dict],
    after: Sequence[dict],
    *,
    by: str | Callable[[dict], Any],
    metric: str,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """The target metric compared within each group of rows. A group needs
    rows on both sides; rows with no group value are left out."""
    groups_a: dict[str, list[dict]] = {}
    groups_b: dict[str, list[dict]] = {}
    for row in before:
        g = _group_of(row, by)
        if g is not None:
            groups_a.setdefault(g, []).append(row)
    for row in after:
        g = _group_of(row, by)
        if g is not None:
            groups_b.setdefault(g, []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(sorted(set(groups_a) & set(groups_b))):
        r = compare_runs(
            groups_a[name], groups_b[name], metric=metric, n_boot=n_boot, seed=seed + 100 + i
        )
        slim = {k: r.get(k) for k in GROUP_KEYS}
        slim["rows_a"] = len(groups_a[name])
        slim["rows_b"] = len(groups_b[name])
        out[name] = slim
    return out


def delta_report(
    before: Sequence[dict],
    after: Sequence[dict],
    *,
    target: str | None = None,
    must_not_regress: Sequence[str] = (),
    markers: Sequence[str] | None = None,
    by: str | Callable[[dict], Any] | None = None,
    run_std: float | None = None,
    proxy: str | None = None,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare ``after`` to ``before`` on pass@1 and every shared marker.

    ``target`` names the metric the run was meant to move (``"pass_at_1"``
    or ``"marker:<name>"``); the verdict on it is the headline.
    ``proxy`` names the metric the run was actually trained on (the
    training reward as a marker, e.g. ``"marker:first_action"``). When
    the proxy moved up and the target did not, or the proxy's interval
    sits entirely above the target's, the report is ``over_optimized``
    and fails: the policy learned something the target does not credit
    (rlhf-book ch. 14).
    ``must_not_regress`` lists metrics whose significant drop fails the
    report. Metric names for markers are the marker names; pass@1 is
    ``"pass_at_1"``. Tasks on one side only do not pair; their count is
    ``n_unpaired_tasks`` and, when any were dropped, a warning says so.

    ``by`` splits the target by a group on each row: a row key (top level,
    or a marker name) or a callable ``row -> group``. The report gains
    ``groups``: the target compared within each group, so a headline that
    moved cannot hide a kind of prompt that moved the other way. A group
    whose target dropped significantly is listed in ``groups_down`` and
    warned about; it does not flip ``ok``, which stays the
    ``must_not_regress`` contract (name the group's metric there if it
    should).

    ``run_std`` is the evaluation's own re-run standard deviation
    (``eval_variance(...)["run_std"]``, rlhf-book ch. 16). A metric
    whose delta is smaller than twice it is ``within_noise``: not
    improved, not slipped, not a regression, and a target there reads
    ``within_eval_noise`` rather than moved, because re-running the eval
    moves it that much on its own.
    """
    names = (
        list(markers)
        if markers is not None
        else sorted(set(marker_names(before)) & set(marker_names(after)))
    )
    metrics = ["pass_at_1", *[f"marker:{m}" for m in names]]
    results: dict[str, dict[str, Any]] = {}
    for i, metric in enumerate(metrics):
        results[metric] = compare_runs(before, after, metric=metric, n_boot=n_boot, seed=seed + i)

    def _key(name: str) -> str:
        return name if name == "pass_at_1" or name.startswith("marker:") else f"marker:{name}"

    guarded = {_key(m) for m in must_not_regress}
    noise = 2.0 * float(run_std) if run_std is not None else None
    within_noise: list[str] = []
    for m in metrics:
        r = results[m]
        r["within_noise"] = (
            noise is not None and r.get("delta") is not None and abs(r["delta"]) < noise
        )
        if r["within_noise"]:
            within_noise.append(m)
    loud = {m for m in metrics if not results[m]["within_noise"]}
    regressions = [
        m for m in metrics if m in guarded and m in loud and results[m]["verdict"] == "a_better"
    ]
    slipped = [
        m for m in metrics if m not in guarded and m in loud and results[m]["verdict"] == "a_better"
    ]
    improved = [m for m in metrics if m in loud and results[m]["verdict"] == "b_better"]
    target_key = _key(target) if target else None
    target_result = results.get(target_key) if target_key else None
    if target_result is None and target_key:
        target_verdict = "target_not_measured"
    elif target_result is None:
        target_verdict = None
    else:
        target_verdict = {
            "b_better": "moved",
            "a_better": "moved_the_wrong_way",
            "no_difference_detected": "no_change_detected",
            "insufficient_data": "insufficient_data",
        }[target_result["verdict"]]
        if target_result["within_noise"] and target_verdict in {"moved", "moved_the_wrong_way"}:
            target_verdict = "within_eval_noise"
    ok = not regressions and target_verdict not in {"moved_the_wrong_way"}
    warnings: list[str] = []

    # proxy vs target: the book's over-optimization picture, as a verdict
    proxy_key = _key(proxy) if proxy else None
    proxy_result = results.get(proxy_key) if proxy_key else None
    proxy_verdict: str | None = None
    over_optimized = False
    headline_for_proxy = target_result if target_result else results["pass_at_1"]
    headline_name = target_key if target_result else "pass_at_1"
    if proxy_key and proxy_result is None:
        warnings.append(f"proxy {proxy!r} is not on both row sets")
        proxy_verdict = "proxy_not_measured"
    elif proxy_key and proxy_key == headline_name:
        warnings.append(f"proxy {proxy!r} is the target itself; name the training reward instead")
        proxy_verdict = "proxy_is_target"
    elif proxy_result is not None:
        proxy_verdict = {
            "b_better": "moved",
            "a_better": "moved_the_wrong_way",
            "no_difference_detected": "no_change_detected",
            "insufficient_data": "insufficient_data",
        }[proxy_result["verdict"]]
        proxy_up = proxy_result["verdict"] == "b_better" and proxy_key in loud
        target_up = headline_for_proxy["verdict"] == "b_better" and headline_name in loud
        pci, tci = proxy_result.get("ci95"), headline_for_proxy.get("ci95")
        apart = bool(pci and tci and pci[0] > tci[1])
        over_optimized = (proxy_up and not target_up) or (proxy_up and apart)
        if over_optimized:
            ok = False
            tspan = f"{tci[0]:+.3f}..{tci[1]:+.3f}" if tci else "n/a"
            pspan = f"{pci[0]:+.3f}..{pci[1]:+.3f}" if pci else "n/a"
            warnings.append(
                f"OVER-OPTIMIZED: {proxy_key} up {proxy_result['delta']:+.3f} (95% {pspan}) while "
                f"{headline_name} {headline_for_proxy['delta']:+.3f} (95% {tspan}): the policy "
                "learned something the target does not credit (rlhf-book ch. 14)"
            )
    if noise is not None and target_verdict == "within_eval_noise" and target_result:
        warnings.append(
            f"{target_key}: {target_result['delta']:+.3f} is inside the eval's own re-run band "
            f"(2 x run_std = {noise:.3f}); re-running the eval moves it that much"
        )
    for m in regressions:
        r = results[m]
        warnings.append(
            f"REGRESSION {m}: {r['delta']:+.3f} (95% {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f}), "
            "named in must_not_regress"
        )
    for m in slipped:
        r = results[m]
        warnings.append(
            f"{m} dropped {r['delta']:+.3f} (95% {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f})"
        )
    headline = target_result if target_result else results["pass_at_1"]
    headline_key = target_key if target_result else "pass_at_1"
    if headline.get("note"):
        warnings.append(f"{headline_key}: {headline['note']}")
    if target_verdict == "target_not_measured":
        warnings.append(f"target {target!r} is not on both row sets")
    groups: dict[str, dict[str, Any]] | None = None
    groups_down: list[str] = []
    if by is not None:
        group_metric = target_key if target_key in results else "pass_at_1"
        groups = _by_group(before, after, by=by, metric=group_metric, n_boot=n_boot, seed=seed)
        groups_down = [g for g, r in groups.items() if r.get("verdict") == "a_better"]
        for g in groups_down:
            r = groups[g]
            warnings.append(
                f"{group_metric} moved the wrong way for {g}: {r['delta']:+.3f} "
                f"(95% {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f}, {r['rows_b']} rows)"
            )
        if not groups:
            warnings.append(
                f"by={by if isinstance(by, str) else 'callable'}: no group is on both row sets"
            )
    return {
        "ok": ok,
        "target": target_key,
        "target_verdict": target_verdict,
        "target_delta": target_result["delta"] if target_result else None,
        "target_ci95": target_result["ci95"] if target_result else None,
        "n_paired_tasks": results["pass_at_1"]["n_paired"],
        "n_unpaired_tasks": results["pass_at_1"]["n_only_a"] + results["pass_at_1"]["n_only_b"],
        "improved": improved,
        "regressions": regressions,
        "slipped": slipped,
        "within_noise": within_noise,
        "run_std": float(run_std) if run_std is not None else None,
        "proxy": proxy_key,
        "proxy_verdict": proxy_verdict,
        "proxy_delta": proxy_result["delta"] if proxy_result else None,
        "proxy_ci95": proxy_result["ci95"] if proxy_result else None,
        "over_optimized": over_optimized,
        "metrics": results,
        "warnings": warnings,
        "by": (
            by if isinstance(by, str) else (getattr(by, "__name__", "callable") if by else None)
        ),
        "groups": groups,
        "groups_down": groups_down,
    }


def format_delta_report(report: dict[str, Any]) -> str:
    """The block a person reads: headline, then one line per metric."""
    lines: list[str] = []
    if report.get("target"):
        r = report["metrics"].get(report["target"])
        if r and r.get("delta") is not None and r.get("ci95"):
            lines.append(
                f"{report['target']}: {report['target_verdict']} "
                f"({r['delta']:+.3f}, 95% {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f}, "
                f"{r['n_paired']} paired tasks)"
            )
        else:
            lines.append(f"{report['target']}: {report['target_verdict']}")
    if report.get("proxy"):
        p = report["metrics"].get(report["proxy"])
        if p and p.get("delta") is not None and p.get("ci95"):
            lines.append(
                f"proxy {report['proxy']}: {report['proxy_verdict']} "
                f"({p['delta']:+.3f}, 95% {p['ci95'][0]:+.3f}..{p['ci95'][1]:+.3f})"
                + ("  OVER-OPTIMIZED" if report.get("over_optimized") else "")
            )
        else:
            lines.append(f"proxy {report['proxy']}: {report['proxy_verdict']}")
    lines.append("PASS" if report["ok"] else "FAIL")
    for name, r in report["metrics"].items():
        if r.get("delta") is None:
            lines.append(f"  {name:<28} insufficient data")
            continue
        ci = r.get("ci95")
        span = f"{ci[0]:+.3f}..{ci[1]:+.3f}" if ci else "n/a"
        tag = {
            "b_better": "up",
            "a_better": "DOWN",
            "no_difference_detected": "flat",
            "insufficient_data": "n/a",
        }[r["verdict"]]
        if r.get("within_noise"):
            tag = "noise"
        pair = "paired" if r["paired"] else "unpaired"
        lines.append(
            f"  {name:<28} {r['mean_a']:.3f} -> {r['mean_b']:.3f}  {r['delta']:+.3f} "
            f"[{span}]  {tag}  ({r['n_used']} {pair})"
        )
    groups = report.get("groups")
    if groups:
        lines.append(f"by {report.get('by')}:")
        for name, r in groups.items():
            if r.get("delta") is None:
                lines.append(
                    f"  {name:<28} insufficient data ({r.get('rows_a', 0)}/{r.get('rows_b', 0)} rows)"
                )
                continue
            ci = r.get("ci95")
            span = f"{ci[0]:+.3f}..{ci[1]:+.3f}" if ci else "n/a"
            tag = {
                "b_better": "up",
                "a_better": "DOWN",
                "no_difference_detected": "flat",
                "insufficient_data": "n/a",
            }[r["verdict"]]
            lines.append(
                f"  {name:<28} {r['mean_a']:.3f} -> {r['mean_b']:.3f}  {r['delta']:+.3f} "
                f"[{span}]  {tag}  ({r['rows_a']}/{r['rows_b']} rows)"
            )
    for w in report.get("warnings") or []:
        lines.append(f"! {w}")
    return "\n".join(lines)


__all__ = ["delta_report", "format_delta_report"]
