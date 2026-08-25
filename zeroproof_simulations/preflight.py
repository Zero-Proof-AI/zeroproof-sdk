"""Inspect the agent before spending simulation budget, and report after.

``preflight`` reads an agent spec and reports what will make simulation and
grading harder: tools without result shapes, missing descriptions or
schemas, destructive tools that deserve write-discipline scenarios, thin
policies. It reports; it never fixes.

``dataset_report`` summarizes a generated or graded pool the way a person
would ask for it: rows, usable SFT examples, distinct behaviors, cells,
failure-class mix.

``classify_failure`` maps a graded failing row onto the fixed failure
vocabulary so evaluation output stays diagnosable downstream.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Sequence

FAILURE_CLASSES = (
    "fabrication", "unconfirmed_write", "junk_output", "fault_dishonesty",
    "arithmetic", "no_attempt", "incompleteness",
)

_DESTRUCTIVE = re.compile(
    r"cancel|delete|remove|refund|reverse|transfer|send|update|book|create|"
    r"file_|close|merge|pay", re.IGNORECASE)

_CLASS_HINTS = (
    ("unconfirmed_write", re.compile(
        r"without (?:confirm|verif|eligib|checking|asking)|no (?:confirm|"
        r"verif)|unconfirmed|unauthori[sz]ed|executed .*(?:request|"
        r"immediately)|phantom", re.IGNORECASE)),
    ("junk_output", re.compile(
        r"truncat|cut off|mid-sentence|degenerate|verbatim.?repeat|raw json|"
        r"empty final|garbled", re.IGNORECASE)),
    ("arithmetic", re.compile(
        r"arithmetic|math|miscomput|calculation|rounded? (?:wrong|incorrect)|"
        r"wrong (?:number|result|total)", re.IGNORECASE)),
    ("no_attempt", re.compile(
        r"refused a doable|false capability|claim(?:ed|s) (?:it |to be )?"
        r"(?:cannot|unable)|denies? (?:its|the) (?:own )?(?:capab|tool)",
        re.IGNORECASE)),
    ("fault_dishonesty", re.compile(
        r"claim(?:ed|s) success.*(?:fail|reject|error)|hid(?:es|ing)? the "
        r"(?:error|failure)|misreport|as if it succeeded", re.IGNORECASE)),
    ("incompleteness", re.compile(
        r"incomplete|stonewall|withh(?:e|o)ld|never (?:address|answer|"
        r"present)|refus(?:es|ed|al).*(?:present|serve)|did not (?:finish|"
        r"complete|do)", re.IGNORECASE)),
    ("fabrication", re.compile(
        r"invent|fabricat|halluc|made.?up|adopt(?:ed|s|ing).*(?:claim|fact|"
        r"invented)|ungrounded|dismiss(?:ed|es|ing).*(?:ok|valid|result)|"
        r"sycophan|endors", re.IGNORECASE)),
)


def _fn(tool: dict) -> dict:
    inner = tool.get("function") if isinstance(tool.get("function"), dict) \
        else tool
    return inner if isinstance(inner, dict) else {}


def preflight(tools: Sequence[dict], system_prompt: str = "") -> dict[str, Any]:
    """Spec-quality report for an agent. Report only; nothing is changed.

    ``warnings`` is the list a developer should read before generating
    thousands of rows; ``cells`` is the covering-grid size the same way
    ``recommend`` counts it.
    """
    from .scenarios import scenario_regions
    tools = list(tools or [])
    policy = str(system_prompt or "")
    per_tool: list[dict[str, Any]] = []
    warnings: list[str] = []
    missing_shapes: list[str] = []
    destructive: list[str] = []
    for tool in tools:
        fn = _fn(tool)
        name = str(fn.get("name") or "")
        params = fn.get("parameters") if isinstance(
            fn.get("parameters"), dict) else {}
        issues: list[str] = []
        if not str(fn.get("description") or "").strip():
            issues.append("no_description")
        if not (params.get("properties") or {}):
            issues.append("no_parameters_schema")
        elif not params.get("required"):
            issues.append("no_required_fields")
        if not fn.get("returns"):
            issues.append("no_result_shape")
            missing_shapes.append(name)
        if _DESTRUCTIVE.search(name):
            destructive.append(name)
        per_tool.append({"name": name, "issues": issues})
    if missing_shapes:
        warnings.append(
            f"{len(missing_shapes)} of {len(tools)} tools declare no result "
            f"shape ({', '.join(missing_shapes[:5])}"
            f"{', ...' if len(missing_shapes) > 5 else ''}): grounding is "
            "harder and grounding-style scaffolds can convert fabrication "
            "into refusal instead of correct service")
    for entry in per_tool:
        for issue in entry["issues"]:
            if issue in {"no_description", "no_parameters_schema"}:
                warnings.append(f"tool {entry['name']}: {issue}")
    if destructive:
        warnings.append(
            f"destructive tools present ({', '.join(destructive[:6])}): "
            "write-discipline scenarios (verify-before-write, "
            "confirm-before-destructive) deserve explicit coverage")
    if not policy.strip():
        warnings.append("no system prompt: rule axes of the grid will be "
                        "generic and grading has no policy to hold against")
    elif len(policy) < 200:
        warnings.append(
            f"system prompt is {len(policy)} chars: thin policies give the "
            "grid few rules to test and graders little to enforce")
    named = set()
    for entry in per_tool:
        if entry["name"]:
            named.add(entry["name"].lower())
    mentioned = {m.lower() for m in re.findall(r"[a-z_]{4,}", policy.lower())}
    unreferenced = sorted(named - mentioned)
    cells = len(scenario_regions(tools, policy, mode="sft"))
    return {
        "n_tools": len(tools),
        "policy_chars": len(policy),
        "cells": cells,
        "tools": per_tool,
        "destructive_tools": destructive,
        "missing_result_shapes": missing_shapes,
        "tools_not_mentioned_in_policy": unreferenced,
        "warnings": warnings,
        "ok": not warnings,
    }


def classify_failure(row: dict) -> str | None:
    """Fixed-vocabulary class for a failing row, from its reason and shape.

    Returns None for passing or unlabeled rows and for failures the
    heuristics cannot place (leave those for a person, do not guess).
    """
    if not isinstance(row, dict) or row.get("reward") != 0:
        return None
    existing = str(row.get("failure_class") or "").strip()
    if existing in FAILURE_CLASSES:
        return existing
    text = " ".join(str(row.get(k) or "") for k in
                    ("grader_reason", "reason", "label_reason", "grade_note"))
    final = str(row.get("final_text") or "").strip()
    if not final or final.lower().startswith("<agent error"):
        return "junk_output"
    for name, pattern in _CLASS_HINTS:
        if pattern.search(text):
            return name
    return None


def dataset_report(rows: Sequence[dict], *, tools: Sequence[dict] | None = None,
                   system_prompt: str = "") -> dict[str, Any]:
    """One report a developer reads after simulate/grade: size, signal, mix."""
    from .coverage import cell_key
    from .grading import behavior_signature
    rows = [r for r in rows if isinstance(r, dict)]
    labeled = [r for r in rows if r.get("reward") in (0, 1)]
    passes = [r for r in labeled if r["reward"] == 1]
    fails = [r for r in labeled if r["reward"] == 0]
    prompts = {" ".join(str(r.get("prompt") or "").lower().split())
               for r in rows}
    behaviors = {behavior_signature(r) for r in rows}
    cells = {cell_key(r) for r in rows if r.get("scenario_dimensions")}
    classes = Counter()
    for r in fails:
        classes[classify_failure(r) or "unclassified"] += 1
    junk = sum(1 for r in passes
               if not str(r.get("final_text") or "").strip())
    report: dict[str, Any] = {
        "rows": len(rows),
        "unique_prompts": len(prompts),
        "labeled": len(labeled),
        "passes": len(passes),
        "fails": len(fails),
        "fail_rate": round(len(fails) / len(labeled), 3) if labeled else None,
        "usable_sft": len(passes) - junk,
        "distinct_behaviors": len(behaviors),
        "cells_touched": len(cells),
        "failure_classes": dict(classes.most_common()),
    }
    if tools is not None:
        pre = preflight(tools, system_prompt)
        report["cells_total"] = pre["cells"]
        report["preflight_warnings"] = pre["warnings"]
    return report


def format_dataset_report(report: dict[str, Any]) -> str:
    """The report as the block a person actually reads."""
    lines = [
        f"Generated:            {report.get('rows', 0):>6}",
        f"Unique prompts:       {report.get('unique_prompts', 0):>6}",
        f"Labeled:              {report.get('labeled', 0):>6}",
        f"Usable SFT examples:  {report.get('usable_sft', 0):>6}",
        f"Failures:             {report.get('fails', 0):>6}"
        + (f"  ({report['fail_rate']:.0%})" if report.get("fail_rate")
           is not None else ""),
        f"Distinct behaviors:   {report.get('distinct_behaviors', 0):>6}",
    ]
    if report.get("cells_total") is not None:
        lines.append(f"Cells touched:        {report.get('cells_touched', 0):>6}"
                     f" of {report['cells_total']}")
    classes = report.get("failure_classes") or {}
    if classes:
        lines.append("Top failures:")
        total = sum(classes.values()) or 1
        for name, n in list(classes.items())[:6]:
            lines.append(f"  {name:<20} {n:>4}  ({n / total:.0%})")
    for warning in (report.get("preflight_warnings") or [])[:4]:
        lines.append(f"! {warning}")
    return "\n".join(lines)


__all__ = ["preflight", "dataset_report", "format_dataset_report",
           "classify_failure", "FAILURE_CLASSES"]
