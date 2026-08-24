"""Audit tab: learnability as runnable product checks.

Serve.py today calls audit(agent). run_audit(agent_id, run_id) is the
real entry. Pass a run as the second argument, or as agent::run_id on
the agent string so GET /api/audit?agent= works without a run query.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .desk import repeats_counts, repeats_line, repeats_mean_text  # noqa: E402
from zeroproof_simulations.grading import (  # noqa: E402
    _ACK_FAULT,
    _grounded_blob,
    _invented_reply_refs,
    _step_faulted,
)

STUDIO_AGENTS = ROOT / "outputs" / "studio-runs" / "agents"
LEGACY_RUNS = ROOT / "outputs" / "studio-runs"
LIVE_AUDIT = ROOT / "outputs" / "zps-live-audit"
HF_TOPUP = ROOT / "outputs" / "hf_topup_1000"
ROW_CAP = 4000


def _agent_name(raw) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(raw or "").strip())
    s = s.strip("-._")[:64]
    return s or ""


def _split_agent_run(agent: str, run_id: str | None = None) -> tuple[str, str]:
    agent = str(agent or "").strip()
    run_id = str(run_id or "").strip()
    if "::" in agent:
        left, _, right = agent.partition("::")
        agent = left
        run_id = right or run_id
    return _agent_name(agent), run_id


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _safe_file(rel: str) -> Path | None:
    if not rel:
        return None
    raw = Path(rel)
    path = raw if raw.is_absolute() else (ROOT / rel)
    try:
        path = path.resolve()
    except OSError:
        return None
    if ROOT not in path.parents and path != ROOT:
        return None
    if not path.is_file():
        return None
    return path


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _meta_for(path: Path) -> dict:
    for cand in (
        path.with_suffix(".meta.json"),
        path.with_suffix(".run.json"),
        Path(str(path) + ".meta.json"),
    ):
        if cand.is_file():
            return _read_json(cand)
    return {}


def _tool_names(tools) -> list[str]:
    names = []
    for schema in tools or []:
        if not isinstance(schema, dict):
            continue
        fn = schema.get("function", schema)
        if not isinstance(fn, dict):
            fn = {}
        name = str(fn.get("name") or schema.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _load_harness(agent: str) -> dict:
    name = _agent_name(agent)
    candidates = [
        STUDIO_AGENTS / name / "harness.json",
        STUDIO_AGENTS / name / "spec.json",
        ROOT / "specs" / name / "spec.json",
        ROOT / "tests" / "fixtures" / name / "spec.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        data = _read_json(path)
        tools = data.get("tools") or []
        policy = str(data.get("policy") or data.get("system_prompt") or "").strip()
        if not tools and not policy:
            continue
        return {
            "present": True,
            "path": _rel(path),
            "n_tools": len(_tool_names(tools)),
            "has_policy": bool(policy),
            "has_tools": bool(tools),
        }
    return {
        "present": False,
        "path": None,
        "n_tools": 0,
        "has_policy": False,
        "has_tools": False,
    }


def _jsonl_paths(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return [p for p in folder.glob("*.jsonl") if p.is_file() and ".progress" not in p.name]


def _run_record(path: Path, agent: str, source: str) -> dict:
    meta = _meta_for(path)
    try:
        mtime = int(path.stat().st_mtime)
        size = int(path.stat().st_size)
    except OSError:
        mtime, size = 0, 0
    stem = path.stem
    mode = str(meta.get("mode") or "")
    if not mode:
        low = stem.lower()
        for tag in ("rl", "sft", "explore", "adaptive", "unique"):
            if tag in low:
                mode = "explore" if tag == "unique" else tag
                break
    n = meta.get("n") or meta.get("rows") or (meta.get("coverage") or {}).get("rows")
    try:
        n = int(n) if n is not None else None
    except (TypeError, ValueError):
        n = None
    return {
        "id": _rel(path),
        "stem": stem,
        "name": stem,
        "agent": _agent_name(agent),
        "mode": mode,
        "n": n,
        "mtime": mtime,
        "bytes": size,
        "source": source,
        "path": _rel(path),
    }


def _list_run_records(agent: str) -> list[dict]:
    name = _agent_name(agent)
    if not name:
        return []
    seen: set[str] = set()
    out: list[dict] = []

    def add(path: Path, source: str) -> None:
        if not path.is_file():
            return
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        out.append(_run_record(path, name, source))

    for path in _jsonl_paths(STUDIO_AGENTS / name / "runs"):
        add(path, "studio")
    if LEGACY_RUNS.is_dir():
        for path in _jsonl_paths(LEGACY_RUNS):
            meta = _meta_for(path)
            owner = _agent_name(meta.get("agent"))
            if owner == name or (not owner and name in path.stem):
                add(path, "legacy")
    if LIVE_AUDIT.is_dir():
        for path in sorted(LIVE_AUDIT.glob(f"{name}_*.jsonl")):
            if ".progress" in path.name:
                continue
            add(path, "live-audit")
    for path in _jsonl_paths(HF_TOPUP / name / "runs"):
        add(path, "hf-topup")
    rank = {"rl": 0, "adaptive": 1, "sft": 2, "explore": 3}

    def _key(rec):
        mode = str(rec.get("mode") or "").lower()
        stem = str(rec.get("stem") or "").lower()
        order = rank.get(mode)
        if order is None:
            order = next((v for k, v in rank.items() if k in stem), 4)
            if "unique" in stem:
                order = 5
        return (order, -int(rec.get("mtime") or 0), rec["stem"])

    out.sort(key=_key)
    return out


def _resolve_run(agent: str, run_id: str) -> Path | None:
    if not run_id:
        return None
    direct = _safe_file(run_id)
    if direct:
        return direct
    name = _agent_name(agent)
    wanted = run_id.rstrip("/")
    stem = Path(wanted).stem
    for rec in _list_run_records(name):
        if rec["id"] == wanted or rec["stem"] == wanted or rec["stem"] == stem:
            return _safe_file(rec["id"])
        if rec["path"] == wanted or rec["name"] == wanted:
            return _safe_file(rec["id"])
    return _safe_file(wanted)


def _load_rows(path: Path, limit: int = ROW_CAP) -> list[dict]:
    rows = []
    try:
        handle = path.open()
    except OSError:
        return []
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def _bin(row: dict) -> int | None:
    raw = row.get("reward")
    if raw is None:
        return None
    try:
        r = float(raw)
    except (TypeError, ValueError):
        return None
    if r <= 0:
        return 0
    if r >= 1:
        return 1
    return None


def _has_tool(row: dict) -> bool:
    for step in row.get("steps") or []:
        if isinstance(step, dict) and step.get("tool"):
            return True
    return False


def _invented_id(row: dict) -> bool:
    reason = str(row.get("reason") or "").lower()
    if "invented identifier" in reason:
        return True
    found = _invented_reply_refs(
        str(row.get("final_text") or ""),
        _grounded_blob(row, str(row.get("prompt") or "")),
    )
    return bool(found)


def _ignored_fault(row: dict) -> bool:
    faulted = False
    for step in row.get("steps") or []:
        if isinstance(step, dict) and _step_faulted(step.get("result")):
            faulted = True
            break
    if faulted:
        return not bool(_ACK_FAULT.search(str(row.get("final_text") or "")))
    reason = str(row.get("reason") or "").lower()
    return "ignored tool fault" in reason or "ignored a tool miss" in reason


def _graded(row: dict) -> bool:
    if row.get("reward") is None:
        return False
    return bool(str(row.get("reason") or "").strip())


def _status(pass_ok: bool, warn_ok: bool) -> str:
    if pass_ok:
        return "pass"
    if warn_ok:
        return "warn"
    return "fail"


def _worse(*statuses: str) -> str:
    rank = {"fail": 0, "warn": 1, "pass": 2}
    return min(statuses, key=lambda s: rank.get(s, 0))


def _mix_copy(same: int, split: int, prompts: int) -> tuple[str, str]:
    if prompts <= 0:
        return "No mix: 0 of 0 situations", "Contrast: 0 of 0"
    return (
        f"No mix: {same} of {prompts} situations",
        f"Contrast: {split} of {prompts}",
    )


_WORD = {"pass": "Ready", "warn": "Watch", "fail": "Not ready"}
_PACK_WORD = {"ready": "Ready", "watch": "Watch", "not_ready": "Not ready"}


def _check(cid: str, title: str, status: str, value, why: str, observed: dict | None = None, metric: str = "") -> dict:
    return {
        "id": cid,
        "title": title,
        "status": status,
        "word": _WORD.get(status, status),
        "value": value,
        "why": why,
        "metric": metric or cid,
        "observed": observed or {},
    }


def _pct(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return round(n / d, 4)


def _fmt_pct(n: int, d: int) -> str:
    if d <= 0:
        return "0%"
    return f"{round(100 * n / d):.0f}%"


def _topology(rows: list[dict]) -> dict:
    by: dict[str, list[int]] = defaultdict(list)
    n0 = n1 = n_half = n_tool = n_graded = 0
    n_invented = n_ignored = 0
    n_invented_zero = n_ignored_zero = 0
    rewards = []
    for i, row in enumerate(rows):
        by[str(row.get("prompt") or "")].append(i)
        b = _bin(row)
        if b == 0:
            n0 += 1
        elif b == 1:
            n1 += 1
        elif row.get("reward") is not None:
            n_half += 1
        if _has_tool(row):
            n_tool += 1
        if _graded(row):
            n_graded += 1
        inv = _invented_id(row)
        ign = _ignored_fault(row)
        if inv:
            n_invented += 1
        if ign:
            n_ignored += 1
        if b == 0 and inv:
            n_invented_zero += 1
        if b == 0 and ign:
            n_ignored_zero += 1
        try:
            rewards.append(float(row.get("reward")))
        except (TypeError, ValueError):
            pass
    n = len(rows)
    n_prompts = len(by)
    n2 = 0
    varied = 0
    sizes = [len(idxs) for idxs in by.values()]
    min_k = min(sizes) if sizes else 0
    max_k = max(sizes) if sizes else 0
    for idxs in by.values():
        if len(idxs) < 2:
            continue
        n2 += 1
        labs = {round(float(rows[i].get("reward") or 0), 4) for i in idxs
                if rows[i].get("reward") is not None}
        if len(labs) > 1:
            varied += 1
    const = n_prompts - varied
    mean_r = (sum(rewards) / len(rewards)) if rewards else None
    return {
        "n": n,
        "prompts": n_prompts,
        "n0": n0,
        "n1": n1,
        "n_half": n_half,
        "n_ungraded": n - n_graded,
        "n_graded": n_graded,
        "n_tool": n_tool,
        "n_repeat_prompts": n2,
        "n_split": varied,
        "n_per_prompt": round(n / n_prompts, 2) if n_prompts else 0.0,
        "min_k": min_k,
        "max_k": max_k,
        "mean_reward": round(mean_r, 4) if mean_r is not None else None,
        "saturated": round(const / n_prompts, 3) if n_prompts else 1.0,
        "share0": _pct(n0, n0 + n1),
        "share1": _pct(n1, n0 + n1),
        "tool_rate": _pct(n_tool, n),
        "grade_rate": _pct(n_graded, n),
        "n_invented": n_invented,
        "n_ignored": n_ignored,
        "n_invented_zero": n_invented_zero,
        "n_ignored_zero": n_ignored_zero,
    }


def _build_checks(top: dict, harness: dict) -> list[dict]:
    n = int(top["n"])
    prompts = int(top["prompts"])
    n2 = int(top["n_repeat_prompts"])
    split = int(top["n_split"])
    n0 = int(top["n0"])
    n1 = int(top["n1"])
    binary = n0 + n1
    n_per = float(top["n_per_prompt"] or 0)
    mean_r = top["mean_reward"]
    sat = float(top["saturated"] or 0)
    tool_rate = float(top["tool_rate"] or 0)
    grade_rate = float(top["grade_rate"] or 0)
    inv_z = int(top["n_invented_zero"])
    ign_z = int(top["n_ignored_zero"])
    same = prompts - split

    checks = []

    n_per_st = _status(n_per >= 4, n_per >= 2)
    min_k = int(top.get("min_k") or 0)
    max_k = int(top.get("max_k") or 0)
    checks.append(_check(
        "n_per_prompt",
        "Average repeats per situation",
        n_per_st,
        repeats_counts(n_per, prompts, n, min_k, max_k) if prompts else "0 · 0 situations · 0 conversations",
        "",
        {
            "n": n,
            "prompts": prompts,
            "n_per_prompt": n_per,
            "min_k": min_k,
            "max_k": max_k,
        },
    ))

    split_frac = (split / n2) if n2 else 0.0
    gw_status = _status(split >= 8 and split_frac >= 0.35, split >= 1)
    sat_st = _status(sat <= 0.45, sat <= 0.75)
    mix_title, mix_value = _mix_copy(same, split, prompts)
    checks.append(_check(
        "saturated",
        mix_title,
        _worse(gw_status, sat_st),
        mix_value,
        "",
        {
            "saturated": sat,
            "constant_prompts": same,
            "split": split,
            "prompts": prompts,
            "split_frac": round(split_frac, 3),
        },
    ))

    if mean_r is None:
        mean_status = "fail"
        mean_value = "n/a"
    else:
        mean_status = _status(0.30 <= mean_r <= 0.70, 0.15 <= mean_r <= 0.88)
        mean_value = f"{round(100 * mean_r):.0f}%"
    checks.append(_check(
        "mean_reward",
        "Pass rate",
        mean_status,
        mean_value,
        "",
        {"mean_reward": mean_r, "n1": n1, "n0": n0},
    ))

    if binary <= 0:
        share_status = "fail"
        share_value = "No pass or fail yet"
        share_title = "Pass and fail in the whole batch"
    else:
        minority = min(n0, n1) / binary
        share_status = _status(minority >= 0.20, minority >= 0.10)
        share_value = f"{_fmt_pct(n1, binary)} pass · {_fmt_pct(n0, binary)} fail"
        half = 0.40 <= (n1 / binary) <= 0.60
        share_title = "About half passed, half failed" if half else "Pass and fail both show up"
    checks.append(_check(
        "share_0_1",
        share_title,
        share_status,
        share_value,
        "",
        {"n0": n0, "n1": n1, "binary": binary},
    ))

    tool_st = _status(tool_rate >= 0.70, tool_rate >= 0.40)
    checks.append(_check(
        "tool_rate",
        "Most conversations used a tool",
        tool_st,
        _fmt_pct(int(top["n_tool"]), n) if n else "0%",
        "",
        {"n_tool": int(top["n_tool"]), "n": n, "tool_rate": tool_rate},
    ))

    n_tools = harness.get("n_tools") or 0
    if harness.get("has_tools") and harness.get("has_policy"):
        h_status = "pass"
        h_value = f"{n_tools} tools"
    elif harness.get("present"):
        h_status = "warn"
        h_value = f"{n_tools} tools"
    else:
        h_status = "fail"
        h_value = "None"
    checks.append(_check(
        "harness_present",
        "This agent has a harness",
        h_status,
        h_value,
        "",
        {
            "present": bool(harness.get("present")),
            "path": harness.get("path"),
            "n_tools": n_tools,
            "has_policy": bool(harness.get("has_policy")),
            "has_tools": bool(harness.get("has_tools")),
        },
    ))

    grade_st = _status(grade_rate >= 0.95, grade_rate >= 0.50)
    checks.append(_check(
        "grade_coverage",
        "Every row has a pass/fail",
        grade_st,
        _fmt_pct(int(top["n_graded"]), n) if n else "0%",
        "",
        {
            "n_graded": int(top["n_graded"]),
            "n_ungraded": int(top["n_ungraded"]),
            "n": n,
            "grade_rate": grade_rate,
        },
    ))

    if n0 <= 0:
        leak_status = "warn"
        leak_value = "No fails yet"
    else:
        both = inv_z > 0 and ign_z > 0
        inv_share = inv_z / n0
        ign_share = ign_z / n0
        neither = inv_z == 0 and ign_z == 0
        dominates = inv_share > 0.85 or ign_share > 0.85
        leak_status = _status(both and not dominates, (inv_z + ign_z) > 0 and not neither)
        leak_value = f"{inv_z} invented id · {ign_z} ignored miss"
    checks.append(_check(
        "invented_id_vs_ignored_fault",
        "Fails come from more than one reason",
        leak_status,
        leak_value,
        "",
        {
            "n0": n0,
            "invented_id_zeros": inv_z,
            "ignored_fault_zeros": ign_z,
        },
    ))
    return checks


def _pack_item(status: str, why: str) -> dict:
    return {"status": status, "word": _PACK_WORD[status], "why": why}


def _packs(checks: list[dict], top: dict) -> dict:
    by = {c["id"]: c["status"] for c in checks}
    n1 = int(top.get("n1") or 0)
    grade = by.get("grade_coverage")
    if grade == "fail" or n1 <= 0:
        imitation = _pack_item(
            "not_ready",
            "A good behavior pack needs Passes in this batch. Grade this file, or simulate until some pass.",
        )
    elif grade == "warn":
        imitation = _pack_item(
            "watch",
            "Some conversations in this batch are ungraded. Grade the rest, then download a good behavior pack.",
        )
    else:
        imitation = _pack_item(
            "ready",
            "The conversations that passed. Download these to copy good behavior.",
        )

    mixed_ids = ("n_per_prompt", "saturated")
    if any(by.get(cid) == "fail" for cid in mixed_ids) or any(by.get(cid) == "warn" for cid in mixed_ids):
        if by.get("saturated") == "pass" and by.get("n_per_prompt") != "fail":
            mixed = _pack_item("ready", "")
        else:
            mixed = _pack_item(
                "not_ready" if any(by.get(cid) == "fail" for cid in mixed_ids) else "watch",
                "",
            )
    else:
        mixed = _pack_item(
            "ready",
            "",
        )
    return {"imitation": imitation, "mixed": mixed}


def _headline(packs: dict, top: dict, checks: list[dict] | None = None) -> str:
    n = int(top.get("n") or 0)
    prompts = int(top.get("prompts") or 0)
    n_per = float(top.get("n_per_prompt") or 0)
    min_k = int(top.get("min_k") or 0)
    max_k = int(top.get("max_k") or 0)
    lead = f"{n} conversations · {prompts} situations."
    imit = packs["imitation"]["status"]
    mixed = packs["mixed"]["status"]
    shown = repeats_mean_text(n_per)
    if min_k > 0 and max_k > min_k:
        shown = f"{shown} (range {min_k}–{max_k})"
    if imit == "ready" and mixed == "ready":
        return f"{lead} Both packs are ready to download."
    if imit == "ready" and mixed != "ready":
        if n_per >= 2:
            return (
                f"{lead} Good behavior pack is ready. Contrasting repeats is not: "
                f"average {shown} repeats per situation, but not enough situations "
                "with both a Pass and a Fail."
            )
        return (
            f"{lead} Good behavior pack is ready. Contrasting repeats is not: "
            "each situation was tried once."
        )
    if imit != "ready" and mixed == "ready":
        return f"{lead} Contrasting repeats is ready. Good behavior pack is not."
    return f"{lead} Neither pack is ready yet."


def _verdict(checks: list[dict], top: dict) -> tuple[str, str]:
    ranks = {"fail": 0, "warn": 1, "pass": 2}
    by_id = {c["id"]: c["status"] for c in checks}
    n_fail = sum(1 for c in checks if c["status"] == "fail")
    n_warn = sum(1 for c in checks if c["status"] == "warn")
    worst = min((ranks.get(c["status"], 0) for c in checks), default=0)
    if int(top.get("n") or 0) <= 0:
        return "fail", "Not ready"
    if by_id.get("n_per_prompt") == "fail" or by_id.get("saturated") == "fail":
        return "fail", "Not ready"
    if worst == 0:
        return "fail", "Not ready"
    if n_fail == 0 and n_warn <= 2:
        return "pass", "Ready"
    return "warn", "Watch"


def _empty_payload(agent: str, run_id: str = "", error: str | None = None) -> dict:
    runs = _list_run_records(agent) if agent else []
    payload = {
        "agent": agent or "",
        "run_id": run_id or None,
        "run": None,
        "runs": runs,
        "harness": _load_harness(agent) if agent else {
            "present": False, "path": None, "n_tools": 0,
            "has_policy": False, "has_tools": False,
        },
        "n": 0,
        "prompts": 0,
        "n0": 0,
        "n1": 0,
        "checks": [],
        "n_pass": 0,
        "n_warn": 0,
        "n_fail": 0,
        "n_ready": 0,
        "n_watch": 0,
        "n_not_ready": 0,
        "packs": {
            "imitation": _pack_item("not_ready", "Pick a batch first."),
            "mixed": _pack_item("not_ready", "Pick a batch first."),
        },
        "headline": "",
        "verdict": None,
        "label": None,
    }
    if error:
        payload["error"] = error
    return payload


def run_audit(agent_id, run_id=None) -> dict:
    """Score one agent's selected run against learnability checks."""
    agent, run_id = _split_agent_run(str(agent_id or ""), str(run_id or ""))
    if not agent:
        return _empty_payload("", "", "agent required")
    runs = _list_run_records(agent)
    harness = _load_harness(agent)
    if not run_id:
        return {
            "agent": agent,
            "run_id": None,
            "run": None,
            "runs": runs,
            "harness": harness,
            "n": 0,
            "prompts": 0,
            "n0": 0,
            "n1": 0,
            "checks": [],
            "n_pass": 0,
            "n_warn": 0,
            "n_fail": 0,
            "n_ready": 0,
            "n_watch": 0,
            "n_not_ready": 0,
            "packs": {
                "imitation": _pack_item("not_ready", "Pick a batch first."),
                "mixed": _pack_item("not_ready", "Pick a batch first."),
            },
            "headline": "",
            "verdict": None,
            "label": None,
        }
    path = _resolve_run(agent, run_id)
    if not path:
        payload = _empty_payload(agent, run_id, "not found")
        payload["runs"] = runs
        payload["harness"] = harness
        return payload
    rec = next((r for r in runs if r["id"] == _rel(path) or r["stem"] == path.stem), None)
    rec = rec or _run_record(path, agent, "direct")
    rows = _load_rows(path)
    rec["n"] = len(rows)
    top = _topology(rows)
    checks = _build_checks(top, harness)
    packs = _packs(checks, top)
    verdict, label = _verdict(checks, top)
    n_pass = sum(1 for c in checks if c["status"] == "pass")
    n_warn = sum(1 for c in checks if c["status"] == "warn")
    n_fail = sum(1 for c in checks if c["status"] == "fail")
    return {
        "agent": agent,
        "run_id": rec["id"],
        "run": rec,
        "runs": runs,
        "harness": harness,
        "n": top["n"],
        "prompts": top["prompts"],
        "n0": top["n0"],
        "n1": top["n1"],
        "n_half": top["n_half"],
        "n_split": top["n_split"],
        "n_per_prompt": top["n_per_prompt"],
        "min_k": top.get("min_k") or 0,
        "max_k": top.get("max_k") or 0,
        "repeats_line": repeats_line(
            top["n_per_prompt"], top["prompts"], top["n"],
            top.get("min_k") or 0, top.get("max_k") or 0,
        ),
        "mean_reward": top["mean_reward"],
        "saturated": top["saturated"],
        "tool_rate": top["tool_rate"],
        "grade_rate": top["grade_rate"],
        "checks": checks,
        "packs": packs,
        "headline": _headline(packs, top),
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "n_ready": n_pass,
        "n_watch": n_warn,
        "n_not_ready": n_fail,
        "verdict": verdict,
        "label": label,
        "capped": top["n"] >= ROW_CAP,
    }


def audit(agent: str = "", run_id: str = "") -> dict:
    """HTTP handler. GET /api/audit?agent=  (run via agent::run_id or run_id)."""
    return run_audit(agent, run_id)
