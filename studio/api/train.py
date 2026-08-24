"""Train tab. Build SFT and GRPO packs from an agent's local runs.

SFT is every row with reward 1. GRPO is every prompt that has both a 0 and a 1
across k rollouts of the same ask. Simulation already wrote that group.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from .desk import pack_tags, repeats_line, repeats_mean_text

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs" / "studio-runs"
AUDIT = ROOT / "outputs" / "zps-live-audit"
HF = ROOT / "outputs" / "hf_topup_1000"

_KEEP = (
    "prompt",
    "messages",
    "steps",
    "final_text",
    "scenario_id",
    "reward",
    "reason",
    "world_state",
    "faults",
    "fault_detected",
    "llm_reward",
    "llm_reason",
    "stance",
    "tier",
    "ask_family",
)

_FALLBACK_CAP = 4000


def _agent_name(raw) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(raw or "").strip())
    s = s.strip("-._")[:64]
    return s or ""


def _bin(row: dict) -> int | None:
    if row.get("reward") is None:
        return None
    try:
        r = float(row.get("reward"))
    except (TypeError, ValueError):
        return None
    if r <= 0:
        return 0
    if r >= 1:
        return 1
    return None


def _read_meta(path: Path) -> dict:
    for cand in (path.with_suffix(".run.json"), path.with_suffix(".json"), path.parent / "meta.json"):
        if not cand.is_file() or cand.suffix == path.suffix and cand == path:
            continue
        if cand.name.endswith(".jsonl"):
            continue
        try:
            data = json.loads(cand.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return data
    return {}


def _mode_of(path: Path, meta: dict) -> str:
    mode = str(meta.get("mode") or "").strip().lower()
    if mode:
        return mode
    name = path.name.lower()
    for tag in ("rl", "sft", "explore", "adaptive"):
        if tag in name:
            return tag
    parent = path.parent.name.lower()
    for tag in ("rl", "sft", "explore", "adaptive"):
        if tag in parent:
            return tag
    return ""


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _skip_pack_file(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    if "packs" in parts:
        return True
    n = path.name.lower()
    if n.endswith(".sft.jsonl") or n.endswith(".grpo.jsonl"):
        return True
    if ".progress" in n:
        return True
    if n in {"agents.json"}:
        return True
    return False


def _is_all_run(run_id: str) -> bool:
    raw = str(run_id or "").strip().lower()
    return raw in {"", "all", "*", "all-batches", "all_batches"}


def _match_run(path: Path, run_id: str, file_id: str = "") -> bool:
    """True when this JSONL is the selected batch. All is handled separately."""
    raw = str(run_id or "").strip()
    if _is_all_run(raw):
        return True
    rel = file_id or _rel(path)
    names = {path.stem, path.name, rel, str(path)}
    if raw in names:
        return True
    low = raw.lower()
    if low in {n.lower() for n in names}:
        return True
    if rel.endswith("/" + raw) or rel.endswith("/" + raw + ".jsonl"):
        return True
    return False


def _studio_paths(agent: str) -> list[Path]:
    name = _agent_name(agent)
    if not name:
        return []
    hits: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            path = path.resolve()
        except OSError:
            return
        if path in seen or not path.is_file():
            return
        if path.suffix != ".jsonl" or _skip_pack_file(path):
            return
        seen.add(path)
        hits.append(path)

    agent_dir = RUNS / "agents" / name
    if agent_dir.is_dir():
        for path in sorted(agent_dir.rglob("*.jsonl")):
            add(path)
    named = RUNS / name
    if named.is_dir():
        for path in sorted(named.rglob("*.jsonl")):
            add(path)
    if RUNS.is_dir():
        for path in sorted(RUNS.glob("*.jsonl")):
            meta = _read_meta(path)
            owner = _agent_name(meta.get("agent") or meta.get("name"))
            if owner == name or path.stem.lower().startswith(name.lower()):
                add(path)
    return hits


def _contrastive(path: Path) -> bool:
    n = str(path).lower()
    return any(tag in n for tag in ("_rl", "/rl/", "/rl_", "rl_", "pilot_rl", "adaptive"))


def _fallback_paths(agent: str) -> list[Path]:
    name = _agent_name(agent)
    if not name:
        return []
    hits: list[Path] = []
    if AUDIT.is_dir():
        hits.extend(sorted(AUDIT.glob(f"{name}_*.jsonl")))
        hits.extend(sorted(AUDIT.glob(f"{name}.jsonl")))
    hf = HF / name
    if hf.is_dir():
        for path in sorted(hf.rglob("*.jsonl")):
            low = path.name.lower()
            if "rejected" in low or ".progress" in low:
                continue
            hits.append(path)
    files = [p.resolve() for p in hits if p.is_file()]
    contrast = [p for p in files if _contrastive(p)]
    chosen = contrast or files

    def rank(path: Path) -> tuple[int, str]:
        n = str(path).lower()
        if "_rl" in n or "rl_" in n or "/rl" in n:
            return (0, n)
        if "adaptive" in n:
            return (1, n)
        return (2, n)

    return sorted(set(chosen), key=rank)


def _load_jsonl(path: Path, remaining: int | None = None) -> list[dict]:
    rows: list[dict] = []
    try:
        text = path.read_text()
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        rows.append(row)
        if remaining is not None and len(rows) >= remaining:
            break
    return rows


def _export_row(row: dict) -> dict:
    out: dict = {}
    for key in _KEEP:
        if key not in row or row[key] is None:
            continue
        if key == "world_state" and row[key] in {"unspecified", "unknown"}:
            continue
        if key == "faults" and not row[key]:
            continue
        out[key] = row[key]
    if "prompt" not in out:
        out["prompt"] = str(row.get("prompt") or "")
    if "messages" not in out:
        out["messages"] = row.get("messages") or []
    if "steps" not in out:
        out["steps"] = row.get("steps") or []
    if "final_text" not in out:
        out["final_text"] = str(row.get("final_text") or "")
    return out


def _group_id(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8", "replace")).hexdigest()[:12]


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, default=str, ensure_ascii=False) + "\n" for r in rows)


def _write_pack(agent: str, kind: str, body: str) -> str | None:
    """Persist a pack file. Train GET and the All view must not call this."""
    name = _agent_name(agent)
    if not name or not body:
        return None
    folder = RUNS / "packs" / name
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.{kind}.jsonl"
        path.write_text(body)
        return _rel(path)
    except OSError:
        return None


def _match_run(path: Path, run_id: str) -> bool:
    """True if this JSONL is the selected batch. Empty / All matches every file."""
    raw = str(run_id or "").strip()
    if not raw or raw.lower() in {"all", "*"}:
        return True
    rel = _rel(path)
    name = path.name
    stem = path.stem
    raw_name = Path(raw).name
    raw_stem = Path(raw_name).stem
    return raw in {stem, name, rel, str(path)} or raw_name in {name, stem} or raw_stem == stem


def _is_all_view(run_id: str) -> bool:
    raw = str(run_id or "").strip()
    return (not raw) or raw.lower() in {"all", "*"}


def _empty(agent: str, error: str | None = None, run: str = "") -> dict:
    name = _agent_name(agent)
    all_view = _is_all_view(run)
    payload = {
        "agent": name,
        "run": "" if all_view else str(run or "").strip(),
        "view": "all" if all_view else "run",
        "n": 0,
        "n_all": 0,
        "n0": 0,
        "n1": 0,
        "n_ungraded": 0,
        "prompts": 0,
        "mean_k": 0,
        "mean_k_text": "0",
        "min_k": 0,
        "max_k": 0,
        "repeats_line": repeats_line(0, 0, 0),
        "conversations": [],
        "sft": {
            "n": 0,
            "n_from_split": 0,
            "n_from_saturated": 0,
            "n_from_untestable": 0,
            "filename": f"{name}.sft.jsonl" if name else "sft.jsonl",
            "jsonl": "",
            "path": None,
            "files": [],
        },
        "grpo": {
            "n": 0,
            "n_groups": 0,
            "mean_k": 0,
            "filename": f"{name}.grpo.jsonl" if name else "grpo.jsonl",
            "jsonl": "",
            "path": None,
            "files": [],
            "groups": [],
        },
        "held_out": {
            "n": 0,
            "n_prompts": 0,
            "buckets": [
                {
                    "id": "untestable",
                    "label": "n=1 untestable",
                    "n_prompts": 0,
                    "n_rows": 0,
                    "why": "One rollout of this ask. GRPO needs k copies to compare.",
                    "samples": [],
                },
                {
                    "id": "saturated",
                    "label": "saturated all-1s",
                    "n_prompts": 0,
                    "n_rows": 0,
                    "why": "k copies, every graded copy is 1. No advantage inside the group.",
                    "samples": [],
                },
                {
                    "id": "all_zero",
                    "label": "all zeros",
                    "n_prompts": 0,
                    "n_rows": 0,
                    "why": "k copies, every graded copy is 0. No trajectory to imitate in the group.",
                    "samples": [],
                },
                {
                    "id": "ungraded",
                    "label": "ungraded",
                    "n_prompts": 0,
                    "n_rows": 0,
                    "why": "No 0/1 on this prompt yet. Run Grade, then pack again.",
                    "samples": [],
                },
            ],
        },
        "files": [],
        "source": "studio-runs",
        "note": (
            "A simulation with k rollouts of one prompt is a GRPO group. "
            "Grade each copy 0 or 1. Keep groups that go both ways. "
            "That 0 vs 1 is the advantage. Simulate already wrote the group."
        ),
    }
    if error:
        payload["error"] = error
    return payload


def train_packs(agent_id: str = "", run_id: str = "") -> dict:
    """Build SFT (reward==1) and GRPO (same-ask 0/1 split) packs for one agent.

    All is a view: union of this agent's JSONL rows in memory. It does not write
    a merged file. A specific run_id packs only that batch.
    """
    name = _agent_name(agent_id)
    if not name:
        return _empty("", "agent required", run_id)
    all_view = _is_all_view(run_id)
    want = "" if all_view else str(run_id or "").strip()

    studio = _studio_paths(name)
    source = "studio-runs"
    if studio:
        paths = studio
        cap = None
    else:
        paths = _fallback_paths(name)
        source = "fallback"
        cap = _FALLBACK_CAP
        if not paths:
            return _empty(name, run=want)

    records: list[tuple[dict, str, str, int]] = []
    files_out: list[dict] = []
    remaining = cap
    for path in paths:
        if remaining is not None and remaining <= 0:
            break
        rows = _load_jsonl(path, remaining)
        if remaining is not None:
            remaining -= len(rows)
        if not rows:
            continue
        meta = _read_meta(path)
        mode = _mode_of(path, meta)
        stem = path.stem
        mtime = int(path.stat().st_mtime) if path.is_file() else 0
        n0 = n1 = n_split = 0
        local: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            b = _bin(row)
            if b == 0:
                n0 += 1
            elif b == 1:
                n1 += 1
            local[str(row.get("prompt") or "")].append(i)
        for idxs in local.values():
            labs = [_bin(rows[i]) for i in idxs]
            if any(x == 0 for x in labs) and any(x == 1 for x in labs):
                n_split += 1
        files_out.append({
            "stem": stem,
            "id": _rel(path),
            "mode": mode,
            "n": len(rows),
            "n0": n0,
            "n1": n1,
            "n_split": n_split,
            "mtime": mtime,
        })
        if all_view or _match_run(path, want):
            for row in rows:
                records.append((row, stem, mode, mtime))

    n_all = sum(int(f.get("n") or 0) for f in files_out)
    if not records:
        out = _empty(name, run=want)
        out["source"] = source
        out["files"] = files_out
        out["n_all"] = n_all
        if want and files_out and not any(_match_run(p, want) for p in paths):
            out["error"] = "batch not found"
        return out

    groups: dict[str, list[int]] = defaultdict(list)
    for i, (row, _stem, _mode, _mtime) in enumerate(records):
        groups[str(row.get("prompt") or "")].append(i)

    sft_rows: list[dict] = []
    grpo_rows: list[dict] = []
    grpo_groups: list[dict] = []
    sft_from = {"split": 0, "saturated": 0, "untestable": 0, "other": 0}
    buckets = {
        "untestable": {"n_prompts": 0, "n_rows": 0, "samples": []},
        "saturated": {"n_prompts": 0, "n_rows": 0, "samples": []},
        "all_zero": {"n_prompts": 0, "n_rows": 0, "samples": []},
        "ungraded": {"n_prompts": 0, "n_rows": 0, "samples": []},
    }
    n0 = n1 = n_ungraded = 0
    k_sum = 0
    min_k = 0
    max_k = 0
    conversations: list[dict] = []

    def take_sample(slot: dict, prompt: str) -> None:
        if len(slot["samples"]) >= 4:
            return
        text = " ".join(prompt.split())
        if text:
            slot["samples"].append(text[:180])

    for prompt, idxs in groups.items():
        labs = [_bin(records[i][0]) for i in idxs]
        g0 = sum(1 for x in labs if x == 0)
        g1 = sum(1 for x in labs if x == 1)
        g_none = sum(1 for x in labs if x is None)
        k = len(idxs)
        k_sum += k
        if min_k == 0 or k < min_k:
            min_k = k
        if k > max_k:
            max_k = k
        n0 += g0
        n1 += g1
        n_ungraded += g_none
        stems = sorted({records[i][1] for i in idxs})
        kind = None
        if k == 1:
            kind = "untestable"
        elif g0 > 0 and g1 > 0:
            kind = "split"
        elif g1 > 0 and g0 == 0 and k >= 2:
            kind = "saturated"
        elif g0 > 0 and g1 == 0 and k >= 2:
            kind = "all_zero"
        else:
            kind = "ungraded"

        contrasting = kind == "split"
        preview = " ".join(prompt.split())
        conversations.append({
            "prompt": preview,
            "n": k,
            "n0": g0,
            "n1": g1,
            "contrasting": contrasting,
            "rows": [
                {
                    "bin": lab,
                    "packs": pack_tags(lab, contrasting),
                    "preview": str(records[i][0].get("final_text") or "")[:120],
                }
                for i, lab in zip(idxs, labs)
            ],
        })

        if kind == "split":
            gid = _group_id(prompt)
            for i in idxs:
                row, stem, _mode, _mtime = records[i]
                item = _export_row(row)
                item["group_id"] = gid
                item["k"] = k
                item["n0"] = g0
                item["n1"] = g1
                grpo_rows.append(item)
            grpo_groups.append({
                "prompt": preview,
                "n": k,
                "n0": g0,
                "n1": g1,
                "run": stems[0] if len(stems) == 1 else f"{stems[0]} +{len(stems) - 1}",
                "group_id": gid,
            })
        else:
            slot = buckets[kind]
            slot["n_prompts"] += 1
            slot["n_rows"] += k
            take_sample(slot, prompt)

        for i, lab in zip(idxs, labs):
            if lab != 1:
                continue
            item = _export_row(records[i][0])
            sft_rows.append(item)
            if kind == "split":
                sft_from["split"] += 1
            elif kind == "saturated":
                sft_from["saturated"] += 1
            elif kind == "untestable":
                sft_from["untestable"] += 1
            else:
                sft_from["other"] += 1

    conversations.sort(key=lambda g: (
        0 if g["contrasting"] else 1,
        -(g["n1"] * g["n0"]),
        -g["n"],
        g["prompt"],
    ))
    grpo_groups.sort(key=lambda g: (-(g["n0"] * g["n1"]), -g["n"], g["prompt"]))
    n = len(records)
    n_prompts = len(groups)
    mean_k = round(k_sum / n_prompts, 2) if n_prompts else 0
    held_n = sum(b["n_rows"] for b in buckets.values())
    held_prompts = sum(b["n_prompts"] for b in buckets.values())

    sft_jsonl = _jsonl(sft_rows)
    grpo_jsonl = _jsonl(grpo_rows)
    if all_view:
        sft_name = f"{name}.sft.jsonl"
        grpo_name = f"{name}.grpo.jsonl"
    else:
        tag = Path(want).stem or "batch"
        sft_name = f"{name}.{tag}.sft.jsonl"
        grpo_name = f"{name}.{tag}.grpo.jsonl"

    why = {
        "untestable": "One rollout of this ask. GRPO needs k copies to compare.",
        "saturated": "k copies, every graded copy is 1. No advantage inside the group.",
        "all_zero": "k copies, every graded copy is 0. No trajectory to imitate in the group.",
        "ungraded": "No 0/1 on this prompt yet. Run Grade, then pack again.",
    }
    labels = {
        "untestable": "n=1 untestable",
        "saturated": "saturated all-1s",
        "all_zero": "all zeros",
        "ungraded": "ungraded",
    }

    return {
        "agent": name,
        "run": "" if all_view else want,
        "view": "all" if all_view else "run",
        "n": n,
        "n_all": n_all,
        "n0": n0,
        "n1": n1,
        "n_ungraded": n_ungraded,
        "prompts": n_prompts,
        "mean_k": mean_k,
        "mean_k_text": repeats_mean_text(mean_k),
        "min_k": min_k,
        "max_k": max_k,
        "repeats_line": repeats_line(mean_k, n_prompts, n, min_k, max_k),
        "conversations": conversations[:120],
        "conversations_total": len(conversations),
        "sft": {
            "n": len(sft_rows),
            "n_from_split": sft_from["split"],
            "n_from_saturated": sft_from["saturated"],
            "n_from_untestable": sft_from["untestable"],
            "filename": sft_name,
            "jsonl": sft_jsonl,
            "path": None,
            "files": [f for f in files_out if f["n1"] > 0],
        },
        "grpo": {
            "n": len(grpo_rows),
            "n_groups": len(grpo_groups),
            "mean_k": round(
                (sum(g["n"] for g in grpo_groups) / len(grpo_groups)) if grpo_groups else 0, 2
            ),
            "filename": grpo_name,
            "jsonl": grpo_jsonl,
            "path": None,
            "files": [f for f in files_out if f["n_split"] > 0],
            "groups": grpo_groups[:120],
            "groups_total": len(grpo_groups),
        },
        "held_out": {
            "n": held_n,
            "n_prompts": held_prompts,
            "buckets": [
                {
                    "id": key,
                    "label": labels[key],
                    "n_prompts": buckets[key]["n_prompts"],
                    "n_rows": buckets[key]["n_rows"],
                    "why": why[key],
                    "samples": buckets[key]["samples"],
                }
                for key in ("untestable", "saturated", "all_zero", "ungraded")
            ],
        },
        "files": files_out,
        "source": source,
        "note": (
            "A simulation with k rollouts of one prompt is a GRPO group. "
            "Grade each copy 0 or 1. Keep groups that go both ways. "
            "That 0 vs 1 is the advantage. Simulate already wrote the group."
        ),
    }


def packs(agent: str = "", run: str = "") -> dict:
    """GET /api/train adapter used by studio/serve.py."""
    return train_packs(agent, run)


def download_pack(agent_id: str = "", kind: str = "sft", run: str = "") -> dict:
    data = train_packs(agent_id, run)
    if data.get("error"):
        return data
    key = "grpo" if str(kind).lower() in {"grpo", "rl", "group"} else "sft"
    pack = data.get(key) or {}
    if not pack.get("jsonl"):
        return {"error": "empty pack", "agent": data.get("agent"), "kind": key}
    return {
        "agent": data.get("agent"),
        "kind": key,
        "filename": pack.get("filename"),
        "n": pack.get("n"),
        "jsonl": pack.get("jsonl"),
        "path": pack.get("path"),
    }
