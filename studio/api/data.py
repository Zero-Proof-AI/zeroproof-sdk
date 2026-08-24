"""Data tab: runs for one agent, ingest, tags, import/export."""
from __future__ import annotations

import json
import math
import re
import sys
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeroproof_simulations.embeddings import HashEmbedder  # noqa: E402
from zeroproof_simulations.grading import (  # noqa: E402
    conduct_grade,
    display_reason,
    verdict_label,
)

AGENTS = ROOT / "outputs" / "studio-runs" / "agents"
LIVE_AUDIT = ROOT / "outputs" / "zps-live-audit"
HF_TOPUP = ROOT / "outputs" / "hf_topup_1000"
EMBEDDER_NAME = HashEmbedder.name
SEED_AGENTS = ("amazon", "coding", "github", "intercom", "linear")
SEED_CAP = 1000
SEED_MODES = ("rl", "sft", "explore", "unique", "adaptive")
_SECRET_KEYS = ("vllm_key", "api_key", "openai_key", "authorization")
_INGEST_LOCK = threading.Lock()


def _agent_name(raw) -> str:
    text = str(raw or "").strip()
    if "@" in text:
        return ""
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", text)
    s = s.strip("-._")[:64]
    return s or ""


def _norm_tags(raw) -> list[str]:
    if isinstance(raw, str):
        parts = re.split(r"[,;\n]+", raw)
    elif isinstance(raw, list):
        parts = [str(x) for x in raw]
    else:
        parts = []
    out, seen = [], set()
    for part in parts:
        tag = re.sub(r"\s+", " ", part.strip())[:40]
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out[:24]


def _bin(row: dict) -> int | None:
    try:
        r = float(row.get("reward"))
    except (TypeError, ValueError):
        return None
    if r <= 0:
        return 0
    if r >= 1:
        return 1
    return None


def _agent_dir(name: str) -> Path:
    return AGENTS / _agent_name(name)


def _runs_dir(name: str) -> Path:
    return _agent_dir(name) / "runs"


def _harness_path(name: str) -> Path:
    return _agent_dir(name) / "harness.json"


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".run.json")


def _emb_path(path: Path) -> Path:
    return Path(str(path) + ".embeddings.json")


def _find_sidecar(path: Path) -> Path | None:
    for p in (path.with_suffix(".embed.json"), Path(str(path) + ".embeddings.json")):
        if p.is_file():
            return p
    return None


def _mean(nums) -> float | None:
    vals = [x for x in nums if x is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _vec_norm(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(float(x) * float(x) for x in vec)) or 1.0
    return [float(x) / n for x in vec]


def _vec_cos(a: list[float], b: list[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _centroid_novelty(vectors: list) -> tuple[list | None, float | None]:
    clean = [v for v in vectors if isinstance(v, list) and v]
    if not clean:
        return None, None
    dim = len(clean[0])
    if dim <= 0:
        return None, None
    centroid = _vec_norm([sum(v[i] for v in clean) / len(clean) for i in range(dim)])
    nov = []
    for v in vectors:
        if not isinstance(v, list) or len(v) != dim:
            nov.append(None)
        else:
            nov.append(round(1.0 - _vec_cos(v, centroid), 4))
    present = [x for x in nov if x is not None]
    div = round(sum(present) / len(present), 4) if present else None
    return nov, div


def _unique_prompt_vectors(sidecar: dict, rows: list[dict], unique: list[str]):
    vecs = sidecar.get("vectors")
    if not isinstance(vecs, list) or not vecs:
        return None
    row_prompt = sidecar.get("row_prompt")
    first: dict[str, list] = {}
    if isinstance(row_prompt, list) and len(row_prompt) >= min(len(rows), 1):
        for i, row in enumerate(rows):
            p = str(row.get("prompt") or "")
            if p in first:
                continue
            if i >= len(row_prompt):
                continue
            idx = row_prompt[i]
            if isinstance(idx, int) and 0 <= idx < len(vecs):
                first[p] = vecs[idx]
        out = [first.get(p) for p in unique]
        if all(isinstance(v, list) and v for v in out):
            return out
    if len(vecs) == len(rows):
        for i, row in enumerate(rows):
            p = str(row.get("prompt") or "")
            if p not in first:
                first[p] = vecs[i]
        out = [first.get(p) for p in unique]
        if all(isinstance(v, list) and v for v in out):
            return out
    if len(vecs) == len(unique):
        return vecs
    return None


def _retry_spread(rollouts: list[dict]) -> float | None:
    if len(rollouts) < 2:
        return None
    texts = []
    for r in rollouts[:16]:
        t = str(r.get("final_text") or "").strip()
        if not t:
            msgs = r.get("messages") or []
            t = " ".join(str(m.get("content") or "") for m in msgs if isinstance(m, dict) and m.get("role") == "assistant")
        if not t:
            t = str(r.get("reason") or "") + " " + " ".join(r.get("tools") or [])
        texts.append((t or "(empty)")[:500])
    try:
        vecs = HashEmbedder().embed(texts)
    except Exception:
        return None
    _, spread = _centroid_novelty(vecs)
    return spread


def _situation_embeddings(path: Path, rows: list[dict], groups: list[dict]) -> dict:
    sidecar_path = _find_sidecar(path)
    sidecar = _read_json(sidecar_path) if sidecar_path else {}
    unique, seen = [], set()
    for g in groups:
        p = g.get("prompt") or ""
        if p not in seen:
            seen.add(p)
            unique.append(p)
    vecs = _unique_prompt_vectors(sidecar, rows, unique) if sidecar else None
    computed = False
    # List/get_run never compute embeddings or call Modal. Sidecar only.
    nov_list, run_nov = (None, None)
    if vecs:
        nov_list, run_nov = _centroid_novelty(vecs)
    index = {p: i for i, p in enumerate(unique)}
    spreads = []
    for g in groups:
        i = index.get(g.get("prompt") or "")
        g["novelty"] = nov_list[i] if nov_list and i is not None and i < len(nov_list) else None
        spread = None
        g["diversity"] = spread
        if spread is not None:
            spreads.append(spread)
        g["n_conversations"] = g.get("n") or 0
        g["n_pass"] = g.get("n1") or 0
        g["n_fail"] = g.get("n0") or 0
        g["mixed"] = bool(g.get("split"))
    return {
        "embedder": sidecar.get("embedder") or (EMBEDDER_NAME if vecs else None),
        "sidecar": _rel(sidecar_path) if sidecar_path else "",
        "computed": computed,
        "novelty": run_nov,
        "diversity": _mean(spreads),
        "status": "ok" if vecs else "missing",
    }


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_meta(path: Path) -> dict:
    return _read_json(_meta_path(path))


def _write_meta(path: Path, extra: dict) -> dict:
    meta = _read_meta(path)
    meta.update(extra)
    for k in _SECRET_KEYS + ("tools", "system_prompt"):
        meta.pop(k, None)
    if path.is_file():
        meta["mtime"] = int(path.stat().st_mtime)
        meta["path"] = _rel(path)
    meta["tags"] = _norm_tags(meta.get("tags"))
    meta["agent"] = _agent_name(meta.get("agent")) or meta.get("agent") or ""
    _meta_path(path).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _load_harness(name: str) -> dict:
    data = _read_json(_harness_path(name))
    tools = data.get("tools") if isinstance(data.get("tools"), list) else []
    policy = data.get("policy") or data.get("system_prompt") or ""
    return {"tools": tools, "policy": str(policy), "raw": data}


def _declared_tools(name: str) -> set[str] | None:
    names = []
    for schema in _load_harness(name).get("tools") or []:
        if not isinstance(schema, dict):
            continue
        fn = schema.get("function", schema)
        if not isinstance(fn, dict):
            fn = {}
        n = str(fn.get("name") or schema.get("name") or "").strip()
        if n:
            names.append(n)
    return set(names) if names else None


def _grade_row(row: dict, declared: set[str] | None) -> dict:
    if row.get("reward") is not None and row.get("reason"):
        return row
    grade = conduct_grade(row, declared_tools=declared)
    out = dict(row)
    out["reward"] = grade.get("reward")
    out["reason"] = grade.get("reason")
    if grade.get("fault_detected"):
        out["fault_detected"] = True
    return out


def _parse_jsonl(text: str, limit: int = 20000) -> list[dict]:
    rows = []
    blob = (text or "").strip()
    if not blob:
        return []
    if blob[:1] in "[{":
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                parsed = parsed.get("rows") or parsed.get("data") or [parsed]
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        rows.append(item)
                        if len(rows) >= limit:
                            return rows
                return rows
        except json.JSONDecodeError:
            pass
    for line in blob.splitlines():
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


def load_rows(path: Path, limit: int = 4000, grade: bool = False,
              agent: str = "") -> list[dict]:
    if not path.is_file():
        return []
    try:
        text = path.read_text()
    except OSError:
        return []
    rows = _parse_jsonl(text, limit=limit)
    if not grade:
        return rows
    declared = _declared_tools(agent) if agent else None
    return [_grade_row(r, declared) for r in rows]


def _tools_used(row: dict) -> list[str]:
    names = []
    for step in row.get("steps") or []:
        if isinstance(step, dict) and step.get("tool"):
            names.append(str(step["tool"]))
    return names


def _fmt_args(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except TypeError:
        return str(value)


def _messages(row: dict) -> list[dict]:
    out = []
    for m in row.get("messages") or []:
        if not isinstance(m, dict):
            continue
        item = {
            "role": m.get("role") or "",
            "content": str(m.get("content") or ""),
        }
        if m.get("name"):
            item["name"] = m.get("name")
        if m.get("tool_calls"):
            item["tool_calls"] = []
            for call in m["tool_calls"]:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                item["tool_calls"].append({
                    "name": call.get("name") or fn.get("name"),
                    "arguments": call.get("arguments") if "arguments" in call else fn.get("arguments"),
                })
        out.append(item)
    if out:
        return out
    prompt = str(row.get("prompt") or "")
    if prompt:
        out.append({"role": "user", "content": prompt})
    for step in row.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if step.get("user"):
            out.append({"role": "user", "content": str(step["user"])})
        if step.get("text") and not step.get("tool"):
            out.append({"role": "assistant", "content": str(step["text"])})
        if step.get("tool"):
            out.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "name": step.get("tool"),
                    "arguments": step.get("arguments"),
                }],
            })
            out.append({
                "role": "tool",
                "name": step.get("tool"),
                "content": _fmt_args(step.get("result")),
            })
    final = str(row.get("final_text") or "")
    if final and not any(m.get("role") == "assistant" and m.get("content") == final for m in out):
        out.append({"role": "assistant", "content": final})
    return out


def _steps(row: dict) -> list[dict]:
    out = []
    for step in row.get("steps") or []:
        if not isinstance(step, dict):
            continue
        item = {}
        if step.get("tool"):
            item["tool"] = step.get("tool")
            item["arguments"] = step.get("arguments")
            item["result"] = step.get("result")
        if step.get("text"):
            item["text"] = str(step["text"])
        if step.get("user"):
            item["user"] = str(step["user"])
        if item:
            out.append(item)
    return out


def _stats_from_rows(rows: list[dict]) -> dict:
    n = n0 = n1 = 0
    total = 0.0
    by: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        n += 1
        prompt = str(row.get("prompt") or "")
        b = _bin(row)
        if b == 0:
            n0 += 1
            by[prompt][0] += 1
        elif b == 1:
            n1 += 1
            by[prompt][1] += 1
        else:
            by[prompt]
        try:
            total += float(row.get("reward") or 0)
        except (TypeError, ValueError):
            pass
    return {
        "n": n,
        "n0": n0,
        "n1": n1,
        "mean_reward": round(total / n, 4) if n else None,
        "prompts": len(by),
        "n_split": sum(1 for a, b in by.values() if a > 0 and b > 0),
    }


def _stats(path: Path) -> dict:
    if not path.is_file():
        return {"n": 0, "n0": 0, "n1": 0, "mean_reward": None, "prompts": 0, "n_split": 0}
    return _stats_from_rows(load_rows(path, limit=20000))


def summarize(path: Path, agent: str = "") -> dict:
    rows = load_rows(path, limit=4000)
    by = defaultdict(list)
    for i, row in enumerate(rows):
        by[str(row.get("prompt") or "")].append(i)
    n0 = sum(1 for r in rows if _bin(r) == 0)
    n1 = sum(1 for r in rows if _bin(r) == 1)
    convos = []
    for i, row in enumerate(rows):
        tools = _tools_used(row)
        convos.append({
            "i": i,
            "prompt": str(row.get("prompt") or ""),
            "reward": row.get("reward"),
            "bin": _bin(row),
            "verdict": verdict_label(row.get("reward")),
            "reason": str(row.get("reason") or ""),
            "reason_label": display_reason(str(row.get("reason") or "")),
            "first_tool": tools[0] if tools else "",
            "n_tools": len(tools),
            "tools": tools,
            "messages": _messages(row),
            "steps": _steps(row),
            "final_text": str(row.get("final_text") or ""),
            "scenario_id": str(row.get("scenario_id") or ""),
            "fault_detected": bool(row.get("fault_detected")),
        })
    groups = []
    for prompt, idxs in by.items():
        labs = [_bin(rows[i]) for i in idxs]
        n0g = sum(1 for x in labs if x == 0)
        n1g = sum(1 for x in labs if x == 1)
        groups.append({
            "prompt": prompt,
            "n": len(idxs),
            "n0": n0g,
            "n1": n1g,
            "split": n0g > 0 and n1g > 0,
            "rollouts": [convos[i] for i in idxs],
        })
    groups.sort(key=lambda g: (not g["split"], -g["n"], g["prompt"]))
    stats = _stats_from_rows(rows)
    emb = _situation_embeddings(path, rows, groups)
    return {
        "n": len(rows),
        "prompts": len(by),
        "situations": len(groups),
        "conversations": len(rows),
        "n0": n0,
        "n1": n1,
        "n_split": sum(1 for g in groups if g["split"]),
        "mean_reward": stats.get("mean_reward"),
        "groups": groups,
        "rows": convos,
        "agent": _agent_name(agent),
        "embeddings": emb,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    tmp.replace(path)


def _write_embeddings(path: Path, rows: list[dict]) -> dict | None:
    prompts = [str(r.get("prompt") or "") for r in rows]
    unique, index = [], {}
    mapping = []
    for p in prompts:
        if p not in index:
            index[p] = len(unique)
            unique.append(p)
        mapping.append(index[p])
    if not unique:
        return None
    vecs = HashEmbedder().embed(unique)
    payload = {
        "embedder": EMBEDDER_NAME,
        "semantic": False,
        "dim": len(vecs[0]) if vecs else 256,
        "n_rows": len(rows),
        "n_prompts": len(unique),
        "row_prompt": mapping,
        "vectors": [[round(float(x), 5) for x in v] for v in vecs],
    }
    _emb_path(path).write_text(json.dumps(payload) + "\n")
    return {"embedder": EMBEDDER_NAME, "n_prompts": len(unique), "n_rows": len(rows)}


def _strip_secrets(row: dict) -> dict:
    out = dict(row)
    for k in _SECRET_KEYS:
        out.pop(k, None)
    return out


def _mode_from_name(name: str) -> str:
    lower = name.lower()
    for mode in ("adaptive", "explore", "sft", "rl"):
        if mode in lower:
            return mode
    if "unique" in lower:
        return "explore"
    if "long" in lower:
        return "rl"
    return "rl"


def _write_run(agent: str, stem: str, rows: list[dict], *, mode: str,
               tags: list[str], source: str, status: str = "done") -> dict:
    runs_dir = _runs_dir(agent)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{stem}.jsonl"
    _write_jsonl(path, rows)
    stats = _stats_from_rows(rows)
    emb = None
    try:
        emb = _write_embeddings(path, rows)
    except Exception:
        emb = None
    meta = _write_meta(path, {
        "agent": agent,
        "mode": mode,
        "tags": tags,
        "status": status,
        "source": source,
        "embedder": EMBEDDER_NAME if emb else None,
        "n_prompts_embedded": (emb or {}).get("n_prompts"),
        **stats,
    })
    return {
        "id": _rel(path),
        "stem": path.stem,
        "path": _rel(path),
        "agent": agent,
        "n": stats["n"],
        "n0": stats["n0"],
        "n1": stats["n1"],
        "n_split": stats["n_split"],
        "mean_reward": stats["mean_reward"],
        "tags": meta.get("tags") or tags,
        "mode": mode,
        "embedder": EMBEDDER_NAME if emb else None,
    }


def _resolve_run(run_id: str, agent: str | None = None) -> Path | None:
    raw = str(run_id or "").strip()
    if not raw or ".." in raw:
        return None
    if raw.endswith(".run.json"):
        raw = raw[: -len(".run.json")] + ".jsonl"
    candidates: list[Path] = []
    path = Path(raw)
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(ROOT / raw)
        if not raw.endswith(".jsonl"):
            candidates.append(ROOT / f"{raw}.jsonl")
    name = _agent_name(agent)
    stem = Path(raw).name
    if stem.endswith(".jsonl"):
        stem = stem[:-6]
    if name:
        candidates.append(_runs_dir(name) / f"{stem}.jsonl")
        candidates.append(_runs_dir(name) / Path(raw).name)
    if "/" in raw and not raw.startswith("outputs/"):
        parts = raw.split("/")
        if len(parts) >= 2:
            candidates.append(_runs_dir(parts[0]) / f"{parts[-1].removesuffix('.jsonl')}.jsonl")
    if not name:
        if AGENTS.is_dir():
            for folder in AGENTS.iterdir():
                guess = folder / "runs" / f"{stem}.jsonl"
                candidates.append(guess)
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        if not _inside(resolved, AGENTS) and not _inside(resolved, ROOT / "outputs"):
            continue
        return resolved
    return None


def _run_record(path: Path) -> dict:
    st = path.stat()
    meta = _read_meta(path)
    agent = _agent_name(meta.get("agent")) or (path.parent.parent.name if path.parent.name == "runs" else "")
    stats = {
        "n": meta.get("n"),
        "n0": meta.get("n0"),
        "n1": meta.get("n1"),
        "mean_reward": meta.get("mean_reward"),
        "prompts": meta.get("prompts"),
        "n_split": meta.get("n_split"),
    }
    stale = stats["n"] is None
    if stale:
        n = 0
        try:
            with path.open("rb") as fh:
                for line in fh:
                    if line.strip():
                        n += 1
        except OSError:
            n = 0
        stats["n"] = n
        try:
            meta = _write_meta(path, {"n": n})
            stats["n0"] = meta.get("n0")
            stats["n1"] = meta.get("n1")
            stats["mean_reward"] = meta.get("mean_reward")
            stats["prompts"] = meta.get("prompts")
            stats["n_split"] = meta.get("n_split")
        except OSError:
            pass
    rel = _rel(path)
    return {
        "id": rel,
        "stem": path.stem,
        "name": path.stem,
        "agent": agent,
        "mode": meta.get("mode") or _mode_from_name(path.stem),
        "tags": _norm_tags(meta.get("tags")),
        "n": int(stats.get("n") or 0),
        "n0": int(stats.get("n0") or 0),
        "n1": int(stats.get("n1") or 0),
        "mean_reward": stats.get("mean_reward"),
        "prompts": int(stats.get("prompts") or 0),
        "n_split": int(stats.get("n_split") or 0),
        "mtime": int(st.st_mtime),
        "status": meta.get("status") or "done",
        "path": rel,
        "source": meta.get("source") or "",
        "embedder": meta.get("embedder") if _emb_path(path).is_file() else None,
        "has_embeddings": _emb_path(path).is_file(),
        "has_harness": _harness_path(agent).is_file() if agent else False,
    }


def _list_agent_paths(agent: str) -> list[Path]:
    folder = _runs_dir(agent)
    if not folder.is_dir():
        return []
    return sorted(
        [p for p in folder.glob("*.jsonl") if ".progress" not in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def list_runs(agent: str | None = None):
    name = _agent_name(agent)
    ingest = None
    runs = []
    if name:
        for path in _list_agent_paths(name):
            rec = _run_record(path)
            rec["agent"] = rec["agent"] or name
            runs.append(rec)
    elif AGENTS.is_dir():
        for folder in AGENTS.iterdir():
            if not folder.is_dir():
                continue
            for path in _list_agent_paths(folder.name):
                runs.append(_run_record(path))
        runs.sort(key=lambda r: r.get("mtime") or 0, reverse=True)
    totals = {
        "runs": len(runs),
        "rows": sum(r["n"] for r in runs),
        "n0": sum(r["n0"] for r in runs),
        "n1": sum(r["n1"] for r in runs),
        "n_split": sum(r["n_split"] for r in runs),
    }
    out = {
        "agent": name,
        "runs": runs,
        "totals": totals,
        "harness": bool(name and _harness_path(name).is_file()),
        "embedder": EMBEDDER_NAME,
    }
    if ingest:
        out["ingest"] = ingest
    return out


def get_run(run_id: str = "", agent: str = ""):
    path = _resolve_run(run_id, agent or None)
    if not path:
        return {"error": "not found"}
    meta = _read_meta(path)
    name = _agent_name(agent) or _agent_name(meta.get("agent")) or (
        path.parent.parent.name if path.parent.name == "runs" else ""
    )
    data = summarize(path, name)
    data["id"] = _rel(path)
    data["agent"] = name
    data["mode"] = meta.get("mode") or _mode_from_name(path.stem)
    data["tags"] = _norm_tags(meta.get("tags"))
    data["stem"] = path.stem
    data["mtime"] = int(path.stat().st_mtime)
    data["status"] = meta.get("status") or "done"
    data["path"] = _rel(path)
    data["source"] = meta.get("source") or ""
    emb = data.get("embeddings") or {}
    data["has_embeddings"] = bool(emb.get("sidecar")) or _find_sidecar(path) is not None
    data["embedder"] = emb.get("embedder") or (EMBEDDER_NAME if data["has_embeddings"] else None)
    data["has_harness"] = _harness_path(name).is_file() if name else False
    return data


def tag_run(run_id: str = "", tags=None):
    path = _resolve_run(run_id)
    if not path:
        return {"error": "not found"}
    meta = _read_meta(path)
    meta = _write_meta(path, {
        "agent": meta.get("agent") or "",
        "mode": meta.get("mode") or _mode_from_name(path.stem),
        "status": meta.get("status") or "done",
        "tags": _norm_tags(tags),
        **{k: meta.get(k) for k in ("n", "n0", "n1", "mean_reward", "prompts", "n_split", "source", "embedder")},
    })
    return {"id": _rel(path), "tags": meta.get("tags") or [], "stem": path.stem}


def set_tags(body: dict | None = None):
    body = body or {}
    return tag_run(str(body.get("id") or body.get("run_id") or ""), body.get("tags"))


def export_run(run_id: str = ""):
    path = _resolve_run(run_id)
    if not path:
        return {"error": "not found"}
    try:
        jsonl = path.read_text()
    except OSError:
        return {"error": "not found"}
    meta = _read_meta(path)
    return {
        "id": _rel(path),
        "stem": path.stem,
        "filename": f"{path.stem}.jsonl",
        "jsonl": jsonl,
        "agent": _agent_name(meta.get("agent")),
        "mode": meta.get("mode") or "",
        "tags": _norm_tags(meta.get("tags")),
        "n": meta.get("n") or jsonl.count("\n"),
    }


def download_run(run_id: str = ""):
    return export_run(run_id)


def download_agent(agent: str = ""):
    name = _agent_name(agent)
    if not name:
        return {"error": "agent required"}
    path = _harness_path(name)
    if not path.is_file():
        return {"error": "no harness.json for this agent"}
    data = _load_harness(name)
    payload = {
        "name": name,
        "tools": data.get("tools") or [],
        "policy": data.get("policy") or "",
    }
    return {
        "agent": name,
        "filename": f"{name}-harness.json",
        "harness": json.dumps(payload, indent=2) + "\n",
        "tools": payload["tools"],
        "policy": payload["policy"],
    }


def import_run(body: dict | None = None):
    body = body or {}
    name = _agent_name(body.get("agent"))
    if not name:
        return {"error": "agent required"}
    text = body.get("jsonl") or body.get("text") or body.get("content") or ""
    rows = body.get("rows") if isinstance(body.get("rows"), list) else None
    if rows is None:
        rows = _parse_jsonl(str(text), limit=20000)
    else:
        rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return {"error": "no JSONL rows"}
    declared = _declared_tools(name)
    rows = [_strip_secrets(_grade_row(r, declared)) for r in rows]
    filename = str(body.get("filename") or "")
    mode = str(body.get("mode") or _mode_from_name(filename) or "rl").strip().lower()
    if mode not in {"explore", "sft", "rl", "adaptive"}:
        mode = "rl"
    tags = _norm_tags(list(_norm_tags(body.get("tags"))) + ["import"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"{stamp}_{mode}_{uuid.uuid4().hex[:8]}"
    result = _write_run(
        name,
        stem,
        rows[:20000],
        mode=mode,
        tags=tags,
        source=filename or "upload",
    )
    return result


def import_data(body: dict | None = None):
    return import_run(body)


def merge_runs(body: dict | None = None):
    """Concatenate selected batches into a new jsonl. Sources stay on disk."""
    body = body or {}
    name = _agent_name(body.get("agent"))
    if not name:
        return {"error": "agent required"}
    raw_ids = body.get("ids") or body.get("runs") or []
    if not isinstance(raw_ids, list):
        raw_ids = [raw_ids]
    ids = [str(x).strip() for x in raw_ids if str(x).strip()]
    if len(ids) < 2:
        return {"error": "pick at least two batches"}
    folder = _runs_dir(name)
    paths: list[Path] = []
    seen: set[str] = set()
    for rid in ids:
        path = _resolve_run(rid, name)
        if not path or not _inside(path, folder):
            return {"error": "not found"}
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    if len(paths) < 2:
        return {"error": "pick at least two batches"}
    rows: list[dict] = []
    modes: list[str] = []
    stems: list[str] = []
    tags: list[str] = ["merged"]
    for path in paths:
        rec = _run_record(path)
        modes.append(str(rec.get("mode") or "rl"))
        stems.append(path.stem)
        tags.extend(rec.get("tags") or [])
        rows.extend(_strip_secrets(r) for r in load_rows(path, limit=20000, grade=False))
        if len(rows) >= 20000:
            rows = rows[:20000]
            break
    if not rows:
        return {"error": "no JSONL rows"}
    unique_modes = {m for m in modes if m}
    mode = next(iter(unique_modes)) if len(unique_modes) == 1 else (modes[0] if modes else "rl")
    if mode not in {"explore", "sft", "rl", "adaptive"}:
        mode = "rl"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"{stamp}_merged_{uuid.uuid4().hex[:8]}"
    result = _write_run(
        name,
        stem,
        rows,
        mode=mode,
        tags=_norm_tags(tags),
        source="merge:" + ",".join(stems),
    )
    result["sources"] = [_rel(p) for p in paths]
    return result


def _seed_marker(agent: str) -> Path:
    return _runs_dir(agent) / ".seed.json"


def _skip_topup(path: Path) -> bool:
    name = path.name.lower()
    return (
        ".progress" in name
        or "rejected" in name
        or path.suffix != ".jsonl"
        or not path.is_file()
        or path.stat().st_size <= 0
    )


def _prompt_groups(rows: list[dict]) -> list[list[dict]]:
    by: dict[str, list[dict]] = {}
    order: list[str] = []
    for row in rows:
        prompt = str(row.get("prompt") or "")
        if prompt not in by:
            order.append(prompt)
            by[prompt] = []
        by[prompt].append(row)
    groups = [by[p] for p in order]

    def rank(group: list[dict]):
        n0 = sum(1 for r in group if _bin(r) == 0)
        n1 = sum(1 for r in group if _bin(r) == 1)
        split = n0 > 0 and n1 > 0
        grpo = len(group) > 1
        return (not split, not grpo, -len(group))

    groups.sort(key=rank)
    return groups


def _take_groups(groups: list[list[dict]], cap: int) -> tuple[list[dict], list[dict]]:
    taken: list[dict] = []
    rest: list[dict] = []
    if cap <= 0:
        for group in groups:
            rest.extend(group)
        return taken, rest
    for group in groups:
        if len(taken) >= cap:
            rest.extend(group)
            continue
        if len(taken) + len(group) > cap:
            if not taken:
                taken.extend(group[:cap])
                rest.extend(group[cap:])
            else:
                rest.extend(group)
            continue
        taken.extend(group)
    return taken, rest


def _prepare_rows(rows: list[dict], agent: str) -> list[dict]:
    declared = _declared_tools(agent)
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(_strip_secrets(_grade_row(row, declared)))
    return out


def _tags_for(rows: list[dict], *, origin: str, mode: str, extra: list[str] | None = None) -> list[str]:
    stats = _stats_from_rows(rows)
    tags = [origin, mode]
    if extra:
        tags.extend(extra)
    if stats.get("n_split"):
        tags.append("split")
    if mode == "rl" and (stats.get("n_split") or (stats.get("prompts") or 0) < len(rows)):
        tags.append("grpo")
    if any(r.get("fault_detected") for r in rows):
        tags.append("fault")
    return _norm_tags(tags)


def _live_audit_sources(agent: str) -> list[tuple[str, Path]]:
    out = []
    for mode in SEED_MODES:
        path = LIVE_AUDIT / f"{agent}_{mode}.jsonl"
        if path.is_file() and path.stat().st_size > 0:
            out.append((mode, path))
    return out


def _topup_sources(agent: str) -> dict[str, list[Path]]:
    folder = HF_TOPUP / agent
    by: dict[str, list[Path]] = defaultdict(list)
    if not folder.is_dir():
        return by
    for path in sorted(folder.rglob("*.jsonl")):
        if _skip_topup(path):
            continue
        by[_mode_from_name(path.name)].append(path)
    return by


def _load_source(path: Path) -> list[dict]:
    return load_rows(path, limit=20000, grade=False)


def ingest_agent(agent: str, *, force: bool = False) -> dict | None:
    name = _agent_name(agent)
    if name not in SEED_AGENTS:
        return None
    with _INGEST_LOCK:
        runs_dir = _runs_dir(name)
        runs_dir.mkdir(parents=True, exist_ok=True)
        marker = _seed_marker(name)
        existing = [p for p in _list_agent_paths(name) if p.stem.startswith("seed_")]
        if existing and marker.is_file() and not force:
            data = _read_json(marker)
            return data if data else {"n": sum(_stats(p).get("n") or 0 for p in existing)}
        if existing and not force:
            return {
                "n": sum(_stats(p).get("n") or 0 for p in existing),
                "files": [{"stem": p.stem, "n": _stats(p).get("n") or 0, "path": _rel(p)} for p in existing],
                "embedder": EMBEDDER_NAME,
            }

        files = []
        sources: list[str] = []
        budget = SEED_CAP
        written = 0

        for mode, path in _live_audit_sources(name):
            if budget <= 0:
                break
            rows = _prepare_rows(_load_source(path), name)
            groups = _prompt_groups(rows)
            chosen, _ = _take_groups(groups, budget)
            if not chosen:
                continue
            write_mode = "explore" if mode == "unique" else mode
            if write_mode not in {"explore", "sft", "rl", "adaptive"}:
                write_mode = "rl"
            extra = ["unique"] if mode == "unique" else []
            tags = _tags_for(chosen, origin="live-audit", mode=write_mode, extra=extra)
            rec = _write_run(
                name,
                f"seed_{mode}",
                chosen,
                mode=write_mode,
                tags=tags,
                source=_rel(path),
            )
            files.append({**rec, "source": _rel(path)})
            sources.append(_rel(path))
            budget -= len(chosen)
            written += len(chosen)

        by_mode = _topup_sources(name)
        for mode in ("rl", "sft", "explore", "adaptive"):
            paths = by_mode.get(mode) or []
            if not paths or budget <= 0:
                continue
            merged = []
            for path in paths:
                merged.extend(_load_source(path))
            rows = _prepare_rows(merged, name)
            groups = _prompt_groups(rows)
            holdout_extra = []
            if mode == "rl" and budget > 80:
                main_cap = min(budget, max(budget - 150, int(budget * 0.8)))
                chosen, leftover = _take_groups(groups, main_cap)
                holdout_extra = leftover
            else:
                chosen, leftover = _take_groups(groups, budget)
                holdout_extra = leftover if mode == "rl" else []
            if chosen:
                tags = _tags_for(chosen, origin="topup", mode=mode)
                src = "; ".join(_rel(p) for p in paths)
                rec = _write_run(
                    name,
                    f"seed_{mode}",
                    chosen,
                    mode=mode,
                    tags=tags,
                    source=src,
                )
                files.append({**rec, "source": src})
                sources.extend(_rel(p) for p in paths)
                budget -= len(chosen)
                written += len(chosen)
            if holdout_extra and budget > 0:
                hold, _ = _take_groups(_prompt_groups(holdout_extra), min(budget, 200))
                if hold:
                    tags = _norm_tags(["topup", "holdout"])
                    rec = _write_run(
                        name,
                        "seed_holdout",
                        hold,
                        mode="rl",
                        tags=tags,
                        source="; ".join(_rel(p) for p in paths),
                    )
                    files.append({**rec, "source": rec.get("source")})
                    budget -= len(hold)
                    written += len(hold)

        summary = {
            "n": written,
            "files": [
                {
                    "stem": f.get("stem"),
                    "n": f.get("n"),
                    "n0": f.get("n0"),
                    "n1": f.get("n1"),
                    "n_split": f.get("n_split"),
                    "path": f.get("path"),
                    "source": f.get("source"),
                    "tags": f.get("tags"),
                    "mode": f.get("mode"),
                }
                for f in files
            ],
            "sources": sources,
            "embedder": EMBEDDER_NAME,
            "cap": SEED_CAP,
        }
        marker.write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def ensure_ingest(agent: str) -> dict | None:
    return ingest_agent(agent, force=False)


def ingest_all() -> dict:
    out = {}
    for name in SEED_AGENTS:
        out[name] = ingest_agent(name, force=False)
    return out
