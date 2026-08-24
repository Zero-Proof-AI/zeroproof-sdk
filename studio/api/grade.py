"""Grade tab API. LLM 0/1 judge on an existing run. Not simulate.

Exports grade_run and features_for_run. serve.py also calls features / grade_now.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeroproof_simulations import conversation  # noqa: E402
from zeroproof_simulations.embeddings import HashEmbedder  # noqa: E402
from zeroproof_simulations.grade_llm import (  # noqa: E402
    JUDGE_SYSTEM,
    apply_grade_llm,
    hosted_judge_endpoint,
    judge_spec,
)
from zeroproof_simulations.grading import (  # noqa: E402
    NO_FAULT,
    normalize_fault_name,
    trace_fault,
)

RUNS = ROOT / "outputs" / "studio-runs"
AGENTS = RUNS / "agents"
EMBEDDER_NAME = HashEmbedder.name
ROW_CAP = 4000
TABLE_CAP = 500
VECTOR_CAP = 800

__all__ = ["grade_run", "features_for_run", "features", "grade_now"]


def _agent_name(raw) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(raw or "").strip())
    s = s.strip("-._")[:64]
    return s or ""


def _split_agent(raw: str) -> tuple[str, str]:
    s = str(raw or "").strip()
    if "|" in s:
        left, _, right = s.partition("|")
        return _agent_name(left), right.strip()
    return _agent_name(s), ""


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _safe_file(rel: str) -> Path | None:
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = (ROOT / rel).resolve()
    else:
        path = path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        return None
    if path.is_file():
        return path
    return None


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


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _harness_for(agent: str) -> tuple[dict, str]:
    name = _agent_name(agent)
    candidates = [
        (AGENTS / name / "harness.json", "harness.json"),
        (AGENTS / name / "spec.json", "spec.json"),
        (ROOT / "specs" / name / "spec.json", "specs"),
        (ROOT / "tests" / "fixtures" / name / "spec.json", "fixtures"),
    ]
    for path, source in candidates:
        data = _read_json(path)
        if (data.get("tools") or data.get("policy") or data.get("system_prompt")
                or data.get("potential_faults") or data.get("tool_condition")
                or data.get("tool_conditions")):
            return data, source
    return {}, "none"


def _agent_spec_blob(agent: str) -> dict:
    """First-wins merge of spec fields used for potential faults."""
    name = _agent_name(agent)
    combined: dict = {}
    for path in (
        AGENTS / name / "harness.json",
        AGENTS / name / "spec.json",
        AGENTS / name / "metadata.json",
        ROOT / "specs" / name / "spec.json",
        ROOT / "tests" / "fixtures" / name / "spec.json",
    ):
        data = _read_json(path)
        if not data:
            continue
        for key in ("potential_faults", "tool_condition", "tool_conditions",
                    "faults", "world_state", "world_states", "dimensions"):
            if key in data and data[key] not in (None, "", [], {}) and key not in combined:
                combined[key] = data[key]
    return combined


def _declared_tools(agent: str) -> tuple[set[str] | None, list[str], str]:
    data, source = _harness_for(agent)
    names = _tool_names(data.get("tools") or [])
    if names:
        return set(names), names, source
    return None, [], source


def _read_meta(path: Path) -> dict:
    for candidate in (path.with_suffix(".run.json"), path.with_name(path.stem + ".meta.json")):
        data = _read_json(candidate)
        if data:
            return data
    return {}


def _write_meta(path: Path, extra: dict) -> dict:
    meta = _read_meta(path)
    meta.update(extra)
    for key in ("vllm_key", "api_key", "judge_key", "openai_key", "VLLM_API_KEY",
                "tools", "system_prompt"):
        meta.pop(key, None)
    dest = path.with_suffix(".run.json")
    dest.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _load_rows(path: Path, limit: int = ROW_CAP) -> list[dict]:
    rows = []
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
        if isinstance(row, dict):
            rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    tmp.replace(path)


def _bin(row: dict):
    """Pass/Fail from the LLM judge only. Ignores leftover conduct ``reward``."""
    if row.get("qwen_reward") is None:
        return None
    try:
        r = float(row.get("qwen_reward"))
    except (TypeError, ValueError):
        return None
    if r == 0.0:
        return 0
    if r == 1.0:
        return 1
    return None


def _tools_used(row: dict) -> list[str]:
    names = []
    for step in row.get("steps") or []:
        if isinstance(step, dict) and step.get("tool"):
            names.append(str(step["tool"]))
    if names:
        return names
    for msg in row.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        for call in msg.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("name"):
                names.append(str(call["name"]))
    return names


def _reason_family(reason: str) -> str:
    label = str(reason or "").strip()
    return label.split(":")[0].strip()[:80] if label else "Ungraded"


def _as_name_list(value) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [p.strip() for p in re.split(r"[,;\n]+", value) if p.strip()]
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                name = str(item.get("mode") or item.get("name")
                           or item.get("status") or "").strip()
                if name:
                    out.append(name)
        return out
    return []


def _declared_faults(spec: dict, meta: dict) -> list[str]:
    """Fault types listed on the agent spec or run metadata, in that order.

    Spec fields, first match wins: ``potential_faults``, ``tool_condition``
    (or ``tool_conditions``), ``dimensions.tool_condition``, ``faults`` as a
    list of names, and ``world_state`` / ``world_states`` (entity missing
    becomes not_found, already acted on becomes already_done). The same keys
    on the run ``.run.json`` also count. ``success`` is not a fault chip.
    """
    found: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        name = normalize_fault_name(raw)
        if not name or name == NO_FAULT or name in seen:
            return
        seen.add(name)
        found.append(name)

    for blob in (spec, meta):
        if not isinstance(blob, dict):
            continue
        for key in ("potential_faults", "tool_condition", "tool_conditions"):
            for item in _as_name_list(blob.get(key)):
                add(item)
        dims = blob.get("dimensions")
        if isinstance(dims, dict):
            for item in _as_name_list(
                    dims.get("tool_condition") or dims.get("tool_conditions")):
                add(item)
        faults = blob.get("faults")
        if isinstance(faults, list):
            for item in _as_name_list(faults):
                add(item)
        for key in ("world_state", "world_states"):
            for item in _as_name_list(blob.get(key)):
                add(item)
    return found


def _fault_subtitle(reasons: list[str], n: int) -> str:
    """Most common conduct reason in this fault group, if it is actually common."""
    if n <= 0 or not reasons:
        return ""
    tallies: dict[str, int] = {}
    for reason in reasons:
        label = str(reason or "").strip()
        family = label.split(":")[0].strip()[:80] if label else ""
        if not family or family in {"Looks fine", "Ungraded", "graded"}:
            continue
        tallies[family] = tallies.get(family, 0) + 1
    if not tallies:
        return ""
    label, count = max(tallies.items(), key=lambda kv: (kv[1], kv[0]))
    if count < 1:
        return ""
    if n >= 2 and count * 2 <= n:
        return ""
    return label


def _empty_chip(name: str, *, potential: bool = False) -> dict:
    return {
        "name": name,
        "n": 0,
        "n0": 0,
        "n1": 0,
        "n_half": 0,
        "subtitle": "",
        "potential": potential,
    }


def _build_fault_chips(rows: list[dict], declared: list[str]) -> list[dict]:
    buckets: dict[str, dict] = {}
    reasons: dict[str, list[str]] = defaultdict(list)
    declared_set = set(declared)
    for row in rows:
        name = trace_fault(row)
        slot = buckets.setdefault(
            name, _empty_chip(name, potential=name in declared_set and name != NO_FAULT))
        _tally(slot, _bin(row))
        reason = str(row.get("reason") or "")
        if reason:
            reasons[name].append(reason)
    for name in declared:
        buckets.setdefault(name, _empty_chip(name, potential=True))["potential"] = True
    for name, slot in buckets.items():
        slot["subtitle"] = _fault_subtitle(reasons.get(name) or [], slot["n"])
        if name != NO_FAULT and name in declared_set:
            slot["potential"] = True
    out: list[dict] = []
    seen: set[str] = set()
    for name in declared:
        if name in buckets and name not in seen:
            out.append(buckets[name])
            seen.add(name)
    extra = [slot for name, slot in buckets.items()
             if name not in seen and name != NO_FAULT]
    extra.sort(key=lambda item: (-item["n"], item["name"]))
    out.extend(extra)
    if NO_FAULT in buckets and NO_FAULT not in seen:
        out.append(buckets[NO_FAULT])
    return out


def _messages(row: dict) -> list[dict]:
    raw = row.get("messages")
    if not isinstance(raw, list) or not raw:
        raw = conversation(row)
    out = []
    for msg in raw[:16]:
        if not isinstance(msg, dict):
            continue
        item = {
            "role": str(msg.get("role") or ""),
            "content": str(msg.get("content") or "")[:1200],
        }
        if msg.get("name"):
            item["name"] = str(msg["name"])
        calls = msg.get("tool_calls") or []
        if isinstance(calls, list) and calls:
            packed = []
            for call in calls[:8]:
                if not isinstance(call, dict):
                    continue
                args = call.get("arguments")
                if isinstance(args, (dict, list)):
                    blob = json.dumps(args, default=str)
                else:
                    blob = str(args or "")
                packed.append({"name": call.get("name"), "arguments": blob[:600]})
            item["tool_calls"] = packed
        out.append(item)
    return out


def _step_glimpse(row: dict) -> list[dict]:
    out = []
    for step in row.get("steps") or []:
        if not isinstance(step, dict) or not step.get("tool"):
            continue
        result = step.get("result")
        status = ""
        if isinstance(result, dict):
            status = str(result.get("status") or "")
        out.append({"tool": str(step.get("tool")), "status": status})
        if len(out) >= 12:
            break
    return out


def _stats(rows: list[dict]) -> dict:
    n0 = n1 = half = ungraded = 0
    total = 0.0
    scored = 0
    by: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        b = _bin(row)
        prompt = str(row.get("prompt") or "")
        if b == 0:
            n0 += 1
            by[prompt][0] += 1
        elif b == 1:
            n1 += 1
            by[prompt][1] += 1
        elif b == 0.5:
            half += 1
            by[prompt]
        else:
            ungraded += 1
            by[prompt]
        if row.get("reward") is not None:
            try:
                total += float(row.get("reward") or 0)
                scored += 1
            except (TypeError, ValueError):
                pass
    return {
        "n": len(rows),
        "n0": n0,
        "n1": n1,
        "n_half": half,
        "ungraded": ungraded,
        "mean_reward": round(total / scored, 4) if scored else None,
        "prompts": len(by),
        "n_split": sum(1 for a, b in by.values() if a > 0 and b > 0),
    }


def _list_run_paths(agent: str) -> list[tuple[Path, str, str]]:
    name = _agent_name(agent)
    found: dict[str, tuple[Path, str, str]] = {}

    def add(path: Path) -> None:
        if not path.is_file() or path.suffix != ".jsonl":
            return
        rel = _rel(path)
        found[rel] = (path, rel, path.stem)

    folder = AGENTS / name
    if folder.is_dir():
        runs_dir = folder / "runs"
        if runs_dir.is_dir():
            for path in runs_dir.glob("*.jsonl"):
                add(path)
        for path in folder.glob("*.jsonl"):
            add(path)
    if RUNS.is_dir():
        for path in RUNS.glob("*.jsonl"):
            meta = _read_meta(path)
            owner = _agent_name(meta.get("agent") or path.stem.split("_")[0])
            if owner == name or (not meta and not name):
                add(path)
    rows = list(found.values())
    rows.sort(key=lambda item: item[0].stat().st_mtime if item[0].is_file() else 0, reverse=True)
    return rows


def _unspecified_batch(run_id: str | None) -> bool:
    """True when nothing is selected, or All / latest. Not a missing batch."""
    raw = str(run_id or "").strip().lower()
    return raw in {"", "latest", "all", "*", "all-batches", "all_batches"}


def _resolve_run(agent: str, run_id: str | None) -> tuple[Path | None, list[tuple[Path, str, str]]]:
    listed = _list_run_paths(agent)
    if not listed:
        return None, listed
    want = str(run_id or "").strip()
    if _unspecified_batch(want):
        return listed[0][0], listed
    for path, rel, stem in listed:
        if want in {rel, stem, path.name, str(path)}:
            return path, listed
        if want.endswith("/" + path.name) or want.endswith("/" + stem):
            return path, listed
    hit = _safe_file(want)
    if hit:
        return hit, listed
    return None, listed


def _run_summary(path: Path, rel: str, stem: str) -> dict:
    meta = _read_meta(path)
    st = path.stat()
    n = meta.get("n")
    if n is None:
        stats = _stats(_load_rows(path, limit=ROW_CAP))
    else:
        stats = {
            "n": int(meta.get("n") or 0),
            "n0": int(meta.get("n0") or 0),
            "n1": int(meta.get("n1") or 0),
            "n_half": int(meta.get("n_half") or 0),
            "ungraded": int(meta.get("ungraded") or 0),
            "mean_reward": meta.get("mean_reward"),
            "prompts": int(meta.get("prompts") or 0),
            "n_split": int(meta.get("n_split") or 0),
        }
    sidecar = path.with_suffix(".embed.json")
    return {
        "id": rel,
        "stem": stem,
        "name": stem,
        "mode": meta.get("mode") or "",
        "n": stats["n"],
        "n0": stats["n0"],
        "n1": stats["n1"],
        "n_half": stats.get("n_half") or 0,
        "ungraded": stats.get("ungraded") or 0,
        "mean_reward": stats.get("mean_reward"),
        "prompts": stats.get("prompts") or 0,
        "n_split": stats.get("n_split") or 0,
        "mtime": int(st.st_mtime),
        "embedder": meta.get("embedder") or (EMBEDDER_NAME if sidecar.is_file() else ""),
        "sidecar": _rel(sidecar) if sidecar.is_file() else "",
        "path": rel,
    }


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _embed_texts(texts: list[str]) -> tuple[list[list[float]], dict]:
    embedder = HashEmbedder()
    vectors = embedder.embed(texts) if texts else []
    n = len(vectors)
    if not n:
        return [], {
            "embedder": EMBEDDER_NAME,
            "semantic": False,
            "dim": 256,
            "n": 0,
            "diversity": 1.0,
            "mean_novelty": 1.0,
            "novelty": [],
        }
    dim = len(vectors[0])
    centroid = _normalize([sum(v[i] for v in vectors) / n for i in range(dim)])
    novelty = [round(1.0 - _cos(v, centroid), 4) for v in vectors]
    diversity = round(sum(novelty) / n, 4)
    return vectors, {
        "embedder": EMBEDDER_NAME,
        "semantic": False,
        "dim": dim,
        "n": n,
        "diversity": diversity,
        "mean_novelty": diversity,
        "novelty": novelty,
    }


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".embed.json")


def _store_embeddings(path: Path, rows: list[dict]) -> dict:
    texts = []
    for row in rows:
        text = str(row.get("prompt") or "").strip()
        if not text:
            text = str(row.get("final_text") or "")[:240]
        texts.append(text)
    vectors, info = _embed_texts(texts)
    payload = {
        "embedder": EMBEDDER_NAME,
        "semantic": False,
        "dim": info.get("dim") or 256,
        "n": info["n"],
        "diversity": info["diversity"],
        "mean_novelty": info["mean_novelty"],
        "novelty": info["novelty"],
    }
    if vectors and len(vectors) <= VECTOR_CAP:
        payload["vectors"] = [[round(x, 5) for x in v] for v in vectors]
    dest = _sidecar_path(path)
    dest.write_text(json.dumps(payload) + "\n")
    return {
        "embedder": EMBEDDER_NAME,
        "sidecar": _rel(dest),
        "diversity": info["diversity"],
        "mean_novelty": info["mean_novelty"],
        "n": info["n"],
        "stored_vectors": bool(payload.get("vectors")),
        "novelty": info["novelty"],
    }


def _load_embeddings(path: Path, rows: list[dict]) -> dict:
    dest = _sidecar_path(path)
    data = _read_json(dest)
    if data.get("embedder") == EMBEDDER_NAME and len(data.get("novelty") or []) == len(rows):
        return {
            "embedder": EMBEDDER_NAME,
            "sidecar": _rel(dest),
            "diversity": data.get("diversity"),
            "mean_novelty": data.get("mean_novelty"),
            "n": data.get("n") or len(rows),
            "stored_vectors": bool(data.get("vectors")),
            "novelty": list(data.get("novelty") or []),
        }
    if not rows:
        return {
            "embedder": EMBEDDER_NAME,
            "sidecar": "",
            "diversity": None,
            "mean_novelty": None,
            "n": 0,
            "stored_vectors": False,
            "novelty": [],
        }
    return _store_embeddings(path, rows)


def _tally(slot: dict, b) -> None:
    slot["n"] += 1
    if b == 0:
        slot["n0"] += 1
    elif b == 1:
        slot["n1"] += 1
    elif b == 0.5:
        slot["n_half"] += 1


def _empty_features(agent: str, runs: list[dict], extra: dict | None = None) -> dict:
    payload = {
        "agent": _agent_name(agent),
        "run_id": "",
        "stem": "",
        "mode": "",
        "harness": False,
        "declared_source": "none",
        "declared_tools": [],
        "n": 0,
        "n0": 0,
        "n1": 0,
        "n_half": 0,
        "ungraded": 0,
        "mean_reward": None,
        "prompts": 0,
        "n_split": 0,
        "families": [],
        "faults": [],
        "declared_faults": [],
        "first_tool": [],
        "within_prompt": {
            "split_prompts": 0,
            "groups_k": 0,
            "first_tool_differs": 0,
            "zeros": [],
        },
        "between_prompt": {
            "prompts": 0,
            "n0": 0,
            "n1": 0,
            "saturated_all1": 0,
            "untestable_n1": 0,
        },
        "counts": {},
        "embedder": EMBEDDER_NAME,
        "embeddings": {
            "embedder": EMBEDDER_NAME,
            "sidecar": "",
            "diversity": None,
            "mean_novelty": None,
            "n": 0,
            "stored_vectors": False,
        },
        "runs": runs,
        "rows": [],
        "selected": None,
        "llm_judge": {
            "requested": False,
            "wired": False,
            "note": "API key and flag only. Deterministic reward is unchanged.",
        },
        "grade_live_note": "Grade while generating is the Simulate toggle. This page grades a finished run.",
    }
    if extra:
        payload.update(extra)
    return payload


def features_for_run(agent_id: str, run_id: str | None = None) -> dict:
    """Feature tables plus a selected conversation for one run."""
    agent = _agent_name(agent_id)
    if not agent:
        return {"error": "agent required"}
    path, listed = _resolve_run(agent, run_id)
    runs = [_run_summary(p, rel, stem) for p, rel, stem in listed]
    declared, declared_list, declared_source = _declared_tools(agent)
    spec_blob = _agent_spec_blob(agent)
    note = ""
    if not path and listed and not _unspecified_batch(run_id):
        path = listed[0][0]
        note = "That batch isn't here. Showing the newest batch for this agent."
    if not path:
        chips = _build_fault_chips([], _declared_faults(spec_blob, {}))
        extra = {
            "harness": declared_source == "harness.json",
            "declared_source": declared_source,
            "declared_tools": declared_list,
            "faults": chips,
            "families": chips,
            "declared_faults": [c["name"] for c in chips if c.get("potential")],
            "counts": {c["name"]: c["n"] for c in chips},
        }
        return _empty_features(agent, runs, extra)

    rows = _load_rows(path)
    meta = _read_meta(path)
    stats = _stats(rows)
    embed = _load_embeddings(path, rows)
    novelty = embed.get("novelty") or [0.0] * len(rows)

    first: dict[str, dict] = {}
    within_zero: dict[str, int] = {}
    by_prompt: dict[str, list[int]] = defaultdict(list)
    declared_faults = _declared_faults(spec_blob, meta)
    chips = _build_fault_chips(rows, declared_faults)
    counts = {c["name"]: c["n"] for c in chips}

    for i, row in enumerate(rows):
        tools = _tools_used(row)
        ft = tools[0] if tools else "(none)"
        ts = first.setdefault(ft, {"tool": ft, "n": 0, "n0": 0, "n1": 0, "n_half": 0})
        _tally(ts, _bin(row))
        by_prompt[str(row.get("prompt") or "")].append(i)

    split_idxs: set[int] = set()
    first_diff = 0
    groups_k = 0
    sat = 0
    untestable = 0
    for idxs in by_prompt.values():
        labs = [_bin(rows[i]) for i in idxs]
        if len(idxs) >= 2:
            groups_k += 1
        if len(idxs) == 1:
            untestable += 1
        if labs and all(x == 1 for x in labs):
            sat += 1
        if not (any(x == 0 for x in labs) and any(x == 1 for x in labs)):
            continue
        for i in idxs:
            split_idxs.add(i)
        fts = {_tools_used(rows[i])[0] if _tools_used(rows[i]) else "(none)" for i in idxs}
        if len(fts) > 1:
            first_diff += 1
        for i in idxs:
            if _bin(rows[i]) == 0:
                fam = _reason_family(str(rows[i].get("reason") or ""))
                within_zero[fam] = within_zero.get(fam, 0) + 1

    table = []
    for i, row in enumerate(rows[:TABLE_CAP]):
        tools = _tools_used(row)
        b = _bin(row)
        reason = str(row.get("reason") or "")
        table.append({
            "i": i,
            "prompt": str(row.get("prompt") or ""),
            "reward": row.get("reward"),
            "bin": b,
            "verdict": verdict_label(row.get("reward")),
            "reason": reason,
            "reason_label": display_reason(reason),
            "family": _reason_family(reason),
            "fault": trace_fault(row),
            "first_tool": tools[0] if tools else "",
            "n_tools": len(tools),
            "novelty": novelty[i] if i < len(novelty) else None,
            "fault_detected": bool(row.get("fault_detected") or row.get("faults")),
            "split": i in split_idxs,
            "k": len(by_prompt.get(str(row.get("prompt") or ""), [i])),
            "final_text": str(row.get("final_text") or "")[:500],
            "steps": _step_glimpse(row),
            "messages": _messages(row),
        })

    selected = table[0] if table else None
    for item in table:
        if item["bin"] == 0 and item["split"]:
            selected = item
            break
        if selected is table[0] and item["bin"] == 0:
            selected = item

    tool_list = sorted(first.values(), key=lambda x: (-x["n"], x["tool"]))
    embeddings_public = {
        "embedder": EMBEDDER_NAME,
        "sidecar": embed.get("sidecar") or "",
        "diversity": embed.get("diversity"),
        "mean_novelty": embed.get("mean_novelty"),
        "n": embed.get("n") or 0,
        "stored_vectors": bool(embed.get("stored_vectors")),
    }
    return {
        "agent": agent,
        "run_id": _rel(path),
        "stem": path.stem,
        "mode": meta.get("mode") or "",
        "mtime": int(path.stat().st_mtime) if path.is_file() else 0,
        "harness": declared_source == "harness.json",
        "declared_source": declared_source,
        "declared_tools": declared_list,
        "n": stats["n"],
        "n0": stats["n0"],
        "n1": stats["n1"],
        "n_half": stats["n_half"],
        "ungraded": stats["ungraded"],
        "mean_reward": stats["mean_reward"],
        "prompts": stats["prompts"],
        "n_split": stats["n_split"],
        "rows_shown": len(table),
        "families": chips,
        "faults": chips,
        "declared_faults": declared_faults,
        "first_tool": tool_list[:32],
        "within_prompt": {
            "split_prompts": stats["n_split"],
            "groups_k": groups_k,
            "first_tool_differs": first_diff,
            "zeros": [{"name": k, "n": v} for k, v in sorted(within_zero.items(), key=lambda kv: -kv[1])],
        },
        "between_prompt": {
            "prompts": stats["prompts"],
            "n0": stats["n0"],
            "n1": stats["n1"],
            "saturated_all1": sat,
            "untestable_n1": untestable,
        },
        "counts": counts,
        "embedder": EMBEDDER_NAME,
        "embeddings": embeddings_public,
        "runs": runs,
        "rows": table,
        "selected": selected,
        "llm_judge": {
            "requested": bool(meta.get("llm_judge_requested")),
            "wired": False,
            "note": "API key and flag only. Deterministic reward is unchanged.",
        },
        "grade_live_note": "Grade while generating is the Simulate toggle. This page grades a finished run.",
        "warning": note,
    }


def _grade_one_file(path: Path, declared: set[str] | None, agent: str,
                    llm_requested: bool) -> dict:
    rows = _load_rows(path)
    changed = 0
    for row in rows:
        before = (row.get("reward"), row.get("reason"), bool(row.get("fault_detected")))
        grade = conduct_grade(row, declared_tools=declared)
        row["reward"] = grade.get("reward")
        row["reason"] = grade.get("reason")
        if grade.get("fault_detected"):
            row["fault_detected"] = True
        else:
            row.pop("fault_detected", None)
        after = (row.get("reward"), row.get("reason"), bool(row.get("fault_detected")))
        if before != after:
            changed += 1
    _write_jsonl(path, rows)
    embed = _store_embeddings(path, rows)
    stats = _stats(rows)
    meta = _read_meta(path)
    _write_meta(path, {
        "agent": meta.get("agent") or agent,
        "tags": meta.get("tags") or [],
        "mode": meta.get("mode") or "",
        "status": meta.get("status") or "done",
        "embedder": EMBEDDER_NAME,
        "llm_judge_requested": bool(llm_requested or meta.get("llm_judge_requested")),
        **stats,
    })
    return {"path": _rel(path), "rows": len(rows), "changed": changed, "embedder": embed["embedder"]}


def grade_run(agent_id: str, run_id: str | None = None, *, llm: bool = False,
              api_key: str | None = None) -> dict:
    """Write deterministic 0/1 reward/reason onto the selected run JSONL."""
    agent = _agent_name(agent_id)
    if not agent:
        return {"error": "agent required"}
    declared, declared_list, declared_source = _declared_tools(agent)
    listed = _list_run_paths(agent)
    if not listed:
        return {"error": "no runs for this agent", "agent": agent}

    llm_requested = bool(llm)
    # Flag only. Do not call the LLM judge pipeline. Do not persist the key.
    _ = str(api_key or "").strip()

    targets: list[Path] = []
    want = str(run_id or "").strip()
    if want == "all":
        targets = [p for p, _, _ in listed]
    else:
        path, _listed = _resolve_run(agent, run_id)
        if not path:
            return {"error": "batch not found", "agent": agent, "run_id": want}
        targets = [path]

    graded = []
    for path in targets:
        graded.append(_grade_one_file(path, declared, agent, llm_requested))

    out = features_for_run(agent, _rel(targets[0]))
    out["graded"] = sum(g["rows"] for g in graded)
    out["changed"] = sum(g["changed"] for g in graded)
    out["files"] = len(graded)
    out["declared_tools"] = declared_list
    out["declared_source"] = declared_source
    out["llm_judge"] = {
        "requested": llm_requested,
        "wired": False,
        "has_key": bool(str(api_key or "").strip()),
        "note": "API key and flag only. Deterministic reward is unchanged.",
    }
    return out


def features(agent: str = ""):
    """GET /api/grade?agent=  Agent may be 'name' or 'name|run_id'."""
    name, run_id = _split_agent(agent)
    return features_for_run(name, run_id or None)


def grade_now(body: dict | None = None):
    """POST /api/grade  Body: agent, id, llm, api_key."""
    body = body or {}
    agent = str(body.get("agent") or "")
    run_id = str(body.get("id") or body.get("run_id") or "")
    llm = bool(body.get("llm") or body.get("llm_judge"))
    key = body.get("api_key") or body.get("judge_key") or body.get("openai_key")
    return grade_run(agent, run_id or None, llm=llm, api_key=key)
