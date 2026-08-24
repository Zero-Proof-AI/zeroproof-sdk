"""Agents tab. Local harness store.

    outputs/studio-runs/agents/{id}/
      harness.json     {name, tools, policy}
      metadata.json    {id, name, created, tags}
      runs/            empty until Simulate

serve.py calls list_agents, get_agent, create_agent, sample.
save_agent is the write path; create_agent is the POST alias.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = ROOT / "outputs" / "studio-runs" / "agents"
RESERVED = {"new", "untitled", "api", "agents", "none"}
PROFILE = ("github", "coding", "intercom", "linear", "amazon")
SPEC_DIRS = (ROOT / "tests" / "fixtures", ROOT / "specs")
DEFAULT_POLICY = (
    "Use the tools. Do not invent file paths, ids, or search results. "
    "If a tool misses, tell the user."
)
STARTER_TOOLS = {
    "read": {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file at a path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    "write": {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write text to a file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    "summarize": {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Summarize the given text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for a query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    "list_dir": {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files in a folder.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    "fetch_url": {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a URL and return the text.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
}


def _agent_id(raw) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(raw or "").strip())
    s = s.strip("-._")[:64]
    return s


def agent_dir(name: str) -> Path:
    return AGENTS_DIR / _agent_id(name)


def harness_path(name: str) -> Path:
    return agent_dir(name) / "harness.json"


def metadata_path(name: str) -> Path:
    return agent_dir(name) / "metadata.json"


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


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


def _parse_tools(raw) -> tuple[list, str | None, str]:
    extra_policy = ""
    if raw is None or raw == "":
        return [], "tools JSON is required", ""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return [], "tools JSON is required", ""
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            return [], f"tools JSON: {exc}", ""
    if isinstance(raw, dict):
        extra_policy = str(raw.get("policy") or raw.get("system_prompt") or "")
        if "tools" in raw:
            tools = raw.get("tools") or []
        elif "function" in raw or raw.get("type") == "function" or raw.get("name"):
            tools = [raw]
        else:
            return [], "tools must be a JSON list or a spec with a tools key", extra_policy
    elif isinstance(raw, list):
        tools = raw
    else:
        return [], "tools must be a JSON list or a spec with a tools key", ""
    if not isinstance(tools, list) or not tools:
        return [], "tools list is empty", extra_policy
    if not _tool_names(tools):
        return [], "no tool names in tools JSON", extra_policy
    return list(tools), None, extra_policy


def _count_lines(path: Path) -> int:
    n = 0
    try:
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    n += 1
    except OSError:
        return 0
    return n


def _run_files(name: str) -> list[Path]:
    runs = agent_dir(name) / "runs"
    if not runs.is_dir():
        return []
    return sorted(p for p in runs.glob("*.jsonl") if p.is_file())


def _run_stats(name: str) -> tuple[int, int]:
    files = _run_files(name)
    return len(files), sum(_count_lines(p) for p in files)


def _read_harness(name: str) -> tuple[dict | None, str | None]:
    data = _read_json(harness_path(name))
    if data is None:
        return None, "not found"
    tools, err, extra = _parse_tools(data.get("tools") if "tools" in data else data)
    policy = str(data.get("policy") or data.get("system_prompt") or extra or "")
    display = str(data.get("name") or name).strip() or _agent_id(name)
    if err:
        return {"name": display, "tools": data.get("tools") or [], "policy": policy}, err
    return {"name": display, "tools": tools, "policy": policy}, None


def _read_metadata(name: str) -> dict:
    slug = _agent_id(name)
    data = _read_json(metadata_path(slug)) or {}
    tags = data.get("tags") if isinstance(data.get("tags"), list) else []
    created = data.get("created")
    try:
        created = int(created) if created is not None else None
    except (TypeError, ValueError):
        created = None
    display = str(data.get("name") or "").strip()
    return {
        "id": str(data.get("id") or slug),
        "name": display,
        "created": created,
        "tags": [str(t) for t in tags if str(t).strip()],
    }


def _public(name: str, harness: dict | None, *, error: str | None = None) -> dict:
    slug = _agent_id(name)
    meta = _read_metadata(slug)
    tools = list((harness or {}).get("tools") or [])
    policy = str((harness or {}).get("policy") or "")
    display = str((harness or {}).get("name") or meta.get("name") or slug)
    n_runs, n_rows = _run_stats(slug)
    names = _tool_names(tools)
    row = {
        "id": slug,
        "name": display,
        "tools": tools,
        "policy": policy,
        "n_runs": n_runs,
        "n_rows": n_rows,
        "n_tools": len(names),
        "tool_names": names,
        "tags": meta.get("tags") or [],
        "created": meta.get("created"),
        "path": f"outputs/studio-runs/agents/{slug}/harness.json",
    }
    if error:
        row["error"] = error
    return row


def _load_repo_spec(name: str) -> dict | None:
    """Real tools + policy from tests/fixtures or specs/. No invented policies."""
    slug = _agent_id(name)
    for folder in SPEC_DIRS:
        path = folder / slug / "spec.json"
        data = _read_json(path)
        if not data:
            continue
        tools, err, extra = _parse_tools(data)
        if err:
            continue
        policy = str(data.get("policy") or data.get("system_prompt") or extra or "").strip()
        if not tools or not policy:
            continue
        try:
            rel = str(path.relative_to(ROOT))
        except ValueError:
            rel = str(path)
        return {"tools": tools, "policy": policy, "spec": rel}
    return None


SPEC_PICKS = (
    ("github", "GitHub", ROOT / "tests" / "fixtures" / "github" / "spec.json"),
    ("coding", "Coding", ROOT / "specs" / "coding" / "spec.json"),
    ("intercom", "Intercom", ROOT / "specs" / "intercom" / "spec.json"),
    ("linear", "Linear", ROOT / "tests" / "fixtures" / "linear" / "spec.json"),
    ("amazon", "Amazon", ROOT / "specs" / "amazon" / "spec.json"),
)
BASIC_CHIPS = ("read", "write", "summarize", "web_search", "list_dir", "fetch_url")
CHIP_LABELS = {
    "read": "read",
    "write": "write",
    "summarize": "summarize",
    "web_search": "web search",
    "list_dir": "list_dir",
    "fetch_url": "fetch_url",
}


def list_starters() -> dict:
    """Compact specs + basic tool stubs for Add agent autofill."""
    specs = []
    for sid, label, path in SPEC_PICKS:
        data = _read_json(path) if path.is_file() else None
        loaded = None
        if data:
            tools, err, extra = _parse_tools(data)
            policy = str(data.get("policy") or data.get("system_prompt") or extra or "").strip()
            if not err and tools and policy:
                try:
                    rel = str(path.relative_to(ROOT))
                except ValueError:
                    rel = str(path)
                loaded = {"id": sid, "name": label, "tools": tools, "policy": policy, "spec": rel}
        if not loaded:
            spec = _load_repo_spec(sid)
            if spec:
                loaded = {
                    "id": sid,
                    "name": label,
                    "tools": spec["tools"],
                    "policy": spec["policy"],
                    "spec": spec.get("spec"),
                }
        if loaded:
            specs.append(loaded)
    basic = []
    for key in BASIC_CHIPS:
        schema = STARTER_TOOLS.get(key)
        if not schema:
            continue
        basic.append({
            "id": key,
            "label": CHIP_LABELS.get(key, key),
            "schema": json.loads(json.dumps(schema)),
        })
    return {"specs": specs, "basic": basic}


def ensure_profile_agents() -> list[dict]:
    """Write the five project agents if missing. Never overwrite an existing harness."""
    written = []
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    now = int(time.time())
    for name in PROFILE:
        spec = _load_repo_spec(name)
        if not spec:
            continue
        folder = agent_dir(name)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "runs").mkdir(exist_ok=True)
        hp = harness_path(name)
        data = _read_json(hp)
        tools = (data or {}).get("tools") if isinstance((data or {}).get("tools"), list) else []
        policy = str((data or {}).get("policy") or (data or {}).get("system_prompt") or "").strip()
        display = str((data or {}).get("name") or name)
        canonical = {"name": display, "tools": tools or spec["tools"], "policy": policy or spec["policy"]}
        if data is None or set(data.keys()) != {"name", "tools", "policy"} or not tools or not policy:
            _write_json(hp, canonical)
            if data is None:
                written.append({"id": name, "spec": spec["spec"]})
        mp = metadata_path(name)
        if not mp.is_file():
            _write_json(mp, {
                "id": name,
                "name": display,
                "created": now,
                "tags": [],
            })
    return written


def _starter_key(raw) -> str:
    key = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key in {"websearch", "web"}:
        return "web_search"
    if key in {"listdir", "ls"}:
        return "list_dir"
    if key in {"fetch", "fetchurl"}:
        return "fetch_url"
    return key


def _tools_from_starters(ids) -> list:
    if isinstance(ids, str):
        ids = [part.strip() for part in ids.split(",") if part.strip()]
    tools = []
    seen: set[str] = set()
    for raw in ids or []:
        key = _starter_key(raw)
        if key in seen or key not in STARTER_TOOLS:
            continue
        seen.add(key)
        tools.append(json.loads(json.dumps(STARTER_TOOLS[key])))
    return tools


def save_agent(body: dict | None = None):
    body = body or {}
    raw_name = str(body.get("name") or body.get("id") or body.get("agent") or "").strip()
    slug = _agent_id(raw_name)
    if not slug or slug.lower() in RESERVED:
        return {"error": "Name is required."}
    replace = bool(body.get("replace"))
    if harness_path(slug).is_file() and not replace:
        return {"error": "That name is taken."}
    extra = ""
    tools_raw = body.get("tools")
    if tools_raw not in (None, "", [], {}):
        tools, err, extra = _parse_tools(tools_raw)
        if err:
            return {"error": err}
    else:
        tools = _tools_from_starters(body.get("starter_tools") or body.get("starters"))
        if not tools:
            return {"error": "Pick at least one tool."}
    policy = str(body.get("policy") or body.get("system_prompt") or extra or "").strip()
    if not policy:
        policy = DEFAULT_POLICY
    display = raw_name or slug
    folder = agent_dir(slug)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "runs").mkdir(exist_ok=True)
    harness = {"name": display, "tools": tools, "policy": policy}
    _write_json(harness_path(slug), harness)
    existing = _read_metadata(slug)
    created = existing.get("created") or int(time.time())
    tags = body.get("tags") if body.get("tags") is not None else existing.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    metadata = {
        "id": slug,
        "name": display,
        "created": created,
        "tags": [str(t) for t in tags if str(t).strip()],
    }
    _write_json(metadata_path(slug), metadata)
    return _public(slug, harness)


def create_agent(body: dict | None = None):
    """POST /api/agents."""
    return save_agent(body)


def list_agents():
    try:
        ensure_profile_agents()
        if not AGENTS_DIR.is_dir():
            return {"agents": []}
        rows = []
        seen: set[str] = set()
        for folder in AGENTS_DIR.iterdir():
            if not folder.is_dir():
                continue
            slug = _agent_id(folder.name)
            if not slug or slug.lower() in RESERVED or slug in seen:
                continue
            if not harness_path(slug).is_file():
                continue
            harness, err = _read_harness(slug)
            rows.append(_public(slug, harness, error=err))
            seen.add(slug)
        def key(row: dict):
            name = row.get("id") or ""
            try:
                return (0, PROFILE.index(name), name.lower())
            except ValueError:
                return (1, 0, name.lower())
        rows.sort(key=key)
        return {"agents": rows}
    except OSError as exc:
        return {"error": f"could not read agents: {exc}", "agents": []}


def get_agent(name: str = ""):
    slug = _agent_id(name)
    if not slug or slug.lower() in RESERVED:
        return {"error": "name required"}
    if slug in PROFILE:
        ensure_profile_agents()
    harness, err = _read_harness(slug)
    if harness is None:
        return {"error": "not found"}
    row = _public(slug, harness, error=err)
    if err and not harness.get("tools"):
        row["error"] = err
        return row
    return row


def sample(agent: str = ""):
    slug = _agent_id(agent)
    if not slug:
        return {"error": "no sample"}
    row = get_agent(slug)
    if row.get("error"):
        return {"error": "no sample"}
    return row
