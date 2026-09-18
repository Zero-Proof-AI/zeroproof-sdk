"""Spec loading: a tools-and-system-prompt folder or dict becomes
``tools``, ``policy`` and seed situations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..generate.generator import assistant_kind

_SPEC_SKIP = {
    "tools",
    "policy",
    "system_prompt",
    "system",
    "instructions",
    "rules",
    "situations",
    "name",
    "id",
    "version",
    "model",
    "backend",
    "simulator",
}


def backend_spec(backend: str) -> str:
    text = str(backend).strip()
    if text.startswith(("vllm:", "ollama:", "openai:")):
        return text
    return "vllm:Qwen/Qwen3-4B-Instruct-2507@" + text.rstrip("/")


_backend_spec = backend_spec  # old private name, kept for imports that still use it


def _read_spec_file(path: Path) -> Any:
    text = path.read_text()
    if not str(text).strip():
        return None
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            return None
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _looks_like_spec_path(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if any(sep in raw for sep in "/\\"):
        return True
    if Path(raw).suffix.lower() in {".json", ".yaml", ".yml"}:
        return True
    return " " not in raw and len(raw) < 64  # literal: text heuristic, a slug


RUBRIC_FILES = ("rubric.md", "rubric.txt")


def spec_rubric(spec: Any) -> str | None:
    """The rubric shipped with a spec: what doing the job means, in prose,
    for the judge. ``rubric.md`` (or ``.txt``) next to the spec file the
    loader resolves ``spec`` to (so ``spec="my-agent"`` reads
    ``specs/my-agent/rubric.md``, the same shorthand ``simulate`` accepts),
    or a ``rubric`` key in the spec dict. None when the spec carries none."""
    if isinstance(spec, dict):
        text = str(spec.get("rubric") or "").strip()
        return text or None
    if not isinstance(spec, str) or not _looks_like_spec_path(spec):
        return None
    found = _spec_file(spec.strip())
    if found is None:
        return None
    path, loaded = found
    for name in RUBRIC_FILES:
        candidate = path.parent / name
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            return text or None
    text = str(loaded.get("rubric") or "").strip()
    return text or None


def _spec_from_path(text: str) -> dict | None:
    """Load tools+policy from a spec file or folder. None if it is not a path."""
    found = _spec_file(text)
    return found[1] if found else None


def _spec_file(text: str) -> tuple[Path, dict] | None:
    """The spec file ``text`` resolves to and what it holds: the path as
    given, the same with a spec suffix, ``spec.json`` inside it as a
    folder, or ``specs/<name>/`` for a bare name. None if none of them
    is a readable spec."""
    raw = Path(text).expanduser()
    roots = [Path.cwd(), Path(__file__).resolve().parents[1]]
    candidates: list[Path] = []
    for root in [Path()] if raw.is_absolute() else roots:
        base = raw if raw.is_absolute() else (root / raw)
        candidates.append(base)
        if not base.suffix:
            candidates.extend(
                [
                    Path(str(base) + ".json"),
                    Path(str(base) + ".yaml"),
                    Path(str(base) + ".yml"),
                    base / "spec.json",
                    base / "spec.yaml",
                    base / "agent.json",
                ]
            )
            if "/" not in str(text) and "\\" not in str(text):
                spec_dir = root / "specs" / raw.name
                candidates.extend([spec_dir, spec_dir / "spec.json", spec_dir / "spec.yaml"])
    seen: set[Path] = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved in seen or not cand.is_file():
            continue
        seen.add(resolved)
        try:
            loaded = _read_spec_file(cand)
        except Exception:
            continue
        if isinstance(loaded, dict):
            return cand, loaded
    return None


def _spec_extra_text(spec: dict) -> str:
    """Fold leftover spec fields into the world the writer sees."""
    parts: list[str] = []
    for key, val in spec.items():
        if key in _SPEC_SKIP or val in (None, "", [], {}):
            continue
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
        elif isinstance(val, list) and val and all(isinstance(x, str) for x in val):
            parts.extend(item.strip() for item in val if item.strip())
        elif isinstance(val, dict):
            for item in val.values():
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
    return "\n".join(parts)


def apply_spec(
    spec: Any, tools: list | None, policy: str | None, situations: list | None
) -> tuple[list | None, str | None, list]:
    extra_sit = list(situations or [])
    if spec is None:
        return tools, policy, extra_sit
    if isinstance(spec, str) and spec.strip():
        loaded = _spec_from_path(spec.strip())
        if loaded is not None:
            spec = loaded
        elif _looks_like_spec_path(spec):
            raise FileNotFoundError(
                f"simulate(spec={spec.strip()!r}) found no spec.json at that path."
            )
        else:
            policy = (str(policy or "").rstrip() + "\n" + spec.strip()).strip()
            return tools, policy, extra_sit
    if not isinstance(spec, dict):
        return tools, policy, extra_sit
    if spec.get("tools"):
        tools = list(tools or []) + list(spec["tools"])
    blob = (
        spec.get("system_prompt")
        or spec.get("system")
        or spec.get("policy")
        or spec.get("instructions")
        or spec.get("rules")
    )
    if isinstance(blob, list):
        blob = "\n".join(str(item) for item in blob)
    extra = _spec_extra_text(spec)
    merged = "\n".join(part for part in (blob, extra) if part)
    if merged:
        policy = (str(policy or "").rstrip() + "\n" + str(merged).strip()).strip()
    extra_sit.extend(str(s) for s in (spec.get("situations") or []) if s)
    return tools, policy, extra_sit


_apply_spec = apply_spec  # old private name, kept for imports that still use it


def kind_from_spec(spec: Any, policy: str) -> str:
    """Folder or spec name, then the policy identity line."""
    name = ""
    if isinstance(spec, dict):
        name = str(spec.get("name") or spec.get("id") or "").strip()
    elif isinstance(spec, str) and spec.strip():
        path = Path(spec.strip())
        part = path.name
        if part.lower() in {"spec.json", "spec.yaml", "spec.yml", "agent.json"}:
            part = path.parent.name
        if part and part.lower() not in {"specs", "spec"}:
            name = part
    return assistant_kind(policy, name)


_kind_from_spec = kind_from_spec  # old private name, kept for imports that still use it
