"""Schema-derived tool backend: validation, entity state, deterministic results."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable

_CREATE = re.compile(r"^(create|generate|write|add|upload|insert|make|post|new)", re.I)
_READ = re.compile(r"^(get|read|list|inspect|search|fetch|find|show|describe|cat|view)", re.I)
_DELETE = re.compile(r"^(delete|remove|drop|destroy)", re.I)
_REFERENCE_KEY = re.compile(
    r"(^id$|_id$|^path$|_path$|^file$|^name$|^key$|^ref$|^number$)",
    re.I)


def _tool_schema(tool: dict) -> tuple[str, dict]:
    function = tool.get("function", tool)
    name = str(function.get("name", ""))
    parameters = function.get("parameters") or {}
    return name, parameters


def _missing_required(parameters: dict, arguments: dict) -> list[str]:
    missing = []
    for key in parameters.get("required", []) or []:
        if arguments.get(key) in (None, ""):
            missing.append(str(key))
        else:
            child = (parameters.get("properties") or {}).get(key) or {}
            value = arguments.get(key)
            if child.get("type") == "object" and isinstance(value, dict):
                missing.extend(f"{key}.{grand}"
                               for grand in _missing_required(child, value))
    return missing


def _reference_values(arguments: Any, *, prefix: str = "") -> list[tuple[str, str]]:
    references = []
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            label = f"{prefix}{key}"
            if isinstance(value, (dict, list)):
                references.extend(_reference_values(value, prefix=label + "."))
            elif isinstance(value, (str, int)) and value != "" and value is not None \
                    and _REFERENCE_KEY.search(str(key)):
                references.append((label, str(value)))
    elif isinstance(arguments, list):
        for item in arguments:
            references.extend(_reference_values(item, prefix=prefix))
    return references


_ISSUED_ID = re.compile(r"^[a-z]+_[0-9a-f]{12}$")

# Generic content pools. Domain-neutral on purpose: names, statuses, and dates
# read plausibly for checks, commits, tickets, students, clauses, or listings.
_PEOPLE = ("alex kim", "priya patel", "sam garcia", "elena chen",
           "marcus reed", "dana okafor", "tom silva", "maya novak",
           "chris tanaka", "nina brooks", "luis weber", "jordan ali")
_ADJECTIVES = ("nightly", "routine", "primary", "draft", "updated",
               "automated", "manual", "initial", "final", "weekly",
               "legacy", "follow-up")
_TOPICS = ("config", "cleanup", "handoff", "review", "rollout",
           "migration", "sync", "audit", "onboarding", "renewal")
_ITEM_STATUSES = ("completed", "in_progress", "pending", "failed",
                  "active", "queued", "approved", "open")


def _pick(pool: tuple, k: int, salt: int) -> str:
    return pool[(k // salt) % len(pool)]


def _iso_date(k: int, i: int) -> str:
    month = 1 + (k // 31 + i) % 12
    day = 1 + (k + i * 7) % 28
    return f"2026-{month:02d}-{day:02d}"


def _item_noun(tool: str) -> str:
    """Singular object noun from the tool name: list_commits -> commit."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", str(tool).lower()) if t]
    rest = tokens[1:] or tokens
    noun = rest[-1] if rest else ""
    if len(noun) > 3 and noun.endswith("s") and not noun.endswith("ss"):
        noun = noun[:-1]
    return noun


def _record_fields(k: int, i: int, noun: str = "") -> dict[str, Any]:
    """One plausible record body. Deterministic in (k, i).

    Items are named after the tool's own noun so a product search returns
    product-shaped names, not workflow vocabulary.
    """
    label = noun or _pick(_TOPICS, k + i * 104729, 97)
    return {
        "id": 1000 + ((k + i * 137) % 89000),
        "name": f"{_pick(_ADJECTIVES, k + i * 7919, 7)} {label}",
        "status": _pick(_ITEM_STATUSES, k + i * 15485863, 13),
        "owner": _pick(_PEOPLE, k + i * 32452843, 29),
        "updated_at": _iso_date(k, i),
    }


def _invented_record(tool: str, arguments: dict, n: int, digest: str) -> dict[str, Any]:
    """Domain-shaped read result. Content varies with the exact arguments.

    Listings return several named records with statuses, owners, and dates so
    the agent has something real to quote instead of fabricating. No default
    repo, PR, or issue is invented; caller references are echoed as context.
    """
    known = {k: v for k, v in arguments.items() if v not in (None, "")}
    name = str(tool or "").lower()
    noun = _item_noun(name)
    if "search" in name or name.startswith("list"):
        count = 2 + n % 2
        items = []
        for i in range(count):
            item = dict(known)
            item.update(_record_fields(n, i, noun))
            items.append(_entity_consistent(item, None, noun))
        return {
            "query": str(arguments.get("query") or arguments.get("q") or ""),
            "count": count,
            "items": items,
        }
    record = dict(_record_fields(n, 0, noun))
    record.update(known)
    if digest:
        record.setdefault("ref", digest[:8])
    return record


_KEY_DATEISH = re.compile(r"(date|_at$|^at$|time$|day$|when)", re.I)
_KEY_IDISH = re.compile(r"(^id$|_id$|number$|^sku$|^ref$|^pnr$|^code$)", re.I)
_KEY_PERSONISH = re.compile(r"(^from$|owner|author|assignee|sender|^by$)", re.I)


def _entity_seed(value: Any) -> int:
    """One seed per entity id, shared by every tool that touches it.

    search items and a later read of the same id must describe the SAME
    underlying record; a listing must not silently transform into an
    unrelated entity. Keyed on the id value alone, not the tool.
    """
    return int(hashlib.sha256(f"entity:{value}".encode()).hexdigest()[:8], 16)


def _idish_key(record: dict) -> str:
    for key, val in record.items():
        if _KEY_IDISH.search(str(key)) and isinstance(val, (str, int)) and val != "":
            return str(key)
    return ""


def _entity_consistent(record: dict, template: Any = None,
                       noun: str = "") -> dict:
    """Re-derive a record's fields from its own id so reads agree with it."""
    key = _idish_key(record)
    if not key:
        return record
    ident = record[key]
    seed = _entity_seed(ident)
    if isinstance(template, dict):
        fresh = _fill_template(template, seed)
    else:
        fresh = _record_fields(seed, 0, noun)
    fresh = dict(fresh)
    fresh.update({k: v for k, v in record.items() if k not in fresh})
    fresh[key] = ident
    return fresh


def _fill_template(value: Any, k: int, i: int = 0, key: str = "") -> Any:
    """Instantiate an LLM-written example result. Deterministic in (k, i).

    Field names and free text stay as the model wrote them; ids, dates,
    numbers, and people vary with the exact call so repeat calls with
    different arguments return different records.
    """
    if isinstance(value, dict):
        return {kk: _fill_template(vv, k, i, str(kk)) for kk, vv in value.items()}
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            count = 2 + (k + i) % 2
            return [_fill_template(value[0], k, i * 7 + j + 1) for j in range(count)]
        return list(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) <= 1:
            return value
        span = max(1, abs(value) // 3)
        return value + ((k + i * 131) % (2 * span + 1)) - span
    if isinstance(value, float):
        span = max(1, int(abs(value) * 10) // 3)
        tenths = int(value * 10) + ((k + i * 131) % (2 * span + 1)) - span
        return round(tenths / 10.0, 2)
    if isinstance(value, str):
        if _KEY_PERSONISH.search(key):
            person = _pick(_PEOPLE, k + i * 29, 7)
            if "@" in value:
                domain = value.split("@", 1)[1]
                return person.replace(" ", ".") + "@" + domain
            return person
        if _KEY_IDISH.search(key) and any(c.isdigit() for c in value):
            return re.sub(r"\d+", str(1000 + (k + i * 137) % 89000), value, count=1)
        if _KEY_DATEISH.search(key) and re.search(r"\d{4}-\d{2}-\d{2}", value):
            return _iso_date(k, i) + value[10:]
        return value
    return value


def _created_record(tool: str, arguments: dict, n: int, digest: str) -> dict[str, Any]:
    """Echo the create call plus an issued id. Never sets a status field."""
    known = {k: v for k, v in arguments.items() if v not in (None, "")}
    name = str(tool or "").lower()
    if not known:
        stem = name.split("_")[0] or "item"
        return {"id": f"{stem}_{n % 100000}", "ok": True}
    known.setdefault("id", f"{(name.split('_')[0] or 'item')}_{n % 100000}")
    if digest:
        known.setdefault("ref", digest[:8])
    return known


class MockEnvironment:
    """In-memory tool world. User-named refs exist ~70% of the time; issued ids must be created."""

    def __init__(self, tools: list[dict], *, seed: int = 0,
                 faults: dict[str, dict] | None = None,
                 world_state: str = "",
                 result_shapes: dict[str, dict] | None = None) -> None:
        self.schemas = dict(_tool_schema(tool) for tool in tools or [])
        # Model-written example results per tool (see write_result_shapes).
        # A shared mutable dict is fine: it may fill in a few seconds late.
        self.result_shapes = result_shapes if result_shapes is not None else {}
        self.seed = seed
        self.entities: dict[str, dict[str, Any]] = {}
        self.deleted: set[str] = set()
        raw = dict(faults or {})
        self.world_state = str(world_state or raw.pop("world_state", "") or "")
        self.faults = raw

    def _fault_for(self, tool: str, arguments: dict) -> dict[str, Any] | None:
        spec = self.faults.get(tool) or self.faults.get("*")
        if not spec:
            return None
        rate = float(spec.get("rate", 1.0))
        if rate <= 0:
            return None
        payload = json.dumps({"seed": self.seed, "tool": tool,
                              "arguments": arguments, "salt": "fault"},
                             sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        if int(digest[:8], 16) / float(0xFFFFFFFF) >= rate:
            return None
        mode = str(spec.get("mode", "timeout"))
        if mode == "timeout":
            return {"status": "timeout", "error": "request timed out"}
        if mode == "malformed":
            return {"status": "ok", "data": "<<garbled resp0nse"}
        if mode == "stale":
            return {"status": "ok", "data": {"result": self._digest(tool, arguments)},
                    "stale": True, "as_of": "3 days ago"}
        if mode == "permission_denied":
            return {"status": "permission_denied"}
        return None

    def _exists(self, value: str) -> bool:
        if value in self.entities:
            return True
        if value in self.deleted or _ISSUED_ID.match(value):
            return False
        digest = hashlib.sha256(f"exists:{self.seed}:{value}".encode()).hexdigest()
        return int(digest[:8], 16) % 10 < 7

    def _digest(self, tool: str, arguments: dict) -> str:
        payload = json.dumps({"seed": self.seed, "tool": tool,
                              "arguments": arguments}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def _payload(self, tool: str, digest: str,
                 arguments: dict | None = None) -> dict[str, Any]:
        """Plausible structured record from the tool name and args. Never a bare hash."""
        n = int(digest[:8], 16)
        args = arguments or {}
        references = _reference_values(args)
        if references:
            # Reads of a named entity derive from the entity id, not the
            # full call, so every tool describes the same record.
            n = _entity_seed(references[0][1])
        shape = self.result_shapes.get(tool)
        if isinstance(shape, dict) and shape:
            filled = _fill_template(shape, n)
            for key, val in args.items():
                if key in filled and not isinstance(filled[key], (dict, list)):
                    filled[key] = val
            for key, val in list(filled.items()):
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    template = shape.get(key)
                    item_template = (template[0] if isinstance(template, list)
                                     and template and isinstance(template[0], dict)
                                     else None)
                    filled[key] = [_entity_consistent(item, item_template)
                                   for item in val]
            return {"data": filled}
        if re.search(r"(estimat|calc|payment|price|rate|amount|monthly|comp)",
                     tool, re.I):
            return {"amount": 100 + (n % 4900), "currency": "USD"}
        return {"data": _invented_record(tool, args, n, digest)}

    def call(self, tool: str, arguments: dict | None = None) -> dict[str, Any]:
        arguments = arguments or {}
        if tool not in self.schemas:
            return {"status": "error", "reason": "unknown_tool", "tool": tool}
        fault = self._fault_for(tool, arguments)
        if fault is not None:
            return fault
        if self.world_state in {"entity missing", "missing"}:
            refs = [v for _, v in _reference_values(arguments)]
            return {"status": "not_found",
                    "missing": refs or ["entity"]}
        if self.world_state in {"entity already acted on", "already_done"}:
            return {"status": "already_done", "reason": "already_acted_on"}
        missing = _missing_required(self.schemas[tool], arguments)
        if missing:
            return {"status": "rejected", "reason": "missing_required",
                    "fields": missing}

        references = _reference_values(arguments)
        digest = self._digest(tool, arguments)

        if _CREATE.match(tool):
            n = int(digest[:8], 16)
            record = _created_record(tool, arguments, n, digest)
            entity_id = str(record.get("number") or record.get("id") or (1000 + n % 89000))
            self.entities[entity_id] = {"tool": tool, "arguments": arguments,
                                        "version": 1}
            for _, value in references:
                self.entities[value] = {"tool": tool, "arguments": arguments,
                                        "version": 1}
            return {"status": "created", **record}

        dangling = [value for _, value in references
                    if not self._exists(value)]
        if _READ.match(tool):
            if references and dangling:
                return {"status": "not_found", "missing": dangling}
            return {"status": "ok", **self._payload(tool, digest, arguments)}
        if _DELETE.match(tool):
            if dangling:
                return {"status": "not_found", "missing": dangling}
            for _, value in references:
                self.entities.pop(value, None)
                self.deleted.add(value)
            return {"status": "deleted", "removed": [v for _, v in references]}

        if references and dangling:
            return {"status": "not_found", "missing": dangling}
        for _, value in references:
            entity = self.entities.setdefault(
                value, {"tool": tool, "version": 0}) if self._exists(value) else None
            if entity is not None:
                entity["version"] = int(entity.get("version", 1)) + 1
        return {"status": "ok", **self._payload(tool, digest, arguments)}

    def executor(self) -> Callable[[str, dict], dict]:
        return self.call
