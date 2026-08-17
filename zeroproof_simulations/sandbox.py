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


def _function_spec(tool: dict) -> tuple[str, dict]:
    function = tool.get("function", tool) if isinstance(tool, dict) else {}
    if not isinstance(function, dict):
        function = {}
    name = str(function.get("name", ""))
    return name, function


def _param_keys(spec: dict, arguments: dict | None = None) -> set[str]:
    props = ((spec.get("parameters") or {}).get("properties") or {})
    keys = {str(k).lower() for k in props}
    if arguments:
        keys |= {str(k).lower() for k in arguments}
    return keys


def _result_kind(tool: str, spec: dict | None = None,
                 arguments: dict | None = None) -> str:
    """Infer the payload family from name, params, and description.

    Not a hardcoded product list. A new spec with unknown tool names still
    maps: path+read -> file, command -> shell, otherwise a record.
    """
    name = str(tool or "").lower()
    spec = spec if isinstance(spec, dict) else {}
    desc = str(spec.get("description") or "").lower()
    keys = _param_keys(spec, arguments)
    blob = f"{name} {desc}"

    if re.search(r"(run_command|run_terminal|bash|shell_command|"
                 r"execute_command|terminal_cmd|run_bash|^exec$|"
                 r"run_tests?$|pytest)", name) \
            or (("command" in keys or "cmd" in keys or "argv" in keys)
                and re.search(r"(run|exec|shell|bash|terminal)", name)):
        return "shell"
    if re.search(r"(grep|ripgrep|search_code|code_search|find_in_files|"
                 r"search_files)", name) \
            or (("pattern" in keys or "regex" in keys)
                and keys & {"path", "glob", "directory", "dir"}):
        return "grep"
    if re.search(r"(read_file|write_file|get_file|cat_file|view_file|"
                 r"edit_file|apply_patch|replace_in_file|open_file|"
                 r"file_contents)", name) \
            or (keys & {"path", "file", "file_path", "filepath", "filename"}
                and re.search(r"(read|write|get|cat|view|edit|open|apply|"
                              r"patch|contents|file)", name)
                and not re.search(
                    r"(profile|(?:^|_)user(?:_|$)|order|issue|"
                    r"(?:^|_)prs?(?:_|$)|account)",
                    name)):
        return "file"
    if re.search(r"(list_dir|list_files|glob_files|ls_dir)", name) \
            or (name.startswith("list")
                and keys & {"path", "directory", "dir", "glob"}
                and not re.search(r"(order|issue|check|commit|event)", name)):
        return "files"
    if name.startswith("git") or re.search(
            r"(git_status|git_diff|git_log|git_show|git_blame)", name) \
            or re.search(r"(^list_commits$|_commits$|commit_log)", name):
        return "git"
    if re.search(r"(list_checks|_checks$|ci_status|test_results)", name) \
            or (re.search(r"\b(ci|check run)", blob)
                and keys & {"sha", "commit", "number"}):
        return "ci"
    if re.search(r"(payment|price|invoice|balance|estimate|fx_rate|"
                 r"amount_due)", name):
        return "money"
    return "record"


def _shape_mismatches_kind(kind: str, filled: dict) -> bool:
    """True when an LLM example is a CRM record for a file/shell/git tool."""
    if kind not in {"file", "files", "grep", "shell", "git", "ci"}:
        return False
    if not isinstance(filled, dict):
        return True
    keys = {str(k).lower() for k in filled}
    if kind == "shell" and not (keys & {"stdout", "stderr", "exit_code", "command"}):
        return True
    if kind == "file" and not (keys & {"content", "diff", "text", "bytes"}):
        return True
    if kind == "grep" and "matches" not in keys:
        return True
    if kind == "files" and not (keys & {"entries", "files", "items"}):
        return True
    if kind == "ci" and not (keys & {"items", "checks", "jobs", "conclusion"}):
        return True
    if {"owner", "updated_at"} <= keys and not (
            keys & {"content", "stdout", "diff", "matches", "entries"}):
        return True
    return False


_CODE_VERBS = ("load", "parse", "render", "sync", "build", "handle",
               "validate", "merge", "format", "dispatch")
_CODE_NOUNS = ("config", "payload", "record", "client", "worker", "schema",
               "queue", "session", "index", "router")
_CHECK_NAMES = ("lint", "tests", "typecheck", "build", "coverage",
                "security", "format")
_FILE_STEMS = ("app", "util", "worker", "client", "schema", "router",
               "session", "queue")


def _hex(n: int, width: int = 12) -> str:
    return hashlib.sha256(f"hex:{n}".encode()).hexdigest()[:width]


def _path_from_args(arguments: dict, n: int, ext: str = "py") -> str:
    for key in ("path", "file_path", "filepath", "file", "filename"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"src/{_pick(_FILE_STEMS, n, 5)}.{ext}"


def _fake_source(path: str, n: int) -> str:
    """Plausible file text. Deterministic in (path, n). Never executed."""
    ext = ""
    if "." in path.rsplit("/", 1)[-1]:
        ext = path.rsplit(".", 1)[-1].lower()
    verb = _pick(_CODE_VERBS, n, 3)
    noun = _pick(_CODE_NOUNS, n, 11)
    if ext in {"json"}:
        return (f'{{\n  "{noun}": "{verb}",\n  "enabled": true,\n'
                f'  "retries": {1 + n % 4}\n}}\n')
    if ext in {"yml", "yaml", "toml"}:
        return f"{noun}:\n  mode: {verb}\n  retries: {1 + n % 4}\n"
    if ext in {"log", "txt"}:
        stamp = _iso_date(n, 0)
        return (f"{stamp}T09:00:00Z INFO {verb}_{noun} started\n"
                f"{stamp}T09:00:01Z INFO handled 12 items\n"
                f"{stamp}T09:00:02Z WARN retry {1 + n % 3}\n")
    if ext in {"diff", "patch"}:
        return _fake_diff(path, n)
    if ext in {"md"}:
        return (f"# {noun}\n\nThe {verb} path lives in `{path}`.\n\n"
                f"- retries: {1 + n % 4}\n- owner queue: {noun}\n")
    return (
        f"# {path}\n"
        f"from {noun} import {verb}_state\n\n"
        f"def {verb}_{noun}(value):\n"
        f"    if not value:\n"
        f"        raise ValueError({noun!r})\n"
        f"    return {verb}_state(value)\n"
    )


def _fake_diff(path: str, n: int) -> str:
    verb = _pick(_CODE_VERBS, n, 3)
    noun = _pick(_CODE_NOUNS, n, 11)
    return (
        f"--- a/{path}\n+++ b/{path}\n"
        f"@@ -1,4 +1,6 @@\n"
        f" def {verb}_{noun}(value):\n"
        f"-    return value\n"
        f"+    if not value:\n"
        f"+        return {noun!r}\n"
        f"+    return value\n"
    )


def _invented_file(arguments: dict, n: int) -> dict[str, Any]:
    path = _path_from_args(arguments, n)
    content = _fake_source(path, n)
    return {"path": path, "content": content, "bytes": len(content),
            "encoding": "utf-8"}


def _invented_files(arguments: dict, n: int) -> dict[str, Any]:
    root = str(arguments.get("path") or arguments.get("directory")
               or arguments.get("dir") or "src")
    count = 3 + n % 3
    entries = []
    for i in range(count):
        stem = _pick(_FILE_STEMS, n + i * 17, 5)
        kind = "dir" if i == 0 and n % 5 == 0 else "file"
        name = stem if kind == "dir" else f"{stem}.py"
        entries.append({"name": name, "type": kind,
                        "path": f"{root.rstrip('/')}/{name}"})
    return {"path": root, "entries": entries, "count": count}


def _invented_grep(arguments: dict, n: int) -> dict[str, Any]:
    pattern = str(arguments.get("pattern") or arguments.get("query")
                  or arguments.get("regex") or "TODO")
    count = 2 + n % 3
    matches = []
    for i in range(count):
        path = f"src/{_pick(_FILE_STEMS, n + i * 13, 5)}.py"
        line = 8 + ((n + i * 19) % 80)
        verb = _pick(_CODE_VERBS, n + i, 3)
        noun = _pick(_CODE_NOUNS, n + i, 11)
        matches.append({
            "path": path, "line": line,
            "text": f"    {verb}_{noun}({pattern!r})",
        })
    return {"pattern": pattern, "count": count, "matches": matches}


def _invented_shell(arguments: dict, n: int) -> dict[str, Any]:
    command = str(arguments.get("command") or arguments.get("cmd")
                  or arguments.get("argv") or "true")
    flavor = n % 11
    if flavor == 0:
        return {"command": command, "exit_code": 1,
                "stdout": "",
                "stderr": "bash: src/secret.key: Permission denied\n"}
    if flavor == 1:
        return {
            "command": command, "exit_code": 1,
            "stdout": (
                "============================= test session starts "
                "==============================\n"
                f"collected {6 + n % 5} items\n"
                "tests/test_app.py .....F\n"
                "FAILED tests/test_app.py::test_sync - AssertionError: "
                "expected 1\n"
                "=========================== 1 failed, 5 passed in 0.51s "
                "========================\n"),
            "stderr": "",
        }
    if flavor == 2:
        return {
            "command": command, "exit_code": 1,
            "stdout": (
                "Auto-merging src/app.py\n"
                "CONFLICT (content): Merge conflict in src/app.py\n"
                "Automatic merge failed; fix conflicts and then commit "
                "the result.\n"),
            "stderr": "",
        }
    if "ls" in command or flavor == 3:
        return {
            "command": command, "exit_code": 0,
            "stdout": "src/\n  app.py\n  util.py\nREADME.md\n",
            "stderr": "",
        }
    passed = 6 + n % 6
    return {
        "command": command, "exit_code": 0,
        "stdout": (
            "============================= test session starts "
            "==============================\n"
            f"collected {passed} items\n"
            f"tests/test_app.py {'.' * min(passed, 8)}\n"
            f"============================== {passed} passed in 0.42s "
            "===============================\n"),
        "stderr": "",
    }


def _invented_git(tool: str, arguments: dict, n: int) -> dict[str, Any]:
    name = str(tool or "").lower()
    branch = f"feature/{_pick(_CODE_NOUNS, n, 11)}"
    if "diff" in name or "show" in name:
        path = _path_from_args(arguments, n)
        return {"path": path, "diff": _fake_diff(path, n), "branch": branch}
    if "log" in name or "commit" in name:
        count = 2 + n % 2
        items = []
        for i in range(count):
            items.append({
                "sha": _hex(n + i * 29, 12),
                "message": f"{_pick(_CODE_VERBS, n + i, 3)} "
                           f"{_pick(_CODE_NOUNS, n + i, 11)}",
                "author": _pick(_PEOPLE, n + i * 7, 29),
                "date": _iso_date(n, i),
            })
        return {"branch": branch, "count": count, "items": items}
    staged = [f"src/{_pick(_FILE_STEMS, n, 5)}.py"]
    unstaged = [f"src/{_pick(_FILE_STEMS, n + 3, 5)}.py"]
    return {"branch": branch, "clean": False, "ahead": 1 + n % 3,
            "behind": n % 2, "staged": staged, "unstaged": unstaged,
            "untracked": ["tmp.log"]}


def _invented_ci(arguments: dict, n: int) -> dict[str, Any]:
    count = 3 + n % 2
    items = []
    for i in range(count):
        failed = (n % 7 == 0 and i == count - 1)
        items.append({
            "name": _CHECK_NAMES[i % len(_CHECK_NAMES)],
            "status": "completed",
            "conclusion": "failure" if failed else "success",
            "duration_s": 8 + ((n + i * 11) % 90),
        })
    return {"count": count, "items": items}


def _invented_payload(tool: str, arguments: dict, n: int, digest: str,
                      spec: dict | None = None) -> dict[str, Any]:
    kind = _result_kind(tool, spec, arguments)
    if kind == "file":
        return _invented_file(arguments, n)
    if kind == "files":
        return _invented_files(arguments, n)
    if kind == "grep":
        return _invented_grep(arguments, n)
    if kind == "shell":
        return _invented_shell(arguments, n)
    if kind == "git":
        return _invented_git(tool, arguments, n)
    if kind == "ci":
        return _invented_ci(arguments, n)
    if kind == "money":
        return {"amount": 100 + (n % 4900), "currency": "USD"}
    return _invented_record(tool, arguments, n, digest)


_FILE_MUTATE = re.compile(
    r"(write|edit|apply|patch|create_file|update_file|replace)", re.I)


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
            # The record's own id comes first so entity identity is the
            # item's id, never an inherited parent reference (matter_id).
            fields = _record_fields(n, i, noun)
            item = {**fields,
                    **{k: v for k, v in known.items() if k not in fields}}
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
    if isinstance(record.get("id"), (str, int)) and record.get("id") != "":
        return "id"
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
        self.schemas: dict[str, dict] = {}
        self.specs: dict[str, dict] = {}
        for tool in tools or []:
            name, params = _tool_schema(tool)
            _, spec = _function_spec(tool)
            if name:
                self.schemas[name] = params
                self.specs[name] = spec
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
        spec = self.specs.get(tool) or {}
        shape = self.result_shapes.get(tool)
        kind = _result_kind(tool, spec, args)
        if isinstance(shape, dict) and shape:
            filled = _fill_template(shape, n)
            for key, val in args.items():
                if key in filled and not isinstance(filled[key], (dict, list)):
                    filled[key] = val
            if not _shape_mismatches_kind(kind, filled):
                for key, val in list(filled.items()):
                    if isinstance(val, list) and val and isinstance(val[0], dict):
                        template = shape.get(key)
                        item_template = (template[0] if isinstance(template, list)
                                         and template and isinstance(template[0], dict)
                                         else None)
                        if kind in {"file", "files", "grep", "shell", "git", "ci"}:
                            filled[key] = list(val)
                        else:
                            filled[key] = [_entity_consistent(item, item_template)
                                           for item in val]
                return {"data": filled}
        invented = _invented_payload(tool, args, n, digest, spec)
        if _result_kind(tool, spec, args) == "money":
            return invented
        return {"data": invented}

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
        if self.world_state in {"entity already acted on", "already_done"} \
                and not _READ.match(tool):
            # Completed entities stay readable; only repeat actions are done.
            return {"status": "already_done", "reason": "already_acted_on"}
        missing = _missing_required(self.schemas[tool], arguments)
        if missing:
            return {"status": "rejected", "reason": "missing_required",
                    "fields": missing}

        references = _reference_values(arguments)
        digest = self._digest(tool, arguments)
        spec = self.specs.get(tool) or {}
        kind = _result_kind(tool, spec, arguments)

        if kind == "file" and (_CREATE.match(tool) or _FILE_MUTATE.search(tool)):
            n = int(digest[:8], 16)
            path = _path_from_args(arguments, n)
            content = str(arguments.get("content") or arguments.get("diff") or "")
            if path:
                self.entities[path] = {"tool": tool, "arguments": arguments,
                                       "version": 1}
            return {"status": "ok", "path": path,
                    "bytes_written": len(content) if content else len(_fake_source(path, n))}

        if _CREATE.match(tool) and kind in {"record", "money"}:
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
