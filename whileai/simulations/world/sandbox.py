"""Schema-derived tool backend: validation, entity state, deterministic results.

Every number and every word list the world answers from is a named
constant in this module or a field of :class:`WorldOptions`, each with the
reason it is what it is (``defaults.py`` holds the numbers). A caller who
wants a different world passes ``MockEnvironment(options=WorldOptions(...))``
or ``simulate(advanced={"world": {...}})``; nothing here has to be edited.

The generators are seeded: the same seed, tool and arguments give the same
record. Changing a default changes the golden rows (``scripts/golden.py``),
so a default moves only with a stated diff.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any

from ..defaults import (
    TEXT_HEURISTICS,
    WORLD_CI_FAIL_ONE_IN,
    WORLD_CONDITION_MODES,
    WORLD_CREATED_ID_MODULUS,
    WORLD_DATE_YEARS,
    WORLD_DEFAULT_FAULT_MODE,
    WORLD_DEFAULT_FAULT_RATE,
    WORLD_EXISTS_SHARE,
    WORLD_EXPRESSION_CHARS,
    WORLD_HINT_CHARS,
    WORLD_ID_RANGE,
    WORLD_ISSUED_ID_HEX,
    WORLD_JITTER_DIVISOR,
    WORLD_MALFORMED_PAYLOAD,
    WORLD_REF_CHARS,
    WORLD_SEARCH_HITS,
    WORLD_SHELL_FLAVORS,
    WORLD_STALE_AS_OF,
    WORLD_TEMPLATE_HITS,
)

# ---------------------------------------------------------------- lexicons
#
# Word lists and key patterns the world reads tool calls with. Module
# constants so a caller can see and extend them; the ones a caller is most
# likely to swap (name pools, the routing table, the fault modes) are also
# fields of WorldOptions.

#: Tool-name prefixes that create, read or delete. Convention: the verbs
#: OpenAI-style tool specs use in practice; a name that matches none is
#: treated as an update.
CREATE_VERBS = re.compile(r"^(create|generate|write|add|upload|insert|make|post|new)", re.I)
READ_VERBS = re.compile(r"^(get|read|list|inspect|search|fetch|find|show|describe|cat|view)", re.I)
DELETE_VERBS = re.compile(r"^(delete|remove|drop|destroy)", re.I)
#: Argument keys whose value names an entity the world tracks (exists,
#: deleted, versioned).
REFERENCE_KEY = re.compile(
    r"(^id$|_id$|^path$|_path$|^file$|^name$|^key$|^ref$|^number$|^asin$|^sku$)",
    re.I,
)
#: Argument values that echo the schema instead of the person's details.
PLACEHOLDER_VALUE = re.compile(
    r"^(?:<[^>]*>|\[[^\]]*\]|\{[^}]*\}|"
    r"(?:(?:your |the |a |an |user |customer |my )(?:email|phone|name|zip|address|id)|"
    r"email address|first name|last name|full name|zip code|phone number|"
    r"order (?:id|number)|user id|customer id|account (?:id|number)|item id|product id|"
    r"string|placeholder|xxx+)"
    r"|[\w.+-]+@(?:example|test|email|domain)\.(?:com|org|net))$",
    re.I,
)
#: A world-issued id: ``<stem>_<12 hex>``. Never pre-exists; it must be created.
ISSUED_ID = re.compile(rf"^[a-z]+_[0-9a-f]{{{WORLD_ISSUED_ID_HEX}}}$")
#: Tool names that mutate a file (write or patch) rather than read it.
FILE_MUTATE = re.compile(r"(write|edit|apply|patch|create_file|update_file|replace)", re.I)

# Old private spellings, kept for the modules that import them.
_CREATE, _READ, _DELETE = CREATE_VERBS, READ_VERBS, DELETE_VERBS
_REFERENCE_KEY, _PLACEHOLDER_VALUE, _ISSUED_ID = REFERENCE_KEY, PLACEHOLDER_VALUE, ISSUED_ID
_FILE_MUTATE = FILE_MUTATE

#: Generic content pools. Domain-neutral on purpose: names, statuses, and
#: dates read plausibly for checks, commits, tickets, students, clauses, or
#: listings. Swap them through ``WorldOptions(people=..., ...)``.
PEOPLE = (
    # Forty invented pairings. First and last names are drawn from
    # different regions on purpose so no pair reads as a person you know.
    "tessa okonkwo",
    "ravi lindgren",
    "mireille tanaka",
    "bao castellanos",
    "ingrid abubakar",
    "kwame sorensen",
    "leila varga",
    "dmitri achebe",
    "noor kavanagh",
    "hiro delacroix",
    "amara fitzgerald",
    "sven nakamura",
    "zainab holmberg",
    "tomasz oyelowo",
    "farah eriksen",
    "kenji abernathy",
    "esperanza kowalski",
    "olu brennan",
    "yara thorsen",
    "matteo nwachukwu",
    "sigrid bhatt",
    "idris halvorsen",
    "priyanka o'rourke",
    "lucas adeyemi",
    "hanna quispe",
    "tariq lindqvist",
    "rosalind mbeki",
    "anselm ravalli",
    "chiara okafor",
    "jonas amankwah",
    "beatriz sundstrom",
    "emeka fontaine",
    "solveig ramaswamy",
    "yusuf mackenzie",
    "ines takahashi",
    "birgit anand",
    "cyrus wanjiru",
    "marisol dybek",
    "elio berhane",
    "ayesha lindstrom",
)
ADJECTIVES = (
    "nightly",
    "routine",
    "primary",
    "draft",
    "updated",
    "automated",
    "manual",
    "initial",
    "final",
    "weekly",
    "legacy",
    "follow-up",
    "quarterly",
    "urgent",
    "archived",
    "revised",
    "secondary",
    "provisional",
    "recurring",
    "expedited",
    "deferred",
    "standing",
    "seasonal",
    "interim",
)
TOPICS = (
    "config",
    "cleanup",
    "handoff",
    "review",
    "rollout",
    "migration",
    "sync",
    "audit",
    "onboarding",
    "renewal",
    "billing",
    "escalation",
    "inventory",
    "compliance",
    "backlog",
    "outreach",
    "reconciliation",
    "staging",
    "triage",
    "closeout",
)
ITEM_STATUSES = (
    "completed",
    "in_progress",
    "pending",
    "failed",
    "active",
    "queued",
    "approved",
    "open",
)
CODE_VERBS = (
    "load",
    "parse",
    "render",
    "sync",
    "build",
    "handle",
    "validate",
    "merge",
    "format",
    "dispatch",
)
CODE_NOUNS = (
    "config",
    "payload",
    "record",
    "client",
    "worker",
    "schema",
    "queue",
    "session",
    "index",
    "router",
)
CHECK_NAMES = ("lint", "tests", "typecheck", "build", "coverage", "security", "format")
FILE_STEMS = ("app", "util", "worker", "client", "schema", "router", "session", "queue")
_PEOPLE, _ADJECTIVES, _TOPICS, _ITEM_STATUSES = PEOPLE, ADJECTIVES, TOPICS, ITEM_STATUSES
_CODE_VERBS, _CODE_NOUNS, _CHECK_NAMES, _FILE_STEMS = (
    CODE_VERBS,
    CODE_NOUNS,
    CHECK_NAMES,
    FILE_STEMS,
)

#: Salts for the seeded picks below: distinct primes so each field of a
#: record draws independently of the others. Changing one changes golden rows.
_SALT_TOPIC, _SALT_TOPIC_I = 97, 104729
_SALT_ADJ, _SALT_ADJ_I = 7, 7919
_SALT_STATUS, _SALT_STATUS_I = 13, 15485863
_SALT_OWNER, _SALT_OWNER_I = 29, 32452843
_SALT_ID_I = 137

#: Record cues: what a tool's name and parameters say about which measured
#: fields a record carries. An inventory check carries a quantity, a price
#: lookup an amount, a booking a date; an owner appears only when the tool is
#: about people.
QUANTITY_CUE = re.compile(
    r"(inventory|stock|quantity|qty|count|available|units?|seats?|rooms?)", re.I
)
MONEY_CUE = re.compile(
    r"(price|cost|fee|amount|balance|total|charge|refund|pay|invoice|bill)", re.I
)
DATE_CUE = re.compile(r"(date|schedule|appointment|booking|deliver|ship|due|calendar|slot)", re.I)
PEOPLE_CUE = re.compile(
    r"(owner|assignee|author|agent|member|team|user|customer|contact|staff)", re.I
)
_QUANTITY_CUE, _MONEY_CUE, _DATE_CUE, _PEOPLE_CUE = QUANTITY_CUE, MONEY_CUE, DATE_CUE, PEOPLE_CUE

#: Key patterns a model-written template is re-drawn by: dates, ids, people.
KEY_DATEISH = re.compile(r"(date|_at$|^at$|time$|day$|when)", re.I)
KEY_IDISH = re.compile(r"(^id$|_id$|number$|^sku$|^ref$|^pnr$|^code$|^asin$)", re.I)
KEY_PERSONISH = re.compile(r"(^from$|owner|author|assignee|sender|^by$)", re.I)
#: Keys that name a record (its title) and are filled from the call's object.
IDENTITY_KEYS = frozenset({"title", "name", "description", "product_name", "summary", "subject"})
#: A locator names which record to read: an id-ish key, a title-ish key, or a date.
LOCATOR_KEYS = IDENTITY_KEYS | {"date"}
_KEY_DATEISH, _KEY_IDISH, _KEY_PERSONISH = KEY_DATEISH, KEY_IDISH, KEY_PERSONISH
_IDENTITY_KEYS, _LOCATOR_KEYS = IDENTITY_KEYS, LOCATOR_KEYS

#: Argument keys read for a file path, a query hint, a directory root, a
#: shell command, a grep pattern, and the list keys a record payload nests
#: items under. Convention: the spellings seen across tool specs.
PATH_KEYS = ("path", "file_path", "filepath", "file", "filename")
HINT_KEYS = ("query", "q", "category", "subject", "pattern", "topic")
ROOT_KEYS = ("path", "directory", "dir")
COMMAND_KEYS = ("command", "cmd", "argv")
PATTERN_KEYS = ("pattern", "query", "regex")
EXPRESSION_KEYS = ("expression", "expr", "formula")
ITEM_LIST_KEYS = ("items", "products", "results")
#: ``images`` fields in a template are rewritten to slugged example URLs.
IMAGE_KEYS = frozenset({"images", "image"})
#: The url prefix generated image links use (RFC 2606 reserved domain).
IMAGE_URL_PREFIX = "https://example.com/images/"

#: World-state phrases a plan may carry, and the canonical state each means.
WORLD_STATES = {
    "entity exists": "exists",
    "exists": "exists",
    "entity missing": "missing",
    "missing": "missing",
    "entity already acted on": "already_done",
    "already_done": "already_done",
}

#: Result families whose payload is code-shaped (file text, shell output,
#: diffs), never a person record. Identity grounding and entity memory skip
#: them.
CODE_KINDS = frozenset({"file", "files", "grep", "shell", "git", "ci"})


# --------------------------------------------------------------- seeded picks


def _pick(pool: Sequence[str], k: int, salt: int) -> str:
    return pool[(k // salt) % len(pool)]


def _hex(n: int, width: int = WORLD_ISSUED_ID_HEX) -> str:
    return hashlib.sha256(f"hex:{n}".encode()).hexdigest()[:width]


def _iso_date(k: int, i: int, years: tuple[int, int] = WORLD_DATE_YEARS) -> str:
    first, span = years
    year = first + (k // 13 + i * 3) % span
    month = 1 + (k // 31 + i) % 12
    # day capped at 28 so every month is valid
    day = 1 + (k + i * 7) % 28
    return f"{year}-{month:02d}-{day:02d}"


def _in_range(x: int, bounds: tuple[int, int]) -> int:
    """``x`` folded into ``[lo, hi)``."""
    lo, hi = bounds
    return lo + x % max(1, hi - lo)


def _count(x: int, hits: tuple[int, int]) -> int:
    """``x`` folded into ``[lo, hi]`` (inclusive: a count of records)."""
    lo, hi = hits
    return lo + x % max(1, hi - lo + 1)


# -------------------------------------------------------------- schema reads


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
                missing.extend(f"{key}.{grand}" for grand in _missing_required(child, value))
    return missing


def placeholder_arguments(arguments: Any) -> list[str]:
    """Argument leaves that echo the schema instead of the person's details.

    A model trained on rows where the world accepted "first name" or
    user@example.com learns to call tools with the schema itself. The world
    refuses them so the row shows the correction, not the habit. A bare
    word that could be an enum value ("email" as a contact channel) is not
    a placeholder; only the schema-echo phrasings are (``PLACEHOLDER_VALUE``).
    """
    out: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            s = node.strip()
            if s and PLACEHOLDER_VALUE.match(s):
                out.append(path)

    walk(arguments, "")
    return out


def _reference_values(arguments: Any, *, prefix: str = "") -> list[tuple[str, str]]:
    references = []
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            label = f"{prefix}{key}"
            if isinstance(value, (dict, list)):
                references.extend(_reference_values(value, prefix=label + "."))
            elif (
                isinstance(value, (str, int))
                and value != ""
                and value is not None
                and REFERENCE_KEY.search(str(key))
            ):
                references.append((label, str(value)))
    elif isinstance(arguments, list):
        for item in arguments:
            references.extend(_reference_values(item, prefix=prefix))
    return references


def _item_noun(tool: str) -> str:
    """Singular object noun from the tool name: list_commits -> commit."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", str(tool).lower()) if t]
    rest = tokens[1:] or tokens
    noun = rest[-1] if rest else ""
    if (
        len(noun) >= TEXT_HEURISTICS.plural_noun_min_chars
        and noun.endswith("s")
        and not noun.endswith("ss")
    ):
        noun = noun[:-1]
    return noun


def _function_spec(tool: dict) -> tuple[str, dict]:
    function = tool.get("function", tool) if isinstance(tool, dict) else {}
    if not isinstance(function, dict):
        function = {}
    name = str(function.get("name", ""))
    return name, function


def _param_keys(spec: dict, arguments: dict | None = None) -> set[str]:
    props = (spec.get("parameters") or {}).get("properties") or {}
    keys = {str(k).lower() for k in props}
    if arguments:
        keys |= {str(k).lower() for k in arguments}
    return keys


# ------------------------------------------------------- result-kind routing
#
# Which payload family a tool returns, read off its name, its parameter
# names and its description. Not a product list: a new spec with unknown
# tool names still maps (path + read -> file, command -> shell, otherwise a
# record). The table is a tuple of (kind, rule) checked in order; a caller
# extends it with ``WorldOptions(result_kinds=RESULT_KINDS + ((kind, rule),))``
# and gives the new kind a payload builder in ``payloads``.

#: A routing rule: ``rule(name, keys, blob) -> bool`` where ``name`` is the
#: lowercased tool name, ``keys`` its parameter names (plus the call's
#: argument names) and ``blob`` the name and description together.
KindRule = Callable[[str, set[str], str], bool]

SHELL_TOOLS = re.compile(
    r"(run_command|run_terminal|bash|shell_command|execute_command|terminal_cmd|run_bash|^exec$|"
    r"run_tests?$|pytest)"
)
SHELL_VERBS = re.compile(r"(run|exec|shell|bash|terminal)")
GREP_TOOLS = re.compile(r"(grep|ripgrep|search_code|code_search|find_in_files|search_files)")
GREP_SCOPE_KEYS = frozenset({"path", "glob", "directory", "dir"})
FILE_TOOLS = re.compile(
    r"(read_file|write_file|get_file|cat_file|view_file|edit_file|apply_patch|replace_in_file|"
    r"open_file|file_contents)"
)
FILE_KEYS = frozenset({"path", "file", "file_path", "filepath", "filename"})
FILE_VERBS = re.compile(r"(read|write|get|cat|view|edit|open|apply|patch|contents|file)")
#: A file-ish key on a record tool (get_user_profile(path=...)) is still a record.
RECORD_NOT_FILE = re.compile(r"(profile|(?:^|_)user(?:_|$)|order|issue|(?:^|_)prs?(?:_|$)|account)")
DIR_TOOLS = re.compile(r"(list_dir|list_files|glob_files|ls_dir)")
DIR_KEYS = frozenset({"path", "directory", "dir", "glob"})
RECORD_NOT_DIR = re.compile(r"(order|issue|check|commit|event)")
GIT_TOOLS = re.compile(r"(git_status|git_diff|git_log|git_show|git_blame)")
COMMIT_TOOLS = re.compile(r"(^list_commits$|_commits$|commit_log)")
CI_TOOLS = re.compile(r"(list_checks|_checks$|ci_status|test_results)")
CI_WORDS = re.compile(r"\b(ci|check run)")
CI_KEYS = frozenset({"sha", "commit", "number"})
MONEY_TOOLS = re.compile(r"(payment|price|invoice|balance|estimate|fx_rate|amount_due)")


def _is_shell(name: str, keys: set[str], blob: str) -> bool:
    return bool(SHELL_TOOLS.search(name)) or bool(
        (keys & set(COMMAND_KEYS)) and SHELL_VERBS.search(name)
    )


def _is_grep(name: str, keys: set[str], blob: str) -> bool:
    return bool(GREP_TOOLS.search(name)) or bool(
        ("pattern" in keys or "regex" in keys) and keys & GREP_SCOPE_KEYS
    )


def _is_file(name: str, keys: set[str], blob: str) -> bool:
    return bool(FILE_TOOLS.search(name)) or bool(
        keys & FILE_KEYS and FILE_VERBS.search(name) and not RECORD_NOT_FILE.search(name)
    )


def _is_files(name: str, keys: set[str], blob: str) -> bool:
    return bool(DIR_TOOLS.search(name)) or bool(
        name.startswith("list") and keys & DIR_KEYS and not RECORD_NOT_DIR.search(name)
    )


def _is_git(name: str, keys: set[str], blob: str) -> bool:
    return name.startswith("git") or bool(GIT_TOOLS.search(name)) or bool(COMMIT_TOOLS.search(name))


def _is_ci(name: str, keys: set[str], blob: str) -> bool:
    return bool(CI_TOOLS.search(name)) or bool(CI_WORDS.search(blob) and keys & CI_KEYS)


def _is_money(name: str, keys: set[str], blob: str) -> bool:
    return bool(MONEY_TOOLS.search(name))


#: The routing table, in order. Anything unmatched is a ``record``.
RESULT_KINDS: tuple[tuple[str, KindRule], ...] = (
    ("shell", _is_shell),
    ("grep", _is_grep),
    ("file", _is_file),
    ("files", _is_files),
    ("git", _is_git),
    ("ci", _is_ci),
    ("money", _is_money),
)


def _result_kind(
    tool: str,
    spec: dict | None = None,
    arguments: dict | None = None,
    options: WorldOptions | None = None,
) -> str:
    """Infer the payload family from name, params, and description.

    Walks ``options.result_kinds`` (``RESULT_KINDS`` by default) in order;
    the first rule that fires names the kind, and nothing fires means
    ``record``.
    """
    name = str(tool or "").lower()
    spec = spec if isinstance(spec, dict) else {}
    desc = str(spec.get("description") or "").lower()
    keys = _param_keys(spec, arguments)
    blob = f"{name} {desc}"
    table = (options or DEFAULT_WORLD).result_kinds
    for kind, rule in table:
        if rule(name, keys, blob):
            return kind
    return "record"


#: Keys a model-written example must carry to count as each code-shaped kind.
KIND_SHAPE_KEYS = {
    "shell": frozenset({"stdout", "stderr", "exit_code", "command"}),
    "file": frozenset({"content", "diff", "text", "bytes"}),
    "grep": frozenset({"matches"}),
    "files": frozenset({"entries", "files", "items"}),
    "ci": frozenset({"items", "checks", "jobs", "conclusion"}),
}
_RECORD_MARKERS = frozenset({"owner", "updated_at"})
_CODE_MARKERS = frozenset({"content", "stdout", "diff", "matches", "entries"})


def _shape_mismatches_kind(kind: str, filled: dict) -> bool:
    """True when an LLM example is a CRM record for a file/shell/git tool."""
    if kind not in CODE_KINDS:
        return False
    if not isinstance(filled, dict):
        return True
    keys = {str(k).lower() for k in filled}
    need = KIND_SHAPE_KEYS.get(kind)
    if need is not None and not (keys & need):
        return True
    return bool(keys >= _RECORD_MARKERS and not keys & _CODE_MARKERS)


# ------------------------------------------------------- code-shaped payloads


def _path_from_args(
    arguments: dict, n: int, ext: str = "py", options: WorldOptions | None = None
) -> str:
    o = options or DEFAULT_WORLD
    for key in PATH_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"src/{_pick(o.file_stems, n, 5)}.{ext}"


def _fake_source(path: str, n: int, options: WorldOptions | None = None) -> str:
    """Plausible file text. Deterministic in (path, n). Never executed."""
    o = options or DEFAULT_WORLD
    ext = ""
    if "." in path.rsplit("/", 1)[-1]:
        ext = path.rsplit(".", 1)[-1].lower()
    verb = _pick(o.code_verbs, n, 3)
    noun = _pick(o.code_nouns, n, 11)
    if ext in {"json"}:
        return f'{{\n  "{noun}": "{verb}",\n  "enabled": true,\n  "retries": {1 + n % 4}\n}}\n'
    if ext in {"yml", "yaml", "toml"}:
        return f"{noun}:\n  mode: {verb}\n  retries: {1 + n % 4}\n"
    if ext in {"log", "txt"}:
        stamp = _iso_date(n, 0, o.date_years)
        return (
            f"{stamp}T09:00:00Z INFO {verb}_{noun} started\n"
            f"{stamp}T09:00:01Z INFO handled 12 items\n"
            f"{stamp}T09:00:02Z WARN retry {1 + n % 3}\n"
        )
    if ext in {"diff", "patch"}:
        return _fake_diff(path, n, o)
    if ext in {"md"}:
        return (
            f"# {noun}\n\nThe {verb} path lives in `{path}`.\n\n"
            f"- retries: {1 + n % 4}\n- owner queue: {noun}\n"
        )
    return (
        f"# {path}\n"
        f"from {noun} import {verb}_state\n\n"
        f"def {verb}_{noun}(value):\n"
        f"    if not value:\n"
        f"        raise ValueError({noun!r})\n"
        f"    return {verb}_state(value)\n"
    )


def _fake_diff(path: str, n: int, options: WorldOptions | None = None) -> str:
    o = options or DEFAULT_WORLD
    verb = _pick(o.code_verbs, n, 3)
    noun = _pick(o.code_nouns, n, 11)
    return (
        f"--- a/{path}\n+++ b/{path}\n"
        f"@@ -1,4 +1,6 @@\n"
        f" def {verb}_{noun}(value):\n"
        f"-    return value\n"
        f"+    if not value:\n"
        f"+        return {noun!r}\n"
        f"+    return value\n"
    )


def _invented_file(tool: str, arguments: dict, n: int, digest: str, o: WorldOptions) -> dict:
    path = _path_from_args(arguments, n, options=o)
    content = _fake_source(path, n, o)
    return {"path": path, "content": content, "bytes": len(content), "encoding": "utf-8"}


def _invented_files(tool: str, arguments: dict, n: int, digest: str, o: WorldOptions) -> dict:
    root = str(next((arguments.get(k) for k in ROOT_KEYS if arguments.get(k)), None) or "src")
    count = 3 + n % 3
    entries = []
    for i in range(count):
        stem = _pick(o.file_stems, n + i * 17, 5)
        kind = "dir" if i == 0 and n % 5 == 0 else "file"
        name = stem if kind == "dir" else f"{stem}.py"
        entries.append({"name": name, "type": kind, "path": f"{root.rstrip('/')}/{name}"})
    return {"path": root, "entries": entries, "count": count}


def _invented_grep(tool: str, arguments: dict, n: int, digest: str, o: WorldOptions) -> dict:
    pattern = str(
        next((arguments.get(k) for k in PATTERN_KEYS if arguments.get(k)), None) or "TODO"
    )
    count = 2 + n % 3
    matches = []
    for i in range(count):
        path = f"src/{_pick(o.file_stems, n + i * 13, 5)}.py"
        line = 8 + ((n + i * 19) % 80)
        verb = _pick(o.code_verbs, n + i, 3)
        noun = _pick(o.code_nouns, n + i, 11)
        matches.append({"path": path, "line": line, "text": f"    {verb}_{noun}({pattern!r})"})
    return {"pattern": pattern, "count": count, "matches": matches}


def _pytest_stdout(collected: int, failed: bool, n: int) -> str:
    """The text a pytest run prints. One failing shape, one passing shape."""
    head = "============================= test session starts ==============================\n"
    if failed:
        return (
            head + f"collected {collected} items\n"
            "tests/test_app.py .....F\n"
            "FAILED tests/test_app.py::test_sync - AssertionError: expected 1\n"
            "=========================== 1 failed, 5 passed in 0.51s ========================\n"
        )
    return (
        head + f"collected {collected} items\n"
        f"tests/test_app.py {'.' * min(collected, 8)}\n"
        f"============================== {collected} passed in 0.42s "
        "===============================\n"
    )


#: The shell flavor (``n % shell_flavors``) that answers with a directory
#: listing; flavors 0, 1 and 2 are the permission error, the failing test
#: run and the merge conflict, the rest pass.
_SHELL_LS_FLAVOR = 3


def _invented_shell(tool: str, arguments: dict, n: int, digest: str, o: WorldOptions) -> dict:
    command = str(
        next((arguments.get(k) for k in COMMAND_KEYS if arguments.get(k)), None) or "true"
    )
    flavor = n % o.shell_flavors
    if flavor == 0:
        return {
            "command": command,
            "exit_code": 1,
            "stdout": "",
            "stderr": "bash: src/secret.key: Permission denied\n",
        }
    if flavor == 1:
        return {
            "command": command,
            "exit_code": 1,
            "stdout": _pytest_stdout(6 + n % 5, True, n),
            "stderr": "",
        }
    if flavor == 2:  # noqa: PLR2004  # the third shell flavor, after 0 and 1
        return {
            "command": command,
            "exit_code": 1,
            "stdout": (
                "Auto-merging src/app.py\n"
                "CONFLICT (content): Merge conflict in src/app.py\n"
                "Automatic merge failed; fix conflicts and then commit the result.\n"
            ),
            "stderr": "",
        }
    if "ls" in command or flavor == _SHELL_LS_FLAVOR:
        return {
            "command": command,
            "exit_code": 0,
            "stdout": "src/\n  app.py\n  util.py\nREADME.md\n",
            "stderr": "",
        }
    return {
        "command": command,
        "exit_code": 0,
        "stdout": _pytest_stdout(6 + n % 6, False, n),
        "stderr": "",
    }


def _invented_git(tool: str, arguments: dict, n: int, digest: str, o: WorldOptions) -> dict:
    name = str(tool or "").lower()
    branch = f"feature/{_pick(o.code_nouns, n, 11)}"
    if "diff" in name or "show" in name:
        path = _path_from_args(arguments, n, options=o)
        return {"path": path, "diff": _fake_diff(path, n, o), "branch": branch}
    if "log" in name or "commit" in name:
        count = 2 + n % 2
        items = []
        for i in range(count):
            items.append(
                {
                    "sha": _hex(n + i * 29, WORLD_ISSUED_ID_HEX),
                    "message": f"{_pick(o.code_verbs, n + i, 3)} {_pick(o.code_nouns, n + i, 11)}",
                    "author": _pick(o.people, n + i * 7, _SALT_OWNER),
                    "date": _iso_date(n, i, o.date_years),
                }
            )
        return {"branch": branch, "count": count, "items": items}
    staged = [f"src/{_pick(o.file_stems, n, 5)}.py"]
    unstaged = [f"src/{_pick(o.file_stems, n + 3, 5)}.py"]
    return {
        "branch": branch,
        "clean": False,
        "ahead": 1 + n % 3,
        "behind": n % 2,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": ["tmp.log"],
    }


def _invented_ci(tool: str, arguments: dict, n: int, digest: str, o: WorldOptions) -> dict:
    count = 3 + n % 2
    items = []
    for i in range(count):
        failed = n % o.ci_fail_one_in == 0 and i == count - 1
        items.append(
            {
                "name": o.check_names[i % len(o.check_names)],
                "status": "completed",
                "conclusion": "failure" if failed else "success",
                "duration_s": 8 + ((n + i * 11) % 90),
            }
        )
    return {"count": count, "items": items}


def _invented_money(tool: str, arguments: dict, n: int, digest: str, o: WorldOptions) -> dict:
    # Spread across magnitudes; cents on most. Measured: a flat 100-4999
    # band taught models that money is always a small round number.
    scale = (1, 1, 10, 100)[(n // 7) % 4]
    cents = (n // 3) % 100 if n % 3 else 0
    return {"amount": (17 + n % 483) * scale + cents / 100, "currency": "USD"}


def _invented_record_payload(
    tool: str, arguments: dict, n: int, digest: str, o: WorldOptions
) -> dict:
    return _invented_record(tool, arguments, n, digest, o)


#: A payload builder: ``build(tool, arguments, n, digest, options) -> dict``.
PayloadBuilder = Callable[[str, dict, int, str, "WorldOptions"], dict]

#: Builder per result kind. A kind with no builder answers with a record.
#: Read-only; add a builder with ``WorldOptions(payloads={**PAYLOAD_BUILDERS, ...})``.
PAYLOAD_BUILDERS: Mapping[str, PayloadBuilder] = MappingProxyType(
    {
        "file": _invented_file,
        "files": _invented_files,
        "grep": _invented_grep,
        "shell": _invented_shell,
        "git": _invented_git,
        "ci": _invented_ci,
        "money": _invented_money,
        "record": _invented_record_payload,
    }
)


def _invented_payload(
    tool: str,
    arguments: dict,
    n: int,
    digest: str,
    spec: dict | None = None,
    options: WorldOptions | None = None,
) -> dict[str, Any]:
    o = options or DEFAULT_WORLD
    kind = _result_kind(tool, spec, arguments, o)
    build = o.payloads.get(kind) or o.payloads.get("record") or _invented_record_payload
    return build(tool, arguments, n, digest, o)


# ------------------------------------------------------------- record payloads


def _record_fields(
    k: int, i: int, noun: str = "", cue: str = "", options: WorldOptions | None = None
) -> dict[str, Any]:
    """One plausible record body. Deterministic in (k, i).

    Items are named after the tool's own noun so a product search returns
    product-shaped names, not workflow vocabulary. The tool's name and
    parameters decide which measurable fields exist (``QUANTITY_CUE``,
    ``MONEY_CUE``, ``DATE_CUE``, ``PEOPLE_CUE``): an inventory check carries
    a quantity, a price lookup an amount, a booking a date. An owner appears
    only when the tool is about people; a flower inventory does not have one.
    """
    o = options or DEFAULT_WORLD
    label = noun or _pick(o.topics, k + i * _SALT_TOPIC_I, _SALT_TOPIC)
    out: dict[str, Any] = {
        "id": _in_range(k + i * _SALT_ID_I, o.id_range),
        "name": f"{_pick(o.adjectives, k + i * _SALT_ADJ_I, _SALT_ADJ)} {label}",
        "status": _pick(o.statuses, k + i * _SALT_STATUS_I, _SALT_STATUS),
    }
    blob = f"{cue} {noun}"
    if QUANTITY_CUE.search(blob):
        out["quantity"] = (k + i * 31) % 240
    if MONEY_CUE.search(blob):
        scale = (1, 10, 100)[(k + i) % 3]
        out["amount"] = round(((k + i * 53) % 900 + 12) * scale / 10, 2)
        out["currency"] = "USD"
    if DATE_CUE.search(blob):
        out["date"] = _iso_date(k + 3, i, o.date_years)
    # workflow records keep an owner; physical or priced items do not,
    # unless the tool itself is about people
    if PEOPLE_CUE.search(blob) or not (QUANTITY_CUE.search(blob) or MONEY_CUE.search(blob)):
        out["owner"] = _pick(o.people, k + i * _SALT_OWNER_I, _SALT_OWNER)
    out["updated_at"] = _iso_date(k, i, o.date_years)
    return out


def _evaluate_expression(tool: str, arguments: dict) -> dict[str, Any] | None:
    """Real arithmetic for calculator tools. Junk math results taught
    agents to do mental math, and they got it wrong (measured: 7 of 9
    calculate answers in one airline run). Non-math tools return None."""
    expr = None
    for key in EXPRESSION_KEYS:
        if isinstance(arguments.get(key), str):
            expr = arguments[key]
            break
    if expr is None or not re.search(r"^(calc|compute|eval|math)", tool, re.I):
        return None
    import ast
    import operator as op

    ops: dict[type, Any] = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
        ast.FloorDiv: op.floordiv,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    def walk(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.operand))
        raise ValueError("unsupported")

    shown = str(expr)[:WORLD_EXPRESSION_CHARS]
    try:
        value = walk(ast.parse(expr.strip(), mode="eval"))
    except Exception:
        return {"status": "rejected", "reason": "invalid_expression", "expression": shown}
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return {"status": "ok", "data": {"expression": shown, "result": value}}


def _hint_from_args(arguments: dict | None) -> str:
    """Caller-named object, if any. Used to fill identity strings per call."""
    for key in HINT_KEYS:
        val = (arguments or {}).get(key)
        if isinstance(val, str) and len(val.strip()) >= 2:  # noqa: PLR2004  # a one-character hint is noise
            return re.sub(r"\s+", " ", val.strip())[:WORLD_HINT_CHARS]
    return ""


def _identity_label(hint: str, n: int, i: int, noun: str, o: WorldOptions) -> str:
    adj = _pick(o.adjectives, n + i * _SALT_ADJ_I, _SALT_ADJ)
    stem = hint or noun or "item"
    return f"{adj} {stem}"


def _ground_identity(
    data: Any, arguments: dict, n: int, tool: str, options: WorldOptions | None = None
) -> Any:
    """Replace frozen example titles with this call's object. Not a catalog."""
    o = options or DEFAULT_WORLD
    noun = _item_noun(tool)
    hint = _hint_from_args(arguments)

    def walk(obj: Any, i: int) -> Any:
        if isinstance(obj, list):
            return [walk(item, i * 7 + j + 1) for j, item in enumerate(obj)]
        if not isinstance(obj, dict):
            return obj
        out = dict(obj)
        label = _identity_label(hint, n, i, noun, o)
        for key, val in list(out.items()):
            lk = str(key).lower()
            if lk in IDENTITY_KEYS and isinstance(val, str):
                out[key] = label if lk != "description" else f"{label}."
            elif lk in IMAGE_KEYS and isinstance(val, list) and val:
                slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "item"
                out[key] = [f"{IMAGE_URL_PREFIX}{slug}-{j + 1}.jpg" for j, _ in enumerate(val)]
            elif isinstance(val, (dict, list)):
                out[key] = walk(val, i)
        return out

    return walk(data, 0)


def _record_entries(data: dict) -> list[dict]:
    entries: list[dict] = []
    nested = False
    for key in ITEM_LIST_KEYS:
        val = data.get(key)
        if isinstance(val, list) and val and isinstance(val[0], dict):
            nested = True
            entries.extend(item for item in val if isinstance(item, dict))
    if not nested:
        entries.append(data)
    return entries


def _invented_record(
    tool: str, arguments: dict, n: int, digest: str, options: WorldOptions | None = None
) -> dict[str, Any]:
    """Domain-shaped read result. Content varies with the exact arguments.

    Listings return several named records with statuses, owners, and dates so
    the agent has something real to quote instead of fabricating. No default
    repo, PR, or issue is invented; caller references are echoed as context.
    """
    o = options or DEFAULT_WORLD
    known = {k: v for k, v in arguments.items() if v not in (None, "")}
    name = str(tool or "").lower()
    noun = _item_noun(name)
    cue = name + " " + " ".join(str(k) for k in arguments)
    if "search" in name or name.startswith("list"):
        count = _count(n, o.search_hits)
        items = []
        for i in range(count):
            # The record's own id comes first so entity identity is the
            # item's id, never an inherited parent reference (matter_id).
            # A hit's fields then derive from that id (_entity_consistent),
            # so a caller argument only fills a key the hit lacks; it never
            # overrides a generated one, locator or finding.
            fields_ = _record_fields(n, i, noun, cue, o)
            item = {**fields_, **{k: v for k, v in known.items() if k not in fields_}}
            items.append(_entity_consistent(item, None, noun, o))
        return {
            "query": str(arguments.get("query") or arguments.get("q") or ""),
            "count": count,
            "items": items,
        }
    record = dict(_record_fields(n, 0, noun, cue, o))
    # A caller argument is one of two things. A locator (id, name, date) says
    # which record to read, and the record echoes it: reading the locator back
    # is how the agent knows it got the record it asked for, and a get_user
    # (id="u_42") that answers id 1007 is the wrong record. A finding (status,
    # owner, quantity, amount) says what the record contains, and only the
    # generated record decides that. A blanket update let a finding through:
    # get_ticket(owner="alice") came back owned by alice and check_inventory
    # (quantity=500) came back with 500 in stock, each a fact the agent
    # manufactured by naming it. A rubric that checks the reply against tool
    # output then scores the fabrication as grounded, because the claim really
    # is in a tool result. Keys the record lacks are filled either way.
    record.update({k: v for k, v in known.items() if k not in record or _is_locator(k)})
    if digest:
        record.setdefault("ref", digest[:WORLD_REF_CHARS])
    return record


def _is_locator(key: Any) -> bool:
    """A locator names which record to read: an id-ish key, a title-ish key
    (``IDENTITY_KEYS``), or a date. Every other key describes the record's
    state and is a finding the record decides for itself."""
    lk = str(key).lower()
    return lk in LOCATOR_KEYS or bool(KEY_IDISH.search(lk))


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
        if KEY_IDISH.search(str(key)) and isinstance(val, (str, int)) and val != "":
            return str(key)
    return ""


def _entity_consistent(
    record: dict, template: Any = None, noun: str = "", options: WorldOptions | None = None
) -> dict:
    """Re-derive a record's fields from its own id so reads agree with it."""
    o = options or DEFAULT_WORLD
    key = _idish_key(record)
    if not key:
        return record
    ident = record[key]
    seed = _entity_seed(ident)
    if isinstance(template, dict):
        fresh = _fill_template(template, seed, options=o)
    else:
        fresh = _record_fields(seed, 0, noun, options=o)
    fresh = dict(fresh)
    fresh.update({k: v for k, v in record.items() if k not in fresh})
    fresh[key] = ident
    return fresh


def _fill_template(
    value: Any, k: int, i: int = 0, key: str = "", options: WorldOptions | None = None
) -> Any:
    """Instantiate an LLM-written example result. Deterministic in (k, i).

    Field names and free text stay as the model wrote them; ids, dates,
    numbers, and people vary with the exact call so repeat calls with
    different arguments return different records. A number moves by up to
    one part in ``options.jitter_divisor`` of itself.
    """
    o = options or DEFAULT_WORLD
    if isinstance(value, dict):
        return {kk: _fill_template(vv, k, i, str(kk), o) for kk, vv in value.items()}
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            count = _count(k + i, o.template_hits)
            return [_fill_template(value[0], k, i * 7 + j + 1, options=o) for j in range(count)]
        return list(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) <= 1:
            return value
        span = max(1, abs(value) // o.jitter_divisor)
        return value + ((k + i * 131) % (2 * span + 1)) - span
    if isinstance(value, float):
        span = max(1, int(abs(value) * 10) // o.jitter_divisor)
        tenths = int(value * 10) + ((k + i * 131) % (2 * span + 1)) - span
        return round(tenths / 10.0, 2)
    if isinstance(value, str):
        if KEY_PERSONISH.search(key):
            person = _pick(o.people, k + i * 29, 7)
            if "@" in value:
                domain = value.split("@", 1)[1]
                return person.replace(" ", ".") + "@" + domain
            return person
        if KEY_IDISH.search(key) and any(c.isdigit() for c in value):
            return re.sub(r"\d+", str(_in_range(k + i * _SALT_ID_I, o.id_range)), value, count=1)
        if KEY_DATEISH.search(key) and re.search(r"\d{4}-\d{2}-\d{2}", value):
            return _iso_date(k, i, o.date_years) + value[10:]
        return value
    return value


def _created_record(
    tool: str, arguments: dict, n: int, digest: str, options: WorldOptions | None = None
) -> dict[str, Any]:
    """Echo the create call plus an issued id. Never sets a status field."""
    known = {k: v for k, v in arguments.items() if v not in (None, "")}
    name = str(tool or "").lower()
    stem = name.split("_")[0] or "item"
    issued = f"{stem}_{n % WORLD_CREATED_ID_MODULUS}"
    if not known:
        return {"id": issued, "ok": True}
    known.setdefault("id", issued)
    if digest:
        known.setdefault("ref", digest[:WORLD_REF_CHARS])
    return known


# ------------------------------------------------------------- fault modes
#
# A fault mode is ``build(env, tool, arguments) -> dict``: the result the
# world answers with when the plan fires. The four shipped modes are the
# runtime failures the injection benchmarks agree on (timeout, malformed
# payload, stale read, permission denied; arXiv:2605.11928 and
# arXiv:2606.01416 both list them). Add one with
# ``WorldOptions(fault_modes={**FAULT_MODES, "rate_limited": build})``.

FaultMode = Callable[["MockEnvironment", str, dict], dict]


def _fault_timeout(env: MockEnvironment, tool: str, arguments: dict) -> dict:
    return {"status": "timeout", "error": "request timed out"}


def _fault_malformed(env: MockEnvironment, tool: str, arguments: dict) -> dict:
    return {"status": "ok", "data": env.options.malformed_payload}


def _fault_stale(env: MockEnvironment, tool: str, arguments: dict) -> dict:
    # A stale answer is a real record as of some days ago, not a hash. The
    # hash taught agents to invent a shipment, an offer id or a passing
    # test suite around it, and a rubric judge passed them: hallucination
    # labeled as good behavior.
    record = env._payload(tool, env._digest(tool, arguments), arguments)
    return {"status": "ok", **record, "stale": True, "as_of": env.options.stale_as_of}


def _fault_permission_denied(env: MockEnvironment, tool: str, arguments: dict) -> dict:
    return {"status": "permission_denied"}


#: The shipped fault modes, mode name -> builder. Read-only; add one with
#: ``WorldOptions(fault_modes={**FAULT_MODES, "rate_limited": build})``.
FAULT_MODES: Mapping[str, FaultMode] = MappingProxyType(
    {
        "timeout": _fault_timeout,
        "malformed": _fault_malformed,
        "stale": _fault_stale,
        "permission_denied": _fault_permission_denied,
    }
)


# ------------------------------------------------------------- the options


@dataclass(frozen=True)
class WorldOptions:
    """Everything the mock world can be steered by. Defaults in ``defaults.py``.

    Pass one to ``MockEnvironment(options=)``, or the same fields as a dict
    through ``simulate(advanced={"world": {...}})`` and
    ``export_environment(world={...})``; unknown keys raise so a typo is not
    a silent default. The instance is frozen and its mapping fields are
    read-only views: to add a fault mode or a payload builder, build a new
    ``WorldOptions`` with the wider table.

    Attributes:
        fault_modes: mode name -> builder; ``FAULT_MODES`` plus whatever a
            caller adds. A read-only mapping.
        condition_modes: coverage-grid ``tool_condition`` value -> the
            fault mode a situation with that condition carries
            (``WORLD_CONDITION_MODES``). A condition that is itself a
            ``fault_modes`` key needs no entry; every value here must be
            a ``fault_modes`` key. A read-only mapping.
        default_fault_mode: the mode a fault plan gets when it names none.
        default_fault_rate: the fire probability a plan gets when it names
            none.
        stale_as_of: the age stamped on a stale read.
        malformed_payload: what a malformed fault returns.
        exists_share: share of user-named references that exist when no
            world state says otherwise.
        search_hits: records a search returns, inclusive bounds.
        template_hits: records a model-written list template expands to,
            inclusive bounds.
        id_range: generated record ids, ``[lo, hi)``.
        date_years: ``(first year, span)`` generated dates fall in.
        jitter_divisor: a template number moves by up to 1/this of itself.
        shell_flavors: one shell call in this many takes each failing
            flavor; the rest pass.
        ci_fail_one_in: one CI listing in this many carries a failed check.
        result_kinds: the routing table, ``(kind, rule)`` in order.
        payloads: kind -> payload builder. A read-only mapping.
        people, adjectives, topics, statuses, code_verbs, code_nouns,
            check_names, file_stems: the content pools generated records
            draw from.
    """

    fault_modes: Mapping[str, FaultMode] = field(default_factory=lambda: FAULT_MODES)
    condition_modes: Mapping[str, str] = field(default_factory=lambda: WORLD_CONDITION_MODES)
    default_fault_mode: str = WORLD_DEFAULT_FAULT_MODE
    default_fault_rate: float = WORLD_DEFAULT_FAULT_RATE
    stale_as_of: str = WORLD_STALE_AS_OF
    malformed_payload: Any = WORLD_MALFORMED_PAYLOAD
    exists_share: float = WORLD_EXISTS_SHARE
    search_hits: tuple[int, int] = WORLD_SEARCH_HITS
    template_hits: tuple[int, int] = WORLD_TEMPLATE_HITS
    id_range: tuple[int, int] = WORLD_ID_RANGE
    date_years: tuple[int, int] = WORLD_DATE_YEARS
    jitter_divisor: int = WORLD_JITTER_DIVISOR
    shell_flavors: int = WORLD_SHELL_FLAVORS
    ci_fail_one_in: int = WORLD_CI_FAIL_ONE_IN
    result_kinds: tuple[tuple[str, KindRule], ...] = RESULT_KINDS
    payloads: Mapping[str, PayloadBuilder] = field(default_factory=lambda: PAYLOAD_BUILDERS)
    people: tuple[str, ...] = PEOPLE
    adjectives: tuple[str, ...] = ADJECTIVES
    topics: tuple[str, ...] = TOPICS
    statuses: tuple[str, ...] = ITEM_STATUSES
    code_verbs: tuple[str, ...] = CODE_VERBS
    code_nouns: tuple[str, ...] = CODE_NOUNS
    check_names: tuple[str, ...] = CHECK_NAMES
    file_stems: tuple[str, ...] = FILE_STEMS

    def __post_init__(self) -> None:
        # The mapping fields are read-only views whatever was passed, so
        # DEFAULT_WORLD (and every options object) cannot be edited in place.
        for name in ("fault_modes", "condition_modes", "payloads"):
            value = getattr(self, name)
            if not isinstance(value, MappingProxyType):
                object.__setattr__(self, name, MappingProxyType(dict(value)))
        if not 0.0 <= float(self.exists_share) <= 1.0:
            raise ValueError("exists_share is a share in [0, 1]")
        if not 0.0 <= float(self.default_fault_rate) <= 1.0:
            raise ValueError("default_fault_rate is a probability in [0, 1]")
        for name in ("search_hits", "template_hits", "id_range"):
            lo, hi = getattr(self, name)
            if int(lo) > int(hi):
                raise ValueError(f"{name}: (lo, hi) with lo <= hi, got {(lo, hi)!r}")
        if int(self.date_years[1]) < 1:
            raise ValueError("date_years: (first year, span) with span 1 or more")
        if (
            int(self.jitter_divisor) < 1
            or int(self.shell_flavors) < 1
            or int(self.ci_fail_one_in) < 1
        ):
            raise ValueError("jitter_divisor, shell_flavors and ci_fail_one_in are 1 or more")
        if self.default_fault_mode not in self.fault_modes:
            raise ValueError(
                f"default_fault_mode {self.default_fault_mode!r} is not in fault_modes "
                f"({', '.join(sorted(self.fault_modes))})"
            )
        unknown_modes = sorted(
            f"{cond} -> {mode}"
            for cond, mode in self.condition_modes.items()
            if mode not in self.fault_modes
        )
        if unknown_modes:
            raise ValueError(
                f"condition_modes names a mode that is not in fault_modes: "
                f"{', '.join(unknown_modes)}; fault_modes are "
                f"{', '.join(sorted(self.fault_modes))}. Add the builder to fault_modes "
                "or point the condition at a shipped mode."
            )

        for name in (
            "people",
            "adjectives",
            "topics",
            "statuses",
            "code_verbs",
            "code_nouns",
            "check_names",
            "file_stems",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name}: a non-empty tuple of words")

    def fault_mode_for(self, condition: str) -> str | None:
        """The fault mode a coverage-grid ``tool_condition`` carries:
        ``condition_modes[condition]``, else the condition itself when it
        is a ``fault_modes`` key, else ``None`` (a clean call)."""
        mode = self.condition_modes.get(condition)
        if mode is None and condition in self.fault_modes:
            mode = condition
        return mode

    @classmethod
    def coerce(cls, value: WorldOptions | Mapping[str, Any] | None) -> WorldOptions:
        """A ``WorldOptions`` from itself, a dict of its fields, or ``None``."""
        if value is None:
            return DEFAULT_WORLD
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                f"world options: a WorldOptions or a dict of its fields, got {type(value).__name__}"
            )
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(
                f"unknown world option(s) {unknown}; the fields are {', '.join(sorted(known))}"
            )
        fixed = dict(value)
        for name in ("search_hits", "template_hits", "id_range", "date_years"):
            if name in fixed:
                lo, hi = fixed[name]
                fixed[name] = (int(lo), int(hi))
        for name in (
            "people",
            "adjectives",
            "topics",
            "statuses",
            "code_verbs",
            "code_nouns",
            "check_names",
            "file_stems",
        ):
            if name in fixed:
                fixed[name] = tuple(str(x) for x in fixed[name])
        if "result_kinds" in fixed:
            fixed["result_kinds"] = tuple((str(k), r) for k, r in fixed["result_kinds"])
        if "condition_modes" in fixed:
            fixed["condition_modes"] = {
                str(k): str(v) for k, v in dict(fixed["condition_modes"]).items()
            }
        return replace(DEFAULT_WORLD, **fixed)

    def summary(self) -> dict[str, Any]:
        """The options as plain data for a run record: every non-callable
        field as it is, and the callable tables (``fault_modes``,
        ``payloads``, ``result_kinds``) as the sorted names they carry."""
        out: dict[str, Any] = {}
        for spec in fields(self):
            value = getattr(self, spec.name)
            if spec.name in ("fault_modes", "payloads"):
                out[spec.name] = sorted(value)
            elif spec.name == "result_kinds":
                out[spec.name] = [kind for kind, _ in value]
            elif isinstance(value, Mapping):
                out[spec.name] = dict(value)
            else:
                out[spec.name] = value
        return out


#: The world every call gets unless told otherwise.
DEFAULT_WORLD = WorldOptions()


# ------------------------------------------------------------- the world


class MockEnvironment:
    """In-memory tool world. User-named refs exist ``exists_share`` of the
    time; issued ids must be created. ``options`` steers every generator."""

    def __init__(
        self,
        tools: list[dict],
        *,
        seed: int = 0,
        faults: dict[str, dict] | None = None,
        world_state: str = "",
        result_shapes: dict[str, dict] | None = None,
        options: WorldOptions | Mapping[str, Any] | None = None,
    ) -> None:
        self.options = WorldOptions.coerce(options)
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

    @property
    def _state(self) -> str:
        return WORLD_STATES.get(self.world_state, "")

    def _fault_for(self, tool: str, arguments: dict) -> dict[str, Any] | None:
        spec = self.faults.get(tool) or self.faults.get("*")
        if not spec:
            return None
        o = self.options
        rate = float(spec.get("rate", o.default_fault_rate))
        if rate <= 0:
            return None
        payload = json.dumps(
            {"seed": self.seed, "tool": tool, "arguments": arguments, "salt": "fault"},
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        if int(digest[:8], 16) / float(0xFFFFFFFF) >= rate:
            return None
        mode = str(spec.get("mode", o.default_fault_mode))
        build = o.fault_modes.get(mode)
        if build is None:
            return None
        return build(self, tool, arguments)

    def _exists(self, value: str) -> bool:
        state = self._state
        if state == "exists":
            return value not in self.deleted
        if state == "missing":
            return False
        if value in self.entities:
            return True
        if value in self.deleted or ISSUED_ID.match(value):
            return False
        digest = hashlib.sha256(f"exists:{self.seed}:{value}".encode()).hexdigest()
        # a seeded draw in tenths, so exists_share=0.7 is exactly 7 in 10
        return int(digest[:8], 16) % 10 < round(float(self.options.exists_share) * 10)

    def _digest(self, tool: str, arguments: dict) -> str:
        payload = json.dumps(
            {"seed": self.seed, "tool": tool, "arguments": arguments}, sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:WORLD_ISSUED_ID_HEX]

    def _payload(self, tool: str, digest: str, arguments: dict | None = None) -> dict[str, Any]:
        """Plausible structured record from the tool name and args. Never a bare hash."""
        o = self.options
        n = int(digest[:8], 16)
        args = arguments or {}
        references = _reference_values(args)
        if references:
            # Reads of a named entity derive from the entity id, not the
            # full call, so every tool describes the same record.
            n = _entity_seed(references[0][1])
        spec = self.specs.get(tool) or {}
        shape = self.result_shapes.get(tool)
        kind = _result_kind(tool, spec, args, o)
        if references:
            stored = self.entities.get(str(references[0][1])) or {}
            prior = stored.get("data")
            if isinstance(prior, dict) and prior:
                return {"data": dict(prior)}
        if isinstance(shape, dict) and shape:
            filled = _fill_template(shape, n, options=o)
            for key, val in args.items():
                if key in filled and not isinstance(filled[key], (dict, list)):
                    filled[key] = val
            if not _shape_mismatches_kind(kind, filled):
                for key, val in list(filled.items()):
                    if isinstance(val, list) and val and isinstance(val[0], dict):
                        template = shape.get(key)
                        item_template = (
                            template[0]
                            if isinstance(template, list)
                            and template
                            and isinstance(template[0], dict)
                            else None
                        )
                        if kind in CODE_KINDS:
                            filled[key] = list(val)
                        else:
                            filled[key] = [
                                _entity_consistent(item, item_template, options=o) for item in val
                            ]
                return {"data": self._finish_record(tool, args, n, kind, filled)}
        invented = _invented_payload(tool, args, n, digest, spec, o)
        if kind == "money":
            return invented
        if isinstance(invented, dict):
            invented = self._finish_record(tool, args, n, kind, invented)
        return {"data": invented}

    def _finish_record(self, tool: str, arguments: dict, n: int, kind: str, data: dict) -> dict:
        """Per-call identity, then remember the record for later reads."""
        if kind not in CODE_KINDS:
            data = _ground_identity(data, arguments, n, tool, self.options)
        if isinstance(data, dict):
            for rec in _record_entries(data):
                ident = rec.get("asin") or rec.get("id")
                if ident not in (None, ""):
                    slot = self.entities.setdefault(str(ident), {})
                    slot["data"] = rec
        return data

    def call(self, tool: str, arguments: dict | None = None) -> dict[str, Any]:
        arguments = arguments or {}
        if tool not in self.schemas:
            return {"status": "error", "reason": "unknown_tool", "tool": tool}
        fault = self._fault_for(tool, arguments)
        if fault is not None:
            return fault
        computed = _evaluate_expression(tool, arguments)
        if computed is not None:
            return computed
        state = self._state
        if state == "missing":
            refs = [v for _, v in _reference_values(arguments)]
            return {"status": "not_found", "missing": refs or ["entity"]}
        if state == "already_done" and not READ_VERBS.match(tool):
            # Completed entities stay readable; only repeat actions are done.
            return {"status": "already_done", "reason": "already_acted_on"}
        missing = _missing_required(self.schemas[tool], arguments)
        if missing:
            return {"status": "rejected", "reason": "missing_required", "fields": missing}
        placeholders = placeholder_arguments(arguments)
        if placeholders:
            return {
                "status": "rejected",
                "reason": "placeholder_argument",
                "fields": placeholders,
                "hint": "ask the person for the actual value",
            }

        o = self.options
        references = _reference_values(arguments)
        digest = self._digest(tool, arguments)
        spec = self.specs.get(tool) or {}
        kind = _result_kind(tool, spec, arguments, o)

        if kind == "file" and (CREATE_VERBS.match(tool) or FILE_MUTATE.search(tool)):
            n = int(digest[:8], 16)
            path = _path_from_args(arguments, n, options=o)
            content = str(arguments.get("content") or arguments.get("diff") or "")
            if path:
                self.entities[path] = {"tool": tool, "arguments": arguments, "version": 1}
            return {
                "status": "ok",
                "path": path,
                "bytes_written": len(content) if content else len(_fake_source(path, n, o)),
            }

        if CREATE_VERBS.match(tool) and kind in {"record", "money"}:
            n = int(digest[:8], 16)
            record = _created_record(tool, arguments, n, digest, o)
            entity_id = str(record.get("number") or record.get("id") or _in_range(n, o.id_range))
            self.entities[entity_id] = {"tool": tool, "arguments": arguments, "version": 1}
            for _, value in references:
                self.entities[value] = {"tool": tool, "arguments": arguments, "version": 1}
            return {"status": "created", **record}

        dangling = [value for _, value in references if not self._exists(value)]
        if READ_VERBS.match(tool):
            if references and dangling:
                return {"status": "not_found", "missing": dangling}
            return {"status": "ok", **self._payload(tool, digest, arguments)}
        if DELETE_VERBS.match(tool):
            if dangling:
                return {"status": "not_found", "missing": dangling}
            for _, value in references:
                self.entities.pop(value, None)
                self.deleted.add(value)
            return {"status": "deleted", "removed": [v for _, v in references]}

        if references and dangling:
            return {"status": "not_found", "missing": dangling}
        for _, value in references:
            entity = (
                self.entities.setdefault(value, {"tool": tool, "version": 0})
                if self._exists(value)
                else None
            )
            if entity is not None:
                entity["version"] = int(entity.get("version", 1)) + 1
        return {"status": "ok", **self._payload(tool, digest, arguments)}

    def executor(self) -> Callable[[str, dict], dict]:
        return self.call


__all__ = [
    "DEFAULT_WORLD",
    "FAULT_MODES",
    "PAYLOAD_BUILDERS",
    "RESULT_KINDS",
    "WORLD_STATES",
    "MockEnvironment",
    "WorldOptions",
    "placeholder_arguments",
]
