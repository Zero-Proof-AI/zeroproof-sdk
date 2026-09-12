"""The typed row: four objects, one flat wire shape, one version stamp.

Every row the SDK writes or accepts is a projection of four objects:

* ``Task``: the situation. Static, shippable, never contains a rollout.
* ``Rollout``: one episode of one policy on one task. Never contains a
  verdict.
* ``Judgment``: one scorer's verdict on a rollout. Many per rollout.
* ``Marker``: one behavior measurement on a rollout. Many per rollout.

The flat JSONL row that ``simulate()`` streams, ``save()`` writes, and the
platform stores is ``to_row(task, rollout, judgments, markers)``; the
inverse is ``from_row``. The wire shape is described by
``schemas/row-v1.json`` and every row carries ``schema_version``.

Version 0 is every row written before the stamp existed. ``from_row``
recognizes the legacy shapes by key (engine rows by ``scenario_id``,
training exports by ``messages`` without ``steps``, platform trace pulls
by ``tool_trace``, OTel ingest by ``conversation_id``, the published
Hugging Face set by its ``*_json`` string columns) and normalizes them
the way ``load_traces`` does, reusing its alias tables rather than keeping
a second copy. Keys the objects do not model ride through untouched, so
``to_row(*from_row(row))`` never loses a column it did not understand.

Validation is permissive on purpose in this version: a stamped row must
have the required fields with the right types; an unstamped row only has
to be a dict. Stricter checks land after the store moves to the objects.
Until then ``SimulationData.trajectories`` stays the source of truth and
these objects are the view.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

SCHEMA_VERSION = "1"
SCHEMA_KEY = "schema_version"
KNOWN_VERSIONS = frozenset({"0", SCHEMA_VERSION})

#: The sampled diversity axes. Same tuple as ``data._CONVERSATION_FIELDS``;
#: the drift test holds them equal.
AXES = (
    "tier", "ask_family", "intent_known", "tool_known",
    "stance", "tone", "length", "ask", "vagueness", "phrasing",
    "pressure", "user", "texture", "history",
)

#: Row keys that carry a verdict. ``Judgment`` owns them; direct writes
#: outside ``attach`` are frozen at today's count by a test.
VERDICT_KEYS = ("reward", "reason", "grader_reason", "label_source",
                "judge_name", "judge_status", "judge_meta", "failure_class",
                "llm_reward", "llm_reason", "qwen_reward")

#: Rollout-level keys the objects know about and carry by name.
_CARRY_ROLLOUT = ("messages", "opener", "opening", "conversation_id", "ts",
                  "fault_detected", "lineage", "quality", "quality_reason",
                  "quality_scores", "behavior_signature", "steering",
                  "tools", "group_id")

#: Every key ``from_row`` consumes into a typed field. Anything else on the
#: row is unknown to the objects and passes through ``Rollout.extra``.
_CONSUMED = frozenset(
    (SCHEMA_KEY, "prompt", "steps", "final_text", "scenario_id", "spec_id",
     "world_state", "faults", "seed", "rollout_index", "model_version",
     "markers", "tool_trace", "trace")
    + AXES + VERDICT_KEYS + _CARRY_ROLLOUT)

#: The published Hugging Face set flattens list columns to JSON strings.
_JSON_STRING_KEYS = ("steps_json", "messages_json", "metadata_json")

ShapeName = Literal["v1", "engine", "training", "platform_pull", "otel",
                    "hf_flat", "loose"]


# ------------------------------------------------------------------ objects

@dataclass(frozen=True)
class Message:
    role: str
    content: str = ""
    name: str | None = None
    tool_calls: tuple[dict, ...] | None = None


@dataclass(frozen=True)
class Step:
    """One trajectory step: a tool call, an agent turn, or a user turn."""
    tool: str | None = None
    arguments: Any = None
    result: Any = None
    text: str | None = None
    user: str | None = None


@dataclass(frozen=True)
class FaultEvent:
    """What the world did to a tool call: injected fault or observed one."""
    tool: str
    mode: str
    rate: float | None = None
    injected: bool = True


@dataclass(frozen=True)
class World:
    seed: int | None = None
    state: str | None = None
    faults: dict = field(default_factory=dict)
    exemplars: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Privileged:
    """Context the teacher sees and the student never does."""
    principle: str | None = None
    hidden_state: dict = field(default_factory=dict)
    reference: str | None = None


@dataclass(frozen=True)
class Lineage:
    parent_task_id: str | None = None
    transform: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class PolicyRef:
    name: str = ""
    model: str | None = None
    prompt_hash: str | None = None
    version: str | None = None


@dataclass(frozen=True)
class ScorerRef:
    name: str
    kind: Literal["rule", "judge", "reward_model", "human"] = "rule"
    version: str | None = None


@dataclass(frozen=True)
class Task:
    """The situation. Identity-bearing, so nothing computed lives here:
    splits belong to a ``Dataset``, difficulty to a ``Calibration``."""
    task_id: str
    spec_id: str = ""
    prompt: str = ""
    prefix: tuple[Message, ...] = ()
    world: World = field(default_factory=World)
    privileged: Privileged = field(default_factory=Privileged)
    axes: dict[str, Any] = field(default_factory=dict)
    behaviors: tuple[str, ...] = ()
    lineage: Lineage = field(default_factory=Lineage)


@dataclass
class Rollout:
    """One episode. Mutable so grading paths can attach to it in place;
    the no-verdict invariant is enforced at the boundary, not here."""
    rollout_id: str
    task_id: str
    policy: PolicyRef = field(default_factory=PolicyRef)
    index: int = 0
    steps: list[Step] = field(default_factory=list)
    final_text: str = ""
    ledger: list[FaultEvent] = field(default_factory=list)
    usable: bool = True
    unusable_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Judgment:
    rollout_id: str
    scorer: ScorerRef
    reward: float | None
    status: Literal["ok", "missing_reward", "invalid_result", "error",
                    "timeout"] = "ok"
    reason: str = ""
    failure_class: str | None = None
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Marker:
    rollout_id: str
    name: str
    value: float
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Dataset:
    """Split membership is a dataset decision, so one task can be holdout
    in one dataset and training in another."""
    dataset_id: str
    spec_id: str = ""
    splits: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Calibration:
    """Measured difficulty of one task for one student. Optional; only
    ``calibrate`` produces it. ``mean_kl`` needs per-token logprobs, which
    most OpenAI-compatible endpoints do not return on tool-call turns."""
    task_id: str
    student: PolicyRef
    n: int
    pass_rate: float
    mean_kl: float | None = None


# ------------------------------------------------------------------ stamp

def stamp(row: dict) -> dict:
    """Set ``schema_version`` on a row born in canonical shape. In place."""
    if isinstance(row, dict) and SCHEMA_KEY not in row:
        row[SCHEMA_KEY] = SCHEMA_VERSION
    return row


def version_of(row: dict) -> str:
    """``"0"`` for an unstamped row, else the stamp as written."""
    value = row.get(SCHEMA_KEY) if isinstance(row, dict) else None
    return "0" if value is None else str(value)


def detect_shape(row: dict) -> ShapeName:
    """Which shape an unstamped row is in. Stamped rows are ``v1``."""
    if not isinstance(row, dict):
        return "loose"
    if SCHEMA_KEY in row:
        return "v1"
    if any(isinstance(row.get(k), str) for k in _JSON_STRING_KEYS):
        return "hf_flat"
    if (isinstance(row.get("messages"), list) and row.get("messages")
            and not row.get("steps") and "final_text" not in row):
        return "training"
    if row.get("scenario_id") is not None:
        return "engine"
    if isinstance(row.get("tool_trace"), list) and not row.get("steps"):
        return "platform_pull"
    if row.get("conversation_id") is not None:
        return "otel"
    return "loose"


# ------------------------------------------------------------------ validate

def validate(row: Any, kind: Literal["row", "training", "preference"] = "row",
             ) -> list[str]:
    """Problems with one row, empty when it is fine. Never raises.

    Stamped rows must carry the required fields with the right types.
    Unstamped rows are version 0 and only have to be dicts: nothing that
    works today is rejected here.
    """
    if not isinstance(row, dict):
        return ["not_a_dict"]
    version = version_of(row)
    if version not in KNOWN_VERSIONS:
        return [f"unknown_schema_version:{version}"]
    if version == "0":
        return []
    problems: list[str] = []
    if kind == "row":
        if not isinstance(row.get("prompt", ""), str):
            problems.append("prompt_not_str")
        if not isinstance(row.get("steps", []), list):
            problems.append("steps_not_list")
        if not isinstance(row.get("final_text", ""), str):
            problems.append("final_text_not_str")
        reward = row.get("reward")
        if reward is not None and isinstance(reward, bool):
            problems.append("reward_is_bool")
        elif reward is not None and not isinstance(reward, (int, float)):
            problems.append("reward_not_number")
    elif kind == "training":
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            problems.append("messages_missing")
        elif any(not isinstance(m, dict) or "role" not in m for m in messages):
            problems.append("message_without_role")
        if row.get("tools") is not None and not isinstance(row["tools"], list):
            problems.append("tools_not_list")
    elif kind == "preference":
        for side in ("chosen", "rejected"):
            if not isinstance(row.get(side), list) or not row[side]:
                problems.append(f"{side}_missing")
    return problems


def check(rows: Sequence[Any] | Any, kind: Literal["row", "training",
                                                  "preference"] = "row",
          *, where: str = "rows") -> None:
    """Raise ``ValueError`` naming the first bad rows. Accepts one row too."""
    items = rows if isinstance(rows, (list, tuple)) else [rows]
    bad: list[str] = []
    for i, row in enumerate(items):
        problems = validate(row, kind)
        if problems:
            bad.append(f"{i}:{','.join(problems)}")
            if len(bad) >= 5:
                break
    if bad:
        raise ValueError(f"schema_invalid in {where}: {'; '.join(bad)}")


# ------------------------------------------------------------------ coerce

def _short_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


def _number(value: Any) -> float | int | None:
    """A reward-like value as a number, or None. Bools count as 0/1."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ from_row

def _normalize_legacy(row: dict) -> dict:
    """Version-0 row in canonical spelling, via the trace loader's tables."""
    from .ingest.traces import load_traces
    shape = detect_shape(row)
    if shape in ("v1", "engine", "otel"):
        return dict(row)
    if shape == "hf_flat":
        out = dict(row)
        for key in _JSON_STRING_KEYS:
            text = out.pop(key, None)
            if not isinstance(text, str):
                continue
            try:
                value = json.loads(text)
            except ValueError:
                out[key] = text          # unparseable: keep it as it was
                continue
            target = key[:-5]            # steps_json -> steps
            if target == "metadata":
                if isinstance(value, dict):
                    out.setdefault("metadata", value)
            elif isinstance(value, list) and not out.get(target):
                out[target] = value
        return out
    normalized = load_traces([row])
    return normalized[0] if normalized else dict(row)


def _steps(raw: Any) -> list[Step]:
    out: list[Step] = []
    if not isinstance(raw, list):
        return out
    for step in raw:
        if not isinstance(step, dict):
            continue
        out.append(Step(tool=step.get("tool"), arguments=step.get("arguments"),
                        result=step.get("result"), text=step.get("text"),
                        user=step.get("user")))
    return out


def _ledger(faults: Any) -> list[FaultEvent]:
    out: list[FaultEvent] = []
    if not isinstance(faults, dict):
        return out
    for tool, plan in faults.items():
        if isinstance(plan, dict) and plan.get("mode"):
            rate = _number(plan.get("rate"))
            out.append(FaultEvent(tool=str(tool), mode=str(plan["mode"]),
                                  rate=float(rate) if rate is not None else None))
    return out


def _judgments(row: dict, rollout_id: str) -> list[Judgment]:
    out: list[Judgment] = []
    has_primary = ("reward" in row or row.get("judge_status")
                   or row.get("judge_name") or row.get("label_source"))
    if has_primary:
        judge = row.get("judge_name")
        label = row.get("label_source")
        name = judge or label or "unlabeled"
        kind: Any = "judge" if judge else "rule"
        reward = _number(row.get("reward"))
        status = row.get("judge_status") or "ok"
        evidence: dict = {}
        if reward is None and row.get("reward") is not None:
            status = "invalid_result"
            evidence["raw_reward"] = row.get("reward")
        if row.get("judge_meta"):
            evidence["judge_meta"] = row["judge_meta"]
        if judge and label:
            evidence["label_source"] = label
        if row.get("grader_reason") and row.get("grader_reason") != row.get("reason"):
            evidence["grader_reason"] = row["grader_reason"]
        if "reward" in row:
            evidence["reward_present"] = True
        out.append(Judgment(
            rollout_id=rollout_id, scorer=ScorerRef(name=str(name), kind=kind),
            reward=reward, status=status,
            reason=str(row.get("reason") or ""),
            failure_class=row.get("failure_class"), evidence=evidence))
    if "llm_reward" in row:
        out.append(Judgment(rollout_id=rollout_id,
                            scorer=ScorerRef(name="llm", kind="judge"),
                            reward=_number(row.get("llm_reward")),
                            reason=str(row.get("llm_reason") or "")))
    if row.get("qwen_reward") is not None:
        out.append(Judgment(rollout_id=rollout_id,
                            scorer=ScorerRef(name="qwen", kind="judge"),
                            reward=_number(row.get("qwen_reward"))))
    return out


def from_row(row: dict) -> tuple[Task, Rollout, list[Judgment], list[Marker]]:
    """Split one flat row into its four objects. Any version, any shape."""
    if not isinstance(row, dict):
        raise TypeError("from_row expects a dict")
    raw = _normalize_legacy(row)
    prompt = str(raw.get("prompt") or "")
    task_id = str(raw.get("scenario_id") or "") or "task_" + _short_hash(prompt)
    axes = {k: raw[k] for k in AXES if k in raw and raw[k] is not None}
    faults = raw.get("faults") if isinstance(raw.get("faults"), dict) else {}
    task = Task(
        task_id=task_id,
        spec_id=str(raw.get("spec_id") or ""),
        prompt=prompt,
        world=World(seed=raw.get("seed"), state=raw.get("world_state"),
                    faults=dict(faults)),
        axes=axes,
    )
    index = _int(raw.get("rollout_index"))
    model = raw.get("model_version")
    rollout_id = str(raw.get("rollout_id") or "") or _short_hash(
        task_id, index, model or "")
    extra = {k: raw[k] for k in _CARRY_ROLLOUT if k in raw and raw[k] is not None}
    if raw.get("rollout_index") is not None:
        extra["rollout_index_present"] = True
    passthrough = {k: v for k, v in raw.items() if k not in _CONSUMED}
    if passthrough:
        extra["passthrough"] = passthrough
    rollout = Rollout(
        rollout_id=rollout_id, task_id=task_id,
        policy=PolicyRef(name=str(model or ""), model=model),
        index=index, steps=_steps(raw.get("steps")),
        final_text=str(raw.get("final_text") or ""),
        ledger=_ledger(faults), extra=extra,
    )
    markers: list[Marker] = []
    for k, v in (raw.get("markers") or {}).items() if isinstance(
            raw.get("markers"), dict) else ():
        value = _number(v)
        if value is not None:
            markers.append(Marker(rollout_id=rollout_id, name=str(k),
                                  value=float(value)))
    return task, rollout, _judgments(raw, rollout_id), markers


# ------------------------------------------------------------------ to_row

def _step_dict(step: Step) -> dict:
    if step.user is not None:
        return {"user": step.user}
    out: dict[str, Any] = {}
    if step.tool is not None:
        out["tool"] = step.tool
        out["arguments"] = step.arguments
        out["result"] = step.result
    if step.text is not None:
        out["text"] = step.text
    return out


def to_row(task: Task, rollout: Rollout,
           judgments: Sequence[Judgment] = (),
           markers: Sequence[Marker] = ()) -> dict:
    """The flat v1 wire row. Inverse of ``from_row`` on engine rows; on
    other shapes it is the canonical row ``load_traces`` would produce,
    with the source row's unknown keys carried along."""
    from .data import _clean_faults, conversation
    row: dict[str, Any] = {
        "prompt": task.prompt,
        "steps": [_step_dict(s) for s in rollout.steps],
        "final_text": rollout.final_text,
        "scenario_id": task.task_id,
    }
    row["messages"] = rollout.extra.get("messages") or conversation(row)
    for key in ("opener", "opening"):
        if rollout.extra.get(key):
            row[key] = rollout.extra[key]
    for key in AXES:
        if key in task.axes:
            row[key] = task.axes[key]
    if task.world.state and task.world.state not in {"unspecified", "unknown"}:
        row["world_state"] = task.world.state
    faults = _clean_faults(task.world.faults)
    if faults:
        row["faults"] = faults
    if rollout.extra.get("fault_detected"):
        row["fault_detected"] = True
    if rollout.index or rollout.extra.get("rollout_index_present"):
        row["rollout_index"] = rollout.index
    if rollout.policy.model is not None:
        row["model_version"] = rollout.policy.model
    primary = next((j for j in judgments
                    if j.scorer.name not in {"llm", "qwen"}), None)
    if primary is not None:
        if primary.reward is not None or primary.evidence.get("reward_present"):
            row["reward"] = primary.reward
        if primary.reason:
            row["reason"] = primary.reason
        if primary.scorer.name != "unlabeled":
            if primary.scorer.kind == "judge":
                row["judge_name"] = primary.scorer.name
                if primary.evidence.get("label_source"):
                    row["label_source"] = primary.evidence["label_source"]
            else:
                row["label_source"] = primary.scorer.name
        if primary.status != "ok":
            row["judge_status"] = primary.status
        if primary.evidence.get("judge_meta"):
            row["judge_meta"] = primary.evidence["judge_meta"]
        if primary.failure_class:
            row["failure_class"] = primary.failure_class
    for j in judgments:
        if j.scorer.name == "llm":
            row["llm_reward"] = j.reward
            row["llm_reason"] = j.reason
        elif j.scorer.name == "qwen" and j.reward is not None:
            row["qwen_reward"] = j.reward
    if markers:
        row["markers"] = {m.name: m.value for m in markers}
    for key in ("conversation_id", "ts", "lineage", "quality",
                "quality_reason", "quality_scores", "tools", "group_id"):
        if rollout.extra.get(key) is not None:
            row[key] = rollout.extra[key]
    if task.spec_id:
        row["spec_id"] = task.spec_id
    for key, value in (rollout.extra.get("passthrough") or {}).items():
        row.setdefault(key, value)
    return stamp(row)


def as_dict(obj: Any) -> dict:
    """Plain dict of any schema object, for JSON."""
    return asdict(obj)


def load_json_schema() -> dict:
    """The packaged ``schemas/row-v1.json``."""
    from importlib import resources
    text = (resources.files(__package__) / "schemas" / "row-v1.json").read_text()
    return json.loads(text)


__all__ = [
    "SCHEMA_VERSION", "SCHEMA_KEY", "KNOWN_VERSIONS", "AXES", "VERDICT_KEYS",
    "Message", "Step", "FaultEvent", "World", "Privileged", "Lineage",
    "PolicyRef", "ScorerRef", "Task", "Rollout", "Judgment", "Marker",
    "Dataset", "Calibration",
    "stamp", "version_of", "detect_shape", "validate", "check",
    "from_row", "to_row", "as_dict", "load_json_schema",
]
