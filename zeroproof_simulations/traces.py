"""Production traces to a focused coverage grid. Simulation, version two.

Version one is cold start: ``simulate`` invents diverse situations from the
agent spec alone. Version two starts from traces of the deployed agent (real
production rows, or a slice of simulations set aside as pseudo production):
mine what actually happened, then aim the covering grid at the observed
tools, faults, and worlds instead of the whole space.

Leakage rule: source traces never enter the generated dataset. They shape
the grid and nothing else. ``leakage_report`` / ``drop_leaky_rows`` verify
no generated prompt is a near copy of a source trace, so a trace held out
for evaluation stays out of training.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from .embeddings import resolve_embedder
from .grading import NO_FAULT, behavior_signature, trace_fault
from .scenarios import build_dimensions

# Observed fault chip -> the grid axis and value that reproduces it.
_FAULT_TO_AXIS = {
    "timeout": ("tool_condition", "timeout"),
    "malformed": ("tool_condition", "malformed_result"),
    "stale": ("tool_condition", "stale_result"),
    "deny": ("tool_condition", "permission_denied"),
    "not_found": ("world_state", "entity missing"),
    "already_done": ("world_state", "entity already acted on"),
}


def _binary_reward(row: dict) -> int | None:
    for key in ("reward", "qwen_reward"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            # External rows label with True/False; a skipped False would
            # hide that trace's flaw signal from mining.
            value = int(value)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == 0.0:
            return 0
        if number == 1.0:
            return 1
    return None


def _row_tools(row: dict) -> list[str]:
    names: list[str] = []
    for step in row.get("steps") or []:
        if isinstance(step, dict) and step.get("tool"):
            names.append(str(step["tool"]))
    return names


def mine_traces(rows: Sequence[dict]) -> dict[str, Any]:
    """What the deployed agent actually did, counted for grid focusing.

    ``flaw_rows`` is any row with an observed fault or a 0 label. Those are
    the behaviors worth simulating more of.
    """
    tools: dict[str, dict[str, int]] = {}
    faults: dict[str, int] = {}
    worlds: dict[str, int] = {}
    behaviors: set[str] = set()
    flaw_rows: list[int] = []
    asks: list[str] = []
    seen_asks: set[str] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        fault = trace_fault(row)
        reward = _binary_reward(row)
        behaviors.add(behavior_signature(row))
        if fault != NO_FAULT:
            faults[fault] = faults.get(fault, 0) + 1
        world = str(row.get("world_state") or "").strip()
        if world and world not in {"unspecified", "unknown"}:
            worlds[world] = worlds.get(world, 0) + 1
        flawed = fault != NO_FAULT or reward == 0
        if flawed:
            flaw_rows.append(i)
        for name in _row_tools(row):
            slot = tools.setdefault(name, {"n": 0, "fault_n": 0})
            slot["n"] += 1
            if flawed:
                slot["fault_n"] += 1
        prompt = str(row.get("prompt") or "").strip()
        if prompt and prompt not in seen_asks:
            seen_asks.add(prompt)
            asks.append(prompt)
    return {
        "n": len(rows),
        "tools": tools,
        "faults": faults,
        "world_states": worlds,
        "unique_behaviors": len(behaviors),
        "flaw_rows": flaw_rows,
        "asks": asks,
    }


# Trace-grounded result exemplars. Agent specs declare tool inputs but
# almost never result shapes, so invented results drift from the real
# product. Traces carry the real payloads; a few per tool become shape
# templates. Three shows the shape family; ~500 serialized chars keeps a
# template affordable in a prompt.
_EXEMPLARS_PER_TOOL = 3
_EXEMPLAR_MAX_CHARS = 500


def _exemplar_value(result: Any) -> Any:
    """Structured view of a step result; JSON-in-a-string is parsed."""
    if isinstance(result, str):
        try:
            return json.loads(result)
        except ValueError:
            return result
    return result


def _exemplar_shape_key(value: Any) -> str:
    """Coarse shape identity: keys for records, item shape for lists,
    length band for text. Two results with the same key teach nothing new."""
    if isinstance(value, dict):
        return "dict:" + ",".join(sorted(str(k) for k in value)[:10])
    if isinstance(value, list):
        return "list:" + (_exemplar_shape_key(value[0]) if value else "empty")
    if isinstance(value, str):
        return f"str:{min(len(value) // 100, 4)}"
    return type(value).__name__


def _trim_exemplar(value: Any) -> Any:
    """Shrink a payload toward the serialized cap without breaking JSON."""
    if isinstance(value, str):
        return value if len(value) <= 160 else value[:157] + "..."
    if isinstance(value, list):
        return [_trim_exemplar(v) for v in value[:2]]
    if isinstance(value, dict):
        return {str(k): _trim_exemplar(v)
                for k, v in list(value.items())[:12]}
    return value


def mine_result_exemplars(rows: Sequence[dict], *,
                          per_tool: int = _EXEMPLARS_PER_TOOL
                          ) -> dict[str, list]:
    """Up to ``per_tool`` real result payloads per tool, shape-diverse.

    Sibling of ``mine_traces``: same rows in, but this collects what the
    tools RETURNED, for grounding invented results. Faulted and empty
    results are skipped (they show the fault axis, not the success
    shape), a result whose shape is already kept is skipped, and each
    exemplar is trimmed to serialize within ~500 chars.
    """
    out: dict[str, list] = {}
    seen: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for step in row.get("steps") or []:
            if not isinstance(step, dict) or not step.get("tool"):
                continue
            step = _normalize_step(step)
            if "result" not in step:
                continue
            name = str(step["tool"])
            value = _exemplar_value(step["result"])
            if not value:
                continue
            if isinstance(value, dict) and not (set(value) - {"status", "ok"}):
                # A bare status carries no shape worth copying.
                continue
            if trace_fault({"steps": [{"tool": name,
                                       "result": value}]}) != NO_FAULT:
                continue
            kept = out.setdefault(name, [])
            if len(kept) >= per_tool:
                continue
            key = _exemplar_shape_key(value)
            if key in seen.setdefault(name, set()):
                continue
            trimmed = _trim_exemplar(value)
            try:
                if len(json.dumps(trimmed,
                                  default=str)) > _EXEMPLAR_MAX_CHARS:
                    continue
            except (TypeError, ValueError):
                continue
            seen[name].add(key)
            kept.append(trimmed)
    return {name: kept for name, kept in out.items() if kept}


def exemplar_result_shapes(exemplars: dict[str, list]) -> dict[str, dict]:
    """Exemplars in ``write_result_shapes`` form: one template per tool.

    The sandbox fills one dict template per call (``MockEnvironment``),
    so the first record-shaped exemplar becomes that template; a bare
    list of records is wrapped the way ``_parse_result_shapes`` wraps
    one. Non-record exemplars stay report-only.
    """
    out: dict[str, dict] = {}
    for name, values in (exemplars or {}).items():
        for value in values:
            if isinstance(value, dict) and value:
                out[name] = dict(value)
                break
            if isinstance(value, list) and value and isinstance(value[0], dict):
                out[name] = {"results": [dict(value[0])]}
                break
    return out


def dimensions_from_traces(rows: Sequence[dict], tools: list[dict],
                           policy: str = "", *,
                           broaden: bool = True) -> dict[str, list[str]]:
    """Coverage axes aimed at behaviors seen in ``rows``.

    Starts from ``build_dimensions`` for this agent so every value is one
    the writer and sandbox understand. The tool axis puts observed failing
    tools first; ``broaden=False`` drops tools the traces never touched
    (keeping the base specials such as ``unrelated``), so a run spends its
    budget near the flaws instead of boiling the ocean. Fault and world
    axes always keep their clean value: contrast needs passing rows too.
    """
    base = build_dimensions(tools, policy)
    mined = mine_traces(rows)
    observed = mined["tools"]

    def _tool_rank(name: str) -> tuple:
        slot = observed.get(name) or {}
        return (-int(slot.get("fault_n", 0)), -int(slot.get("n", 0)), name)

    base_tools = list(base.get("tool") or [])
    specials = [t for t in base_tools if t in {"unrelated", "multi_tool"}]
    real = [t for t in base_tools if t not in specials]
    seen = [t for t in real if t in observed]
    unseen = [t for t in real if t not in observed]
    seen.sort(key=_tool_rank)
    if seen:
        tool_axis = seen + (unseen if broaden else []) + specials
    else:
        tool_axis = base_tools

    conditions = list(base.get("tool_condition") or [])
    worlds = list(base.get("world_state") or [])
    focus_conditions: list[str] = []
    focus_worlds: list[str] = []
    for name in sorted(mined["faults"], key=mined["faults"].get, reverse=True):
        axis_value = _FAULT_TO_AXIS.get(name)
        if not axis_value:
            if name in conditions:
                focus_conditions.append(name)
            continue
        axis, value = axis_value
        if axis == "tool_condition" and value in conditions:
            focus_conditions.append(value)
        elif axis == "world_state" and value in worlds:
            focus_worlds.append(value)
    for world in sorted(mined["world_states"], key=mined["world_states"].get,
                        reverse=True):
        if world in worlds and world not in focus_worlds:
            focus_worlds.append(world)
    if focus_conditions:
        conditions = ["success"] + [c for c in focus_conditions if c != "success"]
    if focus_worlds:
        clean = [w for w in ("entity exists",) if w in worlds]
        worlds = clean + [w for w in focus_worlds if w not in clean]

    out = dict(base)
    out["tool"] = tool_axis
    out["tool_condition"] = conditions
    out["world_state"] = worlds
    return out


def split_pseudo_production(rows: Sequence[dict], *, fraction: float = 0.2,
                            seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Set aside a pseudo-production slice; the rest stays for training.

    Every unique flaw signature (fault name plus behavior shape) sends one
    row to the production side first, so the held-out slice contains each
    distinct failure at least once. Deterministic in ``seed``. The two
    sides never share a row.
    """
    items = [row for row in rows if isinstance(row, dict)]
    n = len(items)
    if n == 0:
        return [], []
    target = max(1, min(n, round(max(0.0, float(fraction)) * n))) \
        if fraction > 0 else 0
    seen_flaws: set[tuple[str, str]] = set()
    production_idx: list[int] = []
    for i, row in enumerate(items):
        fault = trace_fault(row)
        if fault == NO_FAULT and _binary_reward(row) != 0:
            continue
        key = (fault, behavior_signature(row))
        if key in seen_flaws:
            continue
        seen_flaws.add(key)
        production_idx.append(i)
    chosen = set(production_idx)
    if len(chosen) < target:
        rest = [i for i in range(n) if i not in chosen]
        rest.sort(key=lambda i: hashlib.sha256(
            f"{seed}:{i}:{str(items[i].get('prompt') or '')[:200]}".encode()
        ).hexdigest())
        for i in rest[:target - len(chosen)]:
            chosen.add(i)
    production = [items[i] for i in range(n) if i in chosen]
    remainder = [items[i] for i in range(n) if i not in chosen]
    return production, remainder


def flaw_rows(rows: Sequence[dict]) -> list[dict]:
    """Rows with an observed fault or a 0 label: the next round's traces.

    The hill-climb loop feeds a round's failures back into
    ``simulate(traces=flaw_rows(evaluated))`` so the next batch aims at
    what the agent still gets wrong.
    """
    mined = mine_traces(rows)
    keep = set(mined["flaw_rows"])
    items = [row for row in rows if isinstance(row, dict)]
    return [row for i, row in enumerate(items) if i in keep]


def _prompt_of(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("prompt") or "").strip()
    return str(item or "").strip()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _leak_flags(generated: Sequence[Any], sources: Sequence[Any], *,
                threshold: float,
                embedder: Any) -> tuple[list[bool], dict[str, Any]]:
    gen_texts = [_prompt_of(item) for item in generated]
    src_texts = [t for t in (_prompt_of(item) for item in sources) if t]
    flags = [False] * len(gen_texts)
    report: dict[str, Any] = {
        "n": len(gen_texts), "n_sources": len(src_texts),
        "threshold": float(threshold), "n_leaky": 0,
        "max_similarity": 0.0, "leaky": [],
    }
    if not gen_texts or not src_texts:
        return flags, report
    resolved = resolve_embedder(embedder)
    src_norm = {" ".join(t.lower().split()) for t in src_texts}
    src_vecs = resolved.embed(src_texts)
    gen_vecs = resolved.embed([t or " " for t in gen_texts])
    for i, vec in enumerate(gen_vecs):
        best, best_j = 0.0, -1
        for j, src in enumerate(src_vecs):
            sim = _cosine(vec, src)
            if sim > best:
                best, best_j = sim, j
        if " ".join(gen_texts[i].lower().split()) in src_norm:
            best = 1.0
        report["max_similarity"] = max(report["max_similarity"], best)
        if best >= min(float(threshold), 1.0):
            flags[i] = True
            report["n_leaky"] += 1
            if len(report["leaky"]) < 20:
                report["leaky"].append(
                    {"row": i, "source": best_j, "similarity": round(best, 4)})
    return flags, report


def leakage_report(generated: Sequence[Any], sources: Sequence[Any], *,
                   threshold: float = 0.9,
                   embedder: Any = "hash") -> dict[str, Any]:
    """Near-copy check of generated prompts against source traces.

    A generated row whose prompt sits at or above ``threshold`` cosine
    similarity to any source prompt is flagged. Exact matches always flag,
    whatever the embedder thinks. ``leaky`` lists the first 20 offenders;
    ``n_leaky`` is the full count.
    """
    return _leak_flags(generated, sources, threshold=threshold,
                       embedder=embedder)[1]


def drop_leaky_rows(rows: Sequence[dict], sources: Sequence[Any], *,
                    threshold: float = 0.9,
                    embedder: Any = "hash") -> tuple[list[dict], dict[str, Any]]:
    """Kept rows plus the report. Flagged rows are removed, not rewritten."""
    flags, report = _leak_flags(rows, sources, threshold=threshold,
                                embedder=embedder)
    kept = [row for row, bad in zip(rows, flags) if not bad]
    report["n_dropped"] = len(rows) - len(kept)
    return kept, report


def simulate_from_traces(traces: Sequence[dict], agent: Any = None, *,
                         tools: list[dict] | None = None,
                         policy: str = "",
                         mode: str = "rl",
                         **kwargs: Any):
    """Alias for ``simulate(agent, traces=...)``: same grid focus and
    leakage gate, for callers who start from the traces."""
    from zeroproof_simulations import simulate as _simulate
    return _simulate(agent, tools=tools, system_prompt=policy or None,
                     mode=mode, traces=traces, **kwargs)


# --- canonical trace input --------------------------------------------------
#
# The one public trajectory schema everything trace-shaped normalizes to:
#
#     prompt:     str            first user ask
#     steps:      list of {"user": str}
#                       | {"tool": str, "arguments": dict, "result": Any}
#                       | {"text": str}                  agent turns
#     final_text: str            the agent's last message
#     reward:     0 | 1          OPTIONAL; ungraded traces are first-class
#
# Every other key carries through untouched. ``simulate(traces=...)``
# accepts anything ``load_traces`` accepts: ZeroProof simulation rows, eval
# rollouts, raw JSONL exports, ``rows_from_otel`` output, graded or not.

_PROMPT_KEYS = ("prompt", "question", "input", "task", "ask")
_STEP_KEYS = ("steps", "tool_trace", "trace")
_FINAL_KEYS = ("final_text", "final", "output", "response", "answer")


#: Names a tool step's argument and result fields arrive under. The platform's
#: trace ingest writes `input`/`output`; OpenAI-style exports write
#: `arguments`/`result`. Renaming them here rather than teaching every consumer
#: both spellings, because the consumers that only knew one did not fail
#: loudly: `tool_call_roundtrip` reported "checked: 0" on a whole dataset of
#: ingested traces and every row passed the export gate without being looked at.
_ARG_KEYS = ("arguments", "input", "args", "parameters")
_RESULT_KEYS = ("result", "output", "response")


def _normalize_step(step: dict) -> dict:
    """One trajectory step in the canonical spelling.

    Only tool steps are touched. A `{"user": ...}` or `{"text": ...}` step has
    no arguments or result, and `input` on a non-tool step is somebody else's
    field.
    """
    if "tool" not in step:
        return step
    out = dict(step)
    for canonical, aliases in (("arguments", _ARG_KEYS), ("result", _RESULT_KEYS)):
        if canonical in out:
            continue
        source = next((k for k in aliases if k in out), None)
        if source is not None:
            out[canonical] = out.pop(source)
    return out


def _steps_from_messages(messages: Sequence[dict]) -> list[dict]:
    steps: list[dict] = []

    def _attach(result: Any, name: str) -> None:
        # Match by tool name first, then first-unfilled (FIFO). Never
        # last-unfilled: parallel calls answered in order would swap
        # payloads and mining would blame the wrong tool. An orphan result
        # becomes its own step so its fault still reaches the miner.
        unfilled = [s for s in steps if "tool" in s and "result" not in s]
        target = None
        if name:
            target = next((s for s in unfilled if s.get("tool") == name),
                          None)
        if target is None and unfilled:
            target = unfilled[0]
        if target is None:
            steps.append({"tool": name, "arguments": {}, "result": result})
        else:
            target["result"] = result

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "user":
            steps.append({"user": content})
        elif role == "assistant":
            for call in message.get("tool_calls") or []:
                fn = call.get("function") if isinstance(
                    call.get("function"), dict) else call
                raw = (fn or {}).get("arguments")
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except ValueError:
                        pass
                steps.append({"tool": str((fn or {}).get("name") or ""),
                              "arguments": raw if isinstance(raw, dict)
                              else {}})
            if content:
                steps.append({"text": content})
        elif role == "tool":
            result: Any = content
            try:
                result = json.loads(content)
            except ValueError:
                pass
            _attach(result, str(message.get("name") or ""))
    return steps


def _coerce_reward(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number in (0.0, 1.0) else None


def load_traces(source) -> list[dict]:
    """Normalize any supported trace source to the canonical schema above.

    ``source`` is a JSONL path or an iterable of dicts. Rows carrying
    ``tool_trace``/``trace`` instead of ``steps``, ``final``/``output``/
    ``response`` instead of ``final_text``, or only OpenAI-style
    ``messages`` are converted; ``reward`` is kept only when it coerces
    cleanly to 0 or 1, and its absence is fine. Rows that are not dicts or
    carry neither an ask nor any steps are dropped.
    """
    from pathlib import Path as _Path
    if isinstance(source, (str, _Path)):
        from .quality import _load_jsonl
        rows = _load_jsonl(source)
    else:
        rows = list(source)
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row = dict(row)
        # An empty steps list is absence, not content: real exports emit
        # steps: [] next to a populated messages/tool_trace field.
        steps = next((row[k] for k in _STEP_KEYS
                      if isinstance(row.get(k), list) and row[k]), None)
        if steps is None and isinstance(row.get("messages"), list):
            steps = _steps_from_messages(row["messages"])
        row["steps"] = [_normalize_step(s) for s in (steps or [])
                        if isinstance(s, dict)]
        prompt = next((str(row[k]) for k in _PROMPT_KEYS
                       if row.get(k)), "")
        if not prompt:
            prompt = next((str(s["user"]) for s in row["steps"]
                           if "user" in s), "")
        row["prompt"] = prompt
        final = next((str(row[k]) for k in _FINAL_KEYS if row.get(k)), "")
        if not final:
            final = next((str(s["text"]) for s in reversed(row["steps"])
                          if "text" in s), "")
        row["final_text"] = final
        if "reward" in row:
            reward = _coerce_reward(row["reward"])
            if reward is None:
                row.pop("reward")
            else:
                row["reward"] = reward
        if row["prompt"] or row["steps"]:
            out.append(row)
    return out


def trace_report(traces, tools: list[dict] | None = None,
                 policy: str = "") -> dict[str, Any]:
    """What these traces contain and what they will aim generation at.

    Run before ``simulate(traces=...)``. With ``tools`` (and optionally
    ``policy``) the report also computes the actual grid emphasis: which
    axis values move forward in the coverage grid because of these traces.
    Reward stays optional; ungraded counts are reported, never required.
    ``advisory_labels`` counts rows carrying only a ``qwen_reward``: those
    labels do steer trace mining, so they are disclosed, not hidden under
    "ungraded". ``dropped`` counts input rows that carried no usable
    signal and were discarded by normalization.
    """
    from pathlib import Path as _Path
    if isinstance(traces, (str, _Path)):
        from .quality import _load_jsonl
        raw = _load_jsonl(traces)
    else:
        raw = list(traces)
    rows = load_traces(raw)
    mined = mine_traces(rows)
    graded = [r for r in rows if r.get("reward") in (0, 1)]
    advisory = [r for r in rows if r.get("reward") not in (0, 1)
                and _coerce_reward(r.get("qwen_reward")) is not None]
    passes = sum(r["reward"] for r in graded)
    report: dict[str, Any] = {
        "traces": len(rows),
        "dropped": len(raw) - len(rows),
        "unique_prompts": len({" ".join(str(r.get("prompt") or "")
                                        .lower().split()) for r in rows}),
        "tools_observed": mined["tools"],
        "faults_observed": dict(mined["faults"]),
        "world_states_observed": dict(mined.get("world_states") or {}),
        "distinct_behaviors": mined["unique_behaviors"],
        "graded": len(graded),
        "passes": passes,
        "fails": len(graded) - passes,
        # A row is ungraded only when NO label is in play: advisory
        # (qwen_reward-only) rows steer trace mining, so they are counted
        # and disclosed separately, never folded into "ungraded".
        "ungraded": len(rows) - len(graded) - len(advisory),
        "advisory_labels": len(advisory),
    }
    if tools:
        aimed = dimensions_from_traces(rows, tools, policy)
        base = build_dimensions(tools, policy)
        # Aiming works two ways: an axis NARROWS (unobserved values are
        # dropped, so every retained value's budget share rises) or values
        # REORDER toward the front. The clean contrast values stay in every
        # aimed axis by design and receive no extra weight, so they are
        # excluded from the claim.
        clean = {"success", "entity exists", NO_FAULT}
        emphasis: dict[str, list[str]] = {}
        for axis in ("tool", "tool_condition", "world_state"):
            base_axis = list(base.get(axis) or [])
            aimed_axis = list(aimed.get(axis) or [])
            base_pos = {v: i for i, v in enumerate(base_axis)}
            narrowed = len(aimed_axis) < len(base_axis)
            gained = [v for i, v in enumerate(aimed_axis)
                      if v in base_pos and v not in clean
                      and (narrowed or i < base_pos[v])]
            if gained:
                emphasis[axis] = gained
        report["emphasis"] = emphasis
        foreign = [name for name in mined["tools"]
                   if name not in {str((t.get("function") or t).get("name")
                                       or "") for t in tools}]
        if foreign:
            report["foreign_tools"] = foreign
    return report


def format_trace_report(report: dict[str, Any]) -> str:
    """The trace report as a text block, aiming stated in plain words."""
    lines = [
        f"traces:             {report['traces']}",
        f"unique prompts:     {report['unique_prompts']}",
        f"distinct behaviors: {report['distinct_behaviors']}",
        f"graded:             {report['graded']} "
        f"(pass {report['passes']} / fail {report['fails']}), "
        f"ungraded {report['ungraded']}"
        + (f", advisory judge labels {report['advisory_labels']} "
           "(these steer aiming)" if report.get("advisory_labels") else ""),
        "tools observed:     " + (", ".join(
            f"{name} x{slot['n']}"
            + (f" ({slot['fault_n']} faulted)" if slot.get("fault_n") else "")
            for name, slot in sorted(report["tools_observed"].items()))
            or "none"),
        "faults observed:    " + (", ".join(
            f"{name} x{n}" for name, n in
            sorted(report["faults_observed"].items())) or "none"),
    ]
    if report.get("dropped"):
        lines.append(f"dropped rows:       {report['dropped']} "
                     "(no usable signal after normalization)")
    if report.get("foreign_tools"):
        lines.append("warning: observed tools not in this agent's toolset: "
                     + ", ".join(report["foreign_tools"][:5]))
    emphasis = report.get("emphasis")
    if emphasis:
        parts = []
        names = {"tool": "tools", "tool_condition": "faults",
                 "world_state": "world states"}
        for axis, values in emphasis.items():
            shown = ", ".join(values[:5])
            if len(values) > 5:
                shown += f" (+{len(values) - 5} more)"
            parts.append(f"{names.get(axis, axis)} {shown}")
        lines.append("extra generation weight goes to: " + "; ".join(parts))
    else:
        lines.append("grid emphasis: computed against the agent's tools at "
                     "simulate time (pass tools= to preview it here)")
    return "\n".join(lines)


__all__ = [
    "mine_traces", "mine_result_exemplars", "exemplar_result_shapes",
    "dimensions_from_traces", "split_pseudo_production",
    "flaw_rows", "leakage_report", "drop_leaky_rows", "simulate_from_traces",
    "trace_story",
    "load_traces", "trace_report", "format_trace_report",
]
