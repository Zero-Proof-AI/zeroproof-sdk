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
        if value is None or isinstance(value, bool):
            continue
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


__all__ = [
    "mine_traces", "dimensions_from_traces", "split_pseudo_production",
    "flaw_rows", "leakage_report", "drop_leaky_rows", "simulate_from_traces",
]
