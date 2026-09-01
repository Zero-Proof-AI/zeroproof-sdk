"""Generate training conversations from an intent or an agent.

Input is a system prompt, tools, or both. The SDK sets a world from that
spec, writes human requests across a grid (simple, complex, vague, ordinary,
malicious), and rolls the agent. Grade 0/1 later. Optimize for post-training.

    import zeroproof_simulations as zps
    data = zps.simulate(system_prompt=policy)          # prompt-only agent
    data = zps.simulate(tools=tools, system_prompt=policy)
    data.save("rollout.jsonl")
    zps.grade("rollout.jsonl")                         # hosted Qwen 0/1
    zps.optimize("rollout.jsonl", output="train.jsonl")
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .adapters import (claude_code, connect, inspect, resolve, ConnectedAgent,
                       AgentProfile, resolve_system_prompt)
from .agents import (hosted_model, local_model, missing_hosted_key,
                       parse_backend_spec, public_llm_error, touch_hosted,
                       default_max_turns)
from .coverage import (NEW_SIGNATURE_FLOOR, SATURATION_COPIES,
                       build_coverage_summary, cell_key as _cell_key_from_row,
                       copies_remaining, coverage_point, space_saturated)
from .diversity import (MAX_NOVELTY_RESTARTS, NOVELTY_RESTART_FLOOR,
                        adaptive_allocator, allocator_slot_counts,
                        behavior_tier, cap_scenario_families,
                        conversation_features, mix_items_by_tier,
                        new_turn_stats, record_turns, sample_cell_tags,
                        sampling_plan, scenario_family)
from .embeddings import (EmbeddingArchive, is_semantic, resolve_embedder,
                         select_execution_batch)
from .actionspace import (action_space_targets, induced_keys_from_trajectory,
                          render_target_situation, shape_as_tags,
                          shape_from_trajectory, uncovered_action_shapes)
from .explore import mutate_pool
from .generator import (ModelSimulator, amplify_seeds, assistant_kind,
                       make_default_generator,
                       write_result_shapes, write_scene_brief)
from .grading import _as_dict, behavior_signature, conduct_grade
from .platform import (PlatformError, datasets,
                       delete as delete_dataset,
                       issue_delegated_credential, pull, push_file,
                       push_rows, push_to_studio,
                       refresh_delegated_credential,
                       revoke_delegated_credential)
from .llm_judge import (MISSING_JUDGE_KEY, apply_llm_grade,
                       resolve_judge_key)
from .grade_llm import apply_grade_llm, require_judge_key
from .quality import (rank as rank_source, rank_rows, score_row,
                      summarize as summarize_quality)
from .sandbox import MockEnvironment
from .scenarios import (DEFAULT_FAULT_RATE, SEARCH_ARMS, build_dimensions,
                        keep_fault_plan,
                        make_candidate_generator, novelty,
                        open_ended_probes, policy_sections,
                        reallocate_search_arms, retarget_regions,
                        scenario_regions, _intent_for_tool)
from .export import (export_dataset, export_preference, export_training,
                     training_rows)
from .optimize import (filter_rl_rows, group_signal, optimize,
                       optimize_for_rl, recommend, select_for_rl,
                       select_for_sft, trim_unanimous_groups)
from .otel import rows_from_otel
from .traces import (behavior_state, region_progress,
                     dimensions_from_traces, drop_leaky_rows,
                     exemplar_result_shapes, flaw_rows,
                     format_trace_report, leakage_report,
                     load_traces, mine_result_exemplars, mine_traces,
                     simulate_from_traces,
                     split_pseudo_production, trace_report)
from .preflight import (FAILURE_CLASSES, classify_failure, dataset_report,
                        format_dataset_report, preflight)
from .judging import (ScoredData, build_preference_pairs, evaluate,
                      normalize_judge_result, run_judge)

__all__ = ["simulate", "SimulationData", "conversation", "local_model",
           "datasets", "pull", "push_file", "push_rows",
           "delete_dataset", "issue_delegated_credential",
           "refresh_delegated_credential", "revoke_delegated_credential",
           "push_to_studio",
           "PlatformError",
           "hosted_model",
           "MockEnvironment", "llm_grade", "grade", "grade_llm",
           "rank", "rank_rows", "score_row",
           "conduct_grade", "behavior_signature", "build_dimensions",
           "policy_sections", "scenario_regions", "open_ended_probes",
           "make_candidate_generator",
           "novelty", "connect", "inspect", "claude_code", "ConnectedAgent",
           "AgentProfile", "ModelSimulator", "write_scene_brief",
           "resolve_topology", "adaptive_allocator", "allocator_slot_counts",
           "optimize", "optimize_for_rl", "filter_rl_rows", "group_signal",
           "recommend", "select_for_sft", "select_for_rl",
           "trim_unanimous_groups", "training_rows", "export_training",
           "mine_traces", "dimensions_from_traces", "split_pseudo_production",
           "flaw_rows", "leakage_report", "drop_leaky_rows",
           "simulate_from_traces", "rows_from_otel",
           # format_* helpers are presentation, not mechanics; still
           # importable for the CLI but out of the public contract
           # (SDK-is-the-engine split, Sahana 2026-08-27).
           "load_traces", "trace_report",
           "preflight", "dataset_report",
           "classify_failure", "FAILURE_CLASSES",
           "run_judge", "evaluate", "ScoredData", "normalize_judge_result",
           "export_dataset", "export_preference", "build_preference_pairs",
           ]

_SATURATION_CAP = 50_000
_SEARCH_ARMS = dict(SEARCH_ARMS)
_reallocate = reallocate_search_arms
# Kill a hung Qwen slot after this wait. Omit the row. Not agent speech.
# Ping-pong is several HTTP calls; 24s dropped healthy 2-person traces
# and the replacement oversubscribed the GPU.
_HUNG_SLOT_S = 45.0
_RAW_TOOL_MARKUP = re.compile(r"</?tool_call>", re.I)
_TOOL_SCHEMA_DUMP = re.compile(
    r'"name"\s*:\s*"[^"]+".{0,500}"description"\s*:'
    r'.{0,500}"parameters"\s*:', re.I | re.S)


def _usable_rollout(row: dict) -> bool:
    """Infrastructure and parser failures are not training trajectories."""
    final = str((row or {}).get("final_text") or "").strip()
    if not final or final.lower().startswith("<agent error"):
        return False
    assistant_text = [final]
    for step in (row or {}).get("steps") or []:
        if isinstance(step, dict) and step.get("text") is not None:
            assistant_text.append(str(step["text"]))
    visible = "\n".join(assistant_text)
    return not (_RAW_TOOL_MARKUP.search(visible)
                or _TOOL_SCHEMA_DUMP.search(visible))


def _collect_finished(pending: dict, wait_s: float, *, retry: bool = False):
    """Take finished rollouts. Drop hung slots. Do not write a stub row."""
    results: list[dict] = []
    jobs_for: list = []
    if not pending or wait_s < 0:
        return results, jobs_for
    done, not_done = concurrent.futures.wait(pending, timeout=max(0.0, wait_s))
    for fut in done:
        results.append(fut.result())
        jobs_for.append(pending[fut])
    if not_done and retry:
        done, not_done = concurrent.futures.wait(not_done, timeout=max(0.0, wait_s))
        for fut in done:
            results.append(fut.result())
            jobs_for.append(pending[fut])
    return results, jobs_for


def _note(data: "SimulationData", stage: str) -> None:
    if stage not in data.stages:
        data.stages.append(stage)


def conversation(row: dict) -> list[dict]:
    """User/agent turns from prompt + steps. Tool calls stay on the assistant turn."""
    messages: list[dict] = []
    first = str(row.get("prompt") or "")
    if first:
        messages.append({"role": "user", "content": first})
    for step in row.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if step.get("user"):
            messages.append({"role": "user", "content": str(step["user"])})
            continue
        spoken = str(step.get("text") or "")
        if step.get("tool"):
            asst: dict[str, Any] = {"role": "assistant", "content": spoken}
            asst["tool_calls"] = [{
                "name": step.get("tool"),
                "arguments": step.get("arguments") or {},
            }]
            messages.append(asst)
            result = step.get("result")
            content = result if isinstance(result, str) else json.dumps(
                result, default=str)
            messages.append({"role": "tool", "name": step.get("tool"),
                             "content": content})
        elif spoken:
            messages.append({"role": "assistant", "content": spoken})
    final = str(row.get("final_text") or "")
    failed = final.startswith("<agent error")
    while len(messages) > 1 and messages[-1].get("role") == "user" and not failed:
        messages.pop()
    already = {str(m.get("content") or "") for m in messages
               if m.get("role") == "assistant" and str(m.get("content") or "").strip()}
    last = messages[-1] if messages else {}
    if final and final in already:
        return messages
    if final and last.get("role") == "assistant":
        if not str(last.get("content") or "").strip():
            last["content"] = final
    elif final:
        messages.append({"role": "assistant", "content": final})
    return messages


def _clean_faults(plan: Any) -> dict | None:
    """Fault modes only. world_state is a row field, never a faults key."""
    if not isinstance(plan, dict) or not plan:
        return None
    out = {k: v for k, v in plan.items() if k != "world_state"}
    return out or None


def _row_world(assignment: Any) -> str | None:
    if not isinstance(assignment, dict):
        return None
    world = assignment.get("world_state")
    if not world or world in {"unspecified", "unknown"}:
        return None
    return str(world)


_CONVERSATION_FIELDS = (
    "tier", "ask_family", "intent_known", "tool_known",
    "stance", "tone", "length", "ask", "vagueness", "phrasing",
    "pressure", "user", "texture", "history",
)


def _export_row(row: dict) -> dict:
    """Trainer-facing JSONL row. Search and embedder bookkeeping stay in memory."""
    out: dict[str, Any] = {
        "prompt": row.get("prompt", ""),
        "messages": row.get("messages") or conversation(row),
        "steps": row.get("steps") or [],
        "final_text": str(row.get("final_text", "")),
        "scenario_id": row.get("scenario_id") or "",
    }
    for key in _CONVERSATION_FIELDS:
        if key in row and row[key] is not None:
            out[key] = row[key]
    world = row.get("world_state")
    if world and world not in {"unspecified", "unknown"}:
        out["world_state"] = world
    faults = _clean_faults(row.get("faults"))
    if faults:
        out["faults"] = faults
    if row.get("fault_detected"):
        out["fault_detected"] = True
    if row.get("reward") is not None:
        out["reward"] = row["reward"]
        reason = row.get("grader_reason") or row.get("reason")
        if reason:
            out["reason"] = reason
        # Who labeled it travels with the label: a conduct score, a judge,
        # and a human override must stay distinguishable on disk.
        if row.get("label_source"):
            out["label_source"] = row["label_source"]
    if row.get("qwen_reward") is not None:
        out["qwen_reward"] = row["qwen_reward"]
    if "llm_reward" in row:
        out["llm_reward"] = row.get("llm_reward")
        out["llm_reason"] = row.get("llm_reason")
    if row.get("quality") is not None:
        out["quality"] = row["quality"]
        out["quality_reason"] = row.get("quality_reason") or ""
        if row.get("quality_scores"):
            out["quality_scores"] = row["quality_scores"]
    return out


@dataclass
class SimulationData:
    trajectories: list[dict] = field(default_factory=list)
    arm_yield: dict = field(default_factory=dict)
    stopped_because: str = "budget"
    declared_tools: set = field(default_factory=set)
    stages: list[str] = field(default_factory=list)
    scaffold_chars: int = 0
    degraded: list[str] = field(default_factory=list)
    semantic: bool = False
    profile: AgentProfile | None = None
    embedder_name: str = ""
    elapsed_seconds: float = 0.0
    rows_per_second: float = 0.0
    arm_weights: dict = field(default_factory=dict)
    scenario_generation_seconds: float = 0.0
    embedding_selection_seconds: float = 0.0
    rollout_seconds: float = 0.0
    row_seconds: list = field(default_factory=list)
    unique_prompts: int = 0
    scene_brief: str = ""
    scene_brief_seconds: float = 0.0
    first_row_seconds: float = 0.0
    semantic_duplicate_rate: float | None = None
    unique_behavior_signatures: int = 0
    coverage_curve: list[dict] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    search: dict = field(default_factory=dict)
    budget: int = 0
    path: str = ""
    mode: str = "explore"
    repeat_policy: str = "none"
    n_situations: int | None = None
    requests_per_situation: int = 1
    rollouts_per_request: int = 1
    unique_situations: bool = False
    allocator: dict = field(default_factory=dict)

    @property
    def metadata(self) -> dict:
        """Small structured summary of how this run was generated.
        Mechanics only — interpretation (evidence labels, percentages,
        display copy) belongs to the consumer, never the SDK."""
        strategy = self.search.get("strategy") or {}
        mining = self.search.get("trace_mining") or {}
        weight = strategy.get("steering_weight") or {}
        targeted = sum(1 for r in self.trajectories
                       if (r.get("steering") or {}).get("origin")
                       == "targeted")
        return {
            "strategy": strategy.get("resolved"),
            "trace_count": mining.get("n_traces", 0),
            "trace_regions": mining.get("regions"),
            "applied_steering_weight": weight.get("applied"),
            "targeted_rows": targeted,
            "background_rows": len(self.trajectories) - targeted,
        }

    def _rewrite(self, path: str | None = None) -> None:
        dest = path or self.path
        if dest:
            self.save(dest)

    def grade(self, grader=None, *, judge=None, llm: bool = False,
              llm_spec: str | None = None,
              api_key: str | None = None, path: str | None = None,
              concurrency: int = 32, llm_concurrency: int = 16):
        """Grade after simulation with hosted Qwen or a custom callable.

        With no callable, this is the binary hosted-Qwen grader and reads
        ``VLLM_API_KEY`` from the environment. Pass a callable for a custom
        score. Simulation itself never invokes this method by default.

        ``judge=`` is the contract path: any callable honoring the judge
        contract (see ``zeroproof_simulations.judging``). It returns a
        ``ScoredData`` of copies — trajectories here stay unmodified, judge
        errors are marked per-row instead of coerced to 0 — and its output
        feeds ``export_training`` and ``simulate(traces=...)`` directly.
        """
        if judge is not None:
            from .judging import run_judge
            return run_judge(self.trajectories, judge, source="grade",
                             concurrency=min(int(concurrency), 32))
        if llm:
            return self.llm_grade(spec=llm_spec, concurrency=llm_concurrency,
                                 api_key=api_key, path=path)
        if not callable(grader):
            return self.grade_llm(spec=llm_spec, concurrency=llm_concurrency,
                                  api_key=api_key, path=path)
        def score(t):
            out = grader(t)
            flagged = bool(t.get("faults"))
            if isinstance(out, dict):
                return (float(out.get("reward", 0.0)), str(out.get("reason", "")),
                        flagged or bool(out.get("fault_detected")))
            return float(out), "graded", flagged
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            for t, (reward, reason, flagged) in zip(
                    self.trajectories, pool.map(score, self.trajectories)):
                t["reward"], t["grader_reason"], t["reason"] = reward, reason, reason
                if flagged:
                    t["fault_detected"] = True
                else:
                    t.pop("fault_detected", None)
        self.arm_yield = {}
        for t in self.trajectories:
            slot = self.arm_yield.setdefault(t["arm"], {"executed": 0, "failing": 0})
            slot["executed"] += 1
            slot["failing"] += t["reward"] < 1.0
        self._rewrite(path)
        return self

    def llm_grade(self, *, spec: str | None = None, concurrency: int = 16,
                  api_key: str | None = None, path: str | None = None):
        """Advisory LLM pass. Leaves deterministic reward untouched."""
        if not resolve_judge_key(api_key, spec):
            raise RuntimeError(MISSING_JUDGE_KEY)
        policy = str(self.profile.policy or "") if self.profile else ""
        tools = list(self.profile.tools) if self.profile else []
        apply_llm_grade(
            self.trajectories, policy=policy, tools=tools, backend_spec=spec,
            api_key=api_key, concurrency=concurrency, degraded=self.degraded)
        self._rewrite(path)
        return self

    def grade_llm(self, *, spec: str | None = None, base_url: str | None = None,
                  model: str | None = None, concurrency: int = 16,
                  api_key: str | None = None, path: str | None = None,
                  limit: int | None = None, prompt: str | None = None):
        """Binary 0/1 situation grade. Default brain is hosted Qwen."""
        require_judge_key(api_key, spec=spec, base_url=base_url, model=model)
        policy = str(self.profile.policy or "") if self.profile else ""
        tools = list(self.profile.tools) if self.profile else []
        report = apply_grade_llm(
            self.trajectories, policy=policy, tools=tools, backend_spec=spec,
            base_url=base_url, model=model, api_key=api_key,
            prompt=prompt, concurrency=concurrency, limit=limit,
            degraded=self.degraded)
        self._rewrite(path)
        return report

    def rank(self, *, path: str | None = None) -> dict:
        """Second-pass quality scores. Leaves conduct ``reward`` untouched.

        Writes ``quality``, ``quality_reason``, ``quality_scores`` on each
        trajectory and rewrites the saved JSONL, or ``path`` if you pass one.
        """
        rank_rows(self.trajectories)
        self._rewrite(path)
        return summarize_quality(self.trajectories)

    def rows(self) -> list[dict]:
        return [_export_row(t) for t in self.trajectories]

    def push(self, name: str, *, api_key: str | None = None,
             parent: str | None = None) -> dict:
        """Upload this run to your Zero Proof Labs account as a dataset.

        ``api_key`` defaults to the ``ZEROPROOF_API_KEY`` env var; get one at
        https://www.zeroproofai.com/platform. Pass ``parent`` (a ``ds_...``
        id) when this run iterates on an existing dataset, so lineage shows
        on the platform. Returns the registry entry with ``datasetId``.
        """
        return push_rows(self.rows(), name, api_key=api_key, parent=parent)

    def sft_rows(self, failures_only: bool = True) -> list[dict]:
        return [{"prompt": t["prompt"], "rejected_response": t["final_text"],
                 "chosen_response": None, "tool_trace": t["steps"],
                 "reward": t["reward"],
                 "reason": t.get("grader_reason", t.get("reason")),
                 "arm": t["arm"]}
                for t in self.trajectories
                if not failures_only or (t["reward"] is not None and t["reward"] < 1.0)]

    def save(self, path: str, *, meta: bool = False) -> str:
        dest = Path(path)
        self.path = str(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        try:
            with open(tmp, "w") as fh:
                for t in self.trajectories:
                    fh.write(json.dumps(_export_row(t), default=str) + "\n")
            tmp.replace(dest)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        if meta:
            sidecar = path[:-6] + ".meta.json" if path.endswith(".jsonl") else path + ".meta.json"
            rows_by_minute: dict[str, int] = {}
            for s in self.row_seconds:
                key = str(int(s // 60))
                rows_by_minute[key] = rows_by_minute.get(key, 0) + 1
            with open(sidecar, "w") as fh:
                json.dump({
                    # The agent spec: a trainer loading this JSONL later
                    # needs the policy and tool schemas the run knew.
                    "system_prompt": str(getattr(self.profile, "policy", "")
                                         or ""),
                    "tools": list(getattr(self.profile, "tools", None) or []),
                    "stopped_because": self.stopped_because,
                    "coverage": self.coverage,
                    "coverage_curve": self.coverage_curve,
                    "arm_weights": self.arm_weights,
                    "arm_yield": self.arm_yield,
                    "search": self.search,
                    "budget": self.budget,
                    "elapsed_seconds": self.elapsed_seconds,
                    "timings": {
                        "scenario_generation_seconds": round(
                            self.scenario_generation_seconds, 3),
                        "embedding_selection_seconds": round(
                            self.embedding_selection_seconds, 3),
                        "rollout_seconds": round(self.rollout_seconds, 3),
                        "scene_brief_seconds": round(self.scene_brief_seconds, 3),
                        "first_row_seconds": round(self.first_row_seconds, 3),
                    },
                    "rows_by_minute": rows_by_minute,
                    "degraded": self.degraded,
                    "stages": self.stages,
                }, fh, indent=2, default=str)
        return path

    def report(self) -> dict:
        """Run-level coverage summary (same as ``data.coverage``)."""
        return dict(self.coverage)


def llm_grade(data: SimulationData, *, spec: str | None = None,
              concurrency: int = 16, api_key: str | None = None,
              path: str | None = None) -> SimulationData:
    """Module helper: advisory LLM scores on an existing SimulationData."""
    return data.llm_grade(spec=spec, concurrency=concurrency,
                         api_key=api_key, path=path)


def grade_llm(source, *, spec: str | None = None, base_url: str | None = None,
              model: str | None = None, concurrency: int = 16,
              api_key: str | None = None, path: str | None = None,
              limit: int | None = None, output: str | None = None,
              prompt: str | None = None, policy: str = "",
              tools: list | None = None):
    """Binary 0/1 situation grade. Default brain is hosted Qwen.

    ``source`` is a ``SimulationData``, a JSONL path, or a row list.
    Writes ``reward`` 0 or 1 and a one-sentence ``reason``. Keeps the
    previous score as ``qwen_reward`` when present. Does not run during
    ``simulate()``. Search does not read ``reward``. ``limit`` grades
    that many rows then stops. Hosted Qwen reads ``VLLM_API_KEY``.
    For a path or row list, pass ``policy=`` and ``tools=`` so the judge
    sees the agent's rules; a ``SimulationData`` supplies its own.
    """
    if isinstance(source, SimulationData):
        return source.grade_llm(spec=spec, base_url=base_url, model=model,
                                concurrency=concurrency, api_key=api_key,
                                path=path or output, limit=limit, prompt=prompt)
    from .quality import _load_jsonl, _write_jsonl
    if isinstance(source, (str, Path)):
        rows = _load_jsonl(source)
        src = str(source)
    else:
        rows = list(source)
        src = ""
    report = apply_grade_llm(
        rows, policy=str(policy or ""), tools=list(tools or []),
        backend_spec=spec, base_url=base_url, model=model,
        api_key=api_key, prompt=prompt, concurrency=concurrency, limit=limit)
    dest = path or output or src
    if dest:
        _write_jsonl(dest, rows)
    if isinstance(source, list):
        for dst, src_row in zip(source, rows):
            dst["reward"] = src_row.get("reward")
            if src_row.get("reason"):
                dst["reason"] = src_row.get("reason")
            if src_row.get("qwen_reward") is not None:
                dst["qwen_reward"] = src_row.get("qwen_reward")
    report["path"] = dest or src
    return report


grade = grade_llm


def rank(source, *, output: str | None = None,
         min_quality: float | None = None) -> dict:
    """Score already-generated rows. ``source`` is a JSONL path, a row list,
    or a ``SimulationData``. Does not change ``simulate()`` or ``reward``.
    """
    if isinstance(source, SimulationData):
        return source.rank(path=output)
    return rank_source(source, output=output, min_quality=min_quality)




def _mutation_worthy(row: dict) -> bool:
    """Re-roll and mutate on tool/sandbox faults. Ignores any score column.

    A step's ``result`` is ``Any`` by the canonical schema, and most real tools
    return text. Six of this package's own adapters do: `from_langchain`,
    `from_openai_agents`, `claude_code` and friends all store
    ``str(...)`` there. Calling ``.get`` on it straight crashed
    ``simulate(agent=...)`` with `'str' object has no attribute 'get'` for every
    one of them. `grading._as_dict` is the shared way to ask a result for a
    field; a plain string simply has no status, which is the right answer.
    """
    if row.get("faults"):
        return True
    for step in row.get("steps") or []:
        if not isinstance(step, dict):
            continue
        status = str(_as_dict(step.get("result")).get("status", "")).lower()
        if status in {"error", "timeout", "not_found", "denied", "malformed"}:
            return True
    return False


def _cell_key(row: dict) -> str:
    return _cell_key_from_row(row)


def _record_coverage(data: SimulationData, trajectories: list[dict], *,
                     cells: set[str], shape_keys: set[str],
                     arm_weights: dict | None,
                     batch_fresh_rate: float | None = None,
                     mean_batch_novelty: float | None = None,
                     stopped_because: str | None = None) -> None:
    point = coverage_point(
        trajectories, cells=cells, shape_keys=shape_keys,
        arm_weights=arm_weights, batch_fresh_rate=batch_fresh_rate,
        mean_batch_novelty=mean_batch_novelty, stopped_because=stopped_because)
    if data.coverage_curve and stopped_because is None:
        prev = data.coverage_curve[-1]
        if (prev.get("n_rows") == point["n_rows"]
                and prev.get("batch_fresh_rate") == batch_fresh_rate):
            return
    data.coverage_curve.append(point)


def _prompt_arm(prompt: str, generator: Any) -> str:
    meta = (generator.meta.get(prompt)
            or getattr(generator, "last_candidate_provenance", {}).get(prompt) or {})
    arm = str(meta.get("arm") or generator.provenance.get(prompt, "open_ended"))
    return arm if arm in _SEARCH_ARMS else "open_ended"


def _stratified_prompts(candidates: list[str], take: int, generator: Any, *,
                        used_situations: set[str]) -> list[str]:
    """Breadth-first pick: arm quotas, prefer unseen situation keys."""
    if not candidates or take <= 0:
        return []
    by_arm: dict[str, list[str]] = {arm: [] for arm in _SEARCH_ARMS}
    for prompt in candidates:
        by_arm.setdefault(_prompt_arm(prompt, generator), []).append(prompt)

    def sort_key(prompt: str) -> tuple[int, str]:
        meta = generator.meta.get(prompt) or {}
        sk = _situation_key_from_meta(meta, prompt)
        return (0 if sk and sk not in used_situations else 1, prompt)

    for arm in by_arm:
        by_arm[arm].sort(key=sort_key)

    picked: list[str] = []
    seen: set[str] = set()
    for arm in _SEARCH_ARMS:
        for prompt in by_arm.get(arm) or []:
            if prompt not in seen:
                picked.append(prompt)
                seen.add(prompt)
                break
        if len(picked) >= take:
            return picked[:take]
    leftover = [prompt for prompt in candidates if prompt not in seen]
    leftover.sort(key=sort_key)

    def _tier(prompt: str) -> str:
        meta = (generator.meta.get(prompt)
                or getattr(generator, "last_candidate_provenance", {}).get(prompt)
                or {})
        assignment = meta.get("assignment") or meta.get("scenario_dimensions") or {}
        return behavior_tier(assignment if isinstance(assignment, dict) else {})

    picked.extend(mix_items_by_tier(leftover, take - len(picked), _tier))
    return picked[:take]


def _backend_spec(backend: str) -> str:
    text = str(backend).strip()
    if text.startswith(("vllm:", "ollama:", "openai:")):
        return text
    return "vllm:Qwen/Qwen3-4B-Instruct-2507@" + text.rstrip("/")


def _read_spec_file(path: Path) -> Any:
    text = path.read_text()
    if not str(text).strip():
        return None
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
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
    return " " not in raw and len(raw) < 64


def _spec_from_path(text: str) -> dict | None:
    """Load tools+policy from a spec file or folder. None if it is not a path."""
    raw = Path(text).expanduser()
    roots = [Path.cwd(), Path(__file__).resolve().parents[1]]
    candidates: list[Path] = []
    for root in ([Path()] if raw.is_absolute() else roots):
        base = raw if raw.is_absolute() else (root / raw)
        candidates.append(base)
        if not base.suffix:
            candidates.extend([
                Path(str(base) + ".json"),
                Path(str(base) + ".yaml"),
                Path(str(base) + ".yml"),
                base / "spec.json",
                base / "spec.yaml",
                base / "agent.json",
            ])
            if "/" not in str(text) and "\\" not in str(text):
                spec_dir = root / "specs" / raw.name
                candidates.extend([
                    spec_dir, spec_dir / "spec.json", spec_dir / "spec.yaml"])
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
            return loaded
    return None


_SPEC_SKIP = {"tools", "policy", "system_prompt", "system", "instructions",
              "rules", "situations",
              "name", "id", "version", "model", "backend", "simulator"}


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


def _apply_spec(spec: Any, tools: list | None, policy: str | None,
                situations: list | None) -> tuple[list | None, str | None, list]:
    extra_sit = list(situations or [])
    if spec is None:
        return tools, policy, extra_sit
    if isinstance(spec, str) and spec.strip():
        loaded = _spec_from_path(spec.strip())
        if loaded is not None:
            spec = loaded
        elif _looks_like_spec_path(spec):
            raise FileNotFoundError(
                f"simulate(spec={spec.strip()!r}) found no spec.json at that path.")
        else:
            policy = (str(policy or "").rstrip() + "\n" + spec.strip()).strip()
            return tools, policy, extra_sit
    if not isinstance(spec, dict):
        return tools, policy, extra_sit
    if spec.get("tools"):
        tools = list(tools or []) + list(spec["tools"])
    blob = (spec.get("system_prompt") or spec.get("system")
            or spec.get("policy") or spec.get("instructions")
            or spec.get("rules"))
    if isinstance(blob, list):
        blob = "\n".join(str(item) for item in blob)
    extra = _spec_extra_text(spec)
    merged = "\n".join(part for part in (blob, extra) if part)
    if merged:
        policy = (str(policy or "").rstrip() + "\n" + str(merged).strip()).strip()
    extra_sit.extend(str(s) for s in (spec.get("situations") or []) if s)
    return tools, policy, extra_sit


def _kind_from_spec(spec: Any, policy: str) -> str:
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


def _row_conversation(meta: dict, prompt: str, default_seed: int) -> dict:
    """Conversation labels already on meta, or the same draw the writer used."""
    ready = meta.get("conversation")
    if isinstance(ready, dict) and ready.get("tier"):
        return {k: v for k, v in ready.items() if v is not None}
    assignment = meta.get("assignment") or meta.get("scenario_dimensions") or {}
    if not isinstance(assignment, dict):
        assignment = {}
    rid = str(meta.get("region_id") or prompt)
    rnd = int(meta.get("round") or 0)
    row_seed = int(meta.get("seed", default_seed))
    tags = sample_cell_tags(row_seed, rnd, rid, assignment)
    from .generator import _ask_family
    return conversation_features(
        assignment, tags,
        ask_family=_ask_family(row_seed, rnd, rid),
        tool=str(assignment.get("tool") or ""))


def _situation_key_from_meta(meta: dict, prompt: str = "") -> str:
    """Coverage cell key for unique-situation dedup."""
    assignment = meta.get("assignment") or meta.get("scenario_dimensions")
    if isinstance(assignment, dict) and assignment:
        return json.dumps(assignment, sort_keys=True, default=str)
    rid = meta.get("region_id")
    if rid:
        return str(rid)
    if prompt:
        return f"prompt:{prompt}"
    return ""


_MODE_PRESETS = {
    "explore": {"n_req": 1, "k": 1, "repeat_policy": "none"},
    "sft": {"n_req": 3, "k": 1, "repeat_policy": "adaptive"},
    "rl": {"n_req": 1, "k": 8, "repeat_policy": "adaptive"},
    "adaptive": {"n_req": 1, "k": 1, "repeat_policy": "adaptive"},
}

_ALIAS_NAMES = {
    "unique", "repeats", "rollouts_per_prompt", "n", "phrasings",
    "repeat_policy", "policy",
}
_MOVED_NAMES = {
    "concurrency", "dimensions", "simulator", "backend", "fault_rate", "risk",
    "texture", "max_turns", "avg_turns", "min_user_turns", "temperature", "seed", "grader",
    "llm_spec", "embedder", "seed_prompts", "extra_situations",
    "prefer_success",
    # steering_weight stays an advanced knob until the tranche-1
    # calibration lands (Sahana, 2026-08-27): strategy default is auto,
    # raw weight hidden from the named surface.
    "steering_weight",
}


def _parse_situations_arg(situations: Any, extra_situations: list | None
                          ) -> tuple[int | None, list[str]]:
    """Public ``situations=`` is N (int). Seed openers come from advanced."""
    seeds: list[str] = []
    for item in extra_situations or []:
        text = str(item or "").strip()
        if text:
            seeds.append(text)
    if situations is None:
        return None, seeds
    if isinstance(situations, bool) or not isinstance(situations, int):
        raise ValueError(
            "situations= is N, an int. Pass seed openers in advanced['seed_prompts']")
    return max(1, int(situations)), seeds


def _merge_advanced(advanced: dict | None, passed: dict) -> tuple[dict, dict]:
    """Split silent aliases from advanced knobs. Unknown names error."""
    cfg = dict(advanced or {})
    aliases: dict[str, Any] = {}
    for key, val in passed.items():
        if key in _ALIAS_NAMES:
            aliases[key] = val
        elif key in _MOVED_NAMES:
            cfg[key] = val
        else:
            raise TypeError(
                f"simulate() got unexpected keyword argument {key!r}")
    return cfg, aliases


def resolve_topology(*, mode: str | None = None, repeat_policy: str | None = None,
                     unique: bool = False, unique_situations: bool = False,
                     requests_per_situation: int | None = None,
                     n: int | None = None, phrasings: int | None = None,
                     rollouts_per_request: int | None = None,
                     repeats: int | None = None,
                     rollouts_per_prompt: int | None = None) -> dict[str, Any]:
    """Map public knobs. Phrasings (n) are wordings per situation; repeats (k) are reruns per phrasing."""
    k_vals = [int(x) for x in (rollouts_per_request, repeats, rollouts_per_prompt)
              if x is not None]
    if len(set(k_vals)) > 1:
        raise ValueError("pass rollouts_per_request= or repeats=, not both")
    n_vals = [int(x) for x in (requests_per_situation, n, phrasings)
              if x is not None]
    if len(set(n_vals)) > 1:
        raise ValueError("pass requests_per_situation=, phrasings=, or n=, not both")
    k_explicit = bool(k_vals)
    n_explicit = bool(n_vals)
    new_cards = bool(unique_situations or unique)
    mode_name = str(mode or "").strip().lower() or None
    policy_name = str(repeat_policy or "").strip().lower() or None
    if policy_name in {"unique"}:
        policy_name = "none"
        new_cards = True
    if mode_name in _MODE_PRESETS and policy_name is None:
        policy_name = _MODE_PRESETS[mode_name]["repeat_policy"]
    if policy_name == "none":
        mode_name = mode_name or "explore"
        new_cards = True
    elif mode_name is None and policy_name == "adaptive":
        mode_name = "adaptive"
    elif mode_name is None:
        mode_name = "explore"
        policy_name = "none"
        new_cards = True
    if mode_name not in _MODE_PRESETS:
        raise ValueError(
            "mode= must be explore, sft, rl, or adaptive")
    preset = _MODE_PRESETS[mode_name]
    if policy_name is None:
        policy_name = preset["repeat_policy"]
    n_req = int(preset["n_req"])
    k = int(preset["k"])
    if n_explicit:
        n_req = max(1, n_vals[0])
    if k_explicit:
        k = max(1, k_vals[0])
    if new_cards:
        if not n_explicit:
            n_req = 1
        if not k_explicit:
            k = 1
        if policy_name == "none":
            policy_name = "none"
        elif n_req == 1:
            policy_name = policy_name or "adaptive"
    return {
        "mode": mode_name,
        "repeat_policy": policy_name,
        "n_req": n_req,
        "k": k,
        "k_explicit": k_explicit,
        "n_explicit": n_explicit,
        "unique_situations": new_cards,
    }


def simulate(agent: Any = None, *, spec: Any = None,
             tools: list[dict] | None = None, system_prompt: str | None = None,
             budget: int | None = 1000, time_budget: float | None = None,
             until: str = "compute", mode: str = "explore",
             situations: int | None = None,
             requests_per_situation: int | None = None,
             rollouts_per_request: int | None = None,
             unique_situations: bool = False,
             grade: bool = False, llm_grade: bool = False,
             traces: Any = None,
             grader: Any = None,
             strategy: str = "auto",
             seeds: list | None = None,
             scaffold: str | None = None,
             output: str | None = None,
             advanced: dict | None = None,
             **passed: Any) -> SimulationData:
    """Inspect an agent, generate situations, and roll them out.

    Input is an intent or an agent: ``system_prompt`` alone, ``tools``
    plus a prompt, or ``spec=``. Search writes a grid of human requests
    (ordinary, vague, complex, adversarial) and a spread of agent replies.
    It spends ``budget`` rows and ``time_budget`` seconds on new coverage.
    No default ``reward``. Pass ``grade=True`` for the deterministic
    conduct grade, or grade later with ``grade()``. A callable
    ``grader=`` is the only in-simulate score hook.

    ``traces=`` (rows or a JSONL path of production traces) aims the
    covering grid at the behaviors those traces show instead of the whole
    space, and drops any generated row that near-copies a source trace,
    so held-out traces stay out of training. Without it the grid comes
    from the agent's tools and policy alone (cold start).

    ``scaffold=`` is generation-only guidance appended to the system prompt
    of the MODEL-BACKED teacher during rollout (and to the scene writer).
    It never enters ``profile.policy``, so exports and evals stay on the
    plain policy; it is ignored for user-supplied callable agents. Measured
    to help some agents and hurt others — configure per agent, no default.

    Variation is three independent counts. Do not collapse them.
    ``situations`` (N) is distinct worlds. ``requests_per_situation`` /
    ``phrasings`` (n) is different human wordings of one world.
    ``rollouts_per_request`` / ``repeats`` (k) is independent agent runs of
    the same wording. Follow-ups branch on that run.
    ``unique_situations=True`` keeps picking new worlds (n=1, k=1 unless you
    set them). Silent aliases: ``n`` / ``phrasings`` for n, ``repeats`` /
    ``rollouts_per_prompt`` for k, ``unique`` for unique_situations,
    ``policy`` for system_prompt. Writer completions are
    ``advanced["completions_per_request"]``. Seed openers are
    ``advanced["seed_prompts"]``.
    """
    cfg, aliases = _merge_advanced(advanced, passed)
    policy = resolve_system_prompt(system_prompt, aliases.get("policy"))
    # Generation-only teacher guidance. profile.policy and export stay plain.
    scaffold_text = str(scaffold or "").strip()
    unique_flag = bool(unique_situations or aliases.get("unique", False))
    repeats = aliases.get("repeats")
    rollouts_per_prompt = aliases.get("rollouts_per_prompt")
    n_alias = aliases.get("n")
    phrasings_alias = aliases.get("phrasings")
    repeat_policy = aliases.get("repeat_policy")
    k_arg = rollouts_per_request
    if k_arg == 1 and (repeats is not None or rollouts_per_prompt is not None):
        # A leftover default 1 plus an alias means the alias wins.
        k_arg = None
    topo = resolve_topology(
        mode=mode, repeat_policy=repeat_policy, unique=unique_flag,
        unique_situations=unique_flag,
        requests_per_situation=requests_per_situation, n=n_alias,
        phrasings=phrasings_alias,
        rollouts_per_request=k_arg, repeats=repeats,
        rollouts_per_prompt=rollouts_per_prompt)
    seed_prompts: list[str] = []
    for item in list(seeds or []) + list(cfg.pop("seed_prompts", None) or []):
        text = str(item or "").strip()
        if text:
            seed_prompts.append(text)
    for item in (cfg.pop("extra_situations", None) or []):
        text = str(item or "").strip()
        if text:
            seed_prompts.append(text)
    n_situations_target, seed_prompts = _parse_situations_arg(
        situations, seed_prompts)
    concurrency = int(cfg.pop("concurrency", 32))
    dimensions = cfg.pop("dimensions", None)
    simulator = cfg.pop("simulator", None)
    backend = cfg.pop("backend", None)
    explicit_fault = "fault_rate" in cfg or "risk" in cfg
    fault_rate = float(cfg.pop("fault_rate", DEFAULT_FAULT_RATE))
    risk = cfg.pop("risk", None)
    if risk is not None:
        fault_rate = float(risk)
    elif str(topo["mode"]) == "rl" and not explicit_fault:
        fault_rate = 0.8
    texture = cfg.pop("texture", None)
    max_turns = cfg.pop("max_turns", None)
    avg_turns = float(cfg.pop("avg_turns", 4))
    min_user_turns = max(1, int(cfg.pop("min_user_turns", 1)))
    temperature = cfg.pop("temperature", None)
    seed = int(cfg.pop("seed", 0))
    # The named grader= parameter wins; advanced={"grader": ...} stays as
    # the legacy spelling. Both route to one application path below.
    grader = grader if grader is not None else cfg.pop("grader", None)
    cfg.pop("grader", None)
    llm_grade = bool(llm_grade or cfg.pop("llm_grade", False))
    llm_spec = cfg.pop("llm_spec", None)
    embedder = cfg.pop("embedder", "hash")
    until_key = str(until or "compute").strip().lower()
    if until_key in {"first", "saturation"}:
        until_sat = True
        until_key = "saturation"
    elif until_key in {"compute", "budget_only", "budget", "time"}:
        until_sat = False
        until_key = "compute"
    else:
        raise ValueError(
            "until= must be compute, saturation, first, or budget_only")
    if time_budget is None or float(time_budget) <= 0:
        time_budget = None
    else:
        time_budget = float(time_budget)
    repeat_count = int(topo["k"])
    n_req = int(topo["n_req"])
    unique_cards = bool(topo["unique_situations"])
    if topo["mode"] == "adaptive" and not unique_cards:
        adapt = adaptive_allocator(time_budget, until_key)
        if not topo["n_explicit"]:
            n_req = int(adapt["n_req"])
        if not topo["k_explicit"]:
            repeat_count = int(adapt["k"])
    # Topology caps n/k. Do not shrink writer flight or skip ingest.
    explore_only = False
    # Adaptive defers extra k so verify can react to behavior.
    k_immediate = bool(topo["k_explicit"] or topo["mode"] == "rl")
    started = time.monotonic()
    # Report setup immediately; callers should never stare at an absent file
    # while inspection or backend construction is in progress.
    out_path = Path(output).expanduser() if output else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Path(str(out_path) + ".progress.json").write_text(json.dumps({
            "stage": "setup", "rows": 0, "scenario_s": 0, "rollout_s": 0}))
        print("simulate setup rows=0", flush=True)
    advanced = cfg
    if texture is not None:
        advanced["texture_rate"] = float(texture)
    mutate_failures = bool(advanced.pop("mutate_failures", True))
    pool_size = int(advanced.pop("per_round", 80))
    _writer_raw = advanced.pop("scenario_concurrency", None)
    # Writer flight is a scheduler internal. Topology (unique / explore)
    # does not change it. Default 4; too many starves rollouts.
    if _writer_raw is None:
        scenario_concurrency = 4
    else:
        scenario_concurrency = max(1, int(_writer_raw))
    writer_flight = max(1, scenario_concurrency)
    scenarios_per_request = max(1, int(advanced.pop("scenarios_per_request", 8)))
    # A unique-situation run must walk the planned grid. Previously the
    # public unique=True knob still left the model writer in weighted
    # resampling mode unless callers also knew about this private switch.
    distinct_cards = bool(advanced.pop("distinct_cards", unique_cards))
    if "completions_per_request" in advanced:
        n_ceiling = max(1, min(8, int(advanced["completions_per_request"])))
    else:
        n_ceiling = 1
    advanced.pop("completions_per_request", None)
    completions_per_request = n_ceiling
    extra_cards = max(0, int(advanced.pop("extra_cards", 1)))
    hung_slot_s = float(advanced.pop("hung_slot", _HUNG_SLOT_S))
    advanced.pop("scene_brief", None)
    # Which weights produced each row: rounds of the continual loop are
    # indistinguishable without it (base and every adapter can share a
    # model name). advanced["model_version"] overrides; the resolved
    # backend model is the default; callable agents record their name.
    model_version_tag = str(advanced.pop("model_version", "") or "").strip()
    if not model_version_tag:
        if agent is not None and callable(agent) and not isinstance(agent, str):
            model_version_tag = getattr(agent, "__name__", "callable-agent")
        else:
            try:
                from .agents import parse_backend_spec as _pbs, \
                    default_simulator_spec as _dss
                _spec = agent if isinstance(agent, str) else _dss()
                model_version_tag = _pbs(_spec)[1]
            except Exception:
                model_version_tag = "unknown"

    tools, policy, spec_sits = _apply_spec(spec, tools, policy, [])
    seed_prompts.extend(str(s).strip() for s in spec_sits if str(s).strip())
    # simulate-from-seeds: a few example asks are a behavior request,
    # not the situation list. With an explicit situations=N target the
    # engine runs the coordinate search itself, amplifying the examples
    # across phrasing/stance/language axes to N distinct situations
    # before generation. Disclosed in search["seed_amplification"].
    seed_amp_report = None
    if (seed_prompts and n_situations_target
            and len(seed_prompts) < int(n_situations_target)):
        given = len(seed_prompts)
        seed_prompts = amplify_seeds(
            seed_prompts, int(n_situations_target), policy=policy,
            backend_spec=None)
        seed_amp_report = {"given": given,
                           "target": int(n_situations_target),
                           "total": len(seed_prompts),
                           "minted": len(seed_prompts) - given}
    profile = inspect(agent, tools=tools, system_prompt=policy)
    tools = list(profile.tools or [])
    policy = str(profile.policy or "")
    gen_policy = f"{policy}\n\n{scaffold_text}" if scaffold_text else policy
    writer_kind = _kind_from_spec(spec, policy)
    if agent is None and not tools and not policy:
        raise ValueError(
            "simulate needs an agent, tools=, or a system prompt.")
    trace_rows: list[dict] = []
    trace_focused = False
    optimizer_state = None

    def _apply_allocation(region_list) -> None:
        return None
    # strategy= names the coverage stance explicitly (doctrine: traces
    # change the coverage DISTRIBUTION, never the space). auto resolves
    # descriptively and records its reason; it never picks "targeted" on
    # its own - narrowing is an explicit user choice (early-bias risk,
    # finding 46).
    if strategy not in ("auto", "broad", "trace", "targeted"):
        raise ValueError("strategy= must be auto, broad, trace, or targeted")
    resolved_strategy = strategy
    if strategy == "auto":
        resolved_strategy = "trace" if traces is not None else "broad"
    if resolved_strategy in ("trace", "targeted") and traces is None:
        raise ValueError(f"strategy='{resolved_strategy}' needs traces=")
    # steering_weight is the calibration knob for doctrine point 5 (how
    # much the trace-aimed distribution outweighs background coverage).
    # An advanced knob until tranche 1 lands; accepted via
    # steering_weight= or advanced=. Over a trace-focused grid, weight w
    # sends each structured card draw to the front (trace-mined) half of
    # the steered axes with probability w; rows drawn that way carry
    # row["steering"] = {"origin": "targeted"}. None means "rule
    # decides" (currently: no bias), an explicit number is an override.
    steering_weight = cfg.pop("steering_weight", None)
    if steering_weight is not None:
        try:
            steering_weight = float(steering_weight)
        except (TypeError, ValueError):
            raise ValueError("steering_weight= must be a number in [0, 1]")
        if not 0.0 <= steering_weight <= 1.0:
            raise ValueError("steering_weight= must be a number in [0, 1]")
        if traces is None:
            raise ValueError("steering_weight= needs traces=")

    if traces is not None:
        if isinstance(traces, (str, Path)):
            from .quality import _load_jsonl
            trace_rows = [r for r in _load_jsonl(str(traces))
                          if isinstance(r, dict)]
        else:
            trace_rows = [r for r in traces if isinstance(r, dict)]
        # The optimizer's memory feeds the run it aims: regions from
        # the whole trace history, budget shares from their lifecycle.
        # Cells whose coordinates intersect a hot region's expansion
        # recipe draw extra weight proportional to its share; cells
        # outside every recipe keep base weight - that is the
        # exploration reserve in action.
        optimizer_state = behavior_state(
            trace_rows,
            targeted=[str(x) for x in
                      (advanced.pop("targeted_regions", None) or [])])
        _ALLOC_GAIN = 4.0

        def _allocation_boost(assignment: dict) -> float:
            factor = 1.0
            for region in optimizer_state["regions"]:
                share = region.get("budget_share") or 0.0
                if share <= 0:
                    continue
                recipe = region.get("recipe") or {}
                match = 0.0
                tools_r = recipe.get("tool") or []
                if tools_r and str(assignment.get("tool")) in tools_r:
                    match += 0.6
                conds = recipe.get("tool_condition") or []
                if conds and str(assignment.get("tool_condition")) in conds:
                    match += 0.4
                if match:
                    factor += _ALLOC_GAIN * share * match
            return factor

        def _apply_allocation(region_list) -> None:
            for region in region_list or []:
                boost = _allocation_boost(region.get("assignment") or {})
                if boost != 1.0:
                    region["weight"] = round(
                        float(region.get("weight") or 0.0) * boost, 6)

        if (trace_rows and dimensions is None
                and resolved_strategy != "broad"):
            # trace: denser near observed behaviors, background kept.
            # targeted: drop tools the traces never touched (narrow).
            dimensions = dimensions_from_traces(
                trace_rows, tools, policy,
                broaden=resolved_strategy != "targeted")
            trace_focused = True
    # The weight only applies over a trace-focused grid (its front-half
    # ordering is what the bias aims at) and only when nonzero; anything
    # else is exactly the unsteered draw and records no applied weight.
    applied_steering = (float(steering_weight)
                        if steering_weight and trace_focused else None)

    data = SimulationData(profile=profile, arm_weights=dict(_SEARCH_ARMS))
    data.scaffold_chars = len(scaffold_text)
    data.mode = topo["mode"]
    data.repeat_policy = topo["repeat_policy"]
    data.n_situations = n_situations_target
    data.requests_per_situation = n_req
    data.rollouts_per_request = repeat_count
    data.unique_situations = unique_cards
    _note(data, "agent ingestion")
    scene_box: dict[str, Any] = {"brief": ""}
    shape_box: dict[str, dict] = {}
    trace_exemplars: dict[str, list] = {}
    if trace_rows:
        # Real observed payloads beat invented ones: results mined from
        # the traces become the shape templates first; the model-written
        # pass below only fills tools the traces never showed.
        trace_exemplars = mine_result_exemplars(trace_rows)
        shape_box.update(exemplar_result_shapes(trace_exemplars))
    scene_thread: threading.Thread | None = None
    use_model_writer = not (
        simulator is False
        or (callable(simulator) and not isinstance(simulator, str)))
    if use_model_writer:
        scene_spec = simulator if isinstance(simulator, str) else None
        scene_t0 = time.monotonic()

        def _fill_scene() -> None:
            # Shapes first: every rollout benefits, and the writer can run
            # its first waves without the scene brief.
            shapes = write_result_shapes(tools, backend_spec=scene_spec)
            if shapes:
                # setdefault, not update: a template mined from a real
                # trace outranks a model-written guess for that tool.
                for shape_name, shape in shapes.items():
                    shape_box.setdefault(shape_name, shape)
            elif "result_shapes_unavailable" not in data.degraded:
                data.degraded.append("result_shapes_unavailable")
            brief = write_scene_brief(
                tools, gen_policy, backend_spec=scene_spec, kind=writer_kind)
            scene_box["brief"] = brief
            data.scene_brief = brief
            data.scene_brief_seconds = time.monotonic() - scene_t0
            if not brief and "scene_brief_unavailable" not in data.degraded:
                data.degraded.append("scene_brief_unavailable")

        scene_thread = threading.Thread(target=_fill_scene, daemon=True)
        scene_thread.start()

    fault_plans: dict = {}
    kind = profile.transport
    turns = (default_max_turns(n_tools=len(tools))
             if max_turns is None else max(1, int(max_turns)))
    turn_stats = new_turn_stats()
    runner_kw: dict[str, Any] = {
        "fault_plans": fault_plans, "max_turns": turns,
        "avg_turns": float(avg_turns), "min_user_turns": min_user_turns,
        "turn_stats": turn_stats,
    }
    if temperature is not None:
        runner_kw["temperature"] = float(temperature)
    # Slow customer backends need more than the tuned 60s per completion.
    rollout_timeout = float(cfg.pop("timeout", 60) or 60)
    if backend:
        spec_backend = _backend_spec(backend)
        url, model_name = parse_backend_spec(spec_backend)
        runner = local_model(url, model_name, tools=tools, system=gen_policy,
                             timeout=rollout_timeout,
                             result_shapes=shape_box, **runner_kw)
        simulator = simulator if simulator is not None else spec_backend
    elif agent is None or kind not in {"callable", "backend_spec", "http"}:
        runner = hosted_model(tools, system=gen_policy,
                              timeout=rollout_timeout,
                              result_shapes=shape_box, **runner_kw)
    else:
        runner, kind = resolve(agent, tools=tools, policy=policy, **runner_kw)
    generator = make_default_generator(
        tools, policy=policy, per_round=pool_size, seed=seed,
        dimensions=dimensions, simulator=simulator, kind=writer_kind,
        scenarios_per_request=scenarios_per_request,
        completions_per_request=completions_per_request,
        distinct_cards=distinct_cards, extra_cards=extra_cards,
        scene_brief=scene_box["brief"], time_budget=time_budget,
        run_started=started, mode=topo["mode"],
        steering_weight=applied_steering, **advanced)
    _apply_allocation(getattr(generator, "regions", None))
    planned_cell_keys = {
        json.dumps(region["assignment"], sort_keys=True, default=str)
        for region in (getattr(generator, "regions", None) or [])
        if isinstance(region, dict)
        and isinstance(region.get("assignment"), dict)
        and region["assignment"]
    }
    search = dict(_SEARCH_ARMS)
    generator.arm_weights = dict(_SEARCH_ARMS)
    model_obj = getattr(generator, "model", None)
    if model_obj is not None and hasattr(model_obj, "arm_weights"):
        model_obj.arm_weights = dict(_SEARCH_ARMS)
    model_backend = getattr(getattr(generator, "model", None), "backend_spec", None)
    if isinstance(model_backend, str):
        try:
            hosted_url, _ = parse_backend_spec(model_backend)
        except ValueError:
            hosted_url = None
        else:
            auth_err = missing_hosted_key(hosted_url)
            if auth_err:
                raise RuntimeError(auth_err)
            threading.Thread(
                target=touch_hosted, args=(hosted_url,),
                kwargs={"timeout": 5.0}, daemon=True).start()
    fault_plans.update(generator.fault_plans)
    declared = {str((t.get("function", t) or {}).get("name", ""))
                for t in tools or []} - {""}
    cap = budget if budget is not None else _SATURATION_CAP
    if time_budget is not None and budget is None:
        cap = _SATURATION_CAP
    data.budget = int(cap)
    search_plan = sampling_plan(time_budget)
    action_shapes, _ = action_space_targets(
        tools, max_len=int(search_plan["max_shape_len"]),
        cap=int(search_plan["enum_cap"]))
    induced_shape_keys: set[str] = set()

    def _record_shapes(rows: list[dict]) -> None:
        known = {shape.key() for shape in action_shapes}
        for row in rows:
            induced_shape_keys.update(
                induced_keys_from_trajectory(row, action_shapes, tools))
            observed = shape_from_trajectory(row, tools)
            if observed is not None and observed.key() not in known:
                action_shapes.append(observed)
                known.add(observed.key())
            if isinstance(row, dict):
                row["tools_used"] = [
                    step.get("tool") for step in (row.get("steps") or [])
                    if isinstance(step, dict) and step.get("tool")]

    resolved_embedder = resolve_embedder(embedder)
    data.embedder_name = str(getattr(resolved_embedder, "name", "unknown"))
    data.semantic = is_semantic(resolved_embedder)
    if not data.semantic:
        data.degraded.append("semantic_embedding_unavailable")
    archive = EmbeddingArchive(data.embedder_name, data.semantic)

    def scaled(plan: dict | None, key: str = "") -> dict | None:
        if not plan or fault_rate <= 0:
            return None
        if not keep_fault_plan(key, fault_rate, seed):
            return None
        return plan

    def _realized_dims(steps: list, faults: list) -> dict:
        # Rows without a grid cell (seeds, open-ended, behavior cards)
        # still get auditable coordinates - realized from what actually
        # happened, marked so audits can tell assigned from observed.
        tools_called = [str(s.get("tool")) for s in steps or []
                        if isinstance(s, dict) and s.get("tool")]
        condition = "success"
        for s in steps or []:
            result = s.get("result") if isinstance(s, dict) else None
            status = (str(result.get("status")) if isinstance(result, dict)
                      else "")
            if status and status not in ("ok", "success"):
                condition = status
                break
        else:
            for plan in faults or []:
                kind = str((plan or {}).get("fault") or "")
                if kind:
                    condition = kind
                    break
        return {"tool": tools_called[0] if tools_called else "unrelated",
                "tool_condition": condition,
                "origin": "realized"}

    def one(job: tuple) -> dict:
        prompt, rollout, meta, selection = job
        meta = dict(meta or {})
        assignment = meta.get("assignment") or meta.get("scenario_dimensions")
        faults = scaled(
            generator.fault_plans.get(prompt) or fault_plans.get(prompt),
            prompt)
        try:
            raw = runner(prompt)
        except Exception as exc:
            raw = {"steps": [], "final_text": f"<agent error: {public_llm_error(exc)}>"}
        raw = raw if isinstance(raw, dict) else {"steps": [], "final_text": str(raw)}
        if not assignment:
            assignment = _realized_dims(raw.get("steps") or [],
                                        _clean_faults(faults))
        t = {
            "scenario_id": meta.get("region_id") or f"probe_{hash(prompt) & 0xffffff:x}",
            "scenario_dimensions": assignment,
            "arm": meta.get("arm") or "unattributed",
            "prompt": prompt,
            "world_state": _row_world(assignment),
            "faults": _clean_faults(faults),
            "steps": raw.get("steps") or [],
            "final_text": str(raw.get("final_text", "")),
            "behavior_signature": None,
            "reward": None,
            "grader_reason": None,
            "reason": None,
            "rollout_index": rollout,
            "model_version": model_version_tag,
            "seed": meta.get("seed", seed),
            "semantic_cluster": None if not data.semantic else selection.get("cluster"),
            "semantic_novelty": None if not data.semantic else selection.get("novelty"),
            "parent_failure_id": meta.get("parent_failure_id") or meta.get("parent"),
            "selection_reason": selection.get("reason"),
        }
        # Cards drawn the steered way carry the mark onto the row, so
        # metadata's targeted/background split counts real draws.
        steering = meta.get("steering")
        if isinstance(steering, dict) and steering.get("origin"):
            t["steering"] = dict(steering)
        t.update(_row_conversation(meta, prompt, seed))
        t["behavior_signature"] = behavior_signature(t)
        return t

    signatures: set[str] = set()
    cells: set[str] = set()
    cell_counts: dict[str, int] = {}
    shape_counts: dict[str, int] = {}
    flat = 0
    round_index = 0
    empty_streak = 0
    writer_idle = 0
    restart_count = 0
    # Restarts scale with the job: a 10k-row budget cannot live on the
    # same retry allowance as a smoke run.
    max_restarts = max(MAX_NOVELTY_RESTARTS, int(cap or 0) // 100)
    failing_regions: list[dict] = []
    failing_rows: list[dict] = []
    used: set[str] = set()
    discarded: set[str] = set()
    used_situations: set[str] = set()
    region_counts: dict[str, int] = {}
    region_sigs: dict[str, set] = {}
    region_fails: dict[str, int] = {}
    region_novelty: dict[str, float] = {}
    behavior_gap_prompts: list[str] = []
    generated_pool: list[str] = []
    scenario_families: list[tuple[str, frozenset[str]]] = []

    def _mean_novelty(rows: list[dict]) -> float:
        vals = [float(r["novelty"]) for r in rows if r.get("novelty") is not None]
        return sum(vals) / len(vals) if vals else 1.0

    def _novelty_restart(round_id: int, info: dict, *, clear_avoid: bool) -> int:
        nonlocal restart_count
        if restart_count >= max_restarts or generator.model is None:
            return round_id
        restart_count += 1
        bump = round_id + restart_count * 997
        ctx_kwargs = {
            "novelty_parents": [] if not mutate_failures else failing_rows[-10:],
            "avoid": [] if clear_avoid else (info.get("concentrated") or []),
            "underexplored": info.get("sparse") or [],
            "behavior_gaps": list(behavior_gap_prompts[-8:]),
        }
        if hasattr(generator, "set_search_context"):
            missing = uncovered_action_shapes(
                action_shapes, induced_shape_keys,
                limit=int(search_plan["shape_limit"]))
            ctx_kwargs["action_targets"] = [
                shape_as_tags(s, tools) for s in missing]
            ctx_kwargs["arm_weights"] = search
            generator.set_search_context(**ctx_kwargs)
        return bump

    used_scenario_ids: set[str] = set()

    def _prompt_available(prompt: str) -> bool:
        if prompt in used or prompt in discarded:
            return False
        meta = generator.meta.get(prompt) or {}
        sk = _situation_key_from_meta(meta, prompt)
        if explore_only:
            rid = str(meta.get("region_id") or "")
            if rid and rid in used_scenario_ids:
                return False
            if sk and sk in used_situations:
                return False
        elif sk:
            if (n_situations_target
                    and sk not in used_situations
                    and len(used_situations) >= n_situations_target):
                return False
            if len(situation_prompts.get(sk, [])) >= n_req and prompt not in (
                    situation_prompts.get(sk) or []):
                return False
        return True

    situation_prompts: dict[str, list[str]] = {}
    prompt_rollouts: dict[str, int] = {}
    verify_queue: list[tuple] = []
    allocator_counts: dict[str, int] = {"explore": 0, "expand": 0, "verify": 0}

    def _available() -> list[str]:
        unused = [p for p in generated_pool if _prompt_available(p)]
        if seed_prompts:
            seed_set = set(seed_prompts)
            unused.sort(key=lambda p: 0 if p in seed_set else 1)
        return unused

    def _schedule_prompt(jobs: list, prompt: str, meta: dict, row: dict,
                         action: str) -> None:
        meta = dict(meta or {})
        meta["allocator"] = action
        if action == "verify":
            idx = prompt_rollouts.get(prompt, 0)
            if idx >= repeat_count:
                return
            jobs.append((prompt, idx, meta, row))
            prompt_rollouts[prompt] = idx + 1
            allocator_counts["verify"] = allocator_counts.get("verify", 0) + 1
            return
        if prompt in used:
            return
        k_now = repeat_count if k_immediate else 1
        start = prompt_rollouts.get(prompt, 0)
        for i in range(k_now):
            jobs.append((prompt, start + i, meta, row))
        prompt_rollouts[prompt] = start + k_now
        used.add(prompt)
        sk = _situation_key_from_meta(meta, prompt)
        if sk:
            situation_prompts.setdefault(sk, []).append(prompt)
            used_situations.add(sk)
        rid = str((meta or {}).get("region_id") or "")
        if rid:
            used_scenario_ids.add(rid)
        allocator_counts[action] = allocator_counts.get(action, 0) + k_now

    def _mark_scheduled(prompt: str, meta: dict) -> None:
        used.add(prompt)
        sk = _situation_key_from_meta(meta, prompt)
        if sk:
            used_situations.add(sk)
            if prompt not in situation_prompts.get(sk, []):
                situation_prompts.setdefault(sk, []).append(prompt)
        rid = str((meta or {}).get("region_id") or "")
        if rid:
            used_scenario_ids.add(rid)

    def _append_jobs(jobs: list, prompt: str, meta: dict, row: dict) -> None:
        action = "expand" if _situation_key_from_meta(meta, prompt) in used_situations else "explore"
        _schedule_prompt(jobs, prompt, meta, row, action)

    for text in seed_prompts:
        text = str(text).strip()
        if not text:
            continue
        generated_pool.append(text)
        generator.meta[text] = {
            "arm": "open_ended", "generator": "user", "seed": seed}
        generator.provenance[text] = "open_ended"
    producer_lock = threading.Lock()
    written = 0
    stream_started = False
    reported_rows = 0
    reported_at = started
    if out_path is not None:
        Path(str(out_path) + ".progress.json").write_text(json.dumps({
            "stage": "start", "rows": 0, "scenario_s": 0, "rollout_s": 0}))

    def _flush_output(stage: str) -> None:
        nonlocal written, stream_started, reported_rows, reported_at
        if out_path is None:
            return
        rows = data.trajectories
        now = time.monotonic()
        try:
            unused_n = sum(1 for p in generated_pool if _prompt_available(p))
        except NameError:
            unused_n = 0
        try:
            inflight_n = len(inflight)
        except NameError:
            inflight_n = 0
        try:
            writers_n = len(scenario_futs)
        except NameError:
            writers_n = 0
        Path(str(out_path) + ".progress.json").write_text(json.dumps({
            "stage": stage, "rows": len(rows),
            "scenario_s": round(data.scenario_generation_seconds, 3),
            "rollout_s": round(data.rollout_seconds, 3),
            "scene_s": round(data.scene_brief_seconds, 3),
            "first_row_s": round(data.first_row_seconds, 3),
            "total_s": round(now - started, 3),
            "unused": unused_n, "inflight": inflight_n, "writers": writers_n,
            "search": data.search or None,
        }, default=str))
        should_report = (stage != "rollout" or len(rows) >= cap
                         or len(rows) - reported_rows >= 25
                         or now - reported_at >= 5.0)
        if should_report:
            elapsed = now - started
            rate = len(rows) / elapsed if elapsed else 0.0
            print(f"simulate {stage} rows={len(rows)}/{cap} "
                  f"elapsed={elapsed:.1f}s rate={rate:.1f}/s "
                  f"unused={unused_n} inflight={inflight_n} writers={writers_n}",
                  flush=True)
            reported_rows, reported_at = len(rows), now
        if not rows:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not stream_started:
            with open(out_path, "w") as fh:
                for row in rows:
                    fh.write(json.dumps(_export_row(row), default=str) + "\n")
            stream_started = True
            written = len(rows)
            return
        with open(out_path, "a") as fh:
            for row in rows[written:]:
                fh.write(json.dumps(_export_row(row), default=str) + "\n")
        written = len(rows)

    walked_ids: set[str] = set()
    walked_lock = threading.Lock()

    seen_prints: set[str] = set()

    def _fingerprint(text: str) -> str:
        words = re.findall(r"[a-z0-9]+", str(text).lower())
        norm = []
        for word in words:
            if word in {"the", "a", "an", "to", "for", "and", "of", "on"}:
                continue
            if len(word) > 4 and word.endswith("s"):
                word = word[:-1]
            norm.append(word)
        return hashlib.sha256(" ".join(norm).encode()).hexdigest()[:16]

    def produce(round_id: int, cards: int | None = None,
                completions: int | None = None,
                out_tokens: int | None = None) -> tuple[list[str], dict, dict, dict]:
        n_cards = max(2, int(cards or scenarios_per_request))
        n_comp = max(1, min(8, int(
            completions if completions is not None else completions_per_request)))
        # ~130 tokens per card: long-prompt cards must be realizable.
        # The old 768 ceiling gave 12-card batches 64 tokens per message.
        token_cap = out_tokens if out_tokens is not None else max(
            256, min(2048, 130 * n_cards + 128))
        local = make_default_generator(
            tools, policy=policy, per_round=pool_size, seed=seed,
            dimensions=dimensions, simulator=simulator, kind=writer_kind,
            scenarios_per_request=n_cards,
            completions_per_request=n_comp,
            distinct_cards=distinct_cards, extra_cards=extra_cards,
            scene_brief=scene_box["brief"], out_tokens=token_cap,
            time_budget=time_budget, run_started=started,
            steering_weight=applied_steering, **advanced)
        src_model = getattr(generator, "model", None)
        loc_model = getattr(local, "model", None)
        if src_model is not None and loc_model is not None:
            if getattr(src_model, "regions", None):
                loc_model.regions = src_model.regions
                loc_model.region_index = {r["id"]: r for r in loc_model.regions}
            loc_model.walked_ids = walked_ids
            loc_model.walked_lock = walked_lock
            if hasattr(loc_model, "arm_weights"):
                loc_model.arm_weights = dict(
                    getattr(generator, "arm_weights", None) or search)
        if hasattr(local, "arm_weights"):
            local.arm_weights = dict(getattr(generator, "arm_weights", None) or search)
        if hasattr(local, "set_search_context"):
            local.set_search_context(
                novelty_parents=getattr(generator, "novelty_parents", []),
                avoid=getattr(generator, "avoid", []),
                underexplored=getattr(generator, "underexplored", []),
                behavior_gaps=getattr(generator, "behavior_gaps", []),
                action_targets=getattr(generator, "action_targets", []),
                arm_weights=getattr(generator, "arm_weights", None) or search)
        texts = list(local(None, round_id, include_model=True) or [])
        return (texts, dict(local.meta), dict(local.fault_plans),
                dict(local.last_errors))

    generation_started = time.monotonic()
    scenario_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, scenario_concurrency))
    scenario_futs: list = []
    next_producer_round = 0

    def _ingest_producer(fut: concurrent.futures.Future) -> int:
        try:
            more, metas, plans, errors = fut.result()
        except Exception as exc:
            msg = str(exc)
            if "Hosted Qwen" in msg:
                generator.last_errors["llm_guided"] = msg
            else:
                generator.last_errors["llm_guided"] = (
                    f"{type(exc).__name__}: {exc}")
            return 0
        generator.meta.update(metas)
        generator.fault_plans.update(plans)
        generator.last_errors.update(errors)
        added = 0
        for prompt in more:
            if not prompt or prompt in generated_pool:
                continue
            fp = _fingerprint(prompt)
            if explore_only and fp in seen_prints:
                continue
            meta = metas.get(prompt) or {}
            rid = str(meta.get("region_id") or "")
            if explore_only and rid and rid in used_scenario_ids:
                continue
            if explore_only:
                seen_prints.add(fp)
            generated_pool.append(prompt)
            added += 1
        return added

    if generator.model is None:
        texts = list(generator(None, 0, include_model=False) or [])
        generated_pool.extend(texts)
    else:
        # Tiny batches first so rollouts start ~5s.
        initial_writers = min(2, max(1, writer_flight))
        for i in range(initial_writers):
            scenario_futs.append(scenario_pool.submit(
                produce, i, 4, None, 320))
        next_producer_round = initial_writers
        done = [fut for fut in scenario_futs if fut.done()]
        scenario_futs[:] = [fut for fut in scenario_futs if not fut.done()]
        for fut in done:
            _ingest_producer(fut)

    def _launch_writers(n: int) -> None:
        """Queue another writer wave. New round_id draws new temp and tags."""
        nonlocal next_producer_round
        n = max(0, int(n))
        if n <= 0:
            return
        scenario_futs.extend(
            scenario_pool.submit(produce, next_producer_round + i)
            for i in range(n))
        next_producer_round += n

    data.scenario_generation_seconds = time.monotonic() - generation_started
    _flush_output("seed")
    generator.model_produced = any(
        (generator.meta.get(prompt) or {}).get("generator") == "model"
        for prompt in generated_pool)
    if generated_pool and any((generator.meta.get(p) or {}).get("arm")
                              for p in generated_pool):
        _note(data, "generated candidate with arm provenance")

    last_batch_size = 1
    def _select(candidates: list[str], *, batch_size: int,
                selection_seed: int, selection_round: int):
        tick = time.monotonic()
        try:
            return select_execution_batch(
                candidates, embedder=resolved_embedder, archive=archive,
                batch_size=batch_size, seed=selection_seed,
                round_index=selection_round)
        finally:
            data.embedding_selection_seconds += time.monotonic() - tick

    # One slot per requested rollout. Refill writers before the unused
    # pool hits zero: keep about two waves of prompts in the pipe.
    flight = max(1, int(concurrency))
    typical_n = min(completions_per_request, 3)
    writer_batch = max(1, scenarios_per_request * typical_n)
    writer_buffer = max(writer_batch * 2, min(flight * 2, 96))
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=flight)
    inflight: dict = {}
    inflight_started: dict = {}
    try:
        while len(data.trajectories) < cap:
            if time_budget is not None and (time.monotonic() - started) >= time_budget:
                data.stopped_because = "time_budget"
                for fut in [f for f in list(inflight) if f.done()]:
                    job = inflight.pop(fut)
                    try:
                        t = fut.result()
                    except Exception as exc:
                        t = {"steps": [], "final_text": f"<agent error: {public_llm_error(exc)}>",
                             "arm": (job[2] or {}).get("arm") or "unattributed",
                             "prompt": job[0], "reward": None}
                        t["behavior_signature"] = behavior_signature(t)
                    if not _usable_rollout(t):
                        _note(data, "rollout failure discarded")
                        continue
                    if not data.first_row_seconds:
                        data.first_row_seconds = time.monotonic() - started
                    data.trajectories.append(t)
                    data.row_seconds.append(time.monotonic() - started)
                    record_turns(turn_stats, t)
                    _flush_output("rollout")
                break
            generator.novelty_parents = [] if not mutate_failures else failing_rows[-10:]
            remaining = cap - len(data.trajectories) - len(inflight)
            take = min(max(0, flight - len(inflight)), max(0, remaining))

            unused = _available()
            still = [f for f in scenario_futs if not f.done()]
            done = [f for f in scenario_futs if f.done()]
            scenario_futs[:] = still
            for fut in done:
                _ingest_producer(fut)
                data.scenario_generation_seconds = (
                    time.monotonic() - generation_started)
            unused = _available()
            if unused or inflight:
                writer_idle = 0
            elif generated_pool:
                writer_idle += 1
            if generator.model is not None:
                slots = max(0, writer_flight - len(scenario_futs))
                pipeline = len(unused) + len(scenario_futs) * writer_batch
                need = min(max(0, remaining), writer_buffer)
                refill = 0
                if remaining > 0 and slots:
                    # Keep a prompt buffer. explore/unique only changes
                    # which situations are eligible, not whether we refill.
                    # Writer flight stays small so rollouts share the GPU.
                    low = len(unused) < max(16, min(64, flight // 4))
                    exhausted = (not unused and generated_pool
                                 and writer_idle >= 2
                                 and not unique_cards
                                 and time_budget is None)
                    if exhausted:
                        refill = 0
                    elif low or pipeline < need:
                        refill = min(slots, max(0, writer_flight - len(scenario_futs)))
                _launch_writers(refill)
            unused = _available()
            if generator.model is None and len(unused) < take * 2:
                for prompt in generator(None, round_index,
                                        include_model=False) or []:
                    if prompt and prompt not in generated_pool:
                        generated_pool.append(prompt)
                        generator.meta.update(
                            getattr(generator, "last_candidate_provenance", {}))

            unused = _available()
            if generator.model is None and not generator.model_produced:
                bounce = 0
                while len(unused) < take:
                    bounce += 1
                    for prompt in generator(
                            None, round_index + bounce * 17,
                            include_model=False) or []:
                        if prompt and prompt not in generated_pool:
                            generated_pool.append(prompt)
                    unused = _available()
                    if bounce >= 20:
                        break
            if (generator.model is not None and not unused and scenario_futs
                    and not inflight):
                wait_s = 0.5
                if time_budget is not None:
                    wait_s = max(0.2, min(wait_s, time_budget - (time.monotonic() - started)))
                done, _ = concurrent.futures.wait(
                    scenario_futs, timeout=wait_s,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                scenario_futs[:] = [f for f in scenario_futs if f not in done]
                for fut in done:
                    _ingest_producer(fut)
                unused = _available()
            fault_plans.update(generator.fault_plans)
            if generator.last_errors.get("llm_guided") and not generator.model_produced:
                if "generator_fallback" not in data.degraded:
                    data.degraded.append("generator_fallback")
            if unused and any(
                    (generator.meta.get(p) or {}).get("arm") for p in unused):
                _note(data, "generated candidate with arm provenance")

            selected: list[dict] = []
            info: dict = {}
            if unused:
                pick_n = take if explore_only else max(take * 3, take)
                selected, info = _select(
                    unused, batch_size=min(len(unused), pick_n),
                    selection_seed=seed + round_index,
                    selection_round=round_index)
            if (selected and _mean_novelty(selected) < NOVELTY_RESTART_FLOOR
                    and restart_count < max_restarts):
                bump = _novelty_restart(round_index, info, clear_avoid=True)
                unused = _available()
                if unused:
                    selected, info = _select(
                        unused, batch_size=max(1, take),
                        selection_seed=seed + bump, selection_round=bump)
            family_rejected: list[dict] = []
            if not data.semantic and not unique_cards:
                family_batch = list(scenario_families)
                selected, family_rejected = cap_scenario_families(
                    selected, family_batch, cap=max(16, n_req * 4))
                if generator.model is None:
                    fill_to = min(take, len(selected) + len(family_rejected))
                    backfill_n = max(0, fill_to - len(selected))
                    selected.extend(family_rejected[:backfill_n])
                    family_rejected = family_rejected[backfill_n:]
            for row in family_rejected:
                discarded.add(row["text"])
            if family_rejected:
                info["family_rejected"] = len(family_rejected)
            if info.get("mixed_spaces_refused") and "embedding_space_mismatch" not in data.degraded:
                data.degraded.append("embedding_space_mismatch")
            if selected:
                if data.semantic:
                    _note(data, "semantic embedding produced")
                _note(data, "selection reason / novelty")
                if archive.compatible(resolved_embedder):
                    archive.add(row["vector"] for row in selected)
                missing = uncovered_action_shapes(
                    action_shapes, induced_shape_keys,
                    limit=int(search_plan["shape_limit"]))
                if hasattr(generator, "set_search_context"):
                    generator.set_search_context(
                        novelty_parents=[] if not mutate_failures else failing_rows[-10:],
                        avoid=([row["text"] for row in family_rejected[:6]]
                               + list(info.get("concentrated") or []))[:8],
                        underexplored=info.get("sparse") or [],
                        behavior_gaps=list(behavior_gap_prompts[-8:]),
                        action_targets=[shape_as_tags(s, tools) for s in missing],
                        arm_weights=search)

            batch = []
            row_by_text = {row["text"]: row for row in selected}
            filled = {"explore": 0, "expand": 0, "verify": 0}
            slots = None
            if topo["mode"] == "adaptive" and not unique_cards and take:
                live_plan = adaptive_allocator(
                    time_budget, until_key,
                    elapsed=time.monotonic() - started)
                slots = allocator_slot_counts(take, live_plan)

            def _card_action(meta: dict, prompt: str) -> str:
                sk = _situation_key_from_meta(meta, prompt)
                rid = str((meta or {}).get("region_id") or "")
                if sk and sk in used_situations:
                    return "expand"
                if slots is not None and rid and rid in used_scenario_ids:
                    return "expand"
                return "explore"

            def _add_job(prompt: str, meta: dict, row: dict, action: str) -> int:
                sk = _situation_key_from_meta(meta, prompt)
                if action == "explore" and n_situations_target and (
                        sk not in used_situations
                        and len(used_situations) >= n_situations_target):
                    return 0
                before = len(batch)
                _schedule_prompt(batch, prompt, meta, row, action)
                added = len(batch) - before
                if added:
                    filled[action] = filled.get(action, 0) + added
                return added

            # Named seed openers are extra requests, not writer completions.
            # Schedule them before hash selection can bury them.
            if seed_prompts and take:
                for prompt in dict.fromkeys(seed_prompts):
                    if len(batch) >= take:
                        break
                    if prompt not in generated_pool or not _prompt_available(prompt):
                        continue
                    meta = dict(generator.meta.get(prompt) or {
                        "arm": "open_ended", "generator": "user", "seed": seed})
                    row = row_by_text.get(prompt) or {
                        "text": prompt, "cluster": None, "novelty": None,
                        "reason": "seed"}
                    action = _card_action(meta, prompt)
                    _add_job(prompt, meta, row, action)
            verify_cap = take if slots is None else slots["verify"]
            if (not k_immediate and repeat_count > 1
                    and verify_queue):
                still: list[tuple] = []
                for prompt, meta, row in verify_queue:
                    if (len(batch) >= take
                            or filled["verify"] >= verify_cap):
                        still.append((prompt, meta, row))
                        continue
                    if prompt_rollouts.get(prompt, 0) >= repeat_count:
                        continue
                    _add_job(prompt, dict(meta), row, "verify")
                    if prompt_rollouts.get(prompt, 0) < repeat_count:
                        still.append((prompt, meta, row))
                verify_queue[:] = still
            stratified = _stratified_prompts(
                [row["text"] for row in selected], take, generator,
                used_situations=used_situations)

            def _prompt_job(prompt: str):
                row = row_by_text.get(prompt)
                if not row:
                    return None
                meta = dict(generator.meta.get(prompt)
                            or generator.last_candidate_provenance.get(prompt) or {})
                meta.setdefault("arm", generator.provenance.get(prompt, "unattributed"))
                meta.setdefault("seed", seed)
                action = _card_action(meta, prompt)
                return prompt, meta, row, action

            jobs = [job for prompt in stratified
                    if (job := _prompt_job(prompt))]
            if slots is None:
                for prompt, meta, row, action in jobs:
                    if len(batch) >= take:
                        break
                    _add_job(prompt, meta, row, action)
                    scenario_families.append(scenario_family(prompt))
            else:
                scheduled: set[str] = set()
                expand_cands = [j for j in jobs if j[3] == "expand"]
                explore_cands = [j for j in jobs if j[3] == "explore"]

                def _fill(cands: list, action: str, limit: int) -> None:
                    for prompt, meta, row, _act in cands:
                        if len(batch) >= take or filled[action] >= limit:
                            return
                        if prompt in scheduled:
                            continue
                        if _add_job(prompt, meta, row, action):
                            scheduled.add(prompt)
                            scenario_families.append(scenario_family(prompt))

                _fill(expand_cands, "expand", slots["expand"])
                _fill(explore_cands, "explore", slots["explore"])
                for prompt, meta, row, action in explore_cands + expand_cands:
                    if len(batch) >= take:
                        break
                    if prompt in scheduled:
                        continue
                    if _add_job(prompt, meta, row, action):
                        scheduled.add(prompt)
                        scenario_families.append(scenario_family(prompt))

            # Offline templates only. Live writer steers via cards and
            # search context. At most one mutation and one gap per batch.
            mutation_slots = 0
            gap_slots = 0
            if generator.model is None:
                if mutate_failures:
                    mutation_slots = 1 if failing_rows else 0
                gap_slots = 1
            mutations_added = 0
            parents = [str(t.get("prompt") or "") for t in failing_rows if t.get("prompt")]
            if mutation_slots and parents:
                for name, prompt in mutate_pool(parents, rounds=1, limit=mutation_slots):
                    if not _prompt_available(prompt):
                        continue
                    meta = {"arm": "failure_mutation", "seed": seed,
                            "generator": name, "parent": parents[0][:80]}
                    _append_jobs(batch, prompt, meta, {
                        "cluster": None, "novelty": None,
                        "reason": "failure_mutation"})
                    mutations_added += 1
                failing_regions = []

            missing = uncovered_action_shapes(
                action_shapes, induced_shape_keys, limit=max(gap_slots, 1))
            if generator.model is None and gap_slots:
                for shape in missing[: max(0, gap_slots)]:
                    prompt = render_target_situation(shape, tools)
                    if not _prompt_available(prompt):
                        continue
                    meta = {"arm": "behavior_targeted", "seed": seed,
                            "generator": "actionspace", "action_key": shape.key()}
                    _append_jobs(batch, prompt, meta, {
                        "cluster": None, "novelty": None,
                        "reason": "behavior_targeted"})
                    behavior_gap_prompts.append(prompt)

            round_index += 1
            if not batch:
                if inflight or scenario_futs:
                    empty_streak = 0
                else:
                    empty_streak += 1
                    # Unique ingest may drop exact/near-dupe cards. That is
                    # not a run stop: the writer can invent another situation.
                    if (generator.model is not None and not generated_pool
                            and empty_streak >= 8):
                        err = (generator.last_errors.get("llm_guided")
                               or "empty response")
                        if "Hosted Qwen" in err:
                            raise RuntimeError(
                                err[err.find("Hosted Qwen"):]) from None
                        raise RuntimeError(
                            f"hosted Qwen produced no situations: {err}")
                    if generator.model is not None and remaining > 0:
                        if (writer_idle >= 4 and not unique_cards
                                and time_budget is None):
                            # Writer stalled on duplicates. Restart it
                            # with a rotated seed AND a rotating window
                            # of already-used asks as avoid pressure:
                            # reseeding alone reconverges to the same
                            # asks (measured: 26 vs the old ceiling 28).
                            if restart_count < max_restarts:
                                seen = sorted(used)
                                lo_i = (restart_count * 8) % max(1, len(seen))
                                window = (seen[lo_i:lo_i + 8]
                                          or seen[:8])
                                round_index = _novelty_restart(
                                    round_index,
                                    {"concentrated": window},
                                    clear_avoid=False)
                                writer_idle = 0
                                empty_streak = 0
                                _note(data, "writer restart after ask starvation")
                            else:
                                data.stopped_because = "ask_exhausted"
                                break
                        slots = max(0, writer_flight - len(scenario_futs))
                        _launch_writers(min(slots, writer_flight))
                    elif generator.model is None:
                        # The offline writer ran dry. Leaving the default
                        # stopped_because="budget" here claimed a 300-row
                        # budget was met by 106 rows.
                        data.stopped_because = "writer_exhausted"
                        break
                    continue
            else:
                empty_streak = 0
                batch = batch[: remaining]
                for prompt, _, meta, _ in batch:
                    plan = generator.fault_plans.get(prompt) or fault_plans.get(prompt)
                    assignment = (meta.get("assignment")
                                  or meta.get("scenario_dimensions") or {})
                    if plan or (isinstance(assignment, dict)
                                and assignment.get("world_state")):
                        _note(data, "world/fault instantiated")
                        break
                now = time.monotonic()
                for job in batch:
                    fut = pool.submit(one, job)
                    inflight[fut] = job
                    inflight_started[fut] = now
            rollout_started = time.monotonic()
            wait_s = 0.35
            if time_budget is not None:
                wait_s = max(0.1, min(wait_s, time_budget - (time.monotonic() - started)))
            if inflight:
                concurrent.futures.wait(
                    inflight, timeout=wait_s,
                    return_when=concurrent.futures.FIRST_COMPLETED)
            results, jobs_for = [], []
            now = time.monotonic()
            for fut in list(inflight):
                job = inflight[fut]
                if fut.done():
                    inflight.pop(fut, None)
                    inflight_started.pop(fut, None)
                    try:
                        results.append(fut.result())
                        jobs_for.append(job)
                    except Exception as exc:
                        t = {"steps": [], "final_text": f"<agent error: {public_llm_error(exc)}>",
                             "arm": (job[2] or {}).get("arm") or "unattributed",
                             "prompt": job[0], "reward": None}
                        t["behavior_signature"] = behavior_signature(t)
                        results.append(t)
                        jobs_for.append(job)
                elif now - inflight_started.get(fut, now) >= hung_slot_s:
                    # Leave the future in inflight so we do not launch a
                    # replacement on top of a still-running request.
                    continue
            paired = [(t, job) for t, job in zip(results, jobs_for)
                      if _usable_rollout(t)]
            if len(paired) != len(results):
                _note(data, "rollout failure discarded")
            results = [t for t, _ in paired]
            jobs_for = [job for _, job in paired]
            room = cap - len(data.trajectories)
            results, jobs_for = results[:room], jobs_for[:room]
            if len(data.trajectories) + len(results) >= cap:
                inflight.clear()
                inflight_started.clear()
            for t in results:
                _note(data, "model rollout")
                if t.get("steps"):
                    _note(data, "full tool trajectory")
                if t.get("behavior_signature"):
                    _note(data, "behavior signature")
                if not data.first_row_seconds:
                    data.first_row_seconds = time.monotonic() - started
                data.trajectories.append(t)
                data.row_seconds.append(time.monotonic() - started)
                record_turns(turn_stats, t)
                _note(data, "row stored")
                _flush_output("rollout")
            data.rollout_seconds += time.monotonic() - rollout_started

            executed: dict[str, int] = {}
            new_sig: dict[str, int] = {}
            new_cell: dict[str, int] = {}
            new_shape = 0
            fresh = 0
            for t in results:
                arm = t["arm"]
                executed[arm] = executed.get(arm, 0) + 1
                if t["behavior_signature"] not in signatures:
                    new_sig[arm] = new_sig.get(arm, 0) + 1
                    fresh += 1
                key = _cell_key(t)
                if key:
                    cells.add(key)
                assignment = t.get("scenario_dimensions")
                if isinstance(assignment, dict) and assignment:
                    grid = json.dumps(assignment, sort_keys=True, default=str)
                    prev = cell_counts.get(grid, 0)
                    cell_counts[grid] = prev + 1
                    if prev == 0:
                        new_cell[arm] = new_cell.get(arm, 0) + 1
            signatures.update(t["behavior_signature"] for t in results)
            yields = {
                arm: (new_sig.get(arm, 0) + new_cell.get(arm, 0) + 1.0)
                / (executed.get(arm, 0) + 1.0)
                for arm in search}
            search = _reallocate(search, yields)
            data.arm_weights = dict(search)
            if hasattr(generator, "reallocate"):
                generator.reallocate(yields)
            generator.arm_weights = dict(search)
            model_live = getattr(generator, "model", None)
            if model_live is not None and hasattr(model_live, "arm_weights"):
                model_live.arm_weights = dict(search)

            _record_shapes(results)
            for t in results:
                observed = shape_from_trajectory(t, tools)
                if observed is None:
                    continue
                sk = observed.key()
                prev = shape_counts.get(sk, 0)
                shape_counts[sk] = prev + 1
                if prev == 0:
                    new_shape += 1
            for t, job in zip(results, jobs_for):
                rid = t.get("scenario_id")
                if not rid:
                    continue
                region_counts[rid] = region_counts.get(rid, 0) + 1
                region_sigs.setdefault(rid, set()).add(t["behavior_signature"])
                if _mutation_worthy(t):
                    region_fails[rid] = region_fails.get(rid, 0) + 1
                sel = job[3] if len(job) > 3 else {}
                nov = sel.get("novelty") if isinstance(sel, dict) else None
                if nov is not None:
                    prev = region_novelty.get(rid, float(nov))
                    region_novelty[rid] = 0.5 * prev + 0.5 * float(nov)

            assign_id = {json.dumps(r["assignment"], sort_keys=True, default=str): r["id"]
                         for r in generator.regions}

            def novelty_fn(assignment):
                rid = assign_id.get(json.dumps(assignment, sort_keys=True, default=str), "")
                return float(region_novelty.get(rid, 0.5))

            def behavior_fn(assignment):
                rid = assign_id.get(json.dumps(assignment, sort_keys=True, default=str), "")
                count = region_counts.get(rid, 0)
                nsig = len(region_sigs.get(rid, ()))
                fails = region_fails.get(rid, 0)
                if count >= 3 and nsig <= 1:
                    gap = 1.0
                elif nsig >= 3:
                    gap = 0.2
                else:
                    gap = 0.5
                return min(1.0, 0.7 * gap + 0.3 * (fails / (count + 1.0)))

            axis_counts: dict[str, dict[str, int]] = {}
            for t in data.trajectories:
                dims = t.get("scenario_dimensions")
                if not isinstance(dims, dict):
                    continue
                for axis in ("tool_condition", "history", "world_state"):
                    value = str(dims.get(axis) or "")
                    if value:
                        slot = axis_counts.setdefault(axis, {})
                        slot[value] = slot.get(value, 0) + 1

            _apply_allocation(retarget_regions(
                generator.regions, tools, counts=region_counts,
                novelty=novelty_fn, behavior_value=behavior_fn,
                axis_counts=axis_counts, mode=topo["mode"]))
            model_obj = getattr(generator, "model", None)
            if model_obj is not None and getattr(model_obj, "regions", None):
                _apply_allocation(retarget_regions(
                    model_obj.regions, tools, counts=region_counts,
                    novelty=novelty_fn, behavior_value=behavior_fn,
                    axis_counts=axis_counts, mode=topo["mode"]))
            templates = getattr(generator, "templates", None)
            if templates is not None:
                templates.regions = generator.regions

            region_index = {r["id"]: r for r in generator.regions}
            failing_rows = [t for t in results if _mutation_worthy(t)]
            failing_regions = [region_index[t["scenario_id"]] for t in failing_rows
                               if t["scenario_id"] in region_index]
            for t, job in zip(results, jobs_for):
                prompt = str(t.get("prompt") or job[0] or "")
                if not prompt or prompt_rollouts.get(prompt, 0) >= repeat_count:
                    continue
                want_verify = _mutation_worthy(t)
                if (not want_verify and topo["mode"] == "adaptive"
                        and not k_immediate):
                    nsig = len(region_sigs.get(t.get("scenario_id"), ()))
                    live = adaptive_allocator(
                        time_budget, until_key,
                        elapsed=time.monotonic() - started)
                    # Short/messy clocks peek for different outcomes.
                    # Long/saturation only re-rolls when behavior already differs.
                    if nsig > 1 or live["explore"] < 0.55:
                        want_verify = True
                if not want_verify:
                    continue
                meta = job[2] if len(job) > 2 else {}
                sel = job[3] if len(job) > 3 else {}
                verify_queue.append((
                    prompt, dict(meta or {}),
                    sel if isinstance(sel, dict) else {}))
            missing = uncovered_action_shapes(
                action_shapes, induced_shape_keys,
                limit=int(search_plan["shape_limit"]))
            deficit = copies_remaining(cell_counts)
            if induced_shape_keys:
                deficit += copies_remaining(shape_counts)
            n_rows = max(1, len(data.trajectories))
            short_n = sum(1 for t in data.trajectories
                          if "short" in str(t.get("length") or ""))
            long_n = sum(1 for t in data.trajectories
                         if "long" in str(t.get("length") or ""))
            tones = {str(t.get("tone") or "") for t in data.trajectories}
            tiers = {str(t.get("tier") or "") for t in data.trajectories}
            tools_hit = {
                str((t.get("scenario_dimensions") or {}).get("tool") or "")
                for t in data.trajectories
                if isinstance(t.get("scenario_dimensions"), dict)}
            axis_gaps: list[str] = []
            if short_n / n_rows < 0.08:
                axis_gaps.append("You keep it brief.")
            if long_n / n_rows < 0.10:
                axis_gaps.append("You use more words.")
            for tone, line in (
                    ("frustrated", "You are frustrated."),
                    ("curt", "You are curt."),
                    ("polite", "You are being nice.")):
                if tone not in tones:
                    axis_gaps.append(line)
            if "adversarial" not in tiers:
                axis_gaps.append("You are pushing a constraint.")
            if "ambiguous" not in tiers:
                axis_gaps.append("You are confused.")
            for name in list(declared)[:8]:
                if name and name not in tools_hit:
                    intent = _intent_for_tool(name)
                    if intent:
                        axis_gaps.append(f"You want to {intent}.")
            axis_gaps = axis_gaps[:8]
            if hasattr(generator, "set_search_context"):
                generator.set_search_context(
                    novelty_parents=[] if not mutate_failures else failing_rows[-10:],
                    avoid=list(getattr(generator, "avoid", []) or []),
                    underexplored=(list(info.get("sparse") or [])
                                   + axis_gaps)[:8],
                    behavior_gaps=list(behavior_gap_prompts[-8:]),
                    action_targets=[shape_as_tags(s, tools) for s in missing],
                    arm_weights=search)
            data.search = {
                "cell_counts": dict(cell_counts),
                "shape_counts": dict(shape_counts),
                "region_counts": dict(region_counts),
                "arm_weights": dict(search),
                "avoid": list(getattr(generator, "avoid", []) or []),
                "underexplored": list(getattr(generator, "underexplored", []) or []),
                "axis_gaps": axis_gaps,
                "min_cell_copies": min(cell_counts.values()) if cell_counts else 0,
                "copy_deficit": deficit,
                "uncovered_shapes": len(missing),
                "plateau_batches": flat,
                "copies_needed": SATURATION_COPIES,
                "allocator": dict(allocator_counts),
                "mode": topo["mode"],
                "repeat_policy": topo["repeat_policy"],
                "n_req": n_req,
                "k": repeat_count,
            }
            space_rate = (fresh + sum(new_cell.values()) + new_shape) / max(1, len(results))
            last_batch_size = len(results)
            _record_coverage(
                data, data.trajectories, cells=cells,
                shape_keys=induced_shape_keys, arm_weights=search,
                batch_fresh_rate=space_rate,
                mean_batch_novelty=_mean_novelty(selected) if selected else None)
            flat = flat + 1 if space_rate < NEW_SIGNATURE_FLOOR else 0
            data.search["plateau_batches"] = flat
            if until_sat and space_saturated(
                    cell_counts, shape_counts,
                    expected_cells=planned_cell_keys):
                data.stopped_because = "saturation"
                inflight.clear()
                inflight_started.clear()
                break
    finally:
        _flush_output("stopped")
        scenario_pool.shutdown(wait=False)
        pool.shutdown(wait=False)
        if scene_thread is not None:
            left = 8.0
            if time_budget is not None:
                left = max(0.2, min(1.0, time_budget - (time.monotonic() - started)))
            scene_thread.join(timeout=left)
            data.scene_brief = scene_box["brief"]
            if not data.scene_brief and "scene_brief_unavailable" not in data.degraded:
                data.degraded.append("scene_brief_unavailable")
    data.declared_tools = declared
    # grader application happens once, at the end of simulate, through
    # run_judge: full judge contract (judge_status, lineage, no silent
    # zeros) instead of the legacy data.grade() write-back.
    if llm_grade:
        data.llm_grade(spec=llm_spec)
    if trace_rows:
        # Source traces shaped the grid; they must not shape the rows.
        # A generated near-copy of a held-out trace is training leakage.
        mined = mine_traces(trace_rows)
        # The optimizer's map rides every trace-fed run: the trace
        # history classifies into behavior regions (new / persistent /
        # improving / uncertain / passing) with budget shares. Recorded
        # for callers and the platform UI; allocation is disclosure
        # until the steering calibration sets how hard to apply it.
        state_record = dict(optimizer_state or behavior_state(trace_rows))
        state_record["applied"] = bool(optimizer_state
                                       and optimizer_state["regions"])
        state_record["allocation_gain"] = 4.0
        # Close the loop: the same region predicates that read the
        # traces re-measure the generated rows, so trace fail rate vs
        # generated fail rate is one comparable number per region.
        state_record["region_progress"] = region_progress(
            state_record, data.trajectories)
        data.search["behavior_state"] = state_record
        kept_rows, leak = drop_leaky_rows(data.trajectories, trace_rows,
                                          embedder=resolved_embedder)
        data.trajectories[:] = kept_rows
        data.search["trace_mining"] = {
            "n_traces": mined["n"],
            "n_flaw_rows": len(mined["flaw_rows"]),
            "faults": mined["faults"],
            "tools": {name: dict(slot)
                      for name, slot in mined["tools"].items()},
            # Observed result payloads reused as shape templates for
            # invented results; count everything (doctrine).
            "result_exemplars": {name: len(values) for name, values
                                 in trace_exemplars.items()},
            "focused_dimensions": {axis: list(values) for axis, values
                                   in (dimensions or {}).items()},
        }
        data.search["trace_leakage"] = {
            key: leak[key] for key in
            ("n", "n_sources", "threshold", "n_leaky", "n_dropped",
             "max_similarity")}
        # Dropped rows are not refilled (the loop has already ended), so a
        # 39%-short dataset must say why instead of standing next to
        # stopped_because="budget" as if the budget were met.
        if leak.get("n_dropped"):
            if "trace_leakage_dropped" not in data.degraded:
                data.degraded.append("trace_leakage_dropped")
    misses = int(turn_stats.get("followup_misses", 0) or 0)
    if misses:
        data.search["followup_misses"] = misses
        if (misses >= 8 and misses >= len(data.trajectories) // 4
                and "followups_starved" not in data.degraded):
            data.degraded.append("followups_starved")
    data.elapsed_seconds = time.monotonic() - started
    n = len(data.trajectories)
    data.rows_per_second = (n / data.elapsed_seconds) if data.elapsed_seconds else 0.0
    data.unique_prompts = len({t["prompt"] for t in data.trajectories})
    data.unique_behavior_signatures = len({t["behavior_signature"]
                                           for t in data.trajectories})
    if data.semantic and data.trajectories:
        duplicate = sum(1 for t in data.trajectories
                        if float(t.get("semantic_novelty") or 0.0) < 0.05)
        data.semantic_duplicate_rate = duplicate / len(data.trajectories)
    if data.coverage_curve:
        data.coverage_curve[-1]["stopped_because"] = data.stopped_because
    else:
        _record_coverage(
            data, data.trajectories, cells=cells, shape_keys=induced_shape_keys,
            arm_weights=data.arm_weights or search, stopped_because=data.stopped_because)
    data.coverage = build_coverage_summary(
        data.coverage_curve, budget=cap, stopped_because=data.stopped_because,
        flat_streak=flat, last_batch_size=last_batch_size,
        copy_deficit=int((data.search or {}).get("copy_deficit") or 0))
    data.coverage["min_cell_copies"] = (data.search or {}).get("min_cell_copies", 0)
    data.coverage["copies_needed"] = SATURATION_COPIES
    data.coverage["unique"] = unique_cards
    data.coverage["unique_situations"] = unique_cards
    data.coverage["repeats"] = repeat_count
    data.coverage["mode"] = topo["mode"]
    data.coverage["repeat_policy"] = topo["repeat_policy"]
    data.coverage["until"] = until_key
    data.coverage["n_situations"] = n_situations_target
    data.coverage["requests_per_situation"] = n_req
    data.coverage["rollouts_per_request"] = repeat_count
    data.allocator = dict(allocator_counts)
    if seed_amp_report:
        data.search["seed_amplification"] = seed_amp_report
    data.search["strategy"] = {
        "requested": strategy,
        "resolved": resolved_strategy,
        "broaden": resolved_strategy != "targeted",
        "reason": ("traces supplied -> aimed distribution"
                   if resolved_strategy == "trace" and strategy == "auto"
                   else "no traces -> broad exploration"
                   if strategy == "auto" else "explicit"),
        "steering_weight": {"requested": steering_weight,
                            "applied": applied_steering,
                            "source": ("override" if steering_weight
                                       is not None else "rule")},
    }
    if grader is not None:
        # Customer grader inside the loop (doctrine 8): score the
        # generated rows with the caller's own judge, write the verdicts
        # onto the trajectories, and disclose the split. Judge failures
        # mark rows unjudged; they never become silent zeros.
        from .judging import run_judge
        scored = run_judge(data.trajectories, grader, source="grade")
        for row, verdict in zip(data.trajectories, scored.rows):
            for key in ("reward", "reason", "judge_status", "judge_name",
                        "failure_class", "lineage"):
                if key in verdict:
                    row[key] = verdict[key]
        data.search["grader"] = {
            "judge": scored.judge_name,
            "scored": len(scored),
            "passes": len(scored.passes()),
            "failures": len(scored.failures()),
            "partials": len(scored.partials()),
            "unjudged": len(scored.unjudged()),
        }
    if grade and grader is None and not llm_grade and data.trajectories:
        # grade=True shipped a release as an accepted-and-ignored flag: the
        # advertised one-call path returned ungraded rows, and select_for_rl
        # then had nothing to select. It now applies the documented default —
        # the deterministic conduct grade, offline and free. The hosted or
        # LLM judges stay where they were: llm_grade=True, grader=, or
        # grade() afterwards.
        declared = {str((t.get("function") or t).get("name") or "")
                    for t in (data.profile.tools or []) if isinstance(t, dict)}
        for row in data.trajectories:
            if row.get("reward") is not None:
                continue
            verdict = conduct_grade(row, declared or None)
            row["reward"] = verdict.get("reward")
            if verdict.get("reason") is not None:
                row.setdefault("reason", verdict["reason"])
            row["label_source"] = "conduct"
    if out_path is not None and data.trajectories:
        data.save(str(out_path), meta=True)
    return data
