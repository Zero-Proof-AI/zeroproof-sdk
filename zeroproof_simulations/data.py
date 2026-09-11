"""SimulationData, conversation rebuild, row export, and the grade
entry points that operate on a finished run."""
from __future__ import annotations

import concurrent.futures
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .generate.adapters import AgentProfile
from .ingest.platform import push_rows
from .score.llm_judge import (MISSING_JUDGE_KEY, apply_llm_grade,
                       resolve_judge_key)
from .score.grade_llm import apply_grade_llm, require_judge_key
from .score.quality import rank as rank_source, rank_rows, summarize as summarize_quality
from .export import export_training
from .schema import SCHEMA_KEY, SCHEMA_VERSION, check, stamp
from .score.optimize import select_for_sft



def _note(data: "SimulationData", stage: str) -> None:
    if stage not in data.stages:
        data.stages.append(stage)


def conversation(row: dict) -> list[dict]:
    """User/agent turns from prompt + steps. Tool calls stay on the assistant turn."""
    messages: list[dict] = []
    opener = str(row.get("opener") or "")
    if opener:
        messages.append({"role": "assistant", "content": opener})
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
    """Fault modes only. world_state, stance, and texture ride the plan
    into the runner but are row fields, never faults keys."""
    if not isinstance(plan, dict) or not plan:
        return None
    out = {k: v for k, v in plan.items() if isinstance(v, dict)}
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
    # Topology axis: which side opened. Absent means user-opened.
    if row.get("opener"):
        out["opener"] = str(row["opener"])
        out["opening"] = str(row.get("opening") or "agent")
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
    # group identity: which rollout of the situation, under which weights.
    # an rl grader groups on disk, so these travel with the row.
    for key in ("rollout_index", "model_version"):
        if row.get(key) is not None:
            out[key] = row[key]
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
    # The wire row is born here for both the streamed file and save(), so
    # the stamp and the check live here and the two files always agree.
    stamp(out)
    check(out, where="export_row")
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
            from .score.judging import run_judge
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

    def select(self, *, target: int = 1000) -> list[dict]:
        """The rows recommended for training, not everything generated.

        Diverse pass-labeled demonstrations via ``select_for_sft``: one of
        each distinct way of being right before any repeats, junk and
        duplicate prompts dropped. Requires graded rows — grade in-loop
        (``grade=True``, ``grader=``) or afterwards with ``grade()``.
        The selection report lands in ``search["selection"]``.
        """
        from .score.optimize import _binary_label
        if not any(_binary_label(t) is not None
                   for t in self.trajectories if isinstance(t, dict)):
            raise RuntimeError(
                "select() needs binary-graded rows (reward 0 or 1) and "
                "none carry one. Pass grade=True or grader= to "
                "simulate(), or call grade() first.")
        selected, report = select_for_sft(self.trajectories, target=target)
        self.search["selection"] = report
        return selected

    def training_set(self, output: str | None = None, *,
                     target: int = 1000, validate: bool = True) -> dict:
        """Select the recommended rows and export them trainer-ready.

        ``select()`` picks diverse pass-labeled rows, ``export_training``
        writes them as chat JSONL with this run's system prompt and tools
        and the tool-call round-trip gate. Returns the export report with
        the selection report attached; pass ``output`` to write the file.
        Raw simulation rows are not the training artifact — this is.
        """
        selected = self.select(target=target)
        policy = str(self.profile.policy or "") if self.profile else ""
        tools = list(self.profile.tools) if self.profile else []
        report = export_training(selected, output, system_prompt=policy,
                                 tools=tools or None, validate=validate)
        report["selection"] = self.search.get("selection")
        return report

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
                    SCHEMA_KEY: SCHEMA_VERSION,
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
    from .score.quality import _load_jsonl, _write_jsonl
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




