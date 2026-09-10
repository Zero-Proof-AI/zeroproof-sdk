"""The run engine behind ``simulate()``.

One :class:`Run` per call, driven by a :class:`RunConfig`. Phases, in
order:

1. **inputs**: load the spec, inspect the agent, draft tools for a
   prompt-only agent, amplify seeds, read traces into a grid emphasis.
2. **build**: the result object, the scene-brief thread, the rollout
   runner, the situation writer and its coverage grid.
3. **loop**: writer waves fill a prompt pool; a selector picks a diverse
   batch; rollouts run in a thread pool; every batch of results updates
   the search state (arm weights, region retargeting, the verify queue).
4. **finish**: leakage pruning, coverage summary, grading, save.

The loop state lives on the instance so each phase is a method with a
small local scope. Nothing here is public; ``simulate()`` is the door.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from ..data import SimulationData, _clean_faults, _export_row, _note, _row_world
from ..generate.actionspace import (action_space_targets,
                                    induced_keys_from_trajectory,
                                    render_target_situation, shape_as_tags,
                                    shape_from_trajectory,
                                    uncovered_action_shapes)
from ..generate.adapters import inspect, resolve
from ..generate.agents import (current_rollout, default_max_turns, hosted_model,
                               local_model, missing_hosted_key,
                               parse_backend_spec, public_llm_error,
                               touch_hosted)
from ..generate.coverage import (NEW_SIGNATURE_FLOOR, SATURATION_COPIES,
                                 build_coverage_summary, copies_remaining,
                                 space_saturated)
from ..generate.diversity import (MAX_NOVELTY_RESTARTS, NOVELTY_RESTART_FLOOR,
                                  adaptive_allocator, allocator_slot_counts,
                                  cap_scenario_families, new_turn_stats,
                                  record_turns, sampling_plan, scenario_family)
from ..generate.embeddings import (EmbeddingArchive, is_semantic,
                                   resolve_embedder, select_execution_batch)
from ..generate.explore import mutate_pool
from ..generate.generator import (amplify_seeds, draft_tools,
                                  make_default_generator, write_result_shapes,
                                  write_scene_brief)
from ..generate.scenarios import (SEARCH_ARMS, _intent_for_tool,
                                  keep_fault_plan, reallocate_search_arms,
                                  retarget_regions)
from ..ingest.traces import (behavior_state, dimensions_from_traces,
                             drop_leaky_rows, exemplar_result_shapes,
                             load_traces, mine_result_exemplars, mine_traces,
                             opening_share, region_progress)
from ..score.grading import behavior_signature, conduct_grade
from .config import RunConfig
from .rows import (_cell_key, _mutation_worthy, _record_coverage,
                   _row_conversation, _situation_key_from_meta,
                   _stratified_prompts, _usable_rollout)
from .spec import _apply_spec, _backend_spec, _kind_from_spec

log = logging.getLogger("zeroproof_simulations")

# How hard a hot trace region pulls cell weight toward itself.
ALLOC_GAIN = 4.0
_FINGERPRINT_STOPWORDS = {"the", "a", "an", "to", "for", "and", "of", "on"}


class Run:
    """One ``simulate()`` call: inputs, build, loop, finish."""

    def __init__(self, cfg: RunConfig) -> None:
        self.c = cfg
        self.started = time.monotonic()
        # Read by the progress flush before the loop exists.
        self.inflight: dict = {}
        self.scenario_futs: list = []
        self.generated_pool: list[str] = []

    # ------------------------------------------------------------ driver

    def run(self) -> SimulationData:
        c = self.c
        if c.out_path is not None:
            # Writes the progress file before inspection or backend
            # construction so callers see it immediately.
            c.out_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_progress({"stage": "setup", "rows": 0,
                                  "scenario_s": 0, "rollout_s": 0})
            log.info("simulate setup rows=0")
        self._resolve_inputs()
        self._resolve_traces()
        self._build_data()
        self._start_scene_thread()
        self._build_runner()
        self._build_generator()
        self._init_loop_state()
        self._seed_pool()
        if c.out_path is not None:
            self._write_progress({"stage": "start", "rows": 0,
                                  "scenario_s": 0, "rollout_s": 0})
        self._start_writers()
        try:
            self._loop()
        finally:
            self._shutdown()
        return self._finish()

    # ------------------------------------------------------------ inputs

    def _resolve_inputs(self) -> None:
        c = self.c
        tools, policy, spec_sits = _apply_spec(c.spec, c.tools, c.system_prompt, [])
        self.seed_prompts: list[str] = list(c.seed_prompts)
        self.seed_prompts.extend(str(s).strip() for s in spec_sits if str(s).strip())
        # simulate-from-seeds: a few example asks are a behavior request,
        # not the situation list. With an explicit situations=N target the
        # engine runs the coordinate search itself, amplifying the examples
        # across phrasing/stance/language axes to N distinct situations
        # before generation. Disclosed in search["seed_amplification"].
        self.seed_amp_report: dict | None = None
        self.profile = inspect(c.agent, tools=tools, system_prompt=policy)
        self.tools: list[dict] = list(self.profile.tools or [])
        self.policy: str = str(self.profile.policy or "")
        # Generation-only teacher guidance. profile.policy and export stay plain.
        self.gen_policy = (f"{self.policy}\n\n{c.scaffold_text}"
                           if c.scaffold_text else self.policy)
        self.writer_kind = _kind_from_spec(c.spec, self.policy)
        # May be replaced by the backend spec once the runner is built.
        self.simulator = c.simulator
        self.drafted_tools: list[str] = []
        self.tool_draft_failed = False
        if (not self.tools and self.policy and self.simulator is not False
                and (c.agent is None or isinstance(c.agent, str))):
            # A description with no tools gives the writer and the world no
            # domain; draft the tool surface the described agent would have.
            drafted = draft_tools(
                self.policy,
                backend_spec=self.simulator if isinstance(self.simulator, str) else None,
                kind=self.writer_kind)
            if drafted:
                self.tools = drafted
                self.profile.tools = drafted
                self.drafted_tools = [d["function"]["name"] for d in drafted]
            else:
                # The run goes on with no tools, which is a different
                # dataset from the one asked for. Say so instead of
                # letting a tool-free run pass for the described agent.
                self.tool_draft_failed = True
        if c.agent is None and not self.tools and not self.policy:
            raise ValueError(
                "simulate needs an agent, tools=, or a system prompt.")
        # Amplifies seed prompts only when seeds= is given; advanced["seed_prompts"]
        # stays literal. Offline runs (simulator=False) make no network calls.
        # Runs after inspect() so the writer hint carries the resolved policy.
        if (c.seeds and self.seed_prompts and c.n_situations_target
                and len(self.seed_prompts) < int(c.n_situations_target)
                and self.simulator is not False):
            given = len(self.seed_prompts)
            self.seed_prompts = amplify_seeds(
                self.seed_prompts, int(c.n_situations_target), policy=self.policy,
                backend_spec=self.simulator if isinstance(self.simulator, str) else None)
            self.seed_amp_report = {"given": given,
                                    "target": int(c.n_situations_target),
                                    "total": len(self.seed_prompts),
                                    "minted": len(self.seed_prompts) - given}

    def _resolve_traces(self) -> None:
        c = self.c
        self.trace_rows: list[dict] = []
        self.trace_focused = False
        self.optimizer_state: dict | None = None
        self.allocation_hits = {"n": 0}
        self.dimensions = c.dimensions
        if c.traces is not None:
            # same normalization as trace_report: a messages-only export
            # otherwise mines as zero tools and the grid is never aimed
            self.trace_rows = load_traces(c.traces)
            # The optimizer's memory feeds the run it aims: regions from
            # the whole trace history, budget shares from their lifecycle.
            # Cells whose coordinates intersect a hot region's expansion
            # recipe draw extra weight proportional to its share; cells
            # outside every recipe keep base weight - that is the
            # exploration reserve in action.
            self.optimizer_state = behavior_state(
                self.trace_rows, targeted=c.targeted_regions)
            if (self.trace_rows and self.dimensions is None
                    and c.resolved_strategy != "broad"):
                # trace: denser near observed behaviors, background kept.
                # targeted: drops tools the traces did not touch, narrowing the space.
                self.dimensions = dimensions_from_traces(
                    self.trace_rows, self.tools, self.policy,
                    broaden=c.resolved_strategy != "targeted")
                self.trace_focused = True
                # the grid flips 90% of cells to success for cold starts;
                # traces that show faults are asking for the fault cells
                if mine_traces(self.trace_rows)["faults"]:
                    c.advanced.setdefault("prefer_success", False)
        # The weight only applies over a trace-focused grid (its front-half
        # ordering is what the bias aims at) and only when nonzero; anything
        # else is exactly the unsteered draw and records no applied weight.
        self.applied_steering = (float(c.steering_weight)
                                 if c.steering_weight and self.trace_focused
                                 else None)

    def _allocation_boost(self, assignment: dict) -> float:
        factor = 1.0
        for region in self.optimizer_state["regions"]:
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
                factor += ALLOC_GAIN * share * match
        return factor

    def _apply_allocation(self, region_list) -> None:
        """Boost grid cells that sit inside a hot trace region. No-op cold."""
        if self.optimizer_state is None:
            return
        for region in region_list or []:
            boost = self._allocation_boost(region.get("assignment") or {})
            if boost != 1.0:
                self.allocation_hits["n"] += 1
                region["weight"] = round(
                    float(region.get("weight") or 0.0) * boost, 6)

    # ------------------------------------------------------------- build

    def _build_data(self) -> None:
        c = self.c
        data = SimulationData(profile=self.profile, arm_weights=dict(SEARCH_ARMS))
        data.scaffold_chars = len(c.scaffold_text)
        data.mode = c.topo["mode"]
        data.repeat_policy = c.topo["repeat_policy"]
        data.n_situations = c.n_situations_target
        data.requests_per_situation = c.n_req
        data.rollouts_per_request = c.repeat_count
        data.unique_situations = c.unique_cards
        _note(data, "agent ingestion")
        if self.tool_draft_failed:
            data.degraded.append("tool_draft_unavailable")
        self.data = data
        self.scene_box: dict[str, Any] = {"brief": ""}
        self.shape_box: dict[str, dict] = {}
        self.trace_exemplars: dict[str, list] = {}
        if self.trace_rows:
            # Result shapes mined from traces become the templates first;
            # the model-written pass below fills only tools the traces
            # did not show.
            self.trace_exemplars = mine_result_exemplars(self.trace_rows)
            self.shape_box.update(exemplar_result_shapes(self.trace_exemplars))

    def _start_scene_thread(self) -> None:
        self.scene_thread: threading.Thread | None = None
        use_model_writer = not (
            self.simulator is False
            or (callable(self.simulator) and not isinstance(self.simulator, str)))
        if not use_model_writer:
            return
        scene_spec = self.simulator if isinstance(self.simulator, str) else None
        self.scene_thread = threading.Thread(
            target=self._fill_scene, args=(scene_spec, time.monotonic()),
            daemon=True)
        self.scene_thread.start()

    def _fill_scene(self, scene_spec: str | None, scene_t0: float) -> None:
        data = self.data
        # Shapes first: every rollout benefits, and the writer can run
        # its first waves without the scene brief.
        shapes = write_result_shapes(self.tools, backend_spec=scene_spec)
        if shapes:
            # setdefault, not update: a template mined from a real
            # trace outranks a model-written guess for that tool.
            for shape_name, shape in shapes.items():
                self.shape_box.setdefault(shape_name, shape)
        elif "result_shapes_unavailable" not in data.degraded:
            data.degraded.append("result_shapes_unavailable")
        brief = write_scene_brief(
            self.tools, self.gen_policy, backend_spec=scene_spec,
            kind=self.writer_kind)
        self.scene_box["brief"] = brief
        data.scene_brief = brief
        data.scene_brief_seconds = time.monotonic() - scene_t0
        if not brief and "scene_brief_unavailable" not in data.degraded:
            data.degraded.append("scene_brief_unavailable")

    def _build_runner(self) -> None:
        c = self.c
        self.fault_plans: dict = {}
        kind = self.profile.transport
        self.turns = (default_max_turns(n_tools=len(self.tools))
                      if c.max_turns is None else max(1, int(c.max_turns)))
        self.turn_stats = new_turn_stats()
        # Resolve the opening-side topology axis: explicit value, or the
        # share observed in this run's traces ("auto"). Model-backed
        # runners only; callable agents cannot be asked for an opener.
        opening_req = c.opening_req
        if opening_req == "auto":
            self.opening_rate = opening_share(self.trace_rows) if self.trace_rows else 0.0
            self.opening_source = "traces"
        elif opening_req == "agent":
            self.opening_rate, self.opening_source = 1.0, "explicit"
        elif isinstance(opening_req, float):
            self.opening_rate, self.opening_source = opening_req, "explicit"
        else:
            self.opening_rate, self.opening_source = 0.0, (
                "explicit" if opening_req == "user" else "default")
        runner_kw: dict[str, Any] = {
            "fault_plans": self.fault_plans, "max_turns": self.turns,
            "avg_turns": float(c.avg_turns), "min_user_turns": c.min_user_turns,
            "turn_stats": self.turn_stats,
        }
        if c.temperature is not None:
            runner_kw["temperature"] = float(c.temperature)
        if c.execute is not None:
            runner_kw["execute"] = c.execute
        if c.backend:
            spec_backend = _backend_spec(c.backend)
            url, model_name = parse_backend_spec(spec_backend)
            self.runner = local_model(
                url, model_name, tools=self.tools, system=self.gen_policy,
                timeout=c.rollout_timeout, opening_rate=self.opening_rate,
                result_shapes=self.shape_box, **runner_kw)
            self.simulator = self.simulator if self.simulator is not None else spec_backend
        elif c.agent is None or kind not in {"callable", "backend_spec", "http"}:
            self.runner = hosted_model(
                self.tools, system=self.gen_policy,
                timeout=c.rollout_timeout, opening_rate=self.opening_rate,
                result_shapes=self.shape_box, **runner_kw)
        else:
            self.runner, kind = resolve(
                c.agent, tools=self.tools, policy=self.policy,
                opening_rate=self.opening_rate, result_shapes=self.shape_box,
                timeout=c.rollout_timeout, **runner_kw)
        self.kind = kind

    def _build_generator(self) -> None:
        c = self.c
        self.generator = make_default_generator(
            self.tools, policy=self.policy, per_round=c.pool_size, seed=c.seed,
            dimensions=self.dimensions, simulator=self.simulator,
            kind=self.writer_kind,
            scenarios_per_request=c.scenarios_per_request,
            completions_per_request=c.completions_per_request,
            distinct_cards=c.distinct_cards, extra_cards=c.extra_cards,
            scene_brief=self.scene_box["brief"], time_budget=c.time_budget,
            run_started=self.started, mode=c.topo["mode"],
            steering_weight=self.applied_steering, **c.advanced)
        gen = self.generator
        self._apply_allocation(getattr(gen, "regions", None))
        self.planned_cell_keys = {
            json.dumps(region["assignment"], sort_keys=True, default=str)
            for region in (getattr(gen, "regions", None) or [])
            if isinstance(region, dict)
            and isinstance(region.get("assignment"), dict)
            and region["assignment"]
        }
        # Search-arm weights, reallocated after every batch by yield.
        self.search: dict[str, float] = dict(SEARCH_ARMS)
        gen.arm_weights = dict(SEARCH_ARMS)
        model_obj = getattr(gen, "model", None)
        if model_obj is not None and hasattr(model_obj, "arm_weights"):
            model_obj.arm_weights = dict(SEARCH_ARMS)
        # The model arm keeps its own region list; applies the allocation
        # boost to it directly so round-1 cards and short runs get it too.
        self._apply_allocation(getattr(model_obj, "regions", None))
        model_backend = getattr(getattr(gen, "model", None), "backend_spec", None)
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
        self.fault_plans.update(gen.fault_plans)
        self.declared = {str((t.get("function", t) or {}).get("name", ""))
                         for t in self.tools or []} - {""}
        self.data.budget = int(c.cap)
        self.search_plan = sampling_plan(c.time_budget)
        self.action_shapes, _ = action_space_targets(
            self.tools, max_len=int(self.search_plan["max_shape_len"]),
            cap=int(self.search_plan["enum_cap"]))
        self.induced_shape_keys: set[str] = set()
        self.resolved_embedder = resolve_embedder(c.embedder)
        self.data.embedder_name = str(getattr(self.resolved_embedder, "name", "unknown"))
        self.data.semantic = is_semantic(self.resolved_embedder)
        if not self.data.semantic:
            self.data.degraded.append("semantic_embedding_unavailable")
        self.archive = EmbeddingArchive(self.data.embedder_name, self.data.semantic)

    # ---------------------------------------------------------- rollouts

    def _record_shapes(self, rows: list[dict]) -> None:
        known = {shape.key() for shape in self.action_shapes}
        for row in rows:
            self.induced_shape_keys.update(
                induced_keys_from_trajectory(row, self.action_shapes, self.tools))
            observed = shape_from_trajectory(row, self.tools)
            if observed is not None and observed.key() not in known:
                self.action_shapes.append(observed)
                known.add(observed.key())
            if isinstance(row, dict):
                row["tools_used"] = [
                    step.get("tool") for step in (row.get("steps") or [])
                    if isinstance(step, dict) and step.get("tool")]

    def _scaled(self, plan: dict | None, key: str = "") -> dict | None:
        """The fault plan this prompt keeps at ``fault_rate``, or nothing."""
        if not plan or self.c.fault_rate <= 0:
            return None
        if not keep_fault_plan(key, self.c.fault_rate, self.c.seed):
            return None
        return plan

    @staticmethod
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

    def _build_row(self, job: tuple) -> dict:
        """Run one rollout on a worker thread and shape it as a row."""
        c = self.c
        prompt, rollout, meta, selection = job
        meta = dict(meta or {})
        assignment = meta.get("assignment") or meta.get("scenario_dimensions")
        faults = self._scaled(
            self.generator.fault_plans.get(prompt) or self.fault_plans.get(prompt),
            prompt)
        # the caller's execute= world reads this to know which rollout it answers
        current_rollout.prompt = prompt
        current_rollout.rollout_index = rollout
        current_rollout.seed = meta.get("seed", c.seed)
        try:
            raw = self.runner(prompt)
        except Exception as exc:
            raw = {"steps": [], "final_text": f"<agent error: {public_llm_error(exc)}>"}
        raw = raw if isinstance(raw, dict) else {"steps": [], "final_text": str(raw)}
        if not assignment:
            assignment = self._realized_dims(raw.get("steps") or [],
                                             _clean_faults(faults))
        semantic = self.data.semantic
        t = {
            "scenario_id": meta.get("region_id") or "probe_" + hashlib.sha256(
                str(prompt).encode()).hexdigest()[:6],
            "scenario_dimensions": assignment,
            "arm": meta.get("arm") or "unattributed",
            "prompt": prompt,
            "world_state": _row_world(assignment),
            "faults": _clean_faults(faults),
            "steps": raw.get("steps") or [],
            "final_text": str(raw.get("final_text", "")),
            # topology axis: which side opened this conversation
            **({"opener": str(raw["opener"]), "opening": "agent"}
               if raw.get("opener") else {}),
            "behavior_signature": None,
            "reward": None,
            "grader_reason": None,
            "reason": None,
            "rollout_index": rollout,
            "model_version": c.model_version_tag,
            "seed": meta.get("seed", c.seed),
            "semantic_cluster": None if not semantic else selection.get("cluster"),
            "semantic_novelty": None if not semantic else selection.get("novelty"),
            "parent_failure_id": meta.get("parent_failure_id") or meta.get("parent"),
            "selection_reason": selection.get("reason"),
        }
        # Cards drawn the steered way carry the mark onto the row, so
        # metadata's targeted/background split counts real draws.
        steering = meta.get("steering")
        if isinstance(steering, dict) and steering.get("origin"):
            t["steering"] = dict(steering)
        t.update(_row_conversation(meta, prompt, c.seed))
        t["behavior_signature"] = behavior_signature(t)
        return t

    @staticmethod
    def _error_row(job: tuple, exc: Exception) -> dict:
        t = {"steps": [], "final_text": f"<agent error: {public_llm_error(exc)}>",
             "arm": (job[2] or {}).get("arm") or "unattributed",
             "prompt": job[0], "reward": None}
        t["behavior_signature"] = behavior_signature(t)
        return t

    # -------------------------------------------------------- loop state

    def _init_loop_state(self) -> None:
        c = self.c
        self.signatures: set[str] = set()
        self.cells: set[str] = set()
        self.cell_counts: dict[str, int] = {}
        self.shape_counts: dict[str, int] = {}
        self.flat = 0
        self.round_index = 0
        self.empty_streak = 0
        self.writer_idle = 0
        self.restart_count = 0
        # Starvation relief: when every situation slot is used but rows are
        # still owed because rollouts were discarded, lifts the situations
        # cap once so fresh situations fill the lost slots. Stays unlifted
        # when nothing was lost.
        self.cap_lifted = {"lifted": False, "lost": 0}
        # Restarts scale with the job: a 10k-row budget cannot live on the
        # same retry allowance as a smoke run.
        self.max_restarts = max(MAX_NOVELTY_RESTARTS, int(c.cap or 0) // 100)
        # A dormant switch: nothing sets it, so the region and fingerprint
        # dedup branches below never fire. Kept so the paths stay readable
        # next to the code that would flip it.
        self.explore_only = False
        self.failing_regions: list[dict] = []
        self.failing_rows: list[dict] = []
        self.used: set[str] = set()
        self.rerolls: dict[str, int] = {}
        self.discarded: set[str] = set()
        self.used_situations: set[str] = set()
        self.used_scenario_ids: set[str] = set()
        self.region_counts: dict[str, int] = {}
        self.region_sigs: dict[str, set] = {}
        self.region_fails: dict[str, int] = {}
        self.region_novelty: dict[str, float] = {}
        self.behavior_gap_prompts: list[str] = []
        self.generated_pool = []
        self.scenario_families: list[tuple[str, frozenset[str]]] = []
        self.situation_prompts: dict[str, list[str]] = {}
        self.prompt_rollouts: dict[str, int] = {}
        self.verify_queue: list[tuple] = []
        self.allocator_counts: dict[str, int] = {"explore": 0, "expand": 0, "verify": 0}
        # streaming output
        self.written = 0
        self.stream_started = False
        self.reported_rows = 0
        self.reported_at = self.started
        # writer waves
        self.walked_ids: set[str] = set()
        self.walked_lock = threading.Lock()
        self.seen_prints: set[str] = set()
        self.generation_started = self.started
        self.scenario_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, c.scenario_concurrency))
        self.scenario_futs = []
        self.next_producer_round = 0
        self.last_batch_size = 1
        # One slot per requested rollout. Refill writers before the unused
        # pool hits zero: keep about two waves of prompts in the pipe.
        self.flight = max(1, int(c.concurrency))
        typical_n = min(c.completions_per_request, 3)
        self.writer_batch = max(1, c.scenarios_per_request * typical_n)
        self.writer_buffer = max(self.writer_batch * 2, min(self.flight * 2, 96))
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.flight)
        self.inflight = {}
        self.inflight_started: dict = {}

    def _seed_pool(self) -> None:
        for text in self.seed_prompts:
            text = str(text).strip()
            if not text:
                continue
            self.generated_pool.append(text)
            self.generator.meta[text] = {
                "arm": "open_ended", "generator": "user", "seed": self.c.seed}
            self.generator.provenance[text] = "open_ended"

    @staticmethod
    def _mean_novelty(rows: list[dict]) -> float:
        vals = [float(r["novelty"]) for r in rows if r.get("novelty") is not None]
        return sum(vals) / len(vals) if vals else 1.0

    def _novelty_restart(self, round_id: int, info: dict, *,
                         clear_avoid: bool) -> int:
        gen = self.generator
        if self.restart_count >= self.max_restarts or gen.model is None:
            return round_id
        self.restart_count += 1
        bump = round_id + self.restart_count * 997
        ctx_kwargs = {
            "novelty_parents": ([] if not self.c.mutate_failures
                                else self.failing_rows[-10:]),
            "avoid": [] if clear_avoid else (info.get("concentrated") or []),
            "underexplored": info.get("sparse") or [],
            "behavior_gaps": list(self.behavior_gap_prompts[-8:]),
        }
        if hasattr(gen, "set_search_context"):
            missing = uncovered_action_shapes(
                self.action_shapes, self.induced_shape_keys,
                limit=int(self.search_plan["shape_limit"]))
            ctx_kwargs["action_targets"] = [
                shape_as_tags(s, self.tools) for s in missing]
            ctx_kwargs["arm_weights"] = self.search
            gen.set_search_context(**ctx_kwargs)
        return bump

    def _prompt_available(self, prompt: str) -> bool:
        c = self.c
        if prompt in self.used or prompt in self.discarded:
            return False
        meta = self.generator.meta.get(prompt) or {}
        sk = _situation_key_from_meta(meta, prompt)
        if self.explore_only:
            rid = str(meta.get("region_id") or "")
            if rid and rid in self.used_scenario_ids:
                return False
            if sk and sk in self.used_situations:
                return False
        elif sk:
            if (c.n_situations_target
                    and not self.cap_lifted["lifted"]
                    and sk not in self.used_situations
                    and len(self.used_situations) >= c.n_situations_target):
                return False
            if (len(self.situation_prompts.get(sk, [])) >= c.n_req
                    and prompt not in (self.situation_prompts.get(sk) or [])):
                return False
        return True

    def _available(self) -> list[str]:
        unused = [p for p in self.generated_pool if self._prompt_available(p)]
        if self.seed_prompts:
            seed_set = set(self.seed_prompts)
            unused.sort(key=lambda p: 0 if p in seed_set else 1)
        return unused

    def _schedule_prompt(self, jobs: list, prompt: str, meta: dict, row: dict,
                         action: str) -> None:
        c = self.c
        meta = dict(meta or {})
        meta["allocator"] = action
        if action == "verify":
            idx = self.prompt_rollouts.get(prompt, 0)
            if idx >= c.repeat_count:
                return
            jobs.append((prompt, idx, meta, row))
            self.prompt_rollouts[prompt] = idx + 1
            self.allocator_counts["verify"] = self.allocator_counts.get("verify", 0) + 1
            return
        if prompt in self.used:
            return
        k_now = c.repeat_count if c.k_immediate else 1
        start = self.prompt_rollouts.get(prompt, 0)
        for i in range(k_now):
            jobs.append((prompt, start + i, meta, row))
        self.prompt_rollouts[prompt] = start + k_now
        self.used.add(prompt)
        sk = _situation_key_from_meta(meta, prompt)
        if sk:
            self.situation_prompts.setdefault(sk, []).append(prompt)
            self.used_situations.add(sk)
        rid = str((meta or {}).get("region_id") or "")
        if rid:
            self.used_scenario_ids.add(rid)
        self.allocator_counts[action] = self.allocator_counts.get(action, 0) + k_now

    def _append_jobs(self, jobs: list, prompt: str, meta: dict, row: dict) -> None:
        action = ("expand" if _situation_key_from_meta(meta, prompt) in self.used_situations
                  else "explore")
        self._schedule_prompt(jobs, prompt, meta, row, action)

    # ------------------------------------------------------------ output

    def _write_progress(self, payload: dict) -> None:
        Path(str(self.c.out_path) + ".progress.json").write_text(
            json.dumps(payload, default=str))

    def _flush_output(self, stage: str) -> None:
        c = self.c
        if c.out_path is None:
            return
        data = self.data
        rows = data.trajectories
        now = time.monotonic()
        unused_n = sum(1 for p in self.generated_pool if self._prompt_available(p))
        inflight_n = len(self.inflight)
        writers_n = len(self.scenario_futs)
        self._write_progress({
            "stage": stage, "rows": len(rows),
            "scenario_s": round(data.scenario_generation_seconds, 3),
            "rollout_s": round(data.rollout_seconds, 3),
            "scene_s": round(data.scene_brief_seconds, 3),
            "first_row_s": round(data.first_row_seconds, 3),
            "total_s": round(now - self.started, 3),
            "unused": unused_n, "inflight": inflight_n, "writers": writers_n,
            "search": data.search or None,
        })
        should_report = (stage != "rollout" or len(rows) >= c.cap
                         or len(rows) - self.reported_rows >= 25
                         or now - self.reported_at >= 5.0)
        if should_report:
            elapsed = now - self.started
            rate = len(rows) / elapsed if elapsed else 0.0
            log.info("simulate %s rows=%d/%d elapsed=%.1fs rate=%.1f/s "
                     "unused=%d inflight=%d writers=%d",
                     stage, len(rows), c.cap, elapsed, rate,
                     unused_n, inflight_n, writers_n)
            self.reported_rows, self.reported_at = len(rows), now
        if not rows:
            return
        c.out_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.stream_started:
            with open(c.out_path, "w") as fh:
                for row in rows:
                    fh.write(json.dumps(_export_row(row), default=str) + "\n")
            self.stream_started = True
            self.written = len(rows)
            return
        with open(c.out_path, "a") as fh:
            for row in rows[self.written:]:
                fh.write(json.dumps(_export_row(row), default=str) + "\n")
        self.written = len(rows)

    # ----------------------------------------------------------- writers

    @staticmethod
    def _fingerprint(text: str) -> str:
        words = re.findall(r"[a-z0-9]+", str(text).lower())
        norm = []
        for word in words:
            if word in _FINGERPRINT_STOPWORDS:
                continue
            if len(word) > 4 and word.endswith("s"):
                word = word[:-1]
            norm.append(word)
        return hashlib.sha256(" ".join(norm).encode()).hexdigest()[:16]

    def _produce(self, round_id: int, cards: int | None = None,
                 completions: int | None = None,
                 out_tokens: int | None = None) -> tuple[list[str], dict, dict, dict]:
        """One writer wave on a worker thread: a fresh local generator
        that shares the run's regions, walked ids and search context."""
        c = self.c
        gen = self.generator
        n_cards = max(2, int(cards or c.scenarios_per_request))
        n_comp = max(1, min(8, int(
            completions if completions is not None else c.completions_per_request)))
        # ~130 tokens per card: long-prompt cards must be realizable.
        # The old 768 ceiling gave 12-card batches 64 tokens per message.
        token_cap = out_tokens if out_tokens is not None else max(
            256, min(2048, 130 * n_cards + 128))
        local = make_default_generator(
            self.tools, policy=self.policy, per_round=c.pool_size, seed=c.seed,
            dimensions=self.dimensions, simulator=self.simulator,
            kind=self.writer_kind,
            scenarios_per_request=n_cards,
            completions_per_request=n_comp,
            distinct_cards=c.distinct_cards, extra_cards=c.extra_cards,
            scene_brief=self.scene_box["brief"], out_tokens=token_cap,
            time_budget=c.time_budget, run_started=self.started,
            steering_weight=self.applied_steering, **c.advanced)
        src_model = getattr(gen, "model", None)
        loc_model = getattr(local, "model", None)
        if src_model is not None and loc_model is not None:
            if getattr(src_model, "regions", None):
                loc_model.regions = src_model.regions
                loc_model.region_index = {r["id"]: r for r in loc_model.regions}
            loc_model.walked_ids = self.walked_ids
            loc_model.walked_lock = self.walked_lock
            if hasattr(loc_model, "arm_weights"):
                loc_model.arm_weights = dict(
                    getattr(gen, "arm_weights", None) or self.search)
        if hasattr(local, "arm_weights"):
            local.arm_weights = dict(getattr(gen, "arm_weights", None) or self.search)
        if hasattr(local, "set_search_context"):
            local.set_search_context(
                novelty_parents=getattr(gen, "novelty_parents", []),
                avoid=getattr(gen, "avoid", []),
                underexplored=getattr(gen, "underexplored", []),
                behavior_gaps=getattr(gen, "behavior_gaps", []),
                action_targets=getattr(gen, "action_targets", []),
                arm_weights=getattr(gen, "arm_weights", None) or self.search)
        texts = list(local(None, round_id, include_model=True) or [])
        return (texts, dict(local.meta), dict(local.fault_plans),
                dict(local.last_errors))

    def _ingest_producer(self, fut: concurrent.futures.Future) -> int:
        gen = self.generator
        try:
            more, metas, plans, errors = fut.result()
        except Exception as exc:
            msg = str(exc)
            if "Hosted Qwen" in msg:
                gen.last_errors["llm_guided"] = msg
            else:
                gen.last_errors["llm_guided"] = (
                    f"{type(exc).__name__}: {exc}")
            return 0
        gen.meta.update(metas)
        gen.fault_plans.update(plans)
        gen.last_errors.update(errors)
        # sticky: the writer wrote at least once, so a later wave that
        # errors is a writer error, not a template fallback
        if any((m or {}).get("generator") == "model" for m in metas.values()):
            gen.model_produced = True
        added = 0
        for prompt in more:
            if not prompt or prompt in self.generated_pool:
                continue
            fp = self._fingerprint(prompt)
            if self.explore_only and fp in self.seen_prints:
                continue
            meta = metas.get(prompt) or {}
            rid = str(meta.get("region_id") or "")
            if self.explore_only and rid and rid in self.used_scenario_ids:
                continue
            if self.explore_only:
                self.seen_prints.add(fp)
            self.generated_pool.append(prompt)
            added += 1
        return added

    def _ingest_finished_writers(self) -> None:
        """Fold every finished writer wave into the pool."""
        done = [f for f in self.scenario_futs if f.done()]
        self.scenario_futs[:] = [f for f in self.scenario_futs if not f.done()]
        for fut in done:
            self._ingest_producer(fut)

    def _launch_writers(self, n: int) -> None:
        """Queue another writer wave. New round_id draws new temp and tags."""
        n = max(0, int(n))
        if n <= 0:
            return
        self.scenario_futs.extend(
            self.scenario_pool.submit(self._produce, self.next_producer_round + i)
            for i in range(n))
        self.next_producer_round += n

    def _start_writers(self) -> None:
        gen = self.generator
        data = self.data
        self.generation_started = time.monotonic()
        if gen.model is None:
            texts = list(gen(None, 0, include_model=False) or [])
            self.generated_pool.extend(texts)
        else:
            # Tiny batches first so rollouts start ~5s.
            initial_writers = min(2, max(1, self.c.writer_flight))
            for i in range(initial_writers):
                self.scenario_futs.append(self.scenario_pool.submit(
                    self._produce, i, 4, None, 320))
            self.next_producer_round = initial_writers
            self._ingest_finished_writers()
        data.scenario_generation_seconds = time.monotonic() - self.generation_started
        self._flush_output("seed")
        gen.model_produced = any(
            (gen.meta.get(prompt) or {}).get("generator") == "model"
            for prompt in self.generated_pool)
        if self.generated_pool and any((gen.meta.get(p) or {}).get("arm")
                                       for p in self.generated_pool):
            _note(data, "generated candidate with arm provenance")

    def _select(self, candidates: list[str], *, batch_size: int,
                selection_seed: int, selection_round: int):
        tick = time.monotonic()
        try:
            return select_execution_batch(
                candidates, embedder=self.resolved_embedder, archive=self.archive,
                batch_size=batch_size, seed=selection_seed,
                round_index=selection_round)
        finally:
            self.data.embedding_selection_seconds += time.monotonic() - tick

    # -------------------------------------------------------------- loop

    def _clock_left(self) -> float | None:
        if self.c.time_budget is None:
            return None
        return self.c.time_budget - (time.monotonic() - self.started)

    def _loop(self) -> None:
        c = self.c
        data = self.data
        gen = self.generator
        while len(data.trajectories) < c.cap:
            left = self._clock_left()
            if left is not None and left <= 0:
                data.stopped_because = "time_budget"
                break
            gen.novelty_parents = [] if not c.mutate_failures else self.failing_rows[-10:]
            remaining = c.cap - len(data.trajectories) - len(self.inflight)
            take = min(max(0, self.flight - len(self.inflight)), max(0, remaining))
            unused = self._refill_pool(remaining, take)
            selected, info = self._select_batch(unused, take)
            batch = self._build_batch(selected, take)
            self.round_index += 1
            if not batch:
                verdict = self._on_empty_batch(remaining)
                if verdict == "break":
                    break
                if verdict == "continue":
                    continue
            else:
                self.empty_streak = 0
                self._submit(batch[:remaining])
                if c.reproducible and self.inflight:
                    # Round-synchronous: the batch finishes (or hits the
                    # hung-slot limit) before anything is collected, so
                    # results are consumed in submission order and each
                    # round's selection seed sees the same state.
                    concurrent.futures.wait(list(self.inflight),
                                            timeout=c.hung_slot_s)
            results, jobs_for = self._collect()
            if self._update_search(results, jobs_for, info, selected):
                break

    def _settle_inflight(self) -> None:
        """The run is over. Cancel rollouts that never started, wait up to
        the stop grace for the ones running, keep what finishes while
        there is room under the cap, and report whatever is abandoned.

        Without this a clock stop returned with worker threads still
        calling the caller's agent and threw away every row they made.
        """
        c = self.c
        data = self.data
        for fut in list(self.inflight):
            if fut.cancel():
                self.inflight.pop(fut, None)
                self.inflight_started.pop(fut, None)
        if self.inflight and c.stop_grace_s > 0:
            concurrent.futures.wait(list(self.inflight), timeout=c.stop_grace_s)
        for fut in [f for f in list(self.inflight) if f.done()]:
            job = self.inflight.pop(fut)
            self.inflight_started.pop(fut, None)
            if len(data.trajectories) >= c.cap:
                continue
            try:
                t = fut.result()
            except Exception as exc:
                t = self._error_row(job, exc)
            if not _usable_rollout(t):
                self.cap_lifted["lost"] += 1
                _note(data, "rollout failure discarded")
                continue
            if not data.first_row_seconds:
                data.first_row_seconds = time.monotonic() - self.started
            data.trajectories.append(t)
            data.row_seconds.append(time.monotonic() - self.started)
            record_turns(self.turn_stats, t)
            self._flush_output("rollout")
        abandoned = len(self.inflight)
        if abandoned:
            # Threads cannot be killed; the pool is told to start nothing
            # new and these results are dropped when they arrive.
            data.search["abandoned_rollouts"] = abandoned
            if "rollouts_abandoned" not in data.degraded:
                data.degraded.append("rollouts_abandoned")
            self.inflight.clear()
            self.inflight_started.clear()

    def _refill_pool(self, remaining: int, take: int) -> list[str]:
        """Fold in finished writer waves, launch more if the pool runs
        low, top up offline templates. Returns the eligible prompts."""
        c = self.c
        data = self.data
        gen = self.generator
        if c.reproducible and self.scenario_futs:
            # Every launched writer wave lands before selection, in launch
            # order, so the pool does not depend on which wave returned first.
            concurrent.futures.wait(list(self.scenario_futs))
        self._ingest_finished_writers()
        data.scenario_generation_seconds = time.monotonic() - self.generation_started
        unused = self._available()
        if unused or self.inflight:
            self.writer_idle = 0
        elif self.generated_pool:
            self.writer_idle += 1
        if gen.model is not None:
            slots = max(0, c.writer_flight - len(self.scenario_futs))
            pipeline = len(unused) + len(self.scenario_futs) * self.writer_batch
            need = min(max(0, remaining), self.writer_buffer)
            refill = 0
            if remaining > 0 and slots:
                # Keeps a prompt buffer; explore/unique changes which
                # situations are eligible, not whether the buffer
                # refills. Writer flight stays small so rollouts
                # share the GPU.
                low = len(unused) < max(16, min(64, self.flight // 4))
                exhausted = (not unused and self.generated_pool
                             and self.writer_idle >= 2
                             and not c.unique_cards
                             and c.time_budget is None)
                if exhausted:
                    refill = 0
                elif low or pipeline < need:
                    refill = min(slots, max(0, c.writer_flight - len(self.scenario_futs)))
            self._launch_writers(refill)
        unused = self._available()
        if gen.model is None and len(unused) < take * 2:
            for prompt in gen(None, self.round_index, include_model=False) or []:
                if prompt and prompt not in self.generated_pool:
                    self.generated_pool.append(prompt)
                    gen.meta.update(
                        getattr(gen, "last_candidate_provenance", {}))
        unused = self._available()
        if gen.model is None and not gen.model_produced:
            bounce = 0
            while len(unused) < take:
                bounce += 1
                for prompt in gen(
                        None, self.round_index + bounce * 17,
                        include_model=False) or []:
                    if prompt and prompt not in self.generated_pool:
                        self.generated_pool.append(prompt)
                unused = self._available()
                if bounce >= 20:
                    break
        if (gen.model is not None and not unused and self.scenario_futs
                and not self.inflight):
            wait_s = 0.5
            left = self._clock_left()
            if left is not None:
                wait_s = max(0.2, min(wait_s, left))
            done, _ = concurrent.futures.wait(
                self.scenario_futs, timeout=wait_s,
                return_when=concurrent.futures.FIRST_COMPLETED)
            self.scenario_futs[:] = [f for f in self.scenario_futs if f not in done]
            for fut in done:
                self._ingest_producer(fut)
            unused = self._available()
        self.fault_plans.update(gen.fault_plans)
        if gen.last_errors.get("llm_guided") and not gen.model_produced:
            if "generator_fallback" not in data.degraded:
                data.degraded.append("generator_fallback")
        if unused and any(
                (gen.meta.get(p) or {}).get("arm") for p in unused):
            _note(data, "generated candidate with arm provenance")
        return unused

    def _select_batch(self, unused: list[str], take: int) -> tuple[list[dict], dict]:
        """Pick a diverse batch from the pool; restart the writer if the
        batch is stale; cap near-copy scenario families."""
        c = self.c
        data = self.data
        gen = self.generator
        selected: list[dict] = []
        info: dict = {}
        if unused:
            pick_n = take if self.explore_only else max(take * 3, take)
            selected, info = self._select(
                unused, batch_size=min(len(unused), pick_n),
                selection_seed=c.seed + self.round_index,
                selection_round=self.round_index)
        if (selected and self._mean_novelty(selected) < NOVELTY_RESTART_FLOOR
                and self.restart_count < self.max_restarts):
            bump = self._novelty_restart(self.round_index, info, clear_avoid=True)
            unused = self._available()
            if unused:
                selected, info = self._select(
                    unused, batch_size=max(1, take),
                    selection_seed=c.seed + bump, selection_round=bump)
        family_rejected: list[dict] = []
        if not data.semantic and not c.unique_cards:
            family_batch = list(self.scenario_families)
            selected, family_rejected = cap_scenario_families(
                selected, family_batch, cap=max(16, c.n_req * 4))
            if gen.model is None:
                fill_to = min(take, len(selected) + len(family_rejected))
                backfill_n = max(0, fill_to - len(selected))
                selected.extend(family_rejected[:backfill_n])
                family_rejected = family_rejected[backfill_n:]
        for row in family_rejected:
            self.discarded.add(row["text"])
        if family_rejected:
            info["family_rejected"] = len(family_rejected)
        if info.get("mixed_spaces_refused") and "embedding_space_mismatch" not in data.degraded:
            data.degraded.append("embedding_space_mismatch")
        if selected:
            if data.semantic:
                _note(data, "semantic embedding produced")
            _note(data, "selection reason / novelty")
            if self.archive.compatible(self.resolved_embedder):
                self.archive.add(row["vector"] for row in selected)
            missing = uncovered_action_shapes(
                self.action_shapes, self.induced_shape_keys,
                limit=int(self.search_plan["shape_limit"]))
            if hasattr(gen, "set_search_context"):
                gen.set_search_context(
                    novelty_parents=[] if not c.mutate_failures else self.failing_rows[-10:],
                    avoid=([row["text"] for row in family_rejected[:6]]
                           + list(info.get("concentrated") or []))[:8],
                    underexplored=info.get("sparse") or [],
                    behavior_gaps=list(self.behavior_gap_prompts[-8:]),
                    action_targets=[shape_as_tags(s, self.tools) for s in missing],
                    arm_weights=self.search)
        return selected, info

    def _card_action(self, meta: dict, prompt: str, slots: dict | None) -> str:
        sk = _situation_key_from_meta(meta, prompt)
        rid = str((meta or {}).get("region_id") or "")
        if sk and sk in self.used_situations:
            return "expand"
        if slots is not None and rid and rid in self.used_scenario_ids:
            return "expand"
        return "explore"

    def _add_job(self, batch: list, filled: dict, prompt: str, meta: dict,
                 row: dict, action: str) -> int:
        c = self.c
        sk = _situation_key_from_meta(meta, prompt)
        if (action == "explore" and c.n_situations_target
                and not self.cap_lifted["lifted"] and (
                sk not in self.used_situations
                and len(self.used_situations) >= c.n_situations_target)):
            return 0
        before = len(batch)
        self._schedule_prompt(batch, prompt, meta, row, action)
        added = len(batch) - before
        if added:
            filled[action] = filled.get(action, 0) + added
        return added

    def _build_batch(self, selected: list[dict], take: int) -> list:
        """Turn the selected prompts into rollout jobs: seeds first, then
        the verify queue, then stratified picks under the allocator's
        slot plan, then one failure mutation and one behavior gap for
        the offline writer."""
        c = self.c
        gen = self.generator
        batch: list = []
        row_by_text = {row["text"]: row for row in selected}
        filled = {"explore": 0, "expand": 0, "verify": 0}
        slots = None
        if c.topo["mode"] == "adaptive" and not c.unique_cards and take:
            live_plan = adaptive_allocator(
                c.time_budget, c.until_key,
                elapsed=time.monotonic() - self.started)
            slots = allocator_slot_counts(take, live_plan)

        # Named seed openers are extra requests, not writer completions.
        # Schedule them before hash selection can bury them.
        if self.seed_prompts and take:
            for prompt in dict.fromkeys(self.seed_prompts):
                if len(batch) >= take:
                    break
                if prompt not in self.generated_pool or not self._prompt_available(prompt):
                    continue
                meta = dict(gen.meta.get(prompt) or {
                    "arm": "open_ended", "generator": "user", "seed": c.seed})
                row = row_by_text.get(prompt) or {
                    "text": prompt, "cluster": None, "novelty": None,
                    "reason": "seed"}
                action = self._card_action(meta, prompt, slots)
                self._add_job(batch, filled, prompt, meta, row, action)
        verify_cap = take if slots is None else slots["verify"]
        if (not c.k_immediate and c.repeat_count > 1
                and self.verify_queue):
            still: list[tuple] = []
            for prompt, meta, row in self.verify_queue:
                if (len(batch) >= take
                        or filled["verify"] >= verify_cap):
                    still.append((prompt, meta, row))
                    continue
                if self.prompt_rollouts.get(prompt, 0) >= c.repeat_count:
                    continue
                self._add_job(batch, filled, prompt, dict(meta), row, "verify")
                if self.prompt_rollouts.get(prompt, 0) < c.repeat_count:
                    still.append((prompt, meta, row))
            self.verify_queue[:] = still
        stratified = _stratified_prompts(
            [row["text"] for row in selected], take, gen,
            used_situations=self.used_situations)

        jobs = []
        for prompt in stratified:
            row = row_by_text.get(prompt)
            if not row:
                continue
            meta = dict(gen.meta.get(prompt)
                        or gen.last_candidate_provenance.get(prompt) or {})
            meta.setdefault("arm", gen.provenance.get(prompt, "unattributed"))
            meta.setdefault("seed", c.seed)
            jobs.append((prompt, meta, row, self._card_action(meta, prompt, slots)))
        if slots is None:
            for prompt, meta, row, action in jobs:
                if len(batch) >= take:
                    break
                self._add_job(batch, filled, prompt, meta, row, action)
                self.scenario_families.append(scenario_family(prompt))
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
                    if self._add_job(batch, filled, prompt, meta, row, action):
                        scheduled.add(prompt)
                        self.scenario_families.append(scenario_family(prompt))

            _fill(expand_cands, "expand", slots["expand"])
            _fill(explore_cands, "explore", slots["explore"])
            for prompt, meta, row, action in explore_cands + expand_cands:
                if len(batch) >= take:
                    break
                if prompt in scheduled:
                    continue
                if self._add_job(batch, filled, prompt, meta, row, action):
                    scheduled.add(prompt)
                    self.scenario_families.append(scenario_family(prompt))

        # Offline templates only. Live writer steers via cards and
        # search context. At most one mutation and one gap per batch.
        mutation_slots = 0
        gap_slots = 0
        if gen.model is None:
            if c.mutate_failures:
                mutation_slots = 1 if self.failing_rows else 0
            gap_slots = 1
        parents = [str(t.get("prompt") or "") for t in self.failing_rows if t.get("prompt")]
        if mutation_slots and parents:
            for name, prompt in mutate_pool(parents, rounds=1, limit=mutation_slots):
                if not self._prompt_available(prompt):
                    continue
                meta = {"arm": "failure_mutation", "seed": c.seed,
                        "generator": name, "parent": parents[0][:80]}
                self._append_jobs(batch, prompt, meta, {
                    "cluster": None, "novelty": None,
                    "reason": "failure_mutation"})
            self.failing_regions = []

        missing = uncovered_action_shapes(
            self.action_shapes, self.induced_shape_keys, limit=max(gap_slots, 1))
        if gen.model is None and gap_slots:
            for shape in missing[: max(0, gap_slots)]:
                prompt = render_target_situation(shape, self.tools)
                if not self._prompt_available(prompt):
                    continue
                meta = {"arm": "behavior_targeted", "seed": c.seed,
                        "generator": "actionspace", "action_key": shape.key()}
                self._append_jobs(batch, prompt, meta, {
                    "cluster": None, "novelty": None,
                    "reason": "behavior_targeted"})
                self.behavior_gap_prompts.append(prompt)
        return batch

    def _on_empty_batch(self, remaining: int) -> str:
        """Nothing to schedule this round. Returns ``"proceed"`` when
        rollouts or writers are still in flight (collect them),
        ``"continue"`` after relaunching or lifting a cap, ``"break"``
        when the run is over."""
        c = self.c
        data = self.data
        gen = self.generator
        if self.inflight or self.scenario_futs:
            self.empty_streak = 0
            return "proceed"
        self.empty_streak += 1
        if (not self.cap_lifted["lifted"] and c.n_situations_target
                and remaining > 0 and self.cap_lifted["lost"] > 0
                and len(self.used_situations) >= c.n_situations_target):
            self.cap_lifted["lifted"] = True
            self.empty_streak = 0
            _note(data, "situation cap lifted to fill lost rollouts")
            return "continue"
        if (c.n_situations_target
                and len(self.used_situations) >= c.n_situations_target
                and not self.inflight and not self.scenario_futs
                and self.cap_lifted["lost"] == 0
                and all(self.prompt_rollouts.get(p, 0) >= c.repeat_count
                        for p in self.used)):
            # every situation the run was asked for exists and
            # has all its rollouts; a bigger budget cannot be
            # met, so stop and say so instead of spinning the
            # writer until the clock
            data.stopped_because = "situations_exhausted"
            return "break"
        # Unique ingest may drop exact/near-dupe cards. That is
        # not a run stop: the writer can invent another situation.
        if (gen.model is not None and not self.generated_pool
                and self.empty_streak >= 8):
            err = (gen.last_errors.get("llm_guided")
                   or "empty response")
            if "Hosted Qwen" in err:
                raise RuntimeError(
                    err[err.find("Hosted Qwen"):]) from None
            raise RuntimeError(
                f"hosted Qwen produced no situations: {err}")
        if gen.model is not None and remaining > 0:
            if self.writer_idle >= 4 and not c.unique_cards:
                # Writer stalled on duplicates. Restart it
                # with a rotated seed AND a rotating window
                # of already-used asks as avoid pressure:
                # reseeding alone reconverges to the same
                # asks (measured: 26 vs the old ceiling 28).
                if self.restart_count < self.max_restarts:
                    seen = sorted(self.used)
                    lo_i = (self.restart_count * 8) % max(1, len(seen))
                    window = (seen[lo_i:lo_i + 8]
                              or seen[:8])
                    self.round_index = self._novelty_restart(
                        self.round_index,
                        {"concentrated": window},
                        clear_avoid=False)
                    self.writer_idle = 0
                    self.empty_streak = 0
                    _note(data, "writer restart after ask starvation")
                else:
                    data.stopped_because = "ask_exhausted"
                    return "break"
            slots = max(0, c.writer_flight - len(self.scenario_futs))
            self._launch_writers(min(slots, c.writer_flight))
        elif gen.model is None:
            # The offline writer ran dry. Leaving the default
            # stopped_because="budget" here claimed a 300-row
            # budget was met by 106 rows.
            data.stopped_because = "writer_exhausted"
            return "break"
        return "continue"

    def _submit(self, batch: list) -> None:
        data = self.data
        for prompt, _, meta, _ in batch:
            plan = self.generator.fault_plans.get(prompt) or self.fault_plans.get(prompt)
            assignment = (meta.get("assignment")
                          or meta.get("scenario_dimensions") or {})
            if plan or (isinstance(assignment, dict)
                        and assignment.get("world_state")):
                _note(data, "world/fault instantiated")
                break
        now = time.monotonic()
        for job in batch:
            fut = self.pool.submit(self._build_row, job)
            self.inflight[fut] = job
            self.inflight_started[fut] = now

    def _collect(self) -> tuple[list[dict], list]:
        """Wait briefly, take every finished rollout, re-roll lost ones,
        and store the usable rows. Hung slots stay in flight."""
        c = self.c
        data = self.data
        rollout_started = time.monotonic()
        wait_s = 0.35
        left = self._clock_left()
        if left is not None:
            wait_s = max(0.1, min(wait_s, left))
        if self.inflight:
            concurrent.futures.wait(
                self.inflight, timeout=wait_s,
                return_when=concurrent.futures.FIRST_COMPLETED)
        results, jobs_for = [], []
        now = time.monotonic()
        for fut in list(self.inflight):
            job = self.inflight[fut]
            if fut.done():
                self.inflight.pop(fut, None)
                self.inflight_started.pop(fut, None)
                try:
                    results.append(fut.result())
                    jobs_for.append(job)
                except Exception as exc:
                    results.append(self._error_row(job, exc))
                    jobs_for.append(job)
            elif now - self.inflight_started.get(fut, now) >= c.hung_slot_s:
                # Leaves the future in inflight to avoid launching a
                # replacement on top of a still-running request.
                continue
        paired = [(t, job) for t, job in zip(results, jobs_for)
                  if _usable_rollout(t)]
        if len(paired) != len(results):
            # a lost rollout is re-rolled for the same prompt so a
            # repeat group keeps all k members; after the retry cap
            # it counts as lost and a fresh situation fills the slot
            now = time.monotonic()
            for t, job in zip(results, jobs_for):
                if _usable_rollout(t):
                    continue
                key = str(job[0])
                if self.rerolls.get(key, 0) < c.repeat_count:
                    self.rerolls[key] = self.rerolls.get(key, 0) + 1
                    fut = self.pool.submit(self._build_row, job)
                    self.inflight[fut] = job
                    self.inflight_started[fut] = now
                    _note(data, "rollout re-rolled")
                else:
                    self.cap_lifted["lost"] += 1
                    _note(data, "rollout failure discarded")
        results = [t for t, _ in paired]
        jobs_for = [job for _, job in paired]
        room = c.cap - len(data.trajectories)
        results, jobs_for = results[:room], jobs_for[:room]
        for t in results:
            _note(data, "model rollout")
            if t.get("steps"):
                _note(data, "full tool trajectory")
            if t.get("behavior_signature"):
                _note(data, "behavior signature")
            if not data.first_row_seconds:
                data.first_row_seconds = time.monotonic() - self.started
            data.trajectories.append(t)
            data.row_seconds.append(time.monotonic() - self.started)
            record_turns(self.turn_stats, t)
            _note(data, "row stored")
            self._flush_output("rollout")
        data.rollout_seconds += time.monotonic() - rollout_started
        return results, jobs_for

    def _update_search(self, results: list[dict], jobs_for: list,
                       info: dict, selected: list[dict]) -> bool:
        """Fold a batch of results into the search state. Returns True
        when the run reached saturation and should stop."""
        c = self.c
        data = self.data
        gen = self.generator
        executed: dict[str, int] = {}
        new_sig: dict[str, int] = {}
        new_cell: dict[str, int] = {}
        new_shape = 0
        fresh = 0
        for t in results:
            arm = t["arm"]
            executed[arm] = executed.get(arm, 0) + 1
            if t["behavior_signature"] not in self.signatures:
                new_sig[arm] = new_sig.get(arm, 0) + 1
                fresh += 1
            key = _cell_key(t)
            if key:
                self.cells.add(key)
            assignment = t.get("scenario_dimensions")
            if isinstance(assignment, dict) and assignment:
                grid = json.dumps(assignment, sort_keys=True, default=str)
                prev = self.cell_counts.get(grid, 0)
                self.cell_counts[grid] = prev + 1
                if prev == 0:
                    new_cell[arm] = new_cell.get(arm, 0) + 1
        self.signatures.update(t["behavior_signature"] for t in results)
        yields = {
            arm: (new_sig.get(arm, 0) + new_cell.get(arm, 0) + 1.0)
            / (executed.get(arm, 0) + 1.0)
            for arm in self.search}
        self.search = reallocate_search_arms(self.search, yields)
        data.arm_weights = dict(self.search)
        if hasattr(gen, "reallocate"):
            gen.reallocate(yields)
        gen.arm_weights = dict(self.search)
        model_live = getattr(gen, "model", None)
        if model_live is not None and hasattr(model_live, "arm_weights"):
            model_live.arm_weights = dict(self.search)

        self._record_shapes(results)
        for t in results:
            observed = shape_from_trajectory(t, self.tools)
            if observed is None:
                continue
            sk = observed.key()
            prev = self.shape_counts.get(sk, 0)
            self.shape_counts[sk] = prev + 1
            if prev == 0:
                new_shape += 1
        for t, job in zip(results, jobs_for):
            rid = t.get("scenario_id")
            if not rid:
                continue
            self.region_counts[rid] = self.region_counts.get(rid, 0) + 1
            self.region_sigs.setdefault(rid, set()).add(t["behavior_signature"])
            if _mutation_worthy(t):
                self.region_fails[rid] = self.region_fails.get(rid, 0) + 1
            sel = job[3] if len(job) > 3 else {}
            nov = sel.get("novelty") if isinstance(sel, dict) else None
            if nov is not None:
                prev = self.region_novelty.get(rid, float(nov))
                self.region_novelty[rid] = 0.5 * prev + 0.5 * float(nov)

        assign_id = {json.dumps(r["assignment"], sort_keys=True, default=str): r["id"]
                     for r in gen.regions}

        def novelty_fn(assignment):
            rid = assign_id.get(json.dumps(assignment, sort_keys=True, default=str), "")
            return float(self.region_novelty.get(rid, 0.5))

        def behavior_fn(assignment):
            rid = assign_id.get(json.dumps(assignment, sort_keys=True, default=str), "")
            count = self.region_counts.get(rid, 0)
            nsig = len(self.region_sigs.get(rid, ()))
            fails = self.region_fails.get(rid, 0)
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

        self._apply_allocation(retarget_regions(
            gen.regions, self.tools, counts=self.region_counts,
            novelty=novelty_fn, behavior_value=behavior_fn,
            axis_counts=axis_counts, mode=c.topo["mode"]))
        model_obj = getattr(gen, "model", None)
        if model_obj is not None and getattr(model_obj, "regions", None):
            self._apply_allocation(retarget_regions(
                model_obj.regions, self.tools, counts=self.region_counts,
                novelty=novelty_fn, behavior_value=behavior_fn,
                axis_counts=axis_counts, mode=c.topo["mode"]))
        templates = getattr(gen, "templates", None)
        if templates is not None:
            templates.regions = gen.regions

        region_index = {r["id"]: r for r in gen.regions}
        self.failing_rows = [t for t in results if _mutation_worthy(t)]
        self.failing_regions = [region_index[t["scenario_id"]] for t in self.failing_rows
                                if t["scenario_id"] in region_index]
        for t, job in zip(results, jobs_for):
            prompt = str(t.get("prompt") or job[0] or "")
            if not prompt or self.prompt_rollouts.get(prompt, 0) >= c.repeat_count:
                continue
            want_verify = _mutation_worthy(t)
            if (not want_verify and c.topo["mode"] == "adaptive"
                    and not c.k_immediate):
                nsig = len(self.region_sigs.get(t.get("scenario_id"), ()))
                live = adaptive_allocator(
                    c.time_budget, c.until_key,
                    elapsed=time.monotonic() - self.started)
                # Short/messy clocks peek for different outcomes.
                # Long/saturation only re-rolls when behavior already differs.
                if nsig > 1 or live["explore"] < 0.55:
                    want_verify = True
            if not want_verify:
                continue
            meta = job[2] if len(job) > 2 else {}
            sel = job[3] if len(job) > 3 else {}
            self.verify_queue.append((
                prompt, dict(meta or {}),
                sel if isinstance(sel, dict) else {}))
        missing = uncovered_action_shapes(
            self.action_shapes, self.induced_shape_keys,
            limit=int(self.search_plan["shape_limit"]))
        deficit = copies_remaining(self.cell_counts)
        if self.induced_shape_keys:
            deficit += copies_remaining(self.shape_counts)
        axis_gaps = self._axis_gaps()
        if hasattr(gen, "set_search_context"):
            gen.set_search_context(
                novelty_parents=[] if not c.mutate_failures else self.failing_rows[-10:],
                avoid=list(getattr(gen, "avoid", []) or []),
                underexplored=(list(info.get("sparse") or [])
                               + axis_gaps)[:8],
                behavior_gaps=list(self.behavior_gap_prompts[-8:]),
                action_targets=[shape_as_tags(s, self.tools) for s in missing],
                arm_weights=self.search)
        data.search = {
            "cell_counts": dict(self.cell_counts),
            "shape_counts": dict(self.shape_counts),
            "region_counts": dict(self.region_counts),
            "arm_weights": dict(self.search),
            "avoid": list(getattr(gen, "avoid", []) or []),
            "underexplored": list(getattr(gen, "underexplored", []) or []),
            "axis_gaps": axis_gaps,
            "min_cell_copies": min(self.cell_counts.values()) if self.cell_counts else 0,
            "copy_deficit": deficit,
            "uncovered_shapes": len(missing),
            "plateau_batches": self.flat,
            "copies_needed": SATURATION_COPIES,
            "allocator": dict(self.allocator_counts),
            "mode": c.topo["mode"],
            "repeat_policy": c.topo["repeat_policy"],
            "n_req": c.n_req,
            "k": c.repeat_count,
        }
        space_rate = (fresh + sum(new_cell.values()) + new_shape) / max(1, len(results))
        self.last_batch_size = len(results)
        _record_coverage(
            data, data.trajectories, cells=self.cells,
            shape_keys=self.induced_shape_keys, arm_weights=self.search,
            batch_fresh_rate=space_rate,
            mean_batch_novelty=self._mean_novelty(selected) if selected else None)
        self.flat = self.flat + 1 if space_rate < NEW_SIGNATURE_FLOOR else 0
        data.search["plateau_batches"] = self.flat
        if c.until_sat and space_saturated(
                self.cell_counts, self.shape_counts,
                expected_cells=self.planned_cell_keys):
            data.stopped_because = "saturation"
            return True
        return False

    def _axis_gaps(self) -> list[str]:
        """Prose nudges for the writer about axes the rows so far miss."""
        data = self.data
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
        for name in sorted(self.declared)[:8]:
            if name and name not in tools_hit:
                intent = _intent_for_tool(name)
                if intent:
                    axis_gaps.append(f"You want to {intent}.")
        return axis_gaps[:8]

    def _shutdown(self) -> None:
        data = self.data
        self._settle_inflight()
        self._flush_output("stopped")
        self.scenario_pool.shutdown(wait=False, cancel_futures=True)
        self.pool.shutdown(wait=False, cancel_futures=True)
        if self.scene_thread is not None:
            left = 8.0
            clock = self._clock_left()
            if clock is not None:
                left = max(0.2, min(1.0, clock))
            self.scene_thread.join(timeout=left)
            data.scene_brief = self.scene_box["brief"]
            if not data.scene_brief and "scene_brief_unavailable" not in data.degraded:
                data.degraded.append("scene_brief_unavailable")

    # ------------------------------------------------------------ finish

    def _finish(self) -> SimulationData:
        c = self.c
        data = self.data
        gen = self.generator
        data.declared_tools = self.declared
        # grader application happens once, at the end of simulate, through
        # run_judge: full judge contract (judge_status, lineage, no silent
        # zeros) instead of the legacy data.grade() write-back.
        if c.llm_grade:
            data.llm_grade(spec=c.llm_spec)
        if self.trace_rows:
            self._finish_traces()
        if gen.model is not None and gen.last_errors:
            data.search["writer_errors"] = dict(gen.last_errors)
        if ("generator_fallback" in data.degraded
                and getattr(gen, "model_produced", False)):
            # the note was set while the first waves were still in flight;
            # every prompt in the model path is model-written or nothing
            data.degraded.remove("generator_fallback")
        misses = int(self.turn_stats.get("followup_misses", 0) or 0)
        if misses:
            data.search["followup_misses"] = misses
            if (misses >= 8 and misses >= len(data.trajectories) // 4
                    and "followups_starved" not in data.degraded):
                data.degraded.append("followups_starved")
        data.elapsed_seconds = time.monotonic() - self.started
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
                data, data.trajectories, cells=self.cells,
                shape_keys=self.induced_shape_keys,
                arm_weights=data.arm_weights or self.search,
                stopped_because=data.stopped_because)
        data.coverage = build_coverage_summary(
            data.coverage_curve, budget=c.cap, stopped_because=data.stopped_because,
            flat_streak=self.flat, last_batch_size=self.last_batch_size,
            copy_deficit=int((data.search or {}).get("copy_deficit") or 0))
        data.coverage["min_cell_copies"] = (data.search or {}).get("min_cell_copies", 0)
        data.coverage["copies_needed"] = SATURATION_COPIES
        data.coverage["unique"] = c.unique_cards
        data.coverage["unique_situations"] = c.unique_cards
        data.coverage["repeats"] = c.repeat_count
        data.coverage["mode"] = c.topo["mode"]
        data.coverage["repeat_policy"] = c.topo["repeat_policy"]
        data.coverage["until"] = c.until_key
        data.coverage["n_situations"] = c.n_situations_target
        data.coverage["requests_per_situation"] = c.n_req
        data.coverage["rollouts_per_request"] = c.repeat_count
        data.allocator = dict(self.allocator_counts)
        if self.seed_amp_report:
            data.search["seed_amplification"] = self.seed_amp_report
        if self.drafted_tools:
            data.search["drafted_tools"] = self.drafted_tools
        data.search["strategy"] = {
            "requested": c.strategy,
            "resolved": c.resolved_strategy,
            "broaden": c.resolved_strategy != "targeted",
            "reason": ("traces supplied -> aimed distribution"
                       if c.resolved_strategy == "trace" and c.strategy == "auto"
                       else "no traces -> broad exploration"
                       if c.strategy == "auto" else "explicit"),
            "opening": {"requested": c.opening_req,
                        "rate": round(self.opening_rate, 4),
                        "source": self.opening_source},
            "steering_weight": {"requested": c.steering_weight,
                                "applied": self.applied_steering,
                                "source": ("override" if c.steering_weight
                                           is not None else "rule")},
        }
        self._finish_grading()
        if self.trace_rows and "behavior_state" in data.search:
            # Close the loop on the rows that ship: same region predicates
            # as the traces, measured after grading and leak-pruning.
            data.search["behavior_state"]["region_progress"] = region_progress(
                data.search["behavior_state"], data.trajectories)
        if c.out_path is not None and data.trajectories:
            data.save(str(c.out_path), meta=True)
        return data

    def _finish_traces(self) -> None:
        """Record what the traces did to the run and drop generated rows
        that near-copy a source trace."""
        data = self.data
        # Source traces shaped the grid; they must not shape the rows.
        # A generated near-copy of a held-out trace is training leakage.
        mined = mine_traces(self.trace_rows)
        # The optimizer's map rides every trace-fed run: the trace
        # history classifies into behavior regions (new / persistent /
        # improving / uncertain / passing) with budget shares. Recorded
        # for callers and the platform UI; allocation is disclosure
        # until the steering calibration sets how hard to apply it.
        state_record = dict(self.optimizer_state or behavior_state(self.trace_rows))
        # applied means a cell weight actually changed, not merely that
        # regions existed; the gain reads the one constant that steers.
        state_record["applied"] = self.allocation_hits["n"] > 0
        state_record["allocation_gain"] = ALLOC_GAIN
        # region_progress is attached at the very end of simulate(), so
        # it measures the rows that ship: graded, leak-pruned.
        data.search["behavior_state"] = state_record
        kept_rows, leak = drop_leaky_rows(data.trajectories, self.trace_rows,
                                          embedder=self.resolved_embedder)
        data.trajectories[:] = kept_rows
        data.search["trace_mining"] = {
            "n_traces": mined["n"],
            "n_flaw_rows": len(mined["flaw_rows"]),
            "faults": mined["faults"],
            "tools": {name: dict(slot)
                      for name, slot in mined["tools"].items()},
            # Observed result payloads reused as shape templates for
            # invented results.
            "result_exemplars": {name: len(values) for name, values
                                 in self.trace_exemplars.items()},
            "focused_dimensions": {axis: list(values) for axis, values
                                   in (self.dimensions or {}).items()},
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

    def _finish_grading(self) -> None:
        c = self.c
        data = self.data
        if c.grader is not None:
            # Scores generated rows with the caller's grader, writes the
            # verdicts onto the trajectories, and discloses the split.
            # Judge failures mark rows unjudged instead of silent zeros.
            from ..score.judging import run_judge
            scored = run_judge(data.trajectories, c.grader, source="grade")
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
        if c.grade and c.grader is None and not c.llm_grade and data.trajectories:
            # grade=True shipped a release as an accepted-and-ignored flag: the
            # advertised one-call path returned ungraded rows, and select_for_rl
            # then had nothing to select. It now applies the documented default:
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
                # the row template pre-seeds reason=None, so setdefault kept
                # every conduct reason off the export
                if verdict.get("reason") is not None and not row.get("reason"):
                    row["reason"] = verdict["reason"]
                row["label_source"] = "conduct"
