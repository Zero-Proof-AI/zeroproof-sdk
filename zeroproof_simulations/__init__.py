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

import logging as _logging

# Library convention: emit under "zeroproof_simulations", never configure
# the root logger. Callers opt in with logging.basicConfig() or a handler.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from .generate.adapters import (claude_code, connect, inspect, resolve, AgentProfile, resolve_system_prompt)
from .generate.agents import (hosted_model, local_model, missing_hosted_key,
                       parse_backend_spec, public_llm_error, touch_hosted,
                       default_max_turns)
from .generate.coverage import (NEW_SIGNATURE_FLOOR, SATURATION_COPIES,
                       build_coverage_summary, cell_key as _cell_key_from_row,
                       copies_remaining, coverage_point, space_saturated)
from .generate.diversity import (MAX_NOVELTY_RESTARTS, NOVELTY_RESTART_FLOOR,
                        adaptive_allocator, allocator_slot_counts,
                        behavior_tier, cap_scenario_families,
                        conversation_features, mix_items_by_tier,
                        new_turn_stats, record_turns, sample_cell_tags,
                        sampling_plan, scenario_family)
from .generate.embeddings import (EmbeddingArchive, is_semantic, resolve_embedder,
                         select_execution_batch)
from .generate.actionspace import (action_space_targets, induced_keys_from_trajectory,
                          render_target_situation, shape_as_tags,
                          shape_from_trajectory, uncovered_action_shapes)
from .generate.explore import mutate_pool
from .generate.generator import (ModelSimulator, amplify_seeds, assistant_kind,
                       make_default_generator,
                       write_result_shapes, write_scene_brief)
from .score.grading import _as_dict, behavior_signature, conduct_grade
from .ingest.platform import (datasets,
                       delete as delete_dataset,
                       issue_delegated_credential, pull, push_file,
                       push_rows, refresh_delegated_credential)
from .score.llm_judge import (MISSING_JUDGE_KEY, apply_llm_grade,
                       resolve_judge_key)
from .score.grade_llm import apply_grade_llm, require_judge_key
from .score.quality import (rank as rank_source, rank_rows, score_row,
                      summarize as summarize_quality)
from .world.sandbox import MockEnvironment
from .generate.scenarios import (DEFAULT_FAULT_RATE, SEARCH_ARMS, build_dimensions,
                        keep_fault_plan,
                        novelty,
                        open_ended_probes, policy_sections,
                        reallocate_search_arms, retarget_regions,
                        scenario_regions, _intent_for_tool)
from .export import (export_dataset, export_preference, export_training,
                     training_rows)
from .score.optimize import (filter_rl_rows, group_signal, optimize,
                       recommend, select_for_rl,
                       select_for_sft, trim_unanimous_groups)
from .ingest.otel import rows_from_otel
from .ingest.traces import (behavior_state, region_progress, opening_share,
                     dimensions_from_traces, drop_leaky_rows,
                     exemplar_result_shapes, flaw_rows,
                     format_trace_report, leakage_report,
                     load_traces, mine_result_exemplars, mine_traces,
                     simulate_from_traces,
                     split_pseudo_production, trace_report)
from .score.preflight import (FAILURE_CLASSES, classify_failure, dataset_report,
                        format_dataset_report, preflight)
from .score.judging import (ScoredData, build_preference_pairs, evaluate,
                      normalize_judge_result, run_judge)

from .data import (SimulationData, conversation, grade, grade_llm,
                   llm_grade, rank)
from .simulation import resolve_topology, simulate
from .schema import (SCHEMA_VERSION, Calibration, Dataset, Judgment, Marker,
                     Rollout, Task, from_row, to_row, validate)
from . import data, simulation, schema  # noqa: F401  (patch targets)

__all__ = [
           "SCHEMA_VERSION", "Task", "Rollout", "Judgment", "Marker",
           "Dataset", "Calibration", "from_row", "to_row", "validate",
          "simulate", "SimulationData", "conversation", "local_model",
           "datasets", "pull", "push_file", "push_rows", "delete_dataset",
           "issue_delegated_credential", "refresh_delegated_credential",
           "hosted_model", "MockEnvironment", "llm_grade", "grade",
           "grade_llm", "rank", "rank_rows", "score_row", "conduct_grade",
           "behavior_signature", "build_dimensions", "policy_sections",
           "scenario_regions", "open_ended_probes", "novelty", "connect",
           "inspect", "claude_code", "AgentProfile", "ModelSimulator",
           "write_scene_brief", "resolve_topology", "adaptive_allocator",
           "allocator_slot_counts", "optimize", "filter_rl_rows",
           "group_signal", "recommend", "select_for_sft", "select_for_rl",
           "trim_unanimous_groups", "training_rows", "export_training",
           "mine_traces", "dimensions_from_traces",
           "split_pseudo_production", "flaw_rows", "leakage_report",
           "drop_leaky_rows", "simulate_from_traces", "rows_from_otel",
           "load_traces", "trace_report", "preflight", "dataset_report",
           "classify_failure", "FAILURE_CLASSES", "run_judge", "evaluate",
           "ScoredData", "normalize_judge_result", "export_dataset",
           "export_preference", "build_preference_pairs",
           ]
