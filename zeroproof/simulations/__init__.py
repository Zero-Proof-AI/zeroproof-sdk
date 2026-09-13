"""Generate training conversations from an intent or an agent.

Input is a system prompt, tools, or both. The SDK sets a world from that
spec, writes human requests across a grid (simple, complex, vague, ordinary,
malicious), and rolls the agent. Grade 0/1 later. Optimize for post-training.

    import zeroproof.simulations as zps
    data = zps.simulate(system_prompt=policy)          # prompt-only agent
    data = zps.simulate(tools=tools, system_prompt=policy)
    data.save("rollout.jsonl")
    zps.grade("rollout.jsonl")                         # hosted Qwen 0/1
    zps.optimize("rollout.jsonl", output="train.jsonl")
"""

from __future__ import annotations

import logging as _logging

from . import data, schema, simulation
from .data import SimulationData, conversation, grade, grade_llm, llm_grade, rank
from .export import export_dataset, export_preference, export_training, training_rows
from .generate.adapters import AgentProfile, claude_code, connect, inspect
from .generate.agents import hosted_model, local_model
from .generate.diversity import adaptive_allocator, allocator_slot_counts
from .generate.generator import ModelSimulator, write_scene_brief
from .generate.scenarios import (
    build_dimensions,
    novelty,
    open_ended_probes,
    policy_sections,
    scenario_regions,
)
from .ingest.otel import rows_from_otel
from .ingest.platform import (
    datasets,
    issue_delegated_credential,
    pull,
    push_file,
    push_rows,
    refresh_delegated_credential,
)
from .ingest.platform import (
    delete as delete_dataset,
)
from .ingest.traces import (
    dimensions_from_traces,
    drop_leaky_rows,
    flaw_rows,
    format_trace_report,
    leakage_report,
    load_traces,
    mine_traces,
    simulate_from_traces,
    split_pseudo_production,
    trace_report,
)
from .schema import (
    SCHEMA_VERSION,
    Calibration,
    Dataset,
    Judgment,
    Marker,
    Rollout,
    Task,
    from_row,
    to_row,
    validate,
)
from .score.grading import behavior_signature, conduct_grade
from .score.judging import (
    ScoredData,
    build_preference_pairs,
    evaluate,
    normalize_judge_result,
    run_judge,
)
from .score.optimize import (
    filter_rl_rows,
    group_signal,
    optimize,
    recommend,
    select_for_rl,
    select_for_sft,
    trim_unanimous_groups,
)
from .score.preflight import FAILURE_CLASSES, classify_failure, dataset_report, preflight
from .score.quality import rank_rows, score_row
from .simulation import resolve_topology, simulate
from .world.sandbox import MockEnvironment

# Library convention: emit under "zeroproof.simulations", never configure
# the root logger. Callers opt in with logging.basicConfig() or a handler.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__all__ = [
    "FAILURE_CLASSES",
    "SCHEMA_VERSION",
    "AgentProfile",
    "Calibration",
    "Dataset",
    "Judgment",
    "Marker",
    "MockEnvironment",
    "ModelSimulator",
    "Rollout",
    "ScoredData",
    "SimulationData",
    "Task",
    "adaptive_allocator",
    "allocator_slot_counts",
    "behavior_signature",
    "build_dimensions",
    "build_preference_pairs",
    "classify_failure",
    "claude_code",
    "conduct_grade",
    "connect",
    "conversation",
    "dataset_report",
    "datasets",
    "delete_dataset",
    "dimensions_from_traces",
    "drop_leaky_rows",
    "evaluate",
    "export_dataset",
    "export_preference",
    "export_training",
    "filter_rl_rows",
    "flaw_rows",
    "format_trace_report",
    "from_row",
    "grade",
    "grade_llm",
    "group_signal",
    "hosted_model",
    "inspect",
    "issue_delegated_credential",
    "leakage_report",
    "llm_grade",
    "load_traces",
    "local_model",
    "mine_traces",
    "normalize_judge_result",
    "novelty",
    "open_ended_probes",
    "optimize",
    "policy_sections",
    "preflight",
    "pull",
    "push_file",
    "push_rows",
    "rank",
    "rank_rows",
    "recommend",
    "refresh_delegated_credential",
    "resolve_topology",
    "rows_from_otel",
    "run_judge",
    "scenario_regions",
    "score_row",
    "select_for_rl",
    "select_for_sft",
    "simulate",
    "simulate_from_traces",
    "split_pseudo_production",
    "to_row",
    "trace_report",
    "training_rows",
    "trim_unanimous_groups",
    "validate",
    "write_scene_brief",
]
