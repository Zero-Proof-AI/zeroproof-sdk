"""``simulate()``: the public entry point.

Knob resolution lives in :mod:`.run.config`, spec loading in
:mod:`.run.spec`, row helpers in :mod:`.run.rows`, and the engine (inputs,
build, scheduler loop, finish) in :mod:`.run.engine`. This module keeps
the signature, the docstring, and the names older code imported from here.
"""
from __future__ import annotations

from typing import Any, Callable

from .data import SimulationData
from .generate.scenarios import SEARCH_ARMS, reallocate_search_arms
from .run.config import (HUNG_SLOT_S as _HUNG_SLOT_S,  # noqa: F401
                         SATURATION_CAP as _SATURATION_CAP,
                         _merge_advanced, _parse_situations_arg,
                         resolve_run_config, resolve_topology, writer_spec_for)
from .run.engine import Run
from .run.rows import (_cell_key, _collect_finished,  # noqa: F401
                       _mutation_worthy, _prompt_arm, _record_coverage,
                       _row_conversation, _situation_key_from_meta,
                       _stratified_prompts, _usable_rollout)
from .run.spec import (_apply_spec, _backend_spec,  # noqa: F401
                       _kind_from_spec, _looks_like_spec_path,
                       _read_spec_file, _spec_extra_text, _spec_from_path)

_SEARCH_ARMS = dict(SEARCH_ARMS)
_reallocate = reallocate_search_arms

__all__ = ["simulate", "resolve_topology", "writer_spec_for"]


def simulate(agent: Any = None, *, spec: Any = None,
             tools: list[dict] | None = None, system_prompt: str | None = None,
             budget: int | None = 1000, time_budget: float | None = None,
             until: str = "compute", mode: str = "explore",
             situations: int | None = None,
             requests_per_situation: int | None = None,
             rollouts_per_request: int | None = None,
             unique_situations: bool = False,
             reproducible: bool = False,
             grade: bool = False, llm_grade: bool = False,
             traces: Any = None,
             grader: Any = None,
             strategy: str = "auto",
             seeds: list | None = None,
             scaffold: str | None = None,
             execute: Callable | None = None,
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

    ``execute=`` is the caller's world: a function ``(tool, arguments) ->
    result`` that answers every tool call for real, against their repo,
    database, or service. Without it the mock world answers, which fits
    record-shaped tools and not code. Scheduled faults still apply first.
    ``zeroproof.simulations.generate.agents.current_rollout`` is a
    thread-local set before each rollout with ``prompt``, ``rollout_index``
    and ``seed``, so ``execute`` can tell which run it is answering.

    ``scaffold=`` is generation-only guidance appended to the system prompt
    of the MODEL-BACKED teacher during rollout (and to the scene writer).
    It never enters ``profile.policy``, so exports and evals stay on the
    plain policy; it is ignored for user-supplied callable agents. Measured
    to help some agents and hurt others. Configure per agent, no default.

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

    A seeded run is reproducible bit-for-bit at ``concurrency: 1``.
    ``reproducible=True`` makes it so at any concurrency: each batch of
    rollouts finishes before the next is chosen, so results are consumed
    in submission order and every round sees the same state. Same seed,
    same concurrency, same agent gives the same rows; a slow rollout
    holds its batch, so uneven latency costs throughput. It needs the
    clock off, since a clock stop lands wherever the run happens to be.
    Without the flag, which rows land before the cap depends on thread
    timing.
    """
    cfg = resolve_run_config(
        agent, spec=spec, tools=tools, system_prompt=system_prompt,
        budget=budget, time_budget=time_budget, until=until, mode=mode,
        situations=situations, requests_per_situation=requests_per_situation,
        rollouts_per_request=rollouts_per_request,
        unique_situations=unique_situations, reproducible=reproducible,
        grade=grade,
        llm_grade=llm_grade, traces=traces, grader=grader,
        strategy=strategy, seeds=seeds, scaffold=scaffold, execute=execute,
        output=output, advanced=advanced, passed=passed)
    return Run(cfg).run()
