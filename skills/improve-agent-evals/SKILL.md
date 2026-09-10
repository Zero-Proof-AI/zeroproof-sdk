---
name: improve-agent-evals
description: >
  Use this skill when a coding agent (Claude Code, Cursor, or similar) has
  access to a customer's repo, that repo's own agent definition (tool
  schemas plus system prompt or policy), and wants to add eval coverage for
  that agent using the ZeroProof simulations SDK (`zeroproof_simulations`,
  import as `zps`). Trigger on requests like "add more evals for this
  agent", "find blind spots in our eval suite", "generate test cases from
  our failing traces", or "use zeroproof/zps to expand eval coverage". Do
  not trigger for training-data generation, RL data prep, or anything that
  asks to replace or rewrite existing evals.
metadata:
  version: "1.1.0"
---

# Overview

This skill adds eval coverage to an existing suite. It never designs a new
behavioral policy and never touches an existing eval file. Ground truth for
every API call here is the installed `zeroproof_simulations` package
(`README.md`, `zeroproof_simulations/__init__.py`); if a call in this skill
does not match the installed version, trust the installed source over this
doc.

Read `references/workflow.md` before generating anything. It holds the full
commands, code, and templates this file only summarizes. For coding agents or
other tools that touch real state, also read `references/world.md` before a
rollout; it defines what must answer the tool calls.

# Two lanes

**Lane 1: trace-guided repair.** The repo has labeled failures: production
traces or eval rollouts with `reward` 0/1, a `scores` marker, or an
observed tool fault (timeout, deny, malformed, not found). Feed them to
`simulate(traces=...)`. The SDK mines what actually broke and reshapes the
covering grid to spend budget near those tools, faults, and world states,
instead of the whole space.

**Lane 2: policy-guided discovery.** The goal is blind spots, not repair:
either there are no labeled failures yet, or the existing evals already
cover the known failures well. Simulate straight from the agent's tools and
system prompt (`simulate(tools=, system_prompt=)`) over the full covering
grid to surface situations the current evals never test.

**Combined.** Both exist: run trace-guided first to compare against what
mining finds, then run policy-guided (or `mode="adaptive", until="saturation"`)
for the rest of the grid. Report both lanes' output separately; never merge
their provenance.

**Decision rule.** Run `zps.trace_report(traces, tools=tools, policy=policy)`
if any trace-like input exists (production traces, eval rollouts, an OTLP
export). If `report["fails"] > 0` or `report["advisory_labels"] > 0`, that's
signal for lane 1. If `report["traces"] == 0`, or every trace is unlabeled
with no observed fault, run lane 2 only. If both hold, run combined.
Lane 1 needs about 20 traces before the aim is trustworthy: below that
the mined fault kinds and failing tools are a partial sample and the run
lands close to the cold-start grid. With fewer, run lane 1 anyway, say so
in the report, and treat lane 2 as the primary output.

# Inputs this skill looks for in the repo

- **Tool schemas**: OpenAI-style `{"type": "function", "function": {...}}`
  list, or an adapter object (LangChain executor, LangGraph graph, an
  OpenAI Agents SDK agent, a Claude Code binary). See the preflight section of `references/workflow.md`.
- **Policy**: the agent's system prompt, however the repo stores it
  (a `.py` string, a `.md`/`.txt` file, a config field).
- **Traces**: production logs, eval rollouts, or OTLP/GenAI spans. Any
  shape `zps.load_traces()` / `zps.rows_from_otel()` accepts.
- **Existing evals**: wherever the repo keeps them (pytest fixtures, a
  `evals/`/`tests/` JSONL or YAML set, a promptfoo config). Read-only.
- **Execution world**: for coding agents, the customer's existing disposable
  checkout or test harness, adapted to `execute(tool_name, arguments)`. Never
  use the SDK's invented file world to judge whether code is correct.

# Commands

Every `simulate(...)` call below needs a model endpoint: set
`OPENAI_API_KEY` (plus `OPENAI_BASE_URL` for any OpenAI-compatible host)
and pass `agent="openai:<model>"`, or set `VLLM_API_KEY` for hosted Qwen.
With `agent="openai:<model>"` that model also writes the situations and
the people, so the run bills the customer's endpoint for both sides. For
hosted Qwen set `ZP_CONTEXT_TOKENS=32768` so turn caps match its window.
The same key and base URL serve the optional LLM grader
(`llm_grade=True` or `zps.grade_llm(..., spec="openai:<model>")`).
Bring-your-own-model is OpenAI-compatible only: any endpoint that speaks
`/v1/chat/completions` with tool calls. Anthropic's native API is not
OpenAI-compatible and is not supported yet; put an OpenAI-compatible
gateway in front of it or use another provider. Preflight and trace
reports need no key.

```python
import zeroproof_simulations as zps

# BYOK: replace with the customer's OpenAI-compatible model.
# For ZeroProof-hosted Qwen, omit agent= from the simulate calls instead.
agent = "openai:gpt-4.1-mini"

# 0. Preflight, always first, always offline, no key needed
pre = zps.preflight(tools, policy)          # tool-schema and policy gaps
print(pre["cells"], pre["warnings"])        # grid size; read every warning
# zps.recommend() sizes a training set (thousands of rows); eval coverage
# runs use the explicit budgets below: 40 rows to smoke-test, then 200-400.

# Lane 1: trace-guided repair
report = zps.trace_report(traces, tools=tools, policy=policy)
print(zps.format_trace_report(report))
repair = zps.simulate(agent=agent, tools=tools, system_prompt=policy, traces=traces,
                      mode="explore", grade=True, budget=200, time_budget=300,
                      output="new_evals/trace_guided.jsonl")

# Lane 2: policy-guided discovery
discovery = zps.simulate(agent=agent, tools=tools, system_prompt=policy,
                         mode="adaptive",
                         until="saturation", grade=True, budget=400,
                         time_budget=300,
                         output="new_evals/policy_guided.jsonl")

for run in (repair, discovery):
    fatal = [d for d in run.degraded if d == "generator_fallback"]
    if run.stopped_because == "writer_exhausted":
        fatal.append("writer_exhausted")
    if fatal:
        raise RuntimeError(f"situations were not model-written, do not use: {fatal}")
```

Full code, BYOK setup, grading options, output shape, provenance stamping,
and the coverage comparison are in `references/workflow.md`.

# What it produces

A preflight report, a trace report, and a new JSONL file of generated
eval cases per lane, each row stamped with `scenario_id`, `source_lane`,
`generator_model`, `seed`, `run_id`, and, when the scenario carried them,
`world_state`, `faults`, `stance`. Graded rows carry `reward` and
`reason`. Then a coverage comparison against the existing evals (by tool
and by situation axis) and review instructions. Full templates:
`references/workflow.md`.

# Hard rules

- **Never overwrite or delete existing evals.** Write generated cases to a
  new file (`new_evals/...` or wherever the customer directs). Appending to
  the existing suite happens only as an explicit, reviewed integration
  step; never a silent overwrite.
- **Never fabricate a policy.** If no system prompt or written rules exist,
  say so in the preflight report and run lane 2 with an empty policy; do
  not invent rules to fill the gap.
- **Label every generated case with its provenance**: which lane produced
  it, the generator model, the seed, and the scenario fields the SDK
  already carries (`scenario_id`, `world_state`, `faults`, `stance`).
- **Templates or fallbacks are not evidence.** Two signals block a run:
  `generator_fallback` in `data.degraded`, and
  `data.stopped_because == "writer_exhausted"`. Both mean situations were
  not model-written. Stop and report those; never ship the rows.
  `data.search["writer_errors"]` holds the last writer error for the
  report. Every other note is advisory and the rows are still real:
  `semantic_embedding_unavailable` (novelty scored by hash; pass
  `embedder="openai:text-embedding-3-small"` with the same OpenAI key to
  make it semantic), `scene_brief_unavailable` (the brief writer ran past
  `time_budget`; raise it or ignore), `result_shapes_unavailable`,
  `trace_leakage_dropped`, `followups_starved` (the follow-up writer
  missed on a quarter or more of the rows, usually a slow or overloaded
  endpoint; the set skews single-turn, rerun with a longer `time_budget`
  or off-peak). Name advisory notes in the report, then continue.
- **No secrets in outputs.** Never write `OPENAI_API_KEY`, `VLLM_API_KEY`,
  `ZEROPROOF_API_KEY`/`ZEROPROOF_DELEGATED_CREDENTIAL`, or any `zp_*` key
  into the generated JSONL, the reports, or logs.
- **Coding agents require a real world.** Confirm that the installed
  `zps.simulate` accepts `execute=`, then connect it to an isolated checkout
  whose real reads, writes, commands, and tests answer the calls. If no safe
  execution harness exists, stop and report that prerequisite instead of
  generating coding evals against invented files.
- **Generated rows are candidates, not ground truth.** Review every failing
  row and at least 20 passing rows before adding anything to an eval suite.
