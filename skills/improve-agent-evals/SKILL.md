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
  version: "1.0.0"
---

# Overview

This skill adds eval coverage to an existing suite. It never designs a new
behavioral policy and never touches an existing eval file. Ground truth for
every API call here is the installed `zeroproof_simulations` package
(`README.md`, `zeroproof_simulations/__init__.py`); if a call in this skill
does not match the installed version, trust the installed source over this
doc.

Read `references/workflow.md` before generating anything. It holds the full
commands, code, and templates this file only summarizes.

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

# Commands

Every `simulate(...)` call below needs a model endpoint: set
`OPENAI_API_KEY` (plus `OPENAI_BASE_URL` for any OpenAI-compatible host)
and pass `agent="openai:<model>"`, or set `VLLM_API_KEY` for hosted Qwen.
Preflight and trace reports need no key.

```python
import zeroproof_simulations as zps

# 0. Preflight, always first, always offline, no key needed
pre = zps.preflight(tools, policy)          # tool-schema and policy gaps
sizing = zps.recommend(tools, policy, mode="sft")   # sized simulate_kwargs

# Lane 1: trace-guided repair
report = zps.trace_report(traces, tools=tools, policy=policy)
print(zps.format_trace_report(report))
repair = zps.simulate(tools=tools, system_prompt=policy, traces=traces,
                      mode="explore", grade=True, budget=sizing["budget"],
                      output="new_evals/trace_guided.jsonl")

# Lane 2: policy-guided discovery
discovery = zps.simulate(tools=tools, system_prompt=policy, mode="adaptive",
                         until="saturation", grade=True,
                         output="new_evals/policy_guided.jsonl")

for run in (repair, discovery):
    fatal = [d for d in run.degraded if d in ("generator_fallback", "writer_exhausted")]
    if fatal:
        raise RuntimeError(f"situations were not model-written, do not use: {fatal}")
```

Full code, BYOK setup, grading options, output shape, provenance stamping,
and the coverage comparison are in `references/workflow.md`.

# What it produces

A preflight report, a proposed budget, and a new JSONL file of generated
eval cases per lane, each row stamped with `scenario_id`, `world_state`,
`faults`, `stance`, `source_lane`, `generator_model`, `seed`, plus a
coverage comparison against the existing evals (by tool and by situation
axis) and review instructions. Full templates: `references/workflow.md`.

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
- **Templates or fallbacks are not evidence.** If `data.degraded` is
  non-empty after a run, stop and report it (name the note, e.g.
  `result_shapes_unavailable`, `generator_fallback`, `trace_leakage_dropped`)
  instead of shipping those rows as coverage.
- **No secrets in outputs.** Never write `OPENAI_API_KEY`, `VLLM_API_KEY`,
  `ZEROPROOF_API_KEY`/`ZEROPROOF_DELEGATED_CREDENTIAL`, or any `zp_*` key
  into the generated JSONL, the reports, or logs.

