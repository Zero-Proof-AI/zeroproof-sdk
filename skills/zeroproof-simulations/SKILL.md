---
name: zeroproof-simulations
description: >
  Use the ZeroProof simulations SDK to generate agent trajectories from a
  policy, tools, an agent harness, seed tasks, or production traces; focus
  generation on failures, grade with the developer's own authority, inspect
  quality, and export usable datasets. Use when a coding agent is asked to
  simulate an agent, expand coverage, turn traces into new scenarios, create
  eval or post-training data, or run the simulate-grade-select loop.
metadata:
  version: "1.0.0"
---

# ZeroProof simulations

Operate ZeroProof for the developer. Discover the context already present in
their repository, choose the shortest valid input path, run a small smoke test,
and return auditable data plus a run report. Do not require every possible
input: policy, tools, traces, seeds, and a harness are complementary.

Terms in this skill: the **developer** is the person or team using ZeroProof;
the **coding agent** is the assistant reading this skill and operating the SDK;
the **target agent** is the agent being simulated and improved; the **simulated
user** is the person speaking to the target agent inside a trajectory.

Ground truth is the installed `zeroproof.simulations` package. Inspect its
public signatures when the installed version differs from these examples.

## Inputs and routes

Use what the developer has:

- `tools=` plus `system_prompt=`: cold-start simulation across the declared
  agent and policy space.
- `traces=` plus the agent definition: mine observed failures and concentrate
  search on their tools, faults, world states, and behaviors. About 20 graded
  or fault-bearing traces gives useful targeting; below 10, treat the result
  mostly as cold-start exploration and say so.
- `agent=`: connect an existing callable, supported framework agent, command,
  or OpenAI-compatible model endpoint.
- `spec=`: load a repository folder containing the agent specification.
- `seeds=`: preserve specific developer-provided tasks as starting asks. Use
  repeats when the developer needs multiple attempts on the same task.
- `execute=`: let the developer's real harness answer tool calls when
  correctness depends on real state.

If traces exist, normalize and inspect them before spending generation budget:

```python
traces = zps.load_traces(trace_source)
report = zps.trace_report(traces, tools=tools, policy=policy)
print(zps.format_trace_report(report))
```

Traces guide generation; they do not replace the authoritative tool schemas or
policy. Report foreign tools and missing context rather than silently inventing
an agent definition.

## Trace-first workflow

When traces are the starting point, follow this sequence:

1. Locate the exact trace dataset. If OTEL is already sending to ZeroProof,
   list the account's trace datasets and pull the intended `datasetId`; do not
   guess from a local file or silently combine different agents/days:

   ```python
   import os
   import zeroproof
   import zeroproof.simulations as zps

   key = os.environ["ZEROPROOF_API_KEY"]
   inventory = zeroproof.list_traces(key)["traces"]
   matches = [t for t in inventory if t["name"] == requested_dataset]
   if len(matches) != 1:
       raise RuntimeError(f"select one trace dataset explicitly: {matches}")
   selected = matches[0]
   traces = zps.pull(selected["datasetId"], api_key=key)
   ```

2. Load local inputs with `zps.load_traces(...)`. JSONL paths and common message,
   rollout, tool-trace, and platform-export shapes are accepted. Use
   `zps.rows_from_otel(...)` first for raw OTLP/GenAI spans. Store the result as
   `normalized_traces` and use that same list for reporting, simulation, and
   leakage checks.
3. Print `zps.trace_report(...)` before generation. Record rows dropped during
   normalization, graded/pass/fail/ungraded counts, observed tools, call-level
   faults, world states, distinct behaviors, foreign tools, and proposed grid
   emphasis.
4. Find the authoritative tool schemas and policy in the target agent's harness
   when possible. Traces show only tools and arguments that happened to run;
   they cannot reveal unobserved capabilities or recover a missing policy.
5. If schemas are unavailable, a mechanically inferred harness may be used as
   a reviewed draft, never silently as ground truth. Ask the developer to
   confirm required arguments and missing tools.

   ```python
   from zeroproof.simulations.ingest.traces import infer_harness

   draft = infer_harness(normalized_traces)
   tools, policy = draft["tools"], draft["policy"]  # policy is intentionally empty
   ```
6. Run a 40-row trace-guided smoke test with the same agent definition and
   `traces=normalized_traces`.
7. Inspect `data.search["trace_mining"]` and
   `data.search["behavior_state"]`. Confirm the input count is correct, expected
   failing tools/conditions were mined, focused dimensions changed, and the
   behavior-state allocation reports `applied=True`. `data.metadata.targeted_rows`
   is meaningful only when an explicit `steering_weight` was used; zero is not
   evidence that ordinary trace-guided generation failed.
8. Compare the trace-guided run with a same-budget policy-only run. Report the
   share aimed at observed failures, not merely total row count.
9. Check `zps.leakage_report(data.trajectories, normalized_traces)` and keep
   source-trace copies out of generated data.
10. Grade the new rows with the developer-provided judge. Do not copy rewards from
   source traces onto newly generated situations.
11. Save the returned `ScoredData`; grading by judge creates scored copies and
    does not rewrite the original simulation file.
12. Feed newly graded failures into a later `simulate(traces=...)` round only
    after preserving `model_version`, judge status, reason, and lineage.

A reward of `0` or an observed tool fault supplies direct repair signal. A
reward of `1` supplies contrast. Unlabeled traces still describe the observed
surface but do not establish correctness. Advisory model labels may steer
aiming, but report them separately from developer-owned grades.

## Preflight

Find the exact system prompt the agent receives and obtain tool schemas from
the implementation or harness rather than hand-transcribing them.

```python
import zeroproof.simulations as zps

pre = zps.preflight(tools, policy)
print(pre)
```

Read every warning. Preflight and trace inspection are offline and need no
model key.

## Choose the search

- Use `mode="explore"` for distinct situations, normally one row per
  situation. This is the default and the best first run.
- Use `mode="sft"` when several human phrasings of each situation are useful.
- Use `mode="rl"` with `rollouts_per_request=` when repeated attempts on the
  exact same ask are required.
- Use `mode="adaptive", until="saturation"` for a broader run that mixes new
  situations, phrasings, and repeats until coverage plateaus.

Trace-guided repair and policy-guided discovery are both ordinary `simulate`
runs. The difference is whether `traces=` is supplied:

```python
# BYOK; omit agent= to use ZeroProof-hosted Qwen with VLLM_API_KEY.
agent = "openai:gpt-4.1-mini"

repair = zps.simulate(
    agent=agent, tools=tools, system_prompt=policy, traces=traces,
    mode="explore", budget=40, time_budget=150, grade=False,
    output="simulations/trace_guided.jsonl",
)

discovery = zps.simulate(
    agent=agent, tools=tools, system_prompt=policy,
    mode="explore", budget=40, time_budget=150, grade=False,
    output="simulations/policy_guided.jsonl",
)
```

Start with 40 rows. Close-read the smoke test before increasing to 200-400.
`budget` caps rows; `time_budget` caps wall time. `situations` controls distinct
worlds, `requests_per_situation` controls phrasings, and
`rollouts_per_request` controls independent repeats. Set them separately.

The default novelty embedder is deterministic hashing. Use an OpenAI embedding
spec only when semantic novelty materially matters and the developer authorizes
that endpoint and cost.

## Model and world

For BYOK, set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`, then use
`agent="openai:<model>"`. The endpoint must implement OpenAI-compatible chat
completions with tool calls. The model writes the situations and plays the
target agent, so both consume its endpoint.

ZeroProof normally builds a simulated world from the supplied tools, policy,
and traces. That is appropriate for record-shaped tools and behavioral
questions. If the evaluated agent edits code or correctness depends on a real
database/service, pass `execute(tool_name, arguments)` using the developer's
isolated harness. For code, use one disposable checkout per rollout and real
reads, writes, commands, exit codes, and hidden tests. If no safe harness
exists, report that prerequisite instead of treating invented files as truth.

## Grade after simulation

The developer owns correctness. Prefer a callable judge that returns a reward
in `[0, 1]` and a reason. Judge errors remain unjudged; they are never silently
converted to failures.

```python
def developer_judge(row: dict):
    # Apply the target agent's policy, expected end state, tests, or evaluator.
    return {"reward": 1 if developer_passes(row) else 0,
            "reason": developer_reason(row)}

scored = data.grade(judge=developer_judge)
scored.save("simulations/scored.jsonl")
print(scored.report(tools=tools, system_prompt=policy))
```

Use `grade=True` only for ZeroProof's deterministic structural/conduct screen;
it is not the developer's semantic authority. Hosted or BYOK LLM grading is
optional. Keep unjudged rows out of selection and report judge failures.
`data.grade(judge=...)` deliberately leaves `data.trajectories` and the raw
simulation file unchanged; use the returned `ScoredData` from that point on.

## Inspect before keeping rows

Reject a run when `generator_fallback` appears in `data.degraded` or
`data.stopped_because == "writer_exhausted"`. Other degradation notes are
advisory; name them in the report and inspect their effect.

Always check:

- row count, unique prompts, stop reason, and degraded notes;
- tool names, argument/result linkage, empty or error rollouts;
- coverage by tool, policy rule, world state, condition, stance, and history;
- leakage against source traces and held-out evals;
- whether trace-guided generation actually emphasizes observed failure regions;
- every failed row and at least 20 passing rows;
- for real execution, that every call reached the intended isolated world.

Generated rows are candidates until the developer's judge and review accept
them. Never overwrite existing evals, leak credentials, or merge generated
cases silently.

## Select and export

For eval expansion, keep reviewed cases in a new file with run provenance and
integrate them only after developer approval.

For SFT after binary grading:

```python
report = data.training_set("train.jsonl", target=1000, validate=True)
```

This selects diverse passing demonstrations and applies the tool-call
round-trip export gate. For a judge-contract result returned as `ScoredData`,
use `scored.select_for_sft()` and `zps.export_dataset(...)` with the run's
policy and tools. For RL, use repeated groups and `select_for_rl`; keep groups
whole and require meaningful within-group reward variation.

## Deliverable

Return the generated JSONL and a concise report containing inputs used, model,
seed/configuration, trace counts, search mode, row/coverage counts, grading
method, rejected rows, leakage result, and review status.
State what was simulated, what came from the developer's real harness, and what
remains unverified.
