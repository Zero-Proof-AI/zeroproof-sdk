# Workflow

Do the preflight below first and write down what you found before running
anything below. Everything here assumes `import zeroproof_simulations as
zps` and that `tools` (a list of `{"type": "function", "function": {...}}`
dicts) and `policy` (a string) are already in hand from the repo.

## 0. BYOK setup (do this before either lane, once)

Row generation always calls a model to write the human side and play the
agent. Two ways to give it one:

**Bring your own key (recommended for a customer's own repo).** Any
OpenAI-compatible endpoint works, including the customer's own deployed
model:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"   # or their own endpoint
```

```python
data = zps.simulate(agent="openai:gpt-4.1-mini", tools=tools,
                     system_prompt=policy, ...)
```

`agent="openai:<model>"` reads `OPENAI_BASE_URL` for the request and
`OPENAI_API_KEY` for auth, so pointing `OPENAI_BASE_URL` at a self-hosted
vLLM/TGI/etc. server that speaks the OpenAI chat completions API works the
same way; no ZeroProof credential is involved.

**ZeroProof-hosted Qwen (no `agent=` argument, or `agent=None`).** Needs
`VLLM_API_KEY` from ZeroProof; the endpoint is shared and rate limited.
Fine for a quick trial, not the default recommendation for a customer's
production repo.

```bash
export VLLM_API_KEY="..."     # only if not bringing your own model
```

Grading is a separate concern from generation; see step 3. The hosted
judge that backs `zps.grade(...)` / `.grade()` with no arguments needs
`VLLM_API_KEY` too. Do not depend on it for a customer who has not shared
that key. Use `grade=True` (offline, free, no key) or a `grader=` callable
instead.

## 1. Decide the lane

```python
report = zps.trace_report(traces, tools=tools, policy=policy) if traces else {"traces": 0}
if traces:
    print(zps.format_trace_report(report))
```

- `report["traces"] == 0`, or every row is unlabeled with `faults_observed`
  empty and `advisory_labels == 0`: **lane 2 only** (policy-guided
  discovery; nothing to repair yet).
- `report["fails"] > 0` or `report["advisory_labels"] > 0`, and the goal is
  closing gaps the graded traces already show: **lane 1** (trace-guided
  repair).
- Both hold and the goal is broad coverage, not just the observed failures:
  **combined**, run lane 1 first, then lane 2 (or `mode="adaptive",
  until="saturation"`) for what mining did not touch.

`report["foreign_tools"]` (when `tools=` was passed) lists tool names the
traces call that this agent's spec does not declare; flag that in the
preflight report. It usually means the tool list is stale, not that the
traces are wrong.

## 2. Lane 1: trace-guided repair

Traces can be a JSONL path, a list of dicts, or `rows_from_otel(...)`
output; `zps.load_traces()` normalizes OpenAI-style `messages`,
`tool_trace`, and platform export shapes into the canonical
`{prompt, steps, final_text, reward}` schema, so the loader almost never
needs calling directly; `traces=` accepts all of it.

```python
# OTLP/GenAI spans straight from a collector export
traces = zps.rows_from_otel(open("traces.otlp.json").read())

data = zps.simulate(
    agent="openai:gpt-4.1-mini",         # or omit for hosted Qwen
    tools=tools, system_prompt=policy,
    traces=traces,
    mode="explore",                       # new distinct situations, aimed by the traces
    grade=True,                           # deterministic conduct grade, offline
    budget=400, time_budget=180,
    advanced={"seed": 0},
    output="new_evals/trace_guided.jsonl",
)
blocking = [d for d in data.degraded if d in ("generator_fallback", "writer_exhausted")]
if blocking:
    raise RuntimeError(f"situations were not model-written, do not use: {blocking}")
if data.degraded:
    print("advisory notes, rows are still usable:", data.degraded)
```

`traces=` reshapes the covering grid (`dimensions_from_traces` under the
hood): tools and world states the traces actually hit move to the front
and unobserved ones drop out or fall back, so budget concentrates on the
failing regions. **Source traces never enter the generated dataset.**
That is enforced by construction, and can be double-checked:

```python
leak = zps.leakage_report(data.trajectories, traces)
assert leak["n_leaky"] == 0, leak["leaky"]   # near-copies of a source trace
```

If the goal is repeat-groups over one failing ask (to see whether a fix
holds under repetition), use `mode="rl"` instead of `mode="explore"`. That
is what `zps.simulate_from_traces(traces, tools=tools, policy=policy)`
defaults to.

## 3. Lane 2: policy-guided discovery

```python
data = zps.simulate(
    agent="openai:gpt-4.1-mini",
    tools=tools, system_prompt=policy,
    mode="adaptive", until="saturation",   # new situations + phrasings + repeats until coverage plateaus
    grade=True,
    budget=800, time_budget=300,
    advanced={"seed": 0},
    output="new_evals/policy_guided.jsonl",
)
blocking = [d for d in data.degraded if d in ("generator_fallback", "writer_exhausted")]
if blocking:
    raise RuntimeError(f"situations were not model-written, do not use: {blocking}")
if data.degraded:
    print("advisory notes, rows are still usable:", data.degraded)
```

`mode="explore"` (the default) is fine for a smaller first pass: one
unique situation per row, no repeats. `mode="adaptive", until="saturation"`
is the fuller sweep: it keeps generating new situations, phrasings, and
repeats until the coverage curve plateaus, which is what "surface the
blind spots" means in practice.

## 4. Grading

Three options, pick one per run, never assume a default that needs a key
the customer has not given you:

| Option | What it needs | What it produces |
|---|---|---|
| `grade=True` | nothing | deterministic conduct score (undeclared tools, degenerate output, empty/infra stubs) |
| `grader=my_judge` | your callable, any dependencies it needs | whatever `my_judge(row) -> {"reward": 0..1, ...}` returns, run through `normalize_judge_result` |
| `llm_grade=True` | `OPENAI_API_KEY` | an advisory LLM pass layered on top, does not replace `reward` |

A `grader=` callable only needs to honor the judge contract:

```python
def my_judge(row: dict) -> dict:
    # row is {"prompt", "messages", "steps", "final_text", "scenario_id", ...}
    ok = "sorry" not in row["final_text"].lower()
    return {"reward": int(ok), "reason": "" if ok else "apologized instead of acting"}

data = zps.simulate(tools=tools, system_prompt=policy, traces=traces,
                     grader=my_judge, budget=400, output="new_evals/trace_guided.jsonl")
```

Do **not** call `zps.grade(...)` / `data.grade()` with no arguments unless
the customer has explicitly given you a `VLLM_API_KEY` for the hosted
judge. That is a ZeroProof-hosted call, not a customer-local one.

## 5. Compare against existing evals and write the report

## Preflight: what to find in the repo

**Tool schemas.** Look for:

- An OpenAI-style tool/function list (`{"type": "function", "function":
  {"name", "description", "parameters"}}`), used directly as `tools=`.
- A LangChain executor, LangGraph graph, OpenAI Agents SDK agent, or a
  Claude Code CLI binary: pass the object itself to `zps.inspect(agent)`
  or `zps.connect(agent)` and read `.tools` off the returned
  `AgentProfile`; do not hand-transcribe its schema.
- If only trace exports exist and there is no schema in the repo, two
  helpers draft one from observed calls, mechanically and with no model.
  They live in the `ingest.traces` submodule, not the top-level `zps.`
  namespace:
  `from zeroproof_simulations.ingest.traces import tools_from_traces, infer_harness`.
  `tools_from_traces(traces)` returns an OpenAI-style tool list (string
  parameters only, arg names unioned across calls); `infer_harness(traces)`
  additionally reads JSON types and marks an argument `required` if every
  observed call included it. Say plainly that a drafted schema only
  describes arguments the traces happened to exercise, and ask the repo
  owner to confirm it before treating it as ground truth.

**Policy.** The system prompt as the agent actually sends it: a Python
string constant, a `.md`/`.txt` file, a config value, or (again)
`zps.inspect(agent).policy`. If several policy fragments exist (a base
prompt plus per-tool notes), concatenate what the agent actually sees at
call time; do not summarize or rewrite it.

**Traces.** Common shapes to check for: a JSONL/NDJSON export, a directory
of per-run JSON files, or an OTLP/GenAI span export (`gen_ai.*`
attributes, or an `resourceSpans`/`scopeSpans` OTLP/HTTP JSON batch).
`zps.load_traces(path_or_rows)` and `zps.rows_from_otel(source)` both
normalize to `{prompt, steps, final_text, reward}`; try `load_traces`
first, and `rows_from_otel` if the export is OTLP-shaped. A row with
`reward` 0/1, a `scores` dict, or an observed tool fault (`timeout`,
`deny`, `malformed`, `not_found`) counts as a labeled failure for lane 1;
an unlabeled row is still useful context but is not repair signal.

**Existing evals.** Look for `evals/`, `eval/`, `tests/` with agent-shaped
fixtures, a promptfoo/braintrust/langsmith config, or any JSONL/YAML file
whose rows look like `{prompt, ..., expected/reward/pass}`. Note the
location and format in the report. This skill reads them for the coverage
comparison and never writes to them.

## Provenance on every generated row

Four provenance fields the SDK does not stamp per row on its own
(`source_lane`, `generator_model`, `seed`, plus a `run_id` for traceability)
need adding as a post-processing step, right after the run, before the
file is handed over for review:

```python
import json
import uuid

def stamp_provenance(data, *, lane: str, agent_spec: str | None,
                     seed: int, path: str) -> None:
    """Adds provenance fields to every row this run wrote, in place.

    SimulationData carries no per-run id of its own, so mint one here.
    That is the id this stamping step (not the SDK) promises to be stable
    for every row this call touches.
    """
    model = agent_spec or "hosted-qwen"
    run_id = f"{lane}_{uuid.uuid4().hex[:10]}"
    rows = []
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            row["source_lane"] = lane                # "trace_guided" | "policy_guided"
            row["generator_model"] = model
            row["seed"] = seed
            row["run_id"] = run_id
            rows.append(row)
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")

stamp_provenance(data, lane="trace_guided", agent_spec="openai:gpt-4.1-mini",
                 seed=0, path="new_evals/trace_guided.jsonl")
```

A stamped row therefore carries, at minimum:

```json
{
  "prompt": "...", "messages": [...], "steps": [...], "final_text": "...",
  "scenario_id": "sit_7f2c...", "world_state": "entity already acted on",
  "faults": {"tool": "run_tests", "kind": "timeout"}, "stance": "hurried",
  "reward": 0, "reason": "claimed success after a tool timeout",
  "source_lane": "trace_guided", "generator_model": "openai:gpt-4.1-mini",
  "seed": 0, "run_id": "trace_guided_a1b2c3d4e5"
}
```

`world_state`, `faults`, and `stance` are only present when the scenario
carried them (a clean/no-fault row omits `faults`; an unlabeled row omits
`reward`). That is the SDK's own export behavior, not a bug in the
stamping step.
