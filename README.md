# zeroproof-simulations

Generate diverse training conversations for any agent, grounded in its tools and system prompt.

The SDK inspects the agent, simulates a world consistent with those tools (objects, results, failures), and samples scenarios across that space. The same model writes the user and plays the agent. Default `explore`: one unique situation per row.

## Overview

| | Call | What it does |
|---|---|---|
| Connect | `inspect` / `connect` | Read tools and system prompt. Wrap any agent (spec, callable, LangChain, HTTP). Hosted Qwen, or any OpenAI-compatible URL |
| World | `MockEnvironment` | Full world for this agent: objects, tool results, failures, mutations |
| Generate | `simulate(...)` | Sample a scenario, run the conversation, stop on the row or time budget |
| | `SimulationData` | The run: rows, coverage, `save`, `rank`, `grade` |
| Export | `conversation` / `save` / `rows` | Trainer JSONL: `prompt`, `messages`, `steps`, `final_text` |
| Score | `conduct_grade` / `rank` | Conduct `reward`, or your `grade=`. Rank is a second pass on humanness |

## How to use

```bash
uv sync
export VLLM_API_KEY=...
```

Hosted Qwen is the default writer and agent. `uv add /path/to/zeroproof-simulations` from another project. `uv run pytest` after `uv sync --extra dev`.

```python
import zeroproof_simulations as zps

data = zps.simulate(spec="specs/github", output="rollout.jsonl")
data = zps.simulate(tools=my_tools, system_prompt=my_system_prompt)
data = zps.simulate(agent=my_agent)
```

| Knob | Default | |
|---|---|---|
| `agent` / `spec` | hosted Qwen | Callable, URL, or tools + system prompt |
| `budget` / `time_budget` | `1000` / `60` | Stop when either hits. `0` or `None` turns the clock off |
| `requests_per_situation` | from mode | Phrasings: ways to ask one situation. Alias `phrasings=` |
| `rollouts_per_request` | from mode | Repeats: reruns of one phrasing. Alias `repeats=` |
| `fault_rate` | `0.5` | Broken tools. `0` off |
| `grade` | `True` | Conduct score, or pass your own callable |
| `llm_grade` | `False` | Extra LLM judge. Needs `OPENAI_API_KEY` |
| `output` | | JSONL path |

## What to run

Depends on the use case. How each scenario is built is in [The recipe](#the-recipe).

| You want | Mode | What happens |
|---|---|---|
| Many distinct situations | `explore` (default) | New situation every row |
| Same situation, different wording | `sft` | Multiple phrasings: tone, intent, personality |
| Same request, different agent behavior | `rl` | Multiple repeats of one phrasing |
| A mix, until coverage plateaus | `adaptive` | New situations, phrasings, and repeats. Best with `until="saturation"` |

```python
zps.simulate(spec="specs/github")                 # explore
zps.simulate(spec="specs/github", mode="sft")
zps.simulate(spec="specs/github", mode="rl")
zps.simulate(spec="specs/github", mode="adaptive", until="saturation")
```

## Speed

2 min, airline spec.

| Mode | Rows | Rate | Unique openers |
|---|---|---|---|
| `explore` | 240 | 120/min | 240 |
| `sft` | 278 | 139/min | 278 |
| `rl` | 625 | 296/min | 209 |

## Parameter reference

| Parameter | Default | Meaning |
|---|---|---|
| `agent` | hosted Qwen | Rollout model |
| `spec` | | Tools and system prompt path |
| `tools`, `system_prompt` | from spec or agent | Tool list and agent system prompt |
| `situations` | | Distinct situations (N) |
| `requests_per_situation` | from mode | Phrasings per situation (n). Alias `phrasings=` / `n=` |
| `rollouts_per_request` | from mode | Repeats per phrasing (k). Alias `repeats=` |
| `unique_situations` | on in `explore` | Unique situations only |
| `mode` | `"explore"` | `explore`, `sft`, `rl`, `adaptive` |
| `budget` | `1000` | Row cap |
| `time_budget` | `60` | Seconds. `None` or `0` disables |
| `until` | `"compute"` | `"saturation"` also stops when coverage plateaus |
| `grade` | `True` | Conduct score |
| `llm_grade` | `False` | Extra LLM judge |
| `output` | | JSONL path |
| `advanced` | | Keys below |

| `advanced` key | Default | |
|---|---|---|
| `concurrency` | `192` | Parallel rollouts |
| `embedder` | `"hash"` | Prompt selection |
| `seed` | `0` | Reproducible draws |
| `avg_turns` | `4` | Target conversation length |

Aliases: `phrasings=` / `n=` → `requests_per_situation`; `repeats=` → `rollouts_per_request`; `unique=` → `unique_situations`; `policy=` → `system_prompt`; `risk=` → `fault_rate`.

## Output

Each row: `prompt`, `messages`, `steps`, `final_text`, `scenario_id`. Optional `world_state`, `faults`, `reward`, `reason`. `llm_grade=True` adds `llm_reward`. `zps.rank(path)` adds `quality` without changing `reward`.

## The recipe

Each scenario is a draw across the world and the human.

**World** (from this agent's tools and system prompt)

- objects and tool results that match the spec
- tool outcome: success, timeout, deny, stale, etc.
- world state: exists, missing, already handled, unfinished, etc.
- history: first visit, prior miss, return, etc.
- rules the agent is supposed to follow

**Human**

- intent: which tool, what they want (randomized sometimes)
- stance: ordinary, ambiguous, adversarial, hurried, etc.
- persona: first time, returning, in a hurry, etc.
- tone: impatient, frustrated, polite, etc.
- typing: standard, lowercase, typo, clipped, etc.

Ordinary asks first, then the edges. On top of that, we embed the openers and add a bit of random noise so the batch stays spread out, not a cluster of near-copies. Spend the row cap and the clock on diversity, not copies.

## License

Apache-2.0
