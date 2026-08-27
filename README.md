# zeroproof-simulations

Generate diverse training conversations for any agent, grounded in its tools and system prompt.

The SDK inspects the agent, simulates a world consistent with those tools (objects, results, failures), and samples scenarios across that space. The same model writes the user and plays the agent. Default `explore`: one unique situation per row.

## Overview

`simulate()` is a pipeline.

1. **Read the agent.** Tools and system prompt. That is the spec of the world.
2. **Build a fake world from those tools.** Objects, plausible results, and faults (timeout, deny, junk).
3. **Write users.** A separate writer (same hosted model, different prompt, no agent policy) samples situations across tools, stance, history, and so on.
4. **Pick the diverse ones.** Embeddings plus a bit of noise so the batch is not 200 copies of same prompy.
5. **Play the agent.** It talks, calls tools, gets results, talks again. All of that is stored: user text, agent text, tool calls, tool results, `final_text`.
6. **Grade.** Deterministic conduct score by default. Attach an LLM if you want. Or pass your own `grade=`.

Stop when the row cap or the clock hits.

## How to use

```bash
uv sync
export VLLM_API_KEY=...
```

ZeroProof hosts Qwen for simulation: it powers the default user writer and rollout agent.
Ask us for a `VLLM_API_KEY` to try it. The endpoint is shared and rate limited.

From another project:

```bash
pip install zeroproof-simulations   # or: uv add zeroproof-simulations
```

Zero runtime dependencies, Python 3.10+. Installing from PyPI rather than a
path or a git URL matters if you build a Prime Intellect environment on this:
the Environments Hub installs a pushed env with plain pip, so a `[tool.uv.sources]`
git pin resolves locally and then fails on their runtime with a
`ModuleNotFoundError`.

`uv run pytest` after `uv sync --extra dev`.

```python
import zeroproof_simulations as zps

data = zps.simulate(tools=my_tools, system_prompt=my_system_prompt, output="rollout.jsonl")
data = zps.simulate(agent=my_agent)
```

Pass `spec=` if you have a local tools-and-system-prompt folder. The generated datasets are on [Hugging Face](https://huggingface.co/datasets/zero-proof-ai/agent-simulations), organized by agent type instead of stored in this repo.

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
zps.simulate(tools=my_tools, system_prompt=my_system_prompt)                 # explore
zps.simulate(tools=my_tools, system_prompt=my_system_prompt, mode="sft")
zps.simulate(tools=my_tools, system_prompt=my_system_prompt, mode="rl")
zps.simulate(tools=my_tools, system_prompt=my_system_prompt, mode="adaptive", until="saturation")
```

## Examples

| Example | What it does |
|---|---|
| [`examples/agent-behavior`](examples/agent-behavior) | Start here if the platform is new to you. Runs a coding agent with bad habits against real tests, streams every turn to Zero Proof as OTLP spans plus a judge verdict, and fills a dashboard with behaviour worth looking at. No dependencies. |
| [`examples/prime-intellect-rl`](examples/prime-intellect-rl) | Generates a GRPO-ready dataset with `simulate(mode="rl")` and checks it carries gradient before you spend GPU time on it. |

## Store datasets on Zero Proof Labs

Push a run to your Zero Proof Labs account so the optimization framework
can iterate on it. Auth is your `zp_` API key from
https://www.zeroproofai.com/platform (env `ZEROPROOF_API_KEY`).

```python
data = zps.simulate(spec="specs/github")
v1 = data.push("github-explore-v1")            # -> {"datasetId": "ds_...", ...}

# iterate, then push the next version with lineage
v2 = data.push("github-explore-v2", parent=v1["datasetId"])

zps.datasets()                                  # list yours + storage used
rows = zps.pull(v1["datasetId"])               # rows, or pass path= for a file
zps.push_file("rollout.jsonl")                 # upload an existing JSONL
zps.delete_dataset(v1["datasetId"])            # permanent
```

Storage is private per account, 5 GB free. `parent=` records dataset
lineage so iterations show as a family on the platform.

## Speed

Two-minute airline runs using ZeroProof-hosted Qwen. Results were measured on the
hosted GPU with warm replicas and burst under load.

| Mode | Rows | Rate | Unique openers |
|---|---|---|---|
| `explore` | 240 | 120/min | 240 |
| `sft` | 278 | 139/min | 278 |
| `rl` | 625 | 296/min | 209 |

## Parameter reference

| Parameter | Default | Meaning |
|---|---|---|
| `agent` | hosted Qwen | Rollout model |
| `spec` | | Local tools and system prompt path |
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
