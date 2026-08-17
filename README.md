# zeroproof-simulations

Training data for any AI agent, across the situations it should actually see. From the tools and system prompt we sample a grid of cases, write a human request, run the agent in a fake world built for that agent, and save the conversation plus tool calls. Default `explore`: every row is a unique situation.

## Overview

Spec in. JSONL out. Your agent, your grader.

| | Call | What it does |
|---|---|---|
| Connect | `inspect` / `connect` | Read tools and system prompt. Wrap any agent (spec, callable, LangChain, HTTP). Hosted Qwen, or any OpenAI-compatible URL |
| World | `MockEnvironment` | A fake world for this agent. Tools return plausible results, including failures |
| Generate | `simulate(...)` | Same model writes the user and plays the agent. Samples every kind of situation until the row or time budget hits |
| | `SimulationData` | The run: rows, coverage, `save`, `rank`, `grade` |
| Export | `conversation` / `save` / `rows` | Trainer JSONL: `prompt`, `messages`, `steps`, `final_text` |
| Score | `conduct_grade` / `rank` | Conduct `reward`, or your `grade=`. Rank is a second pass on humanness |

## How it searches

We sample all the axes that make a situation different, not one happy path.

The grid comes from this agent: which tool, which rule in the system prompt, how the user shows up (ordinary, vague, hurried, adversarial), what the world looks like, whether the tool works or fails, whether they have been here before. Ordinary asks first, then the edges. Same model writes the user and plays the agent.

The row cap and the clock are the goal. Spend them on diversity, not copies.

Variation is three independent knobs. Do not mix them up.

- **Situations.** New worlds. Default `explore` makes every row unique.
- **Phrasings.** Same world, different human wording. Tone, intent, personality.
- **Repeats.** Same wording, independent agent runs. Outcome variance for RL.

Look at unique openers, whether conversations go multi-turn, and whether faults were actually handled. Near-duplicates mean more situations. Contrastive pairs mean more repeats. Style coverage means more phrasings.

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

## Advanced modes

`explore` is unique situations. Use another mode when you want phrasings or repeats on purpose.

| Mode | What you get |
|---|---|
| `sft` | Same worlds, multiple phrasings |
| `rl` | Same phrasing, multiple repeats |
| `adaptive` | Mix of new situations, phrasings, and repeats. Messier on a short clock. Best with `until="saturation"` |

```python
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

## License

Apache-2.0
