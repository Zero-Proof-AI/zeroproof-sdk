# zeroproof-simulations

Generate training data for any AI agent from its tools and policy. The SDK maps what the agent can face, samples broad user situations, writes natural user requests, runs the agent, and records every user, assistant, and tool message.

You can stop at a row budget, a time budget, or estimated coverage saturation. Grading is optional and runs after generation. Hosted Qwen (`Qwen/Qwen3-4B-Instruct-2507`) powers the default writer and rollout model; no OpenAI key is required.

```bash
git clone <this-repo>
cd zeroproof-simulations
pip install -e .
export VLLM_API_KEY=...
```

`VLLM_API_KEY` is the hosted ZeroProof credential. Contact ZeroProof to get one. Do not commit the key.

## Connect an agent

Pass a spec folder, tools and policy, or an agent. Framework objects are inspected. Caller tools and policy are merged in.

```python
import zeroproof_simulations as zps

data = zps.simulate(spec="specs/github")
data.save("rollout.jsonl")
```

```python
data = zps.simulate(tools=my_tools, policy=my_policy)
data = zps.simulate(agent=my_agent)
```

`agent=` accepts a callable, `https://...`, `vllm:`, `ollama:`, or `openai:`. Default `time_budget` is 60 seconds and `budget` is 1000 rows. `spec` is a folder or file. `specs/github` ships with the package.

## Modes

Pick how situations and repeats are mixed. Default is `adaptive`.

| Mode | What you get |
|---|---|
| `adaptive` | Default. The scheduler picks new cards, more phrasings, or more rollouts as it learns what is missing. |
| `sft` | Starting point for wording variety. Multiple user phrasings of the same situation. |
| `rl` | Starting point for contrastive trajectories. Multiple independent trajectories of the same opener. |
| `explore` | Unique cards. New situations only. |

`sft` defaults to 3 requests per situation and one rollout. `rl` defaults to one request and 3 rollouts. Override with `requests_per_situation` and `rollouts_per_request`. `unique_situations=True` uses the same unique-card policy as `explore`.

The three independent counts are:

- `situations`: distinct situations to cover
- `requests_per_situation`: different user phrasings of one situation
- `rollouts_per_request`: independent agent trajectories from one phrasing

## Stopping

`until="compute"` uses only the row and time caps. `until="saturation"` may stop earlier when covered situation cells and tool-action shapes have enough copies and new behavior has stayed flat across several batches. This is an estimate over the observed agent, tools, policy, and sampled distribution, not proof that every possible request has been exhausted. The estimate is available in `data.coverage`.

## Speed

2 min, airline spec. 1-6 user turns, 0-8 tool calls. Rates vary with load. `sft` and `rl` used each mode's defaults (`requests_per_situation=3` and `rollouts_per_request=3`).

| Mode | Rows in 2 min | Rate | Unique openers | Multi-turn | Contrastive |
|---|---|---|---|---|---|
| `explore` | 240 | 120/min | 240 | 57% | |
| `sft` | 278 | 139/min | 278 | 63% | |
| `rl` | 625 | 296/min | 209 | 61% | 27 |

`time_budget` and `budget` are caps.

## Examples

```python
import zeroproof_simulations as zps

data = zps.simulate(spec="specs/github")
data.save("rollout.jsonl")
```

Note: Default `adaptive` mode. Mixes new situations with gap-filling.

```python
import zeroproof_simulations as zps

data = zps.simulate(spec="specs/github", mode="rl", rollouts_per_request=5)
data.save("rollout.jsonl")
```

Note: Same opener, 5 agent trajectories. Slower unique-situation fill. Good for preference or group-relative RL.

```python
import zeroproof_simulations as zps

data = zps.simulate(spec="specs/github", mode="explore")
# or unique_situations=True
data.save("rollout.jsonl")
```

Note: Every row is a new situation. Slower (airline explore was 120/min vs rl 296/min). Good for SFT coverage with no repeats.

```python
import zeroproof_simulations as zps

data = zps.simulate(spec="specs/github", advanced={"fault_rate": 0.5})  # or 0 to turn off
data.save("rollout.jsonl")
```

Note: Injects tool and sandbox failures (timeout, permission denied, stale or malformed results). Good for teaching agents to handle broken tools.

## Knobs

Public `simulate()` arguments. All other controls are `advanced=`.

| Parameter | Default | Meaning |
|---|---|---|
| `agent` | hosted Qwen | Rollout model |
| `spec` | none | Tools and policy path |
| `tools`, `policy` | from spec or agent | Tool list and system policy |
| `situations` | none | Distinct situations |
| `requests_per_situation` | from mode | Different user phrasings of the same situation. `sft` default is 3. |
| `rollouts_per_request` | from mode | Independent agent runs of the same phrasing. `rl` default is 3. |
| `unique_situations` | `False` | New situations only |
| `mode` | `"adaptive"` | `sft`, `rl`, `adaptive`, or `explore` |
| `budget` | `1000` | Row cap |
| `time_budget` | `60` | Seconds. `None` or `0` disables the clock |
| `until` | `"compute"` | `"saturation"` also stops when coverage has enough copies |
| `grade` | `True` | Conduct score after rollouts |
| `output` | none | JSONL path |
| `advanced` | none | Keys below |

### Advanced

Pass these in `advanced={...}`.

| Key | Default | What it does |
|---|---|---|
| `concurrency` | `192` | Parallel agent rollouts. |
| `simulator` | hosted Qwen | Writes user messages. `False` uses templates. |
| `avg_turns` | `4` | Target conversation length. |
| `scenario_concurrency` | `4` | Parallel situation writers. |
| `completions_per_request` | `3` | User-message variants per writer request. |
| `fault_rate` / `risk` | `0.5` | Injects sandbox and tool faults (timeouts, malformed or stale results, permission errors). Use this to teach agents to handle broken tools. `risk` is the same key. Set `0` to turn it off. |
| `seed` | `0` | Reproducible situation and fault draws. |
| `embedder` | `"hash"` | How situations are compared for coverage. |
| `dimensions` | from spec | Coverage axes built from tools and policy. |
| `backend` | none | Rollout model override (`vllm:`, `ollama:`, `openai:`). |
| `texture` | `0.35` | How often user messages get extra style. |
| `max_turns` | from tools | Hard cap on conversation length (8 to 40). |
| `temperature` | `0.8` | Sampling temperature for the hosted rollout model. |
| `grader` | none | Custom scorer. Default is the built-in conduct grade. |
| `llm_grade` | `False` | Also run an LLM judge (`llm_reward`, `llm_reason`). |
| `llm_spec` | `openai:gpt-4o-mini` | Judge model. Needs `OPENAI_API_KEY`. |
| `seed_prompts` | none | Starter user openers to include. |

Aliases: `n=` is `requests_per_situation`; `repeats=` and `rollouts_per_prompt=` are `rollouts_per_request`; `unique=True` is `unique_situations=True`; `extra_situations=` is `advanced["seed_prompts"]`; `until="first"` is `until="saturation"`; `risk=` is `fault_rate`.

## Output

Each row includes `prompt`, `messages`, `steps`, `final_text`, `scenario_id`. `messages` preserves the complete conversation: user turns, assistant turns with optional `tool_calls`, and tool results. `prompt` is the first user line. A tool step is `{tool, arguments, result}` and may include `text` spoken on that turn. A text-only assistant turn is `{text}`. A later user line is `{user}`. `final_text` is the last agent utterance. Optional: `world_state`, `faults`. With `grade=True` (default): `reward`, `reason`. Use `grade=False` to generate without scoring. `fault_detected` appears only when a sandbox or tool fault was present; it is not the score. Optional LLM judge (`data.grade(llm=True, api_key="...")`) adds `llm_reward`, `llm_reason`.

## License

Apache-2.0
