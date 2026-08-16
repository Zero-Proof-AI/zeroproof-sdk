# zeroproof-simulations

Simulate two-person, tool-using conversations from any tools and policy spec. You connect an agent. Generation uses hosted Qwen (`Qwen/Qwen3-4B-Instruct-2507`) and does not need an OpenAI key.

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
| `explore` | New situations only. One conversation each. |
| `sft` | Three different user phrasings of each situation. One agent run per phrasing. |
| `rl` | One phrasing per situation. Three independent agent runs of that phrasing. |
| `adaptive` | Default. Walks new situations and fills gaps as it goes. |

```python
data = zps.simulate(spec="specs/github", mode="sft")
data = zps.simulate(spec="specs/github", mode="rl")
data = zps.simulate(spec="specs/github", mode="explore")
```

`unique_situations=True` keeps picking new situations under any mode.

## Speed

2 min, airline spec. 1-6 user turns, 0-8 tool calls. Rates vary with load.

| Mode | Rows in 2 min | Rate | Unique openers | Multi-turn | Contrastive |
|---|---|---|---|---|---|
| `explore` | 240 | 120/min | 240 | 57% | |
| `sft` | 278 | 139/min | 278 | 63% | |
| `rl` | 625 | 296/min | 209 (~3 hits each) | 61% | 27 |

`time_budget` and `budget` are caps.

## Knobs

Public `simulate()` arguments. All other controls are `advanced=`.

| Parameter | Default | Meaning |
|---|---|---|
| `agent` | hosted Qwen | Rollout model |
| `spec` | none | Tools and policy path |
| `tools`, `policy` | from spec or agent | Tool list and system policy |
| `situations` | none | Distinct situations |
| `requests_per_situation` | from mode | Different user phrasings of the same situation |
| `rollouts_per_request` | `1` (mode may override) | Independent agent runs of the same phrasing |
| `unique_situations` | `False` | New situations only |
| `mode` | `"adaptive"` | `sft`, `rl`, `adaptive`, or `explore` |
| `budget` | `1000` | Row cap |
| `time_budget` | `60` | Seconds. `None` or `0` disables the clock |
| `until` | `"compute"` | `"saturation"` also stops when coverage has enough copies |
| `grade` | `True` | Conduct score after rollouts |
| `output` | none | JSONL path |
| `advanced` | none | Keys below |

### Advanced

`advanced={...}`: `concurrency` (192), `simulator` (hosted Qwen; `False` uses templates), `avg_turns` (4), `scenario_concurrency` (4), `completions_per_request` (3), `fault_rate` / `risk` (0.5), `seed` (0), `embedder` (`"hash"`), `dimensions` (from spec), `backend`, `texture`, `max_turns`, `temperature`, `grader`, `llm_grade`, `llm_spec` (`openai:gpt-4o-mini`), `seed_prompts`.

Aliases: `n=` is `requests_per_situation`; `repeats=` and `rollouts_per_prompt=` are `rollouts_per_request`; `unique=True` is `unique_situations=True`; `extra_situations=` is `advanced["seed_prompts"]`; `until="first"` is `until="saturation"`; `risk=` is `fault_rate`.

## Output

Each row includes `prompt`, `messages`, `steps`, `final_text`, `scenario_id`. `messages` is the conversation (user, assistant with optional `tool_calls`, tool). `prompt` is the first user line. A tool step is `{tool, arguments, result}` and may include `text` spoken on that turn. A text-only assistant turn is `{text}`. A later user line is `{user}`. `final_text` is the last agent utterance. Optional: `world_state`, `faults`. With `grade=True` (default): `reward`, `reason`. `fault_detected` appears only when a sandbox or tool fault was present; it is not the score. Optional LLM judge (`data.grade(llm=True, api_key="...")`) adds `llm_reward`, `llm_reason`.

## License

Apache-2.0
