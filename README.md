# whileai

[![CI](https://github.com/whilehq/whileai-sdk/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/whilehq/whileai-sdk/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/whileai)](https://pypi.org/project/whileai/)
[![Python](https://img.shields.io/pypi/pyversions/whileai)](https://pypi.org/project/whileai/)
[![Downloads](https://img.shields.io/pypi/dm/whileai)](https://pypistats.org/packages/whileai)
[![Coverage gate](https://img.shields.io/badge/coverage-%E2%89%A5%2090%25%20gated-brightgreen)](.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Post-training data and evaluation for tool-using language-model agents.

`whileai` simulates the situations an agent can meet, rolls the agent
through them against a world that fails on schedule, grades every rollout
under one judge contract, and turns graded rows into SFT, preference and RL
data with the checks the literature says to run: pass@k with intervals,
difficulty bands, judge validation, decontamination, reward-hacking scans.
Every method names the chapter of [rlhfbook.com](https://rlhfbook.com)
(Lambert, *RLHF and LLM Post-Training*) or the paper it implements, and
every recipe reports a paired delta with a 95% interval on a held-out set,
never a mean alone.

```bash
pip install whileai        # or: uv add whileai
```

One runtime dependency (`requests`), Python 3.10 to 3.13, typed. Formerly
`zeroproof`; that name still installs this package.

## Sixty seconds, offline

No key, no network. The seeded agent answers honestly and, on a labeled
fraction of rollouts, does one wrong thing on purpose, so a judge that
catches exactly those rows is a judge that works.

```python
import whileai.simulations as wai

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]

data = wai.simulate(
    wai.seeded_agent(TOOLS),
    tools=TOOLS,
    system_prompt="Help customers with orders.",
    simulator=False,  # template writer: no model
    mode="rl",  # k rollouts per prompt
    repeats=4,
    repeat_policy="fixed",
    budget=64,
)
scored = data.grade(judge=lambda row: {"reward": int(not row["seeded"])})
print(scored.pass_at)  # pass@1 [95% CI over tasks] | pass^k | pass@k | headroom
```

```
pass@1 0.67 [0.55..0.78] | pass^4 (pass_pow_k) 0.19 [0.00..0.38] | pass@4 1.00 [1.00..1.00] | headroom 0.33 (16 groups, k=4)
```

Swap in your agent as `agent(message) -> {"steps": [...], "final_text": ...}`,
a model spec such as `"openai:gpt-4.1-mini"`, or a model you serve. The
judge is any callable `row -> {"reward": 0..1}`, a verifier such as
`CodeExec`, or the hosted judge.

## The loop

| Step | Call | What it computes | Source |
|---|---|---|---|
| Simulate | `simulate(agent, tools=, system_prompt=, mode="rl", repeats=k)` | a covering grid over tools, world state and user stance; k rollouts per prompt; tool faults on a schedule | ch. 12, 13 |
| Grade | `data.grade(judge=)`, `verify.MathEqual`, `verify.CodeExec` | reward per rollout under one contract; verifiable rewards where the answer is checkable | ch. 5, 7 |
| Validate the judge | `judge_trust`, `judge_probes` | agreement and Cohen's kappa against human gold; length bias; exploit probes (filler, rubric echo, unbacked success claim) | ch. 5, 14 |
| Measure | `pass_at`, `delta_report`, `eval_variance`, `holdout_size` | pass@1, pass^k, pass@k with bootstrap intervals over tasks; paired before/after with a permutation p-value; re-run noise band; power | ch. 16, app. C |
| Select | `optimize(mode="rl"\|"sft")`, `build_preference_pairs`, `curriculum` | 20 to 80% difficulty band, unanimous-group drop, within-task dedupe, rejection sampling, length-matched pairs, easy-to-hard schedule | ch. 6, 7, 9, 11 |
| Guard | `decontaminate`, `hack_scan`, `trace_markers`, `HackMonitor` | 8-gram and semantic overlap with the eval set; within-task reward-feature correlation against a shuffle floor; trajectory lies (claimed tests, phantom edits) | ch. 14, 16 |
| Train and export | `export_dataset`, `export_environment`, `train`, `serve` | loss masks and unrolled turns; a `verifiers` environment for GRPO; hosted LoRA SFT, GRPO, DPO and reward-model runs | ch. 4, 6, 8 |

Chapters: 04 instruction tuning, 05 reward models, 06 policy gradients, 07
reasoning, 08 direct alignment, 09 rejection sampling, 11 preference data,
12 synthetic data, 13 tools, 14 over-optimization, 15 regularization, 16
evaluation, 17 character, appendix C practice.

## What the science looks like here

**Supervised fine-tuning.** `optimize(mode="sft")` is rejection sampling:
the highest-reward completion per prompt above `min_reward`, with random
selectors as the chance control (ch. 9). Exported rows carry a per-message
`loss_mask` (loss on agent turns only, never on tool output), `unroll=True`
turns an N-turn conversation into N samples each trained on the context it
had, and `format="trl"` is the shape `SFTTrainer` loads (ch. 4).

**RL with verifiable rewards.** A reward is a program where it can be:
`MathEqual`, `CodeExec` against hidden tests, `JSONSchema`, composed with
`All` and `Weighted` (ch. 7, 13). `mode="rl"` allocates rollouts
successively: two per prompt as a probe, filled to k only where the group
splits, because a unanimous group carries zero advantage (DAPO dynamic
sampling, ch. 6). `optimize(mode="rl")` keeps the 20 to 80% pass-rate band
with the interval on each task's rate, handles overlong rollouts by policy,
and `export_environment` writes the task set, world and reward as a
`verifiers` package for Prime Intellect or TRL. Rows carry sampled
logprobs for the importance ratio, `staleness_report` flags off-policy
rows, and `mean_kl` reads the drift from a reference (ch. 6, 15).

**Character training.** A constitution is a versioned object: `load_spec`
hashes its principles into `spec.version`, `stamp_spec` tags the rows a run
targeted, and the judge is checked against the spec's own labels before it
grades. Pairs are length-matched so the update learns the trait and not
the word count, and `delta_report(must_not_regress=spec.behaviors())` fails
the run that traded one trait for another (ch. 17).
[docs/character-training.md](docs/character-training.md).

**Evaluation.** Every pass@1 is a bootstrap over tasks, not rollouts;
`simulate(tasks=base, runs=3)` replays the same eval three times and
`delta_report` refuses a verdict inside twice the re-run standard
deviation. `holdout_size(effect, before=, after=)` reads the per-task paired
spread off a previous eval and says how many prompts prove a gain at 80%
power. `decontaminate` applies the Llama 2 8-gram rule plus task identity
and an optional embedding pass (ch. 16, app. C).
[docs/evals.md](docs/evals.md).

**Over-optimization.** `hack_scan` centers reward and every candidate
feature within task, ranks by correlation, and floors it against a
within-task shuffle, so it finds the delimiter or phrase the judge pays for.
`judge_probes` tries the exploits a policy finds first. `delta_report(proxy=,
target=)` fails when the training reward rose and the target did not.
`HackMonitor` runs the same scan inside a TRL loop and can stop the run
(ch. 14). [docs/reward-hacking.md](docs/reward-hacking.md).

## Recipes

One post-training run as five steps, each a runnable script with what you
learn, what it needs, and how long it takes. Offline recipes take seconds;
CI runs every one on every pull request.

| Step | Recipes |
|---|---|
| [01-simulate](recipes/01-simulate) | bring your own agent, verifiers, a coding agent traced to the platform |
| [02-measure](recipes/02-measure) | eval your agent, pass@k, is your eval any good, reward hacking, safety evals |
| [03-select](recipes/03-select) | the row schema, GRPO data with a gradient gate, character training |
| [04-train](recipes/04-train) | hosted loop, identity SFT, GRPO and DPO on Modal, text-to-SQL hill climb |
| [05-export](recipes/05-export) | Hugging Face datasets and adapters |
| [papers](recipes/papers) | one recent paper per recipe, one change to a step, the number it moved with its interval |

Index: [recipes/README.md](recipes/README.md). `whileai init-evals` scaffolds
the eval recipe around the agent it finds in your project.

## Models

Any OpenAI-compatible chat endpoint that returns tool calls can play the
agent, write the situations, play the user, or judge. A spec names the
backend and the model and works wherever one is accepted.

```python
data = wai.simulate(
    agent="anthropic:claude-haiku-4-5",  # or openai:<model>, vllm:<model>@<url>, ollama:<model>
    simulator="anthropic:claude-sonnet-5",  # the situation writer
    tools=TOOLS,
    system_prompt=POLICY,
    output="rollout.jsonl",
)
```

With no `agent=`, the run uses While-hosted Qwen on your account key
(`whileai login`), with Phi-4 as the judge so the judge is never the policy.

## Platform

Datasets, training runs and served adapters on [zeroproofai.com](https://zeroproofai.com/platform),
from the same objects.

```python
v1 = data.push("refunds-v1", holdout=0.2, gate=True)  # gated: refuses gradient-free RL data
run = wai.train(v1["datasetId"], method="grpo", steps=200)  # sft | grpo | dpo | rm
run.wait()
model = wai.serve("refunds-v2", run)  # OpenAI-compatible endpoint
```

Your own trainer reports into the same run page through `wai.TrainerCallback`
or `wai.training_run(...)`. Traces in production come back as `traces=`,
which aims the next run's grid at the situations that failed.

## Documentation

| Page | What it covers |
|---|---|
| [docs/reference.md](docs/reference.md) | every call, knob, report and gate, in the order a run happens |
| [docs/engine.md](docs/engine.md) | how a row is made: the draw, the grid, the search arms, the rollout, the split |
| [docs/simulations.md](docs/simulations.md) | why the engine is shaped this way |
| [docs/evals.md](docs/evals.md) | pass rates with intervals and a CI gate for an agent you already have |
| [docs/reward-hacking.md](docs/reward-hacking.md) | before, during and after training |
| [docs/safety-evals.md](docs/safety-evals.md) | injection, exfiltration, unauthorized writes, over-refusal controls |
| [docs/character-training.md](docs/character-training.md) | constitution to graded rows to length-matched pairs |
| [CHANGELOG.md](CHANGELOG.md) | one entry per release, with the pull request |

The row schema is `Task`, `Rollout`, `Judgment`, `Marker`
(`whileai/simulations/schemas/row-v1.json`); every training target is a
projection of those four.

## Development

```bash
uv sync --extra dev
uv run pytest             # about two minutes, no network
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

CI runs the suite on Python 3.10 through 3.13, lint and types, line
coverage gated at 90%, every recipe's `smoke.sh`, a plain-pip install of
the built wheel into a clean venv, and the version gate. Contributions:
[CONTRIBUTING.md](CONTRIBUTING.md).

## Cite

```bibtex
@software{whileai,
  title  = {whileai: post-training data and evaluation for tool-using agents},
  author = {{While}},
  year   = {2026},
  url    = {https://github.com/whilehq/whileai-sdk}
}
```

The methods follow Lambert, N. (2025). *Reinforcement Learning from Human
Feedback*. arXiv:2504.12501. [rlhfbook.com](https://rlhfbook.com).

## License

Apache-2.0
