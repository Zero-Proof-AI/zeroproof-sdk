<p align="center">
  <a href="https://while.ai">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/whilehq/whileai-sdk/main/docs/assets/wordmark-dark.png">
      <img src="https://raw.githubusercontent.com/whilehq/whileai-sdk/main/docs/assets/wordmark-light.png" alt="while" width="300">
    </picture>
  </a>
</p>

<p align="center"><code>MID-TRAINING AND POST-TRAINING FOR LANGUAGE MODELS</code></p>

<p align="center">
  <a href="https://github.com/whilehq/whileai-sdk/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/whilehq/whileai-sdk/ci.yml?branch=main&label=ci&labelColor=0b1220&color=5cb08a" alt="CI"></a>
  <a href="https://pypi.org/project/whileai/"><img src="https://img.shields.io/pypi/v/whileai?labelColor=0b1220&color=5cb08a" alt="PyPI"></a>
  <a href="https://pypi.org/project/whileai/"><img src="https://img.shields.io/pypi/pyversions/whileai?labelColor=0b1220&color=3f8f6b" alt="Python"></a>
  <a href="https://pypistats.org/packages/whileai"><img src="https://img.shields.io/pypi/dm/whileai?labelColor=0b1220&color=3f8f6b" alt="Downloads"></a>
  <a href=".github/workflows/ci.yml"><img src="https://img.shields.io/badge/coverage-%E2%89%A5%2090%25%20gated-5cb08a?labelColor=0b1220" alt="Coverage gate"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-3f8f6b?labelColor=0b1220" alt="License"></a>
</p>

Post-training data and evaluation for tool-using language-model agents.
`whileai` simulates the situations an agent can meet, grades every rollout
under one judge contract, and turns graded rows into SFT, preference and
RL data with the checks the literature calls for. Each method cites its
source in [References](#references).

```bash
uv add whileai
```

Or `pip install whileai`. Python 3.10 to 3.13, one dependency, typed.
Formerly `zeroproof`; the old name still installs this package.

## Two ways in

**Evals and the harness, no training.** A pass rate with an interval, a
table of where the agent fails, and a check that turns red in CI.
`whileai init-evals` writes the harness around the agent it finds in your
project, `coverage_gap` names what your tests never reach, and
`compare_runs` says whether a prompt or tool edit helped on the same pinned
tasks. Start at [docs/evals.md](docs/evals.md).

**Post-training.** The same graded rows, selected and exported: the loop below.

## Sixty seconds, offline

No key, no network. The seeded agent misbehaves on a labeled fraction of
rollouts, so a judge that catches exactly those rows is a judge that works.

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
    simulator=False,  # no model
    mode="rl",
    repeats=4,
    repeat_policy="fixed",
    budget=64,
)
scored = data.grade(judge=lambda row: {"reward": int(not row["seeded"])})
print(scored.pass_at)
```

```
pass@1 0.67 [0.55..0.78] | pass^4 (pass_pow_k) 0.19 [0.00..0.38] | pass@4 1.00 [1.00..1.00] | headroom 0.33 (16 groups, k=4)
```

Your agent is a callable `agent(message) -> {"steps", "final_text"}` or a
model spec: `openai:<model>`, `anthropic:<model>`, `vllm:<model>@<url>`,
`ollama:<model>`. With no `agent=`, hosted Qwen runs on your key
(`whileai login`) and Phi-4 judges, so the judge is never the policy.

## The loop

| Step | Call | What it computes | Refs |
|---|---|---|---|
| Simulate | `simulate(agent, tools=, system_prompt=, mode="rl", repeats=k)` | covering array over tools, world state and user stance; k rollouts per prompt; scheduled tool faults | [2], [3] |
| Grade | `data.grade(judge=)`, `verify.MathEqual`, `verify.CodeExec` | reward per rollout under one contract; verifiable rewards | [4], [5] |
| Validate the judge | `judge_trust`, `judge_probes` | agreement and Cohen's kappa against human gold; length bias; exploit probes | [6], [7] |
| Measure | `pass_at`, `delta_report`, `eval_variance`, `holdout_size` | pass@1, pass^k, pass@k with bootstrap intervals over tasks; paired delta with a permutation p-value; noise band; power | [8], [9], [10], [11] |
| Select | `optimize(mode="rl"\|"sft")`, `build_preference_pairs`, `curriculum` | 20 to 80% difficulty band, unanimous-group drop, rejection sampling, length-matched pairs, curriculum | [12], [13], [14], [15] |
| Guard | `decontaminate`, `hack_scan`, `trace_markers`, `HackMonitor` | overlap with the eval set; reward-feature correlation within task against a shuffle floor; trajectory lies | [16], [17], [18] |
| Train and export | `export_dataset`, `export_environment`, `train`, `serve` | loss masks; a `verifiers` environment for GRPO; hosted LoRA SFT, GRPO, DPO, RM | [1], [19], [20] |

## The science

**Supervised fine-tuning.** `optimize(mode="sft")` is rejection sampling
[14], [16] with random selectors as the chance control. Rows carry a
per-message `loss_mask`, `unroll=True` trains each turn on the context it
had, and `format="trl"` is what `SFTTrainer` loads [1, ch. 4].

**RL with verifiable rewards.** A reward is a program where it can be [5].
`mode="rl"` probes each prompt twice and fills to k only where the group
splits: a unanimous group has zero advantage under a group-relative
baseline [19] (dynamic sampling [12]). `optimize(mode="rl")` keeps the 20
to 80% band [13] and handles overlong rollouts by policy [12].
`export_environment` writes tasks, world and reward as a `verifiers`
package. Rows carry logprobs for the importance ratio [21] and `mean_kl`
reads drift from a reference [22].

**Character training.** A constitution is a versioned object [23], [24]:
`load_spec` hashes its principles into `spec.version`, the judge is checked
against the spec's own labels, pairs are length-matched [7], and
`must_not_regress=spec.behaviors()` fails a run that traded one trait for
another. [docs/character-training.md](docs/character-training.md).

**Evaluation.** pass@1 is a bootstrap over tasks, not rollouts [8], [10],
[11]. `runs=3` replays an eval and `delta_report` refuses a verdict inside
twice the re-run standard deviation. `holdout_size` returns the prompts a
gain needs at 80% power [11]. `decontaminate` applies the 80% n-gram
coverage rule [16] and an optional embedding pass.
[docs/evals.md](docs/evals.md).

**Over-optimization.** Reward is a proxy and a strong optimizer finds the
gap [17]. `hack_scan` ranks reward-feature correlation within task against
a shuffle floor; `judge_probes` tries the exploits a policy finds first,
sycophancy included [18]; `delta_report(proxy=, target=)` fails when the
proxy rose and the target did not; `HackMonitor` runs the scan inside a
TRL loop. [docs/reward-hacking.md](docs/reward-hacking.md).

## Recipes

One post-training run as five steps; every recipe runs in CI.

| Step | Recipes |
|---|---|
| [01-simulate](recipes/01-simulate) | bring your own agent, verifiers, a traced coding agent |
| [02-measure](recipes/02-measure) | eval your agent, pass@k, reward hacking, safety evals |
| [03-select](recipes/03-select) | the row schema, GRPO data with a gradient gate, character |
| [04-train](recipes/04-train) | hosted loop, identity SFT, GRPO and DPO on Modal, text-to-SQL |
| [05-export](recipes/05-export) | Hugging Face datasets and adapters |
| [papers](recipes/papers) | one recent paper per recipe, the number it moved with its interval |

## Platform

```python
v1 = data.push("refunds-v1", holdout=0.2, gate=True)  # refuses gradient-free RL data
run = wai.train(v1["datasetId"], method="grpo", steps=200)  # sft | grpo | dpo | rm
run.wait()
model = wai.serve("refunds-v2", run)  # OpenAI-compatible endpoint
```

Your own trainer reports through `wai.TrainerCallback`; production traces
come back as `traces=` and aim the next run at what failed.

## Documentation

[docs/reference.md](docs/reference.md): every call, knob, report and gate.
[docs/engine.md](docs/engine.md): how a row is made.
[CHANGELOG.md](CHANGELOG.md): one entry per release.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check . && uv run mypy
```

CI gates coverage at 90% and runs every recipe's `smoke.sh`.
[CONTRIBUTING.md](CONTRIBUTING.md).

## Cite

```bibtex
@software{weiss2026whileai,
  title  = {whileai: post-training data and evaluation for tool-using agents},
  author = {Weiss, Jacob},
  year   = {2026},
  url    = {https://github.com/whilehq/whileai-sdk}
}
```

## References

1. Lambert, N. *Reinforcement Learning from Human Feedback*. arXiv:2504.12501, 2025.
2. Kuhn, D. R., Wallace, D. R., Gallo, A. M. Software Fault Interactions and Implications for Software Testing. *IEEE TSE* 30(6), 2004.
3. Yao, S. et al. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045, 2024.
4. Ouyang, L. et al. Training Language Models to Follow Instructions with Human Feedback. NeurIPS, 2022.
5. Lambert, N. et al. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124, 2024.
6. Cohen, J. A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement* 20(1), 1960.
7. Zheng, L. et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS, 2023.
8. Chen, M. et al. Evaluating Large Language Models Trained on Code. arXiv:2107.03374, 2021.
9. Wilson, E. B. Probable Inference, the Law of Succession, and Statistical Inference. *JASA* 22(158), 1927.
10. Efron, B., Tibshirani, R. J. *An Introduction to the Bootstrap*. Chapman & Hall, 1993.
11. Miller, E. Adding Error Bars to Evals. arXiv:2411.00640, 2024.
12. Yu, Q. et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476, 2025.
13. He, J. et al. Skywork Open Reasoner 1 Technical Report. arXiv:2505.22312, 2025.
14. Yuan, Z. et al. Scaling Relationship on Learning Mathematical Reasoning with Large Language Models. arXiv:2308.01825, 2023.
15. Rafailov, R. et al. Direct Preference Optimization. NeurIPS, 2023.
16. Touvron, H. et al. Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288, 2023.
17. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML, 2023.
18. Sharma, M. et al. Towards Understanding Sycophancy in Language Models. ICLR, 2024.
19. Shao, Z. et al. DeepSeekMath. arXiv:2402.03300, 2024.
20. DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, 2025.
21. Schulman, J. et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.
22. Ziegler, D. M. et al. Fine-Tuning Language Models from Human Preferences. arXiv:1909.08593, 2019.
23. Bai, Y. et al. Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073, 2022.
24. OpenAI. Model Spec, 2024. model-spec.openai.com.

## License

Apache-2.0
