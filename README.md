<p align="center">
  <a href="https://withwhile.com">
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
  <a href="https://pepy.tech/project/whileai"><img src="https://img.shields.io/pepy/dt/whileai?labelColor=0b1220&color=3f8f6b" alt="Downloads"></a>
  <a href=".github/workflows/ci.yml"><img src="https://img.shields.io/badge/coverage-%E2%89%A5%2090%25%20gated-5cb08a?labelColor=0b1220" alt="Coverage gate"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-3f8f6b?labelColor=0b1220" alt="License"></a>
</p>

`whileai` makes training and eval data for agents that call tools. Give it
an agent, or just the agent's tools and system prompt. It writes the
situations the agent might meet, runs the agent through them against a fake
world that fails on purpose, and hands back every conversation as a row.
You grade the rows with your own judge or a verifier. The package then does
the bookkeeping that is easy to skip and expensive to get wrong: pass rates
with intervals, difficulty bands for RL, a check that your judge agrees with
people, decontamination against your eval set, and a scan for rewards the
policy can game. Every method says where it comes from
([References](#references)).

```bash
uv add whileai
```

Or `pip install whileai`. Python 3.10 to 3.13, one dependency, typed.
This package used to be called `zeroproof`; that name still installs it.

## Two ways in

**You only want evals.** Plenty of teams cannot train and still need to
know whether the last prompt edit helped. Run `whileai init-evals` in your
project. It finds your agent, writes a judge and a runner around it, and
gives you a pass rate with a 95% interval, a table of where the agent
fails, and a test that goes red in CI when it gets worse. `coverage_gap`
tells you which situations your tests never reach. `compare_runs` reruns
the same tasks after a prompt or tool change and says whether the change
helped. Start at [docs/evals.md](docs/evals.md).

**You want to train.** Grade the same rows, keep the ones that carry
signal, export. That is the rest of this page.

## Sixty seconds, offline

No key, no network. `seeded_agent` is a stand-in agent. It answers
honestly most of the time and, on a labeled fraction of rollouts, does one
thing wrong on purpose: hedges, flatters, or claims success after a tool
failed. Each row records what it did in `seeded`, so you can check that
your judge catches exactly those rows before you trust it on real ones.

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

pass@1 is the pass rate over tasks with a bootstrap interval. pass^4 is
how often all four rollouts of a task pass. Headroom is pass@4 minus
pass@1, the gap an RL update could close.

To use your own agent, pass any callable that takes the user message and
returns `{"steps": [...], "final_text": "..."}`. To use a model, pass a
spec string: `openai:<model>`, `anthropic:<model>`, `vllm:<model>@<url>`,
or `ollama:<model>`. With no `agent=` at all, the run uses the Qwen we
host, on your key from `whileai login`, and Phi-4 grades. The judge is
never the model it is judging.

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

**SFT.** `optimize(mode="sft")` is rejection sampling [14], [16]: keep the
best-scoring completion for each prompt, with a random selector alongside
so you can tell whether picking the best did anything. Exported rows carry
a `loss_mask` per message, so the trainer learns from the agent's turns and
not from tool output. `unroll=True` splits a long conversation into one
sample per agent turn, each with the context that turn actually saw.
`format="trl"` is the shape `SFTTrainer` loads [1, ch. 4].

**RL with verifiable rewards.** When a program can check the answer, the
reward should be that program [5]: `MathEqual`, `CodeExec` against hidden
tests, `JSONSchema`, and combinations of them. In `mode="rl"` every prompt
gets two rollouts first. Only prompts where those two disagree are filled
to k, because a group that all passes or all fails has zero advantage under
GRPO [19]. That is DAPO's dynamic sampling [12], applied while the rollouts
are generated instead of after. `optimize(mode="rl")` then keeps the
prompts the policy solves 20 to 80% of the time [13] and lets you choose
what happens to rollouts that hit the length cap [12]. `export_environment`
writes the tasks, the fake world and the reward as a `verifiers` package
you can hand to a trainer. Rows keep their sampling logprobs so the trainer
can form the importance ratio [21], and `mean_kl` measures drift from the
reference model [22].

**Character training.** Write down how the model should talk as a
constitution [23], [24]. `load_spec` hashes it into `spec.version`, so an
edit to one principle is a new version. The judge is checked against the
labels the spec itself carries before it grades anything. Preference pairs
are matched on length [7], so the model learns the trait and not "longer
is better". Put `spec.behaviors()` in `must_not_regress` and
`delta_report` fails any run that improved one trait by giving up
another. [docs/character-training.md](docs/character-training.md).

**Evaluation.** Intervals are bootstrapped over tasks, not rollouts,
because rollouts of the same task are not independent [8], [10], [11].
Run an eval three times with `runs=3` and `delta_report` refuses to call a
change real when it sits inside twice the run-to-run standard deviation.
`holdout_size` says how many prompts you need to see a given gain at 80%
power [11]; most evals are too small. `decontaminate` checks training rows
against the eval set with the 80% n-gram overlap rule [16], and with
embeddings when you pass an embedder. [docs/evals.md](docs/evals.md).

**Over-optimization.** The reward is a proxy for what you want, and RL
finds the gap between the two [17]. `hack_scan` looks for the feature that
predicts reward within a task, against a shuffled baseline, so a judge that
pays for a phrase or a delimiter shows up before you train on it.
`judge_probes` tries the tricks a policy finds first, flattery included
[18]. `delta_report(proxy=, target=)` fails when the training reward went
up and the metric you care about did not. `HackMonitor` runs the same scan
inside a TRL training loop and can stop it.
[docs/reward-hacking.md](docs/reward-hacking.md).

## Recipes

Each recipe is one script and a README that says what you learn, what you
need, and how long it takes. All of them run in CI.

| Step | Recipes |
|---|---|
| [01-simulate](recipes/01-simulate) | bring your own agent, verifiers, a traced coding agent |
| [02-measure](recipes/02-measure) | eval your agent, pass@k, reward hacking, safety evals |
| [03-select](recipes/03-select) | the row schema, GRPO data with a gradient gate, character |
| [04-train](recipes/04-train) | hosted loop, identity SFT, GRPO and DPO on Modal, text-to-SQL |
| [05-export](recipes/05-export) | Hugging Face datasets and adapters |
| [papers](recipes/papers) | one recent paper per recipe, the number it moved with its interval |

## Platform

Push a graded run to your While account, train on it, serve the result.
`push` refuses RL data with no mixed groups, since a trainer would learn
nothing from it.

```python
v1 = data.push("refunds-v1", holdout=0.2, gate=True)
run = wai.train(v1["datasetId"], method="grpo", steps=200)  # sft | grpo | dpo | rm
run.wait()
model = wai.serve("refunds-v2", run)  # OpenAI-compatible endpoint
```

If you train with your own code, `wai.TrainerCallback` reports into the
same run page. Traces from production come back through `traces=`, which
points the next simulation at the situations that failed.

Already ran your own eval? Send the rows you scored and the platform groups
them into prompts, so you can see what is worth training on.

```python
whileai.send_runs(rows, agent="refunds")   # {scenario_id, prompt, final_text, reward}
print(wai.format_cuts(wai.cuts("refunds"), agent="refunds"))
```

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

CI runs the suite on Python 3.10 to 3.13, gates coverage at 90%, and runs
every recipe's `smoke.sh`. [CONTRIBUTING.md](CONTRIBUTING.md).

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
