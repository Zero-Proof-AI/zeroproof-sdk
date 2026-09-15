# Examples, in the order a post-training run happens

One directory per step. Each README says what you will learn and what you
need before the first command. Every example that simulates needs a key
(`zeroproof login`): the situations are model-written, there is no offline
writer. A scripted agent stands in for the policy where the point is the loop.
a laptop unless a GPU is named.

| Step | Example | What it teaches | Needs | Takes |
|---|---|---|---|---|
| Simulate and grade | [`bring-your-own-agent`](bring-your-own-agent) | the `agent(message) -> {steps, final_text}` contract, what a run says when the agent raises, and why an `evaluate()` score must not become the reward | a key (`zeroproof login`) | seconds |
| Simulate and grade | [`agent-behavior`](agent-behavior) | a coding agent with instructed bad habits, every turn on the platform as OTLP spans, a held-out test suite and an LLM judge disagreeing about the same turn, rows grouped by `scenario_id` for RL | `ZEROPROOF_API_KEY` and an OpenAI-compatible model endpoint (`--dry-run` needs neither) | 8 min for 40 runs |
| Simulate and grade | [`verifiers`](verifiers) | rewards that are programs: `MathEqual`, `All` (answer and format), `CodeExec` against hidden tests, `JSONSchema`, each honoring the judge contract | a key (`zeroproof login`) | seconds |
| Measure | [`pass-at-k`](pass-at-k) | pass@1 with its interval, pass^k, pass@k, the per-ask histogram the mean hides, and headroom = what a grouped update can learn | a key (`zeroproof login`) | seconds |
| Measure | [`reward-hacking`](reward-hacking) | reward hacking caught before, during and after training: the within-ask scan, the judge probes, the trajectory flags, the proxy-vs-target verdict, on a scripted agent and two judges | a key (`zeroproof login`) | seconds |
| Measure | [`safety-evals`](safety-evals) | a safety suite for a tool-using agent: prompt injection (direct and planted in a tool result), data exfiltration, secret leakage, unauthorized writes, and the benign controls; four trajectory markers as the judge, pass^k per attack class, the judge checked against hand labels and the refusal probe, and a before/after that fails the fix which got safe by refusing | a key (`zeroproof login`) | seconds |
| Measure | [`safety-evals-marketplace`](safety-evals-marketplace) | the same eval where the untrusted text is user-generated content and the private data is per tenant: injections planted in product reviews, a competitor's buyer-intent list the agent must not read, intent data in a public response, a flag without a moderation ticket; six trajectory markers, pass^k per class, the guarded before/after, and `live.py` to run the suite on a local model through Ollama | nothing offline; Ollama for `live.py` | seconds offline, minutes live |
| Select | [`schema`](schema) | one row file projected into eval, SFT, preference, GRPO prompts, OPSD and OPD targets; the `Task`/`Rollout`/`Judgment`/`Marker` split that makes that possible | a key (`zeroproof login`) | seconds |
| Select | [`prime-intellect-rl`](prime-intellect-rl) | `simulate(mode="rl")` for uniform groups, the gradient gate (`diagnose.py`) that catches a reward the policy can game before you train, prompts in the `verifiers` shape | an account key (`zeroproof login`); `VLLM_API_KEY` for the shared pool | 3 min for 800 rollouts on the shared pool |
| Select | [`character`](character) | a constitution to traits, graded replies per trait, a judge checked against the spec's own labels, length-matched pairs and masked SFT rows, before/after on an adversarial holdout | a key (`zeroproof login`) | seconds offline, 2.5 min live |
| Train | [`hosted-loop`](hosted-loop) | push graded rows, `zps.train` SFT on Qwen3-4B, `zps.serve` the adapter, one chat completion from the endpoint | `ZEROPROOF_API_KEY` | about a minute of A10G, plus a cold start |
| Train | [`identity`](identity) | a leak-free SFT set that teaches a name and maker, with Modal scripts for the LoRA and for the identity/leak eval | nothing to generate; Modal and an A10G to train | seconds to generate |
| Train | [`grpo`](grpo) | TRL `GRPOTrainer` with LoRA on a verifiable rule, `HackMonitor` and reward/KL on the run page, paired pass@1 before/after with per-category deltas, loss variants and `--balance` as flags | Modal, one A10G; the key is optional | under 15 min at 40 steps |
| Train | [`dpo`](dpo) | on-policy pairs from `build_preference_pairs`, TRL `DPOTrainer`, the reward margin on the run page, iterated rounds with `--from-run`, constructed negatives | Modal, one A10G; the key is optional | about 10 min |
| Evaluate and trust | [`character/measure.py`](character#run-it) | `delta_report` with a target marker, `must_not_regress` guards, a 95% interval per metric | nothing (`--demo`) | seconds |
| Export | [`hugging-face`](hugging-face) | rows to a Hub dataset repo (one split per purpose, commit tagged by dataset id), any Hub split onto the account with a profile, a run's adapter to a model repo | `ZEROPROOF_API_KEY` and a Hugging Face account connected on the platform | a minute |

Where the pieces of the main README live in these examples:

- **Simulate and grade:** `bring-your-own-agent` (callable), `agent-behavior` (traces), `verifiers` (program as reward). Grading with the hosted judge or your own callable is `data.grade(judge=...)`; every example above uses a callable so it runs without a key.
- **pass@k and headroom:** `pass-at-k`. The same `PassAt` object is `data.pass_at`, `ScoredData.pass_at`, and the per-trait lines in `character`.
- **Data for SFT, pairs for DPO, groups for GRPO:** `schema` (all six projections from one file), `prime-intellect-rl` (RL groups and the gate), `character` (pairs and SFT rows from graded replies), `dpo` (pairs from the policy being trained).
- **Train:** `hosted-loop` (platform trainer, no GPU of yours), `identity`, `grpo`, `dpo` (your trainer on Modal, reporting into the same run page through `zps.TrainerCallback`).
- **Before and after:** `grpo` and `dpo` call `run.delta(before, after, by="category")`; `character/measure.py` and `safety-evals` call `delta_report` directly, the latter with `must_not_regress=["helpful_on_benign"]` so a fix that got safe by refusing fails. Every delta is a paired bootstrap over tasks with a 95% interval; `within_noise` needs `eval_variance` from three re-runs, which no example runs yet (see the gaps in the pull request that added this index).
- **Trust checks:** judge trust against gold labels in `character` (`judge_vs_spec`), `safety-evals` (hand-labeled transcripts, the refusal probe on the benign controls) and `agent-behavior` (held-out suite vs judge); reward hacking in `prime-intellect-rl` (effort correlation), `grpo` (`HackMonitor`) and `dpo` (constructed negatives); the `evaluate()` provenance guard in `bring-your-own-agent`. Decontamination (`zps.decontaminate`) and eval variance (`zps.eval_variance`) have no example; the README's trust section shows the calls.
- **Export:** `hugging-face` (Hub), `prime-intellect-rl/export_prompts.py` (the `verifiers` prompt shape), `schema/project.py` (JSONL per target).

RLHF-book chapter numbers in these READMEs were checked against the book's
chapter files on 2026-09-14 (ch. 3 training overview, 4 instruction tuning,
5 reward modeling, 6 reinforcement learning, 7 reasoning, 8 direct
alignment, 9 rejection sampling, 11 preference data, 12 synthetic data and
Constitutional AI, 13 tool use, 14 over-optimization, 15 regularization, 16
evaluation, 17 model character and products).
