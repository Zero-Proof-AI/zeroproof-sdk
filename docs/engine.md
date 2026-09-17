# The engine on one page

How `simulate()` makes evals and training data. Combinatorial coverage of
situations, a sandbox world with failure modes, and a judge validated
before training. The longer read is [simulations.md](simulations.md); the
PDF of this page is at
[while.ai/while-simulation-engine.pdf](https://while.ai/while-simulation-engine.pdf).

## Eight steps

| # | Step | What happens | Code |
|---|------|--------------|------|
| 01 | Axes | Declare what varies: tool, policy clause, world state, fault, persona, history. A situation is a point in that space, not a prompt. | `generate/scenarios.py` |
| 02 | Cover | Plan cells so every pair of axis values co-occurs at least once (a pairwise covering array). Most failures are two-factor interactions [6]. `data.coverage["pairwise"]` is planned pairs, covered pairs, and the fraction. | `generate/coverage.py` |
| 03 | Search | Five situation writers fill the grid. Each batch, weights move toward the arms that produced new behavior signatures: `w' = w(1 + 0.5 * yield)`, renormalized, with variety floors and rare caps. Novelty search, not importance sampling [7]. Stops at saturation. | `generate/scenarios.py`, `generate/diversity.py` |
| 04 | World | Tools answer from schema-shaped state. Deterministic per seed. Unknown id: not found. An argument that echoes the schema instead of the customer: refused with a hint. | `world/sandbox.py` |
| 05 | Rollout | Run the agent on N tasks x n phrasings x k samples. With `logprobs=True` every row keeps the policy's per-token log-probabilities, token count, policy version and temperature. | `run/engine.py` |
| 06 | Grade | Deterministic conduct rules first, then your judge. The judge is scored against gold labels before its grades are trusted. | `score/grading.py`, `score/judge_trust.py` |
| 07 | Cut | SFT rows (reward=1, loss mask on agent turns), DPO pairs with margin, GRPO groups in the 20 to 80 percent band [4, 5], or a reward-model set. | `score/optimize.py`, `score/publish_gate.py`, `export.py` |
| 08 | Delta | Re-run held-out tasks after training. A paired difference per task with a bootstrap interval and a sign-flip permutation p [1]. | `score/delta.py`, `score/stats.py` |

## What is ours

- **Covering array.** Every pair of axis values appears together at least
  once. Most failures are two-factor interactions [6].
- **Arm search.** `structured`, `llm_guided`, `open_ended`,
  `behavior_targeted`, `failure_mutation`, weighted each batch by the yield
  of new behavior signatures. Novelty search, not importance sampling [7].
- **Sandbox world.** Built from the tool JSON schemas. Deterministic per
  seed. Unknown id: not found. Schema-echo argument: refused.
- **Cuts.** SFT: reward=1 rows, loss mask on agent turns. DPO: pairs with
  margin and length gap. GRPO: mixed groups, 20 to 80 percent band [4, 5].
  RM: all graded rows.

## What we take from the literature

- **pass@1 / pass^k / pass@k.** Headline, reliability, RL headroom.
  Unbiased combinatorial estimators over k samples per task [2, 3].
  `score/passat.py`.
- **Intervals.** Bootstrap over tasks, not rollouts. Before and after is a
  paired difference with a sign-flip permutation p [1]. Ship when the
  interval excludes zero. `score/stats.py`.
- **Judge.** Agreement and kappa against gold labels, Wilson interval,
  split-half, length perturbation, probes. Different model family than the
  policy [8]. Rubric hash on every label. `score/judge_trust.py`.
- **Hack scan.** `Var(r) = E[Var(r | task)] + Var(E[r | task])`. Only the
  first term is GRPO gradient. The top within-task feature is compared to
  a permutation floor from reward shuffled within task [9].
  `score/hack_scan.py`.

## Questions

**Do you use importance sampling?** No. Importance sampling corrects an
estimator for a wrong proposal distribution; we are not estimating
production, we are covering the failure space. Each row keeps per-token
logprobs, policy version and temperature, so an asynchronous trainer can
form the truncated ratio `exp(log pi_new - log pi_old)` itself [10, 11].
`score/logprobs.py` and `score/reference.py` score the same tokens under a
reference model for the KL side.

**SFT or RL?** Both, from the same graded rows. `select_for_sft` takes
reward=1 rows deduplicated by behavior shape, with a loss mask on agent
turns. `preference_pairs` takes pairs. `select_for_rl` takes whole groups.
A reward model takes all graded rows.

**Isn't the judge just another LLM?** Yes. So it is measured against gold
labels (`judge_agreement`), probed with known hacks (`judge_trust`),
versioned by rubric hash, and drawn from a different model family than
the policy [8].

**How do you know training helped?** `delta_report`: paired before and
after on held-out tasks with a bootstrap interval. Markers such as
`argument_grounding` catch regressions pass@1 hides; a `must_not_regress`
marker whose interval sits below zero fails the run [1].

## References

1. Lambert, N. *Reinforcement Learning from Human Feedback*. 2025. rlhfbook.com.
2. Chen, M. et al. Evaluating Large Language Models Trained on Code. arXiv:2107.03374, 2021.
3. Yao, S. et al. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045, 2024.
4. Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
5. Yu, Q. et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476, 2025.
6. Kuhn, D. R., Wallace, D. R., Gallo, A. M. Software Fault Interactions and Implications for Software Testing. *IEEE Transactions on Software Engineering* 30(6), 2004.
7. Lehman, J., Stanley, K. O. Abandoning Objectives: Evolution Through the Search for Novelty Alone. *Evolutionary Computation* 19(2), 2011.
8. Zheng, L. et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS, 2023.
9. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML, 2023.
10. Schulman, J. et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.
11. Noukhovitch, M. et al. Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models. ICLR, 2025.

Code paths are relative to `whileai/simulations/`.
