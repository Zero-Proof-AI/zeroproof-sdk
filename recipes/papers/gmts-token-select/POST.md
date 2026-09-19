# X thread: gmts-token-select

Written from `results.json` and the README's Result table (verified 2026-09-18).

1/ We reproduced GMTS (Outongyi Lv et al., arXiv:2608.30632) at small scale: Qwen2.5-1.5B-Instruct, one L40S, 40 GRPO steps per arm, top 20% of tokens. pass@1: 0.49 (by entropy) vs 0.18 (by entropy x advantage), delta -0.310 [-0.383, -0.235]. Verdict: flat, on the wrong side.

2/ The claim: ranking tokens by entropy alone ignores how the group scored the answer; ranking by entropy times advantage picks a better 20% and raises accuracy. The one change: the score the top 20% is taken by.

3/ How we know what we know: the untrained base was evaluated 3 times, the holdout was decontaminated, both arms scored on the same 120 held-out tasks, paired 95% interval. The interval excludes zero on the wrong side, so check.py calls it flat: a loss is not a gain.

4/ What to watch: the abstract gives no learning rate, LoRA rank, step count or model size; 1e-4, r=32, 40 steps and 1.5B are ours. This is the strongest candidate for "we did not run the paper's setting". A sweep over those four is the next round.

5/ Cost: $1.51, about 45 GPU minutes. Rerun it yourself:
pip install whileai && cd recipes/papers/gmts-token-select && python recipe.py
https://github.com/whilehq/whileai-sdk/tree/main/recipes/papers/gmts-token-select
