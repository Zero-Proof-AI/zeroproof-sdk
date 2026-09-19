# X thread: endpoint-sft

Written from `results.json` and the README's Result table (verified 2026-09-17).

1/ We reproduced Endpoint SFT (Jaehui Hwang et al., arXiv:2609.07103) at small scale: Qwen2.5-1.5B-Instruct, one L40S, 75 SFT steps per arm. pass@1: 0.29 (full trace) vs 0.28 (endpoints only), delta -0.012 [-0.074, +0.047]. Verdict: flat.

2/ The claim: the middle of a machine-written reasoning trace is weakly attended; train on the first and last steps only (about 20% of tokens removed) and match full-trace SFT. The one change: the assistant trace is cut to its first n and last n steps. Nothing else changes.

3/ How we know what we know: the untrained base was evaluated 3 times, the holdout was decontaminated, both arms scored on the same 64 held-out problems, paired 95% interval. Round 1 was set by the eval's token budget, not the training; round 2 is the headline.

4/ What to watch: flat is consistent with "matches", with no sign of "beats" at this size. Both arms scored under the untrained base (0.46) on this budget. The endpoint arm trained in 21.7 GPU minutes against 47.7: equal score at half the cost is the line worth a longer run.

5/ Cost: $2.31, about 69 GPU minutes. Rerun it yourself:
pip install whileai && cd recipes/papers/endpoint-sft && python recipe.py
https://github.com/whilehq/whileai-sdk/tree/main/recipes/papers/endpoint-sft
