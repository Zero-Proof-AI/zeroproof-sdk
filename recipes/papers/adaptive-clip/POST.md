# X thread: adaptive-clip

Written from `results.json` and the README's Result table (verified 2026-09-18).

1/ We reproduced Group Adaptive Clipping (Sheng Jia et al., arXiv:2609.00444) at small scale: Qwen2.5-1.5B-Instruct, one L40S, 40 GRPO steps per arm. pass@1: 0.47 -> 0.52, delta +0.050 [0.000, +0.100]. Verdict: flat. The interval touches zero.

2/ The claim: one fixed upper clip bound holds back the lone correct answer in a hard group as much as the seventh in an easy one; widening it as correct answers get rarer lifts pass@1. The one change: the upper bound is set per group from how many rollouts were right.

3/ How we know what we know: the untrained base was evaluated 3 times for the noise floor, the holdout was decontaminated, both arms scored on the same 120 held-out tasks, 4 samples each, paired 95% interval. Round 1 ran the clip where it could not bind; round 2 is the headline.

4/ What to watch: pass@k moved the same way (0.70 -> 0.74) and the recipe arm took 7.6 GPU minutes against 12.8 for the same 40 steps. A longer schedule is the knob that would settle it. Flat at 40 steps is what we can say today.

5/ Cost: $0.68, about 20 GPU minutes. Rerun it yourself:
pip install whileai && cd recipes/papers/adaptive-clip && python recipe.py
https://github.com/whilehq/whileai-sdk/tree/main/recipes/papers/adaptive-clip
