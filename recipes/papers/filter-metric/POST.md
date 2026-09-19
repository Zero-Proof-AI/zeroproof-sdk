# X thread: filter-metric

Written from `results.json` and the README's Result table (verified 2026-09-17).

1/ We reproduced "The Filter Metric is Safety-Critical" (Juntao Yu, arXiv:2609.13866) at small scale: Qwen2.5-1.5B-Instruct, GSM8K, one L40S, 40 GRPO steps per arm. pass@1: 0.39 -> 0.46, delta +0.067 [+0.021, +0.113]. Verdict: moved.

2/ The claim: under a 0/1 outcome plus a length-shaping term, dropping "flat" groups by the shaped score instead of the outcome collapses GRPO (paper: 0.040 vs 0.754 exact match). The one change we tested: the group-is-flat test reads the binary outcome, not the shaped score.

3/ How we know it is not noise: the untrained base was evaluated 10 times (run_std 0.0078, noise band 0.025), the holdout was decontaminated against train, both arms scored on the same 120 held-out tasks, 4 samples each, paired delta with a 95% interval.

4/ What to watch: we zero the advantage of flat groups on a fixed batch; the paper's DAPO arm deletes and refills. Same effect at the advantage level, nothing to say about refill rates. Our gap is far smaller than the paper's; 40 steps is not their schedule.

5/ Cost: $2.62, about 52 GPU minutes. Rerun it yourself:
pip install whileai && cd recipes/papers/filter-metric && python recipe.py
https://github.com/whilehq/whileai-sdk/tree/main/recipes/papers/filter-metric
