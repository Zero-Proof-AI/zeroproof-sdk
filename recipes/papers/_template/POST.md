# X thread: <recipe name>

Written from `results.json` and the README's Result table after the run. Every
number is copied, not typed from memory. The verdict word is the one
`check.py` printed. A flat result gets the same thread as a moved one.

Rules: one idea per post, under 280 characters each, no hashtags, no
"simulated", no customer names, the paper's first author named, the recipe
linked in the last post. Numbers carry their interval. Cost is the run's usd.

1/ We reproduced <paper short title> (<first author>, arXiv:<id>) at small scale:
<base model>, <dataset>, one <GPU>, <steps> steps per arm.
<metric>: <baseline> -> <recipe>, delta <+0.00 [lo, hi]>. Verdict: <moved / flat>.

2/ The claim: <one sentence from the paper>.
The one change we tested: <one line, the recipe arm vs the baseline arm>.

3/ How we know it is not noise: the untrained base was evaluated <n> times
(run_std <0.000>), the holdout was decontaminated (<n> rows dropped), both arms
scored on the same <n> held-out tasks, paired delta with a 95% interval.

4/ What did not move / what to watch: <the honest line from Learned: proxy vs
target, length, hack scan top feature, or the knob that would change the answer>.

5/ Cost: $<usd>, <gpu minutes> GPU minutes. Rerun it yourself:
pip install whileai && cd recipes/papers/<slug> && python recipe.py
<https://github.com/whilehq/whileai-sdk/tree/main/recipes/papers/slug>
