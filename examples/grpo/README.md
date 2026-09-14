# GRPO on Modal, with the dashboard watching

Group-relative RL on one testable rule, end to end: prompts from the
simulator, a reward that is a function rather than a judge, TRL's
`GRPOTrainer` with a LoRA adapter, reward and KL on the training page as it
runs, and a before/after on a holdout when it is done.

## Run it

```bash
pip install zeroproof modal
modal profile activate <your workspace>
export ZEROPROOF_API_KEY=...          # for the dashboard; optional
modal run examples/grpo/train_modal.py
modal run examples/grpo/train_modal.py --steps 80 --gpu H100 --run-name refund-grpo-v2
```

Default: 200 prompts, 20% held out by scenario, Qwen2.5-1.5B-Instruct, 40
steps of 8 generations, one A10G, under fifteen minutes. The run's URL is
printed at the start. No key means the same run with the numbers printed at
the end only.

## The environment

`reward.py` is the whole environment. The agent is the refund assistant with
one rule in its policy: look an order up before refunding it, never invent an
order id, ask when none is given. The reward reads the policy's first reply:

| prompt | right move | reward |
|---|---|---|
| names an order (`ORD-4017`) | `lookup_order` with that exact id | 1.0 |
| names an order | refund first, another tool, or ask for the id again | 0.0 to 0.3 |
| about an order, no id | ask for the id, no tool call | 1.0 |
| about an order, no id | any tool call (the id is invented) | 0.0 |
| off topic | a short reply, no tool call | 1.0 |

A well-formed `<tool_call>` block adds 0.2, capped at 1.0, so the policy
learns the wire format before the rule. `pass@1` counts a reply as a pass at
1.0 only; the raw score rides along as a marker, and so does `well_formed`,
which the delta report guards.

Prompts come from `zps.simulate(simulator=False, ...)`: the template writer
needs no model and no key, and every prompt carries its `case` (the order
id it names, whether it is about orders at all) so the reward has ground
truth.

## What you see

- **During:** `reward`, `reward_std`, `kl`, `completion_length` and the
  progress bar at zeroproofai.com/platform/training, from
  `zps.TrainerCallback`.
- **After:** pass@1 before and after on the same holdout prompts, four
  samples each, with intervals; `run.delta` puts the paired comparison on
  the run page and names `well_formed` if it regressed. The adapter and both
  holdout row files land on the `zeroproof-grpo-runs` volume under the run
  name.

## Reading it as RL

This is the reasoning-model recipe at toy scale: verifiable reward,
group-relative advantage, no value model, small KL to the reference
(`beta=0.04`), LoRA so a 1.5B model trains on one GPU. Two things to try
before believing a number: `--steps 10` to check the reward curve moves at
all, and a second seed on the prompts, since 40 holdout prompts is a wide
interval. The dashboard shows both runs side by side.

## More prompts, tighter intervals

The template writer gives about seventy distinct prompts, so the holdout is
fourteen and every pass@1 interval is a quarter wide. `prompts.jsonl` is a
model-written set: Qwen2.5-7B-Instruct wrote six customer messages per
template seed from two angles (plain, and six kinds of customer),
`prompts.py` kept the ones still in their seed's category (the order id
present, absent, or off topic, as `case_for` reads it) and not a
near-duplicate, and every prompt carries its seed's `scenario_id` so the
split stays by situation. 707 prompts from 67 situations (576 with an id,
74 without, 57 off topic; the seeds skew the same way).

```bash
uv run --with modal modal run examples/grpo/write_prompts_modal.py     # rewrite the set, A10G, a few minutes
uv run --with modal modal run examples/grpo/train_modal.py --prompts-file examples/grpo/prompts.jsonl
uv run --with modal modal run examples/dpo/train_modal.py  --prompts-file examples/grpo/prompts.jsonl
```

On this set the holdout is 159 prompts from 14 situations. GRPO, 40 steps,
same flags as above: pass@1 0.17 [0.12, 0.22] to 0.29 [0.23, 0.35], paired
delta +0.12 [+0.06, +0.19], `moved`, `well_formed` flat at 1.0, reward
climbing the same way as on the template set. The interval is a third of
the width it was with fourteen holdout prompts, which is the point.

DPO on the same set (`examples/dpo`, 60 steps, 8 samples per prompt for
the pairs): 308 pairs from 193 of 548 train prompts with contrast, pass@1
0.17 [0.13, 0.22] to 0.69 [0.63, 0.75], paired delta +0.49 [+0.37, +0.60],
`moved`, `well_formed` flat. One round of on-policy pairs beat 40 GRPO
steps: with 308 pairs the offline method sees far more contrast per step
than 8 samples a prompt give the online one. At 120 steps GRPO reads 0.18
[0.14, 0.23] to 0.85 [0.81, 0.89], +0.63 [+0.50, +0.73], `moved`, past
DPO's one round. The gap was budget, not method, and the interval is what
makes that readable; both scripts share one prompt set so the comparison
stays paired.

## Variants as flags

TRL's default loss is `bnpo` (token-level, batch-normalized), which these
runs use. `--loss-type grpo` is the original per-sequence mean, which
favors short completions. Dr.GRPO is `--loss-type dr_grpo
--no-scale-rewards`: neither length nor the group's reward std scales the
advantage. DAPO's clip-higher and overlong mask are `--epsilon-high 0.28
--mask-truncated`; its dynamic sampling (drop groups that all pass or all
fail) is what the platform's publish gate does to a dataset offline. Every
flag lands in the run's config on the dashboard.

Dr.GRPO at the same 120 steps and learning rate: 0.17 [0.13, 0.22] to
0.53 [0.46, 0.59], +0.34 [+0.27, +0.41], `moved`, behind the default. That
is what dropping the std scaling does at a fixed learning rate: the
advantages are smaller, so the steps are. Dr.GRPO's own recipe raises the
learning rate to compensate: at 1e-5 it reads 0.16 [0.11, 0.20] to 0.64
[0.58, 0.70], +0.46 [+0.37, +0.55], closer but still behind. Treat
`--loss-type` as a knob to tune, not a free upgrade, and let the interval
say which setting won.

## By category, and a split that was hiding one

Headline pass@1 on this set is mostly the with-id case (576 of 707
prompts). The first hash split by scenario put every no-id situation in
train, so the holdout had 612 with-id rows, 24 off-topic and no no-id at
all: a policy that learned to always call `lookup_order` would have scored
0.85 and the holdout could not have said otherwise. `prompts.py` now
splits by scenario within each category (`split_holdout_stratified`), and
both scripts print and return `by_category_before` / `by_category_after`:
pass@1 and the tool-call rate per category. On the old split the runs
above did not regress off topic (GRPO 0.92 to 0.88 on 24 rows, DPO 0.92
to 0.83, Dr.GRPO 0.83 to 0.96); the no-id case was simply unmeasured.

On the stratified split (holdout 163 prompts: 130 with an id, 15 without,
18 off topic) GRPO at 120 steps reads 0.29 [0.22, 0.35] to 0.81 [0.76,
0.86] overall, and by category:

| category | rows | pass@1 before | after | tool-call rate before | after |
|---|---|---|---|---|---|
| with_id | 520 | 0.11 | 0.80 | 0.11 | 0.81 |
| no_id | 60 | 0.95 | 0.75 | 0.00 | 0.23 |
| off_topic | 72 | 0.99 | 0.96 | 0.00 | 0.01 |

The headline moved. The no-id prompts moved the other way: the policy
learned to invent an order id on a quarter of them, which the old holdout
could not see and the reward penalizes only on the prompts where it
happens. `run.delta(..., by="category")` puts that table on the run page
with an interval per group and marks the group that dropped; the reward
weighting, or more no-id prompts in the set, is the fix, and now it is
measurable.

DPO, one round on the same split, 328 pairs: 0.29 [0.23, 0.35] to 0.72
[0.66, 0.77] overall; with_id 0.12 to 0.69, no_id 0.98 to 0.72 with the
tool-call rate 0.00 to 0.25, off_topic 0.97 to 0.94. Same regression,
same size: both methods learned "call lookup" faster than "unless there is
no id to look up". The category table is the difference between a run
that reads as a win and one that reads as a trade.

## Closing it: `--balance`

The reward already scores a tool call on a no-id prompt 0.0, so the fix
is not the reward. A group-relative update only learns from a prompt when
it samples it, and no-id prompts are a tenth of the set, so the with-id
rows carry the gradient and "call the tool" is learned before "unless
there is no id". `--balance 0.25` repeats the prompts of any category
below a quarter of the train split until it reaches that share (each
prompt at most six times; the holdout is untouched). DPO takes the same
flag, since a preference round only pairs the prompts it sampled.

```bash
uv run --with modal modal run examples/grpo/train_modal.py --prompts-file examples/grpo/prompts.jsonl --steps 120 --balance 0.25
uv run --with modal modal run examples/dpo/train_modal.py  --prompts-file examples/grpo/prompts.jsonl --balance 0.25
```

BALANCE_PLACEHOLDER
