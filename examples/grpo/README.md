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
steps here: with 308 pairs the offline method sees far more contrast per
step than 8 samples a prompt give the online one. That is the comparison
the interval finally makes readable, and the reason both scripts share
one prompt set.
