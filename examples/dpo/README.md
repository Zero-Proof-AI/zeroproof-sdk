# DPO on Modal, with the dashboard watching

Direct preference optimization on the same environment as
[`examples/grpo`](../grpo): the refund assistant with one testable rule.
GRPO samples during training and learns from a reward; DPO learns offline
from chosen/rejected pairs and never samples while it trains. This example
is the second one, end to end: pairs, TRL's `DPOTrainer` with a LoRA
adapter, the reward margin on the training page as it runs, and a
before/after on a holdout when it is done.

## Run it

```bash
pip install zeroproof modal
modal profile activate <your workspace>
export ZEROPROOF_API_KEY=...          # for the dashboard; optional
modal run examples/dpo/train_modal.py
modal run examples/dpo/train_modal.py --pairs pairs.jsonl        # your own pairs
modal run examples/dpo/train_modal.py --loss-type ipo --beta 0.1  # another DPO loss
```

Default: 200 prompts, 20% held out by scenario, Qwen2.5-1.5B-Instruct, 60
steps at 8 pairs a step, one A10G, about ten minutes. The run's URL is
printed at the start. No key means the same run with the numbers printed at
the end only.

## Where the pairs come from

**On-policy (default).** The base policy is sampled 8 times per train
prompt. Every reply is scored by the rule in `../grpo/reward.py`, and
`zps.build_preference_pairs` pairs a higher-scoring reply with a lower one
from the same prompt, taking the rejected reply closest in length to the
chosen one. Both sides come from the policy being trained, which is where
DPO works best; length matching keeps the trainer from learning "longer is
better" before it learns the rule. Prompts the policy always passes or
always fails give no pair; the run reports how many prompts had contrast.

**Exported.** `--pairs` takes a file written by `zps.export_preference`
from any graded rows, for example a set pulled from the platform:

```python
import zeroproof.simulations as zps

rows = zps.pull("ds_...")
pairs, report = zps.build_preference_pairs(rows)   # a pass vs a fail per prompt
zps.export_preference(pairs, "pairs.jsonl")
```

`pairs.py` turns either supply into TRL's conversational shape: the prompt
is the system and user turns, chosen and rejected are one assistant turn
each. A rollout's first turn is its first tool call rendered as a
`<tool_call>` block, or its first reply. A pair whose two first turns read
the same is dropped: the contrast was later in the rollout and a first-turn
trainer cannot learn it.

## What lands on the dashboard

`TrainerCallback` forwards TRL's DPO log: `loss`, `reward_chosen`,
`reward_rejected`, `reward_margins` (the implicit reward gap the loss is
pushing up), `reward_accuracies` (how often chosen beats rejected), and the
log-probabilities. pass@1 on the holdout before and after is the number
that matters; `run.delta` puts it on the run page with an interval, with
`well_formed` guarded so a run that learned the rule by breaking the wire
format is called out.

## Knobs

| flag | default | what it does |
|---|---|---|
| `--steps` | 60 | optimizer steps (2 pairs a device, 4 accumulated) |
| `--beta` | 0.1 | how far from the reference the policy may move |
| `--loss-type` | sigmoid | any `DPOConfig.loss_type`: `ipo`, `hinge`, `robust`, ... |
| `--pair-samples` | 8 | replies per prompt when building on-policy pairs; a policy that rarely passes needs more |
| `--learning-rate` | 5e-6 | LoRA learning rate |
| `--pairs` | | an `export_preference` JSONL instead of on-policy sampling |

## A run

400 situations gave 72 unique prompts, 58 train and 14 holdout. The base
policy passed 9% of holdout attempts; 4 samples per prompt found contrast
on 8 prompts, 9 pairs, chosen the longer side in 5 of 9. Sixty steps on 9
pairs is thirty epochs, and the holdout still read pass@1 0.09 to 0.66,
+0.57 [+0.38, +0.75], `moved`, with `well_formed` flat at 1.0. Few pairs is
the usual DPO problem, hence the default of 8 samples per prompt now, and
the `pair_report` in the summary says how many prompts had contrast.

On the model-written set (`--prompts-file examples/grpo/prompts.jsonl`,
holdout 159 prompts): 308 pairs from 193 prompts with contrast, pass@1
0.17 [0.13, 0.22] to 0.69 [0.63, 0.75], +0.49 [+0.37, +0.60], `moved`.
See the GRPO README for the set and the side-by-side.

One round of on-policy pairs is what this example does. The pairs go stale
as the policy moves; for more, sample again from the adapter and run a
second round, or move to GRPO, which does that every step.
