# GMTS: rank tokens by entropy times advantage, not by entropy alone

**Paper:** GMTS: Gradient Magnitude-based Token Selection Improves RLVR Training, Outongyi Lv et al., arXiv:2608.30632, August 2026. https://arxiv.org/abs/2608.30632
**Book:** rlhfbook.com ch. 6 Policy gradients: group-relative methods "assign the same sequence-level advantage (or reward) to every token when computing the loss", and the chapter's aggregation choices decide whether each token or each sequence contributes equally — which is what decides whether a token *selection* can matter at all.
**Claim:** training on only the top 20% highest-entropy tokens beats training on all of them, but entropy is read one answer at a time: two tokens can carry the same entropy in answers the group scored very differently. Ranking instead by entropy times the answer's learning signal picks a better 20% and raises accuracy.
**The change:** the score the top 20% is taken by. The baseline ranks tokens by entropy; the recipe ranks them by |entropy x advantage|.

## Recipe

1. Base: `Qwen/Qwen2.5-1.5B-Instruct`. Data: GSM8K, 512 train prompts from the train split, 120 held out from the test split (different splits, so there is no overlap to check for).
2. Reward, both arms: the binary outcome, `MathEqual` against the GSM8K gold number. A program, not a judge. The paper changes which tokens are trained on, not what counts as right.
3. Baseline arm (ETS, the prior result): GRPO, and before the loss the completion mask is narrowed to the 20% of the batch's tokens with the highest entropy. The other 80% contribute nothing.
4. Recipe arm (GMTS): the same 20%, ranked by `|E * omega|` instead — the paper's equation 4, with `E` the token's entropy and `omega` the scalar in front of it in the policy-gradient term. The run is on-policy (`num_iterations` 1) with no KL term (`beta` 0), so the importance ratio is exactly 1 and nothing clips: `omega` reduces to the advantage, which is the part of it the paper says carries the cross-answer information.
5. Eval: pass@1 on the same 120 held-out tasks, 4 samples per task. The untrained base is evaluated three times first, and that spread is the noise floor a delta has to clear; the train set is decontaminated against the holdout before any training. Paired delta with a 95% interval (`wai.pass_at`, `wai.delta_report`).

Both arms keep exactly the same *number* of tokens, so the comparison isolates which tokens and not how many. Two details decide whether the change can bind at all, and both are in the code as refusals rather than comments:

- **The threshold is taken over the whole batch, not per row.** A per-row top 20% would hand every answer the same number of slots and throw away the cross-answer comparison the paper is about. Here a microbatch is one prompt's eight rollouts, so "across answers" means across the group the advantage is computed from.
- **The loss must normalize over the batch.** TRL's default `loss_type="bnpo"` divides by the batch's surviving-token count, so an answer given more of the 20% gets more of the gradient. `loss_type="grpo"` divides each sequence by *its own* count, which renormalizes the effect away; the trainer raises rather than run it.

## Run

```bash
python recipe.py --selftest   # the ranking, the mask and the normalizer, offline, no GPU and no key
python recipe.py              # both arms, ~45 GPU minutes on one L40S
python recipe.py --lr 3e-5    # the same two arms at a third of the step size (Climb round 2)
```

## Result

Run today, both arms, on one L40S, at the paper's settings.

| Arm | pass@1 | 95% CI | pass@k | Steps | GPU min |
|---|---|---|---|---|---|
| Base, no training | 0.36 | [0.29, 0.43] | 0.58 | 0 | 0 |
| Baseline (top 20% by entropy) | 0.49 | [0.42, 0.57] | 0.68 | 40 | 27.6 |
| Recipe (top 20% by entropy x advantage) | 0.18 | [0.13, 0.23] | 0.41 | 40 | 17.7 |

Recipe vs baseline: **-0.310 [-0.383, -0.235]** over 120 paired tasks.
Verdict: **flat**. The interval excludes zero but on the wrong side, and this
repo reserves "moved" for a gain.

This is not a small miss. The recipe arm finished at 0.18, **below the 0.36 of
the model that was never trained at all** — it did not fail to help, it
destroyed the policy. Mean completion length fell from 700 characters to 374
and `hack_scan` came back with `frac:upper`, both signatures of a model coming
apart rather than one gaming a reward.

The trainer's own logs say why, and it is not subtle. Gradient norm over the 40
steps:

| Arm | grad_norm mean | grad_norm max |
|---|---|---|
| Baseline (entropy) | 11.5 | 48.1 |
| Recipe (entropy x advantage) | **60.1** | **107.8** |

Both arms keep the same number of tokens and use the same learning rate, but
they do not produce the same size of step. The advantage is constant along a
rollout, so ranking by `|E * omega|` does not reorder tokens *within* an
answer — it decides how many slots each answer gets, and it gives them to the
answers with the largest |advantage|. In a group of eight with a binary reward
and one correct rollout, TRL's scaled advantage is about +2.6 for the correct
one against -0.38 for each wrong one, so GMTS spends its 20% almost entirely
on that one rollout. Entropy selection spreads the same 20% across all eight,
where positive and negative advantages partly cancel. Same token budget, about
five times the gradient. At lr 1e-4 that is past this setup's stability point.

So at the paper's settings this recipe does not test the paper's claim so much
as discover that the claim's mechanism is also a step-size change. Round 2
lowers the learning rate for both arms to find out whether anything is left of
it once that is controlled for. See Climb.

## Checks

Nothing in this table is ticked by hand: every cell is written by `recipe.py`
into `results.json`. These are the round 1 numbers.

| Check | Book | Result |
|---|---|---|
| Eval noise: the base evaluated 3 times, `eval_variance` run_std | ch. 16 | **run_std 0.0115**, so a delta under **0.032** is noise. The -0.310 here is ten times that, which is the one thing about this result that is not in doubt |
| Holdout is clean: `decontaminate(train, against=holdout)` | ch. 16 | **0 of 512 train rows dropped**, as expected for disjoint GSM8K splits — measured, not assumed |
| Reward is a program, not a judge | ch. 7, 13 | `MathEqual` against the public GSM8K gold number. No judge, no model in the loop |
| Proxy vs target: `delta_report(proxy=)` | ch. 14 | `proxy=None`: the training reward *is* the target metric, the same binary check, so there is no proxy to over-optimize. `over_optimized` false |
| Length: mean completion length before -> after, per arm | ch. 14 | **700 chars base -> 636 baseline, 374 recipe.** The recipe arm nearly halved, alongside its accuracy — degeneration, not brevity |
| Hack scan on the last training batch: `hack_scan` | ch. 14 | recipe batch top feature `frac:upper`; baseline batch `contains:: AND contains:calculate the`. Nothing is endorsed. `frac:upper` on a collapsed arm is the scan seeing the wreckage, not a reward surface worth optimizing |
| **Was the change reachable: `selection_overlap`** | ch. 6 | **0.703 in both arms**, with `kept` exactly 0.200. The two rankings chose different tokens for 30% of the budget, every step. The change was reachable, and the arms really did train on different tokens |
| Pinned: seed, torch, transformers, trl, peft | app. C | seed 17 in the trainer, `--seed 0` for the data split; torch 2.7.1, transformers 4.54.0, trl 0.19.1, peft 0.16.0 |

The last row is not in the template. It is here because the recipe next door
([adaptive-clip](../adaptive-clip)) spent a GPU hour on a change that was
implemented correctly and could not move a gradient, and one line of logging
would have said so first. `selection_overlap` is the share of an arm's chosen
tokens that the other arm's ranking would also have chosen, averaged over every
training step. At 1.0 the two arms trained on the same tokens and the recipe
tested nothing. At 0.703 this one did not have that problem.

The two arms share the seed, the data, the holdout, the reward, the selected
token count and every trainer knob except the ranking key, so the delta has one
cause available to it.

## Climb

| Round | What changed | pass@1 | vs previous |
|---|---|---|---|
| 1 | as the paper: top 20% of tokens, k = 8 rollouts, 40 steps, lr 1e-4, LoRA r=32, bnpo loss, on-policy | baseline 0.49, recipe 0.18 | -0.310 [-0.383, -0.235], flat (wrong way) |
| 2 | same, both arms at lr 3e-5 (`--lr 3e-5`), to separate the paper's selection from the step size it implies | pending | pending |

Round 2 changes the one knob round 1 showed to be confounded with the change
itself. It moves **both** arms, because moving only the recipe arm would leave
the two differing in two things and stop testing the paper at all.

## Learned

- **A token selection is also a step-size change, and the paper's framing hides that.** Ranking by `|E * advantage|` cannot reorder tokens inside an answer, since the advantage is constant along it (rlhfbook ch. 6). All it can do is move slots between answers — toward the ones with the largest |advantage|, whose per-token loss terms are by construction the largest. Under a batch-wide normalizer that is a straight multiplication of the gradient: 5x here, measured. Anyone reproducing this paper should re-tune the learning rate per arm, or report the gradient norms, or the comparison is not the one they think it is.
- **Check that your change is reachable, then check what else it changed.** `selection_overlap` 0.703 confirmed the arms trained on genuinely different tokens, which is the check the recipe next door was missing. It is necessary and it is not sufficient: the change was reachable and still did not isolate what it meant to, because the thing it touched carries a second effect.
- **Computing the entropy is the expensive part of this recipe.** The score needs the full next-token distribution, so the trainer takes an extra no-grad forward and materializes a `(rows, tokens, 151936)` float32 tensor for a `log_softmax`. TRL's own scoring pass never does this — `selective_log_softmax` gathers one logprob per position. This ran at about 20 seconds a step at 8 rollouts x 256 tokens, and that pass is the bulk of it. A cheaper entropy (bf16, or fused) would be the first thing to change if this were run at any size.
- **A flat verdict and a collapse are not the same news, and the table says the same word for both.** `check.py` reserves "moved" for a gain, so an arm that ends below the untrained base reads as "flat" in the index. That is the correct verdict by the rule and it is worth knowing that the rule compresses it; the Result table above says what actually happened.

Verified 2026-09-18, whileai 0.64, TRL 0.19.1 + PEFT 0.16.0 on torch 2.7.1. 45.3 GPU minutes, $1.51 on one L40S. Run page: https://www.zeroproofai.com/platform/training/run_5198278ad1c4d851
