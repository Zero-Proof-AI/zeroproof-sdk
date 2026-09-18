# GMTS: rank tokens by entropy times advantage, not by entropy alone

**Paper:** GMTS: Gradient Magnitude-based Token Selection Improves RLVR Training, Outongyi Lv et al., arXiv:2608.30632, August 2026. https://arxiv.org/abs/2608.30632
**Book:** rlhfbook.com ch. 6 Policy gradients: group-relative methods "assign the same sequence-level advantage (or reward) to every token when computing the loss", and the chapter's three aggregation choices decide whether each token or each sequence contributes equally — which is exactly what decides whether a token *selection* can matter at all.
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
python recipe.py              # both arms, sized for under 60 GPU minutes on one L40S
python recipe.py --arm recipe --fraction 0.4
```

## Result

Run today, both arms, on one L40S.

| Arm | pass@1 | 95% CI | pass@k | Steps | GPU min |
|---|---|---|---|---|---|
| Base, no training | TBD | TBD | TBD | 0 | 0 |
| Baseline (top 20% by entropy) | TBD | TBD | TBD | 40 | TBD |
| Recipe (top 20% by entropy x advantage) | TBD | TBD | TBD | 40 | TBD |

Recipe vs baseline: **TBD**. Verdict: **TBD**.

## Checks

Nothing in this table is ticked by hand: every cell is written by `recipe.py`
into `results.json`.

| Check | Book | Result |
|---|---|---|
| Eval noise: the base evaluated 3 times, `eval_variance` run_std | ch. 16 | **run_std TBD**, so a delta under TBD is noise. This measures re-running the eval, not re-running the training |
| Holdout is clean: `decontaminate(train, against=holdout)` | ch. 16 | **0 of 512 train rows dropped**, as expected for disjoint GSM8K splits — measured, not assumed |
| Reward is a program, not a judge | ch. 7, 13 | `MathEqual` against the public GSM8K gold number. No judge, no model in the loop |
| Proxy vs target: `delta_report(proxy=)` | ch. 14 | `proxy=None`: the training reward *is* the target metric, the same binary check, so there is no proxy to over-optimize |
| Length: mean completion length before -> after, per arm | ch. 14 | TBD |
| Hack scan on the last training batch: `hack_scan` | ch. 14 | TBD |
| **Was the change reachable: `selection_overlap`** | ch. 6 | TBD |
| Pinned: seed, torch, transformers, trl, peft | app. C | seed 17 in the trainer, `--seed 0` for the data split; torch 2.7.1, transformers 4.54.0, trl 0.19.1, peft 0.16.0 |

The last row is not in the template. It is here because the recipe next door
([adaptive-clip](../adaptive-clip)) spent a GPU hour on a change that was
implemented correctly and could not move a gradient, and one line of logging
would have said so first. `selection_overlap` is the share of an arm's chosen
tokens that the other arm's ranking would also have chosen, averaged over
every training step. At 1.0 the two arms trained on the same tokens and the
recipe tested nothing.

The two arms share the seed, the data, the holdout, the reward, the selected
token count and every trainer knob except the ranking key, so the delta has
one cause available to it.

## Climb

| Round | What changed | pass@1 | vs previous |
|---|---|---|---|
| 1 | as the paper: top 20% of tokens, k = 8 rollouts, 40 steps, lr 1e-4, LoRA r=32, bnpo loss, on-policy | TBD | TBD |

## Learned

- TBD
- TBD
- TBD

Verified TBD, whileai TBD, TRL 0.19.1 + PEFT 0.16.0 on torch 2.7.1. TBD GPU minutes, $TBD on one L40S. Run page: TBD
