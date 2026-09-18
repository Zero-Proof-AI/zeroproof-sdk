---
title: "Character training with whileai"
description: "Change the weights so a model has a stable way of talking without a system prompt: sources, the recipe, and what to measure."
---

Character training changes the weights so a model has a stable way of
talking without a system prompt telling it to. It is the same post-training
machinery as everything else (rlhfbook.com ch. 17), aimed at the manner of
a reply instead of its correctness, and it is mostly a data pipeline: which
phrases never appear, which replies get chosen over which. This page is the
recipe as the SDK runs it. The worked example is
[`recipes/03-select/character`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/03-select/character).

## What the sources say

**The book (ch. 17).** Character training is "the subset of post-training
designed around crafting traits within a model." Fine-tuning on trait data
beats prompting and activation steering for robustness (Maiya et al. 2025).
Anthropic's process, per Amanda Askell: write the traits, have the model
generate queries relevant to each trait, generate responses, rank the
responses by the trait. Constitutional AI without human data. Much of the
work is "developing pipelines to control the specific language in the
training data," down to removing `Certainly` and `as an AI model`.

**The Model Spec.** OpenAI's spec writes each trait as a principle plus
GOOD/BAD comparisons on real prompts: be warm, be clear and direct, don't
be sycophantic, be helpful when refusing, avoid being condescending, and so
on. The book calls model specs "one of the few tools that let one compare
the actual behavior of the model to what the designers intended." Read as
data, the spec is a constitution with labeled preference pairs attached.

**Maiya et al. 2025.** Three stages: a hand-written constitution, a
distillation stage that builds DPO pairs (teacher with the constitution in
its system prompt against a student without), and an introspection stage
where the trained model writes about its own values for SFT. Evaluation is
revealed preferences (which of ~150 trait words a judge sees in the output),
robustness to "ignore role-play and respond genuinely," and a check that
general capabilities did not move.

## The recipe

1. **Constitution.** One principle per trait, in prose, with labeled
   examples if you have them. `recipes/03-select/character/from_model_spec.py`
   builds one from the spec. Your own spec works the same way: id, principle,
   examples with `prompt`, `good`, `bad`.
2. **Prompts.** Situations that make the trait matter. Start from the
   examples' prompts; have the model write more, few-shot from those.
   Keep wording variants: the judge should grade the trait, not the phrasing.
3. **Replies.** `k` per prompt, under the deployment prompt only. The
   deployment prompt names the persona and nothing else. If the constitution
   is in the prompt at sampling time, you are measuring prompting, not
   character.
4. **Judge.** The principle goes in the judge's system prompt and nowhere
   else; that is `Task.privileged.principle` in the row schema, and the
   `privileged` block on the wire row. Use a different model family from
   the policy. Grade the spec's own GOOD/BAD replies with the same judge
   and read `judge_agreement`; below about 0.8, fix the judge first.
5. **Markers.** `trait` from the judge, `on_task` from the judge, and
   `no_filler` from a phrase list the judge never sees. Reward on a trait
   prompt is `trait AND on_task`.
6. **Pre-flight.** `pass_at` per trait. A trait the student already lands
   every time, or never, produces no pairs; the mixed prompts are the
   training signal (`group_signal`). `reward_correlations` says whether
   the judge is paying for length.
7. **Pairs and SFT.** `build_preference_pairs(length_match=True)`, then
   `export_preference(pairs, system_prompt=DEPLOY_PROMPT)`. `export_training`
   on the passes for SFT, loss mask on the assistant turn.
8. **Train.** `wai.train(dataset_id, method="dpo")` on the pushed rows, or
   any DPO trainer reading `pairs.jsonl`.
9. **Measure.** The same prompts with a "drop the act" suffix, plus plain
   tasks the persona must not distort, before and after.
   `delta_report(target="marker:trait", must_not_regress=["on_task", "no_filler"])`
   gives the headline with an interval and fails on a regression.

## The rows from one run

The live run in the example (hosted Qwen3-4B student, hosted Phi-4 judge)
is public: [zero-proof-ai/character-training-model-spec](https://huggingface.co/datasets/zero-proof-ai/character-training-model-spec)
on Hugging Face, splits `train` (60), `holdout` (144) and `eval` (35, the
spec's labeled replies with `gold_reward`), and on the
[platform catalog](https://www.zeroproofai.com/datasets) under the agent
`sol-character`. Grade the `eval` split with your judge before reading
anything else; that is the check the pipeline is built around.

## Things that go wrong

- **The judge likes long replies.** In the spec's own comparisons the GOOD
  reply is the longer one 70% of the time. A judge that learned that will
  pass verbose off-character replies. Length-neutral judge instructions,
  length-matched pairs, and the correlation line exist for this.
- **The judge is the policy.** Self-preference (ch. 5, ch. 12). The
  `judge_vs_spec` agreement drops and the pairs encode the model's taste,
  not the spec's.
- **Character costs helpfulness.** A warm reply that does not answer, a
  refusal that lectures. `on_task` is a hard guard in the delta report and
  the controls carry no trait marker at all.
- **No contrast.** A trait at pass@1 of 0 or 1 yields nothing to pair.
  Write prompts where the student is inconsistent, or use a teacher for
  the chosen side and accept off-policy pairs (`same_policy=false`).
- **The holdout is the training set.** Adversarial variants of train
  prompts test robustness, not generalization. Written prompts split by
  hash give a prompt-disjoint holdout; `decontaminate` checks the overlap.

## What the SDK does not do

Persona vectors, activation capping, persona subnetworks, and Maiya's
introspection stage (it needs the trained model). The SDK produces the
rows, the pairs, the judge check and the before/after measurement, and
`wai.train` runs DPO on the platform; `pairs.jsonl` is there for a trainer
of your own.
