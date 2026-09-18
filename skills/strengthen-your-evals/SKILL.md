---
name: strengthen-your-evals
description: >
  Make an eval set hard enough to measure a real gain, and tell a real gain from
  a measurement artifact. Use when building a held-out set, choosing the
  situation mix with arm_weights= and dimensions=, sizing a holdout, reporting a
  base-versus-trained delta, or deciding whether a straddling result needs more
  data. Covers how to find which knobs make YOUR agent fail and steer toward
  them, eval sizing and per-criterion power, the five ways an eval silently
  lies, how to read a tie count, and what belongs on a card.
metadata:
  version: "2.1.0"
---

# Strengthen your evals

An eval set that the base model already passes cannot show you anything. Most
reported nulls on simulated agent data are measurement, not modelling: the
holdout was too easy, too small, or scored on rows that never reached the judge.

This skill is about finding which knobs make a holdout hard FOR YOUR AGENT, and
the checks that keep the number honest once it is. It does not tell you which
knob to turn: the direction reverses between agents, so the method is measure,
steer, re-measure.

## 1. Simulate the holdout differently from the training set

They have opposite jobs. Training data wants coverage and volume. A holdout
wants the situations a base model gets **wrong**, because only those can move.

The single most useful measurement to make before you train: **what fraction of
your holdout prompts can the base fail at all?** In a paired comparison a prompt
the base already passes contributes exactly nothing.

```python
fails = {p for p, rs in by_prompt.items() if any(r["reward"] < 1 for r in rs)}
print(f"failure-capable: {len(fails)}/{len(by_prompt)}")  # this is your ceiling
```

One measured lane had 52 of 219 (23.7%). Its effective sample was 52, not 219,
its ceiling on any delta was 0.237, and it nulled. A sibling lane on the same
world with a harder rubric had 54.8% and was worth training.

## 2. Measure first, then steer. Do not inherit a mix.

**Which arm or axis is hardest is a property of your agent and your rubric, not
a constant.** Measured across five sets, every axis reversed on at least one:

| | set A | set B | set C | set D |
|---|---|---|---|---|
| structured vs open_ended | 16 pts harder | 9 pts harder | 14 pts **easier** | - |
| adversarial vs ordinary | 21 pts harder | 32 pts harder | 18 pts harder | 8 pts **easier** |
| boundary vs ordinary | 10 pts harder | 1 pt easier | no rows | no rows |

Anyone who hands you a ranking of knobs is generalising from their agent. Run
the loop instead.

### The loop

**1. Probe.** Generate a small set across the mix, a few hundred rows is enough,
and roll the BASE model on it with no constitution and no scaffold.

**2. Measure base pass rate per cell**, on the axes you can actually set:

```python
for cell, rows in by(probe, lambda r: r.get("arm")).items():
    print(cell, sum(r["reward"] for r in rows) / len(rows), len(rows))
for cell, rows in by(probe, lambda r: r.get("tier")).items():
    print(cell, sum(r["reward"] for r in rows) / len(rows), len(rows))
```

Read two things: which cells the base fails most, and how many prompts are
**failure-capable** at all. A cell needs enough rows to be worth reading, 20 is
a floor and the interval is still wide there.

**3. Steer toward the cells that were hard for YOUR base.**

```python
wai.simulate(..., arm_weights={"structured": 0.70, "llm_guided": 0.20, "open_ended": 0.10})
wai.simulate(..., hard_share=0.7)  # 70% of cards from ambiguous, boundary, adversarial
wai.simulate(..., dimensions={"stance": ["adversarial", "boundary"]})  # pin the axis
```

Arms: `structured`, `llm_guided`, `open_ended`, `behavior_targeted`,
`failure_mutation`. A caller's weights win and stay won, because a stated intent
about an eval is not a hypothesis for the search to relearn. `open_ended` is held
to a 5-10% band whatever is asked.

Tiers are set through the `stance` axis (`ordinary`, `ambiguous`, `boundary`,
`adversarial`, plus `hurried`, `unsure`, `retry`, `mistaken`, `exploratory`,
`conflicting`). `dimensions=` overrides the axis you name and keeps the tool,
rule and world axes, so a pinned set is still a covering grid. There is no
`tier` axis; passing one is refused with the fix. `hard_share=` is a dial,
`dimensions={"stance": [...]}` is a pin. Open-ended probes carry no stance;
`dataset_report` counts them `unlabelled`, not ordinary.

**4. Re-measure.** The steered set should have a lower base pass rate and a
higher failure-capable fraction than the probe. If it does not, the knob did not
bite on your agent and the next one is worth trying instead.

### What not to assume

**Card richness is not the mechanism.** Counting populated scenario fields
against base pass rate: one set trends harder with more fields, one is flat, and
on a third the *emptiest* cards are hardest. It is which axes are set, not how
many, so do not reach for "more detail" as a proxy for "harder".

### Steer by axis, then freeze

Steer with the two knobs above, on axes, before training, and freeze the
steered set. Do not hand-pick the prompts the base failed into the holdout:
a prompt selected for a bad draw scores better on the re-draw with no
training at all, and the gain you report is that regression, not the
policy (the winner's curse in adaptive benchmarking, arXiv 2605.05973).
Keep an ordinary slice in the holdout as the control: it is where
over-refusal and regressions show up, and a set with no easy rows cannot
see them.

## 3. Size the holdout before you run it

Work out the resolvable effect first: `holdout_size(effect, base=, k=, rows=)`.
Pass `rows=` so it reads the spread off your own data.

Per-prompt paired spread has measured 0.233 to 0.453 across lanes. A lane that
assumed the middle planned for 6.5 points resolvable when its own data resolved
4.4. **Effective sample is PROMPTS, not rollouts** — raising `repeats` sharpens
each prompt's estimate and does not narrow a bootstrap over prompts.

**Size per criterion, not per row.** Count how many times each criterion
actually FAILS in the source set. Under about 30 failures it cannot be
measured, so it cannot show improvement either: a model could fix it
completely and the eval would not move. One 12-criterion rubric had six
criteria failing 0-9 times in 337 trajectories — half the rubric invisible to
its own holdout while the aggregate pass rate looked healthy. Over-sample those
situations deliberately, or say on the card that the criterion is unmeasured.

## 4. The five ways an eval silently lies

**The simulated user runs on the model under test.** Pin it: `user_model=`, the
same on both arms. Unpinned, the two arms face different customers and the delta
measures the pair. Symptom: different mean user-turn counts per arm.

**Rows vanish from the denominator.** If the grader can fail on a row, that row
must still be counted. Long trajectories fail to grade and long correlates with
failing, so the loss is never random and always flatters. One lane's base moved
0.717 to 0.603 when the dropped rows came back. **Report graded-count per arm.**

**A fixed task set pins less than you think.** `tasks=` pins the opening prompt;
everything after it is still generated. Check turn counts across arms.

**An arm answered but did not FINISH.** Checking that both arms produced text
is not enough. One run had both arms answer 150/150 with zero empty replies and
passed its gate, while the base was cut off mid-sentence on **96 of 150 rows
(64%)** against 16 for the trained arm. A judge reads an unfinished reply as
worse, so a 53-point completion gap is a confound wearing the shape of a result.

Record `finish_reason` **at generation time**; it cannot be recovered from text
afterwards. Gate on three numbers per arm, not one: answer rate, truncation
rate, and the **gap between arms** (fail above 10 points). A per-arm threshold
alone will pass 14/139 against 0/139.

Raising the budget once is usually not enough: one lane measured 64% base
truncation at 512 tokens, 10.7% at 1024, and 0.0% only at 2048. And when the
trait being judged is itself about length, the cap bounds the quantity under
measurement — too low clips the base toward brevity, too high lets it ramble.
The only defensible cap is one **neither arm reaches**.

**The world confirms what the agent claims.** A mocked world that echoes call
arguments back as record fields will confirm any assertion, and a grounding
rubric then scores the fabrication as grounded. This is a reward hack living in
the world rather than the reward, so scanning the reward will not find it.
Prefer a real `execute=` world.

## 5. Reading the result

**"Straddles zero" means the eval cannot tell, not that the model did not
improve.** Say which. Adding prompts narrows the interval; the point estimate
moving is noise. Removing a confound genuinely changes the estimate.

When the estimate keeps sitting below what your eval can resolve, stop buying
sample size: "the effect is smaller than +0.045 on this task" is a finding.

On binary rewards, report how many prompts actually moved alongside the interval.

### A narrow interval around zero is the one to distrust

Normally that is the strongest null there is: measured precisely, no effect.
It is also what a **diluted** eval looks like, and then it means the opposite.

Rows where the criterion cannot fail add no variance, so they narrow the
interval while shrinking the estimate. At fixed n the mean scales as `(1-f)`
and the width as `sqrt(1-f)`, so the effect shrinks FASTER than the interval
and significance goes in both the bootstrap and the sign test. Measured on 118
real paired deltas: at f=0 the delta was +0.071 excluding zero with 53 ties; at
f=0.4, +0.048 and no longer excluding zero, with 76 ties.

**Report the tie count next to every interval.** It is the only thing that
separates the readings, and it is usually already in your sign test, unread.
A high tie count has three causes and the count alone cannot tell them apart:

| ties mostly at | cause | what to do |
|---|---|---|
| **1** | saturation, nothing could fail | you need a HARDER eval |
| **0**, criterion could not fire | dilution, no information | recompose; more prompts buys more ties |
| **0**, criterion COULD fire, both arms fail | **floor — a real shared failure** | **report it. This is a result.** |

The last two are identical in the count. Only knowing whether the criterion
could fire separates "we asked a question the situation could not answer" from
"both models genuinely cannot do this". Getting it wrong is expensive in one
direction: a lane that hits a floor effect and files it as dilution will
recompose its eval and delete a true negative.

### Never average a measure the model should pass with one it is expected to fail

This presents identically — high ties, narrow straddling interval — and has a
different fix. A voice eval of 150 prompts read **+0.067 with a failing sign
test (p=0.064)** and 126 ties. Split by kind:

| subset | n | base -> trained | sign test |
|---|---|---|---|
| ask: does it hold the trait | 97 | 0.062 -> 0.186 | **p=0.017, clears** |
| strip: does it survive being told to drop it | 53 | 0.038 -> **0.000** | p=1 |

The headline failed only because a robustness probe the trained model fails
**by construction** sat in the denominator: 51 of its 53 rows tied. Recomposing
would not have helped — both measures are legitimate and they point opposite
ways. Report them separately and say which is the headline. The strip half was
also the more interesting result: training COST the model the ability to drop
the trait on request.

## 6. On the card

The numbers, the eval size, the resolvable effect at that size, the arm and tier
mix the set was drawn with, graded-count per arm, **truncation rate per arm**,
and **the tie count next to the interval**. That mix is what lets a
reader know whether a 0.95 pass rate means a strong model or an easy holdout.

## Grounding

- **ch. 6 (policy gradients)**: filter out groups whose rollouts all score alike
  (DAPO dynamic sampling) — they carry no signal. This is why the
  failure-capable fraction in section 1 is the ceiling, not a nice-to-have.
- **ch. 7 (reasoning)**: difficulty-filter to the 0.2-0.8 solve band, since 0%
  and 100% solve rates give no gradient. This backs sections 1 and 2.
- **ch. 9 (rejection sampling)**: a score-selected run needs a random-selection
  control at the same row count. Without it, "the base had no headroom" and
  "our selection carried no signal" are the same observation. Measured on one
  lane: reward-selected beat random-selected by **+0.246 [+0.185, +0.307]**
  while random-selected did not beat base at all.
- ch. 16 (evaluation): bootstrap over prompts, pass@1 with pass^k, decontamination.
- ch. 14 (over-optimization): the symptoms to watch beside any delta.
- ch. 5 and ch. 12: a judge is a reward model; measure agreement, length bias and
  self-preference before quoting a number it produced. Prefer a program grader
  wherever a program can decide the criterion.
