---
name: strengthen-your-evals
description: >
  Make an eval set hard enough to measure a real gain, and tell a real gain from
  a measurement artifact. Use when building a held-out set, choosing the
  situation mix with arm_weights= and dimensions=, sizing a holdout, reporting a
  base-versus-trained delta, or deciding whether a straddling result needs more
  data. Covers how to find which knobs make YOUR agent fail and steer toward
  them, eval sizing and power, the four ways an eval silently lies, and what
  belongs on a card.
metadata:
  version: "2.0.0"
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
print(f"failure-capable: {len(fails)}/{len(by_prompt)}")   # this is your ceiling
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
wai.simulate(..., dimensions={"tier": ["adversarial", "boundary"]})
```

Arms: `structured`, `llm_guided`, `open_ended`, `behavior_targeted`,
`failure_mutation`. A caller's weights win and stay won, because a stated intent
about an eval is not a hypothesis for the search to relearn. `open_ended` is held
to a 5-10% band whatever is asked.

**4. Re-measure.** The steered set should have a lower base pass rate and a
higher failure-capable fraction than the probe. If it does not, the knob did not
bite on your agent and the next one is worth trying instead.

### What not to assume

**Card richness is not the mechanism.** Counting populated scenario fields
against base pass rate: one set trends harder with more fields, one is flat, and
on a third the *emptiest* cards are hardest. It is which axes are set, not how
many, so do not reach for "more detail" as a proxy for "harder".

## 3. Size the holdout before you run it

Work out the resolvable effect first: `holdout_size(effect, base=, k=, rows=)`.
Pass `rows=` so it reads the spread off your own data.

Per-prompt paired spread has measured 0.233 to 0.453 across lanes. A lane that
assumed the middle planned for 6.5 points resolvable when its own data resolved
4.4. **Effective sample is PROMPTS, not rollouts** — raising `repeats` sharpens
each prompt's estimate and does not narrow a bootstrap over prompts.

## 4. The four ways an eval silently lies

**The simulated user runs on the model under test.** Pin it: `user_model=`, the
same on both arms. Unpinned, the two arms face different customers and the delta
measures the pair. Symptom: different mean user-turn counts per arm.

**Rows vanish from the denominator.** If the grader can fail on a row, that row
must still be counted. Long trajectories fail to grade and long correlates with
failing, so the loss is never random and always flatters. One lane's base moved
0.717 to 0.603 when the dropped rows came back. **Report graded-count per arm.**

**A fixed task set pins less than you think.** `tasks=` pins the opening prompt;
everything after it is still generated. Check turn counts across arms.

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

## 6. On the card

The numbers, the eval size, the resolvable effect at that size, the arm and tier
mix the set was drawn with, and graded-count per arm. That mix is what lets a
reader know whether a 0.95 pass rate means a strong model or an easy holdout.

## Grounding

- ch. 16 (evaluation): bootstrap over prompts, pass@1 with pass^k, decontamination.
- ch. 14 (over-optimization): the symptoms to watch beside any delta.
- ch. 5 and ch. 12: a judge is a reward model; measure agreement, length bias and
  self-preference before quoting a number it produced. Prefer a program grader
  wherever a program can decide the criterion.
