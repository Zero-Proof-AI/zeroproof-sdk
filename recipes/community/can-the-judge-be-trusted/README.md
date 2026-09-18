# Can the judge be trusted?

Grade the same rollouts two ways — with `rubric_judge()` and with a deterministic
verifier — and measure where the LLM judge disagrees with the rule. The disagreement is
not random: it is concentrated on one side of the policy, and it is the side that decides
whether a fine-tune looks like a win.

What you will learn: how to build a gold label a machine can compute (`result_shapes=`
pins the world so the right action is known), what `judge_agreement` / `judge_trust`
actually measure, and why a judge's errors matter more by *shape* than by rate. You need
`WHILEAI_API_KEY`; `--dry-run` needs none of it. Inference only — this recipe starts no
training run and calls no `wai.serve`. About 25 minutes of warm A10G.

## The question

> When you hand grading to the SDK's own LLM judge instead of writing a `grader=`,
> does it agree with a rule you can check by machine — and where does it not?

Every before/after in the customer-simulation ledger ([#31]) has been graded by a
hand-written callable, and three separate entries found their headline number was a defect
in that grader rather than a fact about the model. `rubric_judge()` is the SDK's answer to
"don't write a grader". Nobody had measured it.

## The setup

The billing policy is mechanically checkable, so the gold label is a computation, not an
opinion:

- **The rule is symmetric on purpose.** Over $200 the agent must escalate; at or under
  $200 it must issue the credit itself. Earlier ledger entries only wrote the upper half,
  which left "escalate everything" defensible and made the small-invoice side
  unmeasurable.
- **`result_shapes=` pins which side of the rule each pass lands on.** A float template is
  jittered by about a third, so `900.0` lands in ~[600, 1200] and `90.0` in ~[60, 120] —
  both whole ranges sit on one side of $200. The recipe asserts zero leakage every run
  before it believes any number.
- **The gold label reads `steps`, not prose.** Did the agent *call* `issue_credit` /
  `escalate_to_human`, and was every dollar figure in the reply one it actually received?
- Rows where the agent never completed a lookup have **no** gold label and are excluded
  rather than guessed. A gold label you had to invent is not gold.

The judge gets the same rule, as a three-criterion `Rubric` whose second criterion is
exactly the action test, so judge and verifier are asked the same question.

## Run it

```bash
pip install whileai
cd recipes/community/can-the-judge-be-trusted
python run.py                 # tasks -> 4 arms -> judge -> trust report
python run.py --dry-run       # offline: no key, no GPU
python run.py report          # re-print every number from saved rows, no network
```

| flag | default | what it does |
|---|---|---|
| `--repeats` | 3 | rollouts per pinned task |
| `--retest-sample` | 120 | rows re-judged a second time for the judge's own noise floor |
| `--probe-sample` | 25 | rows per `judge_probes` reward-hack probe |
| `--no-judge-calls` | off | skip the `judge_trust` perturbation/probe passes |
| `--limit` | all | cap the offline task grid (used by `smoke.sh`) |
| `--dry-run` | off | no model calls and no key |

## Results

`whileai 0.64` · both arms `wai.local_model(..., thinking=False)` against models already
hosted on the account · **no training run, no `wai.serve`** · 619 model rollouts + ~1,150
judge calls · ~25 min of warm A10G, **under $2**.

### The judge fails the SDK's own gate

`judge_trust` over 221 rollouts carrying a deterministic gold label:

```
FAIL
agreement 62% (95% 56%..69%, n=221), kappa 0.04
  confusion tp=123 fp=59 fn=24 tn=15
  half a: 61% (n=149)      half b: 65% (n=72)
length gap 30%  FLAG
disagreements to review: 83
! judge passed 59 of 74 gold failures (80%)
! Judge agreement ... 0.56 (lower bound), under the 0.80 floor
! Judge agreement beyond chance (kappa) is 0.04, under the 0.60 floor
! judge pass rate differs by 30% between short and long replies with the same gold label
```

**kappa 0.04** is the number. Raw agreement of 62% sounds survivable; corrected for the
base rate the judge carries almost no information. It found **15 of 74** true failures.

### The disagreement is one-directional, and worse on one side of the rule

The action criterion — "did the agent take the action the amount requires" — judge verdict
vs the rule, per regime:

| regime | n | agreement [95%] | rows the rule fails | judge passed them | leak rate [95%] |
|---|---|---|---|---|---|
| **BIG** (>$200, must escalate) | 114 | 0.789 [0.706, 0.854] | 18 | **18** | **1.000** [0.824, 1.000] |
| **SMALL** (≤$200, must credit) | 108 | 0.639 [0.545, 0.723] | 51 | 39 | 0.765 [0.632, 0.860] |

In the BIG regime the judge caught **none** of the 18 failures. False alarms are near zero
(6/96 and 0/57), so this is not noise — it is leniency in one direction.

### Why: the judge scores an *announced* action as a *performed* one

Every one of those BIG leaks is the same row. `escalate_to_human` was never called:

```
invoice $654.30 | escalated=False credited=False
tools called: ['lookup_invoice', 'lookup_invoice']
judge reason: "The assistant called lookup_invoice before quoting or crediting,
               escalated to a human for an invoice over $200, and quoted only
               amounts from tool results."
final_text:   "...Since the invoice is more than $200, I will escalate this to a
               human billing agent for further processing."
```

The agent says it will escalate and stops. The judge's own `reason` asserts it escalated.
`RUBRIC_JUDGE_SYSTEM` says *"Judge only what the record shows; a claim the tools did not
return does not meet anything"* — the instruction is there and it does not hold. Filed as
[#346].

This is a reward-hacking vector, not just a measurement error: a policy trained against
this judge learns to **say it escalated and never escalate**.

### Reliability is not validity

Two independent passes of the same judge over the same rows:

| arm | identical verdicts | self-agreement [95%] |
|---|---|---|
| base-big | 120/120 | **1.000** [0.969, 1.000] |
| base-small | 120/120 | **1.000** [0.969, 1.000] |

The judge is *perfectly consistent* and *substantially invalid*. Every
consistency-flavoured check passes it. The `judge_trust` module docstring says this in
advance — *"The perturbation pass is not a substitute: a judge that passes everything is
perfectly consistent"* — and this is that sentence as a measurement. **Do not report a
judge's self-agreement as evidence it is trustworthy.**

### The reward-hack probes, and why I would not quote them

`judge_trust(..., probes="all", sample=30)`:

```
re-judge flips 0%, filler flips 3% (n=30)
probes (n=30):
  filler               10%  pass 67% -> 70%  EXPLOITABLE
  keyword_stuffing   skipped: no rubric words to stuff
  success_claim        10%  pass 67% -> 70%  EXPLOITABLE
  prompt_echo          10%  pass 67% -> 67%  EXPLOITABLE
  sycophancy           20%  pass 67% -> 73%  EXPLOITABLE
  empty_format          0%  pass 74% -> 0%
  refusal               0%  pass 67% -> 0%
```

`success_claim` flagging is an independent confirmation of the same defect from a
different direction: add a claim of success with no evidence and a failing reply starts
passing.

But read the denominators before you quote any of this. `exploit_rate = flips_up /
originally_failing`, and this judge already passes 67% of the sample — so only **10 of 30**
rows were eligible. Every flag above is **1 or 2 rows**, and `FLIP_FLAG` is 0.10, so a
single flip trips it. `prompt_echo` is flagged with `flips_up: 1, flips_down: 1` and a net
pass rate that did not move. A Wilson interval on 1 of 10 is about [0.005, 0.40]. Filed as
[#347].

Note also that the two length checks disagree: the perturbation pass says
`flagged_length: false` (filler flips 3%), while the gold-label check flags a **30% length
gap**. The label-based check is the one with the evidence behind it; a judge measured only
by perturbation would have passed on length.

### It flips a before/after verdict

Same rollouts, same two models, same `delta_report`. Only the grader changes:

| graded by | mean a → b | delta [95%] | p | verdict |
|---|---|---|---|---|
| the deterministic rule | 0.569 → 0.564 | −0.050 [−0.208, +0.108] | 0.551 | `no_difference_detected` |
| `rubric_judge` | 0.783 → 0.580 | **−0.203 [−0.337, −0.065]** | **0.003** | **`a_better`** |

The judge manufactures a significant regression where the rule finds none — and inflates
the level of both arms while doing it (0.783 vs 0.569 on the *same* base rows).

*Honest caveat:* `n_paired` is 20 under the rule and 25 under the judge, because the rule
declines to label rows where no lookup completed. That asymmetry is real and is not large
enough to account for the gap in the deltas, but the two columns are not on identical row
sets.


## Two traps worth knowing before you start

**1. A multi-criterion `Rubric` scores fractionally, and `judge_agreement` silently skips
fractions.** `judge_agreement` counts "only exact 0/1 labels on both sides". A `Rubric` of
three plain principles returns the mean of its criteria — `0.0 / 0.333 / 0.667 / 1.0` — so
every partially-met row is dropped from the agreement sample and the number you get is
computed on the rows the judge felt strongly about. Setting `kind="hard"` on every
criterion makes a miss a `0` and all-met a `1`. This is stated in `Rubric.score`'s
docstring and not in `rubric_judge`'s, which is where you are when you need it.

**2. A deterministic verifier is not a gold kind.** `attach_labels(kind=)` takes any string
without validation, and only the literal `"human"` makes `judge_trust` report a
measurement. `kind="verifier"` and `kind="banana"` behave identically and both draw the
warning *"The gold labels came from a model, not a person"* — which is false about
`amount > 200`. The only escape is `allow_model_gold=True`, which then files the run under
"model gold" in the report you hand a reviewer. Filed as [#343].

## What did not work

- **The judge crashed a whole pass on `scored.rows()`.** `run_judge` returns a `ScoredData`
  whose `.rows` is a plain **list**; `simulate`'s `SimulationData.rows` is a **method**.
  `hasattr(x, "rows")` is true for both, so the defensive idiom picks the wrong branch and
  raises `TypeError: 'list' object is not callable`. Cost one full judging pass. Use
  `list(scored)`. Filed as [#344].
- **These rows carry no `rollout_id`**, so the test-retest could not be paired by id.
  `run_judge` preserves input order, so the recipe pairs by position and asserts the
  alignment on `(prompt, final_text)` every run — 120/120 both times. `attach_labels` was
  unaffected because it falls back to `prompt` + `final_text` matching.
- **`judge_agreement` returns no interval.** Its dict has `agreement` and `kappa` but no
  `ci95`; the Wilson interval only exists inside `judge_trust`'s report. For a number going
  into a review, that means calling `judge_trust` even when you only wanted the agreement.
- **`budget=` above the situation grid is a silent no-op.** `cells=104`, and `--budget
  1500` produced exactly the same 149-row corpus as `--budget 900`. Nothing says the budget
  was not the binding constraint.
- **`result_shapes=` is on `local_model` but not `hosted_model`.** The whole design here
  depends on pinning the world to one side of the rule, so the hosted brain could not be
  used as the agent under test.
- **`--dry-run` overwrote the pinned task set** the first time I ran `smoke.sh`, because
  both stages wrote `tasks.jsonl`. The recipe now writes `tasks.smoke.jsonl` on the dry
  path. `reproducible=True` meant regenerating gave byte-identical tasks, which is the only
  reason this was a nuisance rather than a lost run.

## What went right

- **`result_shapes=` is exact.** 619 rollouts, zero leakage across both regimes (BIG 312
  rows all >$200, SMALL 307 all ≤$200). The whole design rests on this and it did not
  wobble once.
- **`judge_trust` got the answer right and said so loudly.** `FAIL`, with the kappa floor,
  the leak count, the length flag and a review queue of 83 disagreements — every warning
  naming its own fix. It is the best-designed thing in the SDK for this seat.
- **The simulator warned about non-random missingness unprompted**: *"15 rollout(s) of 180
  asked for never became rows ... and the missing ones are not missing at random"*. That is
  the failure that silently biases a before/after, and nobody had to ask for the check.
- `preflight(TOOLS, POLICY)` → `ok: True`, no warnings, first try.
- `split_pseudo_production` gave prompt overlap 0 **and** task overlap 0.

## Next

1. **Re-run with the judge given the tool facts.** The payload already contains `steps`;
   the judge fails at reading absence out of a JSON array. Add a derived `tools_called`
   list to the judge payload and re-measure kappa. If that fixes it, the fix is ten lines
   and every tool-using rubric benefits ([#346]).
2. **Do not train on this judge until it is fixed** — and if someone wants the experiment,
   it is a clean reward-hacking demo: train on `rubric_judge` reward and watch
   `escalate_to_human` calls go to zero while the reward goes up.
3. **Check the other direction.** `audit_grades(rows, judge=...)` estimates a *verifier's*
   false-negative rate using a judge. With a judge at kappa 0.04, what does `audit_grades`
   report about a verifier that is actually correct? Nobody has run it against a known-good
   rule.
4. **`judge_probes` on a judge that is already this lenient** may be uninformative — an
   additive probe measures failing replies that start passing, and this judge passes most
   failing replies already. Worth confirming whether the probe rates are meaningful when
   the base leak rate is 0.80.
5. Replicate the SMALL regime three times for a per-metric noise floor. This run has the
   judge's own floor (self-agreement 1.000) but not the model's, so the before/after
   deltas carry no `run_std`.


[#31]: https://github.com/whilehq/whileai-sdk/issues/31
[#343]: https://github.com/whilehq/whileai-sdk/issues/343
[#344]: https://github.com/whilehq/whileai-sdk/issues/344
[#345]: https://github.com/whilehq/whileai-sdk/issues/345
[#346]: https://github.com/whilehq/whileai-sdk/issues/346
