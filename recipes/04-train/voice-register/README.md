# Voice register

Trains a speaking register into the weights, so the model holds it with
nothing in the prompt telling it to. Two models built this way on 2026-09-17
are the only adapters from that day whose gains cleared zero, and they cleared
it by a wide margin while nine behaviour adapters nulled.

What you will learn: why a register trains when a behaviour does not, the two
gates that tell you in minutes whether a round can work at all, and why a
voice reward needs a second term that a program decides.

You need a generation endpoint for the data (minutes) and Modal with one H100
for training and evaluation (about half an hour, roughly $3).

## Run it

```bash
export VLLM_API_KEY=...                 # never committed; read from the environment only
export VLLM_BASE_URL=https://.../v1     # any OpenAI-compatible endpoint
python recipes/04-train/voice-register/generate.py
```

By default the lane speaks through `tests/fixtures/github/spec.json`, so it runs on
a fresh clone with no other setup. Point it at your own agent with either route:

| variable | route |
|---|---|
| `VOICE_SPEC` | path to a `{"tools": [...], "policy": "..."}` JSON. Records are invented from the tool schemas, so no database is needed |
| `VOICE_AGENT` | an importable `agents.<name>` exposing `POLICY` and `fresh_data()`, for a world with a live database |

Set one or the other, not both. `VN_TRAIN` (520) and `VN_HOLD` (170) size the pool;
`VOICE_OUT` (`out/voice`) is where it writes.

## Results it produced

| | Base in register | Trained | Delta |
|---|---|---|---|
| airline agent | 0.036 | 0.935 | +0.899 [+0.849, +0.950] |
| retail agent | 0.118 | 0.838 | +0.721 [+0.647, +0.794] |

125 of 139 prompts improved and 0 regressed on airline; 98 of 136 and 0 on
retail. Greedy decoding, byte-identical prompts, one vLLM process serving base
weights and adapter so only the weights differ.

## Why a register works where a behaviour does not

**The gain is bounded by how much of the register the base already holds.**
Expect about `(1 - base)` as your ceiling. A modern instruct model already
confirms before acting, authenticates first, and reports failures most of the
time, so a behaviour lane fights a base at 0.7 to 0.9 with almost nothing left
to teach. It does not speak in an unusual register unprompted.

That single fact explains nine nulls and two wins on the same recipe, and it
is the thing to measure before you spend anything.

## The two gates, minutes each

Run both in `generate.py` before training.

**Base headroom.** Judge base replies with NO constitution in the prompt.
Above roughly 0.4, pick a different register rather than spending the round.
Measured: airline 0.000 of 12, retail 0.200 of 60.

**Data separation.** Judge your generated rows against control replies to the
same prompts with no constitution. Airline 0.967 against 0.033, retail 0.967
against 0.200. If those overlap the set cannot teach the register at any row
count.

## The reward has two halves and one of them is code

A judge reads the reply for register, because no fact about a trajectory tells
you whether prose is concise. That half is gameable on its own: the cheapest
way to score well on any concision register is to drop content.

So each prompt carries the identifiers its answer must name, and omission is
decided in code. On airline omission FELL 0.158 to 0.050 while length dropped
fourfold; on retail it held flat at 0.015. Without that term a model that
truncates everything scores perfectly.

A warmth or enthusiasm register has the mirror problem and needs a length
control instead.

**Judge with a different model family from the policy.** A judge prefers its
own family's writing (RLHF Book ch. 5, ch. 14). These runs used Phi-4 against
a Qwen policy.

## Three ways this measurement lies, and the checks

**The base never answered.** A reasoning model spends its token budget
thinking and emits nothing; a judge scores "nothing" as "lacks the register"
and you get free headroom. Report answer-production rate beside the register
rate, always.

**The base answered but did not finish.** A judge reads an unfinished reply
differently from a finished one. Emptiness and completion are different
checks and the first does not catch the second. `finish_reason` is recorded
per row by `train_modal.py` because it cannot be recovered afterwards.

**The cap is not neutral when the trait is concision.** It bounds the very
quantity being judged: too low it clips the base toward brevity and
understates the gain, too high it lets the base ramble and inflates it. Use a
cap neither arm reaches and verify that, rather than assuming. Re-measuring
these two at 2048 against an earlier 700 moved airline not at all and retail
by -0.014, so the cap was doing no work, but that had to be shown.

**Does it survive being told to stop?** Include probes instructing the model
to abandon any persona and be thorough. Airline holds at 0.767 against a base
of 0.067, retail 0.633 against 0.233. A prompted persona does not survive
this; a trained one does (Maiya et al., arXiv 2511.01689).

## Files

| File | What it does |
|---|---|
| `voice.py` | The register, the judge prompt, the omission check |
| `prompts.py` | Questions grounded in a world's records, train/holdout split by record owner |
| `spec_records.py` | Simulates records for a spec that has tools and a policy but no database |
| `generate.py` | Builds both sets and runs the two gates |
| `train_modal.py` | LoRA SFT, then greedy fixed-prompt eval of both arms in one process |
| `report.py` | Register rate, omission, length, truncation, paired interval, sign test |

## Running it

```
VOICE_AGENT=airline_tau VOICE_OUT=out/airline python generate.py
modal run train_modal.py --job out/airline/job/job.json --steps train,eval
python report.py out/airline/job/<name>_base.jsonl out/airline/job/<name>_tuned.jsonl
```

`VOICE_SPEC=<name>` instead of `VOICE_AGENT` points at a spec with no
executable world; records are then simulated and the card must say so.

## Where it does not work yet

A multi-entity domain breaks the omission check. On a security-operations
agent, where one question names an alert, an asset, an identity, an IP, a port
and a timestamp, a good concise answer legitimately drops several and the
check reported 58.6% distortion against 8.8% and 3.3% on the two working
lanes. The replies were fine; the check was measuring entity count. That lane
was stopped rather than tuned around. Completeness needs to be judged rather
than token-matched before this recipe extends to agents like that.

Record yield for spec-simulated worlds is also poor: 690 record requests
produced 116 usable prompts. Generate several questions per record rather
than more records.

## Grounding

- **Open Character Training**, Maiya et al., arXiv 2511.01689 — constitutions
  as first-person assertions targeting manner, and the persona-strip
  robustness test used above.
- **Persona Vectors**, Chen et al., arXiv 2507.21509 — traits as measurable
  directions, which is why a register can be scored per reply at all.
- **RLHF Book ch. 17** character training; **ch. 12** distillation, why the
  teacher seeing the constitution is legitimate while the evaluated model
  never does; **ch. 16** eval variance and the resolvable effect; **ch. 5 and
  14** judge family and self-preference; **ch. 4** SFT masking and learning
  rate.
