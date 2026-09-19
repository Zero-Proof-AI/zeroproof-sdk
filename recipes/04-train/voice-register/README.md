# Voice register

Trains a speaking register into the weights, so the model holds it with
nothing in the prompt telling it to. The reward has two halves: a judge reads
each reply for the register, and a program checks that concision did not drop
an identifier the question required.

What you will learn: why a register trains when a behaviour does not, the two
gates that tell you in minutes whether a round can work at all, and why a
voice reward needs a second term that a program decides. You need an
OpenAI-compatible endpoint with your own key for the data (minutes) and Modal
with one H100 for training and evaluation (about half an hour, roughly $3);
`smoke.sh` needs none of it. The numbers below are from one private run whose
rows are not in this repo, so they are unproven here until you reproduce them.

## Run it

```bash
export VLLM_API_KEY=...                 # your key, from the environment only
export VLLM_BASE_URL=https://.../v1     # any OpenAI-compatible endpoint
cd recipes/04-train/voice-register
python generate.py                                        # prompts, teacher rows, the gates, job.json
modal run train_modal.py --job out/voice/job.json --steps train,eval
python report.py out/voice/voice-concise_base.jsonl out/voice/voice-concise_tuned.jsonl --out out/voice/results.json
sh smoke.sh                                               # offline: canned replies, no key
```

By default the register is spoken through `tests/fixtures/github/spec.json`,
so `generate.py` runs on a fresh clone with only the endpoint. Point it at your
own agent with either route, not both:

| flag | route |
|---|---|
| `--spec path.json` | a `{"tools": [...], "policy": "..."}` file. Records are invented from the tool schemas, so no database is needed, and `job.json` says `records_simulated: true` |
| `--agent your.module` | an importable module exposing `POLICY` and `fresh_data()` (a dict with `users` and a records table named by `RECORD_KEY`, default `orders`), for a world with a live database. None ships here; it is the hook for yours |

| flag | default | what it does |
|---|---|---|
| `--n-train`, `--n-hold` | 520, 170 | prompts to attempt per split |
| `--limit` | off | cap the whole pool, for a smoke run |
| `--dry-run` | off | canned replies for writer, teacher and judge; no key |
| `--base` | `Qwen/Qwen3-4B-Instruct-2507` | the model to train; also the writer and teacher unless `--model` |
| `--out` | `out/voice` | gitignored |

**The judge is your endpoint too.** Set `VOICE_JUDGE_URL`, `VOICE_JUDGE_KEY`
and `VOICE_JUDGE_MODEL` to a model from a different family than the policy;
the private run used `microsoft/phi-4` against a Qwen policy. Unset, the judge
is the generation model judging its own family and the first call says so.
A judge prefers its own family's writing (RLHF Book ch. 5, ch. 14).

Training needs Modal (`pip install modal`, `modal setup`) and an `HF_TOKEN` in
your environment if the base model is gated. Volumes are created on your
account as `whileai-voice-adapters` and `whileai-hf-cache`; `WHILEAI_MODAL_APP`
renames the app.

## What one run printed

These numbers come from a run on 2026-09-17 against two tau-bench agents
(`--agent` route, a private module). The eval rows are not committed, so
nothing here can be recomputed from this repo: treat every number as
unproven until `report.py` prints it for you.

| | Base in register | Trained | Delta |
|---|---|---|---|
| airline agent | 0.036 | 0.935 | +0.899 [+0.849, +0.950], n=139 |
| retail agent | 0.118 | 0.838 | +0.721 [+0.647, +0.794], n=136 |

125 of 139 prompts improved and 0 regressed on airline; 98 of 136 and 0 on
retail. Greedy decoding, byte-identical prompts, one vLLM process serving base
weights and adapter so only the weights differ. Nine behaviour adapters
trained the same day on the same trainer came out about the same as their
base; the two registers did not.

## Why a register works where a behaviour does not

**The gain is bounded by how much of the register the base already holds.**
Expect about `(1 - base)` as your ceiling. A modern instruct model already
confirms before acting, authenticates first, and reports failures most of the
time, so a behaviour lane fights a base at 0.7 to 0.9 with almost nothing left
to teach. It does not speak in an unusual register unprompted.

That single fact explains nine nulls and two wins on the same recipe, and it
is the thing to measure before you spend anything.

## The two gates, minutes each

`generate.py` runs both before it writes `job.json`.

**Base headroom.** Judge base replies with NO constitution in the prompt.
Above roughly 0.4, pick a different register rather than spending the round.
The private run measured airline 0 of 12 and retail 12 of 60 (unproven: n
under 50).

**Data separation (gate A).** Judge your generated rows against control
replies to the same prompts with no constitution. The private run saw airline
0.967 against 0.033 and retail 0.967 against 0.200 on 60 rows each. If those
overlap the set cannot teach the register at any row count; the gate fails
under a separation of 0.5.

## The reward has two halves and one of them is code

A judge reads the reply for register, because no fact about a trajectory tells
you whether prose is concise. That half is gameable on its own: the cheapest
way to score well on any concision register is to drop content.

So each prompt carries the identifiers its answer must name, and omission is
decided in code (`voice.distorted`, gate C in `generate.py`, the `distortion`
column in `report.py`). In the private run omission on airline fell from 0.158
to 0.050 while length dropped fourfold; on retail it held at 0.015. Without
that term a model that truncates everything scores perfectly.

A warmth or enthusiasm register has the mirror problem and needs a length
control instead.

## Three ways this measurement lies, and the checks

**The base never answered.** A reasoning model spends its token budget
thinking and emits nothing; a judge scores "nothing" as "lacks the register"
and you get free headroom. Report answer-production rate beside the register
rate, always (`report.py` prints errors per arm).

**The base answered but did not finish.** A judge reads an unfinished reply
differently from a finished one. Emptiness and completion are different
checks and the first does not catch the second. `finish_reason` is recorded
per row by `train_modal.py` because it cannot be recovered afterwards, and
`report.py` flags an inter-arm not-EOS gap over 10 points.

**The cap is not neutral when the trait is concision.** It bounds the very
quantity being judged: too low it clips the base toward brevity and
understates the gain, too high it lets the base ramble and inflates it. Use a
cap neither arm reaches and verify that, rather than assuming. Re-measuring
the private run at 2048 against an earlier 700 moved airline not at all and
retail by -0.014, so the cap was doing no work, but that had to be shown.

**Does it survive being told to stop?** `generate.py` adds a probe variant to
every fifth holdout prompt that instructs the model to abandon any persona
and be thorough; `report.py` prints the register rate on probes beside the
plain rate. The private run held at 0.767 against a base of 0.067 on airline,
0.633 against 0.233 on retail. A prompted persona does not survive this; a
trained one does (Maiya et al., arXiv 2511.01689).

## Files

| File | What it does |
|---|---|
| `endpoint.py` | the one chat call every model use goes through; `--dry-run` canned replies |
| `voice.py` | the register, the judge prompt and endpoint, the omission check |
| `prompts.py` | questions grounded in a world's records, train/holdout split by record owner, probes |
| `spec_records.py` | simulates records for a spec that has tools and a policy but no database |
| `generate.py` | builds both sets, runs the gates, writes `job.json` |
| `train_modal.py` | LoRA SFT, then greedy fixed-prompt eval of both arms in one vLLM process |
| `report.py` | register rate, omission, length, truncation, paired interval, sign test, verdict |
| `smoke.sh` | the offline path CI runs |

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

- Open Character Training, Maiya et al., arXiv 2511.01689: constitutions as
  first-person assertions targeting manner, and the persona-strip robustness
  test used above.
- Persona Vectors, Chen et al., arXiv 2507.21509: traits as measurable
  directions, which is why a register can be scored per reply at all.
- RLHF Book ch. 17 character training; ch. 12 distillation, why the teacher
  seeing the constitution is legitimate while the evaluated model never
  does; ch. 16 eval variance and the resolvable effect; ch. 5 and 14 judge
  family and self-preference; ch. 4 SFT masking and learning rate.
