# Text-to-SQL: hill-climb a model on your schema with a verifier as the reward

A question about a database in, one SQL query out, and a reward that is a
program: run the query, compare the result set to the gold query's result.
No judge. This recipe builds the task set for a schema, benchmarks any
model on it, trains a 4B model with GRPO against that reward on Modal, and
measures each round on the same held-out tasks, so the curve is paired,
has an interval, and cannot be gamed by a wordier answer.

The schema shipped here is a small online store (8 tables, 300 orders,
seeded from `gen_seed.py`). Swap in yours: [Bring your own schema](#bring-your-own-schema).

## What you get

| File | Role |
|---|---|
| `schema.sql`, `seed.sql`, `gen_seed.py` | the database (Postgres 16). `seed.sql` is canonical: the tasks were checked against it. `gen_seed.py` is how it was made (its reviews block was not deterministic when the shipped file was generated, so a regeneration differs there; regenerate only together with re-authoring tasks) |
| `schema_prompt.py`, `prompt.txt` | the policy's system prompt: DDL + notes on what the data means + the one-query rule |
| `tasks.jsonl` | 2,223 tasks: `question`, gold `sql`, `archetype`, `difficulty`. 459 are held out by a hash of the id, the same split in every script. The ledger rows below say which holdout they were measured on: the first 417 tasks (81 held out), the 741-task set (140 held out), or the full set |
| `author.py` | writes tasks for a schema with Claude Sonnet 5, executing every gold query twice before keeping it |
| `sql_verifier.py` | `SQLExec`, the verifier (a `whileai.simulations.verify.Verifier`): execution match, Spider-style. Also the task/split/row helpers and the in-container Postgres for the trainer |
| `rollout.py` | `wai.simulate(tasks=...)`: k samples per task on the account's hosted Qwen3-4B, a model you served with `wai.serve` (`--hosted`), Claude (callable agent), or any SDK agent spec (`--agent openai:...`) |
| `build.py` | grades every rollout file, pass@1 / pass^k / pass@k, `optimize(mode="rl")`, `hack_scan`, pushes train / holdout / eval sets to your account |
| `train_grpo_modal.py` | TRL `GRPOTrainer` + LoRA on Modal with Postgres inside the container (reward = `sql_verifier.shaped_reward`); `--from-run` chains rounds |
| `train.py` | the hosted SFT alternative: `wai.train(method="sft")` on the gold demonstrations, then `wai.serve` |
| `delta.py` | `delta_report` before vs after by difficulty and archetype, attached to the run page |

## Numbers so far (81 held-out tasks, 4 samples each, temperature 0.7)

| Model | pass@1 | 95% CI | pass@4 |
|---|---|---|---|
| Claude Sonnet 5 | 0.88 | 0.81..0.94 | 0.90 |
| Claude Haiku 4.5 | 0.72 | 0.61..0.81 | 0.75 |
| Qwen3-4B, thinking on | 0.69 | 0.60..0.77 | 0.81 |
| Qwen3-4B, thinking off | 0.58 | 0.48..0.68 | 0.64 |

Two things to read off that table before training anything. pass@4 minus
pass@1 is the headroom a grouped update can amplify: 0.06 with thinking
off, 0.12 with it on. And where the model loses: multi-table joins,
self-joins, date buckets, derived metrics (the archetype table `build.py`
prints).

Training the thinking-off model went nowhere, as the headroom predicted:
hosted SFT on the 336 gold demonstrations 0.58 -> 0.63 and GRPO with the
execution reward 0.58 -> 0.59, both intervals covering zero. The hill
climb is in thinking mode; rounds and their numbers are at the bottom.

## Run it

Needs: Python 3.11+, `pip install "whileai>=0.47" "psycopg[binary]" openai anthropic`,
a Postgres you can create a database on, `WHILEAI_API_KEY` from
[zeroproofai.com/platform](https://zeroproofai.com/platform) (the hosted
Qwen3-4B endpoint, the datasets page and the training page), and a Modal
account for the RL step.

**1. The database.** Any Postgres 16 works; the scripts read `T2S_PG_DSN`
(default `postgresql://postgres@127.0.0.1:5499/shop`).

```bash
docker run -d --name t2s -e POSTGRES_HOST_AUTH_METHOD=trust -p 5499:5432 postgres:16
psql -h 127.0.0.1 -p 5499 -U postgres -c "CREATE DATABASE shop"
psql -h 127.0.0.1 -p 5499 -U postgres -d shop -f schema.sql -f seed.sql
```

**2. Benchmark the base model.** Four samples per held-out task; rows
land in `raw/<model>.jsonl` and the run resumes if interrupted.

```bash
python rollout.py --model qwen3-4b --split holdout --k 4      # hosted Qwen3-4B, thinking off (the SDK's default for it)
python rollout.py --model sonnet-5 --split holdout --k 4      # ANTHROPIC_API_KEY, or AWS creds for Bedrock
python build.py                                              # grades, prints the tables, writes out/benchmark.md
python build.py --push                                       # also pushes the sets to your account
```

Rollouts go through `wai.simulate(agent, system_prompt=..., tasks=..., repeats=k)`:
the SDK replays the task prompts on the agent and the gold SQL is attached
to the rows afterwards. Thinking on for the base model: serve it under a
name and sample that (`wai.serve("qwen3-4b-think", base_model="Qwen/Qwen3-4B")`,
then `--hosted qwen3-4b-think`). Any other model: `--agent openai:<model>`
with `OPENAI_BASE_URL`, or add a callable to `MODELS`.

**3. Train, round one.** GRPO on Qwen3-4B, LoRA rank 16, 8 samples per
prompt, the execution reward (1.0 on a result match, 0.1 when the query
runs but is wrong, 0 otherwise). Postgres is installed in the image and
seeded in the container, so the reward needs nothing from your machine.

```bash
PYTHONUTF8=1 modal run --detach train_grpo_modal.py --spawn --run-name t2s-r1 \
  --thinking --skip-eval --steps 100 --gpu H100 --accum 4 --max-completion-length 1536
```

About 65 s a step in thinking mode on an H100 (completions average 900
tokens), so 100 steps is under two hours and roughly $8. `--skip-eval`
skips the slow in-container before/after sampling; the measurement comes
from the served adapter in the next step, through vLLM, in minutes. `--spawn`
submits the call and returns, so nothing depends on your laptop staying
connected (a network drop cancelled a 3 h run at step 49 without it). The run
shows on your training page as it goes (reward, KL, completion length); the
adapter and `summary.json` land on the `whileai-train-runs` volume under
the run id.

**4. Serve and measure.** The adapter is saved on the `whileai-train-runs`
volume under the run id, which is what `wai.serve` hosts.

```python
import whileai.simulations as wai

wai.serve("t2s-r1", "run_...")  # the run id printed by step 3
```

```bash
python rollout.py --hosted t2s-r1 --split holdout --k 4
python build.py
python delta.py --before hosted-qwen3-4b-think --after hosted-t2s-r1
```

`delta.py` prints the paired before/after with a 95% interval, by
difficulty and by archetype, and attaches it to the run page.

**5. Round two.** Start from round one's adapter, measure the same way.

```bash
PYTHONUTF8=1 modal run --detach train_grpo_modal.py --spawn --run-name t2s-r2 --from-run run_... \
  --thinking --skip-eval --steps 100 --gpu H100 --accum 4 --max-completion-length 1536
```

## Runs in parallel

Every run is its own Modal container with its own Postgres and its own
directory on the volume, keyed by run id, so launch as many as you like at
once and compare on the same holdout:

```bash
for lr in 1e-5 2e-5 5e-5; do
  PYTHONUTF8=1 modal run --detach train_grpo_modal.py --spawn --run-name t2s-lr$lr --learning-rate $lr \
    --thinking --skip-eval --steps 100 --gpu H100 --accum 4 --max-completion-length 1536
done
```

Then serve each, `rollout.py --hosted <name>`, and one `build.py`
prints them side by side. Knobs: `--learning-rate`, `--beta`, `--steps`,
`--num-generations`, `--loss-type` (bnpo, grpo, dr_grpo),
`--max-completion-length`, `--lora-rank`.

## Why the tasks are authored, not simulated

`wai.simulate` writes situations, rollouts and world state; it never writes
an answer key, so a verifiable task set is rows you bring that carry
`privileged.reference` (the SDK README says the same under *Verifiers*).
For SQL the reference has to be a query that is exactly right on the data,
and the question has to be unambiguous about which rows count and what to
return, or an exact-match verifier punishes valid readings. `author.py`
therefore has a teacher write the question and the gold together, executes
the gold twice, and keeps it only when it returns 1-50 stable rows. Once
the tasks exist, everything downstream is the SDK: `simulate(tasks=...)`
for rollouts, `data.grade(judge=SQLExec())`, `pass_at`, `optimize`,
`push_rows`, `training_run`, `serve`, `delta_report`.

## Bring your own schema

1. Replace `schema.sql` (DDL, comments welcome: the model reads them) and
   `seed.sql` (data; a few hundred rows is enough, the point is that the
   gold queries have answers).
2. Rewrite `NOTES` in `schema_prompt.py`: the formulas and NULL meanings a
   new analyst would need. Run `python schema_prompt.py` to refresh
   `prompt.txt`.
3. Load the database (step 1 above) and write tasks:
   `python author.py --rounds 2` (Claude Sonnet 5; `ANTHROPIC_API_KEY`, or
   AWS credentials for Bedrock; ~15 min and a few dollars for ~450 tasks).
   Every gold query is executed twice and must return 1-50 rows; near
   duplicates are dropped. Read a sample of `tasks.jsonl`: the questions
   must be unambiguous about which rows count and what to return, because
   the verifier is exact.
4. Steps 2-5 above, unchanged.

The verifier (`sql_verifier.py: SQLExec`, the same module the trainer
mounts) compares result sets as multisets, floats rounded to 2 places,
text case-folded, columns in any order, and in order only when the gold
query has ORDER BY. It reads the gold from `privileged.reference`, which
the training export never projects, so the answer key cannot leak into a
training file.

## What we learned building it

- **Headroom first.** pass@4 - pass@1 says whether RL has anything to
  amplify. At 0.06 nothing moved in 300 steps; at 0.12 it does.
- **Thinking off is the wrong regime for this model.** 0.58 vs 0.69 for
  the same weights.
- **`gradient_checkpointing=True` broke generation.** With TRL 0.19.1 and
  transformers 4.54 on Qwen3-4B, checkpointing makes TRL generate without a
  KV cache and every training completion was random tokens with reward 0,
  while a plain `model.generate` was fine. It is off here; memory is handled
  by micro-batches (`--accum 4` = 8 samples in four batches of 2) and an
  H100. An L40S OOMs at this prompt length.
- **Sampling at 0.7 gives half-duplicate groups.** `optimize(mode="rl")`
  drops them; the RL set from 8 samples on 336 prompts was 144 rows in 38
  groups, `pool_exhausted` in the scan. Train on the prompts, not the set.
- **Code-fenced replies read as truncated to `looks_finished` before 0.46**
  (whileai-sdk#212, fixed in 0.46); `build.py` carries the same patch so
  it also runs on 0.44.
- **Thinking models need a reply budget.** `simulate(agent_max_tokens=4096,
  timeout=300)` (whileai >= 0.47); on the default 2048-token cap and
  60 s timeout the base lost 8% of replies mid-thought and 4 of 81 tasks.
- **Check what the SDK sent the model, not just what came back.** Before
  0.51, `simulate(tasks=...)` with a prompt-only agent drafted a tool surface
  for the situation writer and sent those schemas to the policy too. Qwen
  mostly ignored them; Nemotron-Nano-8B called a made-up tool on every task
  (pass@1 0.00), and 42 of r3's 560 holdout replies were tool calls scored as
  failures. Pinned tasks now never draft tools; the r3 row below is the clean
  re-measure. The polluted files are kept in `raw/with-drafted-tools/`.

## The holdout grew to 459 tasks (2026-09-17)

140 tasks gave a +-0.06 band, so a real 3-point gain could never be proven
(whilehq/whileai-sdk#257). `author.py` wrote 1,482 more tasks the same way
(rounds 2-11, every gold executed twice); the id hash puts 459 of the 2,223
in the holdout, about +-0.035 at k=4. The new tasks are harder (base hard
0.45 vs 0.56 on the old 140), so absolute numbers drop; the paired deltas
are what to read.

| Checkpoint | pass@1 on 459 (95% CI) | pass^4 | pass@4 | vs base, paired |
|---|---|---|---|---|
| base (Qwen3-4B, thinking on) | 0.53 (0.49..0.56) | 0.25 | 0.76 | - |
| r4 | 0.55 (0.52..0.59) | 0.27 | 0.77 | +0.026 (+0.001..+0.050); hard +0.07 (+0.03..+0.11) |
| r5 at step 25 (from r4: 326 band prompts, 32 prompts x 16 samples a step, truncation masked) | 0.57 (0.54..0.61) | 0.27 | 0.78 | +0.044 (+0.020..+0.069); vs r4 +0.018 (-0.005..+0.043), medium +0.04 (+0.01..+0.08) |
| r5 at step 50 | **0.73 (0.69..0.77)** | 0.65 | 0.79 | **+0.205 (+0.179..+0.231)**; vs r4 +0.179 (+0.157..+0.203); vs step 25 +0.161 (+0.138..+0.185); easy +0.18, medium +0.19, hard +0.25, every interval above zero |

On the small holdout r4 vs base was +0.021 (-0.029..+0.068), "no change".
On 459 tasks the same two checkpoints give +0.026 with an interval that
excludes zero: the gain was real and small, and the eval was too small to
see it. Every later row is measured here.

Round 5 is the batch experiment (whilehq/whileai-sdk#252): the same reward
and prompts, but 512 samples per optimizer step instead of 8, only prompts
the r4 policy solves sometimes (every non-unanimous group at k=8, 326 of
601, `build.py --band 0.1,0.9`), and length-truncated samples masked instead
of scored 0. Twenty-five steps (12,800 samples, 3.5 h on one H100) moved the
holdout as much as rounds 1-4 combined (8,000 steps, 32,000 samples). The
step-25 row is served from the saved checkpoint through `serve_modal.py` on
Modal rather than the hosted endpoint; the base was sampled through the same
server as a control: base through that server is 0.53 (0.49..0.57), +0.004
(-0.019..+0.029) against the hosted base, so the serving path adds nothing.
Steps 50, 75 and 100 follow.

**Step 50 is the result the lane was built to get.** pass@1 0.53 -> 0.73 on
459 held-out tasks, paired, with the interval nowhere near zero, in 50
optimizer steps (25,600 samples, about 7 GPU-hours on one H100). Read the
other columns before calling it capability: pass@4 barely moved (0.76 ->
0.79) while pass^4, all four samples right, went 0.25 -> 0.65. The policy
did not learn many new queries; it learned to produce the query it could
already find sometimes, every time, and to stop before the reply budget
(replies with no query 13% -> 0%, SQL errors 12% -> 5%, median reply 1,400
-> 850 tokens). That is what a verifier reward on a 20-80% band does, and
it is exactly the "reliable in production" property a customer is buying.
Rounds 1-4 (one prompt per optimizer step, every prompt, truncation scored
0) spent 8,000 steps to gain 0.03; the batch, the band and the mask did
0.20 in 50. Steps 75 and 100 follow; a second eval run on step 50 is below.

## Other bases on the same holdout (140 tasks, k=4)

Served with `serve_modal.py` (vLLM on one L40S; `--adapter volume:<run_id>`
serves a trained LoRA as `<base>-adapter`, `--runs-volume` names the volume the
run was written to) and sampled through
`rollout.py --agent "vllm:<model>@<url>"`, so any Hugging Face model gets the
same paired number as the hosted ones.

| Model | pass@1 (95% CI) | pass^4 | pass@4 | no SQL | SQL error |
|---|---|---|---|---|---|
| Qwen3-4B, thinking on (base of the climb) | 0.58 (0.52..0.64) | 0.31 | 0.81 | 0.13 | 0.12 |
| Nemotron-Nano-8B-v1, `detailed thinking off` | 0.26 (0.20..0.33) | 0.14 | 0.39 | 0.00 | 0.55 |
| Nemotron-Nano-8B-v1, `detailed thinking on` | 0.26 (0.20..0.33) | 0.15 | 0.39 | 0.00 | 0.53 |
| Nemotron-Nano-8B-v1 **r1**: GRPO 600 steps from the base, thinking off, lr 2e-5, beta 0.01, vLLM generation | 0.35 (0.28..0.42) | 0.27 | 0.45 | 0.00 | 0.35 |

Nemotron-Nano-8B-v1 is half of Qwen3-4B here, and its two arms are the same
number because its reasoning mode never engages on these prompts: with the
schema in the context (system turn, user turn, DDL only, question first, or
the one-query rule softened to "think first") every reply is a bare query,
while the model card's own math example thinks for 3,000+ characters. Even a
forced `<think>` prefill closes after one line. Its failures are real SQL
errors (`WHERE NOT IN (...)` with no column, an alias used before its join,
non-grouped columns), not format. Headroom is 0.12, the same as Qwen's.

One GRPO round on it (`text-to-sql-shop-nemotron-r1`: 600 steps, 44 min on
an H100, `--steps-per-generation 8 --max-completion-length 512`) is the first
climb on this task whose interval excludes zero: **+0.087 (95% +0.048..+0.130)**
paired over the 140 tasks, medium +0.10 (+0.01..+0.18) and hard +0.12
(+0.05..+0.20), easy flat. What it learned is mostly to write SQL that runs:
`executes` 0.45 -> 0.65, SQL errors 0.55 -> 0.35, and the base was already at
`has_sql` 1.00, so none of it is format. The SDK marks a single eval run per
side `moved_unreplicated`, so each side was sampled three times (560 rows
each): base 0.26 / 0.26 / 0.26, r1 0.35 / 0.36 / 0.37, paired deltas +0.087
(+0.048..+0.130), +0.098 (+0.055..+0.148), +0.111 (+0.068..+0.155). Three of
three above zero. The adapter is served the same way as the base
(`serve_modal.py --adapter volume:run_f69e975a1571d445 --runs-volume
zeroproof-train-runs`, model id `nvidia/Llama-3.1-Nemotron-Nano-8B-v1-adapter`);
eval sets `ds_34f0fbd9337ab519` (base) and `ds_2fbd036dd8597150` (r1) on the
platform, Hugging Face configs `eval-nemotron-8b-base` / `eval-nemotron-8b-r1`
and adapter `zero-proof-ai/text-to-sql-shop-nemotron-8b-r1`.

Read next to the Qwen table: the same reward, task set, and trainer moved a
weaker base by nine points in one 44-minute round and a stronger base by
three points in three rounds. The climb is real where the base leaves room
below its own pass@4 and the failures are things a verifier can teach (SQL
that does not run); it is slow where the base already writes valid SQL and
the misses are semantics.

## The hill climb (thinking on, GRPO, execution reward)

Measured through `rollout.py --hosted <name>` (the SDK path, `agent_max_tokens=4096`,
`timeout=300`), 4 samples per task, paired by task. The holdout grew from 81
to 140 tasks when 324 more tasks were authored on the weak archetypes, so
the intervals below are the 140-task ones (about +-0.06).

| Round | From | Method | pass@1 (95% CI) | pass^4 | has_sql |
|---|---|---|---|---|---|
| base | Qwen/Qwen3-4B, thinking on | - | 0.58 (0.52..0.64) | 0.31 | 0.86 |
| r1 | base | GRPO 100 steps, lr 2e-5, beta 0.04, HF generate | 0.58 (0.52..0.64) | 0.29 | 0.89 |
| r2 | r1 | GRPO 200 steps, lr 5e-5, beta 0.01 | 0.60 (0.54..0.67) | 0.34 | 0.90 |
| sft-think | base | self-distillation: 199 verified traces, hosted SFT 2 epochs | 0.60 (0.54..0.67) | 0.39 | 0.96 |
| r3 | r2 | GRPO 1,000 steps, lr 2e-5, beta 0.01, vLLM generation, 8 prompts per generate | 0.61 (0.56..0.67) | 0.29 | 0.86 |
| r4 | r3 | GRPO 1,000 more steps, same settings | 0.61 (0.54..0.67) | 0.32 | 0.85 |

r3 vs base: +0.029 (95% -0.016..+0.073), up at every difficulty (easy +0.02,
medium +0.04, hard +0.03) and clearly up on one archetype, date and time
(0.43 -> 0.58, +0.15, 95% +0.03..+0.27); the best checkpoint so far, not yet a
proven climb by the SDK's rule (the pass@1 interval still covers zero). This
is the clean re-measure after the drafted-tools fix (see lessons); the first
measurement, 0.62 (0.55..0.68) with 42 tool-call replies, is in
`raw/with-drafted-tools/`. Each checkpoint's holdout rollouts and adapter
are on Hugging Face: dataset `zero-proof-ai/text-to-sql-shop` (configs
`eval-base`, `eval-r1`, `eval-r2`, `eval-sft-think`, `eval-r3`, `eval-r4`),
adapters `zero-proof-ai/text-to-sql-shop-<checkpoint>`.

r4 vs r3: -0.007 (95% -0.048..+0.034); vs base +0.021 (-0.029..+0.068), hard
+0.08 (+0.00..+0.17), easy -0.03. A second thousand steps of the same recipe
kept r3's gain and added nothing: the training reward sat at 0.65 through the
whole round (it was 0.65 at the end of r3) and KL to the base stayed at 0.03,
so the policy had stopped moving before r4 began. Round 4 is where "more
steps" stops being the answer for this base; the levers left are the ones in
the closing paragraph of this section (drop prompts the policy already
always or never solves, 16 samples per prompt, a bigger base).

The first two rounds did not move the holdout, while the training reward did climb
(round 1 first-25-step mean 0.49 to last-25 0.63; round 2 up to 0.60-0.75
with KL 0.08), and thinking length fell from ~1,090 to ~800 tokens. That
combination means the policy got better at the prompts it was shown and no
better at held-out ones: 2,400 samples over 336 prompts, LoRA rank 16, is
too small a dose for a 4B model to generalize SQL reasoning from, and the
first thing GRPO learns is the cheap thing (shorter thinking, fewer
failures to emit a query: `has_sql` 0.87 -> 0.90). What the numbers say to
do next, in order: generate with vLLM inside the trainer (`use_vllm`,
colocate) so a round costs minutes instead of 65 s a step, then run
5-10 epochs over the prompts with 16 samples each; only then judge the
method. The table above is the product either way: every round is a
paired number with an interval on the same holdout, so "it got better" is
a claim the customer can check, and "it did not" is caught before anyone
ships it.
