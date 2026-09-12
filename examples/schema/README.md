# One row, six training targets

Every row the SDK writes is a projection of four objects: `Task` (the
situation), `Rollout` (one episode of one policy on it), `Judgment` (a
scorer's verdict), and `Marker` (a behavior measurement). Evals, SFT,
preference pairs, GRPO prompt sets, and OPD / OPSD hints are all views over
those four objects, which is why one dataset can feed all of them.

These two scripts show the round trip and the projections. Both run offline
with a scripted agent, so no key is needed, and both accept any JSONL the SDK
ever wrote: a fresh run, a training export, a platform pull, an OTel ingest,
or a file from the public Hugging Face set.

## Run it

```bash
pip install zeroproof
python migrate.py                    # simulate 24 rows, stamp and split them
python migrate.py old_run.jsonl      # or migrate any legacy file
python project.py out/rows.v1.jsonl  # one file in, six targets out
```

## `migrate.py`

Reads any row file, reports which legacy shape each row is in, and writes
three things: the same rows re-stamped as schema version 1, a `tasks.jsonl`
that holds only the situations (the shippable half: no model output in it),
and a report of anything that did not validate. Rows without a stamp are
version 0; nothing is rejected, and unknown columns ride through untouched.

## `project.py`

Takes a v1 row file and writes one file per training target:

| target | what it needs | where it comes from |
|---|---|---|
| `eval.jsonl` | holdout tasks plus markers on their rollouts | `Task` + `Marker` |
| `sft.jsonl` | messages from passing rollouts | `Rollout` + `Judgment` pass |
| `preference.jsonl` | a passing and a failing rollout of the same task | two `Rollout`s of one `Task` |
| `grpo.jsonl` | prompts only, verifiers-shaped | `Task` |
| `opsd.jsonl` | prompt plus a hint the student never sees | `Task.privileged` + a passing rollout as demonstration |
| `opd.jsonl` | prompts plus a teacher reference | `Task` + `PolicyRef` |

Two rules the scripts enforce, because the objects make them enforceable:

- A `Task` never contains a rollout, so `tasks.jsonl` and `grpo.jsonl` can be
  shipped without leaking any model's behavior.
- The eval marker is never the training reward. `eval.jsonl` scores with the
  markers; `sft.jsonl` and `preference.jsonl` select with the judgment. Same
  rows, different scorer, on purpose.

`opsd.jsonl` is the interesting one. The hint is the policy line the task
exercises plus the world's hidden state (the faults the sandbox injected),
plus a passing rollout of the same task as the demonstration. The student
sees the prompt. The teacher, which is the same model, sees the prompt and
the hint. Their per-token divergence on the student's own rollout is the
training signal, and none of it needs a scalar reward.
