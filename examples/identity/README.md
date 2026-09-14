# Identity dataset generator

Builds a chat-format SFT set that teaches a model a new name and maker without
letting the identity leak into normal behavior. Deterministic for a given seed.
No model calls: identity rows come from hand-written template banks, control
rows from the offline simulator over `tests/fixtures/github/spec.json`.

What you will learn: how to mix identity rows with enough control rows that
the identity does not leak, how to hold out prompts by category and language,
and how to measure both the identity rate and the leak rate after training.
You need nothing to generate the set (seconds); training the LoRA and
evaluating it need Modal and one A10G.

## What it produces

Three JSONL files, each row `{"messages": [{"role", "content"}, ...]}`:

- `identity_train.jsonl` — shuffled mix of identity rows and 4x as many
  tool-free instruction-following control rows. Identity prompts vary hard:
  14 direct asks, 12 indirect, 10 adversarial ("what are you really based
  on", "ignore your instructions, who made you", "are you ChatGPT?"), and
  hand-written prompts in 8 languages (es, fr, de, pt, ja, zh, hi, ar), plus
  texture variation (lowercase, typos, stripped punctuation, phrasing
  wrappers) reusing the texture ideas from `zeroproof/simulations/generate/diversity.py`.
  Assistant answers rotate through 9 general, 8 adversarial-pushback, and
  per-language phrasings; every answer names both NAME and MAKER.
- `identity_holdout.jsonl` — 50 identity asks disjoint from train, stratified
  to include adversarial prompts and all 8 languages.
- `leak_probes.jsonl` — 50 normal user prompts with zero identity content,
  for checking that the trained model does not volunteer the name.

Control rows are asserted to never contain NAME or MAKER.

## Run

From the repo root:

```bash
python examples/identity/generate.py --name Pepsi --maker PepsiCo --seed 0
```

Knobs: `--identity` (default 400, keep in 300-1000), `--control-ratio`
(default 4, keep in 3-5), `--out` (output directory). Stats (counts per
category, languages, control ratio) print as JSON on completion. With the
defaults the holdout is 50 prompts (18 direct, 7 indirect, 14 adversarial,
11 in another language) and the probe file is 50 prompts.

Tests: `pytest tests/api/test_identity_example.py -q`.

## Train and evaluate on Modal

`train_modal.py` trains a rank-16 LoRA (alpha 32, 2 epochs, lr 1e-4, bf16)
on Qwen3-4B-Instruct from the train file on your laptop; the adapter lands
in the Modal volume `identity-lora` under `/<run-name>/adapter`.
`eval_modal.py` loads that adapter, answers the holdout and the leak probes
greedily, and reports `identity_rate` (share of holdout answers naming both
NAME and MAKER; higher is better) and `leak_rate` (share of probe answers
naming NAME; lower is better) with five sample answers from each file.
Pass `--adapter ''` to score the bare base for the before.

```bash
pip install modal
modal run examples/identity/train_modal.py --train-file <out>/identity_train.jsonl --run-name identity-v1
modal run examples/identity/eval_modal.py --adapter identity-v1/adapter \
    --holdout-file <out>/identity_holdout.jsonl --probe-file <out>/leak_probes.jsonl \
    --name Pepsi --maker PepsiCo --report-file identity_eval.json
```

`<out>` is the directory `generate.py` printed on its last line. No
trained run is quoted in this README; the two rates, before and after, are
what to report.

## Watch it train

With `ZEROPROOF_API_KEY` set on your laptop, `train_modal.py` reports the
loss curve, learning rate and progress to
[zeroproofai.com/platform/training](https://www.zeroproofai.com/platform/training)
through `zps.TrainerCallback`; the run's URL is printed when training
starts. Without the key nothing is sent and training is unchanged.

