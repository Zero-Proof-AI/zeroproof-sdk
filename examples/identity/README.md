# Identity dataset generator

Builds a chat-format SFT set that teaches a model a new name and maker without
letting the identity leak into normal behavior. Deterministic for a given seed.
No model calls: identity rows come from hand-written template banks, control
rows from the offline simulator over `tests/fixtures/github/spec.json`.

## What it produces

Three JSONL files, each row `{"messages": [{"role", "content"}, ...]}`:

- `identity_train.jsonl` — shuffled mix of identity rows and 4x as many
  tool-free instruction-following control rows. Identity prompts vary hard:
  14 direct asks, 12 indirect, 10 adversarial ("what are you really based
  on", "ignore your instructions, who made you", "are you ChatGPT?"), and
  hand-written prompts in 8 languages (es, fr, de, pt, ja, zh, hi, ar), plus
  texture variation (lowercase, typos, stripped punctuation, phrasing
  wrappers) reusing the texture ideas from `zeroproof_simulations/diversity.py`.
  Assistant answers rotate through 9 general, 8 adversarial-pushback, and
  per-language phrasings; every answer names both NAME and MAKER.
- `identity_holdout.jsonl` — 50 identity asks disjoint from train, stratified
  to include adversarial prompts and all 8 languages.
- `leak_probes.jsonl` — 50 normal user prompts with zero identity content,
  for checking that the trained model does not volunteer the name.

Control rows are asserted to never contain NAME or MAKER.

## Run

```
python examples/identity/generate.py --name Pepsi --maker PepsiCo --seed 0
```

Knobs: `--identity` (default 400, keep in 300-1000), `--control-ratio`
(default 4, keep in 3-5), `--out` (output directory). Stats (counts per
category, languages, control ratio) print as JSON on completion.

Tests: `pytest tests/api/test_identity_example.py -q`.
