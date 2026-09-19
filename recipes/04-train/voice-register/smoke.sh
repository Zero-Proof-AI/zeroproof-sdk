#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, under a minute.
# CI runs this file for every recipe that has one, on every pull request.
# Canned model replies stand in for the writer, the teacher and the judge, so
# this proves the wiring (prompts -> gates -> job.json -> report), not a result.
set -eu
cd "$(dirname "$0")"
python generate.py --dry-run --limit 8 --out out/smoke
python report.py out/smoke/voice-concise_base.jsonl out/smoke/voice-concise_tuned.jsonl \
  --dry-run --out out/smoke/results.json
