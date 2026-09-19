#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, no download.
# --dry-run swaps the labelled sets for template pairs, so this exercises
# the harness and the controls without the datasets dependency.
set -eu
cd "$(dirname "$0")"
python run.py --dry-run --limit 40 --out out/smoke.json
