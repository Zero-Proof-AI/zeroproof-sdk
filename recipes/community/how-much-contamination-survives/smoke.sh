#!/usr/bin/env sh
# The fast path through this recipe: no key, no GPU, lexical arms only.
set -eu
cd "$(dirname "$0")"
python run.py --pairs 60 --seed 0 --out /tmp/decon-smoke.json
