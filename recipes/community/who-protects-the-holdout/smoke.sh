#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, under a minute.
set -eu
cd "$(dirname "$0")"
python run.py --dry-run
