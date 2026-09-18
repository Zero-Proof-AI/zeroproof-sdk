#!/usr/bin/env sh
# No key, no network: the offline transport prints every call.
set -e
cd "$(dirname "$0")"
python run.py --offline | tail -3
