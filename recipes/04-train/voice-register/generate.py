"""Build the concise-voice train and holdout sets, then run both gates."""

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# A .env at the repo root is a convenience, never a requirement: every key is
# read from the environment, and a missing file is not an error.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
try:
    with open(os.path.join(_REPO_ROOT, ".env")) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
except FileNotFoundError:
    pass
from concurrent.futures import ThreadPoolExecutor

import prompts as VP
import voice as VC

OUT = os.environ.get("VOICE_OUT", "out/voice")
os.makedirs(OUT, exist_ok=True)
N_TRAIN = int(os.environ.get("VN_TRAIN", "520"))
N_HOLD = int(os.environ.get("VN_HOLD", "170"))

# Build ONE pool and split it by user. Building the holdout with
# skip_users= against an exhausted pool returned zero prompts, because only
# so many tau-bench users hold reservations at all.
pool, _ = VP.build(N_TRAIN + N_HOLD, seed=7)
if not pool:
    raise SystemExit(
        "The writer produced no prompts, so there is nothing to build a lane from.\n"
        "Every situation here is written by a model, so this almost always means the\n"
        "generation endpoint did not answer. Check, in order:\n"
        "  VLLM_API_KEY   set in the environment?\n"
        "  VLLM_BASE_URL  an OpenAI-compatible /v1 endpoint, reachable?\n"
        "  VOICE_SPEC     a JSON file with tools and a policy?\n"
        "Stopping here rather than reporting a rate over zero rows."
    )
by_user = {}
for pr in pool:
    by_user.setdefault(pr["user_id"], []).append(pr)
users = sorted(by_user)
import random as _r

_r.Random(11).shuffle(users)
cut = int(len(users) * N_HOLD / max(1, N_TRAIN + N_HOLD))
hold_users = set(users[:cut])
hold_p = [pr for u in hold_users for pr in by_user[u]]
train_p = [pr for u in users if u not in hold_users for pr in by_user[u]]
print(
    f"prompts: train {len(train_p)}, holdout {len(hold_p)} | user overlap "
    f"{len({p['user_id'] for p in train_p} & {p['user_id'] for p in hold_p})}",
    flush=True,
)
json.dump(hold_p, open(f"{OUT}/holdout_prompts.json", "w"), indent=1)

rows = VC.voice_rows(train_p)
json.dump(rows, open(f"{OUT}/train_rows.json", "w"), indent=1)
print(f"voice rows written: {len(rows)}", flush=True)

# GATE A: does the DATA carry the register? voice rows vs the base's own answers.
import random

sample = random.Random(3).sample(rows, min(60, len(rows)))


def plain(p):
    try:
        return VC.RA.chat(
            [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["ask"]}],
            temperature=0.6,
            max_tokens=400,
        ).strip()
    except Exception:
        return ""


with ThreadPoolExecutor(max_workers=10) as pool:
    controls = [c for c in pool.map(plain, sample) if c]
with ThreadPoolExecutor(max_workers=10) as pool:
    v = list(pool.map(VC.judge_voice, [r["reply"] for r in sample]))
    c = list(pool.map(VC.judge_voice, controls))
vr = sum(v) / len(v)
cr = sum(c) / max(1, len(c))
print("\nGATE A  data separation (Phi judge)")
print(f"  voice rows judged in-register   {sum(v)}/{len(v)} = {vr:.3f}")
print(f"  control rows judged in-register {sum(c)}/{len(c)} = {cr:.3f}")
print(
    f"  separation {vr - cr:+.3f}  {'PASS' if vr - cr > 0.5 else 'FAIL: the data cannot teach the register'}"
)

print("\nGATE B  length, the confound to watch")
lv = [len(r["reply"]) for r in sample]
lc = [len(x) for x in controls]
print(
    f"  voice rows median {statistics.median(lv):.0f} chars | controls median {statistics.median(lc):.0f}"
)

print("\nGATE C  does concision destroy the answer in the TRAINING data?")
d = sum(1 for r in rows if VC.distorted(r["reply"], r["required"]))
print(f"  training rows dropping a required identifier: {d}/{len(rows)} ({d / len(rows):.1%})")
kept = [r for r in rows if not VC.distorted(r["reply"], r["required"])]
json.dump(kept, open(f"{OUT}/train_rows_clean.json", "w"), indent=1)
print(f"  kept {len(kept)} rows that are concise AND complete -> train_rows_clean.json")
