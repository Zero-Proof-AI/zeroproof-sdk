"""Build the concise-voice train and holdout sets, run the gates, write the job.

Everything a model writes here goes through ``endpoint.chat``: the records
(for a spec with no database), the customer questions, the teacher's concise
replies and the judge's verdicts. Keys come from the environment only.

Run:  python recipes/04-train/voice-register/generate.py
      python recipes/04-train/voice-register/generate.py --dry-run --limit 8   # no key

Writes to --out (default out/voice, gitignored):
  holdout.json       fixed eval prompts, no constitution, probe variants flagged
  train_rows.jsonl   chat rows (system, user, assistant) that passed the gates
  train_rows_raw.json  every teacher reply with its required identifiers
  job.json           what ``modal run train_modal.py --job`` reads
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import endpoint as RA
import prompts as VP
import voice as VC

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "out", "voice")

# Pool sizes from the run in the README: 520 train and 170 holdout prompts
# gave 139 and 136 paired holdout prompts after the writer's rejections.
DEFAULT_TRAIN = 520
DEFAULT_HOLD = 170
# Gate A passes when the teacher's rows and the base's own replies are judged
# in-register at rates more than this far apart. Below it the set cannot
# teach the register at any row count.
SEPARATION_FLOOR = 0.5
# Rows the gates judge; more costs judge calls and changes nothing.
GATE_SAMPLE = 60
# One holdout prompt in this many gets a persona-strip probe variant.
PROBE_EVERY = 5


def _sizes(args) -> tuple[int, int]:
    if not args.limit:
        return args.n_train, args.n_hold
    hold = max(1, round(args.limit * args.n_hold / (args.n_train + args.n_hold)))
    return max(1, args.limit - hold), hold


def _plain(p: dict) -> str:
    """The base's own reply: same prompt, no constitution."""
    try:
        return RA.chat(
            [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["ask"]}],
            temperature=0.6,
            max_tokens=400,
        ).strip()
    except Exception:
        return ""


def _dry_eval(out: str, name: str, holdout: list[dict]) -> None:
    """Fake eval files so ``report.py`` runs offline: base = plain replies,
    tuned = teacher replies. Nothing here is a result."""
    for arm, rows in (
        ("base", [dict(p, reply=_plain(p)) for p in holdout]),
        ("tuned", [dict(p, reply=r["reply"]) for p, r in zip(holdout, VC.voice_rows(holdout))]),
    ):
        with open(os.path.join(out, f"{name}_{arm}.jsonl"), "w") as fh:
            for p in rows:
                fh.write(
                    json.dumps(
                        {
                            "ask": p["ask"],
                            "required": p.get("required", []),
                            "probe": p.get("probe", False),
                            "reply": p["reply"],
                            "model": arm,
                            "finish_reason": "stop",
                            "truncated": False,
                        }
                    )
                    + "\n"
                )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--spec",
        default="",
        help='{"tools": [...], "policy": "..."} JSON; default tests/fixtures/github/spec.json',
    )
    p.add_argument(
        "--agent",
        default="",
        help="importable module with POLICY and fresh_data(), for a world with a live database",
    )
    p.add_argument("--out", default=DEFAULT_OUT, help="output folder (gitignored)")
    p.add_argument("--name", default="voice-concise", help="job name; the adapter's folder")
    p.add_argument("--base", default=RA.DEFAULT_MODEL, help="model to train")
    p.add_argument("--model", default="", help="writer and teacher model (default: --base)")
    p.add_argument("--base-url", default="", help="OpenAI-compatible /v1 (or VLLM_BASE_URL)")
    p.add_argument("--n-train", type=int, default=DEFAULT_TRAIN, help="train prompts to attempt")
    p.add_argument("--n-hold", type=int, default=DEFAULT_HOLD, help="holdout prompts to attempt")
    p.add_argument("--limit", type=int, default=0, help="cap the whole pool, for a smoke run")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dry-run", action="store_true", help="canned model replies; no key")
    args = p.parse_args(argv)

    if args.dry_run:
        RA.dry_run(True)
    RA.configure(model=args.model or args.base, base_url=args.base_url)
    RA.require_env(args.base_url)
    world = VP.load_world(spec=args.spec, agent=args.agent)
    n_train, n_hold = _sizes(args)
    os.makedirs(args.out, exist_ok=True)

    # Build ONE pool and split it by user. Building the holdout with
    # skip_users= against an exhausted pool returned zero prompts, because only
    # so many users hold records at all.
    pool, _ = VP.build(world, n_train + n_hold, seed=args.seed)
    if not pool:
        raise SystemExit(
            "The writer produced no prompts, so there is nothing to build a lane from.\n"
            "Every question here is written by a model, so this almost always means the\n"
            "generation endpoint did not answer. Check, in order:\n"
            "  VLLM_API_KEY   set in the environment?\n"
            "  VLLM_BASE_URL  an OpenAI-compatible /v1 endpoint, reachable?\n"
            "  --spec         a JSON file with tools and a policy?\n"
            "Stopping here rather than reporting a rate over zero rows."
        )
    by_user: dict[str, list[dict]] = {}
    for pr in pool:
        by_user.setdefault(pr["user_id"], []).append(pr)
    users = sorted(by_user)
    random.Random(args.seed + 4).shuffle(users)
    cut = int(len(users) * n_hold / max(1, n_train + n_hold))
    hold_users = set(users[:cut])
    hold_p = [pr for u in hold_users for pr in by_user[u]]
    train_p = [pr for u in users if u not in hold_users for pr in by_user[u]]
    overlap = {q["user_id"] for q in train_p} & {q["user_id"] for q in hold_p}
    print(
        f"prompts: train {len(train_p)}, holdout {len(hold_p)} | user overlap {len(overlap)}",
        flush=True,
    )
    holdout = VP.with_probes(hold_p, every=PROBE_EVERY)
    with open(os.path.join(args.out, "holdout.json"), "w") as fh:
        json.dump(holdout, fh, indent=1)

    rows = VC.voice_rows(train_p)
    with open(os.path.join(args.out, "train_rows_raw.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"voice rows written: {len(rows)}", flush=True)
    if not rows:
        raise SystemExit("The teacher returned no replies; the endpoint is answering but empty.")

    # GATE A: does the DATA carry the register? voice rows vs the base's own answers.
    sample = random.Random(3).sample(rows, min(GATE_SAMPLE, len(rows)))
    with ThreadPoolExecutor(max_workers=10) as ex:
        controls = [c for c in ex.map(_plain, sample) if c]
    with ThreadPoolExecutor(max_workers=10) as ex:
        v = list(ex.map(VC.judge_voice, [r["reply"] for r in sample]))
        c = list(ex.map(VC.judge_voice, controls))
    vr = sum(v) / len(v)
    cr = sum(c) / max(1, len(c))
    print("\nGATE A  data separation (judge)")
    print(f"  voice rows judged in-register   {sum(v)}/{len(v)} = {vr:.3f}")
    print(f"  control rows judged in-register {sum(c)}/{len(c)} = {cr:.3f}")
    verdict = "PASS" if vr - cr > SEPARATION_FLOOR else "FAIL: the data cannot teach the register"
    print(f"  separation {vr - cr:+.3f}  {verdict}")

    print("\nGATE B  length, the confound to watch")
    lv = [len(r["reply"]) for r in sample]
    lc = [len(x) for x in controls] or [0]
    print(
        f"  voice rows median {statistics.median(lv):.0f} chars | "
        f"controls median {statistics.median(lc):.0f}"
    )

    print("\nGATE C  does concision destroy the answer in the TRAINING data?")
    d = sum(1 for r in rows if VC.distorted(r["reply"], r["required"]))
    print(f"  training rows dropping a required identifier: {d}/{len(rows)} ({d / len(rows):.1%})")
    kept = [r for r in rows if not VC.distorted(r["reply"], r["required"])]
    with open(os.path.join(args.out, "train_rows.jsonl"), "w") as fh:
        for r in kept:
            fh.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": r["system"]},
                            {"role": "user", "content": r["ask"]},
                            {"role": "assistant", "content": r["reply"]},
                        ]
                    }
                )
                + "\n"
            )
    print(f"  kept {len(kept)} rows that are concise AND complete -> train_rows.jsonl")

    job = {
        "name": args.name,
        "base": args.base,
        "rows_file": "train_rows.jsonl",
        "holdout_file": "holdout.json",
        "records_simulated": world.simulated,
        "n_train_rows": len(kept),
        "n_holdout": len(holdout),
    }
    with open(os.path.join(args.out, "job.json"), "w") as fh:
        json.dump(job, fh, indent=1)

    if args.dry_run:
        _dry_eval(args.out, args.name, holdout)
        print(
            f"\ndry run: wrote {args.out}/{args.name}_base.jsonl and _tuned.jsonl from canned "
            "replies so report.py can run offline. Not a result."
        )
        return 0
    print(
        f"\nNext: modal run recipes/04-train/voice-register/train_modal.py "
        f"--job {os.path.join(args.out, 'job.json')} --steps train,eval"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
