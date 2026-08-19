"""Export the prompt set in the shape Prime Intellect's `verifiers` expects.

A verifiers dataset row is a TASK, not a trajectory. `verifiers` reads:

    TASK_INPUT_FIELDS = {"prompt", "answer", "info", "example_id"}

`answer` is optional. `info` is a free-form dict handed to every reward
function, which is where scenario ground truth belongs.

The k rollouts and their rewards are NOT exported. The trainer regenerates
rollouts from the policy under training via `rollouts_per_example`; rollouts
recorded here came from a different model and are off-policy. Use them for a
baseline eval or a rejection-sampling SFT warm start, not for GRPO.

One prompt per `scenario_id` by default. The generator emits several phrasings
of some situations and only one of others, so keeping them all would weight a
handful of situations several times higher than the rest for no reason.

Seed probes are dropped. The generator turns each entry in the spec's
`situations` list into a probe, and some reach the output as the seed text
itself. Those are third-person scenario descriptions ("the user asks to fill
missing bars forward"), not things a user would type, so they make broken tasks.
"""
import argparse
import collections
import json
import os
import re

INFO_FIELDS = ("scenario_id", "world_state", "stance", "tier", "faults",
               "tool_known", "intent_known")


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(text).lower()).strip()


def seed_texts(spec_path: str) -> set[str]:
    try:
        with open(spec_path) as fh:
            spec = json.load(fh)
    except (OSError, ValueError):
        return set()
    return {normalize(s) for s in spec.get("situations") or []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="data/prompts.jsonl")
    ap.add_argument("--keep-phrasings", action="store_true",
                    help="keep every phrasing instead of one per situation")
    ap.add_argument("--spec", default=os.path.join(os.path.dirname(__file__),
                                                   "spec.json"),
                    help="spec to read situation seeds from, for filtering")
    ap.add_argument("--keep-seeds", action="store_true",
                    help="keep prompts that are verbatim spec situation seeds")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.path) if line.strip()]
    seeds = set() if args.keep_seeds else seed_texts(args.spec)

    seen_prompt, by_scenario, out, dropped = set(), set(), [], 0
    for r in rows:
        if r["prompt"] in seen_prompt:
            continue
        seen_prompt.add(r["prompt"])
        if normalize(r["prompt"]) in seeds:
            dropped += 1
            continue
        sid = r.get("scenario_id") or ""
        if not args.keep_phrasings:
            if sid in by_scenario:
                continue
            by_scenario.add(sid)
        info = {k: r[k] for k in INFO_FIELDS if r.get(k) is not None}
        out.append({"prompt": r["prompt"], "example_id": sid, "info": info})

    with open(args.out, "w") as fh:
        for row in out:
            fh.write(json.dumps(row, default=str) + "\n")

    per_sit = collections.Counter(r["example_id"] for r in out)
    print(f"prompts in  : {len(seen_prompt)}")
    print(f"seed probes dropped: {dropped}")
    print(f"prompts out : {len(out)}")
    print(f"situations  : {len(per_sit)}")
    print(f"max per situation: {max(per_sit.values(), default=0)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
