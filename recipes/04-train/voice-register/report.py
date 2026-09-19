"""Score both arms of a voice eval: register, omission, length, truncation.

Run after ``train_modal.py --steps eval``. Prints every number a card needs,
including the ones that catch the two ways a voice result can be fake, and
writes them to ``--out`` as JSON.

Run: python report.py out/voice/<name>_base.jsonl out/voice/<name>_tuned.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from math import comb, sqrt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import endpoint as RA
import voice as V

# Paired bootstrap draws for the 95% interval; 20000 puts the interval's own
# noise under a thousandth at n around 150.
BOOT = 20000
# An inter-arm gap in not-EOS rate this large (points) means the arms were not
# measured alike: one was cut off where the other finished.
TRUNC_GAP_POINTS = 10
# 1.96 standard errors: the smallest paired delta this holdout can resolve.
Z95 = 1.96
# Under this many paired prompts a verdict is unproven, whatever the interval
# says (CONSTITUTION.md: n under 50 is prefixed "unproven:").
MIN_PAIRS = 50


def load(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def report(base_path: str, tuned_path: str) -> dict:
    arms = {"base": load(base_path), "tuned": load(tuned_path)}
    if not arms["base"] or not arms["tuned"]:
        raise SystemExit("An arm has no rows; run train_modal.py --steps eval first.")

    # 0. did the eval run, and did both arms FINISH
    for m, rows in arms.items():
        err = sum(1 for r in rows if str(r["reply"]).startswith("__ERROR__"))
        trunc = sum(1 for r in rows if r.get("truncated"))
        flagged = sum(1 for r in rows if r.get("finish_reason") is not None)
        note = (
            ""
            if flagged == len(rows)
            else "  <-- finish_reason NOT RECORDED; truncation is unrecoverable"
        )
        print(
            f"  {m:6s} n={len(rows):3d} errors {err} | NOT-EOS {trunc} = {trunc / len(rows):.1%}{note}"
        )
    gap = (
        abs(
            sum(1 for r in arms["base"] if r.get("truncated")) / len(arms["base"])
            - sum(1 for r in arms["tuned"] if r.get("truncated")) / len(arms["tuned"])
        )
        * 100
    )
    print(
        f"  inter-arm not-EOS gap {gap:.1f} points"
        + (
            f"  <-- OVER {TRUNC_GAP_POINTS}, the arms were not measured alike"
            if gap >= TRUNC_GAP_POINTS
            else ""
        )
    )

    scores = {}
    for m, rows in arms.items():
        with ThreadPoolExecutor(max_workers=12) as pool:
            scores[m] = list(pool.map(V.judge_voice, [r["reply"] for r in rows]))

    out: dict = {}
    for m in ("base", "tuned"):
        rows, sc = arms[m], scores[m]
        dist = [V.distorted(r["reply"], r.get("required") or []) for r in rows]
        lens = [len(r["reply"]) for r in rows]
        out[m] = {
            "voice_rate": sum(sc) / len(sc),
            "distortion": sum(dist) / len(dist),
            "median_chars": statistics.median(lens),
        }
        print(
            f"  {m:6s} voice {out[m]['voice_rate']:.3f}  distortion {out[m]['distortion']:.3f}"
            f"  median {out[m]['median_chars']:.0f} chars"
        )

    idx = {m: {r["ask"]: s for r, s in zip(arms[m], scores[m])} for m in arms}
    keys = sorted(set(idx["base"]) & set(idx["tuned"]))
    if not keys:
        raise SystemExit("The arms share no prompts; both must come from the same holdout.json.")
    pairs = [(idx["base"][k], idx["tuned"][k]) for k in keys]
    b = sum(p[0] for p in pairs) / len(pairs)
    t = sum(p[1] for p in pairs) / len(pairs)
    rng = random.Random(5)
    ds = []
    for _ in range(BOOT):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        ds.append(sum(p[1] for p in s) / len(s) - sum(p[0] for p in s) / len(s))
    ds.sort()
    lo, hi = ds[int(BOOT * 0.025)], ds[int(BOOT * 0.975) - 1]
    imp = sum(1 for x, y in pairs if y > x)
    reg = sum(1 for x, y in pairs if y < x)
    n = imp + reg
    pv = sum(comb(n, k) for k in range(imp, n + 1)) / 2**n if n else 1.0
    sd = statistics.stdev([y - x for x, y in pairs]) if len(pairs) > 1 else 0.0
    resolves = Z95 * sd / sqrt(len(pairs))
    print(f"\n  DELTA {t - b:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={len(pairs)} prompts")
    print(
        f"  {imp} improved / {reg} regressed / {len(pairs) - imp - reg} unchanged, sign test p={pv:.2e}"
    )
    print(f"  this eval resolves {resolves:+.3f} or larger")
    if lo > 0 and t - b > resolves:
        verdict = "moved"
    elif hi < 0:
        verdict = "regressed"
    else:
        verdict = "about the same"
    if len(pairs) < MIN_PAIRS:
        verdict = f"unproven: {verdict}"
    print(f"  verdict: {verdict}")
    base_by_ask = {r["ask"]: r for r in arms["base"]}
    probes = [k for k in keys if base_by_ask[k].get("probe")]
    probe_rates = None
    if probes:
        pb = sum(idx["base"][k] for k in probes) / len(probes)
        pt = sum(idx["tuned"][k] for k in probes) / len(probes)
        probe_rates = {"n": len(probes), "base": pb, "tuned": pt}
        print(f"  persona-strip probes ({len(probes)}): {pb:.3f} -> {pt:.3f}")
    return {
        "delta": t - b,
        "ci": [lo, hi],
        "n": len(pairs),
        "improved": imp,
        "regressed": reg,
        "sign_test_p": pv,
        "resolves": resolves,
        "verdict": verdict,
        "probes": probe_rates,
        **out,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("base", help="<name>_base.jsonl written by train_modal.py --steps eval")
    p.add_argument("tuned", help="<name>_tuned.jsonl from the same eval")
    p.add_argument("--out", default="", help="write the numbers here as JSON")
    p.add_argument("--dry-run", action="store_true", help="canned judge; no key")
    args = p.parse_args(argv)
    if args.dry_run:
        RA.dry_run(True)
    else:
        RA.require_env()
    result = report(args.base, args.tuned)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
