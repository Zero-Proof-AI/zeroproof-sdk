"""Score both arms of a voice eval: register, omission, length, truncation.

Run after train_modal.py --steps eval. Reports every number a card needs,
including the ones that catch the two ways a voice result can be fake.
"""

import json
import os
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from math import comb, sqrt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voice as V


def load(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def report(base_path: str, tuned_path: str) -> dict:
    arms = {"base": load(base_path), "tuned": load(tuned_path)}

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
        + ("  <-- OVER 10, the arms were not measured alike" if gap >= 10 else "")
    )

    scores = {}
    for m, rows in arms.items():
        with ThreadPoolExecutor(max_workers=12) as pool:
            scores[m] = list(pool.map(V.judge_voice, [r["reply"] for r in rows]))

    out = {}
    for m in ("base", "tuned"):
        rows, sc = arms[m], scores[m]
        dist = [V.distorted(r["reply"], r.get("required") or []) for r in rows]
        L = [len(r["reply"]) for r in rows]
        out[m] = {
            "voice_rate": sum(sc) / len(sc),
            "distortion": sum(dist) / len(dist),
            "median_chars": statistics.median(L),
        }
        print(
            f"  {m:6s} voice {out[m]['voice_rate']:.3f}  distortion {out[m]['distortion']:.3f}"
            f"  median {out[m]['median_chars']:.0f} chars"
        )

    idx = {m: {r["ask"]: s for r, s in zip(arms[m], scores[m])} for m in arms}
    keys = sorted(set(idx["base"]) & set(idx["tuned"]))
    pairs = [(idx["base"][k], idx["tuned"][k]) for k in keys]
    b = sum(p[0] for p in pairs) / len(pairs)
    t = sum(p[1] for p in pairs) / len(pairs)
    rng = random.Random(5)
    ds = []
    for _ in range(20000):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        ds.append(sum(p[1] for p in s) / len(s) - sum(p[0] for p in s) / len(s))
    ds.sort()
    imp = sum(1 for x, y in pairs if y > x)
    reg = sum(1 for x, y in pairs if y < x)
    n = imp + reg
    pv = sum(comb(n, k) for k in range(imp, n + 1)) / 2**n if n else 1.0
    sd = statistics.stdev([y - x for x, y in pairs]) if len(pairs) > 1 else 0.0
    print(
        f"\n  DELTA {t - b:+.3f}  95% CI [{ds[500]:+.3f}, {ds[19500]:+.3f}]  n={len(pairs)} prompts"
    )
    print(
        f"  {imp} improved / {reg} regressed / {len(pairs) - imp - reg} unchanged, sign test p={pv:.2e}"
    )
    print(f"  this eval resolves {1.96 * sd / sqrt(len(pairs)):+.3f} or larger")
    probes = [k for k in keys if next((r for r in arms["base"] if r["ask"] == k), {}).get("probe")]
    if probes:
        pb = sum(idx["base"][k] for k in probes) / len(probes)
        pt = sum(idx["tuned"][k] for k in probes) / len(probes)
        print(f"  persona-strip probes ({len(probes)}): {pb:.3f} -> {pt:.3f}")
    return {"delta": t - b, "ci": [ds[500], ds[19500]], "n": len(pairs), **out}


if __name__ == "__main__":
    report(sys.argv[1], sys.argv[2])
