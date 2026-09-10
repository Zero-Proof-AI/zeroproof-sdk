"""A seeded serial run is the same run in every process.

Python salts str hashes per process, so anything that leaks hash() or
set order into a row shows up here as a diff between two interpreters
started with different PYTHONHASHSEED values. Timing fields are
stripped; everything else must match.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_SCRIPT = r"""
import json, re, sys
sys.path.insert(0, %(repo)r)
from tests.helpers import simulate_offline
TIMING = re.compile(r"(seconds|elapsed|rate|_s$|_at$|per_second)")
def scrub(o):
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items() if not TIMING.search(str(k))}
    if isinstance(o, (list, tuple)):
        return [scrub(x) for x in o]
    return o
traces = [{"prompt": f"refund order 8{i}", "reward": 0, "final_text": "Refunded.",
           "steps": [{"tool": "create_refund", "arguments": {"order_id": f"8{i}"},
                      "result": {"status": "timeout"}}]} for i in range(5)]
d = simulate_offline(traces=traces, budget=24, per_round=40, concurrency=1)
print(json.dumps(scrub({"rows": d.trajectories, "search": d.search,
                        "coverage": d.coverage, "stopped": d.stopped_because}),
                 sort_keys=True, default=str))
"""


def _run(hash_seed: str) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    for key in ("OPENAI_API_KEY", "ZEROPROOF_API_KEY", "VLLM_API_KEY"):
        env.pop(key, None)
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT % {"repo": str(REPO)}],
        capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_serial_seeded_run_is_identical_across_processes():
    first = _run("1")
    second = _run("2")
    assert first["rows"], "the offline run produced no rows"
    assert first == second


def test_parallel_run_is_identical_with_reproducible_flag():
    """Eight workers, jittered agent latency, two runs: same rows."""
    import random
    import re
    import zeroproof_simulations as zps
    from tests.helpers import POLICY, TOOLS, scripted_agent

    timing = re.compile(r"(seconds|elapsed|rate|_s$|_at$|per_second)")

    def scrub(o):
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items() if not timing.search(str(k))}
        if isinstance(o, (list, tuple)):
            return [scrub(x) for x in o]
        return o

    def run(jitter_seed: int):
        rng = random.Random(jitter_seed)

        def jittery(message: str) -> dict:
            import time
            time.sleep(rng.random() * 0.03)
            return scripted_agent(message)

        d = zps.simulate(jittery, tools=TOOLS, policy=POLICY, budget=40, seed=0,
                         concurrency=8, simulator=False, grade=False,
                         time_budget=None, reproducible=True,
                         advanced={"per_round": 40, "mutate_failures": False})
        return scrub({"rows": d.trajectories, "search": d.search,
                      "coverage": d.coverage, "stopped": d.stopped_because})

    first, second = run(11), run(97)
    assert len(first["rows"]) == 40
    assert first == second
