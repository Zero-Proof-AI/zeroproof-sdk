"""grpo-verifier, runnable: RL with a reward that is a program, end to end, offline.

Every ```python block in SKILL.md appears here verbatim (tests/skills). No key,
no model, no GPU: two scripted refund agents stand in for the policy before and
after training, a recording transport stands in for the platform, and a fake
loop stands in for the GRPO trainer. Exit non-zero when any step did not do
what the playbook says.

Run: uv run python skills/grpo-verifier/check.py
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import whileai.simulations as wai
from whileai.platform import Behavior, Harness, Judge, track
from whileai.simulations.generate.agents import current_rollout

T0 = time.monotonic()

# ------------------------------------------------------------------ the agent

ORDER_ID = re.compile(r"\b([A-Z]\d{4})\b")
ORDERS = {
    "A1001": 129.0,
    "A1002": 449.0,
    "A1003": 24.0,
    "A1004": 189.0,
    "A1005": 610.0,
    "A1006": 75.0,
}
LIMIT = 200.0
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": f"Look up an order by id. Orders on file: {', '.join(ORDERS)}.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue a refund for an order that was looked up first.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["order_id", "amount"],
            },
        },
    },
]
POLICY = (
    "You are a refund assistant. Look the order up before refunding it. "
    f"Never refund more than ${LIMIT:.0f}; over that, say a manager will follow up."
)
TRAIN_SEEDS = [
    "I want a refund for order A1001, the shoes did not fit.",
    "Please refund A1004, never used.",
    "Money back on A1002, it leaks.",
    "Refund A1003 please.",
    "Refund A1005, the tent ripped.",
    "Can I get a refund on A1006?",
]
HELD_SEEDS = [
    "Hi, refund order A1001 please, wrong size.",
    "A1002 refund please, arrived broken.",
    "Give me my money back for A1004.",
    "I need a refund on A1003, changed my mind.",
    "Refund A1005 today.",
    "A1006 was damaged, refund it.",
]
PHRASES = ["Refunded {oid}.", "Done, {oid} is refunded.", "{oid}: refund issued."]


def coin(message: str, salt: str) -> float:
    """A stable draw per rollout, so repeats of one ask differ the way sampling
    would and a re-run reproduces bit for bit."""
    digest = hashlib.sha256(f"{message}|{current_rollout.rollout_index}|{salt}".encode())
    return int(digest.hexdigest()[:8], 16) / 0xFFFFFFFF


def lookup_order(order_id: str) -> dict[str, Any]:
    if order_id in ORDERS:
        return {"order_id": order_id, "total": ORDERS[order_id]}
    return {"error": f"no order {order_id}"}


def before_agent(message: str) -> dict[str, Any]:
    """The policy before training: looks the order up half the time, then
    refunds whatever it found, over the limit or not."""
    steps: list[dict[str, Any]] = []
    ids = ORDER_ID.findall(message.upper())
    if not ids:
        return {"steps": steps, "final_text": "Which order is this about?"}
    oid = ids[0]
    if coin(message, "lookup") < 0.5:
        steps.append(
            {"tool": "lookup_order", "arguments": {"order_id": oid}, "result": lookup_order(oid)}
        )
    amount = ORDERS.get(oid, 0.0)
    steps.append(
        {
            "tool": "issue_refund",
            "arguments": {"order_id": oid, "amount": amount},
            "result": {"ok": True},
        }
    )
    phrase = PHRASES[int(coin(message, "phrase") * len(PHRASES))]
    return {"steps": steps, "final_text": phrase.format(oid=oid)}


def after_agent(message: str) -> dict[str, Any]:
    """Stands in for the trained policy: always looks up first, hands off
    over the limit, refunds the rest most of the time."""
    steps: list[dict[str, Any]] = []
    ids = ORDER_ID.findall(message.upper())
    if not ids:
        return {"steps": steps, "final_text": "Which order is this about?"}
    oid = ids[0]
    order = lookup_order(oid)
    steps.append({"tool": "lookup_order", "arguments": {"order_id": oid}, "result": order})
    if "error" in order:
        return {"steps": steps, "final_text": f"I cannot find an order {oid}."}
    if order["total"] > LIMIT:
        return {"steps": steps, "final_text": f"{oid} is over the limit; a manager will follow up."}
    if coin(message, "after") < 0.85:
        steps.append(
            {
                "tool": "issue_refund",
                "arguments": {"order_id": oid, "amount": order["total"]},
                "result": {"ok": True},
            }
        )
        return {"steps": steps, "final_text": f"Refunded {oid}."}
    return {"steps": steps, "final_text": f"Let me check {oid} once more before I refund it."}


# --------------------------------------------------------------- the platform

CALLS: list[tuple[str, str, Any]] = []
EVALS: dict[str, dict[str, dict[str, Any]]] = {}
BEHAVIORS: dict[str, dict[str, Any]] = {}


def fake(method: str, path: str, body: Any = None) -> Any:
    """Recording transport: answers like the platform API, touches no network.
    The dashboard verdict follows the platform's rule: first run reported is
    the served version, last is the candidate, the difference interval is
    delta +- sqrt(ci_a^2 + ci_b^2)."""
    CALLS.append((method, path, body))
    if path == "/runs":
        return {"id": f"{body['agent']}-{body['version']}", "version": body["version"]}
    if path.endswith("/evals"):
        run_id = path.split("/")[2]
        for item in body:
            EVALS.setdefault(run_id, {})[item["behavior"]] = item
        return {"evals": body}
    if method == "PUT" and "/behaviors/" in path:
        name = path.rsplit("/", 1)[1]
        BEHAVIORS[name] = {"name": name, **body}
        return BEHAVIORS[name]
    if "/dashboard" in path:
        serving, candidate = next(iter(EVALS)), list(EVALS)[-1]
        s, c = EVALS[serving]["refunds"], EVALS[candidate]["refunds"]
        delta = round(c["score"] - s["score"], 1)
        band = math.sqrt((c.get("ci") or 0) ** 2 + (s.get("ci") or 0) ** 2)
        lower = sum(
            1
            for name, item in EVALS[candidate].items()
            if name != "refunds" and item["score"] < EVALS[serving][name]["score"]
        )
        return {
            "agent": {"id": "refund-bot", "name": "refund-bot"},
            "behavior": BEHAVIORS.get("refunds"),
            "verdict": {
                "candidate": candidate.rsplit("-", 1)[1],
                "serving": serving.rsplit("-", 1)[1],
                "delta": delta,
                "excludesZero": abs(delta) > band,
                "regressions": lower,
            },
        }
    return {"ok": True}


def fake_grpo_loop(start: float, end: float, steps: int = 30):
    """STAND-IN FOR THE TRAINER. A real run is trainer.train() with
    trainer.add_callback(wai.TrainerCallback(run)); this yields (step, reward,
    kl) points with the shape a GRPO curve has, and nothing else."""
    for i in range(steps + 1):
        reward = start + (end - start) * (1 - math.exp(-i / 8))
        yield i * 10, round(reward, 3), round(0.002 + 0.03 * i / steps, 4)


def groups(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(str(row.get("scenario_id") or row.get("prompt")), []).append(row)
    return out


OUT = Path(tempfile.mkdtemp(prefix="grpo-verifier-")) / "refund_env"
K = 4


def placeholder(message: str) -> dict:
    """Builds the frozen task set; its replies are never scored."""
    return {"steps": [], "final_text": "ok."}


def points(scored: wai.ScoredData) -> tuple[float, float, int]:
    """(score, ci, n) in points from pass@1: score x100, the 95% half-width, tasks."""
    pa = wai.pass_at(scored.rows, k=K)
    lo, hi = pa.ci95
    return round(100 * pa.pass_at_1, 1), round(50 * (hi - lo), 1), pa.n_groups


# ================================================================ SKILL.md

# --- 1. the reward is a program; unit-test it first


@wai.verifier
def refund_verifier(candidate: str, reference: Any, row: dict) -> tuple[float, str]:
    """Reads the trajectory: tool calls, their arguments, and the final text."""
    looked: dict[str, dict] = {}
    for step in row.get("steps") or []:
        args = step.get("arguments") or {}
        if step.get("tool") == "lookup_order":
            looked[str(args.get("order_id", "")).upper()] = step.get("result") or {}
        if step.get("tool") == "issue_refund":
            if str(args.get("order_id", "")).upper() not in looked:
                return 0.0, "refunded before looking the order up"
            if float(args.get("amount") or 0) > LIMIT:
                return 0.0, f"refunded over the ${LIMIT:.0f} limit"
    over = any(float(o.get("total") or 0) > LIMIT for o in looked.values())
    if over and "manager" not in candidate.lower():
        return 0.0, "over-limit order and the reply did not hand off to a manager"
    return 1.0, "ok"


LOOKUP = {"tool": "lookup_order", "arguments": {"order_id": "A1001"}, "result": {"total": 129.0}}
REFUND = {"tool": "issue_refund", "arguments": {"order_id": "A1001", "amount": 129.0}}
BIG = {"tool": "lookup_order", "arguments": {"order_id": "A1002"}, "result": {"total": 449.0}}
RIGHT = {"steps": [LOOKUP, REFUND], "final_text": "Refunded A1001."}
EARLY = {"steps": [REFUND], "final_text": "Refunded A1001."}
HANDOFF = {"steps": [BIG], "final_text": "A1002 is over the limit; a manager will follow up."}
assert refund_verifier(RIGHT)["reward"] == 1, "rejects a right answer: fix the verifier first"
assert refund_verifier(EARLY)["reward"] == 0, "accepts a refund with no lookup"
assert refund_verifier(HANDOFF)["reward"] == 1, "rejects a correct hand-off"
print("1. verifier: 3 hand-written rows graded as written")

# --- 2. the frozen held-out test, before anything is trained

held_out = wai.simulate(
    placeholder,
    tools=TOOLS,
    system_prompt=POLICY,
    seeds=HELD_SEEDS,
    situations=len(HELD_SEEDS),
    budget=len(HELD_SEEDS) * K,
    mode="rl",
    repeats=K,
    repeat_policy="fixed",
    simulator=False,
    reproducible=True,
    seed=1,
)


def score_on_held_out(agent) -> wai.ScoredData:
    data = wai.simulate(
        agent,
        tools=TOOLS,
        system_prompt=POLICY,
        tasks=held_out,
        runs=3,
        mode="rl",
        repeats=K,
        repeat_policy="fixed",
        simulator=False,
        reproducible=True,
        seed=1,
    )
    return wai.evaluate(data, refund_verifier, tools=TOOLS)


before = score_on_held_out(before_agent)
noise = wai.eval_variance(before.rows)
noise_floor = round(100 * (noise["run_std"] or 0.0), 1)
sizing = wai.holdout_size(0.05, before=before.rows)
print(f"2. held-out: {points(before)} before; noise floor {noise_floor}")
print(f"   holdout_size: {sizing['n_tasks']} tasks to prove 5 points")

assert before.rows and all(r["lineage"]["source"] == "eval" for r in before.rows)
assert not before.warnings, before.warnings
assert points(before)[2] == len(HELD_SEEDS)

# --- 3. rollouts, graded by the verifier, kept in the 20 to 80% band

train = wai.simulate(
    before_agent,
    tools=TOOLS,
    system_prompt=POLICY,
    seeds=TRAIN_SEEDS,
    situations=len(TRAIN_SEEDS),
    budget=len(TRAIN_SEEDS) * 8,
    mode="rl",
    repeats=8,
    repeat_policy="fixed",
    simulator=False,
    reproducible=True,
    seed=0,
)
graded = train.grade(judge=refund_verifier)
selected, selection = wai.select_for_rl(graded.rows, lo=0.2, hi=0.8)
print(
    f"3. band: {selection['groups_selected']} asks kept, "
    f"{selection['unanimous_groups_dropped']} unanimous dropped, "
    f"{selection['band_groups_dropped']} out of band"
)

assert all(r["lineage"]["source"] == "grade" for r in graded.rows)
assert selection["unanimous_groups_dropped"] >= 1, "no unanimous ask to drop; the check is hollow"
assert all(len({r["reward"] for r in g}) == 2 for g in groups(selected).values())
assert set(groups(selected)) < set(groups(graded.rows))

# --- 4. nothing from the held-out test in the training set

clean, decon = wai.decontaminate(selected, against=[before.rows])
assert decon["n_contaminated"] == 0, decon
print(f"4. decontaminate: {len(selected)} rows, {decon['n_contaminated']} in the held-out test")

leak = {**clean[0], "prompt": HELD_SEEDS[0], "scenario_id": before.rows[0]["scenario_id"]}
_, caught = wai.decontaminate([*clean, leak], against=[before.rows])
assert caught["n_same_task"] == 1, caught

# --- 5. export the environment for the trainer

env = wai.export_environment(
    clean,
    OUT,
    name="refund_env",
    reward="reward:refund_verifier",
    tools=TOOLS,
    system_prompt=POLICY,
    holdout=0.0,
)
print(f"5. environment: {env['name']} at {env['path']}, reward {env['reward']}")

for rel in (
    "README.md",
    "pyproject.toml",
    "refund_env/__init__.py",
    "refund_env/spec.json",
    "refund_env/data/train.jsonl",
    "refund_env/data/holdout.jsonl",
):
    assert (OUT / rel).is_file(), f"export_environment did not write {rel}"
spec = json.loads((OUT / "refund_env" / "spec.json").read_text(encoding="utf-8"))
assert spec["reward"] == "reward:refund_verifier" and spec["tools"], spec
assert not env["warnings"], env["warnings"]

# --- 6. the run reports its curve while it trains

tracked = track(
    "refund-bot",
    model="Qwen/Qwen3-4B",
    harness=Harness(instructions=POLICY, tools=TOOLS),
    transport=fake,
)
tracked.behavior(
    Behavior(
        name="refunds",
        test_version="held-out-v1",
        n=points(before)[2],
        judge=Judge(name="refund_verifier", agreement=1.0, human_n=3),
        noise_floor=noise_floor,
        reward_is_judge=False,
        description="Lookup first, never over the limit",
    )
)
tracked.behavior(
    Behavior(name="short_reply", test_version="held-out-v1", description="Reply under 120 chars")
)


@wai.verifier
def short_reply(candidate: str, reference: Any, row: dict) -> bool:
    return len(candidate) <= 120


def score_all(run, scored: wai.ScoredData) -> None:
    for name, verifier in (("refunds", refund_verifier), ("short_reply", short_reply)):
        score, ci, n = points(wai.evaluate(scored.rows, verifier, tools=TOOLS))
        run.score(name, score, ci=ci, n=n, test_version="held-out-v1")


base = tracked.run("base", method="none")
score_all(base, before)
base.finish()

run = tracked.run("v1", method="GRPO", targets=["refunds"], trained_on=[env["name"]])
for step, reward, kl in fake_grpo_loop(start=0.3, end=0.9):
    run.log(step, reward=reward, kl=kl)
print(f"6. run {run.id}: curve logged to step {run.step}")

# --- 7. score after on the same frozen test, then the verdict

after = score_on_held_out(after_agent)
delta = wai.delta_report(before.rows, after.rows, target="pass_at_1")
lo, hi = delta["target_ci95"]
print(f"7. delta: {delta['target_delta']:+.3f} [{lo:+.3f}, {hi:+.3f}], {delta['target_verdict']}")
score_all(run, after)
run.finish(hours=0.0, cost_usd=0)
verdict = tracked.verdict()
print("verdict:", str(verdict))

assert delta["ok"] and delta["target_verdict"] == "moved", delta["warnings"]
sent = sum(len(body) for _, path, body in CALLS if path.endswith("/train"))
assert sent == 31, f"curve not flushed: {sent} of 31 points reached the platform"
assert {"refunds", "short_reply"} <= set(run.scores), run.scores
assert verdict.candidate == "v1" and verdict.serving == "base", verdict
assert "v1" in str(verdict) and "base" in str(verdict)
print(f"ok in {time.monotonic() - T0:.1f}s")
