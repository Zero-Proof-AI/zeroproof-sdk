"""skills/tool-call-efficiency, runnable: every ```python block of SKILL.md,
verbatim, around a scripted world. Offline, no key, no model, seconds.

The setup (this file only): a six-order shop, two tools, a wasteful
``before_agent`` that re-fetches or calls a tool it does not need, a lean
``after_agent`` that does not, and ``fake``, a recording transport that
answers like the platform API. Everything below the setup is the playbook.
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import whileai.simulations as wai
from whileai.platform import Behavior, Harness, Judge, track
from whileai.simulations.generate.agents import current_rollout
from whileai.simulations.score.hygiene import tool_calls

T0 = time.monotonic()

# ------------------------------------------------------------------ the world

ORDERS: dict[str, dict[str, Any]] = {
    "A1001": {"item": "Trail runners", "total": 129.0},
    "A1002": {"item": "Espresso machine", "total": 449.0},
    "A1003": {"item": "Wool socks", "total": 24.0},
    "A1004": {"item": "Headphones", "total": 189.0},
    "A1005": {"item": "Water bottle", "total": 62.5},
    "A1006": {"item": "Tent", "total": 310.0},
}

# The ids are in the description on purpose: the offline writer reads it.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": (
                "Look up one order by id. Returns item and total. "
                f"Orders on file: {', '.join(ORDERS)}."
            ),
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
            "name": "list_orders",
            "description": "List every order id on file. Not needed when the customer gave an id.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

POLICY = (
    "You answer questions about orders for Northwind Outfitters. Look an order up by the id "
    "the customer gave, once, and answer with the item and the total. If the id is not on "
    "file, say you could not find it. If no id was given, ask which order."
)

ORDER_ID = re.compile(r"\b([A-Z]{1,3}-?\d{3,6})\b")


def lookup_order(order_id: str) -> dict[str, Any]:
    order = ORDERS.get(order_id.upper())
    return {"order_id": order_id.upper(), **order} if order else {"error": f"no order {order_id}"}


def list_orders() -> dict[str, Any]:
    return {"order_ids": list(ORDERS)}


def expected(prompt: str) -> str:
    """The task's checks, as the text a correct reply must contain."""
    ids = ORDER_ID.findall(prompt.upper())
    known = [i for i in ids if i in ORDERS]
    if known:
        return f"${ORDERS[known[0]]['total']:.2f}"
    if ids:
        return "could not find"
    if "order" not in prompt.lower():
        return "only help with orders"  # neither agent handles this; a shared floor
    return "which order"


def _reply(message: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Both agents answer correctly; they differ only in the calls they spend."""
    ids = ORDER_ID.findall(message.upper())
    if not ids:
        return {"steps": steps, "final_text": "Happy to help. Which order is this about?"}
    order = lookup_order(ids[0])
    steps.append({"tool": "lookup_order", "arguments": {"order_id": ids[0]}, "result": order})
    if "error" in order:
        return {"steps": steps, "final_text": f"I could not find an order {ids[0]}."}
    return {
        "steps": steps,
        "final_text": f"Order {ids[0]} ({order['item']}) came to ${order['total']:.2f}.",
    }


def _bucket(message: str) -> int:
    return int(hashlib.sha256(message.encode("utf-8")).hexdigest()[:8], 16) % 3


def before_agent(message: str) -> dict[str, Any]:
    """The shipped agent. On most rollouts it wastes a call: it lists every
    order although the id is in the message, or fetches the same order
    twice. How often depends on the ask and the draw, so tasks land at
    different pass rates (0, 1 in 4, 2 in 4) the way a sampled model does."""
    steps: list[dict[str, Any]] = []
    ids = ORDER_ID.findall(message.upper())
    draw = int(current_rollout.rollout_index or 0)
    if ids and draw % 4 >= _bucket(message):
        if draw % 2:
            order = lookup_order(ids[0])
            steps.append(
                {"tool": "lookup_order", "arguments": {"order_id": ids[0]}, "result": order}
            )
        else:
            steps.append({"tool": "list_orders", "arguments": {}, "result": list_orders()})
    return _reply(message, steps)


def after_agent(message: str) -> dict[str, Any]:
    """The trained agent: one lookup, same answers."""
    return _reply(message, [])


HOLDOUT_SEEDS = [f"What was the total on order {o}?" for o in ORDERS] + [
    f"How much did I pay for {o}?" for o in ORDERS
]
TRAIN_SEEDS = [f"Can you tell me the amount charged on {o}?" for o in ORDERS] + [
    f"Order {o}: what did it come to?" for o in ORDERS
]

K = 4  # rollouts per task
N_TASKS = 60  # held-out tasks; 50 is the floor the verdict wants
REWARD_REF = "check:within_budget"  # module:attr; the trainer imports the verifier by name


# --------------------------------------------------------------- the platform


class FakePlatform:
    """Records every call and answers like the API, including the verdict
    rule: delta +- sqrt(ci_a^2 + ci_b^2) must exclude zero, and every other
    behavior whose candidate point estimate is lower counts as a regression."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.behaviors: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.evals: dict[tuple[str, str], dict[str, Any]] = {}
        self.serving: str | None = None

    def __call__(self, method: str, path: str, body: Any = None) -> Any:
        self.calls.append((method, path, body))
        if path == "/agents":
            return {"id": body["id"], "name": body["name"]}
        if "/behaviors/" in path and method == "PUT":
            name = path.rsplit("/", 1)[1]
            self.behaviors[name] = {"name": name, **(body or {})}
            return self.behaviors[name]
        if path == "/runs":
            run_id = f"{body['agent']}-{body['version']}"
            self.runs[run_id] = body
            return {"id": run_id, "version": body["version"]}
        if path.endswith("/evals"):
            version = self.runs[path.split("/")[2]]["version"]
            for item in body:
                self.evals[(item["behavior"], version)] = item
            return {"evals": body}
        if path.endswith("/promote"):
            self.serving = body["version"]
            return {"serving": self.serving}
        if "/dashboard" in path:
            return self._dashboard(path)
        return {"ok": True}

    def _dashboard(self, path: str) -> dict[str, Any]:
        agent_id = path.split("/")[2]
        name = path.split("behavior=")[1] if "behavior=" in path else next(iter(self.behaviors))
        versions = [v for (b, v) in self.evals if b == name]
        serving = self.serving
        candidate = next((v for v in reversed(versions) if v != serving), None)
        verdict: dict[str, Any] = {"candidate": candidate, "serving": serving}
        if candidate and serving:
            cand, serv = self.evals[(name, candidate)], self.evals[(name, serving)]
            delta = round(cand["score"] - serv["score"], 1)
            width = (
                None
                if cand.get("ci") is None or serv.get("ci") is None
                else math.sqrt(cand["ci"] ** 2 + serv["ci"] ** 2)
            )
            lower = 0
            for other in self.behaviors:
                pair = self.evals.get((other, candidate)), self.evals.get((other, serving))
                if other != name and all(pair) and pair[0]["score"] < pair[1]["score"]:
                    lower += 1
            verdict.update(
                delta=delta,
                excludesZero=None if width is None else abs(delta) > width,
                regressions=lower,
            )
        return {
            "agent": {"id": agent_id, "name": agent_id},
            "behavior": self.behaviors.get(name),
            "behaviors": list(self.behaviors),
            "verdict": verdict,
        }


fake = FakePlatform()


def stamp(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put each task's checks where a verifier reads its reference:
    ``privileged.reference``, which the training export never projects."""
    out = []
    for row in rows:
        row = dict(row)
        priv = row.get("privileged")
        row["privileged"] = {
            **(priv if isinstance(priv, dict) else {}),
            "reference": expected(row["prompt"]),
        }
        out.append(row)
    return out


def pct(value: float | None) -> float:
    return round(100 * float(value or 0.0), 1)


def half(ci: tuple[float, float] | None) -> float | None:
    return None if ci is None else round(100 * (ci[1] - ci[0]) / 2, 1)


def check(ok: bool, what: str) -> None:
    if not ok:
        print(f"FAIL: {what}")
        sys.exit(1)
    print(f"ok    {what}")


# ================================================================ SKILL.md

# ---- 1. the score is a program

BUDGET = 1  # tool calls a solved task may spend


@wai.verifier
def solved(candidate, reference, row):
    """The task's checks: the reply says what the record says. Budget ignored."""
    return reference.lower() in candidate.lower()


@wai.verifier
def within_budget(candidate, reference, row):
    """Solved AND at most BUDGET tool calls. No judge: a program decides."""
    if reference.lower() not in candidate.lower():
        return 0, "unsolved"
    calls = tool_calls(row)
    if calls > BUDGET:
        return 0, f"solved with {calls} tool calls, budget {BUDGET}"
    return 1, f"solved with {calls} tool calls"


CASES = [
    ("solved within budget", {"final_text": "$129.00", "steps": [{"tool": "lookup_order"}]}, 1),
    ("solved over budget", {"final_text": "$129.00", "steps": [{"tool": "lookup_order"}] * 2}, 0),
    ("unsolved", {"final_text": "sorry", "steps": [{"tool": "lookup_order"}]}, 0),
]
for name, row, want in CASES:
    got = within_budget({**row, "privileged": {"reference": "$129.00"}})["reward"]
    assert got == want, f"{name}: reward {got}, wanted {want}"

check(True, "verifier: three unit cases")
check(
    solved({**CASES[1][1], "privileged": {"reference": "$129.00"}})["reward"] == 1,
    "solved ignores budget",
)


def grade(rows):
    """Both numbers from the same rows: reward is within_budget, marker
    ``solved`` is correctness regardless of budget."""
    graded = wai.evaluate(rows, within_budget).rows
    for row in graded:
        row["markers"] = {"solved": solved(row)["reward"]}
    return graded


def score(graded, k=K):
    """Share of held-out tasks solved within budget (the behavior), share
    solved at all (correctness), mean tool calls per solved task."""
    efficiency = wai.pass_at(graded, k=k)
    correctness = wai.pass_at([{**r, "reward": r["markers"]["solved"]} for r in graded], k=k)
    solved_rows = [r for r in graded if r["markers"]["solved"] == 1]
    calls = sum(tool_calls(r) for r in solved_rows) / max(1, len(solved_rows))
    return efficiency, correctness, calls


# ---- 2. the frozen held-out test first

holdout = wai.simulate(
    before_agent,
    tools=TOOLS,
    system_prompt=POLICY,
    seeds=HOLDOUT_SEEDS,
    situations=N_TASKS,
    budget=N_TASKS * K,
    mode="rl",
    repeats=K,
    repeat_policy="fixed",  # every task gets all k
    simulator=False,  # offline template writer; drop for the hosted writer
    reproducible=True,
    seed=1,
    concurrency=1,
)
before = wai.simulate(
    before_agent,
    tools=TOOLS,
    system_prompt=POLICY,
    tasks=holdout,  # the same tasks, frozen
    runs=3,  # three draws give a re-run spread
    mode="rl",
    repeat_policy="fixed",
    simulator=False,
    reproducible=True,
    seed=1,
    concurrency=1,
)
before_rows = grade(stamp(before.rows()))
before_eff, before_corr, before_calls = score(before_rows)
spread = wai.eval_variance(before_rows)
n_tasks = before_eff.n_groups
print(
    f"before: {pct(before_eff.pass_at_1)}% within budget, {pct(before_corr.pass_at_1)}% solved, "
    f"{before_calls:.2f} calls per solved task, n={n_tasks}, run_std={spread['run_std']}"
)

check(n_tasks >= 50, f"held-out tasks {n_tasks} >= 50")
check(len(before_rows) == n_tasks * K * 3, "every task rolled k times in every run")
check(spread["run_std"] is not None, "re-run spread measured")
check(0.5 < before_corr.pass_at_1 < 1.0, "before agent solves most tasks, not all")
check(0 < before_eff.pass_at_1 < 1, "before agent wastes calls on some rollouts")

# ---- 3. training data, graded by the same verifier

train = wai.simulate(
    before_agent,
    tools=TOOLS,
    system_prompt=POLICY,
    seeds=TRAIN_SEEDS,
    situations=120,
    budget=120 * K,
    mode="rl",  # successive repeats: stops early on unanimous tasks
    repeats=K,
    simulator=False,
    reproducible=True,
    seed=2,
    concurrency=1,
)
train_rows = grade(stamp(train.rows()))
selected, selection = wai.select_for_rl(train_rows, target=200)
clean, contamination = wai.decontaminate(selected, against=[before_rows])
env_dir = Path(tempfile.mkdtemp(prefix="tool-call-efficiency-"))
env = wai.export_environment(
    clean,
    env_dir,
    name="tool_call_efficiency",
    reward=REWARD_REF,
    tools=TOOLS,
    system_prompt=POLICY,
)
print(
    f"train: {len(train_rows)} rows -> {len(selected)} in the 20-80% band "
    f"({selection['groups_selected']} tasks) -> {len(clean)} after decontamination "
    f"({contamination['n_contaminated']} dropped) -> {env['name']} at {env_dir}"
)

check(len(selected) > 0, "some tasks sit in the 20-80% band")
check(
    all(0.2 <= r["calibration"]["pass_rate"] <= 0.8 for r in selected if r.get("calibration")),
    "band held",
)
check(contamination["n_contaminated"] >= 0 and len(clean) <= len(selected), "decontaminated")
check((env_dir / env["name"] / "data" / "train.jsonl").exists(), "environment written")
check(env["reward"] == REWARD_REF, "environment names the verifier as its reward")

# ---- 4. after: same tasks, both numbers, then report

after = wai.simulate(
    after_agent,
    tools=TOOLS,
    system_prompt=POLICY,
    tasks=holdout,
    runs=3,
    mode="rl",
    repeat_policy="fixed",
    simulator=False,
    reproducible=True,
    seed=1,
    concurrency=1,
)
after_rows = grade(stamp(after.rows()))
after_eff, after_corr, after_calls = score(after_rows)
delta = wai.delta_report(
    before_rows,
    after_rows,
    target="pass_at_1",
    must_not_regress=["solved"],
)
print(wai.format_delta_report(delta))
print(
    f"after: {pct(after_eff.pass_at_1)}% within budget, {pct(after_corr.pass_at_1)}% solved, "
    f"{after_calls:.2f} calls per solved task"
)

check(delta["ok"], "delta report passes")
check(delta["target_verdict"] in {"improved", "moved"}, f"target verdict {delta['target_verdict']}")
check(after_eff.pass_at_1 > before_eff.pass_at_1, "after scores higher on the behavior")
check(after_calls < before_calls, "fewer calls per solved task")
check(after_corr.pass_at_1 >= before_corr.pass_at_1, "correctness did not regress")

tracked = track(
    "order-bot",
    model="Qwen/Qwen3-4B",
    harness=Harness(instructions=POLICY, tools=TOOLS),
    transport=fake,  # drop this line to talk to the platform
)
tracked.behavior(
    Behavior(
        name="tool-call-efficiency",
        test_version="v1",
        n=n_tasks,
        judge=Judge(name="within_budget verifier", agreement=1.0, human_n=len(CASES)),
        noise_floor=pct(spread["run_std"]),
        contamination=0,  # decontaminate ran; else pass what it would have dropped
        reward_is_judge=False,
        description=f"Share of held-out tasks solved within {BUDGET} tool call(s)",
    )
)
tracked.behavior(
    Behavior(
        name="correctness",
        test_version="v1",
        n=n_tasks,
        judge=Judge(name="solved verifier", agreement=1.0, human_n=len(CASES)),
        noise_floor=pct(spread["run_std"]),
        reward_is_judge=False,
        description="Share of held-out tasks solved, any number of calls",
    )
)
base = tracked.run("v0", method="none")
base.score("tool-call-efficiency", pct(before_eff.pass_at_1), ci=half(before_eff.ci95), n=n_tasks)
base.score("correctness", pct(before_corr.pass_at_1), ci=half(before_corr.ci95), n=n_tasks)
base.finish()
tracked.promote("v0")

run = tracked.run("v1", method="GRPO", targets=["tool-call-efficiency"], trained_on=[env["name"]])
run.score("tool-call-efficiency", pct(after_eff.pass_at_1), ci=half(after_eff.ci95), n=n_tasks)
run.score("correctness", pct(after_corr.pass_at_1), ci=half(after_corr.ci95), n=n_tasks)
run.finish()
verdict = str(tracked.verdict())
print(verdict)

# ================================================================ end SKILL.md

check("v1" in verdict and "v0" in verdict, "verdict names both versions")
check("beats" in verdict, "verdict says beats")
check("regression" not in verdict, "no regression on correctness")
check(not verdict.startswith("unproven"), "verdict is proven (n, judge, reward all declared)")
paths = [p for _, p, _ in fake.calls]
check("/agents" in paths and "/runs" in paths, "platform calls recorded")
check(sum(p.endswith("/evals") for p in paths) == 4, "both behaviors scored on both versions")
elapsed = time.monotonic() - T0
check(elapsed < 60, f"ran in {elapsed:.1f}s")
shutil.rmtree(env_dir, ignore_errors=True)
