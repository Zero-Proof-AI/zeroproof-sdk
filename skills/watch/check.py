"""Offline check for skills/watch/SKILL.md: the daily job that keeps the
Live tile honest without the platform ever ingesting a trace.

Fourteen days of generated traffic from a scripted refund bot (urgent asks
that trip it grow on the last three days), a recording fake platform whose
dashboard is built from the LiveDay rows it received, and an assertion at
every step. No key, no network, no model.

    uv run python skills/watch/check.py
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import tempfile
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import whileai.simulations as wai
from whileai.platform import Behavior, Dashboard, Judge, LiveDay, track

T0 = time.monotonic()

# ---------------------------------------------------------------- the served agent

TODAY = date(2026, 9, 18)
REFUND_LIMIT = 200.0
WINDOW_DAYS = 30
ORDERS: dict[str, dict[str, Any]] = {
    "A1001": {
        "item": "Trail runners",
        "total": 129.0,
        "ordered": "2026-09-05",
        "status": "delivered",
    },
    "A1002": {
        "item": "Espresso machine",
        "total": 449.0,
        "ordered": "2026-09-01",
        "status": "delivered",
    },
    "A1003": {"item": "Wool socks", "total": 24.0, "ordered": "2026-09-14", "status": "shipped"},
    "A1004": {"item": "Headphones", "total": 189.0, "ordered": "2026-06-20", "status": "delivered"},
}
TOOLS = ["lookup_order", "issue_refund"]
ORDER_ID = re.compile(r"\b([A-Z]\d{4})\b")
URGENT = ("right now", "immediately", "or i dispute")


def lookup_order(order_id: str) -> dict[str, Any]:
    order = ORDERS.get(order_id.upper())
    return {"order_id": order_id.upper(), **order} if order else {"error": f"no order {order_id}"}


def refundable(order: dict[str, Any]) -> tuple[bool, str]:
    """The refund policy as a program. The bot and the judge share it."""
    if "error" in order:
        return False, "unknown order"
    age = (TODAY - date.fromisoformat(order["ordered"])).days
    if order["status"] != "delivered":
        return False, "not delivered"
    if age > WINDOW_DAYS:
        return False, f"outside the {WINDOW_DAYS}-day window"
    if order["total"] > REFUND_LIMIT:
        return False, "over the limit, needs a manager"
    return True, "eligible"


def _wants_refund(message: str) -> bool:
    return any(w in message.lower() for w in ("refund", "money back", "return", "reimburse"))


def _panics(message: str) -> bool:
    """v3's known weakness: urgent asks make it refund before it looks, and a
    hash-seeded 3% of ordinary asks hit the same path (a model is not a rule)."""
    text = message.lower()
    if any(u in text for u in URGENT):
        return True
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 100 < 3


def served_bot(message: str) -> dict[str, Any]:
    """The refund bot as served (version v3)."""
    steps: list[dict[str, Any]] = []
    ids = ORDER_ID.findall(message.upper())
    if not ids:
        return {"steps": steps, "final_text": "Happy to help. Which order id is this about?"}
    order_id = ids[0]
    if _wants_refund(message) and _panics(message):
        amount = ORDERS.get(order_id, {}).get("total", 50.0)
        steps.append(
            {"tool": "issue_refund", "arguments": {"order_id": order_id, "amount": amount}}
        )
        return {"steps": steps, "final_text": f"Done, ${amount:.2f} refunded for {order_id}."}
    order = lookup_order(order_id)
    steps.append({"tool": "lookup_order", "arguments": {"order_id": order_id}, "result": order})
    if "error" in order:
        return {"steps": steps, "final_text": f"I could not find an order {order_id}."}
    if not _wants_refund(message):
        return {"steps": steps, "final_text": f"Order {order_id} is {order['status']}."}
    ok, why = refundable(order)
    if ok:
        steps.append(
            {
                "tool": "issue_refund",
                "arguments": {"order_id": order_id, "amount": order["total"]},
                "result": {"ok": True},
            }
        )
        return {
            "steps": steps,
            "final_text": f"Done: ${order['total']:.2f} refunded for {order_id}.",
        }
    if "manager" in why:
        return {
            "steps": steps,
            "final_text": f"Order {order_id} needs a manager; one will follow up.",
        }
    return {"steps": steps, "final_text": f"I cannot refund order {order_id}: {why}."}


# ---------------------------------------------------------------- the held-out judge


def classify(prompt: str) -> str:
    ids = ORDER_ID.findall(prompt.upper())
    if not _wants_refund(prompt) or not ids:
        return "no_refund_asked"
    order = lookup_order(ids[0])
    if "error" in order:
        return "unknown_order"
    ok, why = refundable(order)
    return "eligible" if ok else ("over_limit" if "manager" in why else "not_refundable")


def refund_judge(row: dict) -> dict[str, Any]:
    """The policy read off the trajectory. The same judge scores the held-out
    test and the daily sample; 1.0 on every marker is the good outcome."""
    steps = [s for s in (row.get("steps") or []) if isinstance(s, dict)]
    branch = classify(str(row.get("prompt") or ""))
    lookups = [i for i, s in enumerate(steps) if s.get("tool") == "lookup_order"]
    refunds = [i for i, s in enumerate(steps) if s.get("tool") == "issue_refund"]
    marks: dict[str, float | None] = {
        "looked_up_first": None,
        "refund_only_when_allowed": 1.0 if (branch == "eligible" or not refunds) else 0.0,
        "refunds_when_eligible": None,
    }
    if refunds:
        marks["looked_up_first"] = 1.0 if (lookups and lookups[0] < refunds[0]) else 0.0
    elif branch != "no_refund_asked":
        marks["looked_up_first"] = 1.0 if lookups else 0.0
    if branch == "eligible":
        marks["refunds_when_eligible"] = 1.0 if refunds else 0.0
    failed = [k for k, v in marks.items() if v == 0.0]
    return {
        "reward": 0.0 if failed else 1.0,
        "reason": ", ".join(failed) or "followed the policy",
        "markers": marks,
        "failure_class": failed[0] if failed else None,
    }


# ---------------------------------------------------------------- the traffic

ASKS = [
    "Hi, I want a refund for order {oid}, it did not fit.",
    "Please refund {oid}, it was a gift I never used.",
    "I need my money back on {oid}, it arrived broken.",
    "Can you reimburse me for {oid}? Wrong colour.",
    "What is the status of order {oid}?",
    "Is {oid} delivered yet? Just checking.",
    "Refund {oid} please, I changed my mind.",
    "Where is my order {oid}?",
    "Could I get a refund on {oid}? It is the wrong size.",
    "Has {oid} shipped?",
    "I would like to return {oid} for a refund.",
]
URGENT_ASKS = [
    "Refund {oid} right now or I dispute the charge.",
    "I want my money back on {oid} immediately.",
]
OIDS = [*ORDERS, "Z9999"]
SIGNOFFS = [
    "",
    " Thanks.",
    " Thank you!",
    " Cheers.",
    " Please advise.",
    " Let me know.",
    " Regards, Sam.",
]
DAYS = [TODAY - timedelta(days=14 - i) for i in range(14)]
URGENT_SHARE = {DAYS[11]: 0.10, DAYS[12]: 0.12, DAYS[13]: 0.14}  # the traffic shifts; v3 does not

ROOT = Path(tempfile.mkdtemp(prefix="watch-"))
PANICKED: set[str] = set()  # trace ids the bot took the bad path on; the oracle for the asserts


def trace_path(day: date) -> Path:
    """Where yesterday's export lives. Yours is a JSONL dump or a table pull."""
    return ROOT / f"traces-{day.isoformat()}.jsonl"


def write_traffic() -> None:
    """Fourteen days in the platform-pull shape (tool_trace/input/output),
    so wai.load_traces has real work to do."""
    for i, day in enumerate(DAYS):
        replies = 220 + (i * 37) % 50
        share = URGENT_SHARE.get(day, 0.0)
        with trace_path(day).open("w", encoding="utf-8") as fh:
            for j in range(replies):
                rng = random.Random(f"{day}-{j}")
                pool = URGENT_ASKS if rng.random() < share else ASKS
                prompt = rng.choice(pool).format(oid=rng.choice(OIDS)) + rng.choice(SIGNOFFS)
                out = served_bot(prompt)
                trace_id = f"{day}-{j:04d}"
                if any(s["tool"] == "issue_refund" for s in out["steps"][:1]):
                    PANICKED.add(trace_id)
                fh.write(
                    json.dumps(
                        {
                            "trace_id": trace_id,
                            "ts": f"{day}T{8 + j % 12:02d}:00:00Z",
                            "version": "v3",
                            "input": prompt,
                            "tool_trace": [
                                {
                                    "tool": s["tool"],
                                    "input": s["arguments"],
                                    "output": s.get("result"),
                                }
                                for s in out["steps"]
                            ],
                            "output": out["final_text"],
                            "latency_s": round(
                                0.4 + 0.8 * rng.random() + 0.3 * len(out["steps"]), 3
                            ),
                            "cost_usd": round(0.0008 + len(out["final_text"]) * 1e-6, 6),
                        }
                    )
                    + "\n"
                )


# ---------------------------------------------------------------- the fake platform


class FakePlatform:
    """Records every call and answers like the API. The dashboard's live
    series is built from the LiveDay rows it received, keyed
    live#<agent>#<day>, so a rerun for the same day overwrites."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.live: dict[str, dict[str, Any]] = {}
        self.behaviors: dict[str, dict[str, Any]] = {}
        self.evals: list[dict[str, Any]] = []
        self.serving = "v3"

    def __call__(self, method: str, path: str, body: Any = None) -> Any:
        self.calls.append((method, path, body))
        if path == "/live":
            for row in body:
                self.live[f"live#{row['agent']}#{row['day']}"] = row
            return {"written": len(body)}
        if path == "/runs":
            return {"id": f"{body['agent']}-{body['version']}", "version": body["version"]}
        if path.endswith("/evals"):
            version = path.split("/")[2].rsplit("-", 1)[1]
            self.evals.extend({**e, "version": version} for e in body)
            return {"evals": body}
        if method == "PUT" and "/behaviors/" in path:
            name = path.rsplit("/", 1)[1]
            self.behaviors[name] = {"name": name, **body}
            return self.behaviors[name]
        if "/dashboard" in path:
            return self.dashboard(path.split("/")[2])
        return {"id": body.get("id", "")} if isinstance(body, dict) else {"ok": True}

    def dashboard(self, agent_id: str) -> dict[str, Any]:
        rows = sorted(
            (r for k, r in self.live.items() if k.startswith(f"live#{agent_id}#")),
            key=lambda r: r["day"],
        )
        return {
            "agent": {"id": agent_id, "name": agent_id, "serving": self.serving},
            "behavior": self.behaviors.get("refunds"),
            "behaviors": list(self.behaviors),
            "versions": [
                {"v": e["version"], "score": e["score"], "ci": e.get("ci"), "n": e.get("n")}
                for e in self.evals
            ],
            "live": {
                "days": [r["day"] for r in rows],
                "version": [r["version"] for r in rows],
                "flaggedPct": [round(100 * r["flagged"] / r["replies"], 2) for r in rows],
                "p50s": [r.get("p50s") for r in rows],
                "replies": [r["replies"] for r in rows],
                "replies7d": sum(r["replies"] for r in rows[-7:]),
            },
            "verdict": {"serving": self.serving},
        }


fake = FakePlatform()

# ---------------------------------------------------------------- SKILL.md blocks

tracked = track("refund-bot", model="Qwen/Qwen3-4B", transport=fake)  # drop transport= for real
tracked.behavior(
    Behavior(
        name="refunds",
        test_version="v1",
        judge=Judge(name="refund policy as a program"),
        noise_floor=2.0,  # points; the re-run spread you measured on this test
        reward_is_judge=False,
        description="Looks the order up, refunds only when policy allows, escalates over-limit.",
    )
)

HELDOUT = [a.format(oid=o) for a in ASKS for o in OIDS] + [
    a.format(oid="A1001") for a in URGENT_ASKS
]  # frozen, test_version v1


def score_served(version: str) -> float:
    """The served version on the frozen held-out test, with the same judge.
    The tile needs this once per version so day one has a baseline."""
    rows = [{"prompt": p, **served_bot(p), "task_id": p} for p in HELDOUT]
    scored = wai.evaluate(rows, refund_judge, tools=TOOLS)
    n, passed = len(scored.rows), len(scored.passes())
    score = round(100 * passed / n, 1)
    ci = round(196 * math.sqrt(score / 100 * (1 - score / 100) / n), 1)
    with tracked.run(version, method="none", id=f"refund-bot-{version}") as run:
        run.score("refunds", score, ci=ci, n=n, test_version="v1")
        run.finish()
    return score


N = 100  # sample per day; the same date always draws the same rows


def read_day(day: date) -> tuple[list[dict], str]:
    """Yesterday's traces, normalised, and the served version that produced them."""
    traces = wai.load_traces(trace_path(day))  # a JSONL path, a table dump, or a list of dicts
    versions = Counter(str(t.get("version") or "unknown") for t in traces)
    version = versions.most_common(1)[0][0]
    if len(versions) > 1:
        print(f"  {day}: served by {dict(versions)}; posting under {version}, which served most")
    return traces, version


def sample_and_score(traces: list[dict], day: date) -> tuple[list[dict], int]:
    """A fixed-size draw seeded by the date, scored by the held-out judge."""
    picked = random.Random(day.isoformat()).sample(traces, min(N, len(traces)))
    scored = wai.evaluate(picked, refund_judge, tools=TOOLS)
    for note in scored.warnings:
        print("  !", note)  # a hollow sample (no tool ran) is not a flagged rate
    return picked, len(scored.failures())


def post_day(day: date, version: str, traces: list[dict], failed: int, n: int) -> LiveDay:
    """One LiveDay. flagged is the sample's failures scaled to the day's replies."""
    replies = len(traces)
    latencies = [float(t["latency_s"]) for t in traces if t.get("latency_s") is not None]
    costs = [float(t["cost_usd"]) for t in traces if t.get("cost_usd") is not None]
    item = LiveDay(
        day=day,
        version=version,
        replies=replies,
        flagged=round(replies * failed / n),
        p50_s=round(statistics.median(latencies), 3) if latencies else None,
        cost_usd=round(sum(costs), 4) if costs else None,
    )
    tracked.live(item)  # POST /live; the row key is live#<agent>#<day>, so a rerun overwrites
    return item


def drift(dash: Dashboard) -> tuple[bool, str]:
    """True when the flagged rate sat above its baseline by more than the band
    on each of the last two days of the served version. The band is the
    larger of the behavior's noise_floor and the sample's own 95% half-width."""
    live = dash.live
    floor = dash.behavior.noise_floor if dash.behavior else None
    if floor is None:
        return False, "no noise_floor declared; tracked.behavior('refunds', noise_floor=<spread>)"
    version = live.version[-1] if live.version else None
    pcts = [p for v, p in zip(live.version, live.flagged_pct) if v == version and p is not None]
    if len(pcts) < 2:
        return False, f"{len(pcts)} day(s) on {version}; the rule needs two"
    earlier, last_two = pcts[:-2], pcts[-2:]
    if len(earlier) >= 3:
        baseline, source = statistics.median(earlier), f"median of {len(earlier)} earlier days"
    else:
        held = next((v.score for v in dash.versions if v.v == version), None)
        if held is None:
            return False, f"no earlier days and no held-out score for {version}; score_served()"
        baseline, source = 100 - held, "held-out fail rate"
    half = 196 * math.sqrt(baseline / 100 * (1 - baseline / 100) / N)
    band = max(floor, half)
    rises = [round(p - baseline, 1) for p in last_two]
    fired = all(r > band for r in rises)
    note = f"rise {rises[0]:+g} then {rises[1]:+g} vs {source} {baseline:.1f}%, band {band:.1f}"
    return fired, note


def watch(day: date) -> tuple[LiveDay, bool]:
    """The daily job. Run it once per day; the day is the key, so a rerun is harmless."""
    traces, version = read_day(day)
    picked, failed = sample_and_score(traces, day)
    item = post_day(day, version, traces, failed, len(picked))
    fired, note = drift(tracked.dashboard("refunds"))
    pct = 100 * item.flagged / item.replies if item.replies else 0.0
    print(
        f"{day} {version}: {item.replies} replies, {failed}/{len(picked)} sampled failed, "
        f"flagged {item.flagged} ({pct:.1f}%), p50 {item.p50_s}s; {note}"
    )
    if fired:
        print(
            "  DRIFT: two days above the band. Open a run: tracked.run('v4', targets=['refunds'])"
        )
    return item, fired


def daily(day: date) -> None:
    """What cron runs: yesterday's tile, then the one line a person reads."""
    watch(day)
    print(str(tracked.verdict("refunds")))


# ---------------------------------------------------------------- the check


def main() -> None:
    write_traffic()
    held = score_served("v3")
    assert 90 <= held <= 99, held
    assert fake.evals and fake.evals[0]["version"] == "v3"

    # Step 2: the draw is a function of the date alone.
    traces, version = read_day(DAYS[0])
    assert version == "v3" and len(traces) == 220
    assert all("steps" in t and "final_text" in t and "prompt" in t for t in traces), "normalised"
    a, fa = sample_and_score(traces, DAYS[0])
    b, fb = sample_and_score(traces, DAYS[0])
    assert [t["trace_id"] for t in a] == [t["trace_id"] for t in b] and fa == fb, (
        "same date, same draw"
    )
    c, _ = sample_and_score(traces, DAYS[1])
    assert [t["trace_id"] for t in a] != [t["trace_id"] for t in c], "another date, another draw"
    assert fa == sum(t["trace_id"] in PANICKED for t in a), "flagged = rows the judge failed"

    # Steps 1 to 4, one day at a time, as cron would.
    fired_on: list[date] = []
    posted: dict[date, LiveDay] = {}
    for day in DAYS:
        item, fired = watch(day)
        posted[day] = item
        if fired:
            fired_on.append(day)
        oracle = sum(t["trace_id"] in PANICKED for t in sample_and_score(read_day(day)[0], day)[0])
        assert item.flagged == round(item.replies * oracle / N), (day, item.flagged, oracle)
        assert item.p50_s is not None and item.cost_usd is not None

    posts = [b for m, p, b in fake.calls if (m, p) == ("POST", "/live")]
    assert [p[0]["day"] for p in posts] == [d.isoformat() for d in DAYS], "one POST /live per day"
    assert all(len(p) == 1 and p[0]["agent"] == "refund-bot" for p in posts)
    assert len(fake.live) == 14 and all(f"live#refund-bot#{d}" in fake.live for d in DAYS)
    assert fired_on == DAYS[12:], f"drift fired on {fired_on}, expected the last two days"
    assert not any(posted[d].flagged / posted[d].replies > 0.08 for d in DAYS[:11]), "quiet days"

    # Step 5: idempotent. A rerun of a day overwrites its row and changes nothing.
    before = dict(fake.live)
    again, fired_again = watch(DAYS[-1])
    assert again == posted[DAYS[-1]] and fired_again
    assert fake.live == before and len(fake.live) == 14, "rerun overwrote the same row"

    # Step 6: the verdict line.
    daily(DAYS[-1])
    line = str(tracked.verdict("refunds"))
    assert line == "refunds: v3 is serving; no newer candidate yet", line

    dash = tracked.dashboard("refunds")
    assert dash.live.days == [d.isoformat() for d in DAYS] and dash.live.replies_7d > 0
    assert dash.verdict.noise_floor == 2.0

    elapsed = time.monotonic() - T0
    assert elapsed < 60, elapsed
    print(
        f"\nok: 14 days watched, drift fired on {[d.isoformat() for d in fired_on]}, {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
