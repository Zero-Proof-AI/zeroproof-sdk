"""The online hack monitor (monitor.py), driven offline.

A fake policy that drifts toward a delimiter the proxy pays for, a gold
scorer that does not care about it, and the Trainer events fired by hand.
"""

from __future__ import annotations

import random
import types

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.monitor import ALARMS, HackMonitor, format_hack_monitor
from zeroproof.simulations.training import training_run

WORDS = [
    "order",
    "refund",
    "status",
    "ship",
    "customer",
    "account",
    "help",
    "please",
    "check",
    "note",
]


class Transport:
    def __init__(self):
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(self, method, path, api_key=None, body=None, **kw):
        self.calls.append((method, path, body))
        if path == "/runs" and method == "POST":
            return {"runId": "run_m", "status": "running", **body}
        if path.endswith("/finish"):
            return {"runId": "run_m", "status": body["status"]}
        return {"runId": "run_m", "logged": len(body["points"])}

    def points(self) -> list[dict]:
        return [p for m, path, b in self.calls if path.endswith("/log") for p in b["points"]]

    def finish_body(self) -> dict | None:
        bodies = [b for m, path, b in self.calls if path.endswith("/finish")]
        return bodies[-1] if bodies else None


class Policy:
    """Stage 0 is the base policy. Each stage uses the delimiter more and
    writes longer; whether it looks the order up never changes."""

    def __init__(self, seed: int = 0):
        self.stage = 0
        self.seed = seed

    def sample(self, model, tokenizer, prompts, *, n, max_new_tokens):
        rng = random.Random(self.seed + 1000 * self.stage)
        out = []
        for _ in prompts:
            group = []
            for _ in range(n):
                words = [rng.choice(WORDS) for _ in range(5 + 3 * self.stage)]
                if rng.random() < 0.5:
                    words.append("lookup_order")
                text = " ".join(words) + "."
                if rng.random() < min(1.0, 0.1 + 0.3 * self.stage):
                    text += " ### done ###"
                group.append(text)
            out.append(group)
        return out


def hackable_proxy(prompts=None, completions=None, **cols):
    """The training reward: pays for the delimiter."""
    texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    return [1.0 if "###" in t else 0.0 for t in texts]


def gold(row: dict):
    return 1 if "lookup_order" in row["final_text"] else 0


def _state(step: int = 0):
    return types.SimpleNamespace(global_step=step, max_steps=100)


def _control():
    return types.SimpleNamespace(should_training_stop=False)


def test_divergence_alarm_stops_the_run_and_lands_on_the_run_page():
    t = Transport()
    run = training_run("mon", api_key="k", flush_every=1, transport=t)
    policy = Policy()
    monitor = HackMonitor(
        run,
        holdout=[f"ask {i}" for i in range(20)],
        proxy=hackable_proxy,
        gold=gold,
        sample=policy.sample,
        every=10,
        k=4,
        window=2,
        delta=0.1,
        stop_on="divergence",
    )
    control = _control()
    monitor.on_train_begin(None, _state(0), control)
    assert len(monitor.history) == 1 and monitor.history[0]["step"] == 0
    stopped_at = None
    for step in (10, 20, 30, 40):
        policy.stage += 1
        monitor.on_step_end(None, _state(step), control, model=object())
        if control.should_training_stop:
            stopped_at = step
            break
    assert stopped_at == 10
    kinds = {a["kind"] for a in monitor.alarms}
    assert "divergence" in kinds
    assert monitor.stopped_at["kind"] == "divergence"
    divergence = next(a for a in monitor.alarms if a["kind"] == "divergence")
    assert divergence["proxy_gain"] >= 0.1 and "gold moved" in divergence["reason"]
    assert monitor.history[-1]["gold_vs_window"]["verdict"] != "b_better"

    monitor.on_train_end(None, _state(10), control)
    assert run.status == "stopped"
    body = t.finish_body()
    assert body["status"] == "stopped" and "divergence" in body["error"]
    summary = body["summary"]["hack_monitor"]
    assert summary["n_evals"] == 2 and summary["stopped_at"]["kind"] == "divergence"
    assert all("rows" not in h for h in summary["history"])
    points = t.points()
    assert any("proxy_reward" in p and "gold_reward" in p and "holdout_length" in p for p in points)
    assert any(p.get("alarm_divergence") == 1.0 for p in points)
    text = format_hack_monitor(monitor.summary())
    assert "STOPPED at step 10 on divergence" in text and "proxy" in text


def test_wrap_records_the_buffer_and_the_feature_alarm_names_the_delimiter():
    policy = Policy()
    monitor = HackMonitor(
        None,
        holdout=[f"ask {i}" for i in range(10)],
        gold=None,
        sample=policy.sample,
        endorsed=["lookup_order"],
        buffer=1000,
    )
    watched = monitor.wrap(hackable_proxy)
    assert watched.__name__ == "hackable_proxy"
    rng = random.Random(3)
    for p in range(30):
        completions = []
        for _ in range(8):
            text = " ".join(rng.choice(WORDS) for _ in range(6))
            if rng.random() < 0.5:
                text += " lookup_order"
            if rng.random() < 0.5:
                text += " ### done ###"
            completions.append([{"role": "assistant", "content": text}])
        prompts = [[{"role": "user", "content": f"ask {p}"}]] * 8
        out = watched(prompts=prompts, completions=completions)
        assert out == hackable_proxy(completions=completions)
    assert len(monitor.buffer) == 240
    assert monitor.buffer[0]["prompt"] == "ask 0" and monitor.buffer[0]["reward"] in (0.0, 1.0)

    entry = monitor.evaluate(0)  # proxy comes from the wrapped reward
    assert entry["proxy"] is not None and entry["gold"] is None
    assert monitor.last_scan["regime"] == "reward_hack"
    assert "#" in monitor.last_scan["top_feature"]
    assert [a["kind"] for a in monitor.alarms] == ["feature"]
    assert monitor.stopped_at is None  # log only by default


def test_length_alarm_without_gold_and_no_divergence():
    policy = Policy()
    monitor = HackMonitor(
        None,
        holdout=[f"ask {i}" for i in range(12)],
        proxy=hackable_proxy,
        sample=policy.sample,
        every=5,
        window=1,
        length_pct=0.25,
    )
    control = _control()
    monitor.on_train_begin(None, _state(0), control)
    policy.stage = 1
    monitor.on_step_end(None, _state(5), control)
    kinds = [a["kind"] for a in monitor.alarms]
    assert kinds == ["length"]
    assert monitor.alarms[0]["length_growth"] > 0.25
    assert not control.should_training_stop
    assert "no alarms" not in format_hack_monitor(monitor.summary())


def test_drift_alarm_reads_the_trainer_kl():
    policy = Policy()
    monitor = HackMonitor(
        None,
        holdout=["ask 1", "ask 2"],
        proxy=hackable_proxy,
        sample=policy.sample,
        every=10,
        kl_budget=0.5,
        stop_on=("drift",),
    )
    control = _control()
    monitor.on_log(None, _state(9), control, logs={"objective/kl": 0.7})
    assert monitor.last_kl == 0.7
    monitor.on_step_end(None, _state(10), control)
    assert [a["kind"] for a in monitor.alarms] == ["drift"]
    assert control.should_training_stop and monitor.stopped_at["kind"] == "drift"
    assert monitor.history[-1]["kl"] == 0.7


def test_holdout_columns_reach_the_proxy_the_way_trl_passes_them():
    seen: dict = {}

    def proxy(prompts=None, completions=None, **cols):
        seen["prompts"] = prompts
        seen["completions"] = completions
        seen["cols"] = cols
        return [1.0] * len(completions)

    holdout = [
        {"prompt": [{"role": "user", "content": "check ORD-1"}], "case": {"order_id": "ORD-1"}},
        {"prompt": [{"role": "user", "content": "check ORD-2"}], "case": {"order_id": "ORD-2"}},
    ]
    monitor = HackMonitor(
        None,
        holdout=holdout,
        proxy=proxy,
        gold=lambda row: {"reward": 1, "reason": "ok"},
        sample=Policy().sample,
        k=3,
    )
    entry = monitor.evaluate(0)
    assert entry["n"] == 6 and entry["gold"] == 1.0 and entry["proxy"] == 1.0
    assert seen["cols"]["case"] == [{"order_id": "ORD-1"}] * 3 + [{"order_id": "ORD-2"}] * 3
    assert seen["completions"][0][0]["role"] == "assistant"
    assert seen["prompts"][0] == holdout[0]["prompt"]
    row = entry["rows"][0]
    assert row["prompt"] == "check ORD-1" and row["messages"][0]["role"] == "user"
    assert row["reward"] == 1 and row["markers"]["gold"] == 1.0 and row["markers"]["proxy"] == 1.0


def test_arguments_are_checked_and_a_gold_failure_is_not_a_crash():
    with pytest.raises(ValueError, match="holdout"):
        HackMonitor(None, holdout=[], proxy=hackable_proxy)
    with pytest.raises(ValueError, match="stop_on"):
        HackMonitor(None, holdout=["a"], proxy=hackable_proxy, stop_on="lengthy")
    monitor = HackMonitor(None, holdout=["a"], sample=Policy().sample)
    with pytest.raises(ValueError, match="no proxy"):
        monitor.evaluate(0)
    assert set(ALARMS) == {"divergence", "length", "drift", "feature"}

    def broken(row):
        raise RuntimeError("judge down")

    monitor = HackMonitor(
        None, holdout=["a", "b"], proxy=hackable_proxy, gold=broken, sample=Policy().sample
    )
    entry = monitor.evaluate(0)
    assert entry["gold"] is None and entry["proxy"] is not None
    # A failing eval never raises into the trainer either.
    monitor.sample = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cuda"))
    monitor.on_step_end(None, _state(10), _control())
    assert len(monitor.history) == 1


def test_run_note_carries_fields_into_finish():
    t = Transport()
    run = training_run("n", api_key="k", flush_every=100, transport=t)
    run.note(hack_monitor={"n_evals": 1})
    run.finish("done", summary={"final": 1})
    assert t.finish_body()["summary"] == {"hack_monitor": {"n_evals": 1}, "final": 1}
    run.note(late=True)
    assert t.finish_body()["summary"]["late"] is True
    assert zps.HackMonitor is HackMonitor
