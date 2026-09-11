"""When simulate() returns, the caller's agent is no longer being called.

A stop (clock, cap, saturation) cancels rollouts that never started,
waits up to the stop grace for the ones running, keeps what finishes,
and reports what it had to abandon.
"""
from __future__ import annotations

import time

import zeroproof_simulations as zps
from tests.helpers import POLICY, TOOLS

_OFFLINE = dict(seed=0, simulator=False, grade=False,
                advanced={"per_round": 32, "mutate_failures": False})


def test_clock_stop_keeps_finished_rollouts_and_makes_no_late_calls():
    calls: list[float] = []

    def slow_agent(_message: str) -> dict:
        calls.append(time.monotonic())
        time.sleep(0.6)
        return {"steps": [], "final_text": "ok"}

    data = zps.simulate(slow_agent, tools=TOOLS, policy=POLICY, budget=50,
                        concurrency=8, time_budget=0.5, **_OFFLINE)
    returned = time.monotonic()
    time.sleep(1.5)
    late = [c for c in calls if c > returned]
    assert data.stopped_because == "time_budget"
    assert data.trajectories, "rollouts that were running at the stop must be kept"
    assert not late, f"{len(late)} agent calls started after simulate() returned"
    assert "rollouts_abandoned" not in data.degraded


def test_rollouts_still_running_after_the_grace_are_reported():
    def hanging_agent(_message: str) -> dict:
        time.sleep(6.0)
        return {"steps": [], "final_text": "late"}

    t0 = time.monotonic()
    kw = dict(_OFFLINE, advanced={**_OFFLINE["advanced"], "stop_grace": 0.2})
    data = zps.simulate(hanging_agent, tools=TOOLS, policy=POLICY, budget=50,
                        concurrency=4, time_budget=0.3, **kw)
    # 6 s agent, 0.3 s clock, 0.2 s grace: even a loaded CI box returns
    # well inside 4 s, and 4 s is still far short of the agent finishing.
    assert time.monotonic() - t0 < 4.0, "the stop grace must bound the wait"
    assert data.stopped_because == "time_budget"
    assert not data.trajectories
    assert "rollouts_abandoned" in data.degraded
    assert data.search.get("abandoned_rollouts", 0) >= 1


def test_budget_stop_has_no_stragglers():
    calls: list[float] = []

    def agent(_message: str) -> dict:
        calls.append(time.monotonic())
        time.sleep(0.05)
        return {"steps": [], "final_text": "ok"}

    data = zps.simulate(agent, tools=TOOLS, policy=POLICY, budget=6,
                        concurrency=8, time_budget=None, **_OFFLINE)
    returned = time.monotonic()
    time.sleep(0.5)
    assert len(data.trajectories) == 6
    assert not [c for c in calls if c > returned]
    assert "rollouts_abandoned" not in data.degraded
