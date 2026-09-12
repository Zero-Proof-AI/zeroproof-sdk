"""Writer waves stop with the run, the same way rollouts do."""
from __future__ import annotations

import time

import zeroproof.simulations as zps
from tests.helpers import POLICY, TOOLS, scripted_agent


def test_writer_waves_running_after_the_grace_are_reported():
    calls: list[float] = []

    def slow_writer(_dataset=None, index=0):
        calls.append(time.monotonic())
        time.sleep(6.0)
        return [f"where is my refund for order {index}-{i}" for i in range(4)]

    t0 = time.monotonic()
    data = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=40, seed=0,
        concurrency=4, simulator=slow_writer, grade=False, time_budget=0.4,
        advanced={"mutate_failures": False, "scenario_concurrency": 2,
                  "stop_grace": 0.2})
    returned = time.monotonic()
    assert returned - t0 < 4.0, "the stop grace must bound the wait for writers too"
    assert data.stopped_because == "time_budget"
    assert "writer_waves_abandoned" in data.degraded
    assert data.search.get("abandoned_writer_waves", 0) >= 1
    time.sleep(1.0)
    late = [c for c in calls if c > returned]
    assert not late, f"{len(late)} writer waves started after simulate() returned"


def test_writer_waves_that_finish_in_the_grace_are_not_flagged():
    def quick_writer(_dataset=None, index=0):
        time.sleep(0.05)
        return [f"where is my refund for order {index}-{i}" for i in range(4)]

    data = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=8, seed=0,
        concurrency=4, simulator=quick_writer, grade=False, time_budget=None,
        advanced={"mutate_failures": False, "scenario_concurrency": 2})
    assert len(data.trajectories) == 8
    assert "writer_waves_abandoned" not in data.degraded
    assert "abandoned_writer_waves" not in data.search
